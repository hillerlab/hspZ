// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung
// Distributed under the terms of the GNU General Public License, Version 3.0.

// Author : Alejandro Gonzales-Irribarren
// Github : alejandrogzi
// Email  : alejandrxgzi@gmail.com

//! The `seed_filter.cu` device kernels.
//!
//! Where parity is load-bearing the kernels stay literal transcriptions — warp
//! mapping and all — and the two X-drop extension loops stay duplicated because
//! they are duplicated upstream. Almost everything else has been measured and
//! reworked: ballot X-drop termination and deferred entropy in `find_hsps`
//! (rounds 4, 3b), the score-only prefix max (round 5), peeled entry checks and
//! unchecked bounds (rounds 6b, 7), the `align(16)` store and grid 16384
//! (rounds 13, 6), the device-resident count and done-flag scans (rounds 8, 10),
//! the warp-coalesced score gate and dense-anchor compaction (rounds 44–47),
//! device seed generation (`seed_kmers`/`scatter_seeds`, round 68), and
//! `find_hits`' thread-per-seed mapping (round 1). Every change stays
//! byte-identical to the oracle at matched plan and `MAX_HITS`.

use crate::hsp::SegmentPair;

/// `parameters.h: NUM_WARPS`. Both `find_hits` and `find_hsps` launch with
/// `BLOCK_SIZE` = 128 threads = 4 warps.
pub const NUM_WARPS: usize = 4;
/// `parameters.h: BLOCK_SIZE`.
pub const BLOCK_SIZE: u32 = 128;
/// `parameters.h: MAX_BLOCKS` / `MAX_THREADS`, used by the elementwise kernels.
pub const MAX_BLOCKS: u32 = 1 << 10;
pub const MAX_THREADS: u32 = 1024;
/// Grid `find_hsps` is launched with.
///
/// KegAlign hardcodes 1024. `find_hsps` is a grid-stride loop, so this is pure
/// occupancy: measured on both mammalian workloads it improves monotonically to
/// ~8k-32k blocks and regresses again by 65536. 16384 sits at the flat optimum
/// (A -3.9% kernel, B -3.5%, both reproducible and non-overlapping).
pub const HSP_BLOCKS: u32 = 16384;
/// Threads per block for `find_hsps`, kept separate from `BLOCK_SIZE` so its
/// launch geometry can be swept without also moving `find_hits`.
pub const HSP_THREADS: u32 = NUM_WARPS as u32 * 32;
/// Threads per block for the two device-scan kernels, one element per thread.
pub const SCAN_BLOCK: u32 = 256;
/// Round 83 round 2 (`ref-loc-buckets`): threads in the single-block kernels
/// (`scan_exclusive_one_block`, `sort_survivors`), and the largest survivor
/// list the device bitonic sort can restore to ascending order. 4096 `u32` is
/// 16 KiB of shared memory; longer lists go back to the host.
#[cfg(feature = "ref-loc-buckets")]
pub const ONE_BLOCK_THREADS: u32 = 1024;
#[cfg(feature = "ref-loc-buckets")]
pub const SORT_MAX: u32 = 4096;

const WARP_SIZE: u32 = 32;
const FULL_MASK: u32 = 0xFFFF_FFFF;
/// `score_gate`'s sieve modes (`gate-sieve`): score every slot, or bound each
/// lane's own hit first with every lane loading its own reference groups.
#[cfg(feature = "dense-anchors")]
const NO_SIEVE: u32 = 0;
#[cfg(feature = "gate-sieve")]
const SIEVE_SCATTER: u32 = 1;
/// The same bound with the reference groups gathered through shared memory:
/// six lanes load the six groups of one hit, five hits per round.
#[cfg(feature = "gate-sieve")]
const SIEVE_GATHER: u32 = 2;
/// [`SIEVE_SCATTER`] over a query packed without a mask plane
/// (`sieve::QRY_FILL`, round 117): only a drop certifies a side.
#[cfg(feature = "gate-sieve")]
const SIEVE_OPEN: u32 = 3;
/// Shared `u64` slots one warp gathers into: six rows of 32 hits, each row 4
/// slots longer so that neighbouring rows fall on different banks.
#[cfg(feature = "gate-sieve")]
const GATHER_ROW: usize = 36;
#[cfg(feature = "gate-sieve")]
const GATHER_SLOTS: usize = 6 * GATHER_ROW;
/// `parameters.h: NUC`.
const NUC: usize = 8;

#[cuda_host::cuda_module]
pub mod device {
    #[cfg(feature = "dense-anchors")]
    use super::NO_SIEVE;
    use super::{FULL_MASK, NUC, NUM_WARPS, SCAN_BLOCK, SegmentPair, WARP_SIZE};
    #[cfg(feature = "gate-sieve")]
    use super::{GATHER_ROW, GATHER_SLOTS, SIEVE_GATHER, SIEVE_OPEN, SIEVE_SCATTER};
    #[cfg(feature = "dense-anchors")]
    use crate::gpu::sieve;
    use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, thread, warp};

    /// Does nothing. Launched in a loop to price a launch under ZLUDA, where
    /// every dispatch pays PTX-to-HIP translation. The
    /// single argument keeps some marshalling in the measurement; a kernel with
    /// twelve arguments costs more, so treat this as a floor.
    #[kernel]
    pub fn noop(sink: &[u32], mut out: DisjointSlice<u32>) {
        // Never true: keeps both parameters live so the launch marshals
        // arguments like a real kernel, while doing no work.
        let idx = thread::index_1d();
        if idx.get() > sink.len() + out.len() {
            unsafe { *out.get_unchecked_mut(0) = sink[0] };
        }
    }

    /// `find_num_hits`: reference hit count for every seed offset.
    #[kernel]
    pub fn find_num_hits(
        num_seeds: u32,
        index_table: &[u32],
        seed_offsets: &[u64],
        mut seed_hit_num: DisjointSlice<u32>,
    ) {
        let stride = thread::blockDim_x() * thread::gridDim_x();
        let mut id = thread::blockDim_x() * thread::blockIdx_x() + thread::threadIdx_x();

        while id < num_seeds {
            let i = id as usize;
            // N4. Both checks the safe path emits are provably
            // dead: `id < num_seeds` is the loop condition and the host sizes
            // `seed_offsets` from `seeds.len()`; `seed` is a k-mer index, so
            // `seed < 4^kmer_size == index_table.len()` (`seed.rs: kmer_at`
            // accumulates exactly `kmer_size` 2-bit digits and each transition
            // variant XORs a 2-bit mask inside that range). Removing them under
            // ZLUDA cut 24 -> 20 instructions and ran 0.7% SLOWER — that kernel
            // is gather-bound at ~1% of issue capacity. The L4 re-decides.
            #[cfg(feature = "nvidia-find-num-unchecked")]
            // SAFETY: as proven above.
            let (seed, mut num_seed_hit) = unsafe {
                let seed = (*seed_offsets.get_unchecked(i) >> 32) as usize;
                (seed, *index_table.get_unchecked(seed))
            };
            #[cfg(not(feature = "nvidia-find-num-unchecked"))]
            let (seed, mut num_seed_hit) = {
                let seed = (seed_offsets[i] >> 32) as usize;
                (seed, index_table[seed])
            };
            if seed > 0 {
                #[cfg(feature = "nvidia-find-num-unchecked")]
                // SAFETY: `0 < seed < index_table.len()`.
                let prev = unsafe { *index_table.get_unchecked(seed - 1) };
                #[cfg(not(feature = "nvidia-find-num-unchecked"))]
                let prev = index_table[seed - 1];
                num_seed_hit -= prev;
            }
            // SAFETY: `id < num_seeds <= seed_hit_num.len()`.
            unsafe {
                *seed_hit_num.get_unchecked_mut(i) = num_seed_hit;
            }
            id += stride;
        }
    }

    /// First index in `pos[lo..hi]` (half-open) whose value is `>= key`, or `hi`.
    #[inline(always)]
    fn lower_bound_in(pos: &[u32], mut lo: u32, mut hi: u32, key: u32) -> u32 {
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if pos[mid as usize] < key {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// `--paired` (`paired.rs`): [`find_num_hits`] restricted to the seed's own
    /// pair. Seed `j` belongs to pair `p = upper_bound(pair_q_start, j) - 1`,
    /// and its hits are the bucket positions in `[pair_t_lo[p], pair_t_hi[p])`.
    /// That is one contiguous slice of the bucket, because positions ascend
    /// within a bucket (`seed.rs`) and pairs are packed in order. Writes the
    /// slice length (the hit count) and where it starts.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn find_num_hits_paired(
        num_seeds: u32,
        index_table: &[u32],
        pos_table: &[u32],
        seed_offsets: &[u64],
        pair_q_start: &[u32],
        pair_t_lo: &[u32],
        pair_t_hi: &[u32],
        mut seed_hit_num: DisjointSlice<u32>,
        mut seed_lo: DisjointSlice<u32>,
    ) {
        let stride = thread::blockDim_x() * thread::gridDim_x();
        let mut id = thread::blockDim_x() * thread::blockIdx_x() + thread::threadIdx_x();

        while id < num_seeds {
            let i = id as usize;
            let seed_offset = seed_offsets[i];
            let seed = (seed_offset >> 32) as usize;
            let j = (seed_offset & 0xFFFF_FFFF) as u32;
            let end = index_table[seed];
            let start = if seed > 0 { index_table[seed - 1] } else { 0 };

            // upper_bound(pair_q_start, j) - 1; pair 0 starts at 0 <= j.
            let (mut a, mut b) = (0u32, pair_q_start.len() as u32);
            while a < b {
                let mid = a + (b - a) / 2;
                if pair_q_start[mid as usize] <= j {
                    a = mid + 1;
                } else {
                    b = mid;
                }
            }
            let p = a.saturating_sub(1) as usize;

            let lo = lower_bound_in(pos_table, start, end, pair_t_lo[p]);
            let hi = lower_bound_in(pos_table, lo, end, pair_t_hi[p]);
            // SAFETY: `id < num_seeds <=` both output lengths.
            unsafe {
                *seed_hit_num.get_unchecked_mut(i) = hi - lo;
                *seed_lo.get_unchecked_mut(i) = lo;
            }
            id += stride;
        }
    }

    /// Block-local inclusive scan of the per-seed hit counts, in place, plus
    /// this block's total into `block_sums`.
    ///
    /// Hand-rolled from `shuffle_up_sync` + shared memory rather than
    /// cooperative-groups `block_scan`: these are the primitives already proven
    /// to load under the pinned ZLUDA backend, and the pinned cuda-oxide has no
    /// device-wide scan anyway.
    #[kernel]
    pub fn scan_blocks(mut counts: DisjointSlice<u32>, mut block_sums: DisjointSlice<u32>, n: u32) {
        const WARPS: usize = (SCAN_BLOCK / WARP_SIZE) as usize;
        static mut WARP_TOTALS: SharedArray<u32, WARPS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let bid = thread::blockIdx_x();
        let i = bid * SCAN_BLOCK + tid;
        let lane = tid % WARP_SIZE;
        let warp = (tid / WARP_SIZE) as usize;

        // SAFETY: guarded by `i < n`; the tail block reads nothing past the end.
        let mut v = if i < n {
            unsafe { *counts.get_unchecked_mut(i as usize) }
        } else {
            0
        };

        let mut offset = 1;
        while offset < WARP_SIZE {
            let up = warp::shuffle_up_sync(FULL_MASK, v, offset);
            if lane >= offset {
                v += up;
            }
            offset <<= 1;
        }
        if lane == WARP_SIZE - 1 {
            unsafe { WARP_TOTALS[warp] = v };
        }
        thread::sync_threads();

        // Eight values: one thread walking them serially beats any cleverness.
        if tid == 0 {
            let mut acc = 0u32;
            for k in 0..WARPS {
                let total = unsafe { WARP_TOTALS[k] };
                unsafe { WARP_TOTALS[k] = acc };
                acc += total;
            }
            // SAFETY: one slot per block.
            unsafe { *block_sums.get_unchecked_mut(bid as usize) = acc };
        }
        thread::sync_threads();

        v += unsafe { WARP_TOTALS[warp] };
        if i < n {
            // SAFETY: guarded by `i < n`.
            unsafe { *counts.get_unchecked_mut(i as usize) = v };
        }
    }

    /// Counts byte survivor flags per contiguous block without expanding them
    /// to a per-hit `u32` scan.
    #[cfg(feature = "dense-anchors")]
    #[kernel]
    pub fn count_survivors(flags: &[u8], mut block_sums: DisjointSlice<u32>, n: u32) {
        const WARPS: usize = (SCAN_BLOCK / WARP_SIZE) as usize;
        static mut WARP_COUNTS: SharedArray<u32, WARPS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let bid = thread::blockIdx_x();
        let i = bid * SCAN_BLOCK + tid;
        let lane = tid % WARP_SIZE;
        let warp_id = (tid / WARP_SIZE) as usize;
        let keep = i < n && flags[i as usize] != 0;
        let mask = warp::ballot_sync(FULL_MASK, keep);

        if lane == 0 {
            unsafe { WARP_COUNTS[warp_id] = mask.count_ones() };
        }
        thread::sync_threads();

        if tid == 0 {
            let mut total = 0u32;
            for w in 0..WARPS {
                total += unsafe { WARP_COUNTS[w] };
            }
            // SAFETY: one slot per launched block.
            unsafe { *block_sums.get_unchecked_mut(bid as usize) = total };
        }
    }

    /// Emits original hit IDs in increasing order and clears every active flag
    /// for buffer reuse. `offsets` is the host-computed exclusive block prefix.
    #[cfg(feature = "dense-anchors")]
    #[kernel]
    pub fn emit_survivors(
        mut flags: DisjointSlice<u8>,
        offsets: &[u32],
        mut survivor_ids: DisjointSlice<u32>,
        n: u32,
    ) {
        const WARPS: usize = (SCAN_BLOCK / WARP_SIZE) as usize;
        static mut WARP_COUNTS: SharedArray<u32, WARPS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let bid = thread::blockIdx_x();
        let i = bid * SCAN_BLOCK + tid;
        let lane = tid % WARP_SIZE;
        let warp_id = (tid / WARP_SIZE) as usize;
        let keep = i < n && unsafe { *flags.get_unchecked_mut(i as usize) != 0 };
        let mask = warp::ballot_sync(FULL_MASK, keep);

        if lane == 0 {
            unsafe { WARP_COUNTS[warp_id] = mask.count_ones() };
        }
        thread::sync_threads();

        if tid == 0 {
            let mut prefix = 0u32;
            for w in 0..WARPS {
                let count = unsafe { WARP_COUNTS[w] };
                unsafe { WARP_COUNTS[w] = prefix };
                prefix += count;
            }
        }
        thread::sync_threads();

        if keep {
            let before = mask & ((1u32 << lane) - 1);
            let out = offsets[bid as usize] + unsafe { WARP_COUNTS[warp_id] } + before.count_ones();
            // SAFETY: the block prefixes and local ranks cover exactly the
            // allocated survivor range, once each and in increasing `i` order.
            unsafe { *survivor_ids.get_unchecked_mut(out as usize) = i };
        }
        if i < n {
            // SAFETY: one thread owns each active byte.
            unsafe { *flags.get_unchecked_mut(i as usize) = 0 };
        }
    }

    /// Round 83 round 2: the popcount half of `count_survivors`, over the
    /// sorted bitmask the reordered gate writes (one `u32` per 32 gate slots).
    /// One word per thread, so a block covers `32 * SCAN_BLOCK` hits and the
    /// survivor scan gets 32x fewer block sums than the byte-flag path.
    #[cfg(feature = "ref-loc-buckets")]
    #[kernel]
    pub fn count_bits(bits: &[u32], mut block_sums: DisjointSlice<u32>, n_words: u32) {
        const WARPS: usize = (SCAN_BLOCK / WARP_SIZE) as usize;
        static mut WARP_COUNTS: SharedArray<u32, WARPS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let bid = thread::blockIdx_x();
        let i = bid * SCAN_BLOCK + tid;
        let lane = tid % WARP_SIZE;
        let warp_id = (tid / WARP_SIZE) as usize;
        // Every thread stays active through the shuffles: the tail contributes 0.
        let mut c = if i < n_words {
            bits[i as usize].count_ones()
        } else {
            0
        };
        let mut off = WARP_SIZE / 2;
        while off > 0 {
            c += warp::shuffle_down_sync(FULL_MASK, c, off);
            off >>= 1;
        }
        if lane == 0 {
            unsafe { WARP_COUNTS[warp_id] = c };
        }
        thread::sync_threads();

        if tid == 0 {
            let mut total = 0u32;
            for w in 0..WARPS {
                total += unsafe { WARP_COUNTS[w] };
            }
            // SAFETY: one slot per launched block.
            unsafe { *block_sums.get_unchecked_mut(bid as usize) = total };
        }
    }

    /// Round 83 round 2: `emit_survivors` over the sorted bitmask. Set bits are
    /// enumerated in ascending bit order, so slots come out in ascending SORTED
    /// order and `sorted_idx` maps each to its raw hit id. `sort_survivors`
    /// restores ascending raw order afterwards. Nothing is cleared: the gate
    /// overwrites every active word of every chunk.
    #[cfg(feature = "ref-loc-buckets")]
    #[kernel]
    pub fn emit_bits(
        bits: &[u32],
        offsets: &[u32],
        sorted_idx: &[u32],
        mut survivor_ids: DisjointSlice<u32>,
        n_words: u32,
    ) {
        const WARPS: usize = (SCAN_BLOCK / WARP_SIZE) as usize;
        static mut WARP_COUNTS: SharedArray<u32, WARPS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let bid = thread::blockIdx_x();
        let i = bid * SCAN_BLOCK + tid;
        let lane = tid % WARP_SIZE;
        let warp_id = (tid / WARP_SIZE) as usize;
        let word = if i < n_words { bits[i as usize] } else { 0 };
        let c = word.count_ones();

        // Inclusive warp scan of the per-word counts: `s - c` is how many
        // survivors the lower lanes of this warp emit.
        let mut s = c;
        let mut off = 1;
        while off < WARP_SIZE {
            let up = warp::shuffle_up_sync(FULL_MASK, s, off);
            if lane >= off {
                s += up;
            }
            off <<= 1;
        }
        if lane == WARP_SIZE - 1 {
            unsafe { WARP_COUNTS[warp_id] = s };
        }
        thread::sync_threads();

        if tid == 0 {
            let mut prefix = 0u32;
            for w in 0..WARPS {
                let count = unsafe { WARP_COUNTS[w] };
                unsafe { WARP_COUNTS[w] = prefix };
                prefix += count;
            }
        }
        thread::sync_threads();

        let mut out = offsets[bid as usize] + unsafe { WARP_COUNTS[warp_id] } + (s - c);
        let mut m = word;
        while m != 0 {
            let slot = i * WARP_SIZE + m.trailing_zeros();
            // SAFETY: the block prefixes and per-word ranks cover exactly the
            // allocated survivor range, once each; a set bit implies the gate
            // wrote that slot, so `slot < n` indexes `sorted_idx`.
            unsafe { *survivor_ids.get_unchecked_mut(out as usize) = sorted_idx[slot as usize] };
            out += 1;
            m &= m - 1;
        }
    }

    /// `gate-sieve`: [`emit_bits`] over a bitmask indexed by raw hit id, as the
    /// plain sieve kernels write it. A slot is its own id, so the list comes out
    /// ascending and needs no sort. A copy rather than a shared body: routing
    /// `emit_bits` through a helper changed its machine code.
    #[cfg(feature = "gate-sieve")]
    #[kernel]
    pub fn emit_bits_plain(
        bits: &[u32],
        offsets: &[u32],
        mut survivor_ids: DisjointSlice<u32>,
        n_words: u32,
    ) {
        const WARPS: usize = (SCAN_BLOCK / WARP_SIZE) as usize;
        static mut WARP_COUNTS: SharedArray<u32, WARPS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let bid = thread::blockIdx_x();
        let i = bid * SCAN_BLOCK + tid;
        let lane = tid % WARP_SIZE;
        let warp_id = (tid / WARP_SIZE) as usize;
        let word = if i < n_words { bits[i as usize] } else { 0 };
        let c = word.count_ones();

        // Inclusive warp scan of the per-word counts: `s - c` is how many
        // survivors the lower lanes of this warp emit.
        let mut s = c;
        let mut off = 1;
        while off < WARP_SIZE {
            let up = warp::shuffle_up_sync(FULL_MASK, s, off);
            if lane >= off {
                s += up;
            }
            off <<= 1;
        }
        if lane == WARP_SIZE - 1 {
            unsafe { WARP_COUNTS[warp_id] = s };
        }
        thread::sync_threads();

        if tid == 0 {
            let mut prefix = 0u32;
            for w in 0..WARPS {
                let count = unsafe { WARP_COUNTS[w] };
                unsafe { WARP_COUNTS[w] = prefix };
                prefix += count;
            }
        }
        thread::sync_threads();

        let mut out = offsets[bid as usize] + unsafe { WARP_COUNTS[warp_id] } + (s - c);
        let mut m = word;
        while m != 0 {
            let slot = i * WARP_SIZE + m.trailing_zeros();
            // SAFETY: the block prefixes and per-word ranks cover exactly the
            // allocated survivor range, once each.
            unsafe { *survivor_ids.get_unchecked_mut(out as usize) = slot };
            out += 1;
            m &= m - 1;
        }
    }

    /// Round 83 round 2: restores ascending raw-hit order in the survivor list.
    /// One block, bitonic sort in shared memory, `n <= SORT_MAX`; the padding is
    /// `u32::MAX`, which no hit id can take (`n_hits <= max_hits < u32::MAX`),
    /// so the first `n` slots come out as the ids in ascending order. Downstream
    /// (`find_hsps`, `dump_raw`, the census walk, dedup ties) needs that order.
    #[cfg(feature = "ref-loc-buckets")]
    #[kernel]
    pub fn sort_survivors(mut ids: DisjointSlice<u32>, n: u32) {
        const T: u32 = super::ONE_BLOCK_THREADS;
        static mut S: SharedArray<u32, { super::SORT_MAX as usize }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let mut i = tid;
        while i < super::SORT_MAX {
            // SAFETY: `i < SORT_MAX` indexes the shared array; the caller
            // guarantees `n <= SORT_MAX` allocated ids.
            unsafe {
                S[i as usize] = if i < n {
                    *ids.get_unchecked_mut(i as usize)
                } else {
                    u32::MAX
                }
            };
            i += T;
        }
        thread::sync_threads();

        let mut k = 2u32;
        while k <= super::SORT_MAX {
            let mut j = k >> 1;
            while j > 0 {
                let mut t = tid;
                while t < super::SORT_MAX {
                    let l = t ^ j;
                    // Only the lower index of each pair swaps, so every slot is
                    // written by exactly one thread in this step.
                    if l > t {
                        let a = unsafe { S[t as usize] };
                        let b = unsafe { S[l as usize] };
                        if (a > b) == ((t & k) == 0) {
                            unsafe {
                                S[t as usize] = b;
                                S[l as usize] = a;
                            };
                        }
                    }
                    t += T;
                }
                thread::sync_threads();
                j >>= 1;
            }
            k <<= 1;
        }

        let mut i = tid;
        while i < n {
            // SAFETY: `i < n <= SORT_MAX`, one id per slot.
            unsafe { *ids.get_unchecked_mut(i as usize) = S[i as usize] };
            i += T;
        }
    }

    /// Round 83 round 2: exclusive prefix scan of an array in place, in ONE
    /// block, so the bucket histogram's block sums never cross the bus. Any
    /// length: 1024 threads walk the array in tiles carrying the running total
    /// in shared memory. Exact while the total fits `u32`, which it does — the
    /// sums count hits and `n_hits <= u32::MAX`.
    #[cfg(feature = "ref-loc-buckets")]
    #[kernel]
    pub fn scan_exclusive_one_block(mut v: DisjointSlice<u32>, n: u32) {
        const T: u32 = super::ONE_BLOCK_THREADS;
        const WARPS: usize = (T / WARP_SIZE) as usize;
        static mut WARP_TOTALS: SharedArray<u32, WARPS> = SharedArray::UNINIT;
        static mut CARRY: SharedArray<u32, 1> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let lane = tid % WARP_SIZE;
        let warp = (tid / WARP_SIZE) as usize;
        if tid == 0 {
            unsafe { CARRY[0] = 0 };
        }
        thread::sync_threads();

        // `n` and `base` are block-uniform, so every thread runs every tile and
        // reaches every barrier.
        let mut base = 0u32;
        while base < n {
            let i = base + tid;
            // SAFETY: guarded by `i < n`; the tail reads nothing past the end.
            let x = if i < n {
                unsafe { *v.get_unchecked_mut(i as usize) }
            } else {
                0
            };
            let mut s = x;
            let mut off = 1;
            while off < WARP_SIZE {
                let up = warp::shuffle_up_sync(FULL_MASK, s, off);
                if lane >= off {
                    s += up;
                }
                off <<= 1;
            }
            if lane == WARP_SIZE - 1 {
                unsafe { WARP_TOTALS[warp] = s };
            }
            thread::sync_threads();

            // Thirty-two values plus the carry: one thread walking them serially.
            if tid == 0 {
                let mut acc = unsafe { CARRY[0] };
                for w in 0..WARPS {
                    let total = unsafe { WARP_TOTALS[w] };
                    unsafe { WARP_TOTALS[w] = acc };
                    acc += total;
                }
                unsafe { CARRY[0] = acc };
            }
            thread::sync_threads();

            if i < n {
                let excl = unsafe { WARP_TOTALS[warp] } + s - x;
                // SAFETY: guarded by `i < n`.
                unsafe { *v.get_unchecked_mut(i as usize) = excl };
            }
            // Before the next tile overwrites `WARP_TOTALS`.
            thread::sync_threads();
            base += T;
        }
    }

    /// Adds each block's exclusive prefix, turning the block-local scans into
    /// the global inclusive scan `find_hits` expects.
    #[kernel]
    pub fn add_block_offsets(mut counts: DisjointSlice<u32>, offsets: &[u32], n: u32) {
        let i = thread::blockIdx_x() * SCAN_BLOCK + thread::threadIdx_x();
        if i < n {
            let add = offsets[(i / SCAN_BLOCK) as usize];
            // SAFETY: guarded by `i < n`.
            unsafe { *counts.get_unchecked_mut(i as usize) += add };
        }
    }

    /// Device seeder: validate each query window and materialize its k-mer plus the
    /// number of stable output slots it owns. The encoded query uses 0..3 for
    /// uppercase ACGT and >=4 for every byte CPU seeding rejects.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn seed_kmers(
        query: &[u8],
        shape_pos: &[u32],
        seed_size: u32,
        start: u32,
        span: u32,
        per_pos: u32,
        mut kmers: DisjointSlice<u32>,
        mut counts: DisjointSlice<u32>,
    ) {
        const INVALID: u32 = 1 << 31;
        let stride = thread::blockDim_x() * thread::gridDim_x();
        let mut id = thread::blockDim_x() * thread::blockIdx_x() + thread::threadIdx_x();
        while id < span {
            let pos = start + id;
            let mut valid = true;
            let mut j = 0;
            while j < seed_size {
                if query[(pos + j) as usize] >= 4 {
                    valid = false;
                    break;
                }
                j += 1;
            }
            let mut kmer = 0u32;
            if valid {
                let mut t = 0;
                while t < shape_pos.len() {
                    kmer = (kmer << 2) + query[(pos + shape_pos[t]) as usize] as u32;
                    t += 1;
                }
            } else {
                kmer = INVALID;
            }
            unsafe {
                *kmers.get_unchecked_mut(id as usize) = kmer;
                *counts.get_unchecked_mut(id as usize) = if valid { per_pos } else { 0 };
            }
            id += stride;
        }
    }

    /// Stable scatter for the device seeder. The scanned count is an inclusive prefix, so
    /// each valid query position writes the exact CPU order: base k-mer first,
    /// then transition variants in ascending care-position index.
    #[kernel]
    pub fn scatter_seeds(
        kmers: &[u32],
        counts: &[u32],
        start: u32,
        span: u32,
        kmer_size: u32,
        transitions: u32,
        mut seeds: DisjointSlice<u64>,
    ) {
        const INVALID: u32 = 1 << 31;
        const TRANSITION_MASK: u64 = 2;
        let stride = thread::blockDim_x() * thread::gridDim_x();
        let mut id = thread::blockDim_x() * thread::blockIdx_x() + thread::threadIdx_x();
        while id < span {
            let kmer = kmers[id as usize];
            if kmer != INVALID {
                let variants = if transitions != 0 { 1 + kmer_size } else { 1 };
                let at = counts[id as usize] - variants;
                let pos = start + id;
                unsafe {
                    *seeds.get_unchecked_mut(at as usize) = ((kmer as u64) << 32) | pos as u64
                };
                if transitions != 0 {
                    let mut t = 0;
                    while t < kmer_size {
                        let variant = (kmer as u64) ^ (TRANSITION_MASK << (2 * t));
                        unsafe {
                            *seeds.get_unchecked_mut((at + 1 + t) as usize) =
                                (variant << 32) | pos as u64;
                        }
                        t += 1;
                    }
                }
            }
            id += stride;
        }
    }

    /// `find_hits`: expands each seed into one zero-length HSP per reference
    /// occurrence, one thread per seed.
    ///
    /// Hits for a seed are written *backwards* from the seed's exclusive prefix
    /// offset, which is what fixes the order the raw HSP array comes out in.
    ///
    /// KegAlign gives each seed a whole 128-thread block and lets one lane per
    /// warp store, which the measured hits-per-seed distribution says is almost
    /// all idle launch. Block- and warp-per-seed were both implemented and
    /// benchmarked, on apple/orange and on an hg38xmm39 block, then deleted:
    /// thread-per-seed wins by 4.9x on the sparse workload and loses by only
    /// 0.9% end-to-end on the dense one, which is inside noise.
    #[cfg(not(feature = "dense-anchors"))]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn find_hits(
        index_table: &[u32],
        pos_table: &[u32],
        seed_offsets: &[u64],
        seed_size: u32,
        seed_hit_num: &[u32],
        mut hsp: DisjointSlice<SegmentPair>,
        start_seed_index: u32,
        start_hit_index: u32,
        num_seeds: u32,
    ) {
        let seed_slot = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if seed_slot >= num_seeds {
            return;
        }

        let seed_offset = seed_offsets[(seed_slot + start_seed_index) as usize];
        let seed = (seed_offset >> 32) as usize;
        let end = index_table[seed];
        let start = if seed > 0 { index_table[seed - 1] } else { 0 };
        if start == end {
            return; // the common case: no reference hit for this seed
        }

        let query_loc = (seed_offset & 0xFFFF_FFFF) as u32 + seed_size;
        let prefix = seed_hit_num[(seed_slot + start_seed_index) as usize];
        let mut j = 0u32;
        while start + j < end {
            let ref_loc = pos_table[(start + j) as usize] + seed_size;
            let dram_address = prefix
                .wrapping_sub(1)
                .wrapping_sub(start_hit_index)
                .wrapping_sub(j);
            // SAFETY: prefix ranges make every address in bounds and unique.
            // find_hsps reads these two fields, then overwrites the full record
            // before any other consumer; raw writes avoid forming a reference to
            // the still-partially-uninitialized SegmentPair.
            unsafe {
                let hit = hsp.as_mut_ptr().add(dram_address as usize);
                core::ptr::addr_of_mut!((*hit).ref_start).write(ref_loc);
                core::ptr::addr_of_mut!((*hit).query_start).write(query_loc);
            }
            j += 1;
        }
    }

    /// Dense form of `find_hits`: the two input coordinates are
    /// packed into one `u64`, halving the per-hit anchor traffic and storage.
    #[cfg(feature = "dense-anchors")]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn find_hits_dense(
        index_table: &[u32],
        pos_table: &[u32],
        seed_offsets: &[u64],
        seed_size: u32,
        seed_hit_num: &[u32],
        mut anchors: DisjointSlice<u64>,
        start_seed_index: u32,
        start_hit_index: u32,
        num_seeds: u32,
    ) {
        let seed_slot = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if seed_slot >= num_seeds {
            return;
        }

        let seed_offset = seed_offsets[(seed_slot + start_seed_index) as usize];
        let seed = (seed_offset >> 32) as usize;
        let end = index_table[seed];
        let start = if seed > 0 { index_table[seed - 1] } else { 0 };
        if start == end {
            return;
        }

        let query_loc = (seed_offset & 0xFFFF_FFFF) as u32 + seed_size;
        let prefix = seed_hit_num[(seed_slot + start_seed_index) as usize];
        let mut j = 0u32;
        while start + j < end {
            let ref_loc = pos_table[(start + j) as usize] + seed_size;
            let address = prefix
                .wrapping_sub(1)
                .wrapping_sub(start_hit_index)
                .wrapping_sub(j);
            // SAFETY: the same prefix proof as `find_hits`; every address is
            // in bounds and written exactly once.
            unsafe {
                *anchors.get_unchecked_mut(address as usize) =
                    ref_loc as u64 | ((query_loc as u64) << 32);
            }
            j += 1;
        }
    }

    /// `--paired`: [`find_hits_dense`] over the in-pair slice
    /// [`find_num_hits_paired`] found. The count comes from the inclusive
    /// prefix (`seed_hit_num[i] - seed_hit_num[i-1]`), and the walk visits the
    /// same positions in the same ascending order a standalone run of the pair
    /// visits in its whole bucket, so the anchors (and the hit order that fixes
    /// the `MAX_HITS` walk) match it.
    #[cfg(feature = "dense-anchors")]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn find_hits_dense_paired(
        pos_table: &[u32],
        seed_lo: &[u32],
        seed_offsets: &[u64],
        seed_size: u32,
        seed_hit_num: &[u32],
        mut anchors: DisjointSlice<u64>,
        start_seed_index: u32,
        start_hit_index: u32,
        num_seeds: u32,
    ) {
        let seed_slot = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if seed_slot >= num_seeds {
            return;
        }

        let i = (seed_slot + start_seed_index) as usize;
        let prefix = seed_hit_num[i];
        let before = if i > 0 { seed_hit_num[i - 1] } else { 0 };
        let count = prefix - before;
        if count == 0 {
            return;
        }

        let query_loc = (seed_offsets[i] & 0xFFFF_FFFF) as u32 + seed_size;
        let start = seed_lo[i];
        let mut j = 0u32;
        while j < count {
            let ref_loc = pos_table[(start + j) as usize] + seed_size;
            let address = prefix
                .wrapping_sub(1)
                .wrapping_sub(start_hit_index)
                .wrapping_sub(j);
            // SAFETY: the same prefix proof as `find_hits_dense`.
            unsafe {
                *anchors.get_unchecked_mut(address as usize) =
                    ref_loc as u64 | ((query_loc as u64) << 32);
            }
            j += 1;
        }
    }

    /// Round 82: adjacent lanes walk adjacent positions for one seed,
    /// coalescing both the `pos_table` reads and packed-anchor writes.
    #[cfg(feature = "find-hits-warp")]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn find_hits_dense_warp(
        index_table: &[u32],
        pos_table: &[u32],
        seed_offsets: &[u64],
        seed_size: u32,
        seed_hit_num: &[u32],
        mut anchors: DisjointSlice<u64>,
        start_seed_index: u32,
        start_hit_index: u32,
        num_seeds: u32,
    ) {
        let lane = thread::threadIdx_x() % WARP_SIZE;
        let warp = thread::threadIdx_x() / WARP_SIZE;
        let seed_slot = thread::blockIdx_x() * NUM_WARPS as u32 + warp;
        if seed_slot >= num_seeds {
            return;
        }

        let seed_offset = seed_offsets[(seed_slot + start_seed_index) as usize];
        let seed = (seed_offset >> 32) as usize;
        let end = index_table[seed];
        let start = if seed > 0 { index_table[seed - 1] } else { 0 };
        if start == end {
            return;
        }

        let query_loc = (seed_offset & 0xFFFF_FFFF) as u32 + seed_size;
        let prefix = seed_hit_num[(seed_slot + start_seed_index) as usize];
        let mut j = lane;
        while start + j < end {
            let ref_loc = pos_table[(start + j) as usize] + seed_size;
            let address = prefix
                .wrapping_sub(1)
                .wrapping_sub(start_hit_index)
                .wrapping_sub(j);
            // SAFETY: lanes partition the same prefix range the thread mapping
            // owns, so every address is in bounds and written exactly once.
            unsafe {
                *anchors.get_unchecked_mut(address as usize) =
                    ref_loc as u64 | ((query_loc as u64) << 32);
            }
            j += WARP_SIZE;
        }
    }

    /// Round 110, `find-hits-groups`: one warp expands a run of 32 consecutive
    /// seeds, with its lanes spread over the run's hits.
    ///
    /// [`find_hits_dense_warp`] gives every seed a warp, and at 22 hits per seed
    /// most of its time is what a warp pays once per seed, a chain of dependent
    /// loads (seed, index line, first positions), not the hits. Here thread `t`
    /// owns seed slot `t`. Each lane loads its own seed, the index of its first
    /// position (`seed_lo`, which [`find_num_hits_lo`] wrote next to the count,
    /// so the index table is not read a second time) and its inclusive hit
    /// prefix: three sequential streams, and the 32 chains of a run overlap.
    ///
    /// The prefix says how many of the run's hits come before each seed. Lane
    /// `l` of round `i` writes the run's hit `h = 32 i + l`: [`run_owner`] finds
    /// the seed `v` that holds it and the lane stores
    /// `pos[end_v - 1 - (h - before_v)]` at raw address `base + h`. That is
    /// seeds ascending and, inside a seed, descending position: the address
    /// `prefix - 1 - j` the per-seed kernels give hit `j`, so the anchors are
    /// the same words in the same places.
    ///
    /// Four rounds go through the loop together: their four positions are
    /// requested before any is stored, so a warp has four loads in flight. On
    /// an A100 that is what the kernel waits for (unit 0 of hg38 x mm39: 2,042
    /// ms with one round per iteration, 1,686 with two, 1,515 with four; 2,900
    /// for a warp per seed).
    #[cfg(feature = "find-hits-groups")]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn find_hits_groups(
        pos_table: &[u32],
        seed_lo: &[u32],
        seed_offsets: &[u64],
        seed_size: u32,
        seed_hit_num: &[u32],
        mut anchors: DisjointSlice<u64>,
        start_seed_index: u32,
        start_hit_index: u32,
        num_seeds: u32,
    ) {
        let lane = thread::threadIdx_x() % WARP_SIZE;
        let seed_slot = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let live = seed_slot < num_seeds;
        let (mut end, mut count, mut query_loc, mut prefix) = (0u32, 0u32, 0u32, 0u32);
        if live {
            let at = (seed_slot + start_seed_index) as usize;
            let seed_offset = seed_offsets[at];
            prefix = seed_hit_num[at];
            // The prefix is inclusive over the batch's seeds, so the seed's own
            // count is its difference from the seed before.
            let before = if at > 0 { seed_hit_num[at - 1] } else { 0 };
            count = prefix - before;
            end = seed_lo[at] + count;
            query_loc = (seed_offset & 0xFFFF_FFFF) as u32 + seed_size;
        }
        // A warp's slots are consecutive, so its live lanes are a prefix of the
        // warp and lane 0 is live whenever any lane is.
        let live_mask = warp::ballot_sync(FULL_MASK, live);
        if live_mask == 0 {
            return;
        }
        // Hits of the batch before this lane's seed; lane 0's is the run's base.
        let first = prefix.wrapping_sub(count);
        let run_first = warp::shuffle_sync(FULL_MASK, first, 0);
        // Hits of the run before this lane's seed. A lane past the chunk's seeds
        // answers `u32::MAX`, which no hit index reaches, so it is never chosen.
        let before = if live {
            first.wrapping_sub(run_first)
        } else {
            u32::MAX
        };
        let last = 31 - live_mask.leading_zeros();
        let total = warp::shuffle_sync(FULL_MASK, before.wrapping_add(count), last);
        let base = run_first.wrapping_sub(start_hit_index);

        let mut round = 0u32;
        while round < total {
            let h0 = round + lane;
            let h1 = h0 + WARP_SIZE;
            let h2 = h1 + WARP_SIZE;
            let h3 = h2 + WARP_SIZE;
            let (v0, b0) = run_owner(before, h0);
            let (v1, b1) = run_owner(before, h1);
            let (v2, b2) = run_owner(before, h2);
            let (v3, b3) = run_owner(before, h3);
            let end0 = warp::shuffle_sync(FULL_MASK, end, v0);
            let end1 = warp::shuffle_sync(FULL_MASK, end, v1);
            let end2 = warp::shuffle_sync(FULL_MASK, end, v2);
            let end3 = warp::shuffle_sync(FULL_MASK, end, v3);
            let query0 = warp::shuffle_sync(FULL_MASK, query_loc, v0);
            let query1 = warp::shuffle_sync(FULL_MASK, query_loc, v1);
            let query2 = warp::shuffle_sync(FULL_MASK, query_loc, v2);
            let query3 = warp::shuffle_sync(FULL_MASK, query_loc, v3);
            let (ok0, ok1, ok2, ok3) = (h0 < total, h1 < total, h2 < total, h3 < total);
            let (mut r0, mut r1, mut r2, mut r3) = (0u32, 0u32, 0u32, 0u32);
            if ok0 {
                r0 = pos_table[(end0 - 1 - (h0 - b0)) as usize];
            }
            if ok1 {
                r1 = pos_table[(end1 - 1 - (h1 - b1)) as usize];
            }
            if ok2 {
                r2 = pos_table[(end2 - 1 - (h2 - b2)) as usize];
            }
            if ok3 {
                r3 = pos_table[(end3 - 1 - (h3 - b3)) as usize];
            }
            // SAFETY: `base + h` runs over the run's own hit addresses for every
            // `h < total`; the prefix offsets place them inside
            // `0..iter_num_hits`, each once.
            unsafe {
                if ok0 {
                    *anchors.get_unchecked_mut(base.wrapping_add(h0) as usize) =
                        (r0 + seed_size) as u64 | ((query0 as u64) << 32);
                }
                if ok1 {
                    *anchors.get_unchecked_mut(base.wrapping_add(h1) as usize) =
                        (r1 + seed_size) as u64 | ((query1 as u64) << 32);
                }
                if ok2 {
                    *anchors.get_unchecked_mut(base.wrapping_add(h2) as usize) =
                        (r2 + seed_size) as u64 | ((query2 as u64) << 32);
                }
                if ok3 {
                    *anchors.get_unchecked_mut(base.wrapping_add(h3) as usize) =
                        (r3 + seed_size) as u64 | ((query3 as u64) << 32);
                }
            }
            round += 4 * WARP_SIZE;
        }
    }

    /// The seed of a run that holds hit `h`: the last lane with `before <= h`,
    /// and that lane's `before`. Seeds without hits share their `before` with
    /// the next one, so the last such seed always has hits (if `h` is below the
    /// run's total) and is the one that holds `h`. Five probes, each answered
    /// by the lane that owns the probed seed; every lane of the warp must call
    /// it.
    #[cfg(feature = "find-hits-groups")]
    #[inline(always)]
    fn run_owner(before: u32, h: u32) -> (u32, u32) {
        let (mut v, mut before_v) = (0u32, 0u32);
        let mut step = 16u32;
        while step > 0 {
            let b = warp::shuffle_sync(FULL_MASK, before, v + step);
            if b <= h {
                v += step;
                before_v = b;
            }
            step >>= 1;
        }
        (v, before_v)
    }

    /// [`run_owner`] without the owner's `before`: the last lane whose `before`
    /// is at most `h`. With a lane's first position index less its `before`
    /// shuffled from the owner, the index of hit `h` is that value plus `h`.
    #[cfg(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    ))]
    #[inline(always)]
    fn run_owner_lane(before: u32, h: u32) -> u32 {
        let mut v = 0u32;
        let mut step = 16u32;
        while step > 0 {
            if warp::shuffle_sync(FULL_MASK, before, v + step) <= h {
                v += step;
            }
            step >>= 1;
        }
        v
    }

    /// Edge `w` of a k-mer whose positions are `pos_table[start..end]`: the
    /// index of its first position whose anchor falls in bucket `w` or a later
    /// one, so `edge(w)..edge(w + 1)` is the k-mer's slice of bucket `w`.
    /// Positions ascend within a k-mer (`seed.rs`), which is what makes each
    /// slice contiguous. The bucket is the bucket pass's:
    /// `min((pos + seed_size) >> shift, n_buckets - 1)`.
    #[cfg(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    ))]
    #[inline(always)]
    fn bucket_edge(
        pos_table: &[u32],
        start: u32,
        end: u32,
        w: u32,
        shift: u32,
        seed_size: u32,
        n_buckets: u32,
    ) -> u32 {
        if w == 0 {
            start
        } else if w >= n_buckets {
            end
        } else {
            lower_bound_in(
                pos_table,
                start,
                end,
                (w << shift).saturating_sub(seed_size),
            )
        }
    }

    /// Whether a k-mer's word of [`bucket_nibbles`] points at a row of edges
    /// instead of holding the counts.
    #[cfg(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    ))]
    #[inline(always)]
    fn nibble_escapes(w: u32) -> bool {
        w & 0xFF == 0xFF
    }

    /// Round 119: a k-mer's hits per bucket in four bits each, for at most
    /// eight buckets. A table of every k-mer's bucket edges would be eight or
    /// sixteen times the index table, and one random row of it per seed costs
    /// three times what `find_num_hits` pays per seed on an A100, whose L2 holds
    /// most of the index table and little of that; this one is the size of the
    /// index table.
    ///
    /// A count above 14 does not fit: such a k-mer's word is `0xFF` (no word of
    /// counts has two nibbles of 15) and `sat[k]` is 1, for
    /// [`bucket_overflow_rows`] to give it a row of edges.
    #[cfg(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    ))]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn bucket_nibbles(
        index_table: &[u32],
        pos_table: &[u32],
        seed_size: u32,
        shift: u32,
        n_buckets: u32,
        n_kmers: u32,
        mut nibbles: DisjointSlice<u32>,
        mut sat: DisjointSlice<u32>,
    ) {
        let stride = thread::blockDim_x() * thread::gridDim_x();
        let mut k = thread::blockDim_x() * thread::blockIdx_x() + thread::threadIdx_x();
        while k < n_kmers {
            let end = index_table[k as usize];
            let start = if k > 0 {
                index_table[(k - 1) as usize]
            } else {
                0
            };
            let (mut w, mut over) = (0u32, 0u32);
            let mut lo = start;
            let mut b = 0u32;
            while b < n_buckets {
                let hi = bucket_edge(pos_table, start, end, b + 1, shift, seed_size, n_buckets);
                let c = hi - lo;
                if c > 14 {
                    over = 1;
                }
                w |= c.min(14) << (4 * b);
                lo = hi;
                b += 1;
            }
            // SAFETY: `k < n_kmers`, the length of both.
            unsafe {
                *nibbles.get_unchecked_mut(k as usize) = if over != 0 { 0xFF } else { w };
                *sat.get_unchecked_mut(k as usize) = over;
            }
            k += stride;
        }
    }

    /// Round 119: the rows of the k-mers [`bucket_nibbles`] could not hold.
    /// `sat_incl` is the inclusive scan of its `sat`, so a k-mer whose value
    /// differs from the one before it is the `sat_incl[k] - 1`-th of them: its
    /// `1 << row_shift` edges go to that row (the words past edge `n_buckets`
    /// repeat the last edge) and its word becomes `0xFF | row << 8`.
    #[cfg(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    ))]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn bucket_overflow_rows(
        index_table: &[u32],
        pos_table: &[u32],
        seed_size: u32,
        shift: u32,
        n_buckets: u32,
        row_shift: u32,
        n_kmers: u32,
        sat_incl: &[u32],
        mut nibbles: DisjointSlice<u32>,
        mut rows: DisjointSlice<u32>,
    ) {
        let stride = thread::blockDim_x() * thread::gridDim_x();
        let mut k = thread::blockDim_x() * thread::blockIdx_x() + thread::threadIdx_x();
        while k < n_kmers {
            let here = sat_incl[k as usize];
            let before = if k > 0 { sat_incl[(k - 1) as usize] } else { 0 };
            if here != before {
                let row = here - 1;
                let end = index_table[k as usize];
                let start = if k > 0 {
                    index_table[(k - 1) as usize]
                } else {
                    0
                };
                let mut w = 0u32;
                while w < 1 << row_shift {
                    let edge = bucket_edge(pos_table, start, end, w, shift, seed_size, n_buckets);
                    // SAFETY: row `row` is below the number of saturated
                    // k-mers, which sized `rows`; this k-mer owns it.
                    unsafe { *rows.get_unchecked_mut(((row << row_shift) + w) as usize) = edge };
                    w += 1;
                }
                // SAFETY: `k < n_kmers`.
                unsafe { *nibbles.get_unchecked_mut(k as usize) = 0xFF | row << 8 };
            }
            k += stride;
        }
    }

    /// Round 119: how many hits each run of 32 seeds of a chunk
    /// has in each bucket, from the seeds' words of [`bucket_nibbles`]: one
    /// thread per seed, one warp per run (the runs of
    /// [`find_hits_groups_bucketed`]), no anchor read. Bucket-major and
    /// run-minor, so one inclusive scan gives every run the end of its slice of
    /// every bucket. Each seed's word is kept in `seed_nib` for the group
    /// kernel, which then reads no table for a seed whose counts fit; a seed
    /// whose word points at a row reads its edges there.
    #[cfg(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    ))]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn run_bucket_counts(
        seed_offsets: &[u64],
        nibbles: &[u32],
        bounds: &[u32],
        start_seed_index: u32,
        num_seeds: u32,
        n_buckets: u32,
        row_shift: u32,
        num_runs: u32,
        mut counts: DisjointSlice<u32>,
        mut seed_nib: DisjointSlice<u32>,
    ) {
        let tid = thread::threadIdx_x();
        let lane = tid % WARP_SIZE;
        let seed_slot = thread::blockIdx_x() * thread::blockDim_x() + tid;
        let run = seed_slot / WARP_SIZE;
        let live = seed_slot < num_seeds;
        // The live lanes of a warp are its first ones: a warp with one has
        // lane 0, and `run < num_runs`.
        if warp::ballot_sync(FULL_MASK, live) == 0 {
            return;
        }
        let (mut w, mut row, mut lo, mut esc) = (0u32, 0usize, 0u32, false);
        if live {
            // SAFETY: a seed of this chunk; its k-mer indexes the words, a
            // word that escapes names a row, and the seed owns slot
            // `seed_slot` of `seed_nib`.
            unsafe {
                let at = (seed_slot + start_seed_index) as usize;
                let k = (*seed_offsets.get_unchecked(at) >> 32) as u32;
                w = *nibbles.get_unchecked(k as usize);
                *seed_nib.get_unchecked_mut(seed_slot as usize) = w;
                esc = nibble_escapes(w);
                if esc {
                    row = ((w >> 8) << row_shift) as usize;
                    lo = *bounds.get_unchecked(row);
                }
            }
        }
        let mut b = 0u32;
        while b < n_buckets {
            let mut c = (w >> (4 * b)) & 15;
            if esc {
                // SAFETY: edge `b + 1 <= n_buckets` of the k-mer's row.
                let hi = unsafe { *bounds.get_unchecked(row + b as usize + 1) };
                c = hi - lo;
                lo = hi;
            }
            c += warp::shuffle_sync(FULL_MASK, c, lane ^ 16);
            c += warp::shuffle_sync(FULL_MASK, c, lane ^ 8);
            c += warp::shuffle_sync(FULL_MASK, c, lane ^ 4);
            c += warp::shuffle_sync(FULL_MASK, c, lane ^ 2);
            c += warp::shuffle_sync(FULL_MASK, c, lane ^ 1);
            if lane == 0 {
                // SAFETY: lane 0 of run `run < num_runs` owns this slot.
                unsafe { *counts.get_unchecked_mut((b * num_runs + run) as usize) = c };
            }
            b += 1;
        }
    }

    /// Round 119: [`find_hits_groups`] and the bucket pass in one.
    /// A warp expands a run of 32 seeds as there, and stores each anchor
    /// straight at its place in bucket order instead of at its raw address.
    ///
    /// `incl` is the inclusive scan of [`run_bucket_counts`], so the word before
    /// a run's own is where its slice of that bucket starts. A seed's hits of
    /// one bucket are one contiguous slice of `pos_table`, `e[b]..e[b + 1]`: its
    /// start and the counts in its word of `seed_nib`, or the row of edges that
    /// word names when a count does not fit four bits.
    ///
    /// The run's hits are taken bucket by bucket. Inside bucket `b` they are
    /// numbered `g = 0..total` in seed order, a prefix sum across the warp says
    /// how many come before each seed, and lane `l` of a round takes hit
    /// `round + l`: it finds the seed that holds it with five shuffles, loads
    /// one position and stores at `start + g`. So one store instruction writes
    /// 32 consecutive slots, as the raw kernel does. (Taking the hits seed by
    /// seed and looking each one's place up stored about three slots of a
    /// bucket at a time, parts of 32-byte sectors, and the kernel was 100 ms of
    /// unit 0 slower per bucket.) The order inside a bucket is not the raw
    /// order, and since round 116 nothing needs it to be.
    ///
    /// At most eight buckets (the host falls back above that).
    #[cfg(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    ))]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn find_hits_groups_bucketed(
        pos_table: &[u32],
        bounds: &[u32],
        seed_nib: &[u32],
        seed_offsets: &[u64],
        seed_size: u32,
        seed_lo: &[u32],
        incl: &[u32],
        n_buckets: u32,
        row_shift: u32,
        num_runs: u32,
        start_seed_index: u32,
        num_seeds: u32,
        mut sorted_anchor: DisjointSlice<u64>,
    ) {
        let tid = thread::threadIdx_x();
        let lane = tid % WARP_SIZE;
        let seed_slot = thread::blockIdx_x() * thread::blockDim_x() + tid;
        let run = seed_slot / WARP_SIZE;
        let live = seed_slot < num_seeds;
        let mut query_loc = 0u32;
        let (mut w, mut row, mut lo, mut esc) = (0u32, 0usize, 0u32, false);
        if live {
            let at = (seed_slot + start_seed_index) as usize;
            // SAFETY: `at` is a seed of this chunk; `run_bucket_counts` wrote
            // its word at slot `seed_slot` of `seed_nib`, and a word that
            // escapes names a row.
            unsafe {
                let seed_offset = *seed_offsets.get_unchecked(at);
                query_loc = (seed_offset & 0xFFFF_FFFF) as u32 + seed_size;
                w = *seed_nib.get_unchecked(seed_slot as usize);
                lo = *seed_lo.get_unchecked(at);
                esc = nibble_escapes(w);
                if esc {
                    row = ((w >> 8) << row_shift) as usize;
                }
            }
        }
        // The live lanes of a warp are its first ones.
        if warp::ballot_sync(FULL_MASK, live) == 0 {
            return;
        }
        // Lane `b` fetches where this run's slice of bucket `b` starts.
        let mut run_start = 0u32;
        if lane < n_buckets {
            let flat = lane * num_runs + run;
            if flat > 0 {
                // SAFETY: a live warp's run is below `num_runs`, and `incl`
                // holds `n_buckets * num_runs` words.
                run_start = unsafe { *incl.get_unchecked(flat as usize - 1) };
            }
        }

        let mut b = 0u32;
        while b < n_buckets {
            // This seed's hits in bucket `b`: `pos_table[lo..lo + c]`.
            let mut c = (w >> (4 * b)) & 15;
            if esc {
                // SAFETY: edge `b + 1 <= n_buckets` of the k-mer's row.
                c = unsafe { *bounds.get_unchecked(row + b as usize + 1) } - lo;
            }
            // What this lane and the lanes before it hold in the bucket.
            let mut x = c;
            let mut d = 1u32;
            while d < WARP_SIZE {
                let t = warp::shuffle_sync(FULL_MASK, x, lane.wrapping_sub(d) & (WARP_SIZE - 1));
                if lane >= d {
                    x += t;
                }
                d <<= 1;
            }
            let total = warp::shuffle_sync(FULL_MASK, x, WARP_SIZE - 1);
            let before = x - c;
            let start = warp::shuffle_sync(FULL_MASK, run_start, b);
            // Hit `g` of the bucket, held by seed `v`, is the position at index
            // `lo_v + (g - before_v)`: this lane's part of it.
            let pos_key = lo.wrapping_sub(before);

            let mut round = 0u32;
            while round < total {
                let g0 = round + lane;
                let g1 = g0 + WARP_SIZE;
                let g2 = g1 + WARP_SIZE;
                let g3 = g2 + WARP_SIZE;
                let v0 = run_owner_lane(before, g0);
                let v1 = run_owner_lane(before, g1);
                let v2 = run_owner_lane(before, g2);
                let v3 = run_owner_lane(before, g3);
                let at0 = warp::shuffle_sync(FULL_MASK, pos_key, v0).wrapping_add(g0);
                let at1 = warp::shuffle_sync(FULL_MASK, pos_key, v1).wrapping_add(g1);
                let at2 = warp::shuffle_sync(FULL_MASK, pos_key, v2).wrapping_add(g2);
                let at3 = warp::shuffle_sync(FULL_MASK, pos_key, v3).wrapping_add(g3);
                let query0 = warp::shuffle_sync(FULL_MASK, query_loc, v0);
                let query1 = warp::shuffle_sync(FULL_MASK, query_loc, v1);
                let query2 = warp::shuffle_sync(FULL_MASK, query_loc, v2);
                let query3 = warp::shuffle_sync(FULL_MASK, query_loc, v3);
                let (ok0, ok1, ok2, ok3) = (g0 < total, g1 < total, g2 < total, g3 < total);
                let (mut r0, mut r1, mut r2, mut r3) = (0u32, 0u32, 0u32, 0u32);
                // SAFETY: for `g < total` the index is one of the owner's
                // positions of this bucket.
                unsafe {
                    if ok0 {
                        r0 = *pos_table.get_unchecked(at0 as usize) + seed_size;
                    }
                    if ok1 {
                        r1 = *pos_table.get_unchecked(at1 as usize) + seed_size;
                    }
                    if ok2 {
                        r2 = *pos_table.get_unchecked(at2 as usize) + seed_size;
                    }
                    if ok3 {
                        r3 = *pos_table.get_unchecked(at3 as usize) + seed_size;
                    }
                }
                // SAFETY: the scanned counts give the run `total` slots of the
                // bucket from `start`, inside `0..iter_num_hits`, and hit `g`
                // takes slot `start + g`.
                unsafe {
                    if ok0 {
                        *sorted_anchor.get_unchecked_mut((start + g0) as usize) =
                            r0 as u64 | ((query0 as u64) << 32);
                    }
                    if ok1 {
                        *sorted_anchor.get_unchecked_mut((start + g1) as usize) =
                            r1 as u64 | ((query1 as u64) << 32);
                    }
                    if ok2 {
                        *sorted_anchor.get_unchecked_mut((start + g2) as usize) =
                            r2 as u64 | ((query2 as u64) << 32);
                    }
                    if ok3 {
                        *sorted_anchor.get_unchecked_mut((start + g3) as usize) =
                            r3 as u64 | ((query3 as u64) << 32);
                    }
                }
                round += 4 * WARP_SIZE;
            }
            lo += c;
            b += 1;
        }
    }

    /// [`find_num_hits`] that also writes where each seed's positions start in
    /// `pos_table`, for [`find_hits_groups`].
    #[cfg(feature = "find-hits-groups")]
    #[kernel]
    pub fn find_num_hits_lo(
        num_seeds: u32,
        index_table: &[u32],
        seed_offsets: &[u64],
        mut seed_hit_num: DisjointSlice<u32>,
        mut seed_lo: DisjointSlice<u32>,
    ) {
        let stride = thread::blockDim_x() * thread::gridDim_x();
        let mut id = thread::blockDim_x() * thread::blockIdx_x() + thread::threadIdx_x();
        while id < num_seeds {
            let i = id as usize;
            let seed = (seed_offsets[i] >> 32) as usize;
            let end = index_table[seed];
            let start = if seed > 0 { index_table[seed - 1] } else { 0 };
            // SAFETY: `id < num_seeds`, and the host sizes both buffers for it.
            unsafe {
                *seed_hit_num.get_unchecked_mut(i) = end - start;
                *seed_lo.get_unchecked_mut(i) = start;
            }
            id += stride;
        }
    }

    /// Round 83, `ref-loc-buckets`: per-block histogram of the reference
    /// address bucket `(anchor as u32) >> shift`, laid out bucket-major and
    /// block-minor so one inclusive scan of `counts` gives every block its
    /// slice of every bucket. `n_buckets <= 32` is a host invariant.
    #[cfg(feature = "ref-loc-buckets")]
    #[kernel]
    pub fn bucket_count(
        anchors: &[u64],
        n: u32,
        shift: u32,
        n_blocks: u32,
        n_buckets: u32,
        mut counts: DisjointSlice<u32>,
    ) {
        const WARPS: usize = (SCAN_BLOCK / WARP_SIZE) as usize;
        static mut WARP_COUNTS: SharedArray<u32, { WARPS * 32 }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let bid = thread::blockIdx_x();
        let i = bid * SCAN_BLOCK + tid;
        let lane = tid % WARP_SIZE;
        let warp_id = (tid / WARP_SIZE) as usize;
        // `u32::MAX` for the tail keeps those lanes out of every ballot. The
        // clamp only matters if a reference address ever exceeded `ref_len`;
        // any assignment inside `0..n_buckets` still yields a valid stable
        // permutation, so it cannot corrupt the output.
        let b = if i < n {
            let raw = (anchors[i as usize] as u32) >> shift;
            if raw < n_buckets { raw } else { n_buckets - 1 }
        } else {
            u32::MAX
        };

        let mut v = 0u32;
        while v < n_buckets {
            let mask = warp::ballot_sync(FULL_MASK, b == v);
            if lane == 0 {
                unsafe { WARP_COUNTS[warp_id * 32 + v as usize] = mask.count_ones() };
            }
            v += 1;
        }
        thread::sync_threads();

        if tid < n_buckets {
            let mut total = 0u32;
            for w in 0..WARPS {
                total += unsafe { WARP_COUNTS[w * 32 + tid as usize] };
            }
            // SAFETY: `n_buckets * n_blocks` slots, one per (bucket, block).
            unsafe { *counts.get_unchecked_mut((tid * n_blocks + bid) as usize) = total };
        }
    }

    /// Round 83: the scatter half of the stable counting sort. `incl` is the
    /// global inclusive scan of `bucket_count`'s output, so subtracting this
    /// block's own count gives the base of its slice. Bucket-major, and inside
    /// a bucket the original hit order — deterministic, and no atomics.
    #[cfg(feature = "ref-loc-buckets")]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn bucket_scatter(
        anchors: &[u64],
        n: u32,
        shift: u32,
        n_blocks: u32,
        n_buckets: u32,
        incl: &[u32],
        mut sorted_anchor: DisjointSlice<u64>,
        mut sorted_idx: DisjointSlice<u32>,
    ) {
        const WARPS: usize = (SCAN_BLOCK / WARP_SIZE) as usize;
        static mut WARP_PREFIX: SharedArray<u32, { WARPS * 32 }> = SharedArray::UNINIT;
        static mut BLOCK_TOTAL: SharedArray<u32, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let bid = thread::blockIdx_x();
        let i = bid * SCAN_BLOCK + tid;
        let lane = tid % WARP_SIZE;
        let warp_id = (tid / WARP_SIZE) as usize;
        let anchor = if i < n { anchors[i as usize] } else { 0 };
        let b = if i < n {
            let raw = (anchor as u32) >> shift;
            if raw < n_buckets { raw } else { n_buckets - 1 }
        } else {
            u32::MAX
        };

        // The same ballots `bucket_count` ran, plus this lane's own one kept.
        let mut my_mask = 0u32;
        let mut v = 0u32;
        while v < n_buckets {
            let mask = warp::ballot_sync(FULL_MASK, b == v);
            if lane == 0 {
                unsafe { WARP_PREFIX[warp_id * 32 + v as usize] = mask.count_ones() };
            }
            if b == v {
                my_mask = mask;
            }
            v += 1;
        }
        thread::sync_threads();

        if tid < n_buckets {
            let mut prefix = 0u32;
            for w in 0..WARPS {
                let count = unsafe { WARP_PREFIX[w * 32 + tid as usize] };
                unsafe { WARP_PREFIX[w * 32 + tid as usize] = prefix };
                prefix += count;
            }
            unsafe { BLOCK_TOTAL[tid as usize] = prefix };
        }
        thread::sync_threads();

        if i < n {
            let bu = b as usize;
            let rank = (my_mask & warp::lanemask_lt()).count_ones();
            let block_base = incl[(b * n_blocks + bid) as usize] - unsafe { BLOCK_TOTAL[bu] };
            let out = block_base + unsafe { WARP_PREFIX[warp_id * 32 + bu] } + rank;
            // SAFETY: the block bases and local ranks partition `0..n` exactly,
            // one slot per hit.
            unsafe {
                *sorted_anchor.get_unchecked_mut(out as usize) = anchor;
                *sorted_idx.get_unchecked_mut(out as usize) = i;
            }
        }
    }

    /// [`bucket_count`] without its loop (round 114): the same counts, word for
    /// word.
    ///
    /// `bucket_count` and `bucket_scatter` take one vote per bucket, seven trips
    /// of a loop that every lane of every warp walks for the windows of an A100,
    /// and they were measured to pay for each trip (66 ms per bucket on a unit
    /// of 125.6e9 hits). One `match.any` says the same: each lane gets the lanes
    /// of its warp that hold its bucket. The first of them stores the bucket's
    /// count for the warp, over the zero that lane `v` stores for bucket `v`
    /// (a warp barrier between the two stores orders them), and a lane's rank in
    /// the scatter is the number of those lanes before it.
    #[cfg(feature = "ref-loc-buckets")]
    #[kernel]
    pub fn bucket_count_match(
        anchors: &[u64],
        n: u32,
        shift: u32,
        n_blocks: u32,
        n_buckets: u32,
        mut counts: DisjointSlice<u32>,
    ) {
        const WARPS: usize = (SCAN_BLOCK / WARP_SIZE) as usize;
        static mut WARP_COUNTS: SharedArray<u32, { WARPS * 32 }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let bid = thread::blockIdx_x();
        let i = bid * SCAN_BLOCK + tid;
        let lane = tid % WARP_SIZE;
        let warp_id = (tid / WARP_SIZE) as usize;
        let live = i < n;
        // The clamp is `bucket_count`'s. Lanes past the end share a key that no
        // bucket has (`n_buckets <= 32`).
        let b = if live {
            let raw = (anchors[i as usize] as u32) >> shift;
            if raw < n_buckets { raw } else { n_buckets - 1 }
        } else {
            32
        };
        let own = warp::match_any_sync(FULL_MASK, b);
        if lane < n_buckets {
            unsafe { WARP_COUNTS[warp_id * 32 + lane as usize] = 0 };
        }
        // The count below goes over that zero, and it is another lane's store:
        // the barrier is what orders the two (the lanes of a warp are free to
        // run the two `if`s in any order otherwise).
        warp::sync_mask(FULL_MASK);
        if live && (own & warp::lanemask_lt()) == 0 {
            unsafe { WARP_COUNTS[warp_id * 32 + b as usize] = own.count_ones() };
        }
        thread::sync_threads();

        if tid < n_buckets {
            let mut total = 0u32;
            for w in 0..WARPS {
                total += unsafe { WARP_COUNTS[w * 32 + tid as usize] };
            }
            // SAFETY: `n_buckets * n_blocks` slots, one per (bucket, block).
            unsafe { *counts.get_unchecked_mut((tid * n_blocks + bid) as usize) = total };
        }
    }

    /// [`bucket_scatter`] without its loop, as [`bucket_count_match`]: the same
    /// two buffers, word for word.
    #[cfg(feature = "ref-loc-buckets")]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn bucket_scatter_match(
        anchors: &[u64],
        n: u32,
        shift: u32,
        n_blocks: u32,
        n_buckets: u32,
        incl: &[u32],
        mut sorted_anchor: DisjointSlice<u64>,
        mut sorted_idx: DisjointSlice<u32>,
    ) {
        const WARPS: usize = (SCAN_BLOCK / WARP_SIZE) as usize;
        static mut WARP_PREFIX: SharedArray<u32, { WARPS * 32 }> = SharedArray::UNINIT;
        static mut BLOCK_TOTAL: SharedArray<u32, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let bid = thread::blockIdx_x();
        let i = bid * SCAN_BLOCK + tid;
        let lane = tid % WARP_SIZE;
        let warp_id = (tid / WARP_SIZE) as usize;
        let live = i < n;
        let anchor = if live { anchors[i as usize] } else { 0 };
        let b = if live {
            let raw = (anchor as u32) >> shift;
            if raw < n_buckets { raw } else { n_buckets - 1 }
        } else {
            32
        };
        let own = warp::match_any_sync(FULL_MASK, b);
        let before = own & warp::lanemask_lt();
        if lane < n_buckets {
            unsafe { WARP_PREFIX[warp_id * 32 + lane as usize] = 0 };
        }
        // As in `bucket_count_match`: the barrier orders the zero and the count.
        warp::sync_mask(FULL_MASK);
        if live && before == 0 {
            unsafe { WARP_PREFIX[warp_id * 32 + b as usize] = own.count_ones() };
        }
        thread::sync_threads();

        if tid < n_buckets {
            let mut prefix = 0u32;
            for w in 0..WARPS {
                let count = unsafe { WARP_PREFIX[w * 32 + tid as usize] };
                unsafe { WARP_PREFIX[w * 32 + tid as usize] = prefix };
                prefix += count;
            }
            unsafe { BLOCK_TOTAL[tid as usize] = prefix };
        }
        thread::sync_threads();

        if live {
            let bu = b as usize;
            let block_base = incl[(b * n_blocks + bid) as usize] - unsafe { BLOCK_TOTAL[bu] };
            let out = block_base + unsafe { WARP_PREFIX[warp_id * 32 + bu] } + before.count_ones();
            // SAFETY: the block bases and local ranks partition `0..n` exactly,
            // one slot per hit.
            unsafe {
                *sorted_anchor.get_unchecked_mut(out as usize) = anchor;
                *sorted_idx.get_unchecked_mut(out as usize) = i;
            }
        }
    }

    /// [`bucket_scatter_match`] without the raw hit ids (round 116): the same
    /// `sorted_anchor`, word for word, and nothing else. For a chunk whose
    /// survivors go to `find_hsps` by sorted slot: twelve bytes of a hit's twenty
    /// were the scatter's own stores, four of them the id. A copy rather than a
    /// shared body, as for `emit_bits_plain`.
    #[cfg(feature = "ref-loc-buckets")]
    #[kernel]
    pub fn bucket_scatter_match_anchors(
        anchors: &[u64],
        n: u32,
        shift: u32,
        n_blocks: u32,
        n_buckets: u32,
        incl: &[u32],
        mut sorted_anchor: DisjointSlice<u64>,
    ) {
        const WARPS: usize = (SCAN_BLOCK / WARP_SIZE) as usize;
        static mut WARP_PREFIX: SharedArray<u32, { WARPS * 32 }> = SharedArray::UNINIT;
        static mut BLOCK_TOTAL: SharedArray<u32, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let bid = thread::blockIdx_x();
        let i = bid * SCAN_BLOCK + tid;
        let lane = tid % WARP_SIZE;
        let warp_id = (tid / WARP_SIZE) as usize;
        let live = i < n;
        let anchor = if live { anchors[i as usize] } else { 0 };
        let b = if live {
            let raw = (anchor as u32) >> shift;
            if raw < n_buckets { raw } else { n_buckets - 1 }
        } else {
            32
        };
        let own = warp::match_any_sync(FULL_MASK, b);
        let before = own & warp::lanemask_lt();
        if lane < n_buckets {
            unsafe { WARP_PREFIX[warp_id * 32 + lane as usize] = 0 };
        }
        // As in `bucket_count_match`: the barrier orders the zero and the count.
        warp::sync_mask(FULL_MASK);
        if live && before == 0 {
            unsafe { WARP_PREFIX[warp_id * 32 + b as usize] = own.count_ones() };
        }
        thread::sync_threads();

        if tid < n_buckets {
            let mut prefix = 0u32;
            for w in 0..WARPS {
                let count = unsafe { WARP_PREFIX[w * 32 + tid as usize] };
                unsafe { WARP_PREFIX[w * 32 + tid as usize] = prefix };
                prefix += count;
            }
            unsafe { BLOCK_TOTAL[tid as usize] = prefix };
        }
        thread::sync_threads();

        if live {
            let bu = b as usize;
            let block_base = incl[(b * n_blocks + bid) as usize] - unsafe { BLOCK_TOTAL[bu] };
            let out = block_base + unsafe { WARP_PREFIX[warp_id * 32 + bu] } + before.count_ones();
            // SAFETY: the block bases and local ranks partition `0..n` exactly,
            // one slot per hit.
            unsafe { *sorted_anchor.get_unchecked_mut(out as usize) = anchor };
        }
    }

    /// Runs only the promoted score gate and writes one byte for every hit.
    /// The rare survivors are materialized later from a stable compacted ID
    /// list, so the common reject path writes neither an HSP nor a `u32` flag.
    #[cfg(feature = "dense-anchors")]
    // Round 56: `ptxas` sizes registers for an unknown block without this, and the
    // gate is 87% of device time on the whole-genome pair. `HSP_THREADS` is the only
    // shape it is ever launched with.
    #[kernel]
    #[launch_bounds(super::HSP_THREADS)]
    #[allow(clippy::too_many_arguments)]
    pub fn mark_score_survivors(
        ref_seq: &[u8],
        query_seq: &[u8],
        ref_len: u32,
        query_len: u32,
        sub_mat: &[i32],
        xdrop: i32,
        hspthresh: i32,
        num_hits: u32,
        anchors: &[u64],
        mut flags: DisjointSlice<u8>,
    ) {
        // `reordered` is a literal and the body is `inline(always)`, so the
        // permuted path — including the bitmask store, whose pointer is null
        // here and never dereferenced — folds away and this kernel keeps its
        // pre-feature shape. `thread_id` is read here, inside the kernel, so the
        // `launch_bounds` range on `%tid.x` still reaches the body after
        // inlining.
        score_gate(
            thread::threadIdx_x(),
            ref_seq,
            query_seq,
            ref_len,
            query_len,
            sub_mat,
            xdrop,
            hspthresh,
            num_hits,
            anchors,
            false,
            flags.as_mut_ptr(),
            core::ptr::null_mut(),
            NO_SIEVE,
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            0,
            0,
            0,
            0,
            0,
            0,
            core::ptr::null_mut(),
        );
    }

    /// The same gate over anchors permuted into reference-address buckets.
    /// Round 83 round 2: instead of scattering a flag byte to the raw hit id,
    /// each warp accumulates its chunk's 32 keep bits and lane 0 stores ONE
    /// `u32` at the chunk's own sorted slot — coalesced, 1/32 of the bytes, and
    /// the gate no longer reads the `orig` map at all. `emit_bits` maps sorted
    /// slots back through `sorted_idx` and `sort_survivors` restores ascending
    /// raw order, so `find_hsps` is untouched.
    #[cfg(feature = "ref-loc-buckets")]
    #[kernel]
    #[launch_bounds(super::HSP_THREADS)]
    #[allow(clippy::too_many_arguments)]
    pub fn mark_score_survivors_reordered(
        ref_seq: &[u8],
        query_seq: &[u8],
        ref_len: u32,
        query_len: u32,
        sub_mat: &[i32],
        xdrop: i32,
        hspthresh: i32,
        num_hits: u32,
        anchors: &[u64],
        mut flags_bits: DisjointSlice<u32>,
    ) {
        score_gate(
            thread::threadIdx_x(),
            ref_seq,
            query_seq,
            ref_len,
            query_len,
            sub_mat,
            xdrop,
            hspthresh,
            num_hits,
            anchors,
            true,
            core::ptr::null_mut(),
            flags_bits.as_mut_ptr(),
            NO_SIEVE,
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            0,
            0,
            0,
            0,
            0,
            0,
            core::ptr::null_mut(),
        );
    }

    /// [`mark_score_survivors`] behind the lane-local bound of `gpu/sieve.rs`:
    /// each lane first bounds its own hit from the packed planes, and the warp
    /// scores only the hits the bound could not reject. `um`, `ua`, `ub` and
    /// `ubias` are `Classes::kernel_args`; `hspthresh` is `keep_threshold`.
    /// With `count` set, lane 0 also stores each chunk's candidate mask. The
    /// flags are one word of keep bits per 32 hits at the raw hit id, read by
    /// `count_bits` and `emit_bits_plain`.
    #[cfg(feature = "gate-sieve")]
    #[kernel]
    #[launch_bounds(super::HSP_THREADS)]
    #[allow(clippy::too_many_arguments)]
    pub fn mark_score_survivors_sieve(
        ref_seq: &[u8],
        query_seq: &[u8],
        ref_len: u32,
        query_len: u32,
        sub_mat: &[i32],
        xdrop: i32,
        hspthresh: i32,
        num_hits: u32,
        anchors: &[u64],
        mut flags_bits: DisjointSlice<u32>,
        ref_pk: &[u64],
        qry_pk: &[u64],
        qry_bad: &[u32],
        um: i32,
        ua: i32,
        ub: i32,
        ubias: i32,
        count: u32,
        words: u32,
        mut cand_bits: DisjointSlice<u32>,
    ) {
        score_gate(
            thread::threadIdx_x(),
            ref_seq,
            query_seq,
            ref_len,
            query_len,
            sub_mat,
            xdrop,
            hspthresh,
            num_hits,
            anchors,
            false,
            core::ptr::null_mut(),
            flags_bits.as_mut_ptr(),
            SIEVE_SCATTER,
            ref_pk.as_ptr(),
            qry_pk.as_ptr(),
            qry_bad.as_ptr(),
            um,
            ua,
            ub,
            ubias,
            count,
            words,
            cand_bits.as_mut_ptr(),
        );
    }

    /// [`mark_score_survivors_sieve`] over a query without a mask plane (round 117):
    /// `qry_bad` is not read.
    #[cfg(feature = "gate-sieve")]
    #[kernel]
    #[launch_bounds(super::HSP_THREADS)]
    #[allow(clippy::too_many_arguments)]
    pub fn mark_score_survivors_sieve_open(
        ref_seq: &[u8],
        query_seq: &[u8],
        ref_len: u32,
        query_len: u32,
        sub_mat: &[i32],
        xdrop: i32,
        hspthresh: i32,
        num_hits: u32,
        anchors: &[u64],
        mut flags_bits: DisjointSlice<u32>,
        ref_pk: &[u64],
        qry_pk: &[u64],
        qry_bad: &[u32],
        um: i32,
        ua: i32,
        ub: i32,
        ubias: i32,
        count: u32,
        words: u32,
        mut cand_bits: DisjointSlice<u32>,
    ) {
        score_gate(
            thread::threadIdx_x(),
            ref_seq,
            query_seq,
            ref_len,
            query_len,
            sub_mat,
            xdrop,
            hspthresh,
            num_hits,
            anchors,
            false,
            core::ptr::null_mut(),
            flags_bits.as_mut_ptr(),
            SIEVE_OPEN,
            ref_pk.as_ptr(),
            qry_pk.as_ptr(),
            qry_bad.as_ptr(),
            um,
            ua,
            ub,
            ubias,
            count,
            words,
            cand_bits.as_mut_ptr(),
        );
    }

    /// [`mark_score_survivors_reordered`] behind the same bound.
    #[cfg(all(feature = "gate-sieve", feature = "ref-loc-buckets"))]
    #[kernel]
    #[launch_bounds(super::HSP_THREADS)]
    #[allow(clippy::too_many_arguments)]
    pub fn mark_score_survivors_reordered_sieve(
        ref_seq: &[u8],
        query_seq: &[u8],
        ref_len: u32,
        query_len: u32,
        sub_mat: &[i32],
        xdrop: i32,
        hspthresh: i32,
        num_hits: u32,
        anchors: &[u64],
        mut flags_bits: DisjointSlice<u32>,
        ref_pk: &[u64],
        qry_pk: &[u64],
        qry_bad: &[u32],
        um: i32,
        ua: i32,
        ub: i32,
        ubias: i32,
        count: u32,
        words: u32,
        mut cand_bits: DisjointSlice<u32>,
    ) {
        score_gate(
            thread::threadIdx_x(),
            ref_seq,
            query_seq,
            ref_len,
            query_len,
            sub_mat,
            xdrop,
            hspthresh,
            num_hits,
            anchors,
            true,
            core::ptr::null_mut(),
            flags_bits.as_mut_ptr(),
            SIEVE_SCATTER,
            ref_pk.as_ptr(),
            qry_pk.as_ptr(),
            qry_bad.as_ptr(),
            um,
            ua,
            ub,
            ubias,
            count,
            words,
            cand_bits.as_mut_ptr(),
        );
    }

    /// [`mark_score_survivors_reordered_sieve`] over a query without a mask plane (round 117):
    /// `qry_bad` is not read.
    #[cfg(all(feature = "gate-sieve", feature = "ref-loc-buckets"))]
    #[kernel]
    #[launch_bounds(super::HSP_THREADS)]
    #[allow(clippy::too_many_arguments)]
    pub fn mark_score_survivors_reordered_sieve_open(
        ref_seq: &[u8],
        query_seq: &[u8],
        ref_len: u32,
        query_len: u32,
        sub_mat: &[i32],
        xdrop: i32,
        hspthresh: i32,
        num_hits: u32,
        anchors: &[u64],
        mut flags_bits: DisjointSlice<u32>,
        ref_pk: &[u64],
        qry_pk: &[u64],
        qry_bad: &[u32],
        um: i32,
        ua: i32,
        ub: i32,
        ubias: i32,
        count: u32,
        words: u32,
        mut cand_bits: DisjointSlice<u32>,
    ) {
        score_gate(
            thread::threadIdx_x(),
            ref_seq,
            query_seq,
            ref_len,
            query_len,
            sub_mat,
            xdrop,
            hspthresh,
            num_hits,
            anchors,
            true,
            core::ptr::null_mut(),
            flags_bits.as_mut_ptr(),
            SIEVE_OPEN,
            ref_pk.as_ptr(),
            qry_pk.as_ptr(),
            qry_bad.as_ptr(),
            um,
            ua,
            ub,
            ubias,
            count,
            words,
            cand_bits.as_mut_ptr(),
        );
    }

    /// The gather arm of the sieve: the warp's six reference groups per hit,
    /// loaded coalesced instead of by 32 lanes at 32 unrelated addresses.
    ///
    /// Lane `l` loads group `l % 6` of hit `5 * round + l / 6`: six neighbouring
    /// lanes read the 48 contiguous bytes of one hit, five hits per round, seven
    /// rounds per chunk (lanes 30 and 31 idle). Each word goes to shared memory
    /// at row = group, column = hit, and every lane then reads its own column.
    /// The same six words reach the same arithmetic as in the scatter arm.
    ///
    /// # Safety
    /// As `sieve::bound_at` for every anchor of the chunk: lane `i` holds the
    /// reference position of hit `i` in `my_lo`.
    #[cfg(feature = "gate-sieve")]
    #[inline(always)]
    unsafe fn sieve_gather(
        w: usize,
        lane_id: u32,
        my_lo: u32,
        ref_pk: *const u64,
    ) -> (u64, u64, u64, u64, u64, u64) {
        static mut GROUPS: SharedArray<u64, { NUM_WARPS * GATHER_SLOTS }> = SharedArray::UNINIT;

        let row = (lane_id % 6) as usize;
        let column = lane_id / 6;
        let base = w * GATHER_SLOTS;
        let mut round = 0u32;
        while round < 7 {
            let hit = 5 * round + column;
            let loads = column < 5 && hit < WARP_SIZE;
            let ref_loc = warp::shuffle_sync(FULL_MASK, my_lo, if loads { hit } else { 0 });
            if loads {
                let (first, _, _, _) = sieve::locate(ref_loc, 0);
                // SAFETY: the six groups of a packed hit; one lane per slot.
                unsafe {
                    GROUPS[base + row * GATHER_ROW + hit as usize] =
                        *ref_pk.add(first as usize + row);
                }
            }
            round += 1;
        }
        warp::sync_mask(FULL_MASK);
        let at = base + lane_id as usize;
        // SAFETY: column `lane_id` of all six rows was stored above.
        unsafe {
            (
                GROUPS[at],
                GROUPS[at + GATHER_ROW],
                GROUPS[at + 2 * GATHER_ROW],
                GROUPS[at + 3 * GATHER_ROW],
                GROUPS[at + 4 * GATHER_ROW],
                GROUPS[at + 5 * GATHER_ROW],
            )
        }
    }

    /// [`mark_score_survivors_sieve`] with the reference groups gathered
    /// through shared memory ([`sieve_gather`]).
    #[cfg(feature = "gate-sieve")]
    #[kernel]
    #[launch_bounds(super::HSP_THREADS)]
    #[allow(clippy::too_many_arguments)]
    pub fn mark_score_survivors_sieve_gather(
        ref_seq: &[u8],
        query_seq: &[u8],
        ref_len: u32,
        query_len: u32,
        sub_mat: &[i32],
        xdrop: i32,
        hspthresh: i32,
        num_hits: u32,
        anchors: &[u64],
        mut flags_bits: DisjointSlice<u32>,
        ref_pk: &[u64],
        qry_pk: &[u64],
        qry_bad: &[u32],
        um: i32,
        ua: i32,
        ub: i32,
        ubias: i32,
        count: u32,
        words: u32,
        mut cand_bits: DisjointSlice<u32>,
    ) {
        score_gate(
            thread::threadIdx_x(),
            ref_seq,
            query_seq,
            ref_len,
            query_len,
            sub_mat,
            xdrop,
            hspthresh,
            num_hits,
            anchors,
            false,
            core::ptr::null_mut(),
            flags_bits.as_mut_ptr(),
            SIEVE_GATHER,
            ref_pk.as_ptr(),
            qry_pk.as_ptr(),
            qry_bad.as_ptr(),
            um,
            ua,
            ub,
            ubias,
            count,
            words,
            cand_bits.as_mut_ptr(),
        );
    }

    /// [`mark_score_survivors_reordered_sieve`] with the gathered groups.
    #[cfg(all(feature = "gate-sieve", feature = "ref-loc-buckets"))]
    #[kernel]
    #[launch_bounds(super::HSP_THREADS)]
    #[allow(clippy::too_many_arguments)]
    pub fn mark_score_survivors_reordered_sieve_gather(
        ref_seq: &[u8],
        query_seq: &[u8],
        ref_len: u32,
        query_len: u32,
        sub_mat: &[i32],
        xdrop: i32,
        hspthresh: i32,
        num_hits: u32,
        anchors: &[u64],
        mut flags_bits: DisjointSlice<u32>,
        ref_pk: &[u64],
        qry_pk: &[u64],
        qry_bad: &[u32],
        um: i32,
        ua: i32,
        ub: i32,
        ubias: i32,
        count: u32,
        words: u32,
        mut cand_bits: DisjointSlice<u32>,
    ) {
        score_gate(
            thread::threadIdx_x(),
            ref_seq,
            query_seq,
            ref_len,
            query_len,
            sub_mat,
            xdrop,
            hspthresh,
            num_hits,
            anchors,
            true,
            core::ptr::null_mut(),
            flags_bits.as_mut_ptr(),
            SIEVE_GATHER,
            ref_pk.as_ptr(),
            qry_pk.as_ptr(),
            qry_bad.as_ptr(),
            um,
            ua,
            ub,
            ubias,
            count,
            words,
            cand_bits.as_mut_ptr(),
        );
    }

    /// Packs an encoded sequence into the planes the sieve reads
    /// (`gpu/sieve.rs: pack` is the host mirror): one thread per 32-base group.
    /// Symbols outside `ACGT` take the code `fill` and a bad bit; everything
    /// outside the sequence is code 0 and bad. `bad` is written only when
    /// `want_bad` is set (the reference has no bad plane).
    #[cfg(feature = "gate-sieve")]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn pack_planes(
        seq: &[u8],
        len: u32,
        num_groups: u32,
        fill: u32,
        want_bad: u32,
        mut pk: DisjointSlice<u64>,
        mut bad: DisjointSlice<u32>,
    ) {
        let stride = thread::blockDim_x() * thread::gridDim_x();
        let mut g = thread::blockDim_x() * thread::blockIdx_x() + thread::threadIdx_x();
        while g < num_groups {
            let (mut hi, mut lo, mut bd) = (0u32, 0u32, 0u32);
            let mut b = 0u32;
            while b < 32 {
                // Groups in front of the sequence wrap to a position past `len`.
                let pos = g.wrapping_sub(sieve::PADG).wrapping_mul(32).wrapping_add(b);
                if g >= sieve::PADG && pos < len {
                    // SAFETY: `pos < len == seq.len()`.
                    let sym = unsafe { *seq.get_unchecked(pos as usize) } as u32;
                    let code = if sym < 4 {
                        sym
                    } else {
                        bd |= 1 << b;
                        fill
                    };
                    hi |= ((code >> 1) & 1) << b;
                    lo |= (code & 1) << b;
                } else {
                    bd |= 1 << b;
                }
                b += 1;
            }
            // SAFETY: `g < num_groups`, the length of both planes.
            unsafe {
                *pk.get_unchecked_mut(g as usize) = hi as u64 | (lo as u64) << 32;
                if want_bad != 0 {
                    *bad.get_unchecked_mut(g as usize) = bd;
                }
            }
            g += stride;
        }
    }

    /// Body of the score gate, shared by the plain and the reordered kernel.
    ///
    /// `hspthresh` is read only by lane 0's keep test. With `gate-int-keep` the
    /// host passes `keep_threshold(hspthresh)` in its place (`gpu/mod.rs`).
    #[cfg(feature = "dense-anchors")]
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::manual_range_contains)]
    fn score_gate(
        thread_id: u32,
        ref_seq: &[u8],
        query_seq: &[u8],
        ref_len: u32,
        query_len: u32,
        sub_mat: &[i32],
        xdrop: i32,
        hspthresh: i32,
        num_hits: u32,
        anchors: &[u64],
        reordered: bool,
        flags: *mut u8,
        // Round 83 round 2: one `u32` of keep bits per 32-slot chunk, written at
        // the chunk's sorted slot. Null and unused unless `reordered`.
        flags_bits: *mut u32,
        // `gate-sieve` (`gpu/sieve.rs`): with a mode other than `NO_SIEVE` each
        // lane first bounds its own hit and the slot loop walks only the hits
        // the bound could not reject. A literal, like `reordered`: with
        // `NO_SIEVE` everything below folds away and the planes, class constants
        // and candidate words are never read.
        sieve_mode: u32,
        ref_pk: *const u64,
        qry_pk: *const u64,
        qry_bad: *const u32,
        um: i32,
        ua: i32,
        ub: i32,
        ubias: i32,
        count: u32,
        // Round 111: how far an undecided side of the bound is carried, as
        // `right words | left words << 8` (`sieve::Words`).
        words: u32,
        cand_bits: *mut u32,
    ) {
        // The anchor no longer travels through shared memory (round 58): the warp
        // fetches its whole chunk coalesced and broadcasts by shuffle.
        static mut SUB: SharedArray<i32, { NUC * NUC }> = SharedArray::UNINIT;

        let lane_id = thread_id % WARP_SIZE;
        let w = ((thread_id - lane_id) / WARP_SIZE) as usize;
        if (thread_id as usize) < NUC * NUC {
            unsafe { SUB[thread_id as usize] = sub_mat[thread_id as usize] };
        }
        thread::sync_threads();

        // A warp owns CHUNK consecutive anchors so the query side can be reused across
        // the ones that share `query_start` (round 53). 32 was measured against 4, 8,
        // 16 and 64: A prefers 32 because its runs are ~20 long and a longer chunk only
        // adds serialisation, while F (runs ~94) still gains at 64. 32 is the value
        // whose worst per-workload loss is ~0.5 points.
        const CHUNK: u32 = 32;

        // Grid-stride for the production kernel. The reordered kernel instead gives
        // each block one CONTIGUOUS range: with grid-stride a resident block sweeps
        // windows 2 M hits apart, so the wave of resident blocks touches ~8 different
        // reference buckets at once and the L2 never holds one (cycle 3). `per_block`
        // is a multiple of CHUNK, so a warp's chunk never straddles two blocks.
        let (mut base, limit, stride) = if reordered {
            let per_block = num_hits.div_ceil(thread::gridDim_x()).div_ceil(CHUNK) * CHUNK;
            let start = thread::blockIdx_x() * per_block;
            let end = if start < num_hits && num_hits - start > per_block {
                start + per_block
            } else {
                num_hits
            };
            (start + w as u32 * CHUNK, end, NUM_WARPS as u32 * CHUNK)
        } else {
            (
                (thread::blockIdx_x() * NUM_WARPS as u32 + w as u32) * CHUNK,
                num_hits,
                NUM_WARPS as u32 * thread::gridDim_x() * CHUNK,
            )
        };
        while base < limit {
            // Cached query window for this warp's current run. `q_have` is uniform
            // across the warp: every lane walks the same anchors in the same order.
            let mut q_valid = false;
            let mut q_cached_loc = u32::MAX;
            let mut q_word = 0u32;
            let mut q_fast = false;

            // One coalesced fetch of the whole chunk, then broadcast by shuffle.
            //
            // `CHUNK` is a warp, so lane `i` holds anchor `base + i` and each slot
            // reads it with two 32-bit shuffles. What this removes is a *global load
            // at the head of every hit's dependency chain* — the old path had lane 0
            // load the anchor, write it to shared, sync the warp and have every lane
            // read it back, with nothing to overlap the load latency. Round 58.
            let my_idx = base + lane_id;
            // The sieve kernels load without the slice's bounds test, which is
            // 16 instructions and a trap per chunk for an index the two
            // comparisons already bound (`base < num_hits` is the loop's own
            // condition). The plain gate kernels keep the checked load and
            // their machine code.
            let my_anchor = if sieve_mode != NO_SIEVE {
                let at = if my_idx < num_hits { my_idx } else { base };
                // SAFETY: `at < num_hits <= anchors.len()`.
                unsafe { *anchors.get_unchecked(at as usize) }
            } else if my_idx < num_hits {
                anchors[my_idx as usize]
            } else {
                anchors[base as usize]
            };
            let my_lo = my_anchor as u32;
            let my_hi = (my_anchor >> 32) as u32;

            // `gate-sieve`: the slots this warp still has to score, as a mask.
            // Each lane bounds the hit it just loaded; a hit whose bound is
            // certified below the threshold cannot be kept and is never scored.
            let mut rem = 0u32;
            // A literal, like the mode: without the mask plane its loads and the
            // terminal tests fold away.
            #[cfg(feature = "gate-sieve")]
            let masked = sieve_mode != SIEVE_OPEN;
            #[cfg(not(feature = "gate-sieve"))]
            let masked = true;
            if sieve_mode != NO_SIEVE {
                // SAFETY: the host packed both planes for these sequences, and a
                // lane past `num_hits` holds the chunk's first anchor.
                #[cfg(not(feature = "gate-sieve"))]
                let (mut right, mut left) = unsafe {
                    sieve::bound_at(
                        ref_pk, qry_pk, qry_bad, my_lo, my_hi, masked, um, ua, ub, ubias, xdrop,
                    )
                };
                #[cfg(feature = "gate-sieve")]
                let (mut right, mut left) = unsafe {
                    if sieve_mode == SIEVE_GATHER {
                        let (r0, r1, r2, r3, r4, r5) = sieve_gather(w, lane_id, my_lo, ref_pk);
                        let (_, qg, o, qs) = sieve::locate(my_lo, my_hi);
                        sieve::bound_words(
                            r0,
                            r1,
                            r2,
                            r3,
                            r4,
                            r5,
                            qry_pk.add(qg as usize),
                            qry_bad.add(qg as usize),
                            o,
                            qs,
                            true,
                            um,
                            ua,
                            ub,
                            ubias,
                            xdrop,
                        )
                    } else {
                        sieve::bound_at(
                            ref_pk, qry_pk, qry_bad, my_lo, my_hi, masked, um, ua, ub, ubias, xdrop,
                        )
                    }
                };
                // Round 111: a side that has not certified inside the static
                // window is carried on, one word at a time. A lane whose side is
                // certified, or whose two bounds already reach the threshold, has
                // nothing to carry; neither has a lane past the chunk's hits. No
                // lane needs another lane's data for this, so each walks as far
                // as its own hit asks and the warp waits for the longest walk.
                let live = my_idx < num_hits;
                if live && sieve::carry(right, left, hspthresh) {
                    let (rg, qg, o, qs) = sieve::locate(my_lo, my_hi);
                    let mut word = sieve::RIGHT_WORDS;
                    while word < (words & 0xff) && sieve::carry(right, left, hspthresh) {
                        // SAFETY: the planes carry `sieve::PADG` groups of padding
                        // for words up to `sieve::RIGHT_WORDS_MAX`, which the host
                        // never exceeds.
                        right = unsafe {
                            sieve::right_word(
                                ref_pk, qry_pk, qry_bad, rg, qg, o, qs, word, right, masked, um,
                                ua, ub, ubias, xdrop,
                            )
                        };
                        word += 1;
                    }
                }
                if live && sieve::carry(left, right, hspthresh) {
                    let (rg, qg, o, qs) = sieve::locate(my_lo, my_hi);
                    let mut word = sieve::LEFT_WORDS;
                    while word < (words >> 8) && sieve::carry(left, right, hspthresh) {
                        // SAFETY: as above, up to `sieve::LEFT_WORDS_MAX`.
                        left = unsafe {
                            sieve::left_word(
                                ref_pk, qry_pk, qry_bad, rg, qg, o, qs, word, left, masked, um, ua,
                                ub, ubias, xdrop,
                            )
                        };
                        word += 1;
                    }
                }
                rem = warp::ballot_sync(FULL_MASK, live && sieve::passes(right, left, hspthresh));
                if count != 0 && lane_id == 0 {
                    // SAFETY: `base < limit <= num_hits`, one word per chunk.
                    unsafe { *cand_bits.add((base / CHUNK) as usize) = rem };
                }
            }
            // Every sieve kernel writes its chunk's keep bits as one word, as the
            // reordered kernel does: a hit the bound rejected is a 0 bit that no
            // lane has to store.
            let word_flags = reordered || sieve_mode != NO_SIEVE;

            // Keep bits of this chunk, bit `s` for slot `s`; slots past
            // `num_hits` stay 0. Lane 0 owns it and stores it once below.
            let mut chunk_bits = 0u32;
            let mut slot = 0u32;
            loop {
                if sieve_mode != NO_SIEVE {
                    if rem == 0 {
                        break;
                    }
                    slot = rem.trailing_zeros();
                    rem &= rem - 1;
                } else if slot >= CHUNK {
                    break;
                }
                let hid = base + slot;
                let ref_loc = warp::shuffle_sync(FULL_MASK, my_lo, slot);
                let query_loc = warp::shuffle_sync(FULL_MASK, my_hi, slot);
                let mut total = 0i32;

                // The anchor was just read; compare it with the cached window rather than
                // peeking at the next one, which the next iteration loads anyway.
                let q_have = q_valid && query_loc == q_cached_loc;
                q_cached_loc = query_loc;

                #[cfg(feature = "simd-prelude")]
                {
                    let is_right = lane_id < 8;
                    let is_left = lane_id >= 8 && lane_id < 24;
                    let gl = if is_left { lane_id - 8 } else { lane_id };

                    let mut s0 = 0i32;
                    let mut s1 = 0i32;
                    let mut s2 = 0i32;
                    let mut s3 = 0i32;
                    if is_right {
                        let off = 4 * lane_id;
                        let last_r = ref_loc.wrapping_add(off + 3);
                        let last_q = query_loc.wrapping_add(off + 3);
                        // The query side depends only on `query_loc`, so it is loaded once
                        // per run and reused; the reference side is per hit. Splitting the
                        // original combined guard leaves the fast-path condition — and the
                        // per-base slow path below — bit-identical.
                        if !q_have {
                            q_fast = last_q >= query_loc && last_q < query_len;
                            q_word = if q_fast {
                                unsafe { ld_u32(query_seq, query_loc + off, query_len) }
                            } else {
                                0
                            };
                        }
                        if last_r >= ref_loc && last_r < ref_len && q_fast {
                            let rw = unsafe { ld_u32(ref_seq, ref_loc + off, ref_len) };
                            let qw = q_word;
                            let (b0, b1, b2, b3) = (
                                (rw & 0xff) as usize,
                                ((rw >> 8) & 0xff) as usize,
                                ((rw >> 16) & 0xff) as usize,
                                ((rw >> 24) & 0xff) as usize,
                            );
                            let c0 = (qw & 0xff) as usize;
                            let c1 = ((qw >> 8) & 0xff) as usize;
                            let c2 = ((qw >> 16) & 0xff) as usize;
                            let c3 = ((qw >> 24) & 0xff) as usize;
                            unsafe {
                                s0 = SUB[b0 * NUC + c0];
                                s1 = SUB[b1 * NUC + c1];
                                s2 = SUB[b2 * NUC + c2];
                                s3 = SUB[b3 * NUC + c3];
                            }
                        } else {
                            for k in 0..4u32 {
                                let rp = ref_loc.wrapping_add(off + k);
                                let qp = query_loc.wrapping_add(off + k);
                                if rp < ref_len && qp < query_len {
                                    let r = unsafe { *ref_seq.get_unchecked(rp as usize) } as usize;
                                    let q =
                                        unsafe { *query_seq.get_unchecked(qp as usize) } as usize;
                                    let s = unsafe { SUB[r * NUC + q] };
                                    if k == 0 {
                                        s0 = s;
                                    } else if k == 1 {
                                        s1 = s;
                                    } else if k == 2 {
                                        s2 = s;
                                    } else {
                                        s3 = s;
                                    }
                                }
                            }
                        }
                    } else if is_left {
                        let last_off = 4 * gl + 4;
                        if !q_have {
                            q_fast = query_loc >= last_off;
                            q_word = if q_fast {
                                unsafe { ld_u32(query_seq, query_loc - last_off, query_len) }
                            } else {
                                0
                            };
                        }
                        if ref_loc >= last_off && q_fast {
                            let rw = unsafe { ld_u32(ref_seq, ref_loc - last_off, ref_len) };
                            let qw = q_word;
                            // bytes [0,1,2,3] sit at offsets last_off .. last_off-3;
                            // extension order is last_off-3 .. last_off → reverse.
                            let (b0, b1, b2, b3) = (
                                ((rw >> 24) & 0xff) as usize,
                                ((rw >> 16) & 0xff) as usize,
                                ((rw >> 8) & 0xff) as usize,
                                (rw & 0xff) as usize,
                            );
                            let c0 = ((qw >> 24) & 0xff) as usize;
                            let c1 = ((qw >> 16) & 0xff) as usize;
                            let c2 = ((qw >> 8) & 0xff) as usize;
                            let c3 = (qw & 0xff) as usize;
                            unsafe {
                                s0 = SUB[b0 * NUC + c0];
                                s1 = SUB[b1 * NUC + c1];
                                s2 = SUB[b2 * NUC + c2];
                                s3 = SUB[b3 * NUC + c3];
                            }
                        } else {
                            for k in 0..4u32 {
                                let off = 4 * gl + 1 + k;
                                if ref_loc >= off && query_loc >= off {
                                    let r =
                                        unsafe { *ref_seq.get_unchecked((ref_loc - off) as usize) }
                                            as usize;
                                    let q = unsafe {
                                        *query_seq.get_unchecked((query_loc - off) as usize)
                                    } as usize;
                                    let s = unsafe { SUB[r * NUC + q] };
                                    if k == 0 {
                                        s0 = s;
                                    } else if k == 1 {
                                        s1 = s;
                                    } else if k == 2 {
                                        s2 = s;
                                    } else {
                                        s3 = s;
                                    }
                                }
                            }
                        }
                    }

                    q_valid = true;

                    let seg = s0.wrapping_add(s1).wrapping_add(s2).wrapping_add(s3);
                    let (seg_prefix, scanned) = {
                        // Lane-local pair: segment total, and the best prefix inside this
                        // lane floored at 0 — the inherited maximum at the prelude start.
                        let p0 = s0;
                        let p1 = p0.wrapping_add(s1);
                        let p2 = p1.wrapping_add(s2);
                        let p3 = p2.wrapping_add(s3);
                        let local_max = p0.max(p1).max(p2).max(p3).max(0);
                        dual_prefix_sum_max(lane_id, is_right, is_left, gl, seg, local_max)
                    };
                    let before = seg_prefix.wrapping_sub(seg);
                    let c0 = before.wrapping_add(s0);
                    let c1 = c0.wrapping_add(s1);
                    let c2 = c1.wrapping_add(s2);
                    let c3 = c2.wrapping_add(s3);
                    let src1 = if gl >= 1 { lane_id - 1 } else { lane_id };
                    let before_max = {
                        let up1 = warp::shuffle_sync(FULL_MASK, scanned as u32, src1) as i32;
                        if gl == 0 || !(is_right || is_left) {
                            0
                        } else {
                            up1
                        }
                    };
                    let after0 = before_max.max(c0);
                    let after1 = after0.max(c1);
                    let after2 = after1.max(c2);
                    #[cfg(not(feature = "gate-payload-ballot"))]
                    let (right_done, left_done, right_max, left_max) = {
                        let d0 = warp::ballot_sync(FULL_MASK, after0.wrapping_sub(c0) > xdrop);
                        let d1 = warp::ballot_sync(FULL_MASK, after1.wrapping_sub(c1) > xdrop);
                        let d2 = warp::ballot_sync(FULL_MASK, after2.wrapping_sub(c2) > xdrop);
                        let d3 = warp::ballot_sync(FULL_MASK, scanned.wrapping_sub(c3) > xdrop);

                        let first_of = |bits: u32, shift: u32, n_lanes: u32| -> u32 {
                            let tz = (bits >> shift).trailing_zeros();
                            if tz >= n_lanes { 255 } else { tz }
                        };
                        let right_first = (4 * first_of(d0, 0, 8))
                            .min(4 * first_of(d1, 0, 8) + 1)
                            .min(4 * first_of(d2, 0, 8) + 2)
                            .min(4 * first_of(d3, 0, 8) + 3);
                        let left_first = (4 * first_of(d0, 8, 16))
                            .min(4 * first_of(d1, 8, 16) + 1)
                            .min(4 * first_of(d2, 8, 16) + 2)
                            .min(4 * first_of(d3, 8, 16) + 3);
                        let right_edge = ref_loc.wrapping_add(31) >= ref_len
                            || query_loc.wrapping_add(31) >= query_len;
                        let left_edge = ref_loc < 64 || query_loc < 64;
                        let right_done = right_first < 32 || right_edge;
                        let left_done = left_first < 64 || left_edge;

                        let recover = |first: u32, n_bases: u32, origin: u32| -> i32 {
                            let f = if first >= n_bases { n_bases } else { first };
                            if f == 0 {
                                0
                            } else {
                                let sub = f & 3;
                                let src_gl = if sub == 0 { f / 4 - 1 } else { f / 4 };
                                let src = origin + src_gl;
                                let value = if sub == 0 {
                                    scanned
                                } else if sub == 1 {
                                    after0
                                } else if sub == 2 {
                                    after1
                                } else {
                                    after2
                                };
                                warp::shuffle_sync(FULL_MASK, value as u32, src) as i32
                            }
                        };
                        let right_max = recover(right_first, 32, 0);
                        let left_max = recover(left_first, 64, 8);
                        (right_done, left_done, right_max, left_max)
                    };
                    // `gate-payload-ballot`: the same four values from one ballot. A
                    // group's first drop is its first lane with any drop and, inside that
                    // lane, the first base; the maximum before it is local to that lane
                    // (`before_max` before base 0, else `after0..after2`), and a group
                    // without a drop ends on its last lane's `scanned`. Each lane offers
                    // that value, one ballot finds the lanes and the two shuffles
                    // `recover` already paid for read it — no first-of decodes, min
                    // cascades or `recover` branches. The host model
                    // `payload_ballot_matches_first_drop_recovery` checks it against the
                    // default path.
                    #[cfg(feature = "gate-payload-ballot")]
                    let (right_done, left_done, right_max, left_max) = {
                        let q0 = after0.wrapping_sub(c0) > xdrop;
                        let q1 = after1.wrapping_sub(c1) > xdrop;
                        let q2 = after2.wrapping_sub(c2) > xdrop;
                        let q3 = scanned.wrapping_sub(c3) > xdrop;
                        let payload = if q0 {
                            before_max
                        } else if q1 {
                            after0
                        } else if q2 {
                            after1
                        } else if q3 {
                            after2
                        } else {
                            scanned
                        };
                        let dropped = warp::ballot_sync(
                            FULL_MASK,
                            (is_right || is_left) && (q0 || q1 || q2 || q3),
                        );
                        // Lanes 7 and 23 stand in for "no drop": their payload is then
                        // `scanned`, the maximum over the whole group.
                        let right_src = (dropped | 0x80).trailing_zeros();
                        let left_src = 8 + ((dropped >> 8) | 0x8000).trailing_zeros();
                        let right_max = warp::shuffle_sync(FULL_MASK, payload as u32, right_src);
                        let left_max = warp::shuffle_sync(FULL_MASK, payload as u32, left_src);
                        let right_edge = ref_loc.wrapping_add(31) >= ref_len
                            || query_loc.wrapping_add(31) >= query_len;
                        let left_edge = ref_loc < 64 || query_loc < 64;
                        (
                            dropped & 0xff != 0 || right_edge,
                            dropped & 0xff_ff00 != 0 || left_edge,
                            right_max as i32,
                            left_max as i32,
                        )
                    };
                    // The two end broadcasts feed the continuation loops only, and those
                    // run for ~8% of hits. `right_done`/`left_done` are warp-uniform, so
                    // the shuffles sink into the branches: two dependent shuffles removed
                    // from the 92% that resolve inside the prelude. Round 57 — the gate
                    // stalls on dependencies, so a shuffle skipped is latency skipped.
                    if right_done {
                        total += right_max;
                    } else {
                        let right_end = warp::shuffle_sync(FULL_MASK, c3 as u32, 7) as i32;
                        let mut tile = 32i32;
                        let mut prev_score = right_end;
                        let mut prev_max = right_max;
                        loop {
                            let pos_offset = lane_id as i32 + tile;
                            let ref_pos = ref_loc.wrapping_add(pos_offset as u32);
                            let query_pos = query_loc.wrapping_add(pos_offset as u32);
                            let mut score = 0i32;
                            if ref_pos < ref_len && query_pos < query_len {
                                let r =
                                    unsafe { *ref_seq.get_unchecked(ref_pos as usize) } as usize;
                                let q = unsafe { *query_seq.get_unchecked(query_pos as usize) }
                                    as usize;
                                score = unsafe { SUB[r * NUC + q] };
                            }
                            score = prefix_sum(lane_id, score) + prev_score;
                            let candidate = if score > prev_max { score } else { prev_max };
                            let scanned = prefix_max_score(candidate);
                            let dropped = warp::ballot_sync(FULL_MASK, (scanned - score) > xdrop);
                            let first = dropped.trailing_zeros();
                            let max = if first == 0 {
                                prev_max
                            } else {
                                warp::shuffle_sync(FULL_MASK, scanned as u32, first - 1) as i32
                            };
                            let last_offset = tile + WARP_SIZE as i32 - 1;
                            let edge = ref_loc.wrapping_add(last_offset as u32) >= ref_len
                                || query_loc.wrapping_add(last_offset as u32) >= query_len;
                            if dropped != 0 || edge {
                                total += max;
                                break;
                            }
                            prev_score =
                                warp::shuffle_sync(FULL_MASK, score as u32, WARP_SIZE - 1) as i32;
                            prev_max = max;
                            tile += WARP_SIZE as i32;
                        }
                    }

                    if left_done {
                        total += left_max;
                    } else {
                        let left_end = warp::shuffle_sync(FULL_MASK, c3 as u32, 23) as i32;
                        let mut tile = 64i32;
                        let mut prev_score = left_end;
                        let mut prev_max = left_max;
                        loop {
                            let pos_offset = lane_id as i32 + 1 + tile;
                            let mut score = 0i32;
                            if ref_loc >= pos_offset as u32 && query_loc >= pos_offset as u32 {
                                let r = unsafe {
                                    *ref_seq.get_unchecked((ref_loc - pos_offset as u32) as usize)
                                } as usize;
                                let q = unsafe {
                                    *query_seq
                                        .get_unchecked((query_loc - pos_offset as u32) as usize)
                                } as usize;
                                score = unsafe { SUB[r * NUC + q] };
                            }
                            score = prefix_sum(lane_id, score) + prev_score;
                            let candidate = if score > prev_max { score } else { prev_max };
                            let scanned = prefix_max_score(candidate);
                            let dropped = warp::ballot_sync(FULL_MASK, (scanned - score) > xdrop);
                            let first = dropped.trailing_zeros();
                            let max = if first == 0 {
                                prev_max
                            } else {
                                warp::shuffle_sync(FULL_MASK, scanned as u32, first - 1) as i32
                            };
                            let last_offset = tile + WARP_SIZE as i32;
                            let edge =
                                ref_loc < last_offset as u32 || query_loc < last_offset as u32;
                            if dropped != 0 || edge {
                                total += max;
                                break;
                            }
                            prev_score =
                                warp::shuffle_sync(FULL_MASK, score as u32, WARP_SIZE - 1) as i32;
                            prev_max = max;
                            tile += WARP_SIZE as i32;
                        }
                    }
                }

                #[cfg(not(feature = "simd-prelude"))]
                {
                    // Right score: kept byte-for-byte equivalent to the production
                    // gate so its final f32-rounded threshold remains the oracle.
                    let mut tile = 0i32;
                    let mut prev_score = 0i32;
                    let mut prev_max = 0i32;
                    loop {
                        let pos_offset = lane_id as i32 + tile;
                        let ref_pos = ref_loc.wrapping_add(pos_offset as u32);
                        let query_pos = query_loc.wrapping_add(pos_offset as u32);
                        let mut score = 0i32;
                        if ref_pos < ref_len && query_pos < query_len {
                            let r = unsafe { *ref_seq.get_unchecked(ref_pos as usize) } as usize;
                            let q =
                                unsafe { *query_seq.get_unchecked(query_pos as usize) } as usize;
                            score = unsafe { SUB[r * NUC + q] };
                        }

                        score = prefix_sum(lane_id, score) + prev_score;
                        let candidate = if score > prev_max { score } else { prev_max };
                        let scanned = prefix_max_score(candidate);
                        let dropped = warp::ballot_sync(FULL_MASK, (scanned - score) > xdrop);
                        let first = dropped.trailing_zeros();
                        let max = if first == 0 {
                            prev_max
                        } else {
                            warp::shuffle_sync(FULL_MASK, scanned as u32, first - 1) as i32
                        };
                        let last_offset = tile + WARP_SIZE as i32 - 1;
                        let edge = ref_loc.wrapping_add(last_offset as u32) >= ref_len
                            || query_loc.wrapping_add(last_offset as u32) >= query_len;
                        if dropped != 0 || edge {
                            total += max;
                            break;
                        }

                        prev_score =
                            warp::shuffle_sync(FULL_MASK, score as u32, WARP_SIZE - 1) as i32;
                        prev_max = max;
                        tile += WARP_SIZE as i32;
                    }

                    // Left score. The last lane evaluates offset tile + 32.
                    tile = 0;
                    prev_score = 0;
                    prev_max = 0;
                    #[cfg(feature = "left-pair-tile")]
                    let mut left_done = false;
                    #[cfg(not(feature = "left-pair-tile"))]
                    let left_done = false;

                    #[cfg(feature = "left-pair-tile")]
                    {
                        let first_offset = 2 * lane_id + 1;
                        let second_offset = first_offset + 1;
                        let mut first_score = 0i32;
                        let mut second_score = 0i32;
                        if ref_loc >= first_offset && query_loc >= first_offset {
                            let r = unsafe {
                                *ref_seq.get_unchecked((ref_loc - first_offset) as usize)
                            } as usize;
                            let q = unsafe {
                                *query_seq.get_unchecked((query_loc - first_offset) as usize)
                            } as usize;
                            first_score = unsafe { SUB[r * NUC + q] };
                        }
                        if ref_loc >= second_offset && query_loc >= second_offset {
                            let r = unsafe {
                                *ref_seq.get_unchecked((ref_loc - second_offset) as usize)
                            } as usize;
                            let q = unsafe {
                                *query_seq.get_unchecked((query_loc - second_offset) as usize)
                            } as usize;
                            second_score = unsafe { SUB[r * NUC + q] };
                        }

                        let pair_score = first_score.wrapping_add(second_score);
                        let pair_prefix = prefix_sum(lane_id, pair_score);
                        let before_pair = pair_prefix.wrapping_sub(pair_score);
                        let first_cumulative = prev_score
                            .wrapping_add(before_pair)
                            .wrapping_add(first_score);
                        let second_cumulative = prev_score.wrapping_add(pair_prefix);
                        let candidate = first_cumulative.max(second_cumulative).max(prev_max);
                        let scanned = prefix_max_score(candidate);

                        let up = warp::shuffle_up_sync(FULL_MASK, scanned as u32, 1) as i32;
                        let before_max = if lane_id == 0 { prev_max } else { up };
                        let after_first = before_max.max(first_cumulative);
                        let first_drops = warp::ballot_sync(
                            FULL_MASK,
                            after_first.wrapping_sub(first_cumulative) > xdrop,
                        );
                        let second_drops = warp::ballot_sync(
                            FULL_MASK,
                            scanned.wrapping_sub(second_cumulative) > xdrop,
                        );
                        let first = (2 * first_drops.trailing_zeros())
                            .min(2 * second_drops.trailing_zeros() + 1);
                        let max = if first == 0 {
                            prev_max
                        } else {
                            let odd = first & 1 != 0;
                            let source = if odd { first / 2 } else { first / 2 - 1 };
                            let value = if odd { after_first } else { scanned };
                            warp::shuffle_sync(FULL_MASK, value as u32, source) as i32
                        };
                        let edge = ref_loc < 2 * WARP_SIZE || query_loc < 2 * WARP_SIZE;
                        if first < 2 * WARP_SIZE || edge {
                            total += max;
                            left_done = true;
                        } else {
                            prev_score = warp::shuffle_sync(
                                FULL_MASK,
                                second_cumulative as u32,
                                WARP_SIZE - 1,
                            ) as i32;
                            prev_max = max;
                            tile = 2 * WARP_SIZE as i32;
                        }
                    }

                    if !left_done {
                        loop {
                            let pos_offset = lane_id as i32 + 1 + tile;
                            let mut score = 0i32;
                            if ref_loc >= pos_offset as u32 && query_loc >= pos_offset as u32 {
                                let r = unsafe {
                                    *ref_seq.get_unchecked((ref_loc - pos_offset as u32) as usize)
                                } as usize;
                                let q = unsafe {
                                    *query_seq
                                        .get_unchecked((query_loc - pos_offset as u32) as usize)
                                } as usize;
                                score = unsafe { SUB[r * NUC + q] };
                            }

                            score = prefix_sum(lane_id, score) + prev_score;
                            let candidate = if score > prev_max { score } else { prev_max };
                            let scanned = prefix_max_score(candidate);
                            let dropped = warp::ballot_sync(FULL_MASK, (scanned - score) > xdrop);
                            let first = dropped.trailing_zeros();
                            let max = if first == 0 {
                                prev_max
                            } else {
                                warp::shuffle_sync(FULL_MASK, scanned as u32, first - 1) as i32
                            };
                            let last_offset = tile + WARP_SIZE as i32;
                            let edge =
                                ref_loc < last_offset as u32 || query_loc < last_offset as u32;
                            if dropped != 0 || edge {
                                total += max;
                                break;
                            }

                            prev_score =
                                warp::shuffle_sync(FULL_MASK, score as u32, WARP_SIZE - 1) as i32;
                            prev_max = max;
                            tile += WARP_SIZE as i32;
                        }
                    }
                }

                if hid < num_hits && lane_id == 0 {
                    // Production rounds through f32 before this comparison.
                    #[cfg(not(feature = "gate-int-keep"))]
                    let keep = ((total as f32 as f64) as i32 >= hspthresh) as u8;
                    // `gate-int-keep`: the host passed `keep_threshold(hspthresh)` in
                    // this slot — the least total the rounded comparison keeps — so one
                    // integer compare decides the same, with no FP64 on the gate.
                    #[cfg(feature = "gate-int-keep")]
                    let keep = (total >= hspthresh) as u8;
                    if word_flags {
                        chunk_bits |= (keep as u32) << slot;
                    } else {
                        // SAFETY: `hid < num_hits`, one thread per flag byte.
                        unsafe { *flags.add(hid as usize) = keep };
                    }
                }
                slot += 1;
            }
            // One store per chunk. `base` is a multiple of CHUNK == 32 for both
            // traversals, so each word has exactly one owning warp and every
            // word of `ceil(num_hits/32)` is overwritten — no stale bits.
            if word_flags && lane_id == 0 {
                // SAFETY: `base < limit <= num_hits`, so the word index is
                // inside the `ceil(num_hits/32)` the host allocated.
                unsafe { *flags_bits.add((base / CHUNK) as usize) = chunk_bits };
            }
            base += stride;
        }
    }

    /// `find_hsps`: ungapped X-drop extension, one warp per seed hit.
    ///
    /// `log4` is `logf(4.0f)` widened to double, matching KegAlign's
    /// `-entropy/log(4.0f)` where the divisor is computed in single precision.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_range_loop)]
    pub fn find_hsps(
        ref_seq: &[u8],
        query_seq: &[u8],
        ref_len: u32,
        query_len: u32,
        sub_mat: &[i32],
        noentropy: u32,
        xdrop: i32,
        hspthresh: i32,
        num_hits: u32,
        log4: f64,
        #[allow(unused_variables)] anchor_ptr: *const u64,
        #[allow(unused_variables)] survivor_ptr: *const u32,
        mut hsp: DisjointSlice<SegmentPair>,
        mut done: DisjointSlice<u32>,
        // Written only under the `counters` feature; the clean build compiles
        // no counter code at all and leaves this argument untouched.
        #[allow(unused_mut, unused_variables)] mut stats: DisjointSlice<u64>,
    ) {
        static mut REF_LOC: SharedArray<u32, NUM_WARPS> = SharedArray::UNINIT;
        static mut QUERY_LOC: SharedArray<u32, NUM_WARPS> = SharedArray::UNINIT;
        static mut TOTAL_SCORE: SharedArray<i32, NUM_WARPS> = SharedArray::UNINIT;
        static mut PREV_SCORE: SharedArray<i32, NUM_WARPS> = SharedArray::UNINIT;
        static mut PREV_MAX_SCORE: SharedArray<i32, NUM_WARPS> = SharedArray::UNINIT;
        static mut PREV_MAX_POS: SharedArray<i32, NUM_WARPS> = SharedArray::UNINIT;
        static mut EDGE_FOUND: SharedArray<u32, NUM_WARPS> = SharedArray::UNINIT;
        static mut XDROP_FOUND: SharedArray<u32, NUM_WARPS> = SharedArray::UNINIT;
        static mut LEFT_EXTENT: SharedArray<u32, NUM_WARPS> = SharedArray::UNINIT;
        static mut EXTENT: SharedArray<i32, NUM_WARPS> = SharedArray::UNINIT;
        static mut TILE: SharedArray<i32, NUM_WARPS> = SharedArray::UNINIT;
        static mut ENTROPY: SharedArray<f64, NUM_WARPS> = SharedArray::UNINIT;
        static mut SUB: SharedArray<i32, { NUC * NUC }> = SharedArray::UNINIT;
        #[cfg(feature = "counters")]
        static mut TERM: SharedArray<u32, NUM_WARPS> = SharedArray::UNINIT;

        let thread_id = thread::threadIdx_x();
        let block_id = thread::blockIdx_x();
        let num_blocks = thread::gridDim_x();
        let lane_id = thread_id % WARP_SIZE;
        let w = ((thread_id - lane_id) / WARP_SIZE) as usize;
        let last_lane = lane_id == WARP_SIZE - 1;

        if (thread_id as usize) < NUC * NUC {
            // SAFETY: block has BLOCK_SIZE >= NUC*NUC threads, one slot each.
            unsafe {
                SUB[thread_id as usize] = sub_mat[thread_id as usize];
            }
        }
        thread::sync_threads();

        let mut hid0 = block_id * NUM_WARPS as u32;
        while hid0 < num_hits {
            #[cfg(feature = "counters")]
            let (mut right_tiles, mut left_tiles, mut term) = (0u64, 0u64, 0u64);
            // Where the first X-drop lands, in the first tile of each
            // direction — the class that owns ~91% of right terminations.
            #[cfg(feature = "counters")]
            let (mut first_drop_r, mut first_drop_l) = (63u64, 63u64);
            #[cfg(feature = "counters")]
            if lane_id == 0 {
                unsafe { TERM[w] = 0 };
            }
            let hid = hid0 + w as u32;
            if lane_id == 0 {
                let src = if hid < num_hits { hid } else { hid0 };
                #[cfg(not(feature = "dense-anchors"))]
                // SAFETY: `src < num_hits <= hsp.len()`.
                let seed_hit = unsafe { *hsp.get_unchecked_mut(src as usize) };
                #[cfg(not(feature = "dense-anchors"))]
                let (ref_start, query_start) = (seed_hit.ref_start, seed_hit.query_start);
                #[cfg(feature = "dense-anchors")]
                // SAFETY: stable compaction emits one valid original ID per
                // dense work item, and both persistent buffers retain their
                // allocations until this stream is drained.
                let (ref_start, query_start) = unsafe {
                    let original = *survivor_ptr.add(src as usize);
                    let anchor = *anchor_ptr.add(original as usize);
                    (anchor as u32, (anchor >> 32) as u32)
                };
                unsafe {
                    REF_LOC[w] = ref_start;
                    QUERY_LOC[w] = query_start;
                    TOTAL_SCORE[w] = 0;
                }
            }
            warp::sync_mask(FULL_MASK);
            // Invariant for the whole hit, yet re-loaded from shared by every
            // lane on every tile. The instruction census: shared-state loads are ~13% of
            // per-tile instructions and the kernel runs at ~73% of issue peak,
            // so these are the cheapest instructions to remove. Unlike the loop-carrier
            // registerization this
            // needs no commit restructuring — the value never changes.
            let hit_ref_loc = unsafe { REF_LOC[w] };
            let hit_query_loc = unsafe { QUERY_LOC[w] };

            // Score both extensions with the production warp mapping,
            // but keep only the state needed for X-drop and the maximum score.
            // The overwhelmingly common failing hit avoids position recovery,
            // extent tracking, entropy, and output construction below. A rare
            // survivor is recomputed by the unchanged materializer, preserving
            // every coordinate, tie, and output-order decision by construction.
            #[cfg(all(feature = "warp-score-gate", not(feature = "dense-anchors")))]
            {
                let mut gate_total = 0i32;

                // Right score.
                let mut gate_tile = 0i32;
                let mut gate_prev_score = 0i32;
                let mut gate_prev_max = 0i32;
                loop {
                    let pos_offset = lane_id as i32 + gate_tile;
                    let ref_pos = hit_ref_loc.wrapping_add(pos_offset as u32);
                    let query_pos = hit_query_loc.wrapping_add(pos_offset as u32);
                    let mut gate_score = 0i32;
                    if ref_pos < ref_len && query_pos < query_len {
                        let r = unsafe { *ref_seq.get_unchecked(ref_pos as usize) } as usize;
                        let q = unsafe { *query_seq.get_unchecked(query_pos as usize) } as usize;
                        gate_score = unsafe { SUB[r * NUC + q] };
                    }

                    gate_score = prefix_sum(lane_id, gate_score) + gate_prev_score;
                    let candidate = if gate_score > gate_prev_max {
                        gate_score
                    } else {
                        gate_prev_max
                    };
                    let scanned = prefix_max_score(candidate);
                    let dropped = warp::ballot_sync(FULL_MASK, (scanned - gate_score) > xdrop);
                    let first = dropped.trailing_zeros();
                    let gate_max = if first == 0 {
                        gate_prev_max
                    } else {
                        warp::shuffle_sync(FULL_MASK, scanned as u32, first - 1) as i32
                    };
                    let last_offset = gate_tile + WARP_SIZE as i32 - 1;
                    let edge = hit_ref_loc.wrapping_add(last_offset as u32) >= ref_len
                        || hit_query_loc.wrapping_add(last_offset as u32) >= query_len;
                    if dropped != 0 || edge {
                        gate_total += gate_max;
                        break;
                    }

                    gate_prev_score =
                        warp::shuffle_sync(FULL_MASK, gate_score as u32, WARP_SIZE - 1) as i32;
                    gate_prev_max = gate_max;
                    gate_tile += WARP_SIZE as i32;
                }

                // Left score. The last lane evaluates offset tile + 32.
                gate_tile = 0;
                gate_prev_score = 0;
                gate_prev_max = 0;
                loop {
                    let pos_offset = lane_id as i32 + 1 + gate_tile;
                    let mut gate_score = 0i32;
                    if hit_ref_loc >= pos_offset as u32 && hit_query_loc >= pos_offset as u32 {
                        let r = unsafe {
                            *ref_seq.get_unchecked((hit_ref_loc - pos_offset as u32) as usize)
                        } as usize;
                        let q = unsafe {
                            *query_seq.get_unchecked((hit_query_loc - pos_offset as u32) as usize)
                        } as usize;
                        gate_score = unsafe { SUB[r * NUC + q] };
                    }

                    gate_score = prefix_sum(lane_id, gate_score) + gate_prev_score;
                    let candidate = if gate_score > gate_prev_max {
                        gate_score
                    } else {
                        gate_prev_max
                    };
                    let scanned = prefix_max_score(candidate);
                    let dropped = warp::ballot_sync(FULL_MASK, (scanned - gate_score) > xdrop);
                    let first = dropped.trailing_zeros();
                    let gate_max = if first == 0 {
                        gate_prev_max
                    } else {
                        warp::shuffle_sync(FULL_MASK, scanned as u32, first - 1) as i32
                    };
                    let last_offset = gate_tile + WARP_SIZE as i32;
                    let edge =
                        hit_ref_loc < last_offset as u32 || hit_query_loc < last_offset as u32;
                    if dropped != 0 || edge {
                        gate_total += gate_max;
                        break;
                    }

                    gate_prev_score =
                        warp::shuffle_sync(FULL_MASK, gate_score as u32, WARP_SIZE - 1) as i32;
                    gate_prev_max = gate_max;
                    gate_tile += WARP_SIZE as i32;
                }

                // Production rounds through f32 before its final threshold
                // comparison. Mirroring that upper bound also preserves exact
                // output for unusually large custom scores near an f32 ULP.
                let gate_upper = (gate_total as f32 as f64) as i32;
                if gate_upper < hspthresh {
                    if hid < num_hits && lane_id == 0 {
                        unsafe {
                            *hsp.get_unchecked_mut(hid as usize) = SegmentPair {
                                ref_start: hit_ref_loc,
                                query_start: hit_query_loc,
                                len: 0,
                                score: 0,
                            };
                            *done.get_unchecked_mut(hid as usize) = 0;
                        }
                    }
                    warp::sync_mask(FULL_MASK);
                    hid0 += NUM_WARPS as u32 * num_blocks;
                    continue;
                }
            }

            // ---------------------------------------------------------------
            // Right extension
            if lane_id == 0 {
                unsafe {
                    TILE[w] = 0;
                    XDROP_FOUND[w] = 0;
                    EDGE_FOUND[w] = 0;
                    ENTROPY[w] = 1.0;
                    PREV_SCORE[w] = 0;
                    PREV_MAX_SCORE[w] = 0;
                    PREV_MAX_POS[w] = -1;
                    EXTENT[w] = 0;
                }
            }
            warp::sync_mask(FULL_MASK);

            loop {
                #[cfg(feature = "counters")]
                {
                    right_tiles += 1;
                }
                let pos_offset = lane_id as i32 + unsafe { TILE[w] };
                let ref_pos = hit_ref_loc.wrapping_add(pos_offset as u32);
                let query_pos = hit_query_loc.wrapping_add(pos_offset as u32);
                let mut thread_score = 0i32;

                if ref_pos < ref_len && query_pos < query_len {
                    // SAFETY: `ref_len`/`query_len` are the lengths of these
                    // very slices, and the branch above just tested against
                    // them — so Rust's bounds checks here are provably dead. The
                    // census says they are not free: `cvt` + `setp` + `bra`
                    // twice, 6 of ~129 instructions in a kernel running at ~73%
                    // of issue throughput.
                    let r_chr = unsafe { *ref_seq.get_unchecked(ref_pos as usize) } as usize;
                    let q_chr = unsafe { *query_seq.get_unchecked(query_pos as usize) } as usize;
                    thread_score = unsafe { SUB[r_chr * NUC + q_chr] };
                }
                warp::sync_mask(FULL_MASK);

                thread_score = prefix_sum(lane_id, thread_score);
                thread_score += unsafe { PREV_SCORE[w] };

                let prev_max_score = unsafe { PREV_MAX_SCORE[w] };
                let candidate = if thread_score > prev_max_score {
                    thread_score
                } else {
                    prev_max_score
                };
                warp::sync_mask(FULL_MASK);
                // Score only: the winning position is recovered once below
                // rather than dragged through every step of the scan.
                let scanned = prefix_max_score(candidate);

                // KegAlign runs a prefix-OR of the per-lane X-drop test, resets
                // the dropped lanes to the tile-entry maximum, then re-runs the
                // prefix max to undo the reset — 15 shuffles to answer one
                // question. The answer is fully determined by the *first* lane
                // that drops: lanes before it keep their running maximum, which
                // is monotonic, so the survivor is simply lane `first - 1`.
                // Ties inside that run carry the earliest position either way,
                // so the result is identical, not merely equivalent.
                let dropped = warp::ballot_sync(FULL_MASK, (scanned - thread_score) > xdrop);
                let xdrop_done = dropped != 0;
                let first = dropped.trailing_zeros(); // 32 when nothing dropped
                #[cfg(feature = "counters")]
                if right_tiles == 1 {
                    first_drop_r = first as u64;
                }
                let max_score = if first == 0 {
                    prev_max_score
                } else {
                    warp::shuffle_sync(FULL_MASK, scanned as u32, first - 1) as i32
                };
                // Position recovery. A candidate only exceeds the inherited
                // maximum strictly, so a winner equal to it means no lane
                // improved and the inherited position stands. Otherwise the
                // winner is some lane's own score, and because the scan reads
                // lane `L - offset` and takes it on `>=`, the position the
                // carried version would have kept is the EARLIEST such lane's.
                let max_pos = if max_score == prev_max_score {
                    unsafe { PREV_MAX_POS[w] }
                } else {
                    let winners =
                        warp::ballot_sync(FULL_MASK, thread_score == max_score && lane_id < first);
                    winners.trailing_zeros() as i32 + (pos_offset - lane_id as i32)
                };
                warp::sync_mask(FULL_MASK);

                if last_lane {
                    unsafe {
                        if xdrop_done {
                            #[cfg(feature = "counters")]
                            {
                                TERM[w] = 1;
                            }
                            TOTAL_SCORE[w] += max_score;
                            XDROP_FOUND[w] = 1;
                            EXTENT[w] = max_pos;
                            PREV_MAX_POS[w] = max_pos;
                            TILE[w] = max_pos;
                        } else if ref_pos >= ref_len || query_pos >= query_len {
                            #[cfg(feature = "counters")]
                            {
                                TERM[w] = if ref_pos >= ref_len { 2 } else { 4 };
                            }
                            TOTAL_SCORE[w] += max_score;
                            EDGE_FOUND[w] = 1;
                            EXTENT[w] = max_pos;
                            PREV_MAX_POS[w] = max_pos;
                            TILE[w] = max_pos;
                        } else {
                            PREV_SCORE[w] = thread_score;
                            PREV_MAX_SCORE[w] = max_score;
                            PREV_MAX_POS[w] = max_pos;
                            TILE[w] += WARP_SIZE as i32;
                        }
                    }
                }
                warp::sync_mask(FULL_MASK);

                // Peeled entry check: both flags were cleared immediately
                // above, so testing them before the first tile is dead work —
                // and with ~91% of candidates running exactly one right tile it
                // is paid once per candidate rather than amortized.
                if unsafe { XDROP_FOUND[w] != 0 || EDGE_FOUND[w] != 0 } {
                    break;
                }
            }
            warp::sync_mask(FULL_MASK);
            #[cfg(feature = "counters")]
            {
                term |= (unsafe { TERM[w] } as u64) << 40;
                if lane_id == 0 {
                    unsafe { TERM[w] = 0 };
                }
            }
            warp::sync_mask(FULL_MASK);

            // ---------------------------------------------------------------
            // Left extension
            if lane_id == 0 {
                unsafe {
                    TILE[w] = 0;
                    XDROP_FOUND[w] = 0;
                    EDGE_FOUND[w] = 0;
                    PREV_SCORE[w] = 0;
                    PREV_MAX_SCORE[w] = 0;
                    PREV_MAX_POS[w] = 0;
                    LEFT_EXTENT[w] = 0;
                }
            }
            warp::sync_mask(FULL_MASK);

            loop {
                #[cfg(feature = "counters")]
                {
                    left_tiles += 1;
                }
                let pos_offset = lane_id as i32 + 1 + unsafe { TILE[w] };
                let (ref_loc, query_loc) = (hit_ref_loc, hit_query_loc);
                let mut thread_score = 0i32;

                if ref_loc >= pos_offset as u32 && query_loc >= pos_offset as u32 {
                    // SAFETY: `pos_offset >= 1` here, and the anchors satisfy
                    // `ref_loc <= ref_len` / `query_loc <= query_len` (a seed
                    // hit ends at most at the sequence end), so both indices are
                    // at most `len - 1`. Same dead-bounds-check argument as the
                    // right extension.
                    let r_chr =
                        unsafe { *ref_seq.get_unchecked((ref_loc - pos_offset as u32) as usize) }
                            as usize;
                    let q_chr = unsafe {
                        *query_seq.get_unchecked((query_loc - pos_offset as u32) as usize)
                    } as usize;
                    thread_score = unsafe { SUB[r_chr * NUC + q_chr] };
                }

                thread_score = prefix_sum(lane_id, thread_score);
                thread_score += unsafe { PREV_SCORE[w] };

                let prev_max_score = unsafe { PREV_MAX_SCORE[w] };
                let candidate = if thread_score > prev_max_score {
                    thread_score
                } else {
                    prev_max_score
                };
                warp::sync_mask(FULL_MASK);
                // Score only: the winning position is recovered once below
                // rather than dragged through every step of the scan.
                let scanned = prefix_max_score(candidate);

                // KegAlign runs a prefix-OR of the per-lane X-drop test, resets
                // the dropped lanes to the tile-entry maximum, then re-runs the
                // prefix max to undo the reset — 15 shuffles to answer one
                // question. The answer is fully determined by the *first* lane
                // that drops: lanes before it keep their running maximum, which
                // is monotonic, so the survivor is simply lane `first - 1`.
                // Ties inside that run carry the earliest position either way,
                // so the result is identical, not merely equivalent.
                let dropped = warp::ballot_sync(FULL_MASK, (scanned - thread_score) > xdrop);
                let xdrop_done = dropped != 0;
                let first = dropped.trailing_zeros(); // 32 when nothing dropped
                #[cfg(feature = "counters")]
                if left_tiles == 1 {
                    first_drop_l = first as u64;
                }
                let max_score = if first == 0 {
                    prev_max_score
                } else {
                    warp::shuffle_sync(FULL_MASK, scanned as u32, first - 1) as i32
                };
                // Position recovery. A candidate only exceeds the inherited
                // maximum strictly, so a winner equal to it means no lane
                // improved and the inherited position stands. Otherwise the
                // winner is some lane's own score, and because the scan reads
                // lane `L - offset` and takes it on `>=`, the position the
                // carried version would have kept is the EARLIEST such lane's.
                let max_pos = if max_score == prev_max_score {
                    unsafe { PREV_MAX_POS[w] }
                } else {
                    let winners =
                        warp::ballot_sync(FULL_MASK, thread_score == max_score && lane_id < first);
                    winners.trailing_zeros() as i32 + (pos_offset - lane_id as i32)
                };
                warp::sync_mask(FULL_MASK);

                if last_lane {
                    unsafe {
                        if xdrop_done {
                            #[cfg(feature = "counters")]
                            {
                                TERM[w] = 1;
                            }
                            TOTAL_SCORE[w] += max_score;
                            XDROP_FOUND[w] = 1;
                            LEFT_EXTENT[w] = max_pos as u32;
                            EXTENT[w] += LEFT_EXTENT[w] as i32;
                            PREV_MAX_POS[w] = max_pos;
                            TILE[w] = max_pos;
                        } else if ref_loc < pos_offset as u32 || query_loc < pos_offset as u32 {
                            #[cfg(feature = "counters")]
                            {
                                TERM[w] = if ref_loc < pos_offset as u32 { 2 } else { 4 };
                            }
                            TOTAL_SCORE[w] += max_score;
                            EDGE_FOUND[w] = 1;
                            LEFT_EXTENT[w] = max_pos as u32;
                            EXTENT[w] += LEFT_EXTENT[w] as i32;
                            PREV_MAX_POS[w] = max_pos;
                            TILE[w] = max_pos;
                        } else {
                            PREV_SCORE[w] = thread_score;
                            PREV_MAX_SCORE[w] = max_score;
                            PREV_MAX_POS[w] = max_pos;
                            TILE[w] += WARP_SIZE as i32;
                        }
                    }
                }
                warp::sync_mask(FULL_MASK);

                // Same peel on the left loop, which every candidate enters.
                if unsafe { XDROP_FOUND[w] != 0 || EDGE_FOUND[w] != 0 } {
                    break;
                }
            }

            #[cfg(feature = "counters")]
            {
                term |= (unsafe { TERM[w] } as u64) << 43;
                if lane_id == 0 {
                    unsafe { TERM[w] = 0 };
                }
            }
            #[cfg(feature = "counters")]
            warp::sync_mask(FULL_MASK);

            // ---------------------------------------------------------------
            // Entropy correction
            let total_score = unsafe { TOTAL_SCORE[w] };
            if total_score >= hspthresh && total_score <= 3 * hspthresh && noentropy == 0 {
                // KegAlign maintains these counts through every tile of every
                // extension, but only candidates inside the score band ever read
                // them — 0.0148% of hits on a mammalian block. Recounting here
                // costs one re-walk of an interval that is 3 tiles long on
                // average, and skips the bookkeeping for the other 99.985%.
                //
                // The interval is CLOSED: both extension maxima are counted
                // positions, so it spans `len + 1` bases starting at the HSP's
                // reference start. Only matching, unambiguous pairs count, which
                // is exactly what the incremental version accumulated: anything
                // beyond the running maximum went to `count_del` and was
                // discarded, and out-of-range lanes always sat beyond it.
                let left = unsafe { LEFT_EXTENT[w] };
                let ref_start = hit_ref_loc - left;
                let query_start = hit_query_loc - left;
                let positions = unsafe { EXTENT[w] } + 1;
                // All four counts ride in one u64, 16 bits each: accumulation
                // is branchless and the warp reduction is a single scan instead
                // of four, with less live state in the hottest kernel. That is
                // why the packed form is kept.
                //
                // It also happens to be a workaround: the four-accumulator,
                // four-scan version produced PTX that `ptxas` accepted but the
                // pinned ZLUDA backend refused to load with `DriverError(500)`.
                // Bisected to that construct — not the loop, not the indexing.
                // The packed form is retained on its own merits; if native
                // NVIDIA measurement ever shows the unpacked form wins there,
                // that is when specialization gets earned.
                //
                // each field holds at most 65535. Overflow would
                // carry into the neighbouring base's count, so the ceiling is
                // load-bearing: entropy only runs when total_score is within
                // [hspthresh, 3*hspthresh] and every match adds ~91-100, which
                // bounds matches at roughly 3*hspthresh/91 — unreachable below
                // an hspthresh in the millions (upstream's `short count[4]`
                // wraps even earlier). Widen to 32-bit fields across two u64s
                // if that ever becomes reachable.
                // All four counts ride in one u64, 16 bits each: accumulation
                // is branchless and the warp reduction is a single scan instead
                // of four, with less live state in the hottest kernel.
                //
                // The four-counter/four-scan alternative (N6) was implemented
                // and measured on an NVIDIA L4: `find_hsps` +1.90% (A), +1.36%
                // (B), +0.50% (apple) — worse on every workload. The codegen
                // census said why: 42 shuffles vs 22, 9/6 local stores/loads vs
                // 4/1, and 53 more 32-bit registers, because `c[r]` with a
                // runtime index spills. Packed wins on native CUDA on its own
                // merits, not just as the ZLUDA workaround it originally was
                // (round 15).
                //
                // each field holds at most 65535. Overflow would
                // carry into the neighbouring base's count, so the ceiling is
                // load-bearing: entropy only runs when total_score is within
                // [hspthresh, 3*hspthresh] and every match adds ~91-100, which
                // bounds matches at roughly 3*hspthresh/91 — unreachable below
                // an hspthresh in the millions (upstream's `short count[4]`
                // wraps even earlier). Widen to 32-bit fields across two u64s
                // if that ever becomes reachable.
                let mut packed = 0u64;
                let mut i = lane_id as i32;
                while i < positions {
                    let r = ref_seq[(ref_start + i as u32) as usize] as usize;
                    let q = query_seq[(query_start + i as u32) as usize] as usize;
                    if r == q && r < 4 {
                        packed += 1u64 << (16 * r);
                    }
                    i += WARP_SIZE as i32;
                }
                // Lane 31 holds the warp total, which is where entropy is
                // computed and the only place these are read.
                let packed = prefix_sum_u64(lane_id, packed);
                let count = [
                    (packed & 0xFFFF) as i32,
                    ((packed >> 16) & 0xFFFF) as i32,
                    ((packed >> 32) & 0xFFFF) as i32,
                    ((packed >> 48) & 0xFFFF) as i32,
                ];
                warp::sync_mask(FULL_MASK);

                let total_count = count[0] + count[1] + count[2] + count[3];
                #[cfg(feature = "counters")]
                {
                    term |= 1 << 46;
                }
                #[cfg(feature = "counters")]
                if last_lane && total_count >= 20 {
                    // Only lane 31 knows this; lane 0 does the store, so it has
                    // to travel through shared memory.
                    unsafe { TERM[w] = 1 };
                }
                #[cfg(feature = "counters")]
                warp::sync_mask(FULL_MASK);
                #[cfg(feature = "counters")]
                {
                    term |= (unsafe { TERM[w] } as u64 & 1) << 47;
                }
                if last_lane && total_count >= 20 {
                    let denom = (unsafe { EXTENT[w] } + 1) as f64;
                    let mut entropy = 0.0f64;
                    for i in 0..4 {
                        let p = count[i] as f64 / denom;
                        entropy += p * if count[i] != 0 { ln(p) } else { 0.0 };
                    }
                    unsafe {
                        ENTROPY[w] = -entropy / log4;
                    }
                }
            }
            warp::sync_mask(FULL_MASK);

            if hid < num_hits && lane_id == 0 {
                let entropy = unsafe { ENTROPY[w] };
                let extent = unsafe { EXTENT[w] };
                // SAFETY: `hid < num_hits`, and both slices are sized to the
                // chunk's hit count.
                unsafe {
                    if ((total_score as f32 as f64) * entropy) as i32 >= hspthresh {
                        let left = LEFT_EXTENT[w];
                        *hsp.get_unchecked_mut(hid as usize) = SegmentPair {
                            ref_start: hit_ref_loc - left,
                            query_start: hit_query_loc - left,
                            len: extent as u32,
                            score: if entropy > 0.0 {
                                (total_score as f64 * entropy) as i32
                            } else {
                                0
                            },
                        };
                        *done.get_unchecked_mut(hid as usize) = 1;
                        #[cfg(feature = "counters")]
                        {
                            term |= 1 << 48;
                        }
                    } else {
                        *hsp.get_unchecked_mut(hid as usize) = SegmentPair {
                            ref_start: hit_ref_loc,
                            query_start: hit_query_loc,
                            len: 0,
                            score: 0,
                        };
                        *done.get_unchecked_mut(hid as usize) = 0;
                    }
                }
            }
            #[cfg(feature = "counters")]
            if hid < num_hits && lane_id == 0 {
                // SAFETY: `hid < num_hits <= stats.len()`.
                unsafe {
                    // Two words per candidate: the tile/termination record, and
                    // the anchor, from which the host derives the diagonal and
                    // the tile-quantized evaluated interval.
                    *stats.get_unchecked_mut(2 * hid as usize) = (right_tiles & 0xF_FFFF)
                        | ((left_tiles & 0xF_FFFF) << 20)
                        | term
                        | (first_drop_r << 49)
                        | (first_drop_l << 55);
                    *stats.get_unchecked_mut(2 * hid as usize + 1) =
                        hit_ref_loc as u64 | ((hit_query_loc as u64) << 32);
                }
            }
            warp::sync_mask(FULL_MASK);

            hid0 += NUM_WARPS as u32 * num_blocks;
        }
    }

    /// `compress_output`: stream compaction driven by the inclusive scan of
    /// `done`.
    #[kernel]
    pub fn compress_output(
        done: &[u32],
        hsp: &[SegmentPair],
        mut reduced: DisjointSlice<SegmentPair>,
        num_hits: u32,
    ) {
        let stride = thread::blockDim_x() * thread::gridDim_x();
        let mut id = thread::blockDim_x() * thread::blockIdx_x() + thread::threadIdx_x();

        while id < num_hits {
            let index = done[id as usize];
            if id > 0 {
                if index > done[id as usize - 1] {
                    // SAFETY: `index <= num_anchors <= reduced.len()`.
                    unsafe {
                        *reduced.get_unchecked_mut(index as usize - 1) = hsp[id as usize];
                    }
                }
            } else if index == 1 {
                unsafe {
                    *reduced.get_unchecked_mut(0) = hsp[0];
                }
            }
            id += stride;
        }
    }

    /// Packed 4-byte load of encoded sequence. Addresses are consecutive `u32`s
    /// across a group, so neighboring lanes still touch neighboring bases.
    #[cfg(feature = "simd-prelude")]
    #[inline(always)]
    unsafe fn ld_u32(seq: &[u8], pos: u32, len: u32) -> u32 {
        // `read_unaligned` becomes four `ld.global.b8`, which is what this exists to
        // avoid. A volatile 32-bit load is one `ld.global.b32` — but NVIDIA *faults*
        // on an unaligned one (`DriverError(716, "misaligned address")`), while RDNA
        // through ZLUDA tolerates it. That asymmetry shipped a default build that
        // could not run on a T4 at all; every arm of Kaggle kernel v22 died on it.
        //
        // Read the two aligned words that straddle the window and funnel them. The
        // second word is only touched when it is wholly inside the sequence, so the
        // tail of a record falls back to bytes rather than reading past the buffer.
        let base = (pos & !3) as usize;
        let sh = (pos & 3) * 8;
        let ptr = unsafe { seq.as_ptr().add(base) as *const u32 };
        if sh == 0 {
            return unsafe { core::ptr::read_volatile(ptr) };
        }
        if base as u32 + 8 <= len {
            let lo = unsafe { core::ptr::read_volatile(ptr) };
            let hi = unsafe { core::ptr::read_volatile(ptr.add(1)) };
            return (lo >> sh) | (hi << (32 - sh));
        }
        let mut w = 0u32;
        let mut k = 0u32;
        while k < 4 {
            w |= (unsafe { *seq.get_unchecked((pos + k) as usize) } as u32) << (8 * k);
            k += 1;
        }
        w
    }

    /// One scan carrying both quantities the prelude needs, instead of a sum scan
    /// followed by a max scan that has to wait for it.
    ///
    /// `(sum, max)` is a monoid under `(s1,m1) . (s2,m2) = (s1+s2, max(m1, s1+m2))` —
    /// total score and best prefix score of a concatenation. Scanned inclusively it
    /// gives every lane its prefix sum *and* the running maximum over all prefixes,
    /// which is what the two serial scans produce.
    ///
    /// Round 51 measured why depth is the thing to cut: removing 9 of 22 prelude
    /// shuffles changed nothing (+0.00%), while adding 8 *on the dependency chain* cost
    /// 3%. Same shuffle count here, half the dependent depth.
    #[inline(always)]
    fn dual_prefix_sum_max(
        lane_id: u32,
        is_right: bool,
        is_left: bool,
        gl: u32,
        mut s: i32,
        mut m: i32,
    ) -> (i32, i32) {
        let mut offset = 1u32;
        while offset < 16 {
            let src = if gl >= offset {
                lane_id - offset
            } else {
                lane_id
            };
            let up_s = warp::shuffle_sync(FULL_MASK, s as u32, src) as i32;
            let up_m = warp::shuffle_sync(FULL_MASK, m as u32, src) as i32;
            if (is_right || is_left) && gl >= offset {
                // the shuffled lane is the earlier segment: (up_s, up_m) . (s, m)
                let cand = up_s.wrapping_add(m);
                m = if up_m >= cand { up_m } else { cand };
                s = up_s.wrapping_add(s);
            }
            offset <<= 1;
        }
        (s, m)
    }

    /// `__shfl_up_sync` inclusive add-scan (`thread_score` in both extensions).
    #[inline(always)]
    fn prefix_sum(lane_id: u32, mut v: i32) -> i32 {
        let mut offset = 1;
        while offset < WARP_SIZE {
            let up = warp::shuffle_up_sync(FULL_MASK, v as u32, offset) as i32;
            if lane_id >= offset {
                v += up;
            }
            offset <<= 1;
        }
        v
    }

    /// Inclusive add-scan over `u64`, used to reduce the packed entropy counts.
    #[inline(always)]
    fn prefix_sum_u64(lane_id: u32, mut v: u64) -> u64 {
        let mut offset = 1;
        while offset < WARP_SIZE {
            let up = warp::shuffle_up_u64_sync(FULL_MASK, v, offset);
            if lane_id >= offset {
                v += up;
            }
            offset <<= 1;
        }
        v
    }

    /// Inclusive prefix max over scores. `shuffle_up` reads lane `L - offset`,
    /// so taking it on `>=` resolves ties to the *earliest* lane — which is what
    /// pins the argmax the caller recovers, and what KegAlign depends on.
    #[inline(always)]
    fn prefix_max_score(mut score: i32) -> i32 {
        let mut offset = 1;
        while offset < WARP_SIZE {
            let up = warp::shuffle_up_sync(FULL_MASK, score as u32, offset) as i32;
            // No `lane_id >= offset` guard. `shuffle_up` hands a lane its own
            // value when the source lane is out of range, so for lanes below
            // `offset` this reduces to `max(score, score)` — already a no-op,
            // and the tie rule is unaffected because the value is unchanged.
            // The sum scan genuinely needs its guard (there it would double the
            // value); this one was paying 5 `selp` per tile for nothing.
            if up >= score {
                score = up;
            }
            offset <<= 1;
        }
        score
    }

    /// Natural log for `x` in `(0, 1]`.
    ///
    /// libdevice's `__nv_log` is not reachable from cuda-oxide, so this is an
    /// atanh series over the reduced mantissa, truncated after the `z^7/15` term.
    /// Measured on p = c/d, 1 <= c < d <= 4096 (2026-09-28): relative error up to
    /// 3.4e-14 (212 ulp, near m = sqrt 2), and the bits differ from `__nv_log` in
    /// about half of all arguments. A score or the threshold test flips only when
    /// `total * entropy` lies within ~1e-10 of an integer — estimated at ~2e-4
    /// expected differences per whole-genome run; none has been observed. A port
    /// of `__nv_log` would make this bit-identical to the oracle by construction.
    #[inline(always)]
    fn ln(x: f64) -> f64 {
        const LN2: f64 = core::f64::consts::LN_2;
        const SQRT2: f64 = core::f64::consts::SQRT_2;

        let bits = x.to_bits();
        let mut exp = ((bits >> 52) & 0x7FF) as i32 - 1023;
        let mut m = f64::from_bits((bits & 0x000F_FFFF_FFFF_FFFF) | 0x3FF0_0000_0000_0000);
        if m > SQRT2 {
            m *= 0.5;
            exp += 1;
        }

        // ln(m) = 2 * atanh((m-1)/(m+1)); |s| <= 0.1716 so z <= 0.0295.
        let s = (m - 1.0) / (m + 1.0);
        let z = s * s;
        let poly = 1.0
            + z * (1.0 / 3.0
                + z * (1.0 / 5.0
                    + z * (1.0 / 7.0
                        + z * (1.0 / 9.0
                            + z * (1.0 / 11.0 + z * (1.0 / 13.0 + z * (1.0 / 15.0)))))));
        2.0 * s * poly + exp as f64 * LN2
    }
}

/// Host models of the per-tile maximum logic, used to prove that carrying the
/// position through the prefix scan and recovering it once at the end agree on
/// every tie case before any of it reaches the GPU.
#[cfg(test)]
mod tests {
    const W: usize = 32;

    /// Round 83 round 2 (`ref-loc-buckets`): host mirrors of `emit_bits`'
    /// ascending bit enumeration and of `sort_survivors`' bitonic network, over
    /// both survivor orders, a masked tail word, an all-zero and an all-one
    /// bitmask. Both device paths are exact algorithms, so the mirror is the
    /// specification: the emitted list is the raw ids at the set bits in
    /// ascending SORTED-slot order, and the restored list must be ascending RAW
    /// order — what `dump_raw`, the census walk and dedup ties read. Above
    /// `SORT_MAX` the fallback is `sort_unstable`, which must agree with it.
    #[cfg(feature = "ref-loc-buckets")]
    #[test]
    fn sorted_bitmask_emits_ascending_slots_and_restores_ascending_ids() {
        use super::SORT_MAX;

        // `emit_bits`: ascending bits inside a word, ascending words.
        fn emit(bits: &[u32], sorted_idx: &[u32], n: u32) -> Vec<u32> {
            let mut out = Vec::new();
            for (w, &word) in bits.iter().enumerate() {
                let mut m = word;
                while m != 0 {
                    let slot = w as u32 * 32 + m.trailing_zeros();
                    assert!(slot < n, "set bit at slot {slot} past {n} hits");
                    out.push(sorted_idx[slot as usize]);
                    m &= m - 1;
                }
            }
            out
        }

        // `sort_survivors`: one block, bitonic over SORT_MAX slots padded with
        // `u32::MAX` (no hit id can take that value).
        fn bitonic(ids: &[u32]) -> Vec<u32> {
            let mut s = vec![u32::MAX; SORT_MAX as usize];
            s[..ids.len()].copy_from_slice(ids);
            let mut k = 2usize;
            while k <= SORT_MAX as usize {
                let mut j = k / 2;
                while j > 0 {
                    for t in 0..SORT_MAX as usize {
                        let l = t ^ j;
                        if l > t && (s[t] > s[l]) == ((t & k) == 0) {
                            s.swap(t, l);
                        }
                    }
                    j >>= 1;
                }
                k <<= 1;
            }
            s.truncate(ids.len());
            s
        }

        let n: u32 = 100; // 3 full words plus a 4-bit tail
        let words = n.div_ceil(32) as usize;
        // Slot -> raw id: the identity permutation, and the reversal, which is
        // the worst case for the restoration.
        let ident: Vec<u32> = (0..n).collect();
        let rev: Vec<u32> = (0..n).rev().collect();

        // All zero: nothing emitted, nothing to sort.
        let zero = vec![0u32; words];
        assert!(emit(&zero, &ident, n).is_empty());

        // All one with the tail masked, both permutations.
        let mut ones = vec![u32::MAX; words];
        ones[words - 1] = (1u32 << (n % 32)) - 1;
        for map in [&ident, &rev] {
            let got = emit(&ones, map, n);
            assert_eq!(got.len() as u32, n, "every hit survives");
            assert_eq!(got, map[..n as usize].to_vec(), "ascending slot order");
            let sorted = bitonic(&got);
            assert_eq!(sorted, ident, "restored to ascending raw ids");
            let mut host = got.clone();
            host.sort_unstable();
            assert_eq!(host, sorted, "host fallback agrees with the network");
        }

        // A scattered pattern, including the tail word.
        let slots: Vec<u32> = vec![0, 1, 31, 32, 63, 64, 96, 99];
        let mut bits = vec![0u32; words];
        for &sl in &slots {
            bits[(sl / 32) as usize] |= 1 << (sl % 32);
        }
        for map in [&ident, &rev] {
            let got = emit(&bits, map, n);
            let expect: Vec<u32> = slots.iter().map(|&sl| map[sl as usize]).collect();
            assert_eq!(got, expect, "ascending slot order, not raw order");
            let sorted = bitonic(&got);
            let mut expect_sorted = expect.clone();
            expect_sorted.sort_unstable();
            assert_eq!(sorted, expect_sorted, "restored to ascending raw ids");
        }
    }

    /// Hillis-Steele prefix max over `(score, pos)`, taking the earlier lane on
    /// ties — exactly what `prefix_max` does, since `shuffle_up` reads lane
    /// `L - offset`.
    fn scan_pairs(v: &mut [(i32, i32); W]) {
        let mut offset = 1usize;
        while offset < W {
            let prev = *v;
            for l in offset..W {
                if prev[l - offset].0 >= prev[l].0 {
                    v[l] = prev[l - offset];
                }
            }
            offset <<= 1;
        }
    }

    fn scan_scores(v: &mut [i32; W]) {
        let mut offset = 1usize;
        while offset < W {
            let prev = *v;
            for l in offset..W {
                if prev[l - offset] >= prev[l] {
                    v[l] = prev[l - offset];
                }
            }
            offset <<= 1;
        }
    }

    /// The unguarded form the kernel now uses: lanes below `offset` see their
    /// own value from `shuffle_up`, so the step is `max(x, x)`.
    fn scan_scores_unguarded(v: &mut [i32; W]) {
        let mut offset = 1usize;
        while offset < W {
            let prev = *v;
            for l in 0..W {
                let up = if l >= offset {
                    prev[l - offset]
                } else {
                    prev[l]
                };
                if up >= prev[l] {
                    v[l] = up;
                }
            }
            offset <<= 1;
        }
    }

    #[test]
    fn dropping_the_max_scan_lane_guard_is_a_no_op() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..50_000 {
            let mut a = [0i32; W];
            for slot in a.iter_mut() {
                *slot = (rnd() % 11) as i32 - 5;
            }
            let (mut g, mut u) = (a, a);
            scan_scores(&mut g);
            scan_scores_unguarded(&mut u);
            assert_eq!(g, u, "guard removal changed the scan for {a:?}");
        }
    }

    fn first_drop(scanned: &[i32; W], ts: &[i32; W], xdrop: i32) -> usize {
        (0..W)
            .find(|&l| scanned[l].wrapping_sub(ts[l]) > xdrop)
            .unwrap_or(W)
    }

    /// Current production shape: position rides through the scan.
    fn carried(ts: &[i32; W], pms: i32, pmp: i32, xdrop: i32, base: i32) -> (i32, i32) {
        let mut v = [(0i32, 0i32); W];
        for l in 0..W {
            v[l] = if ts[l] > pms {
                (ts[l], l as i32 + base)
            } else {
                (pms, pmp)
            };
        }
        scan_pairs(&mut v);
        let scores = core::array::from_fn(|l| v[l].0);
        let first = first_drop(&scores, ts, xdrop);
        if first == 0 { (pms, pmp) } else { v[first - 1] }
    }

    /// Proposed shape: scan scores only, recover the position once.
    fn recovered(ts: &[i32; W], pms: i32, pmp: i32, xdrop: i32, base: i32) -> (i32, i32) {
        let mut scores: [i32; W] = core::array::from_fn(|l| if ts[l] > pms { ts[l] } else { pms });
        scan_scores(&mut scores);
        let first = first_drop(&scores, ts, xdrop);
        let winning = if first == 0 { pms } else { scores[first - 1] };
        let pos = if winning == pms {
            pmp
        } else {
            // Earliest lane before the drop whose score is the winner.
            (0..first)
                .find(|&l| ts[l] == winning)
                .expect("a lane produced the winner") as i32
                + base
        };
        (winning, pos)
    }

    #[test]
    fn score_gate_uses_both_extensions_and_the_production_upper_bound() {
        let (right, left, threshold) = (1_400i32, 1_600i32, 3_000i32);
        assert!(
            right < threshold,
            "the right extension alone must not decide the gate"
        );
        assert!(((right + left) as f32 as f64) as i32 >= threshold);

        // Above 2^24 the production f32 conversion can round a raw score up.
        // The gate mirrors it so unusual custom scores cannot lose an HSP.
        let (raw, threshold) = (16_777_219i32, 16_777_220i32);
        assert!(raw < threshold);
        assert!((raw as f32 as f64) as i32 >= threshold);
    }

    #[cfg(feature = "left-pair-tile")]
    fn scan_sums_wrapping(v: &mut [i32; W]) {
        let mut offset = 1usize;
        while offset < W {
            let prev = *v;
            for l in offset..W {
                v[l] = prev[l].wrapping_add(prev[l - offset]);
            }
            offset <<= 1;
        }
    }

    #[cfg(feature = "left-pair-tile")]
    fn old_left_from(
        scores: &[i32],
        valid: usize,
        xdrop: i32,
        mut tile: usize,
        mut prev_score: i32,
        mut prev_max: i32,
    ) -> i32 {
        loop {
            let mut cumulative = [0i32; W];
            let mut acc = prev_score;
            for (lane, score) in cumulative.iter_mut().enumerate() {
                acc = acc.wrapping_add(if tile + lane < valid {
                    scores[tile + lane]
                } else {
                    0
                });
                *score = acc;
            }
            let mut maxima = cumulative.map(|score| score.max(prev_max));
            scan_scores(&mut maxima);
            let first = first_drop(&maxima, &cumulative, xdrop);
            let max = if first == 0 {
                prev_max
            } else {
                maxima[first - 1]
            };
            if first < W || valid < tile + W {
                return max;
            }
            prev_score = cumulative[W - 1];
            prev_max = max;
            tile += W;
        }
    }

    #[cfg(feature = "left-pair-tile")]
    fn paired_prelude(
        scores: &[i32],
        valid: usize,
        xdrop: i32,
        prev_score: i32,
        prev_max: i32,
    ) -> (usize, i32, i32) {
        let mut first_scores = [0i32; W];
        let mut second_scores = [0i32; W];
        let mut pair_sums = [0i32; W];
        for lane in 0..W {
            first_scores[lane] = if 2 * lane < valid {
                scores[2 * lane]
            } else {
                0
            };
            second_scores[lane] = if 2 * lane + 1 < valid {
                scores[2 * lane + 1]
            } else {
                0
            };
            pair_sums[lane] = first_scores[lane].wrapping_add(second_scores[lane]);
        }
        scan_sums_wrapping(&mut pair_sums);

        let mut first_cumulative = [0i32; W];
        let mut second_cumulative = [0i32; W];
        let mut pair_maxima = [0i32; W];
        for lane in 0..W {
            let before = if lane == 0 { 0 } else { pair_sums[lane - 1] };
            first_cumulative[lane] = prev_score
                .wrapping_add(before)
                .wrapping_add(first_scores[lane]);
            second_cumulative[lane] = prev_score.wrapping_add(pair_sums[lane]);
            pair_maxima[lane] = prev_max
                .max(first_cumulative[lane])
                .max(second_cumulative[lane]);
        }
        scan_scores(&mut pair_maxima);

        let mut first = 2 * W;
        for lane in 0..W {
            let before = if lane == 0 {
                prev_max
            } else {
                pair_maxima[lane - 1]
            };
            let after_first = before.max(first_cumulative[lane]);
            if first == 2 * W && after_first.wrapping_sub(first_cumulative[lane]) > xdrop {
                first = 2 * lane;
            }
            if first == 2 * W && pair_maxima[lane].wrapping_sub(second_cumulative[lane]) > xdrop {
                first = 2 * lane + 1;
            }
        }
        let max = match first {
            0 => prev_max,
            p if p & 1 == 0 => pair_maxima[p / 2 - 1],
            p if p < 2 * W => {
                let lane = p / 2;
                let before = if lane == 0 {
                    prev_max
                } else {
                    pair_maxima[lane - 1]
                };
                before.max(first_cumulative[lane])
            }
            _ => pair_maxima[W - 1],
        };
        (first, max, second_cumulative[W - 1])
    }

    #[cfg(feature = "left-pair-tile")]
    fn paired_left(
        scores: &[i32],
        valid: usize,
        xdrop: i32,
        prev_score: i32,
        prev_max: i32,
    ) -> i32 {
        let (first, max, end_score) = paired_prelude(scores, valid, xdrop, prev_score, prev_max);
        if first < 2 * W || valid < 2 * W {
            max
        } else {
            old_left_from(scores, valid, xdrop, 2 * W, end_score, max)
        }
    }

    #[cfg(feature = "left-pair-tile")]
    #[test]
    fn paired_left_prelude_matches_two_tiles_and_fallback() {
        const N: usize = 192;
        let mut scores = [1i32; N];
        for drop in 0..2 * W {
            scores.fill(1);
            scores[drop] = -8;
            assert_eq!(paired_prelude(&scores, N, 7, 0, 0).0, drop, "drop {drop}");
            assert_eq!(
                paired_left(&scores, N, 7, 0, 0),
                old_left_from(&scores, N, 7, 0, 0, 0),
                "drop {drop}"
            );
        }
        for edge in 1..=2 * W {
            scores.fill(1);
            assert_eq!(
                paired_left(&scores, edge, 7, 0, 0),
                old_left_from(&scores, edge, 7, 0, 0, 0),
                "edge {edge}"
            );
        }

        let mut state = 0x517c_c1b7_2722_0a95u64;
        for case in 0..100_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            for score in &mut scores {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *score = if case & 7 == 0 {
                    state as i32
                } else {
                    (state % 251) as i32 - 150
                };
            }
            let a = state as i32;
            state = state.rotate_left(17).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let b = state as i32;
            let prev_score = a.min(b);
            let prev_max = a.max(b);
            let valid = 1 + state as usize % N;
            let xdrop = (state.rotate_right(11) % 1_001) as i32;
            assert_eq!(
                paired_left(&scores, valid, xdrop, prev_score, prev_max),
                old_left_from(&scores, valid, xdrop, 0, prev_score, prev_max),
                "case {case}, valid {valid}, xdrop {xdrop}"
            );
        }
    }

    /// Deterministic pseudo-random tile shapes covering the M7.1 case list:
    /// maxima inherited from the previous tile, maxima created inside the tile,
    /// ties against the inherited maximum, ties within the tile, first dropping
    /// lane at 0, and no dropping lane at all.
    #[test]
    fn score_only_recovery_matches_carried_position() {
        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut rnd = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let (mut inherited, mut created, mut tie_prev, mut tie_within) = (0, 0, 0, 0);
        let (mut drop_at_zero, mut no_drop) = (0, 0);

        for case in 0..200_000 {
            // Small alphabets make exact ties common, which is the point.
            let span = [2i64, 3, 5, 9][case % 4];
            let pms = (rnd() % 7) as i32;
            let pmp = (rnd() % 5) as i32;
            let xdrop = (rnd() % 6) as i32;
            let base = (rnd() % 3) as i32;
            let mut ts = [0i32; W];
            let mut acc = 0i32;
            for slot in ts.iter_mut() {
                acc += (rnd() % (2 * span as u64 + 1)) as i32 - span as i32;
                *slot = acc;
            }

            let a = carried(&ts, pms, pmp, xdrop, base);
            let b = recovered(&ts, pms, pmp, xdrop, base);
            assert_eq!(
                a, b,
                "case {case}: ts={ts:?} pms={pms} pmp={pmp} xdrop={xdrop}"
            );

            let mut scores: [i32; W] =
                core::array::from_fn(|l| if ts[l] > pms { ts[l] } else { pms });
            scan_scores(&mut scores);
            let first = first_drop(&scores, &ts, xdrop);
            if a.0 == pms {
                inherited += 1
            } else {
                created += 1
            }
            if ts.iter().filter(|&&t| t == pms).count() > 0 {
                tie_prev += 1
            }
            if (0..W).any(|i| (i + 1..W).any(|j| ts[i] == ts[j])) {
                tie_within += 1
            }
            if first == 0 {
                drop_at_zero += 1
            }
            if first == W {
                no_drop += 1
            }
        }

        // The equivalence is only meaningful if the cases actually occurred.
        assert!(
            inherited > 1000,
            "inherited maxima barely exercised: {inherited}"
        );
        assert!(created > 1000, "in-tile maxima barely exercised: {created}");
        assert!(
            tie_prev > 1000,
            "ties with the inherited maximum: {tie_prev}"
        );
        assert!(tie_within > 1000, "ties within a tile: {tie_within}");
        assert!(
            drop_at_zero > 100,
            "first dropping lane = 0: {drop_at_zero}"
        );
        assert!(no_drop > 100, "no dropping lane: {no_drop}");
    }

    /// The tie rule itself, pinned: `shuffle_up` reads the *earlier* lane, so
    /// `>=` keeps the earliest position among equal maxima.
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn ties_resolve_to_the_earliest_lane() {
        let mut v = [(0i32, 0i32); W];
        for l in 0..W {
            v[l] = (5, l as i32);
        }
        scan_pairs(&mut v);
        assert_eq!(
            v[W - 1],
            (5, 0),
            "equal maxima must carry lane 0's position"
        );
    }

    fn scan_sum_width(v: &mut [i32], width: usize) {
        let mut offset = 1usize;
        while offset < width {
            let prev = v.to_vec();
            for l in offset..width {
                v[l] = prev[l].wrapping_add(prev[l - offset]);
            }
            offset <<= 1;
        }
    }

    fn scan_max_width(v: &mut [i32], width: usize) {
        let mut offset = 1usize;
        while offset < width {
            let prev = v.to_vec();
            for l in offset..width {
                if prev[l - offset] >= prev[l] {
                    v[l] = prev[l - offset];
                }
            }
            offset <<= 1;
        }
    }

    /// Current 32-lane tile: one base per lane, wrapping arithmetic.
    fn scalar_tile(
        scores: &[i32],
        valid: usize,
        xdrop: i32,
        prev_score: i32,
        prev_max: i32,
    ) -> (usize, i32, i32) {
        let mut cumulative = [0i32; W];
        let mut acc = prev_score;
        for lane in 0..W {
            acc = acc.wrapping_add(if lane < valid { scores[lane] } else { 0 });
            cumulative[lane] = acc;
        }
        let mut maxima = cumulative.map(|score| score.max(prev_max));
        scan_scores(&mut maxima);
        let first = first_drop(&maxima, &cumulative, xdrop);
        let max = if first == 0 {
            prev_max
        } else {
            maxima[first - 1]
        };
        let done_at = if first < W {
            first
        } else if valid < W {
            valid
        } else {
            W
        };
        (done_at, max, cumulative[W - 1])
    }

    /// 4-base grouped prelude used by the SIMD score gate.
    /// `n_lanes` is 8 (right, 32 bases) or 16 (left, 64 bases).
    #[allow(clippy::needless_range_loop)]
    fn simd4_prelude(
        scores: &[i32],
        valid: usize,
        n_lanes: usize,
        xdrop: i32,
        prev_score: i32,
        prev_max: i32,
    ) -> (usize, i32, i32) {
        let n_bases = n_lanes * 4;
        let mut seg = vec![0i32; n_lanes];
        let mut base = vec![[0i32; 4]; n_lanes];
        for lane in 0..n_lanes {
            for k in 0..4 {
                let idx = lane * 4 + k;
                base[lane][k] = if idx < valid { scores[idx] } else { 0 };
                seg[lane] = seg[lane].wrapping_add(base[lane][k]);
            }
        }
        scan_sum_width(&mut seg, n_lanes);

        let mut c = vec![[0i32; 4]; n_lanes];
        let mut cand = vec![0i32; n_lanes];
        for lane in 0..n_lanes {
            let before = if lane == 0 { 0 } else { seg[lane - 1] };
            let mut acc = prev_score.wrapping_add(before);
            for k in 0..4 {
                acc = acc.wrapping_add(base[lane][k]);
                c[lane][k] = acc;
            }
            cand[lane] = prev_max
                .max(c[lane][0])
                .max(c[lane][1])
                .max(c[lane][2])
                .max(c[lane][3]);
        }
        let mut scanned = cand.clone();
        scan_max_width(&mut scanned, n_lanes);

        let mut first = n_bases;
        let mut after = vec![[0i32; 4]; n_lanes];
        for lane in 0..n_lanes {
            let before_max = if lane == 0 {
                prev_max
            } else {
                scanned[lane - 1]
            };
            after[lane][0] = before_max.max(c[lane][0]);
            after[lane][1] = after[lane][0].max(c[lane][1]);
            after[lane][2] = after[lane][1].max(c[lane][2]);
            after[lane][3] = after[lane][2].max(c[lane][3]);
            for k in 0..4 {
                if first == n_bases && after[lane][k].wrapping_sub(c[lane][k]) > xdrop {
                    first = lane * 4 + k;
                }
            }
        }
        if valid < n_bases && first == n_bases {
            first = valid;
        }
        let f = if first >= n_bases { n_bases } else { first };
        let max = if f == 0 {
            prev_max
        } else {
            let src = if f & 3 == 0 { f / 4 - 1 } else { f / 4 };
            let sub = if f & 3 == 0 { 3 } else { (f & 3) - 1 };
            if f & 3 == 0 {
                scanned[src]
            } else {
                after[src][sub]
            }
        };
        (first.min(n_bases), max, c[n_lanes - 1][3])
    }

    fn scalar_run(scores: &[i32], valid: usize, xdrop: i32, prev_score: i32, prev_max: i32) -> i32 {
        let mut tile = 0usize;
        let mut ps = prev_score;
        let mut pm = prev_max;
        loop {
            let remain = valid.saturating_sub(tile);
            let slice = if tile < scores.len() {
                &scores[tile..]
            } else {
                &[] as &[i32]
            };
            let (done_at, max, end) = scalar_tile(slice, remain.min(W), xdrop, ps, pm);
            if done_at < W || remain <= W {
                return max;
            }
            ps = end;
            pm = max;
            tile += W;
        }
    }

    #[cfg(feature = "simd-prelude")]
    #[test]
    fn simd4_prelude_matches_scalar_tiles() {
        const N: usize = 192;
        let mut scores = [1i32; N];

        for drop in 0..32 {
            scores.fill(1);
            scores[drop] = -8;
            let s = scalar_tile(&scores, N, 7, 0, 0);
            let g = simd4_prelude(&scores, N, 8, 7, 0, 0);
            assert_eq!(g.0, drop, "right drop {drop}");
            assert_eq!(g.1, s.1, "right max at drop {drop}");
        }
        for drop in 0..64 {
            scores.fill(1);
            scores[drop] = -8;
            let s = scalar_run(&scores, N, 7, 0, 0);
            let g = simd4_prelude(&scores, N, 16, 7, 0, 0);
            assert_eq!(g.0, drop, "left drop {drop}");
            assert_eq!(g.1, s, "left max at drop {drop}");
        }
        for edge in 0..=64 {
            scores.fill(1);
            let sr = scalar_tile(&scores, edge.min(32), 7, 0, 0);
            let gr = simd4_prelude(&scores, edge.min(32), 8, 7, 0, 0);
            assert_eq!(gr.1, sr.1, "right edge {edge}");
            let sl = scalar_run(&scores, edge, 7, 0, 0);
            let gl = simd4_prelude(&scores, edge, 16, 7, 0, 0);
            assert_eq!(gl.1, sl, "left edge {edge}");
        }

        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for case in 0..100_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            for score in &mut scores {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *score = if case & 7 == 0 {
                    state as i32
                } else {
                    (state % 251) as i32 - 150
                };
            }
            let a = state as i32;
            state = state.rotate_left(17).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let b = state as i32;
            let prev_score = a.min(b);
            let prev_max = a.max(b);
            let xdrop = (state.rotate_right(11) % 1_001) as i32;
            let right_valid = 1 + (state as usize % 33);
            let left_valid = 1 + (state.rotate_left(5) as usize % 65);

            let sr = scalar_tile(&scores, right_valid.min(32), xdrop, prev_score, prev_max);
            let gr = simd4_prelude(&scores, right_valid.min(32), 8, xdrop, prev_score, prev_max);
            assert_eq!(gr.1, sr.1, "case {case} right max");
            assert_eq!(gr.2, sr.2, "case {case} right end");

            let sl = scalar_run(&scores, left_valid, xdrop, prev_score, prev_max);
            let gl = simd4_prelude(&scores, left_valid, 16, xdrop, prev_score, prev_max);
            if left_valid <= 64 {
                assert_eq!(gl.1, sl, "case {case} left max valid={left_valid}");
            }
        }
    }

    /// One warp of `score_gate`'s SIMD prelude, lane for lane, up to what the
    /// first-drop recovery reads: lanes 0-7 hold the right extension's bases
    /// 0-31 (four per lane), lanes 8-23 the left's 0-63, lanes 24-31 idle with
    /// zero scores. Same wrapping arithmetic, the same `dual_prefix_sum_max`
    /// steps and the same `before_max` shuffle as the kernel.
    struct PreludeWarp {
        active: [bool; W],
        c: [[i32; 4]; W],
        before_max: [i32; W],
        after: [[i32; 3]; W],
        scanned: [i32; W],
    }

    /// `(right_done, left_done, right_max, left_max)`.
    type Recovery = (bool, bool, i32, i32);

    fn prelude_warp(s: &[[i32; 4]; W]) -> PreludeWarp {
        let is_left = |l: usize| (8..24).contains(&l);
        let gl = |l: usize| if is_left(l) { l - 8 } else { l };
        let active: [bool; W] = core::array::from_fn(|l| l < 24);
        let seg: [i32; W] = core::array::from_fn(|l| {
            let [s0, s1, s2, s3] = s[l];
            s0.wrapping_add(s1).wrapping_add(s2).wrapping_add(s3)
        });
        let local_max: [i32; W] = core::array::from_fn(|l| {
            let [s0, s1, s2, s3] = s[l];
            let p1 = s0.wrapping_add(s1);
            let p2 = p1.wrapping_add(s2);
            let p3 = p2.wrapping_add(s3);
            s0.max(p1).max(p2).max(p3).max(0)
        });
        // `dual_prefix_sum_max`: every lane shuffles the pre-step values.
        let (mut sum, mut max) = (seg, local_max);
        let mut offset = 1usize;
        while offset < 16 {
            let (up_s, up_m) = (sum, max);
            for l in 0..W {
                if active[l] && gl(l) >= offset {
                    let cand = up_s[l - offset].wrapping_add(up_m[l]);
                    max[l] = if up_m[l - offset] >= cand {
                        up_m[l - offset]
                    } else {
                        cand
                    };
                    sum[l] = up_s[l - offset].wrapping_add(up_s[l]);
                }
            }
            offset <<= 1;
        }
        let scanned = max;
        let mut c = [[0i32; 4]; W];
        let mut before_max = [0i32; W];
        let mut after = [[0i32; 3]; W];
        for l in 0..W {
            let mut acc = sum[l].wrapping_sub(seg[l]);
            for k in 0..4 {
                acc = acc.wrapping_add(s[l][k]);
                c[l][k] = acc;
            }
            before_max[l] = if gl(l) == 0 || !active[l] {
                0
            } else {
                scanned[l - 1]
            };
            let a0 = before_max[l].max(c[l][0]);
            let a1 = a0.max(c[l][1]);
            after[l] = [a0, a1, a1.max(c[l][2])];
        }
        PreludeWarp {
            active,
            c,
            before_max,
            after,
            scanned,
        }
    }

    /// Drop flags of one lane, base by base, exactly as the kernel tests them.
    fn drop_flags(p: &PreludeWarp, l: usize, xdrop: i32) -> [bool; 4] {
        [
            p.after[l][0].wrapping_sub(p.c[l][0]) > xdrop,
            p.after[l][1].wrapping_sub(p.c[l][1]) > xdrop,
            p.after[l][2].wrapping_sub(p.c[l][2]) > xdrop,
            p.scanned[l].wrapping_sub(p.c[l][3]) > xdrop,
        ]
    }

    /// Default path, part 1: four ballots decoded by `first_of` and the min
    /// cascades into each group's first dropping base `4 * lane + k` (lane
    /// counted within the group), or >= 1020 when the group has no drop.
    fn default_first_drops(p: &PreludeWarp, xdrop: i32) -> (u32, u32) {
        let mut d = [0u32; 4];
        for l in 0..W {
            for (k, dropped) in drop_flags(p, l, xdrop).into_iter().enumerate() {
                d[k] |= u32::from(dropped) << l;
            }
        }
        let first_of = |bits: u32, shift: u32, n_lanes: u32| -> u32 {
            let tz = (bits >> shift).trailing_zeros();
            if tz >= n_lanes { 255 } else { tz }
        };
        let right_first = (4 * first_of(d[0], 0, 8))
            .min(4 * first_of(d[1], 0, 8) + 1)
            .min(4 * first_of(d[2], 0, 8) + 2)
            .min(4 * first_of(d[3], 0, 8) + 3);
        let left_first = (4 * first_of(d[0], 8, 16))
            .min(4 * first_of(d[1], 8, 16) + 1)
            .min(4 * first_of(d[2], 8, 16) + 2)
            .min(4 * first_of(d[3], 8, 16) + 3);
        (right_first, left_first)
    }

    /// Default path, part 2: `recover` shuffles the running maximum before the
    /// first drop (base `n_bases` when there is none) from its lane.
    fn default_recovery(
        p: &PreludeWarp,
        xdrop: i32,
        right_edge: bool,
        left_edge: bool,
    ) -> Recovery {
        let (right_first, left_first) = default_first_drops(p, xdrop);
        let recover = |first: u32, n_bases: u32, origin: u32| -> i32 {
            let f = if first >= n_bases { n_bases } else { first };
            if f == 0 {
                0
            } else {
                let sub = f & 3;
                let src_gl = if sub == 0 { f / 4 - 1 } else { f / 4 };
                let src = (origin + src_gl) as usize;
                if sub == 0 {
                    p.scanned[src]
                } else {
                    p.after[src][sub as usize - 1]
                }
            }
        };
        (
            right_first < 32 || right_edge,
            left_first < 64 || left_edge,
            recover(right_first, 32, 0),
            recover(left_first, 64, 8),
        )
    }

    /// `gate-payload-ballot`: one ballot of "this active lane drops somewhere",
    /// each lane's payload is its maximum before its own first drop, and each
    /// group reads the payload of its first dropping lane (or of its last lane).
    #[allow(clippy::needless_range_loop)]
    fn payload_recovery(
        p: &PreludeWarp,
        xdrop: i32,
        right_edge: bool,
        left_edge: bool,
    ) -> Recovery {
        let mut dropped = 0u32;
        let mut payload = [0i32; W];
        for l in 0..W {
            let [q0, q1, q2, q3] = drop_flags(p, l, xdrop);
            payload[l] = if q0 {
                p.before_max[l]
            } else if q1 {
                p.after[l][0]
            } else if q2 {
                p.after[l][1]
            } else if q3 {
                p.after[l][2]
            } else {
                p.scanned[l]
            };
            dropped |= u32::from(p.active[l] && (q0 || q1 || q2 || q3)) << l;
        }
        let right_src = (dropped | 0x80).trailing_zeros();
        let left_src = 8 + ((dropped >> 8) | 0x8000).trailing_zeros();
        // The same two lanes, written as "first dropping lane of the group,
        // else its last lane".
        let (mr, ml) = (dropped & 0xff, (dropped >> 8) & 0xffff);
        assert_eq!(right_src, if mr != 0 { mr.trailing_zeros() } else { 7 });
        assert_eq!(left_src, if ml != 0 { 8 + ml.trailing_zeros() } else { 23 });
        (
            dropped & 0xff != 0 || right_edge,
            dropped & 0xff_ff00 != 0 || left_edge,
            payload[right_src as usize],
            payload[left_src as usize],
        )
    }

    /// `gate-payload-ballot` against the default first-drop recovery on
    /// random warps: substitution-like scores, ties, base-0 drops, wrapping
    /// totals, xdrop across [0, 1e6] (and beyond, for the idle-lane mask), random
    /// edge flags. Every result the continuation and the keep read must agree.
    #[test]
    fn payload_ballot_matches_first_drop_recovery() {
        const HOXD70: [i32; 9] = [91, -114, -31, -123, 100, -125, -31, -114, -125];
        const RICH: [i32; 5] = [100, 91, 91, -31, -114];
        const CLIFFS: [i32; 4] = [-1000, 5, 0, 100];
        const TIES: [i32; 6] = [0, 0, 1, -1, 100, 100];
        const WARPS: u32 = 400_000;
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        // Per group: first drop at base 0 of lane 0, at k = 0..3 elsewhere, in
        // the group's last lane, and no drop at all.
        let mut seen = [[0u32; 7]; 2];
        for warp in 0..WARPS {
            let mut s = [[0i32; 4]; W];
            for lane in s.iter_mut().take(24) {
                for v in lane.iter_mut() {
                    let r = rnd();
                    *v = match warp % 5 {
                        0 => HOXD70[(r % 9) as usize],
                        1 => RICH[(r % 5) as usize],
                        2 => CLIFFS[(r % 4) as usize],
                        3 => TIES[(r % 6) as usize],
                        // Any magnitude up to the full range, so sums wrap.
                        _ => (r as i32) >> (r >> 59),
                    };
                }
            }
            let r = rnd();
            let xdrop = if warp >= WARPS - 20_000 {
                [-1, -1000, i32::MIN, i32::MAX][(r % 4) as usize]
            } else {
                match r % 8 {
                    0 => 0,
                    1 => 910,
                    2 => 50,
                    3 => 1_000_000,
                    _ => ((r >> 8) % 1_000_001) as i32,
                }
            };
            let right_edge = (r >> 40) & 3 == 0;
            let left_edge = (r >> 42) & 3 == 0;
            let p = prelude_warp(&s);
            let want = default_recovery(&p, xdrop, right_edge, left_edge);
            let got = payload_recovery(&p, xdrop, right_edge, left_edge);
            assert_eq!(
                got, want,
                "warp {warp} xdrop {xdrop} edges {right_edge}/{left_edge}"
            );
            // While no partial sum wraps (every mode but the full-range one), the
            // warp model agrees with the per-group model that
            // `simd4_prelude_matches_scalar_tiles` checks against scalar tiles; a
            // prelude starts from score 0, maximum 0. Past a wrap the kernel's
            // (sum, max) scan and that model's absolute prefix maxima part ways,
            // and `prelude_warp` follows the kernel.
            if warp % 5 != 4 {
                let bases = |lanes: core::ops::Range<usize>| -> Vec<i32> {
                    lanes.flat_map(|l| s[l]).collect()
                };
                let (rf, rmax, _) = simd4_prelude(&bases(0..8), 32, 8, xdrop, 0, 0);
                let (lf, lmax, _) = simd4_prelude(&bases(8..24), 64, 16, xdrop, 0, 0);
                assert_eq!(
                    (rf < 32 || right_edge, lf < 64 || left_edge, rmax, lmax),
                    want,
                    "warp {warp}: warp model vs simd4_prelude"
                );
            }

            let (right_first, left_first) = default_first_drops(&p, xdrop);
            for (g, (first, n_bases)) in [(right_first, 32), (left_first, 64)]
                .into_iter()
                .enumerate()
            {
                let case = if first >= n_bases {
                    6
                } else if first == 0 {
                    0
                } else if first / 4 == n_bases / 4 - 1 {
                    5
                } else {
                    1 + (first & 3) as usize
                };
                seen[g][case] += 1;
            }
        }
        for (g, cases) in seen.iter().enumerate() {
            assert!(
                cases.iter().all(|&n| n > 0),
                "group {g} left a case untested: {cases:?}"
            );
        }
    }
}
