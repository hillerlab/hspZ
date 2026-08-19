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
//! reworked: ballot X-drop termination and deferred entropy in `find_hsps`, the
//! score-only prefix max, peeled entry checks and unchecked bounds, the
//! `align(16)` store and grid 16384, the device-resident count and done-flag
//! scans, the warp-coalesced score gate and dense-anchor compaction, device
//! seed generation (`seed_kmers`/`scatter_seeds`), and `find_hits`'
//! thread-per-seed mapping. Every change stays byte-identical to the oracle at
//! matched plan and `MAX_HITS`.

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

const WARP_SIZE: u32 = 32;
const FULL_MASK: u32 = 0xFFFF_FFFF;
/// `parameters.h: NUC`.
const NUC: usize = 8;

#[cuda_host::cuda_module]
pub mod device {
    use super::{FULL_MASK, NUC, NUM_WARPS, SCAN_BLOCK, SegmentPair, WARP_SIZE};
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
            // Both checks the safe path emits are provably
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

    /// Validate each query window and materialize its k-mer plus the
    /// number of stable output slots it owns. The encoded query uses 0..3 for
    /// uppercase ACGT and >=4 for every byte CPU seeding rejects.
    #[kernel]
    // a kernel launch signature, not a design to refactor
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

    /// Stable scatter. The scanned count is an inclusive prefix, so
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
    ///
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

    /// Runs only the score gate and writes one byte for every hit.
    /// The rare survivors are materialized later from a stable compacted ID
    /// list, so the common reject path writes neither an HSP nor a `u32` flag.
    #[cfg(feature = "dense-anchors")]
    // `ptxas` sizes registers for an unknown block without this, and the gate
    // is 87% of device time on the whole-genome pair. `HSP_THREADS` is the only
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
        // The anchor does not travel through shared memory: the warp fetches
        // its whole chunk coalesced and broadcasts by shuffle.
        static mut SUB: SharedArray<i32, { NUC * NUC }> = SharedArray::UNINIT;

        let thread_id = thread::threadIdx_x();
        let lane_id = thread_id % WARP_SIZE;
        let w = ((thread_id - lane_id) / WARP_SIZE) as usize;
        if (thread_id as usize) < NUC * NUC {
            unsafe { SUB[thread_id as usize] = sub_mat[thread_id as usize] };
        }
        thread::sync_threads();

        // A warp owns CHUNK consecutive anchors so the query side can be reused
        // across the ones that share `query_start`. 32 was measured against 4, 8,
        // 16 and 64: A prefers 32 because its runs are ~20 long and a longer chunk only
        // adds serialisation, while F (runs ~94) still gains at 64. 32 is the value
        // whose worst per-workload loss is ~0.5 points.
        const CHUNK: u32 = 32;

        let stride = NUM_WARPS as u32 * thread::gridDim_x() * CHUNK;
        let mut base = (thread::blockIdx_x() * NUM_WARPS as u32 + w as u32) * CHUNK;
        while base < num_hits {
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
            // read it back, with nothing to overlap the load latency.
            let my_idx = base + lane_id;
            let my_anchor = if my_idx < num_hits {
                anchors[my_idx as usize]
            } else {
                anchors[base as usize]
            };
            let my_lo = my_anchor as u32;
            let my_hi = (my_anchor >> 32) as u32;

            let mut slot = 0u32;
            while slot < CHUNK {
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
                    let is_left = (8..24).contains(&lane_id);
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
                            let b0 = (rw & 0xff) as usize;
                            let b1 = ((rw >> 8) & 0xff) as usize;
                            let b2 = ((rw >> 16) & 0xff) as usize;
                            let b3 = ((rw >> 24) & 0xff) as usize;
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
                            let b0 = ((rw >> 24) & 0xff) as usize;
                            let b1 = ((rw >> 16) & 0xff) as usize;
                            let b2 = ((rw >> 8) & 0xff) as usize;
                            let b3 = (rw & 0xff) as usize;
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
                    // The two end broadcasts feed the continuation loops only, and those
                    // run for ~8% of hits. `right_done`/`left_done` are warp-uniform, so
                    // the shuffles sink into the branches: two dependent shuffles removed
                    // from the 92% that resolve inside the prelude. The gate
                    // stalls on dependencies, so a shuffle skipped is latency
                    // skipped.
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
                    let keep = ((total as f32 as f64) as i32 >= hspthresh) as u8;
                    unsafe { *flags.get_unchecked_mut(hid as usize) = keep };
                }
                slot += 1;
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
            // Where the first X-drop lands, in the first tile of each direction
            // — the class that owns ~91% of right terminations.
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
            // lane on every tile. Shared-state loads are ~13% of per-tile
            // instructions and the kernel runs at ~73% of issue peak, so these
            // are the cheapest instructions to remove — and unlike a scan, this
            // needs no commit restructuring: the value never changes.
            let hit_ref_loc = unsafe { REF_LOC[w] };
            let hit_query_loc = unsafe { QUERY_LOC[w] };

            // Score both extensions with the production warp mapping, but keep
            // only the state needed for X-drop and the maximum score.
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
                // The packed form doubles as a workaround: the four-accumulator,
                // four-scan version produced PTX that `ptxas` accepted but the
                // pinned ZLUDA backend refused to load with `DriverError(500)`,
                // bisected to that construct — not the loop, not the indexing.
                // It is retained on its own merits; if native NVIDIA measurement
                // ever shows the unpacked form wins there, that is when
                // specialization gets earned.
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
                // The four-counter/four-scan alternative was implemented and
                // measured on an NVIDIA L4: `find_hsps` +1.90% (A), +1.36% (B),
                // +0.50% (apple) — worse on every workload. The codegen census
                // said why: 42 shuffles vs 22, 9/6 local stores/loads vs 4/1,
                // and 53 more 32-bit registers, because `c[r]` with a runtime
                // index spills. Packed wins on native CUDA on its own merits.
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
                    // lane index, mirrored from the device scan
                    #[allow(clippy::needless_range_loop)]
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
    /// Removing 9 of 22 prelude shuffles changed nothing (+0.00%), while adding
    /// 8 *on the dependency chain* cost 3% — depth is the thing to cut. Same
    /// shuffle count here, half the dependent depth.
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
    /// libdevice's `__nv_log` is not reachable from cuda-oxide, so
    /// this is an atanh series over the reduced mantissa — relative error under
    /// 1e-15, which moves a truncated HSP score by ~1e-11 at most. Swap it for
    /// the real intrinsic if cuda-oxide ever exposes libdevice.
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
    // step distance in the host reimplementation of the warp scan
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
    fn simd4_prelude(
        scores: &[i32],
        valid: usize,
        n_lanes: usize,
        xdrop: i32,
        prev_score: i32,
        prev_max: i32,
    ) -> (usize, i32, i32) {
        // lane index, mirrored from the device scan
        #[allow(clippy::needless_range_loop)]
        let n_bases = n_lanes * 4;
        let mut seg = vec![0i32; n_lanes];
        let mut base = vec![[0i32; 4]; n_lanes];
        for lane in 0..n_lanes {
            // lane index, mirrored from the device scan
            #[allow(clippy::needless_range_loop)]
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
}
