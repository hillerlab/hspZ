// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung
// Distributed under the terms of the GNU General Public License, Version 3.0.

// Author : Alejandro Gonzales-Irribarren
// Github : alejandrogzi
// Email  : alejandrxgzi@gmail.com

//! Chromosome-aware work planning.
//!
//! Removes the assumption "one input genome = one GPU block" by grouping whole
//! records into bins and pairing reference bins with query bins. Records stay
//! atomic — a bin never cuts a chromosome — so every HSP remains
//! record-relative and no coordinate stitching is needed.
//!
//! Reference and query targets are independent: `--seq-block-size` sets the
//! reference layout and `--query-block-size` (defaulting to it) the query
//! layout, so a reference geometry can be chosen without paying for a query
//! split. `--kegalign-bins` switches binning to KegAlign's
//! sequential fill for matched-granularity benchmarking, and refuses to shrink
//! below it because that would unmatch the granularity.
//!
//! The planner sees only metadata (id, name, length, original order); bases
//! stay in the `Genome`. Serial and multi-worker execution consume the same
//! [`WorkUnit`]s: `assign_bins` distributes reference bins across `--gpus`
//! workers by deterministic LPT, and `plan_within_budget` halves the target
//! until the largest unit fits device memory. GPU and host preflights
//! (`plan_within_budget` / `host_preflight`) run before any CUDA allocation.
//!
//! Block layout sets the `MAX_HITS` chunk boundaries and therefore the dedup
//! scope: a different layout produces a legitimately different HSP set, so
//! each layout carries its own frozen digest.

/// One record's metadata, as the planner needs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordMeta {
    pub id: u32,
    pub name: String,
    pub len: u64,
    /// Position in the input file, which is the tie-break that keeps binning
    /// deterministic and the order records are packed within a bin.
    pub ordinal: u32,
}

/// A group of whole records executed together as one GPU block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bin {
    pub id: u32,
    /// Record ids, always in original input order.
    pub record_ids: Vec<u32>,
    pub total_bp: u64,
}

/// One reference-bin × query-bin pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkUnit {
    /// Stable logical position, independent of execution order. Output ordering,
    /// `-D` threshold history and `-Z` entry order all follow this so results do
    /// not depend on GPU completion order.
    pub ordinal: u32,
    pub reference_bin: u32,
    pub query_bin: u32,
}

/// A complete plan: how both genomes are binned, and the pairs to run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub reference_bins: Vec<Bin>,
    pub query_bins: Vec<Bin>,
    pub units: Vec<WorkUnit>,
}

/// Assigns reference bins to workers, deterministically.
///
/// Every query bin runs against every reference bin, so
/// `cost(R) = reference_bp x total_query_bp` orders bins exactly as `total_bp`
/// does — LPT on `total_bp` is the same schedule with less arithmetic. Longest bin
/// first into the currently lightest worker, ties on bin id, and each worker's list
/// is returned in bin order so it executes its share in ordinal order.
pub fn assign_bins(bins: &[Bin], workers: usize) -> Vec<Vec<u32>> {
    let workers = workers.max(1);
    let mut order: Vec<&Bin> = bins.iter().collect();
    order.sort_by(|a, b| b.total_bp.cmp(&a.total_bp).then(a.id.cmp(&b.id)));
    let mut load = vec![0u64; workers];
    let mut out = vec![Vec::new(); workers];
    for b in order {
        // `min_by_key` keeps the first minimum, so equal loads go to the lowest
        // worker index — the tie-break that makes this reproducible.
        let w = (0..workers).min_by_key(|&i| load[i]).unwrap();
        load[w] += b.total_bp;
        out[w].push(b.id);
    }
    for ids in &mut out {
        ids.sort_unstable();
    }
    out
}

/// KegAlign's block rule, reproduced exactly, for matched-granularity runs.
///
/// `main.cpp` fills blocks in *input order* and closes one as soon as the
/// accumulated length exceeds `seq_block_size` — so a block overshoots the target
/// by up to one record, and membership depends on input order rather than on any
/// balancing. That is a different rule from [`bin_records`]' LPT, and comparing HSP
/// algorithms requires the same membership on both sides: block layout decides the
/// `MAX_HITS` chunking and the dedup scope, so unmatched blocks make the outputs
/// legitimately differ before either kernel runs.
pub fn bin_records_sequential(records: &[RecordMeta], target_bp: u64) -> Vec<Bin> {
    let mut bins: Vec<Bin> = Vec::new();
    let mut cur: Vec<u32> = Vec::new();
    let mut cur_bp = 0u64;
    let mut order: Vec<&RecordMeta> = records.iter().collect();
    order.sort_by_key(|r| r.ordinal);
    for r in order {
        cur.push(r.id);
        cur_bp += r.len;
        // `>`, not `>=`: KegAlign closes the block only once it is *over* target.
        if cur_bp > target_bp {
            bins.push(Bin {
                id: bins.len() as u32,
                record_ids: std::mem::take(&mut cur),
                total_bp: cur_bp,
            });
            cur_bp = 0;
        }
    }
    if !cur.is_empty() {
        bins.push(Bin {
            id: bins.len() as u32,
            record_ids: cur,
            total_bp: cur_bp,
        });
    }
    bins
}

/// Groups records into bins of roughly `target_bp`, keeping records atomic.
///
/// `--seq-block-size` is the *target*, not a ceiling: a 249 Mbp chr1 stays one
/// 249 Mbp bin against a 200 Mbp target rather than being split. Longest record
/// first into the currently smallest bin, which is the standard LPT
/// heuristic and is deterministic given the ordinal tie-break.
pub fn bin_records(records: &[RecordMeta], target_bp: u64) -> Vec<Bin> {
    if records.is_empty() {
        return Vec::new();
    }
    let total: u64 = records.iter().map(|r| r.len).sum();
    let target = target_bp.max(1);
    let n_bins = (total.div_ceil(target) as usize).clamp(1, records.len());

    // Longest first; equal lengths fall back to input order so the result does
    // not depend on the sort's stability.
    let mut order: Vec<&RecordMeta> = records.iter().collect();
    order.sort_by(|a, b| b.len.cmp(&a.len).then(a.ordinal.cmp(&b.ordinal)));

    let mut loads: Vec<(u64, Vec<u32>)> = vec![(0, Vec::new()); n_bins];
    for r in order {
        // Smallest current load wins; index breaks ties so this is reproducible.
        let (i, _) = loads
            .iter()
            .enumerate()
            .min_by_key(|(i, (bp, _))| (*bp, *i))
            .expect("n_bins >= 1");
        loads[i].0 += r.len;
        loads[i].1.push(r.id);
    }

    // Normalise: drop empties, restore input order inside each bin, then order
    // bins by their first record so bin ids follow the genome.
    let by_ordinal: std::collections::HashMap<u32, u32> =
        records.iter().map(|r| (r.id, r.ordinal)).collect();
    let mut bins: Vec<(u64, Vec<u32>)> = loads.into_iter().filter(|(_, v)| !v.is_empty()).collect();
    for (_, ids) in bins.iter_mut() {
        ids.sort_by_key(|id| by_ordinal[id]);
    }
    bins.sort_by_key(|(_, ids)| by_ordinal[&ids[0]]);
    bins.into_iter()
        .enumerate()
        .map(|(i, (total_bp, record_ids))| Bin {
            id: i as u32,
            record_ids,
            total_bp,
        })
        .collect()
}

/// Builds the plan: bin both sides, then pair every reference bin with every
/// query bin.
///
/// Reference bin outermost in the ordinal sequence, because the executor builds
/// one `SeedTable` per reference bin and reuses it across all query bins.
/// Ordinals therefore run R0×Q0, R0×Q1, …, R1×Q0, … which is exactly the order
/// serial execution wants and the order MPS must commit results in.
// the single-target entry point: used by the tests below, and it documents the
// common case where reference and query share one block target
#[allow(dead_code)]
pub fn plan(reference: &[RecordMeta], query: &[RecordMeta], target_bp: u64) -> Plan {
    plan_with(reference, query, target_bp, target_bp, false)
}

/// [`plan`], with KegAlign's sequential block fill instead of LPT when
/// `kegalign_bins` is set (Mode A).
/// The two sides take independent targets. Splitting the reference is what
/// balances workers; splitting the query only multiplies `R x Q` work units, and every one
/// of those carries a query swap and its own `MAX_HITS` chunk boundary on *every* worker.
/// Before this, one flag did both, so choosing a reference layout forced a query layout.
/// `query_target_bp == target_bp` reproduces the old plan exactly.
pub fn plan_with(
    reference: &[RecordMeta],
    query: &[RecordMeta],
    target_bp: u64,
    query_target_bp: u64,
    kegalign_bins: bool,
) -> Plan {
    let bin = if kegalign_bins {
        bin_records_sequential
    } else {
        bin_records
    };
    let reference_bins = bin(reference, target_bp);
    let query_bins = bin(query, query_target_bp);
    let mut units = Vec::with_capacity(reference_bins.len() * query_bins.len());
    let mut ordinal = 0u32;
    for r in &reference_bins {
        for q in &query_bins {
            units.push(WorkUnit {
                ordinal,
                reference_bin: r.id,
                query_bin: q.id,
            });
            ordinal += 1;
        }
    }
    Plan {
        reference_bins,
        query_bins,
        units,
    }
}

/// Device bytes one work unit needs, from hspZ's actual allocations.
///
/// Deliberately built from the real formulas rather than a bytes/bp constant, so
/// the preflight tracks the code: `pos_table` is one `u32` per indexed reference
/// position, `index_table` is `4^kmer_size` entries, the three sequence buffers
/// are one byte per base (reference, query, query reverse complement), and the
/// per-batch hit buffers are bounded by `max_hits`.
pub fn unit_device_bytes(
    ref_bp: u64,
    query_bp: u64,
    kmer_size: usize,
    step: u32,
    max_hits: u32,
) -> u64 {
    let indexed = ref_bp / step.max(1) as u64;
    let pos_table = indexed * 4;
    let index_table = (1u64 << (2 * kmer_size)) * 4;
    let sequences = ref_bp + 2 * query_bp;
    // Per batch: one SegmentPair (16 B) and one done flag (4 B) per hit, plus
    // the compacted anchors. These are the grown-once buffers in `Engine`.
    let hits = max_hits as u64 * (16 + 4);
    pos_table + index_table + sequences + hits
}

/// Raises the bin count until the largest planned unit fits, or reports the
/// record that cannot fit at all.
///
/// Returns the accepted plan and the largest unit estimate. A single record that
/// exceeds capacity is a hard error: v1 does not split records, and discovering
/// it through a CUDA OOM mid-run is exactly what this avoids.
// the planner needs the whole device budget; a struct here would only move the list
#[allow(clippy::too_many_arguments)]
pub fn plan_within_budget(
    reference: &[RecordMeta],
    query: &[RecordMeta],
    target_bp: u64,
    query_target_bp: u64,
    budget_bytes: u64,
    kmer_size: usize,
    step: u32,
    max_hits: u32,
    kegalign_bins: bool,
) -> Result<(Plan, u64), String> {
    let worst_of = |p: &Plan| {
        p.units
            .iter()
            .map(|u| {
                let r = p.reference_bins[u.reference_bin as usize].total_bp;
                let q = p.query_bins[u.query_bin as usize].total_bp;
                unit_device_bytes(r, q, kmer_size, step, max_hits)
            })
            .max()
            .unwrap_or(0)
    };
    // Mode A: the block layout must equal KegAlign's, so shrinking the target to
    // fit is not an option — it would silently unmatch the granularity the whole
    // comparison rests on. Fit at the requested size or say why not.
    if kegalign_bins {
        let p = plan_with(
            reference,
            query,
            target_bp.max(1),
            query_target_bp.max(1),
            true,
        );
        let worst = worst_of(&p);
        if worst <= budget_bytes {
            return Ok((p, worst));
        }
        return Err(format!(
            "matched-granularity blocks of {} bp need {:.1} GB per work unit, only {:.1} GB \
             free: pick a block size both tools can run, do not let the planner shrink it",
            target_bp,
            worst as f64 / 1e9,
            budget_bytes as f64 / 1e9
        ));
    }
    let mut target = target_bp.max(1);
    let mut qtarget = query_target_bp.max(1);
    for _ in 0..24 {
        let p = plan_with(reference, query, target, qtarget, false);
        let worst = p
            .units
            .iter()
            .map(|u| {
                let r = p.reference_bins[u.reference_bin as usize].total_bp;
                let q = p.query_bins[u.query_bin as usize].total_bp;
                unit_device_bytes(r, q, kmer_size, step, max_hits)
            })
            .max()
            .unwrap_or(0);
        if worst <= budget_bytes {
            return Ok((p, worst));
        }
        // A bin holding one atomic record cannot be made smaller.
        let stuck = p.reference_bins.iter().any(|b| {
            b.record_ids.len() == 1
                && p.query_bins.iter().any(|q| {
                    unit_device_bytes(b.total_bp, q.total_bp, kmer_size, step, max_hits)
                        > budget_bytes
                })
                && q_is_single(&p, budget_bytes, kmer_size, step, max_hits, b.total_bp)
        });
        if stuck {
            let biggest = reference
                .iter()
                .chain(query.iter())
                .max_by_key(|r| r.len)
                .map(|r| format!("{} ({} bp)", r.name, r.len))
                .unwrap_or_else(|| "<none>".into());
            return Err(format!(
                "record {biggest} exceeds GPU capacity ({budget_bytes} bytes available); \
                 intra-record splitting is not supported in v1"
            ));
        }
        // Both sides halve, so an explicit `--query-block-size` keeps its ratio to the
        // reference target while the plan shrinks into device memory.
        target /= 2;
        qtarget = (qtarget / 2).max(1);
    }
    Err("could not find a bin size that fits GPU memory".into())
}

/// True when every offending query bin is already a single record, i.e. halving
/// the target cannot help either side.
fn q_is_single(
    p: &Plan,
    budget: u64,
    kmer_size: usize,
    step: u32,
    max_hits: u32,
    ref_bp: u64,
) -> bool {
    p.query_bins
        .iter()
        .filter(|q| unit_device_bytes(ref_bp, q.total_bp, kmer_size, step, max_hits) > budget)
        .all(|q| q.record_ids.len() == 1)
}

/// Host-memory estimate for a multi-worker run.
///
/// Conservative: sums the dominant host-resident allocations rather than
/// modelling their exact overlap. Exact formulas only (capacity × size_of), no
/// `bp × magic_constant`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostEstimate {
    /// Raw input records, held once by the caller and shared by every worker.
    pub shared: u64,
    /// Per-worker peak with reference prefetch: the next bin's build overlaps
    /// the current query bin + pinned seed slots.
    pub per_worker_prefetch: u64,
    /// Per-worker peak without prefetch: the build is inline, so it does not
    /// overlap the query phase.
    pub per_worker_no_prefetch: u64,
}

pub fn host_estimate(
    plan: &Plan,
    ref_bp_total: u64,
    qry_bp_total: u64,
    kmer_size: usize,
    step: u32,
    threads: usize,
    max_seeds: usize,
) -> HostEstimate {
    let index = (1u64 << (2 * kmer_size)) * 4; // index_table, 4 B/entry
    let largest_ref = plan
        .reference_bins
        .iter()
        .map(|b| b.total_bp)
        .max()
        .unwrap_or(0);
    let largest_qry = plan
        .query_bins
        .iter()
        .map(|b| b.total_bp)
        .max()
        .unwrap_or(0);
    let pos = largest_ref / step.max(1) as u64 * 4; // pos_table, 4 B/indexed bp
    let counts = threads as u64 * index; // build_parallel pass-1 transient
    let packed_ref = largest_ref * 2; // buf + enc (no rc for a reference bin)
    let query = largest_qry * 4; // buf + rc + enc + enc_rc
    let pinned = 2 * max_seeds as u64 * 8; // two u64 seed slots

    HostEstimate {
        shared: ref_bp_total + qry_bp_total,
        per_worker_prefetch: packed_ref + counts + index + pos + query + pinned,
        per_worker_no_prefetch: (packed_ref + counts + index + pos).max(query + pinned),
    }
}

/// Chooses whether reference prefetch is safe for `workers` workers against
/// `available` host bytes. Pure, so it is unit-testable without a
/// GPU. `Ok(true)` keeps prefetch, `Ok(false)` disables it, `Err` means even the
/// no-prefetch shape cannot fit.
pub fn host_preflight(
    est: &HostEstimate,
    assignment: &[Vec<u32>],
    available: u64,
) -> Result<bool, String> {
    let with = host_peak(est, assignment, true);
    if with <= available {
        return Ok(true);
    }
    let without = host_peak(est, assignment, false);
    if without <= available {
        return Ok(false);
    }
    Err(format!(
        "host memory: {} worker(s) need ~{:.1} GB (prefetch) or ~{:.1} GB (no prefetch), \
         only ~{:.1} GB available",
        assignment.len(),
        with as f64 / 1e9,
        without as f64 / 1e9,
        available as f64 / 1e9
    ))
}

/// Estimated host peak for a concrete assignment.
///
/// A worker that owns a single reference bin has nothing to prefetch, so it costs
/// the no-prefetch shape whatever the flag says. Charging every worker the prefetch
/// shape overestimated a 4-worker multi5 run by 52% against measured RSS;
/// assignment-aware it is +19%, still conservative but usefully so.
pub fn host_peak(est: &HostEstimate, assignment: &[Vec<u32>], prefetch: bool) -> u64 {
    est.shared
        + assignment
            .iter()
            .map(|bins| {
                if prefetch && bins.len() > 1 {
                    est.per_worker_prefetch
                } else {
                    est.per_worker_no_prefetch
                }
            })
            .sum::<u64>()
}

/// A bin turned into exactly what the engine consumes.
///
/// Mirrors what `Prepared` holds for a whole genome: raw bytes for seeding and
/// encoded bytes for the GPU, and for a query side both strands of each. Built
/// only through [`sequence::pack`] / [`sequence::reverse_complement`], so a bin
/// is byte-identical to the same records loaded as an entire input.
pub struct PackedBin {
    /// Packed bases, records joined by `SEP` with a trailing separator.
    pub buf: Vec<u8>,
    /// Bin-local chromosome table; names are the original record names, so every
    /// emitted coordinate stays chromosome-relative.
    pub chrs: Vec<crate::sequence::Chr>,
    /// Bases excluding the trailing separator.
    pub block_len: usize,
    /// Reverse complement of `buf`, built from the *bin* — never sliced
    /// out of a whole-genome reverse complement.
    pub rc: Vec<u8>,
    pub rc_chrs: Vec<crate::sequence::Chr>,
    /// Device alphabet forms of `buf` and `rc`.
    pub enc: Vec<u8>,
    pub enc_rc: Vec<u8>,
}

impl PackedBin {
    /// Materializes one bin. `records` supplies `(name, bases)` for every record
    /// id the bin holds, in the bin's own (input) order.
    ///
    /// `want_rc` skips the reverse complement for reference bins, which never
    /// need one — that is half the packing work and, on a 249 Mbp bin, ~500 MB.
    pub fn build<'a>(
        records: impl IntoIterator<Item = (&'a str, &'a [u8])>,
        prefix: &str,
        want_rc: bool,
    ) -> Self {
        let (buf, chrs, block_len) = crate::sequence::pack(records, prefix);
        let enc = crate::sequence::encode(&buf[..block_len.min(buf.len())]);
        let (rc, rc_chrs) = if want_rc {
            crate::sequence::reverse_complement(&buf, &chrs, block_len)
        } else {
            (Vec::new(), Vec::new())
        };
        let enc_rc = if want_rc {
            crate::sequence::encode(&rc)
        } else {
            Vec::new()
        };
        PackedBin {
            buf,
            chrs,
            block_len,
            rc,
            rc_chrs,
            enc,
            enc_rc,
        }
    }
}

#[cfg(test)]
mod tests {
    /// Mode A: hspZ must reproduce KegAlign's block membership exactly, or the two
    /// tools dedup in different scopes and their outputs differ before any kernel
    /// runs. The rule is sequential fill in input order, closing a block once it is
    /// *over* target — so blocks overshoot and the last one may be short.
    #[test]
    fn sequential_bins_match_kegalign_block_fill() {
        use super::{RecordMeta, bin_records_sequential};
        let rec = |id: u32, len: u64| RecordMeta {
            id,
            name: format!("chr{id}"),
            len,
            ordinal: id,
        };
        // 100, 60, 50, 300, 10 against a target of 120:
        //   100+60 = 160 > 120  -> block 0 = {0,1}
        //   50 <= 120, +300 = 350 > 120 -> block 1 = {2,3}
        //   10 left over        -> block 2 = {4}
        let recs = vec![rec(0, 100), rec(1, 60), rec(2, 50), rec(3, 300), rec(4, 10)];
        let bins = bin_records_sequential(&recs, 120);
        let ids: Vec<Vec<u32>> = bins.iter().map(|b| b.record_ids.clone()).collect();
        assert_eq!(ids, vec![vec![0, 1], vec![2, 3], vec![4]]);
        assert_eq!(
            bins.iter().map(|b| b.total_bp).collect::<Vec<_>>(),
            vec![160, 350, 10]
        );
        // Input order decides membership: unlike LPT, sorting by length must not
        // change anything.
        assert_eq!(bin_records_sequential(&recs, 120), bins);
        // One record longer than the target is its own block, never split.
        assert_eq!(bin_records_sequential(&[rec(0, 500)], 120).len(), 1);
    }

    /// The assignment must be balanced, deterministic, and a partition — no bin
    /// run twice (duplicate output) and none dropped (missing alignments).
    /// `--query-block-size` is a *query-only* lever: the reference layout and
    /// therefore `assign_bins` must not move when it changes, and an unset flag
    /// (which the CLI turns into the reference target) must reproduce the old
    /// plan exactly.
    #[test]
    fn query_block_size_changes_only_the_query_side() {
        use super::{plan, plan_with};
        let mk = |lens: &[u64]| -> Vec<RecordMeta> {
            lens.iter()
                .enumerate()
                .map(|(i, &len)| RecordMeta {
                    id: i as u32,
                    name: format!("c{i}"),
                    len,
                    ordinal: i as u32,
                })
                .collect()
        };
        let r = mk(&[250, 240, 200, 190, 180, 170, 160]);
        let q = mk(&[195, 180, 160, 155, 150, 145, 60]);

        // Equal targets reproduce the single-target plan bit for bit.
        let base = plan(&r, &q, 400);
        assert_eq!(
            plan_with(&r, &q, 400, 400, false),
            base,
            "equal targets must be identical"
        );

        // A coarser query target leaves the reference bins and units-per-reference-bin
        // structure alone, and only collapses query bins.
        let coarse = plan_with(&r, &q, 400, 10_000, false);
        assert_eq!(
            coarse.reference_bins, base.reference_bins,
            "reference layout must not move"
        );
        assert_eq!(
            coarse.query_bins.len(),
            1,
            "one query bin at a target above the total"
        );
        assert!(
            coarse.query_bins.len() < base.query_bins.len(),
            "query bins must collapse"
        );
        assert_eq!(
            coarse.units.len(),
            coarse.reference_bins.len(),
            "units = R x 1"
        );
        // Ordinals stay dense and ascending, so output ordering is still well defined.
        for (i, u) in coarse.units.iter().enumerate() {
            assert_eq!(u.ordinal, i as u32);
        }
        // Every record still appears exactly once on each side.
        for (bins, n) in [
            (&coarse.reference_bins, r.len()),
            (&coarse.query_bins, q.len()),
        ] {
            let mut ids: Vec<u32> = bins.iter().flat_map(|b| b.record_ids.clone()).collect();
            ids.sort_unstable();
            assert_eq!(ids, (0..n as u32).collect::<Vec<_>>());
        }
        // And the worker assignment is untouched, which is the whole point.
        assert_eq!(
            super::assign_bins(&coarse.reference_bins, 4),
            super::assign_bins(&base.reference_bins, 4)
        );
    }

    #[test]
    fn assign_bins_is_a_balanced_deterministic_partition() {
        use super::{Bin, assign_bins};
        let bins: Vec<Bin> = [100u64, 90, 80, 70, 10]
            .iter()
            .enumerate()
            .map(|(i, &bp)| Bin {
                id: i as u32,
                record_ids: vec![i as u32],
                total_bp: bp,
            })
            .collect();

        for workers in [1usize, 2, 3, 8] {
            let a = assign_bins(&bins, workers);
            assert_eq!(
                a,
                assign_bins(&bins, workers),
                "assignment must be deterministic"
            );
            let mut all: Vec<u32> = a.iter().flatten().copied().collect();
            all.sort_unstable();
            assert_eq!(
                all,
                vec![0, 1, 2, 3, 4],
                "every bin exactly once ({workers} workers)"
            );
            for ids in &a {
                let mut sorted = ids.clone();
                sorted.sort_unstable();
                assert_eq!(*ids, sorted, "a worker runs its bins in ordinal order");
            }
        }

        // LPT on 100,90,80,70,10: 100->w0, 90->w1, 80->w1 (90<100), 70->w0,
        // 10->w0 (tie goes to the lower index). So 180 vs 170 — within one bin of
        // perfect, which is the point of longest-first.
        let two = assign_bins(&bins, 2);
        let load = |ids: &Vec<u32>| ids.iter().map(|&i| bins[i as usize].total_bp).sum::<u64>();
        assert_eq!((load(&two[0]), load(&two[1])), (180, 170));
        assert_eq!((&two[0], &two[1]), (&vec![0, 3, 4], &vec![1, 2]));
        // More workers than bins: the extra ones get nothing, and nothing is lost.
        assert_eq!(
            assign_bins(&bins, 8)
                .iter()
                .filter(|v| v.is_empty())
                .count(),
            3
        );
    }

    use super::*;

    /// The prefetch fallback — keep prefetch when it fits,
    /// disable it when only the no-prefetch shape fits, hard-error when neither
    /// does. Shared bytes are counted once, per-worker bytes times the worker
    /// count.
    #[test]
    fn host_preflight_keeps_prefetch_then_falls_back_then_errors() {
        use super::{HostEstimate, host_peak, host_preflight};
        // shared 100, per-worker prefetch 300, no-prefetch 150.
        let est = HostEstimate {
            shared: 100,
            per_worker_prefetch: 300,
            per_worker_no_prefetch: 150,
        };
        let multi = |w: usize| vec![vec![0u32, 1]; w]; // every worker owns 2 bins
        // 1 worker: 100 + 300 = 400 fits → prefetch kept.
        assert_eq!(host_preflight(&est, &multi(1), 400), Ok(true));
        // 4 workers: 100 + 4*300 = 1300 fails, 100 + 4*150 = 700 fits → disabled.
        assert_eq!(host_preflight(&est, &multi(4), 1000), Ok(false));
        // 4 workers: even 700 fails → hard error.
        assert!(host_preflight(&est, &multi(4), 500).is_err());

        // A worker owning ONE bin has nothing to prefetch, so it costs the
        // no-prefetch shape even when prefetch is on. Four such workers are
        // 100 + 4*150 = 700, not 1300 — the difference between a spurious fallback
        // (or a spurious hard error) and running.
        let one_each: Vec<Vec<u32>> = (0..4).map(|i| vec![i]).collect();
        assert_eq!(host_peak(&est, &one_each, true), 700);
        assert_eq!(host_peak(&est, &one_each, false), 700);
        assert_eq!(host_preflight(&est, &one_each, 700), Ok(true));
        // Mixed 2/1/1: only the first worker prefetches.
        let mixed = vec![vec![0u32, 1], vec![2], vec![3]];
        assert_eq!(host_peak(&est, &mixed, true), 100 + 300 + 150 + 150);
        // Shared counted once, never per worker.
        assert_eq!(
            host_peak(&est, &multi(2), true) - host_peak(&est, &multi(1), true),
            300
        );
    }

    /// Packing a bin must equal packing those same records as an entire input,
    /// byte for byte. With the shared packer this is a regression test rather
    /// than a two-implementation equivalence proof.
    #[test]
    fn packing_a_bin_equals_packing_those_records_as_a_whole_input() {
        let cases: Vec<Vec<(&str, &[u8])>> = vec![
            vec![("chrA", b"ACGTACGTAC".as_slice())],
            vec![
                ("chrA", b"ACGT".as_slice()),
                ("chrB", b"TTTTGGGG".as_slice()),
            ],
            vec![
                ("chrA", b"ACGTN".as_slice()),
                ("chrB", b"acgtACGT".as_slice()),
                ("chrC", b"NNNN".as_slice()),
            ],
            // Record shorter than a seed window, and an empty record.
            vec![
                ("tiny", b"AC".as_slice()),
                ("empty", b"".as_slice()),
                ("chrZ", b"GGGG".as_slice()),
            ],
            // Ns and soft masking right at the boundaries.
            vec![("a", b"NNACGTnn".as_slice()), ("b", b"nnACGTNN".as_slice())],
        ];
        for recs in cases {
            let (want_buf, want_chrs, want_len) = crate::sequence::pack(recs.clone(), "");
            let bin = PackedBin::build(recs.clone(), "", true);
            assert_eq!(bin.buf, want_buf, "packed bytes differ for {recs:?}");
            assert_eq!(bin.block_len, want_len, "block_len differs for {recs:?}");
            assert_eq!(bin.chrs.len(), want_chrs.len());
            for (a, b) in bin.chrs.iter().zip(&want_chrs) {
                assert_eq!((a.start, a.len, &a.name), (b.start, b.len, &b.name));
            }
            // The reverse complement must come from the bin and agree with
            // the shared helper on the same packed block.
            let (want_rc, want_rc_chrs) =
                crate::sequence::reverse_complement(&want_buf, &want_chrs, want_len);
            assert_eq!(
                bin.rc, want_rc,
                "bin reverse complement differs for {recs:?}"
            );
            assert_eq!(bin.rc_chrs.len(), want_rc_chrs.len());
            for (a, b) in bin.rc_chrs.iter().zip(&want_rc_chrs) {
                assert_eq!((a.start, a.len, &a.name), (b.start, b.len, &b.name));
            }
        }
    }

    #[test]
    fn a_reference_bin_skips_the_reverse_complement() {
        let recs: Vec<(&str, &[u8])> = vec![("chrA", b"ACGTACGT".as_slice())];
        let r = PackedBin::build(recs.clone(), "", false);
        assert!(
            r.rc.is_empty() && r.enc_rc.is_empty(),
            "reference bins never need an RC"
        );
        let q = PackedBin::build(recs, "", true);
        assert!(!q.rc.is_empty() && !q.enc_rc.is_empty());
    }

    fn recs(lens: &[u64]) -> Vec<RecordMeta> {
        lens.iter()
            .enumerate()
            .map(|(i, &len)| RecordMeta {
                id: i as u32,
                name: format!("chr{}", i + 1),
                len,
                ordinal: i as u32,
            })
            .collect()
    }

    #[test]
    fn a_record_larger_than_the_target_stays_atomic() {
        // chr1 is 249 Mbp against a 200 Mbp target: it must remain one bin, not
        // be split to satisfy the target.
        let r = recs(&[249_000_000]);
        let bins = bin_records(&r, 200_000_000);
        assert_eq!(bins.len(), 1);
        assert_eq!(bins[0].total_bp, 249_000_000);
        assert_eq!(bins[0].record_ids, vec![0]);
    }

    #[test]
    fn lpt_balances_and_is_deterministic() {
        let r = recs(&[100, 90, 80, 70, 60, 50]);
        let a = bin_records(&r, 150);
        let b = bin_records(&r, 150);
        assert_eq!(a, b, "planning must be reproducible");
        let total: u64 = a.iter().map(|x| x.total_bp).sum();
        assert_eq!(total, 450);
        // 450/150 = 3 bins; LPT on (100,90,80,70,60,50) gives 150/150/150.
        assert_eq!(a.len(), 3);
        for bin in &a {
            assert_eq!(bin.total_bp, 150, "{a:?}");
        }
    }

    #[test]
    fn records_keep_input_order_inside_a_bin() {
        // Long-first assignment would otherwise leave descending order.
        let r = recs(&[10, 100, 20]);
        let bins = bin_records(&r, 1_000);
        assert_eq!(bins.len(), 1);
        assert_eq!(bins[0].record_ids, vec![0, 1, 2], "input order preserved");
    }

    #[test]
    fn bins_are_ordered_by_their_first_record() {
        let r = recs(&[50, 100, 50, 100]);
        let bins = bin_records(&r, 100);
        let firsts: Vec<u32> = bins.iter().map(|b| b.record_ids[0]).collect();
        let mut sorted = firsts.clone();
        sorted.sort_unstable();
        assert_eq!(firsts, sorted, "bin ids follow the genome: {bins:?}");
    }

    #[test]
    fn equal_lengths_break_ties_by_input_order() {
        let r = recs(&[100, 100, 100, 100]);
        let a = bin_records(&r, 200);
        let b = bin_records(&r, 200);
        assert_eq!(a, b);
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn ordinals_run_reference_outermost() {
        // The executor builds one SeedTable per reference bin and reuses it over
        // every query bin, so ordinals must group by reference bin.
        let p = plan(&recs(&[100, 100]), &recs(&[100, 100]), 100);
        assert_eq!(p.reference_bins.len(), 2);
        assert_eq!(p.query_bins.len(), 2);
        let seq: Vec<(u32, u32, u32)> = p
            .units
            .iter()
            .map(|u| (u.ordinal, u.reference_bin, u.query_bin))
            .collect();
        assert_eq!(seq, vec![(0, 0, 0), (1, 0, 1), (2, 1, 0), (3, 1, 1)]);
    }

    #[test]
    fn empty_input_plans_nothing() {
        let p = plan(&[], &recs(&[100]), 100);
        assert!(p.units.is_empty() && p.reference_bins.is_empty());
    }

    #[test]
    fn preflight_shrinks_bins_until_the_plan_fits() {
        // Two 200 Mbp records with a budget that only admits one at a time.
        let r = recs(&[200_000_000, 200_000_000]);
        let q = recs(&[10_000_000]);
        let budget = unit_device_bytes(200_000_000, 10_000_000, 12, 1, 16_711_680) + 1;
        let (p, worst) = plan_within_budget(
            &r,
            &q,
            400_000_000,
            400_000_000,
            budget,
            12,
            1,
            16_711_680,
            false,
        )
        .unwrap();
        assert!(worst <= budget, "worst {worst} budget {budget}");
        assert_eq!(
            p.reference_bins.len(),
            2,
            "target halved until each bin held one record"
        );
    }

    #[test]
    fn preflight_refuses_an_unsplittable_record() {
        // One record that cannot fit however the bins are arranged.
        let r = recs(&[3_000_000_000]);
        let q = recs(&[1_000_000]);
        let err = plan_within_budget(
            &r,
            &q,
            200_000_000,
            200_000_000,
            1 << 30,
            12,
            1,
            16_711_680,
            false,
        )
        .expect_err("must refuse rather than OOM later");
        assert!(err.contains("exceeds GPU capacity"), "{err}");
        assert!(
            err.contains("intra-record splitting is not supported"),
            "{err}"
        );
    }

    #[test]
    fn device_estimate_tracks_the_real_allocations() {
        // pos_table dominates: 4 bytes per indexed reference base.
        let one_gbp = unit_device_bytes(1_000_000_000, 0, 12, 1, 0);
        assert!(
            one_gbp > 4_000_000_000,
            "pos_table must be 4 B/bp: {one_gbp}"
        );
        // --step 3 indexes a third of the positions.
        let strided = unit_device_bytes(1_000_000_000, 0, 12, 3, 0);
        assert!(
            strided < one_gbp / 2,
            "stride must reduce pos_table: {strided}"
        );
    }
}
