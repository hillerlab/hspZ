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
//! split (round 80). `--kegalign-bins` switches binning to KegAlign's
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
//! each layout carries its own frozen digest (rounds 74–76).

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
/// Every query bin runs against every reference bin, so the scheduling cost
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

/// One contiguous slot slice of a single reference bin (round 90).
///
/// A worker executes its visits in ascending bin id, and the slots inside a
/// visit in ascending slot order. `queries` is a scheduling coordinate only:
/// on the single-query path one slot is one query bin (`queries == 0..Q` is
/// whole-bin ownership); on the batch path one slot is one `(job, unit)`
/// entry of that bin's scheduling list (job-list order, then that job's
/// original unit order) and never a new query-bin id. Each entry carries its
/// original [`WorkUnit`], so execution still calls `seed_and_filter_all` on
/// the original units with their original query-bin ids and ordinals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Visit {
    pub bin: usize,
    pub queries: std::ops::Range<usize>,
}

/// Static unit-level partition of a frozen plan over `workers` workers.
///
/// Pure and deterministic. With `N = R*Q` units, quotas are `N div W` and one
/// more (exactly `N mod W` large). A quota `c` gives `c div Q` whole-bin slots
/// plus a residual of `c mod Q` units. Whole slots take the largest bins first
/// (LPT on reference bp exactly as [`assign_bins`], ties by bin id, only workers
/// with a free slot). The leftover bins form one residual stream of units in
/// descending reference-bp order (ties by bin id), then query order, cut into
/// consecutive pieces; the large/small quota
/// order along the stream is chosen by an `O(W^2)` DP over how many of each have
/// been placed, minimising total bin visits (a piece at stream offset `S` with
/// length `d` costs `1 + (((S mod Q) + d - 1) / Q)` visits, `0` when `d == 0`).
/// Ties break towards fewer visits on large-quota workers, then earliest large
/// quotas. `W == 1` is one whole-bin visit per bin, identical to today.
///
/// Thin wrapper over [`unit_partition_dims`]: slots are the query bins, so the
/// behaviour is exactly the round-90 partition this has always been.
pub fn unit_partition(plan: &Plan, workers: usize) -> Vec<Vec<Visit>> {
    if plan.units.is_empty() && !plan.reference_bins.is_empty() && !plan.query_bins.is_empty() {
        return vec![Vec::new(); workers.max(1)];
    }
    unit_partition_dims(&plan.reference_bins, plan.query_bins.len(), workers)
}

/// Dimension-based core of [`unit_partition`] shared by both callers (round 96).
///
/// `slots_per_bin` (`M`) is the slot count in every reference bin: the query-bin
/// count on the single-query path, `M = Σ_jobs Q_j` on the batch path. Total
/// units are `N = R*M`; quotas, whole-bin slots, the residual stream and the
/// quota-order DP are exactly [`unit_partition`]'s with `Q := M`. Returned
/// ranges index the caller's per-bin scheduling lists, whose entries carry the
/// original units (built by `run::build_slot_lists`).
#[allow(clippy::needless_range_loop)]
pub fn unit_partition_dims(
    reference_bins: &[Bin],
    slots_per_bin: usize,
    workers: usize,
) -> Vec<Vec<Visit>> {
    let w = workers.max(1);
    let r = reference_bins.len();
    let q = slots_per_bin;
    let n = r.saturating_mul(q);
    if r == 0 || q == 0 || n == 0 {
        return vec![Vec::new(); w];
    }
    if w == 1 {
        let mut v: Vec<Visit> = reference_bins
            .iter()
            .map(|b| Visit {
                bin: b.id as usize,
                queries: 0..q,
            })
            .collect();
        v.sort_by_key(|x| x.bin);
        return vec![v];
    }
    let a = n / w;
    let rem = n % w;
    let (nl, large_c, small_c) = if rem == 0 { (0, a, a) } else { (rem, a + 1, a) };
    let ns = w - nl;
    let (d_large, d_small) = (large_c % q, small_c % q);
    // DP over (#small placed, #large placed); stream offset is fixed by the
    // counts, so the cost of the next piece is known. Best is (total visits,
    // visits on large-quota pieces); exact ties keep the small arm, which
    // places large quotas earliest (fixed order).
    let visits_at = |s: usize, d: usize| -> u32 {
        if d == 0 {
            0
        } else {
            (1 + ((s % q) + d - 1) / q) as u32
        }
    };
    let mut best: Vec<Vec<Option<(u32, u32)>>> = vec![vec![None; nl + 1]; ns + 1];
    let mut large_last: Vec<Vec<bool>> = vec![vec![false; nl + 1]; ns + 1];
    best[0][0] = Some((0, 0));
    for i in 0..=ns {
        for j in 0..=nl {
            let Some((tot, lg)) = best[i][j] else {
                continue;
            };
            let s = i * d_small + j * d_large;
            if i < ns {
                let cand = (tot + visits_at(s, d_small), lg);
                let replace = match best[i + 1][j] {
                    None => true,
                    Some(cur) => cand < cur,
                };
                if replace {
                    best[i + 1][j] = Some(cand);
                    large_last[i + 1][j] = false;
                }
            }
            if j < nl {
                let v = visits_at(s, d_large);
                let cand = (tot + v, lg + v);
                let replace = match best[i][j + 1] {
                    None => true,
                    Some(cur) => cand < cur,
                };
                if replace {
                    best[i][j + 1] = Some(cand);
                    large_last[i][j + 1] = true;
                }
            }
        }
    }
    let mut is_large = vec![false; w];
    {
        let (mut i, mut j) = (ns, nl);
        while i + j > 0 {
            if large_last[i][j] {
                is_large[i + j - 1] = true;
                j -= 1;
            } else {
                i -= 1;
            }
        }
    }
    let quota = |x: usize| {
        if is_large[x] { large_c } else { small_c }
    };
    // Whole-bin slots, largest bins first into the lightest worker with a free
    // slot (ties by worker index), exactly like `assign_bins`.
    let mut order: Vec<&Bin> = reference_bins.iter().collect();
    order.sort_by(|x, y| y.total_bp.cmp(&x.total_bp).then(x.id.cmp(&y.id)));
    let mut load = vec![0u64; w];
    let mut used = vec![0usize; w];
    let mut taken = vec![false; r];
    let mut visits: Vec<Vec<Visit>> = vec![Vec::new(); w];
    for b in order {
        let mut pick: Option<usize> = None;
        for x in 0..w {
            if used[x] >= quota(x) / q {
                continue;
            }
            pick = Some(match pick {
                None => x,
                Some(p) => {
                    if (load[x], x) < (load[p], p) {
                        x
                    } else {
                        p
                    }
                }
            });
        }
        let Some(p) = pick else { break };
        load[p] += b.total_bp;
        used[p] += 1;
        taken[b.id as usize] = true;
        visits[p].push(Visit {
            bin: b.id as usize,
            queries: 0..q,
        });
    }
    // Residual stream: leftover bins (descending bp, ties by id), slot order.
    let mut rest: Vec<&Bin> = reference_bins
        .iter()
        .filter(|b| !taken[b.id as usize])
        .collect();
    rest.sort_by(|x, y| y.total_bp.cmp(&x.total_bp).then(x.id.cmp(&y.id)));
    let mut stream: Vec<(usize, usize)> = Vec::with_capacity(rest.len() * q);
    for b in &rest {
        for qq in 0..q {
            stream.push((b.id as usize, qq));
        }
    }
    let mut off = 0usize;
    for x in 0..w {
        let d = quota(x) % q;
        let mut k = off;
        let end = off + d;
        while k < end {
            let b = stream[k].0;
            let mut e = k + 1;
            while e < end && stream[e].0 == b {
                e += 1;
            }
            visits[x].push(Visit {
                bin: b,
                queries: stream[k].1..stream[e - 1].1 + 1,
            });
            k = e;
        }
        off = end;
    }
    debug_assert_eq!(off, stream.len());
    for v in &mut visits {
        v.sort_by_key(|t| t.bin);
        debug_assert!(
            v.windows(2).all(|w| w[0].bin != w[1].bin),
            "two visits of one bin on a worker"
        );
    }
    visits
}

/// KegAlign's block rule, reproduced exactly, for matched-granularity runs
/// (matched-granularity mode).
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
/// `--seq-block-size` is the *target*, not a ceiling: a 249 Mbp chr1 stays
/// one 249 Mbp bin against a 200 Mbp target rather than being split. Longest
/// record first into the currently smallest bin, which is the standard LPT
/// heuristic and is deterministic given the ordinal tie-break.
pub fn bin_records(records: &[RecordMeta], target_bp: u64) -> Vec<Bin> {
    if records.is_empty() {
        return Vec::new();
    }
    let total: u64 = records.iter().map(|r| r.len).sum();
    let n_bins = layout_n_bins(total, target_bp, records.len());
    bin_records_n(records, n_bins)
}

/// `n_bins = clamp(ceil(total/target), 1, n_records)` — the count the normal
/// planner actually uses. Sequential overshoot (`bin_records_sequential`) is a
/// different rule and is only `--kegalign-bins`.
pub fn layout_n_bins(total_bp: u64, target_bp: u64, n_records: usize) -> usize {
    if n_records == 0 {
        return 0;
    }
    (total_bp.div_ceil(target_bp.max(1)) as usize).clamp(1, n_records)
}

/// LPT-packs records into exactly `n_bins` bins (or fewer if some stay empty,
/// which they do not when `n_bins ≤ n_records`). Same assignment as
/// [`bin_records`] at the matching count.
pub fn bin_records_n(records: &[RecordMeta], n_bins: usize) -> Vec<Bin> {
    if records.is_empty() || n_bins == 0 {
        return Vec::new();
    }
    let n_bins = n_bins.clamp(1, records.len());

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
pub fn plan(reference: &[RecordMeta], query: &[RecordMeta], target_bp: u64) -> Plan {
    plan_with(reference, query, target_bp, target_bp, false)
}

/// [`plan`], with KegAlign's sequential block fill instead of LPT when
/// `kegalign_bins` is set (Mode A).
/// The two sides take independent targets (round 80). Splitting the reference is what
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
    from_bins(reference_bins, query_bins)
}

/// Default block target both sides fall back to: the KegAlign-matched digest
/// (`-B`'s CLI default). `-B 0` resolves to this at `workers <= 1`, and the
/// automatic layout derives its reference count from it.
pub const DEFAULT_BLOCK_TARGET: u64 = 500_000_000;

/// Rejected combination: the automatic layout bins by LPT record balance,
/// which cannot reproduce KegAlign's sequential fill.
pub const AUTO_KEGALIGN_ERROR: &str =
    "automatic layout uses LPT bins; drop --kegalign-bins or give -B explicitly";

/// Automatic reference-bin count for `-B 0` at `workers >= 2` (rank-2 rule):
/// the smallest multiple of `workers` that does not coarsen the default
/// layout (`R_def`), capped at one bin per record so no worker idles by
/// construction. Divisibility is the whole point: `R % W == 0` keeps every
/// worker holding the same number of bins.
pub fn auto_ref_bin_count(n_records: usize, r_def: usize, workers: usize) -> usize {
    if n_records == 0 {
        return 0;
    }
    let w = workers.max(1);
    n_records.min(r_def.max(1).div_ceil(w).saturating_mul(w))
}

/// Resolved automatic layout: the plan plus the targets that reproduce it
/// through the normal planner, for the manifest and the layout line.
pub struct AutoLayout {
    pub plan: Plan,
    /// `Some((R, Q))` when the automatic candidate did not fit the device budget
    /// and the plan fell back to the ordinary default policy (Grok r88 review:
    /// `-B 0` must never be harder to admit than the default).
    pub fell_back: Option<(usize, usize)>,
    /// `ceil(total_ref_bp / R)`: any explicit `-B` in
    /// `(total/R, total/(R-1)]` round-trips through [`layout_n_bins`] to the
    /// same `R`, and [`bin_records_n`] is a pure function of the count, so an
    /// explicit rerun with this target rebuilds these bins exactly.
    pub seq_target: u64,
    /// Whole-query total when Q=1 collapses the query (same round-trip), else
    /// the target actually used (explicit override or default).
    pub query_target: u64,
}

/// Everything [`auto_layout`]'s fit checks need beyond metadata and budgets.
/// `threads`/`max_seeds` feed [`host_estimate`]; the caller derives them the
/// same way `run` does so the layout-time check agrees with the real preflight.
pub struct AutoCtx {
    pub kmer_size: usize,
    pub step: u32,
    pub max_hits: u32,
    pub wga_chunk_size: u32,
    pub transitions: bool,
    pub threads: usize,
    pub max_seeds: usize,
}

/// Automatic layout for `-B 0` (rank-2 policy).
///
/// `workers <= 1` resolves trivially to today's default plan — no Q=1 attempt,
/// no new bins — so a one-worker run is byte-identical to `-B 500000000`.
/// At `workers >= 2` the reference takes [`auto_ref_bin_count`] bins via
/// [`bin_records_n`], and the query collapses to one bin iff *both* the device
/// fit ([`worst_unit_bytes`] against `budget_bytes`, the same budget
/// [`plan_within_budget`] uses) and the host fit ([`host_preflight`] against
/// `host_available`, already discounted; `None` means unknown and admits)
/// allow it — otherwise the default query blocks are kept. An explicit
/// `query_override` is honoured as given and skips the fit check. The query
/// target is never derived from the reference target.
///
/// Fit-checks and falls back, never shrinks: a final plan that still exceeds
/// the device budget errors with an explicit-flags hint, exactly like
/// matched-granularity mode, instead of coarsening the layout behind the
/// caller's back. Kernels, chunking and emitter ordinals are untouched —
/// [`from_bins`] keeps dense ref-outer ordinals.
#[allow(clippy::too_many_arguments)]
pub fn auto_layout(
    reference: &[RecordMeta],
    query: &[RecordMeta],
    workers: usize,
    exec_workers: usize,
    budget_bytes: u64,
    host_available: Option<u64>,
    query_override: Option<u64>,
    ctx: &AutoCtx,
) -> Result<AutoLayout, String> {
    let ref_total: u64 = reference.iter().map(|r| r.len).sum();
    let qry_total: u64 = query.iter().map(|r| r.len).sum();
    if workers.max(1) <= 1 {
        let q = query_override.unwrap_or(DEFAULT_BLOCK_TARGET);
        let plan = plan_with(reference, query, DEFAULT_BLOCK_TARGET, q.max(1), false);
        return Ok(AutoLayout {
            plan,
            fell_back: None,
            seq_target: DEFAULT_BLOCK_TARGET,
            query_target: q,
        });
    }
    let r_def = layout_n_bins(ref_total, DEFAULT_BLOCK_TARGET, reference.len());
    let r = auto_ref_bin_count(reference.len(), r_def, workers);
    let reference_bins = bin_records_n(reference, r);
    let seq_target = match r {
        0 => DEFAULT_BLOCK_TARGET,
        _ => ref_total.div_ceil(r as u64).max(1),
    };
    let (query_bins, query_target) = match query_override {
        Some(q) => (bin_records(query, q.max(1)), q),
        None => {
            let single = bin_records_n(query, 1);
            let cand = from_bins(reference_bins.clone(), single.clone());
            let device_ok = worst_unit_bytes(
                &cand,
                ctx.kmer_size,
                ctx.step,
                ctx.max_hits,
                ctx.wga_chunk_size,
                ctx.transitions,
            )
            .map(|worst| worst <= budget_bytes)
            .unwrap_or(false);
            let host_ok = match host_available {
                None => true,
                Some(avail) => {
                    let est = host_estimate(
                        &cand,
                        ref_total,
                        qry_total,
                        ctx.kmer_size,
                        ctx.step,
                        ctx.threads,
                        ctx.max_seeds,
                    );
                    host_preflight(
                        &est,
                        &assign_bins(&reference_bins, exec_workers.max(1)),
                        avail,
                    )
                    .is_ok()
                }
            };
            if device_ok && host_ok {
                (single, qry_total.max(1))
            } else {
                (
                    bin_records(query, DEFAULT_BLOCK_TARGET),
                    DEFAULT_BLOCK_TARGET,
                )
            }
        }
    };
    let plan = from_bins(reference_bins, query_bins);
    let worst = worst_unit_bytes(
        &plan,
        ctx.kmer_size,
        ctx.step,
        ctx.max_hits,
        ctx.wga_chunk_size,
        ctx.transitions,
    )?;
    if worst > budget_bytes {
        if query_override.is_none() {
            // The candidate does not fit: run exactly what `-B 500000000` would
            // (including plan_within_budget's own halving), never an error.
            let (candidate_r, candidate_q) = (plan.reference_bins.len(), plan.query_bins.len());
            let (plan, _worst) = plan_within_budget(
                reference,
                query,
                DEFAULT_BLOCK_TARGET,
                DEFAULT_BLOCK_TARGET,
                budget_bytes,
                ctx.kmer_size,
                ctx.step,
                ctx.max_hits,
                false,
                ctx.wga_chunk_size,
                ctx.transitions,
            )?;
            return Ok(AutoLayout {
                plan,
                fell_back: Some((candidate_r, candidate_q)),
                seq_target: DEFAULT_BLOCK_TARGET,
                query_target: DEFAULT_BLOCK_TARGET,
            });
        }
        return Err(format!(
            "automatic layout (R={} Q={}) with the explicit --query-block-size needs {:.1} GB per work unit, only {:.1} GB budgeted: \
             drop --query-block-size or give -B explicitly",
            plan.reference_bins.len(),
            plan.query_bins.len(),
            worst as f64 / 1e9,
            budget_bytes as f64 / 1e9
        ));
    }
    Ok(AutoLayout {
        plan,
        fell_back: None,
        seq_target,
        query_target,
    })
}

/// The one stderr line an automatic layout prints: the explicit flags that
/// reproduce it plus the per-worker bin counts (`owners`) under the layout
/// worker count, so the run is reproducible without `-B 0`.
pub fn auto_layout_line(workers: usize, seq_target: u64, query_target: u64, plan: &Plan) -> String {
    let owners = assign_bins(&plan.reference_bins, workers.max(1))
        .iter()
        .map(|bins| bins.len().to_string())
        .collect::<Vec<_>>()
        .join("+");
    let ref_total: u64 = plan.reference_bins.iter().map(|b| b.total_bp).sum();
    let n_records: usize = plan.reference_bins.iter().map(|b| b.record_ids.len()).sum();
    let r = plan.reference_bins.len();
    // The printed -B reproduces these bins only if the count-first planner
    // derives the same R from it (true for genome-scale inputs; tiny toy sets
    // can break it). Say so rather than print flags that would not replay.
    let hint = if layout_n_bins(ref_total, seq_target.max(1), n_records) == r {
        String::new()
    } else {
        " [flags do not reproduce this R on this input; use --dump-manifest/--from-manifest]".into()
    };
    format!(
        "layout: auto W={workers} -> --seq-block-size {seq_target} --query-block-size \
         {query_target} (R={r} Q={}, {} units, owners {owners}){hint}",
        plan.query_bins.len(),
        plan.units.len()
    )
}

fn from_bins(reference_bins: Vec<Bin>, query_bins: Vec<Bin>) -> Plan {
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

/// Packed block length: `total_bp + nrecords - 1` (`sequence::pack` joins
/// records with `SEP` and drops the trailing separator from `block_len`).
fn packed_len(total_bp: u64, nrecords: usize) -> Result<u64, String> {
    if nrecords == 0 {
        return Err("packed length of empty bin".into());
    }
    let seps = (nrecords as u64) - 1;
    let packed = total_bp
        .checked_add(seps)
        .ok_or_else(|| "packed length overflow".to_string())?;
    if u32::try_from(packed).is_err() || usize::try_from(packed).is_err() {
        return Err(format!("packed length {packed} does not fit u32/usize"));
    }
    Ok(packed)
}

pub(crate) fn packed_bin_len(side: &str, b: &Bin) -> Result<u64, String> {
    packed_len(b.total_bp, b.record_ids.len()).map_err(|e| format!("{side} bin {}: {e}", b.id))
}

fn max_packed(bins: &[Bin]) -> Result<u64, String> {
    let mut m = 0u64;
    for b in bins {
        m = m.max(packed_len(b.total_bp, b.record_ids.len())?);
    }
    Ok(m)
}

/// Conservative device-byte bound for one work unit, from hspz allocations.
///
/// Lifetimes are conservative, not exact. Terms map to real buffers:
/// - `index_table`: `4 * 4^k` (`Engine::index_table`)
/// - `pos_table`: `4 * ceil(ref_packed / step)` (`Engine::pos_table`)
/// - encoded ref: `ref_packed` (`Engine::ref_seq`)
/// - query: `2 * query_packed` (`query_seq` + `query_rc_seq`); swap overlap
///   is added in [`worst_unit_bytes`]
/// - seed slots: `16S` — two `u64` slots (`seed_slots[2]`)
/// - hit counts: `8S` — `buf_hit_num` (`u32`), covering nonpersistent old/new
///   replacement as well as the persistent `4S` buffer
/// - kmer scratch: `4C` (`buf_seed_kmer`)
/// - shape: `4k` (`seed_shape`)
/// - matrix: `256` (`sub_mat`, 64 × `i32`)
/// - scan: `12*ceil(C/256)+8*ceil(S/256)+12*ceil(H/256)` (`d_block_sums` /
///   `buf_seed_offsets` / hit-count and done-scan offsets). The dense path keeps
///   the *survivor* offsets array (`_survivor_offsets`) alive through the done
///   scan — three H-scaled `u32` buffers (survivor offsets + done sums + done
///   offsets), so `H` scans at 12 B/ceil(H/256); non-dense stays 8 (done sums +
///   done offsets only).
/// - hits: dense `49H` (anchor 8 + flag 1 + survivor 4 + HSP 16 + done 4 +
///   reduced 16); sparse `36H` (persistent 20H + reduced 16H). Under the
///   `counters` feature add `16H` (`d_stats`, 2 x `u64` per materialized hit
///   alive through the reduced buffers); default coefficients unchanged.
/// - buckets: under `ref-loc-buckets` add `12H` (permuted anchor 8 + raw index
///   4), `ceil(H/8)` for the sorted bitmask flags (round 2) and
///   `4*32*ceil(H/256) + 8*ceil(32*ceil(H/256)/256)` for the bucket histogram
///   and its block sums; zero otherwise. The survivor sort needs no device
///   scratch — it is one block of shared memory, or a host round trip.
/// - 64 MiB unmodelled headroom
///
/// `C = wga_chunk_size`, `S = C * (1+k if transitions else 1)`, `H = max_hits`.
/// `max_hits == 0` is allowed for arithmetic tests; a manifest still rejects it.
/// Never lowers `max_hits`.
///
/// No checked arithmetic inside: `ref_bp`/`query_bp` are u64, the rest are ≤ u32
/// with `k` bounded 4..=15 (matching `Shape::parse`), so every intermediate and
/// the total stay far below u128::MAX. The inputs are validated first, and only
/// the final conversion to u64 can fail (a huge sum).
pub fn unit_device_bytes(
    ref_bp: u64,
    query_bp: u64,
    kmer_size: usize,
    step: u32,
    max_hits: u32,
    wga_chunk_size: u32,
    transitions: bool,
) -> Result<u64, String> {
    if step == 0 {
        return Err("step must be positive".into());
    }
    if wga_chunk_size == 0 {
        return Err("wga_chunk_size must be positive".into());
    }
    if !(4..=15).contains(&kmer_size) {
        return Err(format!("k={kmer_size} outside Shape::parse range 4..=15"));
    }
    let k = kmer_size as u128;
    let c = u128::from(wga_chunk_size);
    let per_pos = if transitions { 1 + k } else { 1 };
    let s = c * per_pos;
    if s > u128::from(u32::MAX) {
        return Err("seed count S exceeds u32::MAX".into());
    }
    let h = u128::from(max_hits);
    // Same coefficient used for the H-scan term, chosen at compile time.
    let hit_scan_b: u128 = if cfg!(feature = "dense-anchors") {
        12
    } else {
        8
    };
    let hit_b: u128 = if cfg!(feature = "dense-anchors") {
        49
    } else {
        36
    };
    // `counters` keeps `d_stats` (2 x u64 per materialized hit) alive through
    // the reduced buffers; census adds host-only buffers, not device capacity.
    let counters_b: u128 = if cfg!(feature = "counters") { 16 } else { 0 };
    // `ref-loc-buckets` keeps the permuted anchor copy (8) and its raw-hit index
    // (4) alive across the gate, plus the bucket-major histogram
    // (`32 * ceil(H/256)` u32) and its block sums, held twice for the host round
    // trip. Never lowers `max_hits`: a unit that no longer fits is replanned or
    // rejected by the existing fit logic.
    let bucket_b: u128 = if cfg!(feature = "ref-loc-buckets") {
        12
    } else {
        0
    };
    let bucket_scan_b: u128 = if cfg!(feature = "ref-loc-buckets") {
        let counts = 32 * h.div_ceil(256);
        4 * counts + 8 * counts.div_ceil(256)
    } else {
        0
    };
    // Round 2: the gate's keep bits, one u32 per 32 hits, additive to the byte
    // flags the default path still allocates. Never lowers `max_hits`.
    let bucket_bits_b: u128 = if cfg!(feature = "ref-loc-buckets") {
        h.div_ceil(8)
    } else {
        0
    };

    // The largest term is below 2^66 and the full sum below 2^67;
    // only the final conversion to u64 can overflow.
    let total = u128::from(ref_bp).div_ceil(u128::from(step)) * 4
        + (1u128 << (2 * kmer_size)) * 4
        + u128::from(ref_bp)
        + 2 * u128::from(query_bp)
        + h * hit_b
        + h * counters_b
        + h * bucket_b
        + bucket_scan_b
        + bucket_bits_b
        + 16 * s
        + 8 * s
        + 4 * c
        + 4 * k
        + 256
        + 12 * c.div_ceil(256)
        + 8 * s.div_ceil(256)
        + hit_scan_b * h.div_ceil(256)
        + 64 * 1024 * 1024;
    u64::try_from(total).map_err(|_| "device estimate exceeds u64".into())
}

/// Conservative bound for the largest Cartesian unit, using packed lengths.
///
/// Full Cartesian means max packed reference × max packed query is sufficient.
/// Query swap allocates the RHS while the old buffers still live: one query bin
/// is `2 Qmax` already in [`unit_device_bytes`]; more than one query bin adds
/// `Qmax` (a `3 Qmax` bound covering both replacement phases, any bin order).
pub fn worst_unit_bytes(
    plan: &Plan,
    kmer_size: usize,
    step: u32,
    max_hits: u32,
    wga_chunk_size: u32,
    transitions: bool,
) -> Result<u64, String> {
    let _ = unit_device_bytes(0, 0, kmer_size, step, max_hits, wga_chunk_size, transitions)?;
    if plan.reference_bins.is_empty() || plan.query_bins.is_empty() {
        return Ok(0);
    }
    let rmax = max_packed(&plan.reference_bins)?;
    let qmax = max_packed(&plan.query_bins)?;
    let unit = unit_device_bytes(
        rmax,
        qmax,
        kmer_size,
        step,
        max_hits,
        wga_chunk_size,
        transitions,
    )?;
    if plan.query_bins.len() > 1 {
        unit.checked_add(qmax)
            .ok_or_else(|| "query-swap overlap exceeds u64".into())
    } else {
        Ok(unit)
    }
}

/// Physical hit-capacity ceiling for a frozen plan under a device budget.
///
/// `H = max_hits` is the semantic chunk target; the return `C >= H` is a
/// physical allowance only — the largest capacity whose
/// [`worst_unit_bytes`] still fits `budget_bytes`. Reuses the estimator, does
/// not replan, mutate the plan, or allocate hit buffers. Callers apply any
/// kernel-safe ceiling separately; this proves no kernel index bound.
pub fn max_hit_capacity(
    plan: &Plan,
    budget_bytes: u64,
    kmer_size: usize,
    step: u32,
    max_hits: u32,
    wga_chunk_size: u32,
    transitions: bool,
) -> Result<u32, String> {
    if max_hits == 0 {
        return Err("max_hits must be positive".into());
    }
    let base = worst_unit_bytes(plan, kmer_size, step, max_hits, wga_chunk_size, transitions)?;
    if base > budget_bytes {
        return Err(format!(
            "max_hits {max_hits} needs {base} bytes per work unit, only {budget_bytes} bytes budgeted"
        ));
    }
    if plan.reference_bins.is_empty() || plan.query_bins.is_empty() {
        return Ok(max_hits);
    }
    let mut lo = max_hits;
    let mut hi = u32::MAX;
    while lo < hi {
        let mid = (lo as u64 + (hi as u64 - lo as u64).div_ceil(2)) as u32;
        let cost = worst_unit_bytes(plan, kmer_size, step, mid, wga_chunk_size, transitions)?;
        if cost <= budget_bytes {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    Ok(lo)
}

/// Per-device byte budget when `workers` share `devices` by `w % devices`.
///
/// Divides free VRAM by `ceil(W/D)` so the most-loaded device is covered.
pub fn worker_device_budget(free_bytes: u64, workers: usize, devices: usize) -> u64 {
    free_bytes / workers.max(1).div_ceil(devices.max(1)) as u64
}

/// Raises the bin count until the largest planned unit fits, or reports the
/// record that cannot fit at all.
///
/// Returns the accepted plan and the largest unit estimate. A single record that
/// exceeds capacity is a hard error: v1 does not split records, and discovering
/// it through a CUDA OOM mid-run is exactly what this avoids. Intermediate
/// oversized packed bins shrink; matched-granularity and frozen layouts fail.
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
    wga_chunk_size: u32,
    transitions: bool,
) -> Result<(Plan, u64), String> {
    let _ = unit_device_bytes(0, 0, kmer_size, step, max_hits, wga_chunk_size, transitions)?;
    let worst_of =
        |p: &Plan| worst_unit_bytes(p, kmer_size, step, max_hits, wga_chunk_size, transitions);
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
        let worst = worst_of(&p)?;
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
    let unsplittable = || {
        let biggest = reference
            .iter()
            .chain(query.iter())
            .max_by_key(|r| r.len)
            .map(|r| format!("{} ({} bp)", r.name, r.len))
            .unwrap_or_else(|| "<none>".into());
        format!(
            "record {biggest} exceeds GPU capacity ({budget_bytes} bytes available); \
             intra-record splitting is not supported in v1"
        )
    };
    let mut target = target_bp.max(1);
    let mut qtarget = query_target_bp.max(1);
    for _ in 0..24 {
        let p = plan_with(reference, query, target, qtarget, false);
        match worst_of(&p) {
            Ok(worst) if worst <= budget_bytes => return Ok((p, worst)),
            Ok(_) | Err(_) if can_split(&p) => {
                // Both sides halve, so an explicit `--query-block-size` keeps its
                // ratio to the reference target while the plan shrinks.
                target /= 2;
                qtarget = (qtarget / 2).max(1);
            }
            Ok(_) | Err(_) => return Err(unsplittable()),
        }
    }
    Err("could not find a bin size that fits GPU memory".into())
}

fn can_split(p: &Plan) -> bool {
    p.reference_bins.iter().any(|b| b.record_ids.len() > 1)
        || p.query_bins.iter().any(|b| b.record_ids.len() > 1)
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
    host_estimate_sizes(
        largest_ref,
        largest_qry,
        ref_bp_total,
        qry_bp_total,
        kmer_size,
        step,
        threads,
        max_seeds,
    )
}

/// As [`host_estimate`], with the largest reference and query bins supplied by
/// the caller instead of read from one plan. Batch mode holds every job in RAM
/// and picks both maxima over all jobs, so it calls this directly; the formula
/// itself exists once.
#[allow(clippy::too_many_arguments)]
pub(crate) fn host_estimate_sizes(
    largest_ref: u64,
    largest_qry: u64,
    ref_bp_total: u64,
    qry_bp_total: u64,
    kmer_size: usize,
    step: u32,
    threads: usize,
    max_seeds: usize,
) -> HostEstimate {
    let index = (1u64 << (2 * kmer_size)) * 4; // index_table, 4 B/entry
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

/// FNV-1a over names, lengths and bases, in input order. A frozen manifest
/// refuses to run if this does not match the files on the node.
pub(crate) const FNV_OFFSET: u64 = 0xcbf29ce484222325;

pub fn records_hash(records: &[(String, Vec<u8>)]) -> u64 {
    let mut h = FNV_OFFSET;
    for (name, seq) in records {
        h = fnv1a(name.as_bytes(), h);
        h = fnv1a(&(seq.len() as u64).to_le_bytes(), h);
        h = fnv1a(seq, h);
    }
    h
}

pub(crate) fn fnv1a(data: &[u8], mut h: u64) -> u64 {
    const P: u64 = 0x0100_0000_01b3;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(P);
    }
    h
}

/// FNV-1a of the running executable. Called only when writing or checking a
/// frozen manifest, never on ordinary execution.
pub fn executable_hash() -> Result<u64, String> {
    use std::io::Read;
    let path = std::env::current_exe().map_err(|e| e.to_string())?;
    let mut file = std::fs::File::open(&path).map_err(|e| e.to_string())?;
    let mut buf = [0u8; 64 * 1024];
    let mut h = FNV_OFFSET;
    loop {
        let n = file.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        h = fnv1a(&buf[..n], h);
    }
    Ok(h)
}

/// Every Cargo.toml feature flag that is on in this binary, in declaration
/// order. Empty when built with `--no-default-features`.
pub fn compiled_features() -> String {
    [
        ("counters", cfg!(feature = "counters")),
        (
            "nvidia-find-num-unchecked",
            cfg!(feature = "nvidia-find-num-unchecked"),
        ),
        (
            "nvidia-uninit-seed-buffers",
            cfg!(feature = "nvidia-uninit-seed-buffers"),
        ),
        ("device-seeds", cfg!(feature = "device-seeds")),
        ("device-seeds-check", cfg!(feature = "device-seeds-check")),
        ("warp-score-gate", cfg!(feature = "warp-score-gate")),
        ("dense-anchors", cfg!(feature = "dense-anchors")),
        ("find-hits-warp", cfg!(feature = "find-hits-warp")),
        ("ref-loc-buckets", cfg!(feature = "ref-loc-buckets")),
        ("left-pair-tile", cfg!(feature = "left-pair-tile")),
        ("simd-prelude", cfg!(feature = "simd-prelude")),
    ]
    .into_iter()
    .filter_map(|(name, on)| on.then_some(name))
    .collect::<Vec<_>>()
    .join(",")
}

/// Frozen execution plan: bins, ordinals, resolved hit cap, seed/scoring
/// identity. A node given this file *validates fit and fails*; it does not
/// re-run `plan_within_budget`'s shrink loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanManifest {
    pub version: u32,
    pub hspz_version: String,
    pub features: String,
    pub max_hits: u32,
    pub hsp_blocks: u32,
    pub seed: String,
    pub step: u32,
    pub transitions: bool,
    pub xdrop: i32,
    pub hspthresh: i32,
    pub noentropy: bool,
    pub wga_chunk_size: u32,
    pub lastz_interval_size: u32,
    pub kegalign_bins: bool,
    pub seq_block_size: u64,
    pub query_block_size: u64,
    pub ref_hash: u64,
    pub qry_hash: u64,
    pub executable_hash: u64,
    pub sub_mat: Vec<i32>,
    pub strand: String,
    pub target_prefix: String,
    pub query_prefix: String,
    pub plan: Plan,
}

impl PlanManifest {
    pub const FORMAT: u32 = 2;

    pub fn write<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        writeln!(w, "hspz-manifest {}", self.version)?;
        writeln!(w, "hspz_version {}", self.hspz_version)?;
        writeln!(w, "features {}", self.features)?;
        writeln!(w, "max_hits {}", self.max_hits)?;
        writeln!(w, "hsp_blocks {}", self.hsp_blocks)?;
        writeln!(w, "seed {}", self.seed)?;
        writeln!(w, "step {}", self.step)?;
        writeln!(w, "transitions {}", self.transitions as u8)?;
        writeln!(w, "xdrop {}", self.xdrop)?;
        writeln!(w, "hspthresh {}", self.hspthresh)?;
        writeln!(w, "noentropy {}", self.noentropy as u8)?;
        writeln!(w, "wga_chunk_size {}", self.wga_chunk_size)?;
        writeln!(w, "lastz_interval_size {}", self.lastz_interval_size)?;
        writeln!(w, "kegalign_bins {}", self.kegalign_bins as u8)?;
        writeln!(w, "seq_block_size {}", self.seq_block_size)?;
        writeln!(w, "query_block_size {}", self.query_block_size)?;
        writeln!(w, "ref_hash {:016x}", self.ref_hash)?;
        writeln!(w, "qry_hash {:016x}", self.qry_hash)?;
        writeln!(w, "executable_hash {:016x}", self.executable_hash)?;
        writeln!(w, "sub_mat {}", join_csv(&self.sub_mat))?;
        writeln!(w, "strand {}", self.strand)?;
        writeln!(
            w,
            "target_prefix_hex {}",
            encode_hex(self.target_prefix.as_bytes())
        )?;
        writeln!(
            w,
            "query_prefix_hex {}",
            encode_hex(self.query_prefix.as_bytes())
        )?;
        writeln!(w, "n_ref_bins {}", self.plan.reference_bins.len())?;
        writeln!(w, "n_query_bins {}", self.plan.query_bins.len())?;
        writeln!(w, "n_units {}", self.plan.units.len())?;
        for b in &self.plan.reference_bins {
            writeln!(
                w,
                "bin R {} {} {}",
                b.id,
                b.total_bp,
                join_csv(&b.record_ids)
            )?;
        }
        for b in &self.plan.query_bins {
            writeln!(
                w,
                "bin Q {} {} {}",
                b.id,
                b.total_bp,
                join_csv(&b.record_ids)
            )?;
        }
        for u in &self.plan.units {
            writeln!(w, "unit {} {} {}", u.ordinal, u.reference_bin, u.query_bin)?;
        }
        Ok(())
    }

    pub fn read(s: &str) -> Result<Self, String> {
        const SCALAR_KEYS: &[&str] = &[
            "hspz-manifest",
            "hspz_version",
            "features",
            "max_hits",
            "hsp_blocks",
            "seed",
            "step",
            "transitions",
            "xdrop",
            "hspthresh",
            "noentropy",
            "wga_chunk_size",
            "lastz_interval_size",
            "kegalign_bins",
            "seq_block_size",
            "query_block_size",
            "ref_hash",
            "qry_hash",
            "executable_hash",
            "sub_mat",
            "strand",
            "target_prefix_hex",
            "query_prefix_hex",
            "n_ref_bins",
            "n_query_bins",
            "n_units",
        ];
        let mut scalars = std::collections::HashMap::<&str, (usize, &str)>::new();
        let mut reference_bins = Vec::new();
        let mut query_bins = Vec::new();
        let mut units = Vec::new();

        for (lineno, raw) in s.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let line_no = lineno + 1;
            let err = |m: &str| format!("manifest line {line_no}: {m}");
            let key = line.split_whitespace().next().unwrap();
            let rest = line[key.len()..].trim();
            if SCALAR_KEYS.contains(&key) {
                if scalars.insert(key, (line_no, rest)).is_some() {
                    return Err(err(&format!("duplicate {key}")));
                }
                continue;
            }
            match key {
                "bin" => {
                    let mut p = rest.split_whitespace();
                    let side = p.next().ok_or_else(|| err("bin missing side"))?;
                    let id = p.next().ok_or_else(|| err("bin missing id"))?;
                    let total_bp = p.next().ok_or_else(|| err("bin missing bp"))?;
                    let ids = p.next().ok_or_else(|| err("bin missing record ids"))?;
                    if p.next().is_some() {
                        return Err(err("trailing tokens"));
                    }
                    let id: u32 = id.parse().map_err(|_| err("bad bin id"))?;
                    let total_bp: u64 = total_bp.parse().map_err(|_| err("bad bin bp"))?;
                    let record_ids = parse_ids(ids).map_err(|e| err(&e))?;
                    let bin = Bin {
                        id,
                        record_ids,
                        total_bp,
                    };
                    match side {
                        "R" => reference_bins.push(bin),
                        "Q" => query_bins.push(bin),
                        _ => return Err(err("bin side must be R or Q")),
                    }
                }
                "unit" => {
                    let mut p = rest.split_whitespace();
                    let ordinal = p.next().ok_or_else(|| err("unit missing ordinal"))?;
                    let reference_bin = p.next().ok_or_else(|| err("unit missing ref"))?;
                    let query_bin = p.next().ok_or_else(|| err("unit missing query"))?;
                    if p.next().is_some() {
                        return Err(err("trailing tokens"));
                    }
                    units.push(WorkUnit {
                        ordinal: ordinal.parse().map_err(|_| err("bad ordinal"))?,
                        reference_bin: reference_bin.parse().map_err(|_| err("bad ref bin"))?,
                        query_bin: query_bin.parse().map_err(|_| err("bad query bin"))?,
                    });
                }
                _ => return Err(err(&format!("unknown key {key}"))),
            }
        }

        if !scalars.contains_key("hspz-manifest") {
            return Err("not an hspz-manifest".into());
        }
        for k in SCALAR_KEYS {
            if !scalars.contains_key(k) {
                return Err(format!("missing {k}"));
            }
        }

        let one = |k: &str| -> Result<(usize, &str), String> {
            let (line, rest) = scalars[k];
            Ok((line, one_token(rest, line, k)?))
        };
        let opt = |k: &str| -> Result<(usize, &str), String> {
            let (line, rest) = scalars[k];
            Ok((line, opt_token(rest, line, k)?))
        };

        let (line, tok) = one("hspz-manifest")?;
        let version: u32 = parse_at(tok, line, "format version")?;
        if version != Self::FORMAT {
            return Err(format!("manifest line {line}: unsupported format version"));
        }

        let (_, tok) = one("hspz_version")?;
        let hspz_version = tok.to_string();
        let (_, tok) = opt("features")?;
        let features = tok.to_string();
        let (line, tok) = one("max_hits")?;
        let max_hits: u32 = parse_at(tok, line, "max_hits")?;
        let (line, tok) = one("hsp_blocks")?;
        let hsp_blocks: u32 = parse_at(tok, line, "hsp_blocks")?;
        let (_, tok) = one("seed")?;
        let seed = tok.to_string();
        let (line, tok) = one("step")?;
        let step: u32 = parse_at(tok, line, "step")?;
        let (line, tok) = one("transitions")?;
        let transitions = parse_flag(tok, line, "transitions")?;
        let (line, tok) = one("xdrop")?;
        let xdrop: i32 = parse_at(tok, line, "xdrop")?;
        let (line, tok) = one("hspthresh")?;
        let hspthresh: i32 = parse_at(tok, line, "hspthresh")?;
        let (line, tok) = one("noentropy")?;
        let noentropy = parse_flag(tok, line, "noentropy")?;
        let (line, tok) = one("wga_chunk_size")?;
        let wga_chunk_size: u32 = parse_at(tok, line, "wga_chunk_size")?;
        let (line, tok) = one("lastz_interval_size")?;
        let lastz_interval_size: u32 = parse_at(tok, line, "lastz_interval_size")?;
        let (line, tok) = one("kegalign_bins")?;
        let kegalign_bins = parse_flag(tok, line, "kegalign_bins")?;
        let (line, tok) = one("seq_block_size")?;
        let seq_block_size: u64 = parse_at(tok, line, "seq_block_size")?;
        let (line, tok) = one("query_block_size")?;
        let query_block_size: u64 = parse_at(tok, line, "query_block_size")?;
        let (line, tok) = one("ref_hash")?;
        let ref_hash = parse_hex_u64(tok, line, "ref_hash")?;
        let (line, tok) = one("qry_hash")?;
        let qry_hash = parse_hex_u64(tok, line, "qry_hash")?;
        let (line, tok) = one("executable_hash")?;
        let executable_hash = parse_hex_u64(tok, line, "executable_hash")?;
        let (line, tok) = one("sub_mat")?;
        let sub_mat = parse_sub_mat(tok).map_err(|e| format!("manifest line {line}: {e}"))?;
        let (line, tok) = one("strand")?;
        if !matches!(tok, "plus" | "minus" | "both") {
            return Err(format!(
                "manifest line {line}: strand must be plus, minus or both"
            ));
        }
        let strand = tok.to_string();
        let (line, tok) = opt("target_prefix_hex")?;
        let target_prefix = decode_hex_utf8(tok, "target_prefix")
            .map_err(|e| format!("manifest line {line}: {e}"))?;
        let (line, tok) = opt("query_prefix_hex")?;
        let query_prefix = decode_hex_utf8(tok, "query_prefix")
            .map_err(|e| format!("manifest line {line}: {e}"))?;
        let (line, tok) = one("n_ref_bins")?;
        let n_ref_bins: usize = parse_at(tok, line, "n_ref_bins")?;
        let (line, tok) = one("n_query_bins")?;
        let n_query_bins: usize = parse_at(tok, line, "n_query_bins")?;
        let (line, tok) = one("n_units")?;
        let n_units: usize = parse_at(tok, line, "n_units")?;

        if n_ref_bins != reference_bins.len() {
            return Err(format!(
                "n_ref_bins {n_ref_bins} != {}",
                reference_bins.len()
            ));
        }
        if n_query_bins != query_bins.len() {
            return Err(format!(
                "n_query_bins {n_query_bins} != {}",
                query_bins.len()
            ));
        }
        if n_units != units.len() {
            return Err(format!("n_units {n_units} != {}", units.len()));
        }
        if max_hits == 0 {
            return Err("manifest max_hits must be the resolved cap, not 0".into());
        }
        for (name, v) in [
            ("hsp_blocks", hsp_blocks),
            ("step", step),
            ("wga_chunk_size", wga_chunk_size),
            ("lastz_interval_size", lastz_interval_size),
        ] {
            if v == 0 {
                return Err(format!("manifest {name} must be positive"));
            }
        }

        let plan = Plan {
            reference_bins,
            query_bins,
            units,
        };
        validate_topology(&plan)?;
        Ok(PlanManifest {
            version,
            hspz_version,
            features,
            max_hits,
            hsp_blocks,
            seed,
            step,
            transitions,
            xdrop,
            hspthresh,
            noentropy,
            wga_chunk_size,
            lastz_interval_size,
            kegalign_bins,
            seq_block_size,
            query_block_size,
            ref_hash,
            qry_hash,
            executable_hash,
            sub_mat,
            strand,
            target_prefix,
            query_prefix,
            plan,
        })
    }

    /// Input files on this node must be the same records the planner hashed.
    pub fn check_inputs(
        &self,
        ref_records: &[(String, Vec<u8>)],
        qry_records: &[(String, Vec<u8>)],
    ) -> Result<(), String> {
        let rh = records_hash(ref_records);
        let qh = records_hash(qry_records);
        if rh != self.ref_hash {
            return Err(format!(
                "manifest ref_hash {:016x} != this node's {:016x}",
                self.ref_hash, rh
            ));
        }
        if qh != self.qry_hash {
            return Err(format!(
                "manifest qry_hash {:016x} != this node's {:016x}",
                self.qry_hash, qh
            ));
        }
        Ok(())
    }

    pub fn check_software(&self) -> Result<(), String> {
        let ver = env!("CARGO_PKG_VERSION");
        if self.hspz_version != ver {
            return Err(format!(
                "manifest hspz_version {} != this binary {ver}",
                self.hspz_version
            ));
        }
        let feat = compiled_features();
        if self.features != feat {
            return Err(format!(
                "manifest features '{}' != this binary '{feat}'",
                self.features
            ));
        }
        // A byte-identical executable is the ceiling here: replaying across independently
        // rebuilt artifacts would need source/toolchain identity instead.
        let got = executable_hash()?;
        if self.executable_hash != got {
            return Err(format!(
                "manifest executable_hash {:016x} != this binary {:016x}",
                self.executable_hash, got
            ));
        }
        Ok(())
    }

    /// Fail rather than shrink bins or the hit cap. The cap is load-bearing.
    /// Uses this manifest's `step`, `wga_chunk_size`, `transitions`, and `max_hits`.
    pub fn check_fit(&self, budget_bytes: u64, kmer_size: usize) -> Result<u64, String> {
        validate_topology(&self.plan)?;
        let worst = worst_unit_bytes(
            &self.plan,
            kmer_size,
            self.step,
            self.max_hits,
            self.wga_chunk_size,
            self.transitions,
        )?;
        if worst > budget_bytes {
            return Err(format!(
                "frozen plan needs {:.1} GB per work unit, only {:.1} GB free; \
                 refusing to replan or lower max_hits ({})",
                worst as f64 / 1e9,
                budget_bytes as f64 / 1e9,
                self.max_hits
            ));
        }
        Ok(worst)
    }

    /// Topology, then each side's record ids are a permutation of `record_meta`
    /// ids with matching `total_bp` and a packed length that fits `u32`/`usize`.
    pub fn validate_records(
        &self,
        reference: &[RecordMeta],
        query: &[RecordMeta],
    ) -> Result<(), String> {
        validate_topology(&self.plan)?;
        validate_side_records("reference", &self.plan.reference_bins, reference)?;
        validate_side_records("query", &self.plan.query_bins, query)?;
        Ok(())
    }
}

pub(crate) fn join_csv(xs: &[impl std::fmt::Display]) -> String {
    xs.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn one_token<'a>(rest: &'a str, line: usize, name: &str) -> Result<&'a str, String> {
    let mut it = rest.split_whitespace();
    match (it.next(), it.next()) {
        (Some(tok), None) => Ok(tok),
        (None, _) => Err(format!("manifest line {line}: missing {name}")),
        _ => Err(format!(
            "manifest line {line}: trailing tokens after {name}"
        )),
    }
}

fn opt_token<'a>(rest: &'a str, line: usize, name: &str) -> Result<&'a str, String> {
    let mut it = rest.split_whitespace();
    match (it.next(), it.next()) {
        (None, _) => Ok(""),
        (Some(tok), None) => Ok(tok),
        _ => Err(format!(
            "manifest line {line}: trailing tokens after {name}"
        )),
    }
}

fn parse_at<T: std::str::FromStr>(tok: &str, line: usize, name: &str) -> Result<T, String> {
    tok.parse::<T>()
        .map_err(|_| format!("manifest line {line}: bad {name}"))
}

fn parse_flag(tok: &str, line: usize, name: &str) -> Result<bool, String> {
    match tok {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err(format!("manifest line {line}: {name} must be 0 or 1")),
    }
}

fn parse_hex_u64(tok: &str, line: usize, name: &str) -> Result<u64, String> {
    if tok.len() != 16 || !tok.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("manifest line {line}: bad {name}"));
    }
    u64::from_str_radix(tok, 16).map_err(|_| format!("manifest line {line}: bad {name}"))
}

fn parse_ids(s: &str) -> Result<Vec<u32>, String> {
    if s.is_empty() {
        return Ok(Vec::new());
    }
    s.split(',')
        .map(|p| p.parse::<u32>().map_err(|_| format!("bad record id '{p}'")))
        .collect()
}

fn parse_sub_mat(s: &str) -> Result<Vec<i32>, String> {
    let v: Result<Vec<i32>, String> = s
        .split(',')
        .map(|p| {
            p.parse::<i32>()
                .map_err(|_| format!("bad sub_mat entry '{p}'"))
        })
        .collect();
    let v = v?;
    if v.len() != 64 {
        return Err(format!("sub_mat must have 64 entries, got {}", v.len()));
    }
    Ok(v)
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}

fn decode_hex_utf8(s: &str, what: &str) -> Result<String, String> {
    if !s.len().is_multiple_of(2) {
        return Err(format!("odd {what} hex length"));
    }
    let mut bytes = Vec::with_capacity(s.len() / 2);
    let raw = s.as_bytes();
    let mut i = 0;
    while i < raw.len() {
        let hi = hex_digit(raw[i]).ok_or_else(|| format!("invalid {what} hex"))?;
        let lo = hex_digit(raw[i + 1]).ok_or_else(|| format!("invalid {what} hex"))?;
        bytes.push((hi << 4) | lo);
        i += 2;
    }
    String::from_utf8(bytes).map_err(|_| format!("invalid UTF-8 in {what}"))
}

fn hex_digit(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn validate_topology(plan: &Plan) -> Result<(), String> {
    validate_side_bins("reference", &plan.reference_bins)?;
    validate_side_bins("query", &plan.query_bins)?;
    let r = plan.reference_bins.len();
    let q = plan.query_bins.len();
    let expect = match r.checked_mul(q) {
        Some(n) if n <= u32::MAX as usize => n,
        _ => return Err(format!("unit count {r}*{q} exceeds u32::MAX")),
    };
    if plan.units.len() != expect {
        return Err(format!("unit count {} != {r}*{q}", plan.units.len()));
    }
    let mut pairs = std::collections::HashSet::with_capacity(expect);
    for (i, u) in plan.units.iter().enumerate() {
        if u.ordinal != i as u32 {
            return Err(format!("unit {i}: ordinal {} != index", u.ordinal));
        }
        if (u.reference_bin as usize) >= r {
            return Err(format!(
                "unit {i}: reference_bin {} out of range",
                u.reference_bin
            ));
        }
        if (u.query_bin as usize) >= q {
            return Err(format!("unit {i}: query_bin {} out of range", u.query_bin));
        }
        if !pairs.insert((u.reference_bin, u.query_bin)) {
            return Err(format!(
                "unit {i}: duplicate pair {} x {}",
                u.reference_bin, u.query_bin
            ));
        }
    }
    Ok(())
}

fn validate_side_bins(side: &str, bins: &[Bin]) -> Result<(), String> {
    if bins.len() > u32::MAX as usize {
        return Err(format!("{side} bin count exceeds u32::MAX"));
    }
    let mut seen = std::collections::HashSet::new();
    for (i, b) in bins.iter().enumerate() {
        if b.id != i as u32 {
            return Err(format!("{side} bin {i}: id {} != index", b.id));
        }
        if b.record_ids.is_empty() {
            return Err(format!("{side} bin {i}: empty record_ids"));
        }
        packed_bin_len(side, b)?;
        for &id in &b.record_ids {
            if !seen.insert(id) {
                return Err(format!("{side} bin {i}: duplicate record id {id}"));
            }
        }
    }
    Ok(())
}

fn validate_side_records(side: &str, bins: &[Bin], records: &[RecordMeta]) -> Result<(), String> {
    if records.len() > u32::MAX as usize {
        return Err(format!("{side} record count exceeds u32::MAX"));
    }
    for (i, rec) in records.iter().enumerate() {
        if rec.id != i as u32 {
            return Err(format!("{side} metadata id {} != index {i}", rec.id));
        }
    }
    let n = records.len();
    let mut seen = vec![false; n];
    for b in bins {
        for &id in &b.record_ids {
            let idx = id as usize;
            if idx >= n {
                return Err(format!("{side} bin {}: record id {id} out of range", b.id));
            }
            if seen[idx] {
                return Err(format!("{side} bin {}: duplicate record id {id}", b.id));
            }
            seen[idx] = true;
        }
    }
    if let Some(i) = seen.iter().position(|&ok| !ok) {
        return Err(format!("{side} missing record {i}"));
    }
    for b in bins {
        let mut sum = 0u64;
        for &id in &b.record_ids {
            sum = sum
                .checked_add(records[id as usize].len)
                .ok_or_else(|| format!("{side} bin {}: total_bp overflow", b.id))?;
        }
        if sum != b.total_bp {
            return Err(format!(
                "{side} bin {}: total_bp {} != record length sum {sum}",
                b.id, b.total_bp
            ));
        }
        packed_bin_len(side, b)?;
    }
    Ok(())
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
    /// Mode A: hspz must reproduce KegAlign's block membership exactly, or the two
    /// tools dedup in different scopes and their outputs differ before any kernel
    /// runs. The rule is sequential fill in input order, closing a block once it is
    /// *over* target — so blocks overshoot and the last one may be short.
    /// Bases are addressed with u32, so a bin whose packed length does not fit
    /// is an error (`hspz index` calls this before building).
    #[test]
    fn packed_bin_length_beyond_u32_is_an_error() {
        use super::{Bin, packed_bin_len};
        let fits = Bin {
            id: 0,
            record_ids: vec![0, 1],
            total_bp: u64::from(u32::MAX) - 1,
        };
        assert_eq!(packed_bin_len("reference", &fits), Ok(u64::from(u32::MAX)));
        let over = Bin {
            id: 3,
            record_ids: vec![0],
            total_bp: u64::from(u32::MAX) + 1,
        };
        let err = packed_bin_len("reference", &over).unwrap_err();
        assert!(
            err.contains("reference bin 3") && err.contains("does not fit u32"),
            "{err}"
        );
    }

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

    /// The assignment must be balanced, deterministic, and a partition — no
    /// bin run twice (duplicate output) and none dropped (missing alignments).
    /// Round 80: `--query-block-size` is a *query-only* lever. The reference layout and
    /// therefore `assign_bins` must not move when it changes, and an unset flag (which the
    /// CLI turns into the reference target) must reproduce the old plan exactly.
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

    /// The packing invariant: packing a bin must equal packing those same records as
    /// an entire input, byte for byte. With the shared packer this is a
    /// regression test rather than a two-implementation equivalence proof.
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

    const TEST_C: u32 = 250_000;

    fn bytes(
        ref_bp: u64,
        query_bp: u64,
        k: usize,
        step: u32,
        h: u32,
        c: u32,
        transitions: bool,
    ) -> u64 {
        unit_device_bytes(ref_bp, query_bp, k, step, h, c, transitions).unwrap()
    }

    #[test]
    fn preflight_shrinks_bins_until_the_plan_fits() {
        // Two 200 Mbp records with a budget that only admits one at a time.
        let r = recs(&[200_000_000, 200_000_000]);
        let q = recs(&[10_000_000]);
        let budget = bytes(200_000_000, 10_000_000, 12, 1, 16_711_680, TEST_C, true) + 1;
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
            TEST_C,
            true,
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
            TEST_C,
            true,
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
        let one_gbp = bytes(1_000_000_000, 0, 12, 1, 0, TEST_C, true);
        assert!(
            one_gbp > 4_000_000_000,
            "pos_table must be 4 B/bp: {one_gbp}"
        );
        // --step 3 indexes a third of the positions: 4 B/bp over 1 Gbp falls from
        // 4 GB to 1.33 GB. Assert that saving directly rather than a ratio against
        // the whole estimate, which also folds in the reference and headroom terms
        // that a future reference-representation change could move without
        // touching pos_table.
        let strided = bytes(1_000_000_000, 0, 12, 3, 0, TEST_C, true);
        assert!(
            one_gbp - strided > 2_600_000_000,
            "stride must reduce pos_table: {one_gbp} -> {strided}"
        );
    }

    #[test]
    fn bin_records_n_matches_target_count() {
        let r = recs(&[100, 90, 80, 70, 60, 50]);
        let from_target = bin_records(&r, 150);
        let from_n = bin_records_n(&r, from_target.len());
        assert_eq!(from_target, from_n);
        assert_eq!(bin_records_n(&r, 1).len(), 1);
        assert_eq!(bin_records_n(&r, 6).len(), 6);
        assert_eq!(layout_n_bins(450, 150, 6), 3);
    }

    fn auto_test_ctx() -> AutoCtx {
        AutoCtx {
            kmer_size: 12,
            step: 1,
            max_hits: 1000,
            wga_chunk_size: 250_000,
            transitions: true,
            threads: 4,
            max_seeds: 10_000,
        }
    }

    /// `-B 0` rank-2 rule: `R = min(n_records, ceil(R_def / W) * W)` — the
    /// smallest multiple of W that does not coarsen the default layout.
    #[test]
    fn auto_ref_bin_count_follows_the_rank2_rule() {
        // 7 x 499 Mbp + 5 x 1 bp: total 3,493,000,005 bp, so R_def = 7 over 12
        // records against the 500 Mbp default.
        let mut lens = vec![499_000_000u64; 7];
        lens.extend_from_slice(&[1; 5]);
        let r = recs(&lens);
        assert_eq!(r.len(), 12);
        assert_eq!(layout_n_bins(3_493_000_005, 500_000_000, 12), 7);
        assert_eq!(auto_ref_bin_count(12, 7, 2), 8);
        assert_eq!(auto_ref_bin_count(12, 7, 4), 8);
        assert_eq!(auto_ref_bin_count(12, 7, 3), 9);
        assert_eq!(auto_ref_bin_count(12, 7, 8), 8);
        assert_eq!(auto_ref_bin_count(5, 7, 4), 5, "one bin per record caps R");
        assert_eq!(auto_ref_bin_count(0, 7, 4), 0);
        // The full path agrees, and its printed targets rebuild the same plan.
        let q = recs(&[10, 10]);
        let auto = auto_layout(&r, &q, 2, 2, u64::MAX, None, None, &auto_test_ctx()).unwrap();
        assert_eq!(auto.plan.reference_bins.len(), 8);
        assert_eq!(
            plan_with(&r, &q, auto.seq_target, auto.query_target, false),
            auto.plan
        );
    }

    /// Resolving 0 at W=1 is the default plan exactly — no Q=1 attempt, no new
    /// bins — which is what makes one-worker `-B 0` byte-identical to today.
    #[test]
    fn auto_layout_at_w1_is_the_default_plan() {
        let r = recs(&[250, 240, 200, 190, 180, 170, 160]);
        let q = recs(&[195, 180, 160, 155, 150, 145, 60]);
        let ctx = auto_test_ctx();
        let auto = auto_layout(&r, &q, 1, 1, u64::MAX, None, None, &ctx).unwrap();
        assert_eq!(
            auto.plan,
            plan_with(&r, &q, 500_000_000, 500_000_000, false)
        );
        assert_eq!(
            (auto.seq_target, auto.query_target),
            (500_000_000, 500_000_000)
        );
        // An explicit query target is honoured as given, still with no Q=1.
        let auto = auto_layout(&r, &q, 1, 1, u64::MAX, None, Some(10_000), &ctx).unwrap();
        assert_eq!(auto.plan, plan_with(&r, &q, 500_000_000, 10_000, false));
        assert_eq!(auto.query_target, 10_000);
    }

    /// Q=1 needs *both* budgets; either failure falls back to the default
    /// query blocks, and an explicit query target is honoured as given.
    #[test]
    fn auto_layout_q1_needs_both_budgets() {
        // 2 x 100 bp reference over 2 workers: R_def = 1, so R = 2.
        let r = recs(&[100, 100]);
        // 2 x 600 Mbp query: Q=1 holds 1.2 Gbp where the default holds 600
        // Mbp, so a budget between the two estimates separates the paths.
        let q = recs(&[600_000_000, 600_000_000]);
        let ctx = auto_test_ctx();
        let wb = |p: &Plan| {
            worst_unit_bytes(
                p,
                ctx.kmer_size,
                ctx.step,
                ctx.max_hits,
                ctx.wga_chunk_size,
                ctx.transitions,
            )
            .unwrap()
        };
        let q1 = from_bins(bin_records_n(&r, 2), bin_records_n(&q, 1));
        let defq = from_bins(bin_records_n(&r, 2), bin_records(&q, 500_000_000));
        assert_eq!(defq.query_bins.len(), 2);
        let (w1, w0) = (wb(&q1), wb(&defq));
        assert!(
            w1 > w0,
            "fixture must separate Q=1 ({w1}) from default-Q ({w0})"
        );

        // Both budgets admit: Q=1, and the printed targets rebuild the plan
        // through the normal planner.
        let one = auto_layout(&r, &q, 2, 2, w1, None, None, &ctx).unwrap();
        assert_eq!(one.plan, q1);
        assert_eq!(one.query_target, 1_200_000_000);
        assert_eq!(
            plan_with(&r, &q, one.seq_target, one.query_target, false),
            one.plan
        );
        assert_eq!(
            auto_layout_line(2, one.seq_target, one.query_target, &one.plan),
            "layout: auto W=2 -> --seq-block-size 100 --query-block-size 1200000000 \
             (R=2 Q=1, 2 units, owners 1+1)"
        );

        // Device budget admits the fallback but not Q=1: default query blocks.
        let poor = auto_layout(&r, &q, 2, 2, w0, None, None, &ctx).unwrap();
        assert_eq!(poor.plan, defq);
        assert_eq!(poor.query_target, 500_000_000);
        assert_eq!(
            plan_with(&r, &q, poor.seq_target, poor.query_target, false),
            poor.plan
        );

        // Host budget rejects Q=1 under ample device memory: same fallback.
        let cramped = auto_layout(&r, &q, 2, 2, w1, Some(1), None, &ctx).unwrap();
        assert_eq!(cramped.plan, defq);

        // Explicit query target is honoured as given, fit permitting.
        let expl = auto_layout(&r, &q, 2, 2, w0, None, Some(500_000_000), &ctx).unwrap();
        assert_eq!(expl.plan, defq);
        assert_eq!(expl.query_target, 500_000_000);
    }

    /// The manifest stores the resolved targets (never 0) and round-trips.
    #[test]
    fn auto_manifest_round_trips_resolved_targets() {
        let r = recs(&[100, 100]);
        let q = recs(&[50, 50]);
        let auto = auto_layout(&r, &q, 2, 2, u64::MAX, None, None, &auto_test_ctx()).unwrap();
        assert_eq!(auto.plan.reference_bins.len(), 2);
        assert_eq!(auto.plan.query_bins.len(), 1);
        let mut m = manifest_fixture(auto.plan.clone());
        m.seq_block_size = auto.seq_target;
        m.query_block_size = auto.query_target;
        assert_ne!(m.seq_block_size, 0);
        assert_ne!(m.query_block_size, 0);
        let back = PlanManifest::read(&manifest_text(&m)).unwrap();
        assert_eq!(back, m);
        assert_eq!(
            (back.seq_block_size, back.query_block_size),
            (auto.seq_target, auto.query_target)
        );
    }

    fn manifest_fixture(plan: Plan) -> PlanManifest {
        PlanManifest {
            version: PlanManifest::FORMAT,
            hspz_version: env!("CARGO_PKG_VERSION").into(),
            features: compiled_features(),
            executable_hash: executable_hash().unwrap(),
            max_hits: 16_711_680,
            hsp_blocks: 16384,
            seed: "12of19".into(),
            step: 1,
            transitions: true,
            xdrop: 910,
            hspthresh: 3000,
            noentropy: false,
            wga_chunk_size: 250_000,
            lastz_interval_size: 10_000_000,
            kegalign_bins: false,
            seq_block_size: 200,
            query_block_size: 200,
            ref_hash: 1,
            qry_hash: 2,
            sub_mat: vec![0; 64],
            strand: "both".into(),
            target_prefix: String::new(),
            query_prefix: String::new(),
            plan,
        }
    }

    fn manifest_text(m: &PlanManifest) -> String {
        let mut buf = Vec::new();
        m.write(&mut buf).unwrap();
        String::from_utf8(buf).unwrap()
    }

    fn drop_key(text: &str, key: &str) -> String {
        text.lines()
            .filter(|l| l.split_whitespace().next() != Some(key))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn set_key(text: &str, key: &str, rest: &str) -> String {
        text.lines()
            .map(|l| {
                if l.split_whitespace().next() == Some(key) {
                    format!("{key} {rest}")
                } else {
                    l.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn plan_manifest_round_trips_and_rejects_a_zero_cap() {
        let r = recs(&[100, 100, 100, 100]);
        let q = recs(&[50, 50]);
        let p = plan(&r, &q, 200);
        let m = manifest_fixture(p.clone());
        let text = manifest_text(&m);
        let back = PlanManifest::read(&text).unwrap();
        assert_eq!(back, m);
        assert_eq!(back.plan, p);
        assert_eq!(back.max_hits, 16_711_680);
        m.check_software().unwrap();
        m.validate_records(&r, &q).unwrap();
        let worst = m.check_fit(u64::MAX, 12).unwrap();
        assert_eq!(
            worst,
            worst_unit_bytes(&p, 12, 1, 16_711_680, m.wga_chunk_size, m.transitions).unwrap()
        );
        assert!(
            m.check_fit(1, 12)
                .unwrap_err()
                .contains("refusing to replan")
        );

        let zero = text.replace("max_hits 16711680", "max_hits 0");
        assert!(
            PlanManifest::read(&zero)
                .unwrap_err()
                .contains("resolved cap")
        );
    }

    #[test]
    fn manifest_strict_parse_rejects_malformed_scalars() {
        let p = plan(&recs(&[100, 100, 100, 100]), &recs(&[50, 50]), 200);
        let base = manifest_text(&manifest_fixture(p));
        PlanManifest::read(&set_key(&base, "features", "")).unwrap();
        let cases: &[(&str, String, &str)] = &[
            ("legacy", set_key(&base, "hspz-manifest", "1"), "format"),
            ("dup", format!("{base}features x\n"), "duplicate"),
            ("unknown", format!("{base}nope 1\n"), "unknown"),
            (
                "dup-header",
                format!("{base}hspz-manifest 2\n"),
                "duplicate",
            ),
            ("missing", drop_key(&base, "seed"), "missing"),
            ("missing-features", drop_key(&base, "features"), "missing"),
            ("bool", set_key(&base, "transitions", "true"), "0 or 1"),
            (
                "ids",
                base.replace("bin Q 0 100 0,1", "bin Q 0 100 0,x"),
                "record id",
            ),
            (
                "empty-id",
                base.replace("bin Q 0 100 0,1", "bin Q 0 100 0,"),
                "record id",
            ),
            ("counts", set_key(&base, "n_units", "99"), "n_units"),
            ("trail", set_key(&base, "step", "1 2"), "trailing"),
            ("strand", set_key(&base, "strand", "forward"), "strand"),
        ];
        for (name, text, needle) in cases {
            let err = PlanManifest::read(text).expect_err(name);
            assert!(err.contains(needle), "{name}: expected {needle:?} in {err}");
        }
    }

    #[test]
    fn manifest_topology_and_record_membership() {
        let r = recs(&[10, 20]);
        let q = recs(&[5, 6]);
        let p = plan(&r, &q, 1);
        let m0 = manifest_fixture(p);
        m0.validate_records(&r, &q).unwrap();

        let empty: Vec<RecordMeta> = vec![];
        manifest_fixture(plan(&empty, &q, 1))
            .validate_records(&empty, &q)
            .unwrap();
        assert!(
            manifest_fixture(plan(&empty, &q, 1))
                .validate_records(&r, &q)
                .unwrap_err()
                .contains("reference")
        );

        let mut alt = m0.clone();
        alt.plan.units.swap(0, 1);
        alt.plan.units[0].ordinal = 0;
        alt.plan.units[1].ordinal = 1;
        alt.validate_records(&r, &q).unwrap();

        let check = |mutate: &dyn Fn(&mut PlanManifest), needle: &str| {
            let mut m = m0.clone();
            mutate(&mut m);
            let err = m.validate_records(&r, &q).expect_err(needle);
            assert!(err.contains(needle), "expected {needle:?} in {err}");
        };
        check(&|m| m.plan.reference_bins[0].id = 9, "id");
        check(&|m| m.plan.reference_bins[0].record_ids.clear(), "empty");
        check(&|m| m.plan.units.swap(0, 1), "ordinal");
        check(
            &|m| {
                m.plan.units[1].reference_bin = m.plan.units[0].reference_bin;
                m.plan.units[1].query_bin = m.plan.units[0].query_bin;
            },
            "duplicate",
        );
        check(
            &|m| {
                m.plan.units.pop();
            },
            "unit count",
        );
        check(
            &|m| m.plan.reference_bins[0].record_ids = vec![99],
            "out of range",
        );
        check(&|m| m.plan.reference_bins[0].total_bp += 1, "total_bp");

        let r1 = recs(&[10, 20]);
        let q1 = recs(&[5]);
        let mut miss = manifest_fixture(plan(&r1, &q1, 1000));
        miss.plan.reference_bins[0].record_ids = vec![0];
        miss.plan.reference_bins[0].total_bp = 10;
        assert!(
            miss.validate_records(&r1, &q1)
                .unwrap_err()
                .contains("missing")
        );

        let big = vec![RecordMeta {
            id: 0,
            name: "big".into(),
            len: u32::MAX as u64 + 1,
            ordinal: 0,
        }];
        let q_small = recs(&[1]);
        let overflow = manifest_fixture(Plan {
            reference_bins: vec![Bin {
                id: 0,
                record_ids: vec![0],
                total_bp: u32::MAX as u64 + 1,
            }],
            query_bins: vec![Bin {
                id: 0,
                record_ids: vec![0],
                total_bp: 1,
            }],
            units: vec![WorkUnit {
                ordinal: 0,
                reference_bin: 0,
                query_bin: 0,
            }],
        });
        let err = overflow.validate_records(&big, &q_small).unwrap_err();
        assert!(err.contains("packed") && err.contains("reference"), "{err}");
    }

    #[test]
    fn manifest_matrix_and_prefix_roundtrip() {
        let mut m = manifest_fixture(plan(&recs(&[10]), &recs(&[10]), 100));
        m.sub_mat = (0..64).map(|i| i - 40).collect();
        m.target_prefix = " \t\n\u{1b}héllo".into();
        m.query_prefix = "世界\0".into();
        let text = manifest_text(&m);
        let back = PlanManifest::read(&text).unwrap();
        assert_eq!(back.sub_mat, m.sub_mat);
        assert_eq!(back.sub_mat.len(), 64);
        assert_eq!(back.target_prefix, m.target_prefix);
        assert_eq!(back.query_prefix, m.query_prefix);

        for (key, rest, needle) in [
            ("sub_mat", "1,2,3", "64"),
            ("target_prefix_hex", "ff", "UTF-8"),
            ("query_prefix_hex", "zz", "hex"),
            ("target_prefix_hex", "abc", "odd"),
        ] {
            let err = PlanManifest::read(&set_key(&text, key, rest)).unwrap_err();
            assert!(err.contains(needle), "{key} {rest}: {err}");
        }
    }

    #[test]
    fn manifest_software_identity_mismatch() {
        let m0 = manifest_fixture(plan(&recs(&[10]), &recs(&[10]), 100));
        m0.check_software().unwrap();

        let mut m = m0.clone();
        m.hspz_version = "0.0.0".into();
        assert!(m.check_software().unwrap_err().contains("hspz_version"));
        m = m0.clone();
        m.features = format!("x{}", m.features);
        assert!(m.check_software().unwrap_err().contains("features"));
        m = m0.clone();
        m.executable_hash ^= 1;
        assert!(m.check_software().unwrap_err().contains("executable_hash"));

        let feat = compiled_features();
        let known = [
            "counters",
            "nvidia-find-num-unchecked",
            "nvidia-uninit-seed-buffers",
            "device-seeds",
            "device-seeds-check",
            "warp-score-gate",
            "dense-anchors",
            "find-hits-warp",
            "ref-loc-buckets",
            "left-pair-tile",
            "simd-prelude",
        ];
        for f in feat.split(',').filter(|s| !s.is_empty()) {
            assert!(known.contains(&f), "unexpected feature {f}");
        }
        #[cfg(feature = "simd-prelude")]
        assert!(feat.split(',').any(|f| f == "simd-prelude"));
        // A default feature is part of the identity a manifest records.
        #[cfg(feature = "ref-loc-buckets")]
        assert!(feat.split(',').any(|f| f == "ref-loc-buckets"));
        #[cfg(not(feature = "counters"))]
        assert!(!feat.split(',').any(|f| f == "counters"));
    }

    #[test]
    fn manifest_fit_and_positive_caps() {
        let r = recs(&[100, 100, 100, 100]);
        let q = recs(&[50, 50]);
        let p = plan(&r, &q, 200);
        let m = manifest_fixture(p.clone());
        let worst = m.check_fit(u64::MAX, 12).unwrap();
        assert_eq!(
            worst,
            worst_unit_bytes(&p, 12, 1, m.max_hits, m.wga_chunk_size, m.transitions).unwrap()
        );
        assert!(
            m.check_fit(1, 12)
                .unwrap_err()
                .contains("refusing to replan")
        );

        let mut bad = m.clone();
        bad.plan.units[0].reference_bin = 99;
        assert!(bad.check_fit(u64::MAX, 12).is_err());

        let text = manifest_text(&m);
        for (key, needle) in [
            ("max_hits", "resolved cap"),
            ("hsp_blocks", "positive"),
            ("step", "positive"),
            ("wga_chunk_size", "positive"),
            ("lastz_interval_size", "positive"),
        ] {
            let err = PlanManifest::read(&set_key(&text, key, "0")).unwrap_err();
            assert!(err.contains(needle), "{key}: {err}");
        }
    }

    #[test]
    fn runtime_seed_staging_tracks_chunk_shape_transitions_not_step() {
        let k14 = 14usize;
        let c1m = 1_000_000u32;
        let h = 0u32;
        let with = bytes(0, 0, k14, 1, h, c1m, true);
        let stepped = bytes(0, 0, k14, 3, h, c1m, true);
        assert_eq!(
            with, stepped,
            "seed staging must not depend on step when ref_bp=0"
        );
        let no_tr = bytes(0, 0, k14, 1, h, c1m, false);
        assert!(with > no_tr, "transitions grow S: {with} vs {no_tr}");
        let default_c = bytes(0, 0, k14, 1, h, TEST_C, true);
        assert!(with > default_c, "C=1M vs 250k: {with} vs {default_c}");
        let k12 = bytes(0, 0, 12, 1, h, c1m, true);
        assert_ne!(with, k12, "k=14 vs k=12 must change index and S");

        let err = |r: Result<u64, String>, needle: &str| {
            let e = r.expect_err(needle);
            assert!(e.contains(needle), "expected {needle:?} in {e}");
        };
        err(unit_device_bytes(0, 0, 12, 0, 0, TEST_C, true), "step");
        err(unit_device_bytes(0, 0, 12, 1, 0, 0, true), "wga_chunk_size");
        err(unit_device_bytes(0, 0, 16, 1, 0, TEST_C, true), "k=");
        err(unit_device_bytes(0, 0, 3, 1, 0, TEST_C, true), "k=");
        err(
            unit_device_bytes(0, 0, 15, 1, 0, u32::MAX, true),
            "u32::MAX",
        );
        assert!(unit_device_bytes(u64::MAX, u64::MAX, 15, 1, u32::MAX, TEST_C, true).is_err());
    }

    #[test]
    fn packed_query_overlap_and_adaptive_split() {
        let r = recs(&[100]);
        let q1 = recs(&[1_000_000]);
        let q2 = recs(&[1_000_000, 1_000_000]);
        let one_q = plan(&r, &q1, 10_000_000);
        let two_q = plan_with(&r, &q2, 10_000_000, 1, false);
        assert_eq!(one_q.query_bins.len(), 1);
        assert_eq!(two_q.query_bins.len(), 2);
        let w1 = worst_unit_bytes(&one_q, 12, 1, 0, TEST_C, true).unwrap();
        let w2 = worst_unit_bytes(&two_q, 12, 1, 0, TEST_C, true).unwrap();
        assert_eq!(w2 - w1, 1_000_000, "3Qmax vs 2Qmax differs by Qmax");

        let two_rec = Plan {
            reference_bins: vec![Bin {
                id: 0,
                record_ids: vec![0, 1],
                total_bp: 100,
            }],
            query_bins: vec![Bin {
                id: 0,
                record_ids: vec![0],
                total_bp: 10,
            }],
            units: vec![WorkUnit {
                ordinal: 0,
                reference_bin: 0,
                query_bin: 0,
            }],
        };
        let packed = worst_unit_bytes(&two_rec, 12, 1, 0, TEST_C, false).unwrap();
        let raw = bytes(100, 10, 12, 1, 0, TEST_C, false);
        let with_sep = bytes(101, 10, 12, 1, 0, TEST_C, false);
        assert_eq!(packed, with_sep);
        assert!(packed > raw, "separator must add packed length");

        let huge_r = recs(&[2_200_000_000, 2_200_000_000]);
        let tiny_q = recs(&[1_000]);
        let one = bytes(2_200_000_000, 1_000, 12, 1, 0, TEST_C, true);
        let (p, worst) = plan_within_budget(
            &huge_r,
            &tiny_q,
            4_400_000_000,
            4_400_000_000,
            one + 1,
            12,
            1,
            0,
            false,
            TEST_C,
            true,
        )
        .expect("splittable packed overflow must shrink");
        assert_eq!(p.reference_bins.len(), 2);
        assert!(worst <= one + 1);

        let err = plan_within_budget(
            &recs(&[2_200_000_000]),
            &tiny_q,
            u64::MAX,
            u64::MAX,
            one / 2,
            12,
            1,
            0,
            false,
            TEST_C,
            true,
        )
        .expect_err("atomic oversized record");
        assert!(err.contains("exceeds GPU capacity"), "{err}");

        let fit_one = bytes(200_000_000, 10_000_000, 12, 1, 0, TEST_C, true) + 1;
        let err = plan_within_budget(
            &recs(&[200_000_000, 200_000_000]),
            &recs(&[10_000_000]),
            400_000_000,
            400_000_000,
            fit_one,
            12,
            1,
            0,
            true,
            TEST_C,
            true,
        )
        .expect_err("matched-granularity must not shrink");
        assert!(err.contains("matched-granularity"), "{err}");

        let mut forged = manifest_fixture(plan(&recs(&[10]), &recs(&[10]), 100));
        forged.plan.reference_bins[0].total_bp = u64::MAX;
        assert!(
            forged.check_fit(u64::MAX, 12).is_err(),
            "huge total must fail even at u64::MAX"
        );
        forged.plan.reference_bins[0].total_bp = 10;
        forged.step = 0;
        assert!(forged.check_fit(u64::MAX, 12).is_err());
        forged.step = 1;
        assert!(forged.check_fit(u64::MAX, 99).is_err());
    }

    #[test]
    fn hit_scan_terms_and_frozen_cap() {
        let hit_b: u64 = if cfg!(feature = "dense-anchors") {
            49
        } else {
            36
        };
        // Dense keeps the survivor-offsets array alive through the done scan, so
        // its H-scan term is 12 B/ceil(H/256); non-dense is 8 (see the doc on
        // `unit_device_bytes`).
        let hit_scan_b: u64 = if cfg!(feature = "dense-anchors") {
            12
        } else {
            8
        };
        // `counters` keeps 2 x u64 per materialized hit alive through reduce.
        let counters_b: u64 = if cfg!(feature = "counters") { 16 } else { 0 };
        // `ref-loc-buckets` keeps the permuted anchor + its raw index alive
        // across the gate; its histogram is a second, coarser scan bucket.
        let bucket_b: u64 = if cfg!(feature = "ref-loc-buckets") {
            12
        } else {
            0
        };
        let delta = |h0: u32, h1: u32| {
            bytes(0, 0, 12, 1, h1, TEST_C, true) - bytes(0, 0, 12, 1, h0, TEST_C, true)
        };
        let scan = |h: u32| hit_scan_b * u64::from(h).div_ceil(256);
        let bscan = |h: u32| {
            if cfg!(feature = "ref-loc-buckets") {
                let counts = 32 * u64::from(h).div_ceil(256);
                4 * counts + 8 * counts.div_ceil(256)
            } else {
                0
            }
        };
        // Round 2 sorted bitmask flags: 1 bit per hit, so it steps once per 8.
        let bmask = |h: u32| {
            if cfg!(feature = "ref-loc-buckets") {
                u64::from(h).div_ceil(8)
            } else {
                0
            }
        };
        assert_eq!(
            delta(255, 256),
            hit_b + counters_b + bucket_b + bmask(256) - bmask(255),
            "same scan bucket: only hit term"
        );
        assert_eq!(
            delta(256, 257),
            hit_b + counters_b + bucket_b + scan(257) - scan(256) + bscan(257) - bscan(256)
                + bmask(257)
                - bmask(256),
            "crossing 256 must add scan increment"
        );
        assert_eq!(
            delta(0, 1),
            hit_b + counters_b + bucket_b + hit_scan_b + bscan(1) - bscan(0) + bmask(1) - bmask(0)
        );
        // Scoped coefficient pin: per-hit slope above scan buckets is hit + counters.
        let slope = (bytes(0, 0, 12, 1, 2000, TEST_C, true)
            - bytes(0, 0, 12, 1, 1000, TEST_C, true)
            - (scan(2000) - scan(1000))
            - (bscan(2000) - bscan(1000))
            - (bmask(2000) - bmask(1000)))
            / 1000;
        assert_eq!(slope, hit_b + counters_b + bucket_b);

        let cap = 16_711_680u32;
        let r = recs(&[100]);
        let q = recs(&[50]);
        let full = bytes(100, 50, 12, 1, cap, TEST_C, true);
        let zero = bytes(100, 50, 12, 1, 0, TEST_C, true);
        assert!(full > zero);
        let err = plan_within_budget(&r, &q, 200, 200, zero + 1, 12, 1, cap, false, TEST_C, true)
            .expect_err("must not lower max_hits to fit");
        assert!(err.contains("exceeds GPU capacity"), "{err}");

        let p = plan(&r, &q, 200);
        let mut m = manifest_fixture(p);
        let cap_before = m.max_hits;
        let a = m.check_fit(u64::MAX, 12).unwrap();
        m.wga_chunk_size = 1_000_000;
        let b = m.check_fit(u64::MAX, 12).unwrap();
        assert_ne!(a, b, "frozen fit must use manifest wga_chunk_size");
        assert_eq!(m.max_hits, cap_before);
        assert!(
            m.check_fit(1, 12)
                .unwrap_err()
                .contains("refusing to replan")
        );
        assert_eq!(m.max_hits, cap_before);
    }

    #[test]
    fn worker_device_budget_divides_shared_devices() {
        assert_eq!(worker_device_budget(1000, 1, 4), 1000);
        assert_eq!(worker_device_budget(1000, 4, 4), 1000);
        assert_eq!(worker_device_budget(1000, 5, 4), 500);
        assert_eq!(worker_device_budget(1000, 8, 4), 500);
        assert_eq!(worker_device_budget(900, 3, 1), 300);
        assert_eq!(worker_device_budget(1000, 0, 0), 1000);
    }

    /// rank5b CPU-only ownership contract (preparation, no shard runtime):
    /// reference bins over G*N workers via `assign_bins`, G workers per node;
    /// owned ordinals by actual refbin membership, exact partition.
    #[test]
    fn rank5b_reference_ownership_exact_partition() {
        let mut cases = 0u32;
        let (mut saw_gapped, mut saw_nonzero_start) = (false, false);
        let (mut saw_empty_worker, mut saw_empty_node, mut validated_alt) = (false, false, false);
        for n_ref in 1..=8usize {
            for n_qry in 1..=4usize {
                let r = recs(
                    &(0..n_ref)
                        .map(|i| 50 + ((i * 53 + n_qry * 7 + n_ref) % 3) as u64 * 50)
                        .collect::<Vec<_>>(),
                );
                let q = recs(
                    &(0..n_qry)
                        .map(|i| 40 + ((i * 29 + n_ref * 11) % 3) as u64 * 40)
                        .collect::<Vec<_>>(),
                );
                let p = plan_with(&r, &q, 150, 150, false);
                assert!(!p.reference_bins.is_empty() && !p.units.is_empty());
                for g in 1..=3usize {
                    for n_nodes in 1..=4usize {
                        cases += 1;
                        let mut units = p.units.clone();
                        if cases.is_multiple_of(2) {
                            units.reverse();
                            for (i, u) in units.iter_mut().enumerate() {
                                u.ordinal = i as u32;
                            }
                        }
                        let total = g * n_nodes;
                        let global = assign_bins(&p.reference_bins, total);
                        assert_eq!(global, assign_bins(&p.reference_bins, total));
                        assert!(global.iter().all(|w| w.windows(2).all(|w| w[0] < w[1])));
                        saw_empty_worker |= global.iter().any(|w| w.is_empty());
                        let mut node_bins: Vec<Vec<u32>> = global
                            .chunks(g)
                            .map(|c| c.iter().flat_map(|w| w.iter().copied()).collect())
                            .collect();
                        node_bins.iter_mut().for_each(|nb| nb.sort_unstable());
                        let nbins = p.reference_bins.len() as u32;
                        let mut all_bins: Vec<u32> =
                            node_bins.iter().flat_map(|v| v.iter().copied()).collect();
                        all_bins.sort_unstable();
                        assert_eq!(all_bins, (0..nbins).collect::<Vec<_>>());
                        let node_units: Vec<Vec<u32>> = node_bins
                            .iter()
                            .map(|b| {
                                units
                                    .iter()
                                    .filter(|u| b.contains(&u.reference_bin))
                                    .map(|u| u.ordinal)
                                    .collect()
                            })
                            .collect();
                        for seq in &node_units {
                            let mut sorted = seq.clone();
                            sorted.sort_unstable();
                            assert_eq!(*seq, sorted);
                            saw_gapped |= seq.windows(2).any(|w| w[1] != w[0] + 1);
                            saw_nonzero_start |= !seq.is_empty() && seq[0] != 0;
                        }
                        saw_empty_node |= node_units.iter().any(|s| s.is_empty());
                        let mut union: Vec<u32> =
                            node_units.iter().flat_map(|v| v.iter().copied()).collect();
                        union.sort_unstable();
                        assert_eq!(union, (0..units.len() as u32).collect::<Vec<_>>());
                        if cases.is_multiple_of(2) && units.len() > 1 && !validated_alt {
                            manifest_fixture(Plan {
                                reference_bins: p.reference_bins.clone(),
                                query_bins: p.query_bins.clone(),
                                units,
                            })
                            .validate_records(&r, &q)
                            .unwrap();
                            validated_alt = true;
                        }
                    }
                }
            }
        }
        assert_eq!(cases, 8 * 4 * 3 * 4);
        assert!(saw_gapped);
        assert!(saw_nonzero_start);
        assert!(saw_empty_worker);
        assert!(saw_empty_node);
        assert!(validated_alt);
    }

    #[test]
    fn max_hit_capacity_exact_boundary_and_brute_force_agreement() {
        let cost =
            |p: &super::Plan, h: u32| super::worst_unit_bytes(p, 12, 1, h, TEST_C, true).unwrap();
        // Multi-query plan charges the swap overlap (3 Qmax vs 2 Qmax).
        let p = super::plan_with(&recs(&[100]), &recs(&[50, 60]), 10_000, 1, false);
        assert_eq!(p.query_bins.len(), 2);
        let before = p.clone();
        let h = 100u32;
        assert_eq!(
            super::max_hit_capacity(&p, cost(&p, h), 12, 1, h, TEST_C, true).unwrap(),
            h
        );
        let richer = cost(&p, h + 25);
        assert_eq!(
            super::max_hit_capacity(&p, richer, 12, 1, h, TEST_C, true).unwrap(),
            h + 25
        );
        assert!(cost(&p, h + 26) > richer);
        // Scan-bucket edges plus a tiny brute-force window agree.
        for &edge in &[1u32, 255, 256, 257, 511, 512] {
            let budget = cost(&p, edge + 40);
            let got = super::max_hit_capacity(&p, budget, 12, 1, edge, TEST_C, true).unwrap();
            let mut want = edge;
            for c in edge..=edge + 80 {
                if cost(&p, c) <= budget {
                    want = c;
                }
            }
            assert_eq!(got, want, "edge {edge}");
        }
        let (lo, budget) = (10u32, cost(&p, 30));
        let mut want = lo;
        for c in lo..=60 {
            if cost(&p, c) <= budget {
                want = c;
            }
        }
        assert_eq!(
            super::max_hit_capacity(&p, budget, 12, 1, lo, TEST_C, true).unwrap(),
            want
        );
        assert_eq!(p, before, "helper must not mutate the plan");
    }

    #[test]
    fn max_hit_capacity_rejects_empty_and_bad_inputs() {
        let p = super::plan(&recs(&[100]), &recs(&[50, 60]), 200);
        let h = 100u32;
        let exact = super::worst_unit_bytes(&p, 12, 1, h, TEST_C, true).unwrap();
        let before = p.clone();
        assert!(super::max_hit_capacity(&p, exact - 1, 12, 1, h, TEST_C, true).is_err());
        assert!(super::max_hit_capacity(&p, exact, 12, 1, 0, TEST_C, true).is_err());
        assert!(super::max_hit_capacity(&p, exact, 12, 0, h, TEST_C, true).is_err());
        assert!(super::max_hit_capacity(&p, exact, 3, 1, h, TEST_C, true).is_err());
        assert!(super::max_hit_capacity(&p, exact, 12, 1, h, 0, true).is_err());
        assert_eq!(p, before);
        let empty = super::plan(&[], &recs(&[50]), 100);
        assert!(empty.reference_bins.is_empty());
        assert_eq!(
            super::max_hit_capacity(&empty, u64::MAX, 12, 1, h, TEST_C, true).unwrap(),
            h
        );
    }
}

/// Round 90: unit-level static partition (task.md Part 2).
#[cfg(test)]
mod unit_partition_tests {
    use super::{Bin, Plan, Visit, WorkUnit, unit_partition};

    fn toy_plan(ref_bps: &[u64], q: usize) -> Plan {
        let reference_bins = ref_bps
            .iter()
            .enumerate()
            .map(|(i, &bp)| Bin {
                id: i as u32,
                record_ids: Vec::new(),
                total_bp: bp,
            })
            .collect::<Vec<_>>();
        let query_bins = (0..q)
            .map(|i| Bin {
                id: i as u32,
                record_ids: Vec::new(),
                total_bp: 1,
            })
            .collect::<Vec<_>>();
        let mut units = Vec::new();
        let mut ordinal = 0u32;
        for r in &reference_bins {
            for qq in &query_bins {
                units.push(WorkUnit {
                    ordinal,
                    reference_bin: r.id,
                    query_bin: qq.id,
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

    /// Canonical whole-genome shape: descending bp order 3,4,2,1,0,6,5.
    fn canonical() -> Plan {
        let mut bps = [0u64; 7];
        bps[3] = 700;
        bps[4] = 600;
        bps[2] = 500;
        bps[1] = 400;
        bps[0] = 300;
        bps[6] = 200;
        bps[5] = 100;
        toy_plan(&bps, 6)
    }

    fn v(bin: usize, lo: usize, hi: usize) -> Visit {
        Visit {
            bin,
            queries: lo..hi,
        }
    }

    fn total_visits(part: &[Vec<Visit>]) -> usize {
        part.iter().map(Vec::len).sum()
    }

    /// Exact cover, quotas, contiguity and in-bin order for any partition.
    #[allow(clippy::needless_range_loop)]
    fn check_invariants(plan: &Plan, workers: usize, part: &[Vec<Visit>]) {
        let r = plan.reference_bins.len();
        let q = plan.query_bins.len();
        let n = plan.units.len();
        let w = workers.max(1);
        assert_eq!(part.len(), w);
        let (a, rem) = (n / w, n % w);
        let mut seen = vec![vec![false; q]; r.max(1)];
        let mut large = 0;
        for (x, visits) in part.iter().enumerate() {
            let mut units = 0;
            let mut bins: Vec<usize> = Vec::new();
            for t in visits {
                assert!(!t.queries.is_empty(), "worker {x}: empty visit");
                assert!(t.bin < r, "worker {x}: bin out of range");
                assert!(t.queries.end <= q, "worker {x}: range past Q");
                bins.push(t.bin);
                for qq in t.queries.clone() {
                    assert!(!seen[t.bin][qq], "worker {x}: unit covered twice");
                    seen[t.bin][qq] = true;
                    units += 1;
                }
            }
            let mut sorted = bins.clone();
            sorted.sort_unstable();
            assert_eq!(bins, sorted, "worker {x}: visits not in bin order");
            assert!(units == a || units == a + 1, "worker {x}: quota {units}");
            if rem > 0 && units == a + 1 {
                large += 1;
            }
        }
        for b in 0..r {
            assert!(seen[b].iter().all(|&s| s), "bin {b}: unit missing");
        }
        if rem > 0 {
            assert_eq!(large, rem, "exactly N mod W large quotas");
        }
    }

    #[test]
    fn canonical_w2() {
        let p = canonical();
        let got = unit_partition(&p, 2);
        assert_eq!(
            got,
            vec![
                vec![v(0, 0, 6), v(1, 0, 6), v(3, 0, 6), v(5, 0, 3)],
                vec![v(2, 0, 6), v(4, 0, 6), v(5, 3, 6), v(6, 0, 6)],
            ]
        );
        assert_eq!(total_visits(&got), 8);
        assert_eq!(total_visits(&got) - 7, 1, "1 extra replica");
        check_invariants(&p, 2, &got);
    }

    #[test]
    fn canonical_w4() {
        let p = canonical();
        let got = unit_partition(&p, 4);
        assert_eq!(
            got,
            vec![
                vec![v(0, 0, 5), v(3, 0, 6)],
                vec![v(0, 5, 6), v(4, 0, 6), v(6, 0, 3)],
                vec![v(2, 0, 6), v(5, 0, 1), v(6, 3, 6)],
                vec![v(1, 0, 6), v(5, 1, 6)],
            ]
        );
        assert_eq!(total_visits(&got), 10);
        assert_eq!(total_visits(&got) - 7, 3, "3 extra replicas");
        check_invariants(&p, 4, &got);
    }

    #[test]
    fn w1_is_one_visit_per_bin() {
        let p = canonical();
        let got = unit_partition(&p, 1);
        let want: Vec<Visit> = (0..7).map(|b| v(b, 0, 6)).collect();
        assert_eq!(got, vec![want]);
        check_invariants(&p, 1, &got);
    }

    #[test]
    fn q1_needs_no_split() {
        let p = toy_plan(&[50, 40, 30, 20], 1);
        for w in 1..=5 {
            let got = unit_partition(&p, w);
            assert_eq!(total_visits(&got), 4, "W={w}");
            for visits in &got {
                for t in visits {
                    assert_eq!(t.queries, 0..1);
                }
            }
            check_invariants(&p, w, &got);
        }
    }

    #[test]
    fn fewer_units_than_workers_leaves_empty_quotas() {
        let p = toy_plan(&[30, 20], 1);
        let got = unit_partition(&p, 5);
        let counts: Vec<usize> = got
            .iter()
            .map(|visits| visits.iter().map(|t| t.queries.len()).sum())
            .collect();
        assert_eq!(counts, vec![1, 1, 0, 0, 0]);
        assert_eq!(total_visits(&got), 2);
        check_invariants(&p, 5, &got);
    }

    #[test]
    fn deterministic_and_quota_shaped_elsewhere() {
        for (bps, q, w) in [
            (vec![9, 8, 7, 6, 5], 3, 2),
            (vec![100, 1, 1, 1], 4, 3),
            (vec![30, 20, 10], 2, 4),
            (vec![5], 5, 3),
        ] {
            let p = toy_plan(&bps, q);
            let (a, b) = (unit_partition(&p, w), unit_partition(&p, w));
            assert_eq!(a, b, "deterministic for {bps:?} Q={q} W={w}");
            check_invariants(&p, w, &a);
        }
    }

    #[test]
    fn batch_7bins_100slots_w2() {
        // Round 96: 7 reference bins x 100 single-block jobs (M = 100 slots in
        // every bin, N = 700 units). Same core with Q := M.
        let p = canonical();
        let got = super::unit_partition_dims(&p.reference_bins, 100, 2);
        assert_eq!(
            got,
            vec![
                vec![v(0, 0, 100), v(1, 0, 100), v(3, 0, 100), v(5, 0, 50)],
                vec![v(2, 0, 100), v(4, 0, 100), v(5, 50, 100), v(6, 0, 100)],
            ]
        );
        assert_eq!(total_visits(&got), 8);
        assert_eq!(total_visits(&got) - 7, 1, "1 extra replica");
        let counts: Vec<usize> = got
            .iter()
            .map(|visits| visits.iter().map(|t| t.queries.len()).sum())
            .collect();
        assert_eq!(counts, vec![350, 350]);
    }

    #[test]
    fn batch_7bins_100slots_w4() {
        let p = canonical();
        let got = super::unit_partition_dims(&p.reference_bins, 100, 4);
        assert_eq!(
            got,
            vec![
                vec![v(0, 0, 75), v(3, 0, 100)],
                vec![v(0, 75, 100), v(4, 0, 100), v(6, 0, 50)],
                vec![v(2, 0, 100), v(5, 0, 25), v(6, 50, 100)],
                vec![v(1, 0, 100), v(5, 25, 100)],
            ]
        );
        assert_eq!(total_visits(&got), 10);
        assert_eq!(total_visits(&got) - 7, 3, "3 extra replicas");
        let counts: Vec<usize> = got
            .iter()
            .map(|visits| visits.iter().map(|t| t.queries.len()).sum())
            .collect();
        assert_eq!(counts, vec![175, 175, 175, 175]);
    }

    #[test]
    fn batch_dims_agree_with_wrapper_on_single_query_shapes() {
        // The extracted core with M = Q must reproduce the wrapper exactly.
        for (bps, q, w) in [
            (vec![700, 600, 500, 400, 300, 200, 100], 6, 2),
            (vec![700, 600, 500, 400, 300, 200, 100], 6, 4),
            (vec![9, 8, 7, 6, 5], 3, 2),
            (vec![100, 1, 1, 1], 4, 3),
            (vec![30, 20, 10], 2, 4),
        ] {
            let p = toy_plan(&bps, q);
            assert_eq!(
                super::unit_partition_dims(&p.reference_bins, q, w),
                unit_partition(&p, w),
                "core must agree for {bps:?} Q={q} W={w}"
            );
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn minimal_visits_against_brute_force_oracle() {
        // Every R,Q,W <= 3: enumerate all quota-respecting assignments of the
        // N units to W workers and take the minimum (bin,worker) touch count.
        // Visit count is contiguity-blind, so the oracle needs no contiguity
        // filter; the construction must still attain the minimum.
        for r in 1..=3 {
            for q in 1..=3 {
                for w in 1..=3 {
                    let bps: Vec<u64> = (0..r).map(|i| (r - i) as u64 * 10 + 1).collect();
                    let p = toy_plan(&bps, q);
                    let n = r * q;
                    let (a, rem) = (n / w, n % w);
                    let mut best = usize::MAX;
                    let mut assign = vec![0usize; n];
                    loop {
                        let mut counts = vec![0usize; w];
                        for &x in &assign {
                            counts[x] += 1;
                        }
                        if counts.iter().all(|&c| c == a || (rem > 0 && c == a + 1)) {
                            let mut pair = vec![vec![false; w]; r];
                            for (u, &x) in assign.iter().enumerate() {
                                pair[u / q][x] = true;
                            }
                            let mut touches = 0;
                            for b in 0..r {
                                touches += pair[b].iter().filter(|&&t| t).count();
                            }
                            best = best.min(touches);
                        }
                        let mut k = 0;
                        while k < n {
                            assign[k] += 1;
                            if assign[k] < w {
                                break;
                            }
                            assign[k] = 0;
                            k += 1;
                        }
                        if k == n {
                            break;
                        }
                    }
                    let got = total_visits(&unit_partition(&p, w));
                    assert_eq!(got, best, "R={r} Q={q} W={w}");
                }
            }
        }
    }
}
