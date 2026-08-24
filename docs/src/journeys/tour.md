# From genomes to deterministic segment files

<!-- @source: src/run.rs::run -->
<!-- @source: src/plan.rs::plan_within_budget -->
<!-- @source: src/seed.rs::SeedTable::build_parallel -->
<!-- @source: src/gpu/mod.rs::Engine::seed_and_filter -->
<!-- @source: src/gpu/kernels.rs::find_hsps -->
<!-- @source: src/hsp.rs::dedup_and_order -->

## 0.1 — From genomes to segments
<!-- @id: overview -->
WHAT GOES IN: Softmasked FASTA, FASTA.gz, or 2bit reference and query records.
WHAT HAPPENS: hspZ plans chromosome bins, finds seed hits, filters and extends HSPs, then orders them for LASTZ.
WHAT COMES OUT: Non-empty plus/minus `.segments` files or one reproducible tarball.
INVARIANT: Input format does not change the decoded sequence set.

```mermaid
sequenceDiagram
    participant I as Input records
    participant P as Planner
    participant G as GPU Seed + Filter
    participant E as Ordered emitter
    I->>P: names, lengths, original order
    P->>G: reference × query WorkUnits
    G-->>E: accepted HSPs + ordinal
    E-->>I: LASTZ-compatible .segments
```

## 0.2 — Plan chromosome-atomic work
<!-- @id: planning -->
WHAT GOES IN: Record metadata, block targets, worker count, and device memory.
WHAT HAPPENS: The planner builds deterministic bins, forms a reference × query grid, and halves unsafe targets.
WHAT COMES OUT: Ordered WorkUnits assigned to GPU workers by reference bin.
INVARIANT: A chromosome is never split to satisfy a target or a memory budget.

```mermaid
sequenceDiagram
    participant M as Record metadata
    participant P as LPT planner
    participant B as Memory preflight
    participant W as WorkUnits
    M->>P: id, length, input order
    P->>B: chromosome-atomic bins
    B-->>P: fit or halve target
    P->>W: reference bins × query bins
```

## 0.3 — Build and query a spaced-seed index
<!-- @id: seeding -->
WHAT GOES IN: One packed reference bin, one query interval, and a seed shape such as 12of19.
WHAT HAPPENS: Reference positions are scattered into stable buckets; exact and permitted transition query seeds look them up.
WHAT COMES OUT: A cumulative stream of reference/query anchor positions.
INVARIANT: Full seed windows containing an invalid base are rejected, and reference bucket order is stable.

```mermaid
sequenceDiagram
    participant R as Reference bin
    participant T as Stable SeedTable
    participant Q as Query interval
    participant A as Anchor stream
    R->>T: count, scan, deterministic scatter
    Q->>Q: exact seed + transition variants
    Q->>T: lookup each packed k-mer
    T-->>A: reference positions in bucket order
```

## 0.4 — Reject weak hits before materialization
<!-- @id: filtering -->
WHAT GOES IN: Packed anchors in exact hit-stream order.
WHAT HAPPENS: A score-only X-drop gate marks plausible anchors; scans compact survivor IDs without reordering them.
WHAT COMES OUT: A dense list of the rare anchors worth full extension.
INVARIANT: Compaction preserves the original hit order and MAX_HITS chunk scope.

```mermaid
sequenceDiagram
    participant A as Packed anchors
    participant S as Score gate
    participant C as Stable compaction
    participant H as HSP materializer
    A->>S: score-only left + right X-drop
    S-->>C: one survivor flag per hit
    C->>C: count, scan, emit original IDs
    C-->>H: dense survivors in hit order
```

## 0.5 — Recover exact HSPs and apply entropy
<!-- @id: xdrop-entropy -->
WHAT GOES IN: A surviving anchor and the encoded reference/query sequences.
WHAT HAPPENS: One warp finds right and left X-drop maxima, recovers boundaries, and conditionally recounts matching bases for entropy.
WHAT COMES OUT: An accepted SegmentPair with exact endpoints and score, or rejection.
INVARIANT: Earliest-maximum ties and KegAlign float conversion points are preserved.

```mermaid
sequenceDiagram
    participant A as Surviving anchor
    participant X as Warp X-drop
    participant E as Entropy gate
    participant H as SegmentPair
    A->>X: extend right in 32-base tiles
    A->>X: extend left in 32-base tiles
    X->>E: score + closed HSP interval
    E-->>H: accept when threshold semantics pass
```

## 0.6 — Make multi-GPU completion deterministic
<!-- @id: deterministic-output -->
WHAT GOES IN: WorkUnits distributed across independently finishing GPU workers.
WHAT HAPPENS: Results enter an ordinal buffer, then one emitter performs host dedup, LASTZ ordering, coordinate conversion, and optional partitioning.
WHAT COMES OUT: The same bytes for one or many GPUs within the same layout.
INVARIANT: Worker completion order never controls file order or partition history.

```mermaid
sequenceDiagram
    participant G0 as GPU worker 0
    participant G1 as GPU worker 1
    participant O as Ordinal buffer
    participant E as Emitter
    participant F as Segment files
    G1->>O: finish WorkUnit 3
    G0->>O: finish WorkUnit 2
    O->>E: replay WorkUnit 2
    O->>E: replay WorkUnit 3
    E-->>F: deterministic bytes
```
