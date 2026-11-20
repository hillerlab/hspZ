# Accepted HSPs become LASTZ segments

<!-- @source: src/gpu/mod.rs::tail_lean -->
<!-- @source: src/hsp.rs::dedup_and_order -->
<!-- @source: src/hsp.rs::records -->
<!-- @source: src/hsp.rs::render_records -->
<!-- @source: src/run.rs::Emitter::emit_unit -->
<!-- @source: src/partition.rs::Partitioner::plan -->

## G.1 — Compact accepted HSPs
<!-- @id: g-compact -->
WHAT GOES IN: Materialized SegmentPairs and their done flags, one per survivor in survivor order.
WHAT HAPPENS: The device scans the done flags and copies accepted records into increasing materializer order, and the host receives only the active prefix. A chunk of at most 4,096 survivors skips all of it (0.1.2): the host copies the survivors' records as E.5 left them and keeps those with a positive score, which are exactly the accepted ones (an accepted record holds its truncated score, at least the threshold; a rejected one holds 0), in the same order. With `--hspthresh` below 2 every chunk takes the device path.
WHAT COMES OUT: A dense vector of raw accepted HSPs for one MAX_HITS chunk.
INVARIANT: Compaction changes storage density, never the relative order of accepted records; both paths give the same vector.

```mermaid
sequenceDiagram
    participant H as Materialized HSPs
    participant D as Done scan
    participant C as Stable compactor
    participant R as Raw accepted HSPs
    H->>C: candidate records
    D->>C: inclusive accepted ranks
    C-->>R: accepted records in order
    H-->>R: at most 4,096 survivors: copied, positive scores kept
```

## G.2 — Reproduce oracle deduplication
<!-- @id: g-dedup -->
WHAT GOES IN: Raw accepted HSPs from one chunk and strand.
WHAT HAPPENS: A stable diagonal sort makes equivalent runs adjacent; unique-copy compares each record with the previous input record.
WHAT COMES OUT: One survivor for each oracle equivalence run.
INVARIANT: Comparing with the last kept record instead would be wrong because containment is not transitive.

```mermaid
sequenceDiagram
    participant R as Raw HSPs
    participant S as Diagonal stable sort
    participant U as Previous-input dedup
    participant K as Kept HSPs
    R->>S: diagonal, ref start, len, score
    S->>U: adjacent contained runs
    U-->>K: first item of each run
```

## G.3 — Put survivors in LASTZ order
<!-- @id: g-lastz-order -->
WHAT GOES IN: Deduplicated SegmentPairs.
WHAT HAPPENS: A second stable sort orders query start, reference start, length, then descending score.
WHAT COMES OUT: LASTZ query-major segment order.
INVARIANT: Sort stability and unsigned wrapping diagonal semantics are part of byte parity.

```mermaid
sequenceDiagram
    participant K as Kept HSPs
    participant L as LASTZ stable sort
    participant O as Ordered segments
    K->>L: query/ref/len/score keys
    L-->>O: query-major order
```

## G.4 — Convert to chromosome coordinates
<!-- @id: g-coordinates -->
WHAT GOES IN: Block-relative SegmentPairs, chromosome tables, and strand.
WHAT HAPPENS: Upper-bound lookup finds each chromosome; starts become one-based and ends remain inclusive. Minus records are traversed in reverse.
WHAT COMES OUT: Eight numeric/text fields for each `.segments` line.
INVARIANT: Coordinate conversion never stitches across the `&` separator between chromosomes.

```mermaid
sequenceDiagram
    participant H as Ordered HSPs
    participant C as Chr tables
    participant R as Record converter
    participant T as Segment text
    H->>R: block-relative start + extent
    R->>C: upper-bound chromosome lookup
    C-->>R: name + block offset
    R-->>T: 1-based inclusive fields
```

## G.5 — Emit deterministic files or one archive
<!-- @id: g-emit -->
WHAT GOES IN: Completed WorkUnits arriving in any GPU completion order.
WHAT HAPPENS: Ordinals are replayed; optional diagonal partitioning runs before a directory or reproducible tar sink writes non-empty entries. With `--query-list`, every job has its own ordinal cursor, `-D` history and sink.
WHAT COMES OUT: `tmp<n>.block<q>.r<r>.{plus,minus}[.splitN].segments` files.
INVARIANT: Worker completion order never changes filenames, partition history, archive entry order, or bytes.

```mermaid
sequenceDiagram
    participant W as Completed WorkUnits
    participant O as Ordinal buffer
    participant P as Optional partitioner
    participant S as Output sink
    W->>O: out-of-order results
    O->>P: next ordinal only
    P-->>S: whole or split records
    S-->>S: directory or reproducible tar
```
