# Reject weak anchors before materialization

<!-- @source: src/gpu/kernels.rs::mark_score_survivors -->
<!-- @source: src/gpu/kernels.rs::count_survivors -->
<!-- @source: src/gpu/kernels.rs::emit_survivors -->
<!-- @source: src/gpu/mod.rs::seed_and_filter -->

## D.1 — Feed coalesced anchor chunks to a warp
<!-- @id: d-coalesced-anchors -->
WHAT GOES IN: Packed `(reference-end, query-end)` anchors in raw hit order.
WHAT HAPPENS: A warp fetches 32 consecutive anchors coalescently and broadcasts one anchor at a time to its lanes.
WHAT COMES OUT: Shared query windows and per-anchor reference coordinates ready for scoring.
INVARIANT: Anchor order and identity are unchanged by the warp's memory mapping.

```mermaid
sequenceDiagram
    participant A as Packed anchors
    participant W as Score-gate warp
    participant Q as Query-window cache
    participant S as X-drop scorer
    A->>W: 32 adjacent anchors
    W->>Q: reuse equal query positions
    W-->>S: one anchor at a time
```

## D.2 — Compute a score-only X-drop bound
<!-- @id: d-score-only -->
WHAT GOES IN: One anchor, encoded sequences, substitution matrix, and X-drop.
WHAT HAPPENS: Lanes score right and left prefixes, stop on the first X-drop or edge, and retain only the best total score.
WHAT COMES OUT: An upper score for the anchored ungapped segment.
INVARIANT: The threshold value rounds through KegAlign's `f32` conversion before comparison.

```mermaid
sequenceDiagram
    participant A as Anchor
    participant R as Right prefix
    participant L as Left prefix
    participant G as Threshold gate
    A->>R: score until X-drop or edge
    A->>L: score until X-drop or edge
    R-->>G: right maximum
    L-->>G: left maximum
    G->>G: KegAlign float comparison
```

## D.3 — Write one byte per decision
<!-- @id: d-flags -->
WHAT GOES IN: The score-only total and hsp threshold K.
WHAT HAPPENS: The common reject path writes zero; plausible anchors write a one-byte survivor flag.
WHAT COMES OUT: A dense flag vector aligned with the original anchor array.
INVARIANT: Rejection creates no HSP record, while every active flag slot is overwritten before reuse.

```mermaid
sequenceDiagram
    participant G as Threshold gate
    participant F as Byte flags
    participant R as Reject path
    participant K as Keep path
    G->>R: rounded score below K
    R-->>F: 0
    G->>K: rounded score at least K
    K-->>F: 1
```

## D.4 — Compact survivors without reordering
<!-- @id: d-compact -->
WHAT GOES IN: One byte flag per raw anchor.
WHAT HAPPENS: Device blocks count flags, the host scans only block sums, and stable local ranks emit original anchor IDs.
WHAT COMES OUT: A small, increasing list of survivor IDs for full materialization.
INVARIANT: Stable compaction preserves raw hit order exactly, including under forced MAX_HITS chunking.

```mermaid
sequenceDiagram
    participant F as Byte flags
    participant C as Block counts
    participant H as Host prefix
    participant E as Stable emitter
    participant S as Survivor IDs
    F->>C: ballots of kept flags
    C-->>H: one count per block
    H-->>E: exclusive block offsets
    E-->>S: original IDs in order
```
