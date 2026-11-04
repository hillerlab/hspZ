#!/usr/bin/env python3
"""Fixture for `run --paired` (assets/tests/paired-identity.sh).

Real chain gaps, written the way make_lastz_chains' GPU RepeatFiller will:
one record per gap and side, the same name on both sides, uppercased
(RepeatFiller's [unmask]), and the query side reverse-complemented for '-'
chains so every pair is plus strand. Gaps follow RepeatFiller's predicate:
chain score >= 1000, 30 <= gap <= 20000 on both sides.

Plus edge pairs: empty records, records shorter than / exactly one seed span,
and unmasked low-complexity pairs (poly-A, AT repeat) whose hits overflow a
small --max-hits.

Usage: gen_paired_fixture.py CHAIN.gz TARGET.fa QUERY.fa OUT [N_GAPS] [SMALL_C]
  TARGET.fa / QUERY.fa must be samtools-faidx indexed.
Writes OUT/{targets,queries}.fa (every pair), OUT/c_{targets,queries}.fa
(pairs that fit one chunk at -C SMALL_C), OUT/d_{targets,queries}.fa (the
low-complexity pairs and their neighbours) and OUT/pairs.tsv.
"""
import gzip
import subprocess
import sys

SPAN = 19  # 12of19
chain_path, t_fa, q_fa, out = sys.argv[1:5]
n_gaps = int(sys.argv[5]) if len(sys.argv) > 5 else 300
small_c = int(sys.argv[6]) if len(sys.argv) > 6 else 2000

gaps = []  # (t_name, t_from, t_to, q_name, q_from, q_to, strand), 1-based inclusive forward
with gzip.open(chain_path, "rt") as fh:
    for line in fh:
        p = line.split()
        if not p or p[0].startswith("#"):
            continue
        if p[0] == "chain":
            h, pos = p, [int(p[5]), int(p[10])]
            continue
        if len(p) == 3:
            size, dt, dq = map(int, p)
            tb, qb = pos[0] + size, pos[1] + size
            tg, qg = tb + dt, qb + dq
            if int(h[1]) >= 1000 and 30 <= dt <= 20000 and 30 <= dq <= 20000:
                q_size, strand = int(h[8]), h[9]
                qf, qt = (q_size - qg + 1, q_size - qb) if strand == "-" else (qb + 1, qg)
                gaps.append((h[2], tb + 1, tg, h[7], qf, qt, strand))
            pos = [tg, qg]

# Deterministic stride, half the gaps from each strand (the dominant chain of a
# real pair is often one strand, which would otherwise leave '-' untested).
picked = []
for strand in "+-":
    pool = [g for g in gaps if g[6] == strand]
    want = n_gaps - len(picked) if strand == "-" else n_gaps // 2
    picked += pool[::max(1, len(pool) // max(1, want))][:want]
gaps = picked


def faidx(fa, regions):
    out = subprocess.run(["samtools", "faidx", fa, *regions], check=True,
                         capture_output=True, text=True).stdout
    seqs, cur = [], None
    for line in out.splitlines():
        if line.startswith(">"):
            cur = []
            seqs.append(cur)
        else:
            cur.append(line.strip())
    return ["".join(s).upper() for s in seqs]


COMP = str.maketrans("ACGTN", "TGCAN")
t_seqs = faidx(t_fa, [f"{g[0]}:{g[1]}-{g[2]}" for g in gaps])
q_seqs = faidx(q_fa, [f"{g[3]}:{g[4]}-{g[5]}" for g in gaps])
pairs = []
for i, (g, ts, qs) in enumerate(zip(gaps, t_seqs, q_seqs)):
    if g[6] == "-":
        qs = qs.translate(COMP)[::-1]
    pairs.append((f"g{i:07d}", ts, qs, f"{g[0]}:{g[1]}-{g[2]}\t{g[3]}:{g[4]}-{g[5]}\t{g[6]}"))

# Edge pairs, spliced in the middle so they sit between real gaps.
mid = len(pairs) // 2
low = [
    ("e_empty", "", "", "empty"),
    ("e_short", "ACGTACGTAC", "ACGTACGTAC", "shorter than one span"),
    ("e_span", "TTGACCTAGGACTTACGGATCCAGT", t_seqs[0][:SPAN] if t_seqs else "A" * SPAN, "query exactly one span"),
    ("e_polyA", "A" * 3000, "A" * 3000, "low complexity"),
    ("e_AT", "AT" * 1000, "AT" * 1000, "low complexity"),
]
pairs[mid:mid] = low


def write(prefix, chosen):
    with open(f"{out}/{prefix}targets.fa", "w") as t, open(f"{out}/{prefix}queries.fa", "w") as q:
        for name, ts, qs, _ in chosen:
            t.write(f">{name}\n")
            q.write(f">{name}\n")
            for k in range(0, len(ts), 60):
                t.write(ts[k:k + 60] + "\n")
            for k in range(0, len(qs), 60):
                q.write(qs[k:k + 60] + "\n")


write("", pairs)
write("c_", [p for p in pairs if len(p[2]) - SPAN + 1 <= small_c])
write("d_", pairs[mid - 5:mid + len(low) + 5])
with open(f"{out}/pairs.tsv", "w") as f:
    for name, ts, qs, note in pairs:
        f.write(f"{name}\t{len(ts)}\t{len(qs)}\t{note}\n")
minus = sum(1 for g in gaps if g[6] == "-")
print(f"{len(pairs)} pairs ({len(gaps)} gaps, {minus} minus-strand, {len(low)} edge); "
      f"c_ subset at -C {small_c}: {sum(1 for p in pairs if len(p[2]) - SPAN + 1 <= small_c)}")
