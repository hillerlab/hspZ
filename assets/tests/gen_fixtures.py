#!/usr/bin/env python3
"""Generate the tiny synthetic parity fixtures the harness compares against the oracle.

Deterministic: a fixed PRNG seed, so the oracle and the Rust port are always
compared on the same bytes. Sequences are sized so a clean match clears the
default hspthresh of 3000 (~33 matching bases at +91..100 each).
"""

import os
import random
import sys

COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def rc(s):
    return s.translate(COMP)[::-1]


def rand(n, rng):
    return "".join(rng.choice("ACGT") for _ in range(n))


def fasta(path, records):
    with open(path, "w") as fh:
        for name, seq in records:
            fh.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i : i + 60] + "\n")


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    rng = random.Random(20260811)
    base = rand(600, rng)
    cases = {}

    # A perfect match must produce one long HSP on the plus strand.
    cases["identical"] = ([("ref", base)], [("qry", base)])

    # One substitution: the X-drop must ride straight through it.
    sub = list(base)
    sub[300] = {"A": "C", "C": "A", "G": "T", "T": "G"}[sub[300]]
    cases["mismatch"] = ([("ref", base)], [("qry", "".join(sub))])

    # A 5 bp deletion splits the alignment into two ungapped segments on
    # different diagonals.
    cases["indel"] = ([("ref", base)], [("qry", base[:300] + base[305:])])

    # Reverse strand only: every HSP must land in the minus file.
    cases["revcomp"] = ([("ref", base)], [("qry", rc(base))])

    # A tiled repeat gives one seed thousands of reference hits, exercising the
    # hit-expansion path and the containment dedup.
    unit = rand(20, rng)
    cases["repeat"] = ([("ref", unit * 30)], [("qry", unit * 30)])

    # An N run kills every seed whose window touches it and scores as
    # bad_score during extension.
    with_n = base[:250] + "N" * 30 + base[280:]
    cases["ambiguous"] = ([("ref", base)], [("qry", with_n)])

    # Divergence in the middle: long enough to trip X-drop at the default 910,
    # short enough that the flanks stay above hspthresh on their own.
    diverged = base[:250] + rand(60, rng) + base[310:]
    cases["xdrop"] = ([("ref", base)], [("qry", diverged)])

    # Several records per file: exercises the '&' separators, the chromosome
    # tables and the reverse-complement chromosome remapping.
    multi_r = [("r1", base[:200]), ("r2", base[200:400]), ("r3", base[400:])]
    multi_q = [("q1", base[:200]), ("q2", rc(base[200:400])), ("q3", base[400:])]
    cases["multi"] = (multi_r, multi_q)

    # Soft-masked bases score as bad_score and never seed.
    cases["softmask"] = ([("ref", base)], [("qry", base[:250].lower() + base[250:])])

    # A non-default substitution matrix: uniform +100/-50 makes mismatches far
    # cheaper than the blastz default, so extensions run through divergence that
    # the default matrix would X-drop on. Reuses the diverged pair above.
    cases["scoring"] = ([("ref", base)], [("qry", diverged)])
    with open(os.path.join(outdir, "scoring.matrix"), "w") as fh:
        fh.write(
            "# non-default scoring set for the --scoring parity fixture\n"
            "bad_score          = X:-1000\n"
            "fill_score         = -100\n"
            "gap_open_penalty   =   400\n"
            "gap_extend_penalty =    30\n"
            "\n"
            "     A     C     G     T\n"
            "A  100   -50   -50   -50\n"
            "C  -50   100   -50   -50\n"
            "G  -50   -50   100   -50\n"
            "T  -50   -50   -50   100\n"
        )

    for name, (r, q) in cases.items():
        fasta(os.path.join(outdir, f"{name}.ref.fa"), r)
        fasta(os.path.join(outdir, f"{name}.qry.fa"), q)
    print("\n".join(sorted(cases)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fixtures")
