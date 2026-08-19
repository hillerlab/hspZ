#!/usr/bin/env bash
# Classifies the difference between two HSP sets. Dedup runs per `seed_and_filter`
# call, so chunk and bin structure legitimately change the output in exactly two
# ways, and in no others:
#
#   DUPLICATES-ONLY  distinct anchors identical; the extra lines are exact
#                    duplicates of anchors both sides keep (`max_hits` chunking)
#   CONTAINED-EXTRA  one side keeps a segment the other absorbed into the segment
#                    containing it on the same diagonal — scope-sensitive
#                    `contained()` removal, a real alignment, not a duplicate
#
# Anything else is FAIL: a lost anchor, or an extra that is neither. Comparing
# distinct *counts* would pass a run that dropped one anchor and gained another,
# so the sets are compared as sets and every differing line is classified.
#
# Usage: assets/tests/multiplicity_gate.sh <label> <A dir|file> <B dir|file>
#        assets/tests/multiplicity_gate.sh --self-test
# Exit 0: EXACT, DUPLICATES-ONLY or CONTAINED-EXTRA.  Exit 1: FAIL.
set -uo pipefail
export LC_ALL=C

gate() {  # label A B
    local label=$1 a=$2 b=$3
    local t; t=$(mktemp -d "${TMPDIR:-/tmp}/hspZ-gate-XXXXXX")
    # shellcheck disable=SC2064
    trap "rm -rf '$t'" RETURN

    local src
    for side in a b; do
        src=$([ "$side" = a ] && echo "$a" || echo "$b")
        if [ -d "$src" ]; then
            # shellcheck disable=SC2086
            cat "$src"/*.segments 2>/dev/null
        else
            cat "$src" 2>/dev/null
        fi | sort > "$t/$side"
        uniq "$t/$side" > "$t/${side}u"
    done

    local na nb da db dupa dupb ma mb
    na=$(wc -l < "$t/a"); nb=$(wc -l < "$t/b")
    da=$(wc -l < "$t/au"); db=$(wc -l < "$t/bu")
    dupa=$((na - da)); dupb=$((nb - db))
    ma=$(uniq -c "$t/a" | awk '{if($1>m)m=$1} END{print m+0}')
    mb=$(uniq -c "$t/b" | awk '{if($1>m)m=$1} END{print m+0}')

    if [ "$na" = 0 ] && [ "$nb" = 0 ]; then
        printf 'gate %-20s EMPTY  both sides have no HSPs\n' "$label"
        return 1
    fi
    if cmp -s "$t/a" "$t/b"; then
        printf 'gate %-20s EXACT  %s HSPs, %s distinct\n' "$label" "$na" "$da"
        return 0
    fi

    # (a) distinct-anchor sets, compared as sets. They can legitimately differ in
    # one way only: `contained()` removal is scope-sensitive, so a segment that a
    # wider chunk absorbed into the one containing it survives when the pair
    # straddles two chunks. Those extras are real alignments, not duplicates, so
    # they get their own verdict rather than being folded into half (b).
    if ! cmp -s "$t/au" "$t/bu"; then
        comm -23 "$t/au" "$t/bu" > "$t/only_a"
        comm -13 "$t/au" "$t/bu" > "$t/only_b"
        if ! python3 - "$t/au" "$t/bu" "$t/only_a" "$t/only_b" > "$t/contained"; then
            printf 'gate %-20s FAIL   distinct anchors differ: %s vs %s (only-A %s, only-B %s)\n' \
                "$label" "$da" "$db" "$(wc -l < "$t/only_a")" "$(wc -l < "$t/only_b")"
            head -4 "$t/contained" | sed 's/^/         /'
            return 1
        fi <<'PY'
import sys

def load(path):
    """Segments keyed by (r_chr, q_chr, strand, diagonal) -> ref intervals.

    `hsp.rs contained()` is same-diagonal plus ref-interval containment, and the
    diagonal is preserved by the chromosome-relative rewrite within one pair.
    """
    d = {}
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) != 8:
            continue
        rs, qs = int(f[1]), int(f[4])
        d.setdefault((f[0], f[3], f[6], rs - qs), []).append((rs, int(f[2])))
    return d

au, bu = load(sys.argv[1]), load(sys.argv[2])
bad = 0
for path, other in ((sys.argv[3], bu), (sys.argv[4], au)):
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) != 8:
            print(f"unparseable: {line.rstrip()}")
            bad += 1
            continue
        rs, re_, qs = int(f[1]), int(f[2]), int(f[4])
        near = other.get((f[0], f[3], f[6], rs - qs), [])
        if not any(s <= rs and e >= re_ or s >= rs and e <= re_ for s, e in near):
            print(f"not a contained pair: {line.rstrip()}  "
                  f"({len(near)} same-diagonal segments in the other arm)")
            bad += 1
raise SystemExit(1 if bad else 0)
PY
        printf 'gate %-20s CONTAINED-EXTRA  every extra segment is a contained pair: only-A %s, only-B %s; distinct %s vs %s; total %s vs %s\n' \
            "$label" "$(wc -l < "$t/only_a")" "$(wc -l < "$t/only_b")" \
            "$da" "$db" "$na" "$nb"
        return 0
    fi

    # (b) every differing line must be an exact duplicate of a kept line. Implied
    # by (a), and checked anyway: it is the half that would have to break for a
    # dropped-unique/gained-duplicate swap to slip through a counts-only check.
    comm -3 "$t/a" "$t/b" | sed 's/^\t//' | sort -u > "$t/diff"
    local ndiff notdup
    ndiff=$(wc -l < "$t/diff")
    notdup=$(comm -23 "$t/diff" "$t/au" | wc -l)
    if [ "$notdup" != 0 ]; then
        printf 'gate %-20s FAIL   %s differing lines are not duplicates of kept anchors\n' \
            "$label" "$notdup"
        comm -23 "$t/diff" "$t/au" | head -3 | sed 's/^/         /'
        return 1
    fi

    printf 'gate %-20s DUPLICATES-ONLY  distinct %s == %s; total %s vs %s; dups %s vs %s; max mult %s vs %s (%s anchors)\n' \
        "$label" "$da" "$db" "$na" "$nb" "$dupa" "$dupb" "$ma" "$mb" "$ndiff"
    return 0
}

self_test() {
    local t; t=$(mktemp -d "${TMPDIR:-/tmp}/hspZ-gate-self-XXXXXX")
    local rc=0 out
    mk() { mkdir -p "$t/$1"; printf '%b' "$2" > "$t/$1/x.segments"; }
    expect() {  # want_verdict want_exit label A B
        out=$(gate "$3" "$t/$4" "$t/$5"); local got=$?
        if [ "$got" != "$2" ] || ! printf '%s' "$out" | grep -q "$1"; then
            printf 'self-test FAIL (%s): exit %s want %s :: %s\n' "$3" "$got" "$2" "$out"
            rc=1
        fi
    }
    mk same_a 'a\nb\nc\n';   mk same_b 'a\nb\nc\n'
    mk dup_a  'a\na\nb\n';   mk dup_b  'a\nb\n'
    mk lost_a 'a\nb\n';      mk lost_b 'a\nc\n'
    # The failure the second half exists for: same distinct COUNT, different set,
    # and a duplicate added to keep the totals plausible.
    mk swap_a 'a\nb\nb\n';   mk swap_b 'a\nc\nc\n'
    # Real segment lines, because the contained-pair check parses coordinates.
    # `keep` is what both arms agree on; `inner` sits inside it on the same
    # diagonal (accepted), `far` is on the same diagonal but disjoint (not).
    keep='chr1\t100\t200\tchrX\t50\t150\t+\t1000\n'
    inner='chr1\t120\t180\tchrX\t70\t130\t+\t900\n'
    far='chr1\t300\t400\tchrX\t250\t350\t+\t900\n'
    mk cont_a "$keep";       mk cont_b "$keep$inner"
    mk far_a  "$keep";       mk far_b  "$keep$far"
    expect EXACT            0 self_exact same_a same_b
    expect DUPLICATES-ONLY  0 self_dups  dup_a  dup_b
    expect FAIL             1 self_lost  lost_a lost_b
    expect FAIL             1 self_swap  swap_a swap_b
    expect CONTAINED-EXTRA  0 self_cont  cont_a cont_b
    expect FAIL             1 self_far   far_a  far_b
    rm -rf "$t"
    [ "$rc" = 0 ] && echo "multiplicity_gate self-test: 6/6 pass"
    return $rc
}

case "${1:-}" in
    --self-test) self_test;;
    "" | -h | --help) sed -n '2,18p' "$0"; exit 2;;
    *) [ $# -eq 3 ] || { echo "usage: $0 <label> <A> <B>" >&2; exit 2; }
       gate "$1" "$2" "$3";;
esac
