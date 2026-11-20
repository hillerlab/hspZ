#!/usr/bin/env bash
# Identity gate for the score-gate sieve (ZLUDA or NVIDIA, ~9 min on the local box).
#
# Claim: with the sieve the binary writes the same bytes as without it, for
# every option that reaches the score gate, and every lane decides as the host
# model says. Each arm runs the control (the same binary with
# HSPZ_GATE_SIEVE=0, its plain gate kernels; or another binary given as H0) and
# the sieve on one reference/query pair and compares the digests
# of all segment files; the sieve run carries HSPZ_GATE_SIEVE_CHECK=1, which
# recomputes every hit on the host and must report zero candidate mismatches,
# zero flag mismatches, zero kept-but-rejected hits and zero plane mismatches.
# An arm also states whether the sieve must be on or off there.
#
# `--paired` has its own script (assets/tests/paired-identity.sh).
#
# The fixture used for the release: hg38 chr20 against mm39
# chr19:20000000-24000000 (`samtools faidx`, record names as samtools writes
# them), 192,899,566 seed hits, default-arm digest 70152625c1a75cb3.
#
# Usage: FIX=<dir with ref.fa qry.fa> assets/tests/sieve-identity.sh
#        (H1=binary under test, H0=control binary, W=workdir to override)
set -uo pipefail
S=${TMPDIR:-/tmp}
for _v in CUDA_HOME ZLUDA; do
  eval "[ \"\${$_v+x}\" ] && _s_$_v=\$$_v"
done
_f=${HSPZ_TEST_ENV:-${XDG_CONFIG_HOME:-$HOME/.config}/hspz/test.env}
[ -f "$_f" ] && . "$_f"
for _v in CUDA_HOME ZLUDA; do
  eval "[ \"\${_s_$_v+x}\" ] && $_v=\$_s_$_v"
done
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
FIX=${FIX:?set FIX to a directory holding the chr20 fixture (ref.fa, qry.fa)}
FIX=$(cd "$FIX" && pwd)
H1=${H1:-$ROOT/target/release/hspZ}
H0=${H0:-$H1}
export CUDA_HOME=${CUDA_HOME:?set CUDA_HOME (CUDA toolkit root) in the environment or ~/.config/hspz/test.env}
export LD_LIBRARY_PATH=${ZLUDA:+$ZLUDA:}$CUDA_HOME/lib
W=${W:-$S/hspz-sieve-identity}; rm -rf "$W"; mkdir -p "$W"

digest() { (cd "$1" && cat $(ls *.segments 2>/dev/null | LC_ALL=C sort) | sha256sum | cut -c1-16); }
fail=0
printf '%-34s %-5s %-17s %-10s %s\n' arm sieve digest rho check

# arm NAME on|off REF QRY [hspz options...]; environment prefixes go through ENVV.
arm() {
  local name=$1 want=$2 ref=$3 qry=$4; shift 4
  # The control also expands seeds with the kernels of before round 110 (one warp or thread per seed) and buckets
  # with those of before round 114 (a vote per bucket), so an arm compares the group kernel and the bucket kernels
  # as well as the sieve; an arm's own ENVV may override that.
  env HSPZ_FIND_HITS_GROUPS=0 HSPZ_BUCKET_VOTES=loop ${ENVV:-} HSPZ_GATE_SIEVE=0 "$H0" run -r "$ref" -q "$qry" -o "$W/$name.0" --max-hits "${MAXH:-16711680}" "$@" \
    2> "$W/$name.0.err" || { echo "$name: control run failed"; tail -3 "$W/$name.0.err"; fail=$((fail+1)); return; }
  env ${ENVV:-} HSPZ_GATE_SIEVE_CHECK=1 "$H1" run -r "$ref" -q "$qry" -o "$W/$name.1" --max-hits "${MAXH:-16711680}" --time "$@" \
    2> "$W/$name.1.err" || { echo "$name: sieve run failed"; tail -3 "$W/$name.1.err"; fail=$((fail+1)); return; }
  local d0 d1 state rho check verdict=ok
  d0=$(digest "$W/$name.0"); d1=$(digest "$W/$name.1")
  state=$(grep -m1 -o 'gate sieve: o[nf]*' "$W/$name.1.err" | awk '{print $3}')
  rho=$(grep -m1 '^#sieve candidates' "$W/$name.1.err" | grep -o '([0-9.]*%)' | tr -d '()')
  check=$(grep -m1 '^#sieve check' "$W/$name.1.err" | sed 's/^#sieve check: //')
  [ "$d0" = "$d1" ] || verdict="DIGEST $d0 vs $d1"
  [ "${state:-none}" = "$want" ] || verdict="STATE ${state:-none}, expected $want"
  if [ "$want" = on ]; then
    # Every tally after the model's keep count must be zero.
    echo "$check" | grep -q 'candidate mismatches 0, flag mismatches 0, kept but rejected 0, plane word mismatches 0$' \
      || verdict="CHECK ${check:-missing}"
  fi
  # Under HSPZ_BUCKET_ORDER_CHECK every checked chunk must be the stable permutation by bucket.
  if grep -q '^#bucket-check' "$W/$name.1.err"; then
    local nb nok
    nb=$(grep -c '^#bucket-check chunk=' "$W/$name.1.err")
    nok=$(grep -c '^#bucket-check chunk=.* perm=ok monotone=ok stable=ok anchors=ok' "$W/$name.1.err")
    { [ "$nb" -gt 0 ] && [ "$nb" = "$nok" ] && ! grep -q '^#bucket-check-violation' "$W/$name.1.err"; } \
      || verdict="BUCKET ORDER $nok of $nb chunks"
  fi
  [ "$verdict" = ok ] || fail=$((fail+1))
  printf '%-34s %-5s %-17s %-10s %s\n' "$name" "${state:-none}" "$d1" "${rho:--}" "$verdict"
  [ -n "${KEEP:-}" ] || rm -rf "$W/$name.0" "$W/$name.1"
}

R=$FIX/ref.fa; Q=$FIX/qry.fa
arm defaults                 on  "$R" "$Q"
arm hspthresh-2400           on  "$R" "$Q" -H 2400
arm hspthresh-1500           on  "$R" "$Q" -H 1500
arm seed-14of22              on  "$R" "$Q" -s 14of22
arm notransition             on  "$R" "$Q" -n
arm xdrop-600                on  "$R" "$Q" -x 600
arm xdrop-999                on  "$R" "$Q" -x 999
# Without the query's mask plane (round 117) a masked base need not end an extension: the sieve stays on.
arm xdrop-1200               on  "$R" "$Q" -x 1200
arm ambiguous-n              off "$R" "$Q" -a n
arm ambiguous-iupac          off "$R" "$Q" -a iupac
arm strand-plus              on  "$R" "$Q" -S plus
arm strand-minus             on  "$R" "$Q" -S minus
arm noentropy                on  "$R" "$Q" -E
arm small-chunks             on  "$R" "$Q" -C 60000
ENVV="HSPZ_REF_BUCKETS=1" arm buckets-forced            on  "$R" "$Q"
ENVV="HSPZ_REF_BUCKETS=1" arm buckets-forced-2400       on  "$R" "$Q" -H 2400
ENVV="HSPZ_GATE_SIEVE=0"  arm sieve-switched-off        none "$R" "$Q"
# The gather arm: the same bound with the reference groups read through shared memory.
ENVV="HSPZ_GATE_SIEVE=gather" arm gather                    on  "$R" "$Q"
ENVV="HSPZ_GATE_SIEVE=gather" arm gather-2400               on  "$R" "$Q" -H 2400
ENVV="HSPZ_GATE_SIEVE=gather" arm gather-14of22             on  "$R" "$Q" -s 14of22
ENVV="HSPZ_GATE_SIEVE=gather HSPZ_REF_BUCKETS=1" arm gather-buckets-forced     on  "$R" "$Q"
# `auto`: gathered on the plain gate, per lane under the bucket pass.
ENVV="HSPZ_GATE_SIEVE=auto HSPZ_REF_BUCKETS=0" arm auto-plain                on  "$R" "$Q"
ENVV="HSPZ_GATE_SIEVE=auto HSPZ_REF_BUCKETS=1" arm auto-buckets-forced       on  "$R" "$Q"
grep -q 'gathered on the plain gate only' "$W/auto-plain.1.err" || { echo "auto-plain: not the auto arm"; fail=$((fail+1)); }
for a in gather gather-2400 gather-14of22 gather-buckets-forced; do
  grep -q 'gathered reference' "$W/$a.1.err" || { echo "$a: the gather kernels did not run"; fail=$((fail+1)); }
done
grep -q 'gathered reference' "$W/defaults.1.err" && { echo "defaults: ran the gather kernels"; fail=$((fail+1)); }

# A custom matrix with milder transitions: the class scores must follow the file.
cat > "$W/mild.q" <<'EOF'
    A    C    G    T
A   91 -114  -20 -123
C -114  100 -125  -20
G  -20 -125  100 -114
T -123  -20 -114   91
EOF
arm scoring-mild-transitions on  "$R" "$Q" -c "$W/mild.q"
grep -q 'transition -20' "$W/scoring-mild-transitions.1.err" || { echo "scoring arm: class scores did not follow the matrix"; fail=$((fail+1)); }

# IUPAC codes other than N loosen the transversion class to -100, in the reference and (without the mask
# plane, round 117) in the query; with the plane a query code is not terminal and the sieve must switch off.
python3 - "$R" "$Q" "$W" <<'EOF'
import sys
ref, qry, out = sys.argv[1:4]
def spike(path, dest, every):
    name, seq = None, []
    for line in open(path):
        if line.startswith(">"): name = line
        else: seq.append(line.rstrip("\n"))
    s = bytearray("".join(seq).encode())
    for i in range(every // 2, len(s), every):
        if s[i] in b"ACGT": s[i] = b"RYKMSW"[(i // every) % 6]
    with open(dest, "w") as f:
        f.write(name)
        for i in range(0, len(s), 60): f.write(s[i:i + 60].decode() + "\n")
spike(ref, out + "/ref_iupac.fa", 1499)
spike(qry, out + "/qry_iupac.fa", 5003)
EOF
arm iupac-in-reference       on  "$W/ref_iupac.fa" "$Q"
grep -q 'transversion -100' "$W/iupac-in-reference.1.err" || { echo "iupac arm: transversion class is not -100"; fail=$((fail+1)); }
arm iupac-in-query           on  "$R" "$W/qry_iupac.fa"
grep -q 'transversion -100' "$W/iupac-in-query.1.err" || { echo "iupac-in-query: transversion class is not -100"; fail=$((fail+1)); }

# The guard. Two diverged copies of one tandem array: nearly every hit extends, so
# the bound rejects almost nothing; the guard must see that on its first sample
# and run the plain gate on the batches after it. Same bytes with and without it.
python3 - "$W" <<'PY'
import random, sys
out = sys.argv[1]
rng = random.Random(20261118)
unit = [rng.choice("ACGT") for _ in range(171)]
def array(copies):
    seq = []
    for _ in range(copies):
        seq += [rng.choice("ACGT") if rng.random() < 0.03 else b for b in unit]
    return "".join(seq)
for name, copies in (("sat_ref", 120), ("sat_qry", 40)):
    s = array(copies)
    with open(f"{out}/{name}.fa", "w") as f:
        f.write(f">{name}\n")
        for i in range(0, len(s), 60): f.write(s[i:i + 60] + "\n")
PY
arm satellite-guard          on  "$W/sat_ref.fa" "$W/sat_qry.fa" -C 2000
grep -q '^#sieve guard: the bound passed more than half' "$W/satellite-guard.1.err" \
  || { echo "satellite-guard: the guard never held the plain gate"; fail=$((fail+1)); }
ENVV="HSPZ_GATE_SIEVE_GUARD=0" arm satellite-no-guard        on  "$W/sat_ref.fa" "$W/sat_qry.fa" -C 2000
grep -q '^#sieve guard' "$W/satellite-no-guard.1.err" && { echo "satellite-no-guard: the guard ran"; fail=$((fail+1)); }
grep -q '^#sieve guard' "$W/defaults.1.err" && { echo "defaults: the guard held on mouse against human"; fail=$((fail+1)); }

# Bucket pass with a window sized for the packed planes: a chunk the sieve does
# not gate skips the pass. Forced on here (the local card's L2 is too small for
# the rule to choose it): with the mask plane a query with IUPAC codes switches the sieve off for
# the unit, and on the tandem array the guard holds the plain gate.
ENVV="HSPZ_REF_BUCKETS=1 HSPZ_GATE_SIEVE_MASK=1" arm buckets-iupac-in-query    off "$R" "$W/qry_iupac.fa"
ENVV="HSPZ_REF_BUCKETS=1" arm buckets-satellite-guard   on  "$W/sat_ref.fa" "$W/sat_qry.fa" -C 2000

# Round 111: an undecided side of the bound is carried on to 128 right / 160 left bases by default. These arms run
# the static window of rounds 104-110 and the middle setting; every other arm above runs the default with the host
# recomputing each hit's walk.
ENVV="HSPZ_SIEVE_WORDS=2,3" arm window-static             on  "$R" "$Q"
ENVV="HSPZ_SIEVE_WORDS=2,3" arm window-static-2400        on  "$R" "$Q" -H 2400
ENVV="HSPZ_SIEVE_WORDS=3,4" arm window-middle             on  "$R" "$Q"
ENVV="HSPZ_SIEVE_WORDS=3,4 HSPZ_REF_BUCKETS=1" arm window-middle-buckets     on  "$R" "$Q" -C 60000
ENVV="HSPZ_SIEVE_WORDS=2,3" arm window-static-satellite   on  "$W/sat_ref.fa" "$W/sat_qry.fa" -C 2000
grep -q 'an undecided side carried to 128 right, 160 left' "$W/defaults.1.err" || { echo "defaults: the bound is not carried to its longest window"; fail=$((fail+1)); }
grep -q 'an undecided side carried' "$W/window-static.1.err" && { echo "window-static: the static window was not used"; fail=$((fail+1)); }

# Round 110: seeds expanded by runs of 32 per warp. `check` runs the per-seed kernel and the group kernel on each
# chunk and stops at the first anchor that differs; small chunks cut the runs at arbitrary seeds.
ENVV="HSPZ_FIND_HITS_GROUPS=check" arm find-hits-check           on  "$R" "$Q"
ENVV="HSPZ_FIND_HITS_GROUPS=check" arm find-hits-check-chunks    on  "$R" "$Q" -C 60000
ENVV="HSPZ_FIND_HITS_GROUPS=check" arm find-hits-check-satellite on  "$W/sat_ref.fa" "$W/sat_qry.fa" -C 2000
grep -q 'of packed planes' "$W/buckets-forced.1.err" || { echo "buckets-forced: the window is not plane-sized"; fail=$((fail+1)); }

# Round 114: the bucket kernels take one match.any instead of a vote per bucket. The permutation check (anchors
# and raw ids of every chunk: a permutation, ascending by bucket, stable, each anchor its own) runs on 1, 4, 8, 16
# and 31 buckets, on chunks that end inside a warp, and once on the kernels of before.
for shift in 26 24 23 22 21; do
  ENVV="HSPZ_REF_BUCKETS=1 HSPZ_REF_BUCKET_SHIFT=$shift HSPZ_BUCKET_ORDER_CHECK=1000000" arm bucket-order-shift-$shift on "$R" "$Q"
done
ENVV="HSPZ_REF_BUCKETS=1 HSPZ_REF_BUCKET_SHIFT=23 HSPZ_BUCKET_ORDER_CHECK=1000000" arm bucket-order-small-chunks on "$R" "$Q" -C 60000
ENVV="HSPZ_BUCKET_VOTES=loop HSPZ_REF_BUCKETS=1 HSPZ_REF_BUCKET_SHIFT=23 HSPZ_BUCKET_ORDER_CHECK=1000000" arm bucket-order-loop-kernels on "$R" "$Q"
for a in bucket-order-shift-26 bucket-order-shift-21 bucket-order-small-chunks bucket-order-loop-kernels; do
  grep -q '^#bucket-check chunk=' "$W/$a.1.err" || { echo "$a: the permutation check did not run"; fail=$((fail+1)); }
done

# Round 116: a bucketed chunk's survivors go to find_hsps by sorted slot: the scatter writes no raw id and nothing
# is sorted back. Every arm above that runs the bucket pass without the permutation check takes that path, against
# a control in raw order (below, the control's own switch). The arms with the check keep the ids, and there the
# anchors-only scatter runs beside the checked one and must write the same anchors word for word.
ENVV="HSPZ_REF_BUCKETS=1 HSPZ_SURVIVOR_ORDER=raw" arm survivors-raw-order       on  "$R" "$Q"
ENVV="HSPZ_REF_BUCKETS=1 HSPZ_SURVIVOR_ORDER=raw" arm survivors-raw-order-2400  on  "$R" "$Q" -H 2400
ENVV="HSPZ_REF_BUCKETS=1 HSPZ_REF_BUCKET_SHIFT=23" arm survivors-slots-chunks   on  "$R" "$Q" -C 60000
for a in bucket-order-shift-26 bucket-order-shift-24 bucket-order-shift-23 bucket-order-shift-22 bucket-order-shift-21 bucket-order-small-chunks; do
  n=$(grep -c '^#bucket-check-anchors-only' "$W/$a.1.err"); ok=$(grep -c '^#bucket-check-anchors-only .* equal=ok' "$W/$a.1.err")
  { [ "$n" -gt 0 ] && [ "$n" = "$ok" ]; } || { echo "$a: the anchors-only scatter differs or did not run ($ok of $n)"; fail=$((fail+1)); }
done
grep -q '^  sort_survivors' "$W/buckets-forced.1.err" && { echo "buckets-forced: survivors were sorted back to raw order"; fail=$((fail+1)); }
grep -q '^  sort_survivors' "$W/survivors-raw-order.1.err" || { echo "survivors-raw-order: the raw-order path did not run"; fail=$((fail+1)); }

# Round 117: the bound without the query's mask plane is the default and every arm above without a gather
# setting runs it, the host recomputing each walk the same way. These arms keep the plane (the kernels of rounds
# 104 to 116 with the terminal base tested once per word), where the sieve is off at xdrop 1200 and with IUPAC
# codes in the query.
ENVV="HSPZ_GATE_SIEVE_MASK=1" arm mask-plane                on  "$R" "$Q"
ENVV="HSPZ_GATE_SIEVE_MASK=1" arm mask-plane-2400           on  "$R" "$Q" -H 2400
ENVV="HSPZ_GATE_SIEVE_MASK=1 HSPZ_REF_BUCKETS=1" arm mask-plane-buckets        on  "$R" "$Q"
ENVV="HSPZ_GATE_SIEVE_MASK=1" arm mask-plane-xdrop-1200     off "$R" "$Q" -x 1200
ENVV="HSPZ_GATE_SIEVE_MASK=1" arm mask-plane-iupac-query    off "$R" "$W/qry_iupac.fa"
ENVV="HSPZ_GATE_SIEVE_MASK=1" arm mask-plane-satellite      on  "$W/sat_ref.fa" "$W/sat_qry.fa" -C 2000
grep -q 'query without a mask plane' "$W/defaults.1.err" || { echo "defaults: the query still has its mask plane"; fail=$((fail+1)); }
grep -q 'query without a mask plane' "$W/mask-plane.1.err" && { echo "mask-plane: the plane was dropped"; fail=$((fail+1)); }

# Round 119: a bucketed chunk's anchors are written in bucket order from its seeds, with no raw pass, no count over
# anchors and no scatter; it is on wherever survivors go by sorted slot, so the bucketed arms above already run it.
# The control has the group kernels off and keeps the count and the scatter. `check` runs both on every chunk and
# compares each bucket's anchors on the host; `=0` is the count and the scatter with everything else on.
for shift in 25 24 23; do
  ENVV="HSPZ_REF_BUCKETS=1 HSPZ_REF_BUCKET_SHIFT=$shift HSPZ_HITS_BUCKETED=check" arm hits-bucketed-check-$shift on "$R" "$Q"
done
ENVV="HSPZ_REF_BUCKETS=1 HSPZ_REF_BUCKET_SHIFT=24 HSPZ_HITS_BUCKETED=check" arm hits-bucketed-small-chunks on "$R" "$Q" -C 60000
ENVV="HSPZ_REF_BUCKETS=1 HSPZ_REF_BUCKET_SHIFT=23" arm hits-bucketed-2400        on  "$R" "$Q" -H 2400
ENVV="HSPZ_REF_BUCKETS=1 HSPZ_REF_BUCKET_SHIFT=23" arm hits-bucketed-iupac       on  "$W/ref_iupac.fa" "$W/qry_iupac.fa"
ENVV="HSPZ_REF_BUCKETS=1 HSPZ_REF_BUCKET_SHIFT=23 HSPZ_HITS_BUCKETED=0" arm hits-bucketed-off on "$R" "$Q"
# The group kernels' own check compares raw anchors, so under it the anchors must still be written raw.
ENVV="HSPZ_REF_BUCKETS=1 HSPZ_REF_BUCKET_SHIFT=23 HSPZ_FIND_HITS_GROUPS=check" arm hits-bucketed-groups-check on "$R" "$Q"
for a in hits-bucketed-check-25 hits-bucketed-check-24 hits-bucketed-check-23 hits-bucketed-small-chunks; do
  grep -q '^bucket order from the seeds check: .* every bucket the same anchors' "$W/$a.1.err" \
    || { echo "$a: the check did not run"; fail=$((fail+1)); }
done
for a in hits-bucketed-2400 hits-bucketed-iupac survivors-slots-chunks; do
  grep -q '^bucket order from the seeds: [0-9]* chunks' "$W/$a.1.err" || { echo "$a: the anchors were not written from the seeds"; fail=$((fail+1)); }
done
grep -q '^bucket order from the seeds' "$W/hits-bucketed-off.1.err" && { echo "hits-bucketed-off: the anchors were written from the seeds"; fail=$((fail+1)); }
grep -q '^  ref_bucket_scatter' "$W/hits-bucketed-off.1.err" || { echo "hits-bucketed-off: the scatter did not run"; fail=$((fail+1)); }
grep -q '^bucket order from the seeds' "$W/hits-bucketed-groups-check.1.err" && { echo "hits-bucketed-groups-check: the anchors were written from the seeds"; fail=$((fail+1)); }
grep -q '^find_hits by groups check: .* every anchor equal' "$W/hits-bucketed-groups-check.1.err" || { echo "hits-bucketed-groups-check: the check did not run"; fail=$((fail+1)); }
grep -q 'query without a mask plane' "$W/gather.1.err" && { echo "gather: the gathered arm lost its mask plane"; fail=$((fail+1)); }

# Round 122: the transition fold. A folded engine is seeded without transitions and finds each position's hits in
# one list; the control has neither the fold nor the sieve. `check` compares the tables the device built with the
# host's, k-mer by k-mer. A batch of --max-hits hits or more is cut at plain seeds and must be sent again with
# them (MAXH makes some batches, or most, such a batch; where most are, the engine drops its fold after 16 and
# the pass goes on with plain seeds); HSPZ_FOLD_CAPACITY makes the first cut batch take the fold's memory.
FB="HSPZ_REF_BUCKETS=1 HSPZ_REF_BUCKET_SHIFT=24"
ENVV="$FB HSPZ_FOLD=check" arm fold-check-24 on "$R" "$Q"
ENVV="HSPZ_REF_BUCKETS=1 HSPZ_REF_BUCKET_SHIFT=25 HSPZ_FOLD=check" arm fold-check-25 on "$R" "$Q"
ENVV="$FB HSPZ_FOLD=1 HSPZ_HITS_BUCKETED=check" arm fold-bucketed-check on "$R" "$Q"
ENVV="$FB HSPZ_FOLD=1" arm fold-2400 on "$R" "$Q" -H 2400
ENVV="$FB HSPZ_FOLD=check" arm fold-iupac on "$W/ref_iupac.fa" "$W/qry_iupac.fa"
ENVV="$FB HSPZ_FOLD=1 HSPZ_DEVICE_SEEDS=1" arm fold-device-seeds on "$R" "$Q"
ENVV="$FB HSPZ_FOLD=1 HSPZ_FOLD_RUN=32" arm fold-run-32 on "$R" "$Q"
ENVV="$FB HSPZ_FOLD=1 HSPZ_FOLD_RUN=1" arm fold-run-1 on "$R" "$Q"
MAXH=1700000 ENVV="$FB HSPZ_FOLD=1" arm fold-cut-batches on "$R" "$Q" -C 60000
MAXH=1700000 ENVV="$FB HSPZ_FOLD=1 HSPZ_DEVICE_SEEDS=1" arm fold-cut-batches-device-seeds on "$R" "$Q" -C 60000
MAXH=1000000 ENVV="$FB HSPZ_FOLD=1" arm fold-mostly-cut on "$R" "$Q" -C 60000
MAXH=1000000 ENVV="$FB HSPZ_FOLD=1 HSPZ_DEVICE_SEEDS=1" arm fold-mostly-cut-device-seeds on "$R" "$Q" -C 60000
MAXH=4000000 ENVV="$FB HSPZ_FOLD=1 HSPZ_FOLD_CAPACITY=3000000" arm fold-released on "$R" "$Q"
MAXH=4000000 ENVV="$FB HSPZ_FOLD=1 HSPZ_FOLD_CAPACITY=3000000 HSPZ_DEVICE_SEEDS=1" arm fold-released-device-seeds on "$R" "$Q"
ENVV="$FB HSPZ_FOLD=0" arm fold-off on "$R" "$Q"
for a in fold-check-24 fold-check-25 fold-iupac; do
  grep -q '^  transition fold: .* (checked against the host.s)' "$W/$a.1.err" || { echo "$a: the tables were not checked"; fail=$((fail+1)); }
done
for a in fold-check-24 fold-check-25 fold-bucketed-check fold-2400 fold-iupac fold-device-seeds fold-run-32 fold-run-1; do
  grep -q '^transition fold: 1 reference tables folded, [1-9][0-9]* batches on them, 0 sent again' "$W/$a.1.err" \
    || { echo "$a: the batches were not on the folded tables"; fail=$((fail+1)); }
done
grep -q '^bucket order from the seeds check: .* every bucket the same anchors' "$W/fold-bucketed-check.1.err" \
  || { echo "fold-bucketed-check: the check did not run"; fail=$((fail+1)); }
for a in fold-cut-batches fold-cut-batches-device-seeds; do
  grep -q '^transition fold: 1 reference tables folded, [1-9][0-9]* batches on them, [1-9][0-9]* sent again .* 0 tables dropped' "$W/$a.1.err" \
    || { echo "$a: not both folded batches and batches sent again"; fail=$((fail+1)); }
done
for a in fold-mostly-cut fold-mostly-cut-device-seeds; do
  grep -q '^transition fold: 1 reference tables folded, [0-9]* batches on them, 16 sent again .* 1 tables dropped' "$W/$a.1.err" \
    || { echo "$a: the fold was not dropped after 16 batches sent again"; fail=$((fail+1)); }
  grep -q '^  transition fold: dropped for this reference, a quarter of its batches' "$W/$a.1.err" \
    || { echo "$a: the fold was not dropped for its cut batches"; fail=$((fail+1)); }
done
for a in fold-released fold-released-device-seeds; do
  grep -q '^transition fold: 1 reference tables folded, 0 batches on them, 1 sent again .* 1 tables dropped' "$W/$a.1.err" \
    || { echo "$a: the tables were not dropped at the first cut batch"; fail=$((fail+1)); }
  grep -q '^  transition fold: dropped for this reference, a chunk needs its memory' "$W/$a.1.err" \
    || { echo "$a: the fold was not dropped for a chunk"; fail=$((fail+1)); }
done
grep -q 'transition fold' "$W/fold-off.1.err" && { echo "fold-off: the fold was built"; fail=$((fail+1)); }

echo
grep -L 'find_hits by groups check: .* equal to the per-seed kernel' "$W"/find-hits-check*.1.err 2>/dev/null | grep -q . \
  && { echo "find-hits-check: an arm did not report its check"; fail=$((fail+1)); }
if [ $fail -eq 0 ]; then echo "sieve-identity: PASS (every arm byte-identical, every check clean)"; else echo "sieve-identity: $fail FAILURE(S); logs in $W"; fi
exit $fail
