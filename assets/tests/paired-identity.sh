#!/usr/bin/env bash
# Identity gate for `run --paired` (ZLUDA or NVIDIA, a few minutes).
#
# Claim: the lines `run --paired` writes for pair i are byte-identical to what a
# standalone `run -S plus` of pair i's two records writes. Every arm compares
# the whole paired.segments against the per-pair standalone outputs
# concatenated in pair order (each pair's tmpN files in numeric order).
#
# Arms (same K, same flags on both sides):
#   A  defaults, every pair
#   B  -B small enough to force several batches, every pair (reuses A's runs)
#   C  -C small (several whole records per call), pairs that fit one chunk
#   D  --max-hits small, low-complexity pairs + neighbours: forces the
#      oversized-call re-issue (H stays above the largest per-seed count)
#   E  HSPZ_REF_BUCKETS=1 (bucket pass forced on; ZLUDA turns it off), every
#      pair (reuses A's runs)
#
# Fixture: assets/tests/gen_paired_fixture.py CHAIN.gz TARGET.fa QUERY.fa FIX 300 2000
# Usage: FIX=<fixture dir> assets/tests/paired-identity.sh   (H=binary W=workdir K=hspthresh)
set -uo pipefail
S=${TMPDIR:-/tmp}
# Machine paths are never committed: CUDA_HOME and ZLUDA come from the per-user
# file ${HSPZ_TEST_ENV:-${XDG_CONFIG_HOME:-$HOME/.config}/hspz/test.env}
# (plain KEY=value lines). Pre-exported values win over the file.
for _v in CUDA_HOME ZLUDA; do
  eval "[ \"\${$_v+x}\" ] && _s_$_v=\$$_v"
done
_f=${HSPZ_TEST_ENV:-${XDG_CONFIG_HOME:-$HOME/.config}/hspz/test.env}
[ -f "$_f" ] && . "$_f"
for _v in CUDA_HOME ZLUDA; do
  eval "[ \"\${_s_$_v+x}\" ] && $_v=\$_s_$_v"
done
FIX=${FIX:?set FIX to an assets/tests/gen_paired_fixture.py output dir}
FIX=$(cd "$FIX" && pwd)
H=${H:-$(cd "$(dirname "$0")/../.." && pwd)/target/release/hspZ}
K=${K:-2000}
export CUDA_HOME=${CUDA_HOME:?set CUDA_HOME (CUDA toolkit root) in the environment or ~/.config/hspz/test.env}
export LD_LIBRARY_PATH=${ZLUDA:+$ZLUDA:}$CUDA_HOME/lib
W=${W:-$S/hspz-paired-identity}; rm -rf "$W"; mkdir -p "$W"

row() { printf '%-46s %s\n' "$1" "$2"; }
fail=0

# expected <targets.fa> <queries.fa> <name> [flags...]: per-pair standalone runs,
# concatenated in pair order into $W/<name>.expected.
expected() {
  local t=$1 q=$2 name=$3; shift 3
  local d="$W/$name.pairs"; mkdir -p "$d"
  # One file per record, named by pair index, for both sides (awk keeps empty records).
  for side in t q; do
    local fa; [ $side = t ] && fa=$t || fa=$q
    awk -v d="$d" -v s=$side '/^>/ { n++; f = sprintf("%s/%06d.%s.fa", d, n, s) } { print > f }' "$fa"
  done
  : > "$W/$name.expected"
  for tf in "$d"/*.t.fa; do
    local i; i=$(basename "$tf" .t.fa)
    "$H" run -S plus --hspthresh "$K" -r "$tf" -q "$d/$i.q.fa" -o "$d/$i.out" "$@" > /dev/null 2> "$d/$i.err" \
      || { echo "standalone pair $i failed:"; tail -3 "$d/$i.err"; exit 1; }
    ls "$d/$i.out"/*.segments > /dev/null 2>&1 && cat $(ls "$d/$i.out"/*.segments | sort -V) >> "$W/$name.expected"
  done
}

# check <arm> <expected name> <targets.fa> <queries.fa> [flags...]
check() {
  local arm=$1 exp=$2 t=$3 q=$4; shift 4
  "$H" run --paired -S plus --hspthresh "$K" -r "$t" -q "$q" -o "$W/$arm" --time "$@" 2> "$W/$arm.err" \
    || { row "$arm" "FAIL (run): $(tail -1 "$W/$arm.err")"; fail=$((fail+1)); return; }
  local info; info=$(grep -o 'paired: [0-9]* pairs.*' "$W/$arm.err")
  if cmp -s "$W/$arm/paired.segments" "$W/$exp.expected"; then
    row "$arm" "MATCH $(wc -l < "$W/$exp.expected") lines; $info"
  else
    row "$arm" "DIFF  ($info)"
    diff "$W/$arm/paired.segments" "$W/$exp.expected" | head -4
    fail=$((fail+1))
  fi
}

expected "$FIX/targets.fa" "$FIX/queries.fa" all
expected "$FIX/c_targets.fa" "$FIX/c_queries.fa" c -C 2000
expected "$FIX/d_targets.fa" "$FIX/d_queries.fa" d --max-hits 1000000

check "A defaults"                 all "$FIX/targets.fa"   "$FIX/queries.fa"
check "B -B 200000 (several batches)" all "$FIX/targets.fa" "$FIX/queries.fa" -B 200000
check "C -C 2000"                  c   "$FIX/c_targets.fa" "$FIX/c_queries.fa" -C 2000
check "D --max-hits 1000000"       d   "$FIX/d_targets.fa" "$FIX/d_queries.fa" --max-hits 1000000
HSPZ_REF_BUCKETS=1 check "E HSPZ_REF_BUCKETS=1" all "$FIX/targets.fa" "$FIX/queries.fa"

echo "mismatches: $fail"
exit $fail
