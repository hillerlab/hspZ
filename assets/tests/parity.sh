#!/usr/bin/env bash
# Exact-parity harness.
#
# Runs the KegAlign C++/CUDA oracle and hspZ on every fixture and requires
# byte-identical .segments output. Fixture levels:
#   1. tiny synthetic  - generated here, covers the parity checklist
#   2. small genomes   - KegAlign's own apple/orange test data
#   3. large fragment  - any ref/query pair passed as $1 $2
#
# Usage: assets/tests/parity.sh [big_ref.fa big_query.fa]
set -uo pipefail

cd "$(dirname "$0")/../.."
ROOT=$PWD
HSPZ=${HSPZ:-$ROOT/target/release/hspZ}
KEGALIGN=${KEGALIGN:-/tmp/kegalign/build/kegalign}
TESTDATA=${TESTDATA:-/tmp/kegalign/test-data}
WORK=${WORK:-$(mktemp -d /tmp/hspz-parity-XXXXXX)}

CUDACONDA=${CUDACONDA:-/home/alejandro/opt/cudaconda}
export LD_LIBRARY_PATH=${ZLUDA:-/home/alejandro/opt/zluda}:$CUDACONDA/lib
# The Thrust/CUB oracle aborts (rc=134) under ZLUDA without a legacy-stream shim.
# `hspZ compare` no longer defaults this -- a developer path must not ship in the
# published container's --help -- so the harness supplies it, and skips it when absent
# so a native-CUDA host needs no shim at all.
HIPFIX=${HIPFIX:-/home/alejandro/opt/zluda-guide/hipfix.so}
preload=()
[ -f "$HIPFIX" ] && preload=(--ld-preload "$HIPFIX")
# cuda-bindings' build script needs cuda.h; without CUDA_HOME it probes
# /usr/local/cuda, fails, and the unit suite silently does not run.
export CUDA_HOME=${CUDA_HOME:-$CUDACONDA}

[ -x "$HSPZ" ]     || { echo "no hspZ binary at $HSPZ - run: cargo oxide build -- --release"; exit 1; }
[ -x "$KEGALIGN" ] || { echo "no oracle at $KEGALIGN"; exit 1; }

echo "workdir: $WORK"
mkdir -p "$WORK/fixtures"
FIXTURES=$(python3 assets/tests/gen_fixtures.py "$WORK/fixtures")

pass=0; fail=0; env_fail=0; failed=(); envfailed=()

check() {  # name ref query [scoring-matrix]
    local name=$1 ref=$2 qry=$3 scoring=${4:-}
    local extra=()
    [ -n "$scoring" ] && extra=(--scoring "$scoring")
    printf '%-24s ' "$name"
    local log="$WORK/$name.log"
    if "$HSPZ" compare --reference "$ref" --query "$qry" "${extra[@]}" "${preload[@]}" \
            --kegalign "$KEGALIGN" --workdir "$WORK/$name" >"$log" 2>&1; then
        printf 'PASS  %s\n' "$(grep -o 'IDENTICAL.*' "$log" | head -1)"
        pass=$((pass+1))
    elif grep -q 'kegalign exited signal' "$log"; then
        # an oracle abort is an environment failure, not an
        # algorithm result — but recognition stays narrow. The verified
        # signature is a teardown crash *after* complete output; an abort with
        # no output proves nothing either way and is reported as unverified.
        sig=$(grep -o 'signal: [0-9]* ([A-Z]*)' "$log" | head -1)
        if [ "$(ls "$WORK/$name/cpp" 2>/dev/null | grep -c '\.segments$')" -gt 0 ]; then
            printf 'ENV   oracle teardown after valid output (%s)\n' "$sig"
        else
            printf 'ENV?  oracle aborted with NO output, unverified (%s)\n' "$sig"
        fi
        env_fail=$((env_fail+1)); envfailed+=("$name")
    else
        printf 'FAIL\n'
        sed -n '/HSP parity/,$p' "$log" | head -8 | sed 's/^/    /'
        grep -E '^(error|Rust |C\+\+ )' "$log" | head -4 | sed 's/^/    /'
        fail=$((fail+1)); failed+=("$name")
    fi
}

echo "--- unit checks ---"
# The test binary links the embedded device bundle, so it has to go through the
# oxide backend like the main build does.
# Piping into grep hid a build failure behind a zero exit status once, which
# reported "all checks passed" with the unit suite never run. Capture, then judge.
if cargo oxide test -- --release > "$WORK/unit.log" 2>&1; then
    grep -E 'test result' "$WORK/unit.log" | tail -3
    pass=$((pass+1))
else
    printf '%-24s FAIL  cargo oxide test failed (%s)\n' unit_tests "$WORK/unit.log"
    grep -E 'error|FAILED|^test .* FAILED' "$WORK/unit.log" | head -6 | sed 's/^/    /'
    fail=$((fail+1)); failed+=(unit_tests)
fi
# The multiplicity gate is the correctness authority for multi-bin output, so it
# is itself tested: a gate that always passes is worse than no gate.
assets/tests/multiplicity_gate.sh --self-test || { echo "gate self-test FAILED"; exit 1; }

echo "--- level 1: tiny synthetic ---"
for f in $FIXTURES; do
    matrix=""
    [ -f "$WORK/fixtures/$f.matrix" ] && matrix="$WORK/fixtures/$f.matrix"
    check "$f" "$WORK/fixtures/$f.ref.fa" "$WORK/fixtures/$f.qry.fa" "$matrix"
done

echo "--- max_hits chunking fallback ---"
# The device scan skips the cumulative-array copy whenever num_hits <= max_hits,
# which is every normal run; a tiny --max-hits forces the exact fallback. The
# extension work must be partition-invariant (same raw HSPs), and the run must be
# deterministic. The FINAL count legitimately differs: KegAlign dedups per chunk,
# so the partition is semantically load-bearing, not just a memory bound.
#
# Guarded on the test data, and on a non-empty raw count. Without the guard this
# check passed vacuously in any fresh environment: with no apple.fasta.gz both arms
# produce empty output, so `[ "$raw_def" = "$raw_cap" ]` and `diff -r` of two empty
# directories both succeed and it printed `PASS ... ()`. The non-empty assertion
# covers the same failure with the data present but hspZ crashing.
if [ -f "$TESTDATA/apple.fasta.gz" ] && [ -f "$TESTDATA/orange.fasta.gz" ]; then
    fb() { "$HSPZ" run --reference "$TESTDATA/apple.fasta.gz" --query "$TESTDATA/orange.fasta.gz" \
            --output "$1" ${2:+--max-hits $2} 2>&1 | tr -d ' '; }
    mkdir -p "$WORK/fb_def" "$WORK/fb_cap" "$WORK/fb_cap2"
    raw_def=$(fb "$WORK/fb_def" | grep -o '#rawHSPs:[0-9]*')
    raw_cap=$(fb "$WORK/fb_cap" 1000 | grep -o '#rawHSPs:[0-9]*')
    fb "$WORK/fb_cap2" 1000 >/dev/null
    printf '%-24s ' "max_hits_fallback"
    if [ -n "$raw_def" ] && [ "$raw_def" = "$raw_cap" ] \
       && diff -r "$WORK/fb_cap" "$WORK/fb_cap2" >/dev/null; then
        printf 'PASS  raw HSPs partition-invariant (%s), chunked run deterministic\n' "$raw_cap"
        pass=$((pass+1))
    else
        printf 'FAIL  raw def=%s cap=%s\n' "${raw_def:-<empty>}" "${raw_cap:-<empty>}"
        fail=$((fail+1)); failed+=("max_hits_fallback")
    fi
else
    printf '%-24s %s\n' "max_hits_fallback" "skipped (no \$TESTDATA)"
fi

echo "--- level 2: small genomes ---"
if [ -f "$TESTDATA/apple.fasta.gz" ]; then
    check apple_orange "$TESTDATA/apple.fasta.gz" "$TESTDATA/orange.fasta.gz"
else
    echo "  skipped (no $TESTDATA)"
fi

echo "--- level 2.5: multi-block vs single-block ---"
# 6x4 bins over a repetitive fixture, two properties, both permanent:
#   a) with no max_hits chunking, multi-bin output must equal single-block
#      EXACTLY. One record per bin means zero internal SEPs, which is the
#      strongest SEP-layout independence test.
#   b) when chunking IS forced, the two arms may differ only in duplicate
#      multiplicity — the disposition-A gate: distinct anchors byte-identical and
#      every difference an exact duplicate of a kept anchor. Chunk boundaries
#      decide which side keeps the spare copy, so neither arm is the clean one.
# Repetitive, because chunking needs many hits per seed: a shared motif pool is
# what pushes a 20 kbp record past --max-hits 400000.
python3 - "$WORK/rep_ref.fa" "$WORK/rep_qry.fa" <<'PY'
import random, sys
random.seed(11)
pool = ["".join(random.choice("ACGT") for _ in range(200)) for _ in range(40)]
for path, seed, nrec in ((sys.argv[1], 7, 6), (sys.argv[2], 23, 4)):
    random.seed(seed)
    with open(path, "w") as f:
        for c in range(nrec):
            s = ""
            while len(s) < 20000:
                m = list(random.choice(pool))
                for _ in range(2):
                    m[random.randrange(len(m))] = random.choice("ACGT")
                s += "".join(m)
            f.write(f">chr{c}\n{s[:20000]}\n")
PY
for arm in single multi; do
    bs=1000000; [ "$arm" = multi ] && bs=20000
    for cap in "" 400000; do
        d="$WORK/mb_$arm$cap"; rm -rf "$d"; mkdir -p "$d"
        "$HSPZ" run --reference "$WORK/rep_ref.fa" --query "$WORK/rep_qry.fa" \
            --output "$d" --seq-block-size $bs ${cap:+--max-hits $cap} >/dev/null 2>&1
    done
done
# Four switches that must be invisible. --no-ref-prefetch only moves *when* the
# next bin's pack + seed table is built, --no-async-stages only moves *when* the host
# waits for the GPU, and --no-async-seed-copy only moves *which stream* carries the
# seed upload and how far ahead it is issued (all three shipped on; an earlier round
# measured the pair at -1.51% on an L4). `--gpus 2` is the spec: two workers own three
# reference bins each and the output must still be identical, because the emitter
# replays units by ordinal rather than by completion order. On a one-GPU box the two
# workers time-slice it, which is a correctness configuration, not a fast one.
for extra in --no-ref-prefetch --no-async-stages --no-async-seed-copy "--gpus 2"; do
    d="$WORK/mb_multi$(echo "$extra" | tr -d ' -')"; rm -rf "$d"; mkdir -p "$d"
    # shellcheck disable=SC2086
    "$HSPZ" run --reference "$WORK/rep_ref.fa" --query "$WORK/rep_qry.fa" --output "$d" \
        --seq-block-size 20000 $extra >/dev/null 2>&1
done
for c in "multiblock:mb_single:mb_multi:exact" \
         "multiblock_chunked:mb_single400000:mb_multi400000:gate" \
         "ref_prefetch:mb_multi:mb_multinorefprefetch:exact" \
         "no_async_stages:mb_multi:mb_multinoasyncstages:exact" \
         "no_async_seed_copy:mb_multi:mb_multinoasyncseedcopy:exact" \
         "two_gpu_workers:mb_multi:mb_multigpus2:exact"; do
    IFS=: read -r name a b req <<< "$c"
    out=$(assets/tests/multiplicity_gate.sh "$name" "$WORK/$a" "$WORK/$b"); rc=$?
    verdict=$(printf '%s' "$out" | head -1 | sed "s/^gate  *$name  *//")
    printf '%-24s ' "$name"
    if [ $rc -ne 0 ]; then
        printf 'FAIL  %s\n' "$verdict"
        fail=$((fail+1)); failed+=("$name")
    elif [ "$req" = exact ] && [ "${verdict:0:5}" != EXACT ]; then
        # Without chunking there is no known mechanism for a difference, so
        # duplicates-only here would be a real regression, not the accepted case.
        printf 'FAIL  unchunked multi-bin must be exact: %s\n' "$verdict"
        fail=$((fail+1)); failed+=("$name")
    else
        printf 'PASS  %s\n' "$verdict"
        [ "$req" = gate ] && [ "${verdict:0:5}" = EXACT ] &&
            printf '%-24s NOTE  fixture no longer forces chunking; gate path unexercised\n' ""
        pass=$((pass+1))
    fi
done

echo "--- level 3: chromosome fragment ---"
if [ $# -ge 2 ]; then
    check "$(basename "$1")" "$1" "$2"
else
    echo "  skipped (pass ref.fa query.fa to run it)"
fi

echo
echo "$pass passed, $fail failed, $env_fail environment"
[ $env_fail -eq 0 ] || printf 'environment failures (oracle aborted, rerun): %s\n' "${envfailed[*]}"
if [ $fail -ne 0 ]; then
    printf 'failed: %s\n' "${failed[*]}"
    echo "note: an unpatched oracle differs on entropy-corrected scores near a"
    echo "      sequence end - see assets/tests/kegalign-uninit-rchr.patch"
    exit 1
fi
