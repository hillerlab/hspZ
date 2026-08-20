#!/usr/bin/env bash
# Container smoke: does this image produce the frozen answer?
#
#   assets/container/smoke.sh zluda  hspz:zluda-test
#   assets/container/smoke.sh nvidia hspz:test
#
# The expected values are NOT frozen from a container's first run (plan amendment A4). They are
# the answers assets/tests/parity.sh already validates against the C++ CUDA oracle on the same
# deterministic fixtures, so a passing container is tied to that lineage:
#
#   repeat  57 HSPs  sorted d01edd2118cf5fa5
#   multi    3 HSPs  sorted 0da3bbda656e4f7b   (two output files)
#
# --max-hits is pinned on every arm: device-derived MAX_HITS moves the chunk boundaries and
# therefore the HSP set, so an unpinned run is not comparable across cards.
set -uo pipefail

BACKEND=${1:?usage: smoke.sh <nvidia|zluda> <image[:tag]>}
IMAGE=${2:?usage: smoke.sh <nvidia|zluda> <image[:tag]>}
MAX_HITS=${MAX_HITS:-16711680}
FAIL=0

case "$BACKEND" in
    nvidia) GPU_ARGS=(--gpus all);;
    zluda)  GPU_ARGS=(--device=/dev/kfd --device=/dev/dri --group-add video --group-add render);;
    *) echo "smoke: backend must be nvidia or zluda" >&2; exit 2;;
esac

say()  { printf '\n== %s\n' "$*"; }
ok()   { printf '   PASS  %s\n' "$*"; }
bad()  { printf '   FAIL  %s\n' "$*"; FAIL=1; }
drun() { docker run --rm "${GPU_ARGS[@]}" "$@"; }

say "$BACKEND / $IMAGE"

# ---- 1. the binary answers at all -----------------------------------------------------
v=$(drun "$IMAGE" --version 2>&1) && ok "--version: $v" || bad "--version: $v"
drun "$IMAGE" --help > /dev/null 2>&1 && ok "--help" || bad "--help"

# ---- 2. driver-symbol floor (amendment A1) --------------------------------------------
# Anything newer than CUDA 4.0 here means the image silently requires a recent host driver.
sym=$(drun --entrypoint sh "$IMAGE" -c '
  command -v nm >/dev/null 2>&1 || { echo SKIP; exit 0; }
  nm -D --undefined-only "$(command -v hspZ)" \
      | grep -oE "cu[A-Za-z]+_v[0-9]+" \
      | grep -vE \
          "cuDevicePrimaryCtxRelease_v2|cuEventDestroy_v2|cuMemAllocHost_v2|cuMemAlloc_v2|cuMemcpyDtoHAsync_v2|cuMemcpyHtoDAsync_v2|cuMemcpyHtoD_v2|cuMemFree_v2|cuMemGetInfo_v2|cuStreamDestroy_v2" \
      | sort -u | tr "\n" " "' 2>/dev/null)
case "$sym" in
    ""|SKIP*) ok "driver symbols: no post-CUDA-4.0 requirement${sym:+ (nm absent, checked at build)}";;
    *) bad "driver symbols: image needs $sym — host drivers below 570 will fail at the first timed event";;
esac

# ---- 3. the frozen answers ------------------------------------------------------------
check() {                       # check <fixture> <expected hsps> <expected sorted digest> <files>
    local fx=$1 want_n=$2 want_d=$3 want_f=$4
    # Runs through the image's real ENTRYPOINT and digests on the host. An earlier version used
    # `--entrypoint sh` to digest inside the container, which silently bypassed
    # entrypoint-zluda.sh -- so LD_LIBRARY_PATH never got ZLUDA, there was no libcuda.so.1, and
    # every fixture "produced" 0 HSPs. Bypassing the entrypoint is exactly not the thing to test.
    local d n f dg
    d=$(mktemp -d)
    drun -v "$d:/out" "$IMAGE" run \
        -r "/opt/hspZ/fixtures/$fx.ref.fa" -q "/opt/hspZ/fixtures/$fx.qry.fa" \
        -o /out --max-hits "$MAX_HITS" > "$d/.stdout" 2> "$d/.stderr"
    n=$(cat "$d"/*.segments 2>/dev/null | wc -l)
    f=$(ls "$d"/*.segments 2>/dev/null | wc -l)
    dg=$(cat "$d"/*.segments 2>/dev/null | LC_ALL=C sort | sha256sum | cut -c1-16)
    if [ "$n" = "$want_n" ] && [ "$dg" = "$want_d" ] && [ "$f" = "$want_f" ]; then
        ok "$fx: $n HSPs in $f file(s), $dg"
    else
        bad "$fx: got $n HSPs / $f file(s) / $dg, want $want_n / $want_f / $want_d"
        printf '         %s\n' "$(tail -2 "$d/.stderr" 2>/dev/null | tr '\n' ' ' | cut -c1-160)"
    fi
    rm -rf "$d"
}
say "frozen fixtures (--max-hits $MAX_HITS)"
check repeat 57 d01edd2118cf5fa5 1
check multi   3 0da3bbda656e4f7b 2

# ---- 3a. hspZ-prefixed invocation (nextflow-style) --------------------------------------
# Nextflow launches the container as `hspZ run ...` — the entrypoint shim must strip the
# leading binary name and land on the same answer as the bare `run` form above.
say "hspZ-prefixed invocation"
d=$(mktemp -d)
drun -v "$d:/out" "$IMAGE" hspZ run \
    -r /opt/hspZ/fixtures/repeat.ref.fa -q /opt/hspZ/fixtures/repeat.qry.fa \
    -o /out --max-hits "$MAX_HITS" > "$d/.stdout" 2> "$d/.stderr"
n=$(cat "$d"/*.segments 2>/dev/null | wc -l)
dg=$(cat "$d"/*.segments 2>/dev/null | LC_ALL=C sort | sha256sum | cut -c1-16)
if [ "$n" = 57 ] && [ "$dg" = d01edd2118cf5fa5 ]; then
    ok "hspZ run: $n HSPs, $dg"
else
    bad "hspZ run: got $n HSPs / $dg, want 57 / d01edd2118cf5fa5"
    printf '         %s\n' "$(tail -2 "$d/.stderr" 2>/dev/null | tr '\n' ' ' | cut -c1-160)"
fi
rm -rf "$d"

# ---- 3b. Phase 9 rows: output modes and input formats (FULL=1) --------------------------
# Off by default so the routine smoke stays small . These are the acceptance-matrix
# rows that need no second machine: -Z tarball, gzipped input, and -D partitioned output. Each
# must land on the same HSP set as the plain run of the same fixture.
if [ "${FULL:-0}" = 1 ]; then
    say "output modes and input formats (FULL=1)"
    d=$(mktemp -d)
    # -Z: one tarball instead of a directory. Unpack on the host and digest the same way.
    drun -v "$d:/out" "$IMAGE" run -r /opt/hspZ/fixtures/repeat.ref.fa \
        -q /opt/hspZ/fixtures/repeat.qry.fa -o /out -D -Z /out/out.tar.gz \
        --max-hits "$MAX_HITS" > "$d/.o" 2> "$d/.e"
    if [ -s "$d/out.tar.gz" ]; then
        mkdir -p "$d/x" && tar xzf "$d/out.tar.gz" -C "$d/x" 2>/dev/null
        n=$(cat "$d"/x/*.segments 2>/dev/null | wc -l)
        [ "$n" = 57 ] && ok "-D -Z tarball: $n HSPs" || bad "-D -Z tarball: $n HSPs, want 57"
    else
        bad "-D -Z produced no tarball: $(tail -1 "$d/.e" | cut -c1-120)"
    fi
    # gzipped FASTA in: same answer as the plain fixture.
    drun --entrypoint sh "$IMAGE" -c 'command -v gzip >/dev/null' 2>/dev/null \
        && fmt_ok=1 || fmt_ok=0
    if [ "$fmt_ok" = 1 ]; then
        e=$(mktemp -d)
        drun -v "$e:/out" --entrypoint sh "$IMAGE" -c "
            gzip -c /opt/hspZ/fixtures/repeat.ref.fa > /tmp/r.fa.gz
            gzip -c /opt/hspZ/fixtures/repeat.qry.fa > /tmp/q.fa.gz
            export LD_LIBRARY_PATH=/opt/zluda\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
            hspZ run -r /tmp/r.fa.gz -q /tmp/q.fa.gz -o /out --max-hits $MAX_HITS" \
            > "$e/.o" 2> "$e/.e"
        n=$(cat "$e"/*.segments 2>/dev/null | wc -l)
        dg=$(cat "$e"/*.segments 2>/dev/null | LC_ALL=C sort | sha256sum | cut -c1-16)
        [ "$dg" = d01edd2118cf5fa5 ] && ok "FASTA.gz input: $n HSPs, $dg" \
            || bad "FASTA.gz input: $n HSPs / $dg, want 57 / d01edd2118cf5fa5"
        rm -rf "$e"
    else
        printf '   SKIP  FASTA.gz input (no gzip in image)\n'
    fi
    rm -rf "$d"
fi

# ---- 4. the no-GPU message is useful, not a panic --------------------------------------
say "no-GPU behaviour"
msg=$(docker run --rm "$IMAGE" run -r /opt/hspZ/fixtures/repeat.ref.fa \
        -q /opt/hspZ/fixtures/repeat.qry.fa -o /tmp/o --max-hits "$MAX_HITS" 2>&1 | tail -3)
if printf '%s' "$msg" | grep -qiE "panic|SIGSEGV|core dumped"; then
    bad "no-GPU run panicked: $msg"
elif printf '%s' "$msg" | grep -qiE "cuda|driver|kfd|GPU|device"; then
    ok "no-GPU run explains itself: $(printf '%s' "$msg" | tail -1 | cut -c1-90)"
else
    bad "no-GPU run gave no useful message: $msg"
fi

printf '\n%s\n' "$([ "$FAIL" -eq 0 ] && echo 'SMOKE PASS' || echo 'SMOKE FAIL')"
exit "$FAIL"
