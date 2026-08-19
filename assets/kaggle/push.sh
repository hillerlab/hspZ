#!/usr/bin/env bash
# One command: push a stage to Kaggle, follow it, download its output.
#
#   assets/kaggle/push.sh sweep            # six-pair correctness sweep   (~30 min with queue)
#   assets/kaggle/push.sh fast             # sweep + a 1,2,2,1 set on the big pair  (~1.5 h)
#   WAIT=1 assets/kaggle/push.sh sweep     # block until done, then download (what CI uses)
#
# The kernel clones this repository at HSPZ_SHA, so *push your commit first* — a
# local-only change will not be in the run. HSPZ_SHA defaults to HEAD, and the push
# refuses if HEAD is not on the remote, because silently benchmarking an older tree
# is the worst possible failure here.
set -euo pipefail
STAGE=${1:-sweep}
case $STAGE in fast|sweep) ;; *) echo "unknown stage: $STAGE (expected sweep or fast)" >&2; exit 2;; esac
KERNEL=${KERNEL:-alejandrogzi/hspz-multigpu}
cd "$(dirname "$0")/../.."
SHA=${HSPZ_SHA:-$(git rev-parse HEAD)}

# `git branch -r --contains` exits 0 with *empty* output when no remote branch has the
# commit, so the exit status is not the test — the output is. Kernel v20 died 2.4 s in
# on `git checkout <sha>` because of this, after the commit landed on a branch that had
# not been pushed.
if [ -z "$(git branch -r --contains "$SHA" 2>/dev/null)" ]; then
    echo "push: $SHA is on no remote branch — push the branch first, or set HSPZ_SHA" >&2
    echo "      (local branches with it: $(git branch --contains "$SHA" | tr -d ' *' | paste -sd,))" >&2
    exit 1
fi
if [ -n "$(git status --porcelain -- src assets/kaggle assets/tests)" ]; then
    echo "push: WARNING working tree has uncommitted changes; the kernel will run $SHA" >&2
fi

k=assets/kaggle/kernel
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
cp "$k/kernel-metadata.json" "$k/run.py" "$tmp/"
# The stage and the pinned SHA travel as literals in the script: Kaggle script
# kernels take no arguments, and a kernel that reads them from the environment
# would silently run the wrong stage on a rerun from the web UI.
python3 - "$tmp/run.py" "$STAGE" "$SHA" <<'PY'
import sys
p, stage, sha = sys.argv[1:4]
s = open(p).read()
before = s
s = s.replace('SHA = os.environ.get("HSPZ_SHA", "")', f'SHA = os.environ.get("HSPZ_SHA", "{sha}")')
s = s.replace('STAGE = os.environ.get("HSPZ_STAGE", "sweep")', f'STAGE = os.environ.get("HSPZ_STAGE", "{stage}")')
if s == before:
    raise SystemExit("push: neither SHA nor STAGE was substituted -- run.py's literals moved")
open(p, "w").write(s)
PY

# The accelerator travels in kernel-metadata.json as `machine_shape`, and the value
# is case-sensitive: `NvidiaTeslaT4` (Kaggle's T4 option is a pair), NvidiaTeslaP100,
# Tpu1VmV38 — the list is in kagglesdk's KernelPushRequest docstring, not in any
# public doc page. Lower-cased spellings are accepted and silently ignored, which
# costs a run on the default single P100. ACCEL=... overrides for probing.
echo "pushing $KERNEL  stage=$STAGE  sha=$SHA${ACCEL:+  accelerator=$ACCEL}"
kaggle kernels push -p "$tmp" ${ACCEL:+--accelerator "$ACCEL"}

# Report what the server will actually run on, before anyone reads a timing.
pull=$(mktemp -d); trap 'rm -rf "$tmp" "$pull"' EXIT
if kaggle kernels pull "$KERNEL" -p "$pull" -m >/dev/null 2>&1; then
    shape=$(python3 -c "import json;print(json.load(open('$pull/kernel-metadata.json')).get('machine_shape','?'))")
    echo "machine_shape: $shape"
    if [ "$shape" != "NvidiaTeslaT4" ]; then
        echo "push: WARNING machine_shape is '$shape', not NvidiaTeslaT4 (= T4 x2)." >&2
        echo "      The environment gate will VOID the run; fix the metadata." >&2
    fi
fi
# WAIT=1 turns this into a blocking, self-checking run: poll to completion, pull the logs and
# the output, and fail on a kernel error. CI wants one command, and so does a human who would
# otherwise sit refreshing the web UI. Kaggle reports a *finished* kernel as "complete" even
# when its own gates recorded FAIL, so the caller still has to read correctness.tsv.
if [ "${WAIT:-0}" = 1 ]; then
    echo "waiting for $KERNEL (a fast stage is ~1.5 h; a sweep ~30 min including the queue)"
    status=
    for _ in $(seq 1 240); do
        sleep 60
        status=$(kaggle kernels status "$KERNEL" 2>&1 || true)
        echo "  $status"
        case $status in
            *complete*) break ;;
            *error*|*cancel*) echo "push: kernel did not finish cleanly" >&2; break ;;
        esac
    done
    mkdir -p "results/kaggle/$STAGE"
    kaggle kernels logs "$KERNEL" > "results/kaggle/$STAGE/kernel.log" 2>&1 || true
    kaggle kernels output "$KERNEL" -p "results/kaggle/$STAGE" --force || {
        echo "push: could not download the output" >&2; exit 1; }
    case $status in *complete*) ;; *) exit 1 ;; esac
    echo "collected into results/kaggle/$STAGE"
    exit 0
fi

echo "following (ctrl-c is safe — the job keeps running):"
sleep 20
kaggle kernels status "$KERNEL" || true
kaggle kernels output "$KERNEL" -p "results/kaggle/$STAGE" 2>/dev/null || true
cat <<TXT

Collect later, from anywhere:
  kaggle kernels status $KERNEL
  kaggle kernels logs   $KERNEL
  kaggle kernels output $KERNEL -p results/kaggle/$STAGE
TXT
