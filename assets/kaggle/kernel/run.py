#!/usr/bin/env python3
"""hspZ GPU validation on Kaggle 2x Tesla T4.

A script kernel, and Kaggle uploads exactly one file for those, so the build, the arms, the
gates and the report all live here. The repository is cloned only for the Rust source at a
pinned SHA, which push.sh writes into this file before uploading.

Stages (STAGE below, set by push.sh):
    sweep   the six species pairs, one 1-GPU arm each, every one gated against a frozen
            per-workload digest. ~12 min including the build.
    fast    sweep, plus a 1,2,2,1 set on the 49-work-unit pair in `-D -Z` mode, so
            multi-GPU byte-identity is checked where there is real work to divide. ~1.5 h.

Every gate raises. A benchmark that quietly ran on one GPU is worse than one that failed,
and an EXACT verdict over zero files is worse still.
"""

import csv
import json
import os
import pathlib
import resource
import shutil
import statistics
import subprocess
import time

WORK = pathlib.Path("/kaggle/working")
LOGS = WORK / "logs"
SRC = pathlib.Path("/kaggle/tmp/hspZ")
SCRATCH = pathlib.Path("/kaggle/tmp/bench")
DATASET = pathlib.Path(
    os.environ.get("HSPZ_DATASET", "/kaggle/input/hspz-multigpu-benchmark")
)

REPO = os.environ.get("HSPZ_REPO", "https://github.com/hillerlab/hspZ.git")
SHA = os.environ.get("HSPZ_SHA", "")  # push.sh pins this
STAGE = os.environ.get("HSPZ_STAGE", "sweep")  # push.sh pins this
# The `version` stage's reference point: the SHA every accepted T4x2 record was
BLOCK = os.environ.get("HSPZ_BLOCK", "40000000")  # every record its own bin -> 7x7
# The small pairs are one 60-64 Mbp record against one 4 Mbp record. Split the
# reference into ~4 bins so a 2-GPU arm actually has work for both devices and the
# comparison means something — at one work unit the second worker would idle and the
# "1 vs 2 GPUs" gate would prove nothing. Both arms of a pair use the same block size,
# so their MAX_HITS chunk scope is identical and the comparison stays exact.
SMALL_BLOCK = os.environ.get("HSPZ_SMALL_BLOCK", "20000000")
THREADS = os.environ.get("HSPZ_THREADS", str(os.cpu_count() or 4))

# Frozen by the accepted T4x2 jobs on this exact 7x7 dataset/block plan. A new
# performance arm must reproduce it before its timing is considered.
CANON_HSPS = 2_235_722
CANON_DISTINCT = 2_235_673
CANON_DIGEST = "bcc0dff63ebccf2c"

# The small species-pair suite (round 49). Each is a whole 60-64 Mbp chromosome
# against the 4 Mbp window that actually aligns to it, so they cost seconds on a T4
# and still exercise the real path. Their point is divergence coverage: every
# optimization before round 49 was decided on human x mouse alone, where only 0.017%
# of hits survive the score gate. `dog-dog` survives at a different order of
# magnitude, and a change that helps one and hurts the other would have gone unseen.
WORKLOADS = [
    ("A_hs_mm", "hg38.chr20.fa", "mm39.chr19.mid4M.fa"),
    ("B_hs_mm", "hg38.chr19.fa", "mm39.chr17.mid4M.fa"),
    ("C_hs_bat", "hg38.chr20.fa", "HLmyoMyo6.m19p13.a.fa"),
    ("D_mm_bat", "mm39.chr19.fa", "HLmyoMyo6.m19p13.b.fa"),
    ("E_dog_bat", "canFam6.chr14.fa", "HLmyoMyo6.m19p11.fa"),
    ("F_dog_dog", "canFam6.chr14.fa", "ROSCfam.chr14.mid4M.fa"),
]
# Per-workload T4 output, frozen from kernel v23 (SHA 2a6b716, the first run after the
# alignment fix). A T4 and an L4 derive different MAX_HITS and therefore different chunk
# boundaries, so these are device-class constants: F differs from the local ZLUDA run by
# 11 duplicate copies for exactly that reason. Comparing a run against these is a
# cross-session gate, which is worth more than comparing two arms of the same run.
WORKLOAD_CANON = {
    "A_hs_mm": (4248, 4248, "852ce647a1f9cd74"),
    "B_hs_mm": (8908, 8908, "6b7e1d28f6fc0c57"),
    "C_hs_bat": (6547, 6547, "34ece3bf7c883391"),
    "D_mm_bat": (534, 533, "a28ed6155a2da6d3"),
    "E_dog_bat": (6638, 6636, "040289bd816ab8c7"),
    "F_dog_dog": (19087, 19076, "8076da1fedb7c262"),
}


def sh(cmd, check=True, **kw):
    print(f"+ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check, text=True, **kw)


def out(cmd):
    return subprocess.run(
        cmd, shell=True, text=True, capture_output=True
    ).stdout.strip()


def gate_environment():
    LOGS.mkdir(parents=True, exist_ok=True)
    listing, smi = out("nvidia-smi -L"), out("nvidia-smi")
    names = out("nvidia-smi --query-gpu=name,memory.total,uuid --format=csv,noheader")
    lines = [
        f"date            {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"stage           {STAGE}",
        f"repo            {REPO}",
        f"sha             {SHA or '(branch tip)'}",
        f"block           {BLOCK}",
        f"threads         {THREADS}",
        f"CUDA_VISIBLE_DEVICES {os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}",
        f"cpus            {os.cpu_count()}",
        f"ram_gb          {os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9:.1f}",
        f"disk_free_gb    {shutil.disk_usage('/kaggle').free / 1e9:.1f}",
        f"driver          {out('nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1')}",
        f"nvcc            {out('nvcc --version | tail -2 | head -1')}",
        "",
        listing,
        "",
        names,
        "",
        smi,
    ]
    (WORK / "environment.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]), flush=True)

    gpus = [l for l in listing.splitlines() if l.startswith("GPU ")]
    if len(gpus) != 2:
        raise RuntimeError(
            f"VOID: expected two physical GPUs, nvidia-smi -L shows {len(gpus)}"
        )
    uuids = {l.rsplit("UUID: ", 1)[-1].rstrip(")") for l in gpus}
    if len(uuids) != 2:
        raise RuntimeError(
            f"VOID: entries share a UUID ({uuids}) — not distinct devices"
        )
    if not all("T4" in l for l in gpus):
        raise RuntimeError(f"VOID: expected Tesla T4s, got {gpus}")
    for line in names.splitlines():
        if int(line.split(",")[1].strip().split()[0]) < 14000:
            raise RuntimeError(f"VOID: {line} — expected ~15 GB per device")
    print("gate: two distinct Tesla T4s, >=14 GB each — PASS", flush=True)


def threads_for(gpus, variant="baseline"):
    """The `--threads` value that gives the machine THREADS total.

    `--threads` became a machine budget divided across workers (amendment A). On a
    SHA that predates it every worker resolved the full value independently, so a
    2-GPU arm took twice the cores it asked for — and the scaling number would then
    be blamed on Kaggle's 4 cores rather than on the budget bug. Detect which
    behaviour the benchmarked source has and pass whichever value means "THREADS
    total", so both SHAs are measured under the same host budget.
    """
    total = int(THREADS)
    src = SRCS.get(variant, SRC)
    # Test for the division itself, not for the substring "per_worker": the older
    # SHA also contains `est.per_worker_prefetch` in its host-budget report, so a
    # substring test says every tree has amendment A. Under that false positive both
    # SHAs got --threads 4 at 2 GPUs, which is 4 per worker (8 threads on 4 cores)
    # on a tree without the division — the confound this stage exists to avoid.
    # Jobs 2-4 (kernel v17-v19) ran with it; their 2-GPU arms were oversubscribed.
    src_run = (src / "src/run.rs").read_text()
    per_worker_budget = "resolve_threads(args.threads) / workers" in src_run
    return str(total if per_worker_budget else max(1, total // max(1, gpus)))


def checkout(path, sha):
    """Clone at a pinned SHA and give that checkout the T4's arch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(path, ignore_errors=True)
    sh(f"git clone --quiet {REPO} {path}")
    if sha:
        sh(f"git -C {path} checkout --quiet {sha}")
    (path / ".cargo").mkdir(exist_ok=True)
    (path / ".cargo/cuda-oxide.toml").write_text(
        "# Kaggle: Tesla T4 is compute capability 7.5; sm_80 PTX will not load here.\n"
        'default-arch = "sm_75"\n'
    )
    return out(f"git -C {path} rev-parse HEAD")


def build():
    head = checkout(SRC, SHA)
    t = time.time()
    if not shutil.which("rustup"):
        sh(
            "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none"
        )
    os.environ["PATH"] = f"{pathlib.Path.home()}/.cargo/bin:" + os.environ["PATH"]
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    os.environ["PATH"] = f"{os.environ['CUDA_HOME']}/bin:" + os.environ["PATH"]
    os.environ["LD_LIBRARY_PATH"] = (
        f"{os.environ['CUDA_HOME']}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    )
    if not pathlib.Path(os.environ["CUDA_HOME"], "include/cuda.h").exists():
        raise RuntimeError(f"VOID: no cuda.h under CUDA_HOME={os.environ['CUDA_HOME']}")

    # cuda-bindings runs bindgen, which needs a real clang installation — not just
    # a libclang.so. Kaggle's image ships only the Python `clang` package's bundled
    # library, which has no resource directory, so bindgen resolves `#include
    # <stddef.h>` against nothing and dies on /usr/include/stdlib.h. Internet is
    # enabled, so install clang and take both paths from it.
    if not shutil.which("clang") or not shutil.which("time"):
        # `time` is not in the Kaggle image either, and the arms report max RSS.
        sh(
            "apt-get -qq update && apt-get -qq install -y clang libclang-dev time",
            check=False,
        )
    if not shutil.which("clang"):
        raise RuntimeError("VOID: no clang; cuda-bindings' bindgen cannot run")
    res = out("clang -print-resource-dir")
    if not res or not pathlib.Path(res, "include/stddef.h").exists():
        raise RuntimeError(f"VOID: clang resource dir has no stddef.h ({res!r})")
    os.environ["BINDGEN_EXTRA_CLANG_ARGS"] = f"-resource-dir {res} " + os.environ.get(
        "BINDGEN_EXTRA_CLANG_ARGS", ""
    )
    # Prefer the libclang that belongs to that clang, not the Python package's copy.
    lib = out(
        "find /usr/lib /usr/local/lib -name 'libclang.so*' -not -path '*dist-packages*' "
        "2>/dev/null | head -1"
    )
    if lib:
        os.environ["LIBCLANG_PATH"] = str(pathlib.Path(lib).parent)
    print(
        f"clang {out('clang --version | head -1')}\n"
        f"LIBCLANG_PATH={os.environ.get('LIBCLANG_PATH', '(default)')}\n"
        f"BINDGEN_EXTRA_CLANG_ARGS={os.environ['BINDGEN_EXTRA_CLANG_ARGS']}",
        flush=True,
    )

    # Linking wants `-lcuda`, i.e. a file literally named libcuda.so. A Kaggle image
    # has the driver as libcuda.so.1 and no toolkit stub, so the link fails at the
    # very end of the build. Point the linker at whatever exists, creating the
    # unversioned name in scratch if only the versioned one is there.
    cand = [
        f"{os.environ['CUDA_HOME']}/lib64/stubs/libcuda.so",
        "/usr/lib/x86_64-linux-gnu/libcuda.so",
    ]
    libdir = next(
        (str(pathlib.Path(c).parent) for c in cand if pathlib.Path(c).exists()), None
    )
    if not libdir:
        # The container runtime can mount the driver anywhere (`find` under /usr/lib
        # came up empty on a machine whose GPUs work), so ask the loader.
        real = out("ldconfig -p | awk '/libcuda\\.so/ {print $NF; exit}'") or out(
            "find / -xdev -name 'libcuda.so*' 2>/dev/null | head -1"
        )
        print(
            f"libcuda candidates: ldconfig={out('ldconfig -p | grep -c libcuda')} "
            f"picked={real!r}",
            flush=True,
        )
        if not real:
            raise RuntimeError("VOID: no libcuda.so* on this image; cannot link")
        shim = pathlib.Path("/kaggle/tmp/link")
        shim.mkdir(parents=True, exist_ok=True)
        (shim / "libcuda.so").unlink(missing_ok=True)
        (shim / "libcuda.so").symlink_to(real)
        libdir = str(shim)
        print(f"link: {real} -> {shim}/libcuda.so", flush=True)
    os.environ["LIBRARY_PATH"] = libdir + ":" + os.environ.get("LIBRARY_PATH", "")
    print(f"LIBRARY_PATH={os.environ['LIBRARY_PATH']}", flush=True)

    # Every cargo command runs *inside the checkout*: rust-toolchain.toml pins the
    # nightly there, and Kaggle's pre-existing ~/.rustup has no default toolchain, so
    # a cargo invocation from anywhere else fails with "could not choose a version of
    # cargo to run" — which is exactly how the first attempt died.
    sh(f"cd {SRC} && rustup show active-toolchain")
    if not shutil.which("cargo-oxide"):
        sh(
            f"cd {SRC} && cargo install --quiet --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide"
        )
    # One binary. Both stages here exercise the shipped default build; the seeding path is
    # chosen at runtime on the worker count, so a feature-gated second build would be
    # byte-identical anyway.
    variants = [("baseline", SRC, ())]
    binaries = {}
    build_lines = [f"repo {REPO}", f"sha {head}", "arch sm_75"]
    for name, src, features in variants:
        SRCS[name] = src
        target = src / "target" / f"kaggle-{name}"
        started = time.time()
        flags = " ".join(features)
        sh(
            f"cd {src} && cargo oxide build -- --release {flags} --target-dir {target} 2>&1 | tail -40"
        )
        hspZ = target / "release/hspZ"
        if not hspZ.exists():
            raise RuntimeError(f"{name} build produced no binary")
        binaries[name] = hspZ
        build_lines += [
            f"{name}.features {flags or '(none)'}",
            f"{name}.seconds {time.time() - started:.0f}",
            f"{name}.sha256 {out('sha256sum ' + str(hspZ)).split()[0]}",
            f"{name}.src {src}",
            f"{name}.version {out(str(hspZ) + ' --version')}",
        ]
    build_lines += [
        f"build_seconds {time.time() - t:.0f}",
        f"libclang {os.environ.get('LIBCLANG_PATH', '(default)')}",
    ]
    (WORK / "build.txt").write_text("\n".join(build_lines) + "\n")
    print(f"build: {head} for sm_75 in {time.time() - t:.0f}s", flush=True)
    return binaries, head


def inputs():
    """The attached dataset. Kaggle decompresses `.gz` on upload, so match both."""
    root = pathlib.Path("/kaggle/input")
    print(
        "input mounts:",
        [str(p) for p in root.glob("*")] if root.exists() else "(no /kaggle/input)",
        flush=True,
    )
    # Take the dataset directory if it is where we expect, else any mount that has
    # the two files — a renamed slug should not cost a six-minute build.
    dirs = (
        [DATASET]
        if DATASET.exists()
        else sorted(p for p in root.glob("*") if p.is_dir())
    )
    ref = qry = None
    for d in dirs:
        ref = next(iter(sorted(d.rglob("hg38*.fa*"))), None) or ref
        qry = next(iter(sorted(d.rglob("mm39*.fa*"))), None) or qry
        if ref and qry:
            break
    if not (ref and qry):
        raise RuntimeError(
            f"VOID: no hg38/mm39 input under /kaggle/input "
            f"(looked in {[str(d) for d in dirs]})"
        )
    print(
        f"input: {ref} ({ref.stat().st_size / 1e9:.2f} GB), {qry} ({qry.stat().st_size / 1e9:.2f} GB)",
        flush=True,
    )
    man = next(iter(ref.parent.glob("manifest.txt")), None)
    if man:
        shutil.copy(man, WORK / "input-manifest.txt")
        print(man.read_text(), flush=True)
    return ref, qry


def small_workloads():
    """Locate the species-pair files in the attached dataset.

    A workload whose files are not there is skipped with a printed note rather than
    failing the job: the dataset version and the kernel version move independently,
    and a missing pair must not cost a 90-minute run.
    """
    root = pathlib.Path("/kaggle/input")
    index = {}
    if root.exists():
        # Kaggle decompresses an uploaded `x.fa.gz` either to `x.fa` or to a
        # *directory* `x.fa/` containing `x.fa`, seemingly at random across files in
        # one upload. Both shapes appear in this dataset, so match files only and let
        # the nested copy win by name.
        for f in root.rglob("*.fa"):
            if f.is_file():
                index[f.name] = f
    found, missing = [], []
    for wid, ref, qry in WORKLOADS:
        if ref in index and qry in index:
            found.append((wid, index[ref], index[qry]))
        else:
            missing.append(wid)
    if missing:
        print(
            f"small workloads missing from the dataset, skipped: {', '.join(missing)}",
            flush=True,
        )
    print(f"small workloads available: {[w[0] for w in found]}", flush=True)
    return found


def row(**kw):
    new = not BENCH.exists()
    with BENCH.open("a", newline="") as f:
        w = csv.DictWriter(f, BENCH_COLS, delimiter="\t", extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: kw.get(k, "") for k in BENCH_COLS})


def verdict(comparison, v, detail=""):
    new = not CORR.exists()
    with CORR.open("a", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if new:
            w.writerow(["comparison", "verdict", "detail"])
        w.writerow([comparison, v, detail])
    print(f"  gate {comparison:28} {v} {detail}", flush=True)


def field(text, pattern, idx=None):
    """First numeric field of the last line matching `pattern` (phase rows have a
    variable number of name words, so fixed columns are a trap)."""
    hit = [l for l in text.splitlines() if pattern in l]
    if not hit:
        return None
    parts = hit[-1].split()
    if idx is not None:
        return parts[idx] if idx < len(parts) else None
    for p in parts:
        try:
            return float(p.replace("%", "")) if "." in p else float(p)
        except ValueError:
            continue
    return None


def phase(text, name, gpu=False):
    """Host or CUDA-event milliseconds from one exact timing-table row."""
    hit = [l for l in text.splitlines() if l.startswith(f"  {name}")]
    if not hit:
        return 0
    if gpu:
        try:
            return 0 if hit[-1].split()[-2] == "-" else float(hit[-1].split()[-2])
        except (ValueError, IndexError):
            return 0
    return field(hit[-1], name) or 0


def seed_mechanism(text):
    stages = ("seed k-mers", "seed scan blocks", "seed add offsets", "seed scatter")
    device = all(any(l.startswith(f"  {s}") for l in text.splitlines()) for s in stages)
    async_upload = any(
        l.startswith("  (H->D seeds (standalone))") for l in text.splitlines()
    )
    return device, async_upload, int(field(text, "seed uploads:") or 0)


def arm(
    hspZ, ref, qry, label, gpus, extra=(), env=None, block=None, variant="baseline"
):
    d = SCRATCH / label
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    tar = SCRATCH / f"{label}.tar.gz"
    tar.unlink(missing_ok=True)
    cmd = [
        str(hspZ),
        "run",
        "--reference",
        str(ref),
        "--query",
        str(qry),
        "--output",
        str(d),
        "--seq-block-size",
        block or BLOCK,
        "--threads",
        threads_for(gpus, variant),
        "--gpus",
        str(gpus),
        "--time",
        *extra,
    ]
    e = dict(os.environ, **(env or {}))
    t = time.time()
    t0 = time.strftime(SMI_TIME)
    tfile = SCRATCH / f"{label}.time"
    prefix = (
        ["/usr/bin/time", "-v", "-o", str(tfile)]
        if pathlib.Path("/usr/bin/time").exists()
        else []
    )
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    p = subprocess.run([*prefix, *cmd], text=True, capture_output=True, env=e)
    wall_s = time.time() - t
    if not prefix:
        # No GNU time: ru_maxrss is the high-water mark over all children so far, so
        # it is the max of this arm and every earlier one — honest for the largest
        # arm, an over-report for a later smaller one. Recorded either way.
        after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        tfile.write_text(
            f"\tMaximum resident set size (kbytes): {max(after, before)}\n"
        )
    (LOGS / f"{label}.err").write_text(p.stderr)
    (LOGS / f"{label}.out").write_text(p.stdout)
    timef = (
        (SCRATCH / f"{label}.time").read_text()
        if (SCRATCH / f"{label}.time").exists()
        else ""
    )

    lc = [l for l in p.stderr.splitlines() if "lifecycle:" in l]
    R = Q = units = builds = engines = ref_uploads = swaps = 0
    if lc:
        f = lc[-1].split()
        R, units, builds, engines, ref_uploads, swaps = (
            int(f[1]),
            int(f[4]),
            int(f[7]),
            int(f[9]),
            int(f[11]),
            int(f[14]),
        )
        Q = units // R if R else 0
    # Stage 8: reference-side work must equal R, never R x Q, on both arms.
    mech = (
        "ok"
        if (builds == engines == ref_uploads == R and swaps == units)
        else (
            f"VOID R={R} builds={builds} engines={engines} uploads={ref_uploads} swaps={swaps} units={units}"
        )
    )

    dig_dir = d
    if tar.exists():
        dig_dir = d / "x"
        dig_dir.mkdir(exist_ok=True)
        sh(f"tar xzf {tar} -C {dig_dir}")
    segs = sorted(dig_dir.glob("*.segments"))
    canon = SCRATCH / f"{label}.canon"
    if segs:
        sh(f"cat {dig_dir}/*.segments | LC_ALL=C sort > {canon}")
        n = int(out(f"wc -l < {canon}"))
        dn = int(out(f"uniq {canon} | wc -l"))
        dg = out(f"sha256sum {canon}")[:16]
    else:
        n = dn = 0
        dg = "-"

    device_seeds, async_seed_upload, seed_uploads = seed_mechanism(p.stderr)
    device_ms = sum(
        phase(p.stderr, s, gpu=True)
        for s in ("seed k-mers", "seed scan blocks", "seed add offsets", "seed scatter")
    )
    row(
        stage=STAGE,
        arm=label,
        variant=variant,
        gpus=gpus,
        wall_ms=field(p.stderr, "total wall") or round(wall_s * 1000, 1),
        core_ms=sum(
            float(l.split()[-2])
            for l in p.stderr.splitlines()
            if l.startswith(("  find_hsps", "  find_hits", "  find_num_hits"))
        ),
        gpu_ms=field(p.stderr, "gpu busy") or 0,
        rss_kb=field(timef, "Maximum resident") or 0,
        hsps=n,
        distinct=dn,
        digest=dg,
        ref_bins=R,
        query_bins=Q,
        work_units=units,
        max_hits=field(p.stderr, "max_hits:") or 0,
        prefetch=(lambda x: x.split()[-1] if x else "?")(
            next((l for l in p.stderr.splitlines() if "host budget:" in l), "")
        ),
        threads=threads_for(gpus, variant),
        host_est_mib=field(p.stderr, "host budget: estimated peak") or 0,
        device_seeds=str(device_seeds).lower(),
        async_seed_upload=str(async_seed_upload).lower(),
        seed_uploads=seed_uploads,
        seed_cpu_ms=phase(p.stderr, "(seed generation (standalone))"),
        seed_exposed_ms=phase(p.stderr, "seed generation (exposed)"),
        seed_h2d_ms=phase(p.stderr, "(H->D seeds (standalone))"),
        seed_h2d_exposed_ms=phase(p.stderr, "H->D seeds"),
        seed_device_ms=device_ms,
        seed_count_ms=phase(p.stderr, "seed count round trip"),
        t0=t0,
        t1=time.strftime(SMI_TIME),
    )
    print(
        f"  {label:14} gpus={gpus} rc={p.returncode} wall={wall_s:.1f}s hsps={n} mech={mech}",
        flush=True,
    )
    if p.returncode != 0:
        verdict(
            f"exit:{label}", "FAIL", f"rc={p.returncode}: {p.stderr.strip()[-300:]}"
        )
    if mech != "ok":
        verdict(f"lifecycle:{label}", "FAIL", mech)
    if label.startswith("fast-"):
        canonical = (n, dn, dg) == (CANON_HSPS, CANON_DISTINCT, CANON_DIGEST)
        verdict(
            f"canonical:{label}",
            "EXACT" if canonical else "FAIL",
            f"{n} HSPs, {dn} distinct, {dg}",
        )
    elif WORKLOAD_CANON:
        # A recorded per-workload digest is a cross-session gate; without one the
        # session-internal 1-vs-2-GPU comparison is still the load-bearing check.
        want = WORKLOAD_CANON.get(label.rsplit("-", 1)[0])
        if want:
            ok = (n, dn, dg) == want
            verdict(
                f"canonical:{label}",
                "EXACT" if ok else "FAIL",
                f"{n} HSPs, {dn} distinct, {dg}"
                + ("" if ok else f" (expected {want[0]}, {want[1]}, {want[2]})"),
            )
    return d if not tar.exists() else dig_dir


def compare(label, a, b, tarballs=False):
    """Byte-for-byte first; only if that fails does the class of difference matter."""

    def classification():
        gate = out(
            f"cd {SRC} && tests/multiplicity_gate.sh {label} "
            f"{SCRATCH}/{label.split('-vs-')[0]}.canon "
            f"{SCRATCH}/{label.split('-vs-')[1]}.canon"
        )
        return gate.splitlines()[0] if gate else "difference could not be classified"

    fa = sorted(p.name for p in a.glob("*.segments"))
    fb = sorted(p.name for p in b.glob("*.segments"))
    if not fa or not fb:
        # Two empty directories compare equal, which is how kernel v22 reported
        # "EXACT (0 files)" for six pairs whose arms had all died with a driver
        # error. An empty output is a failure, never a match.
        return verdict(
            label, "FAIL", f"no output to compare ({len(fa)} vs {len(fb)} files)"
        )
    if fa != fb:
        return verdict(
            label,
            "FAIL",
            f"file sets differ ({len(fa)} vs {len(fb)}); {classification()}",
        )
    diff = [n for n in fa if (a / n).read_bytes() != (b / n).read_bytes()]
    detail = f"{len(fa)} files"
    if tarballs:
        ta, tb = SCRATCH / f"{a.parent.name if a.name == 'x' else a.name}.tar.gz", None
        ta = SCRATCH / f"{label.split('-vs-')[0]}.tar.gz"
        tb = SCRATCH / f"{label.split('-vs-')[1]}.tar.gz"
        if ta.exists() and tb.exists():
            detail += (
                ", tarballs byte-identical"
                if ta.read_bytes() == tb.read_bytes()
                else ", TARBALLS DIFFER"
            )
            if ta.read_bytes() != tb.read_bytes():
                return verdict(label, "FAIL", detail)
    if diff:
        return verdict(label, "FAIL", f"{len(diff)} file(s) differ; {classification()}")
    verdict(label, "EXACT", detail)


def sampler():
    f = (WORK / "gpu.csv").open("w")
    p = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv",
            "-l",
            "1",
        ],
        stdout=f,
    )
    return p, f


def gpu_samples():
    samples = []
    gpu = WORK / "gpu.csv"
    if not gpu.exists():
        return samples
    for line in gpu.read_text().splitlines()[1:]:
        f = [x.strip() for x in line.split(",")]
        if len(f) >= 4 and f[1].isdigit():
            try:
                samples.append(
                    (
                        f[0].split(".")[0],
                        int(f[1]),
                        float(f[2].split()[0]),
                        float(f[3].split()[0]),
                    )
                )
            except ValueError:
                pass
    return samples


def report():
    """Human-readable summary next to the machine-readable TSVs."""
    bench = list(csv.DictReader(BENCH.open(), delimiter="\t")) if BENCH.exists() else []
    corr = list(csv.DictReader(CORR.open(), delimiter="\t")) if CORR.exists() else []
    L = [
        f"# hspZ on Kaggle 2x Tesla T4 -- stage `{STAGE}`\n",
        "## Environment\n",
        "```text",
        (WORK / "environment.txt").read_text().strip(),
        "```\n",
        "## Build\n",
        "```text",
        (WORK / "build.txt").read_text().strip(),
        "```\n",
        "## Correctness\n",
        "| comparison | verdict | detail |",
        "| --- | --- | --- |",
    ]
    for r in corr:
        L.append(f"| {r['comparison']} | **{r['verdict']}** | {r['detail']} |")
    if not corr:
        L.append("| _(none)_ | | |")
    bad = [r for r in corr if r["verdict"] not in ("EXACT", "PASS")]
    L += [
        "",
        "**All comparisons EXACT.**"
        if corr and not bad
        else f"**{len(bad)} comparison(s) not EXACT.**",
        "",
        "Absolute HSP counts are device-class constants, not universal ones: `MAX_HITS` is",
        "derived from device memory (a T4's 15 GB gives ~58 M, an L4's 23 GB ~94 M) and it",
        "sets the per-chunk dedup boundary. Every gate here compares against digests frozen",
        "on this same 2x T4 shape.",
        "",
        "## Arms\n",
        "| arm | gpus | wall ms | gpu ms | HSPs | distinct | digest | R | Q | units |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for r in bench:
        L.append(
            "| {arm} | {gpus} | {wall_ms} | {gpu_ms} | {hsps} | {distinct} | `{digest}` "
            "| {ref_bins} | {query_bins} | {work_units} |".format(**r)
        )
    (WORK / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def main():
    """Two stages only.

    sweep  the six species pairs, one 1-GPU arm each, every one gated against the frozen
           per-workload digests in WORKLOAD_CANON. ~12 min of compute. Each reference here is
           a single record, so there is exactly one work unit however the block is set -- a
           2-GPU arm would idle the second device and its "1 vs 2 GPUs EXACT" would be a
           tautology. What these arms gate is the output itself.
    fast   sweep, plus a 1,2,2,1 set on the 49-work-unit pair in `-D -Z` mode, so machine
           drift cancels within one session and multi-GPU identity is checked on a workload
           that really has work to divide. ~1.5 h.
    """
    if STAGE not in ("sweep", "fast"):
        raise RuntimeError(
            f"VOID: unknown stage {STAGE!r}; this kernel runs sweep or fast"
        )
    for d in (WORK, LOGS, SCRATCH):
        d.mkdir(parents=True, exist_ok=True)
    gate_environment()
    binaries, head = build()
    hspZ = binaries["baseline"]
    ref, qry = (None, None) if STAGE == "sweep" else inputs()

    smp, fh = sampler()
    try:
        for wid, wref, wqry in small_workloads():
            arm(hspZ, wref, wqry, f"{wid}-1gpu", 1, block=SMALL_BLOCK)
        if STAGE == "fast":
            tar = lambda n: str(SCRATCH / f"{n}.tar.gz")
            a1 = arm(hspZ, ref, qry, "fast-1gpu-a", 1, ["-D", "-Z", tar("fast-1gpu-a")])
            a2 = arm(hspZ, ref, qry, "fast-2gpu-a", 2, ["-D", "-Z", tar("fast-2gpu-a")])
            arm(hspZ, ref, qry, "fast-2gpu-b", 2, ["-D", "-Z", tar("fast-2gpu-b")])
            arm(hspZ, ref, qry, "fast-1gpu-b", 1, ["-D", "-Z", tar("fast-1gpu-b")])
            compare("fast-1gpu-a-vs-fast-2gpu-a", a1, a2, tarballs=True)
    finally:
        smp.terminate()
        fh.close()
        gpu_samples()

    (WORK / "mechanisms.json").write_text(
        json.dumps(
            {
                "sha": head,
                "stage": STAGE,
                "block": BLOCK,
                "small_block": SMALL_BLOCK,
                "threads": THREADS,
                "cpus": os.cpu_count(),
                "gpus": out("nvidia-smi -L").splitlines(),
                "binaries": {
                    n: {"sha256": out(f"sha256sum {p}").split()[0]}
                    for n, p in binaries.items()
                },
            },
            indent=2,
        )
        + "\n"
    )
    report()
    # Keep the evidence, drop the payload: the outputs are gigabytes and already digested.
    shutil.rmtree(SCRATCH, ignore_errors=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
