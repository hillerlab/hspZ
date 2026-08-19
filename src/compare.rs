// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung
// Distributed under the terms of the GNU General Public License, Version 3.0.

// Author : Alejandro Gonzales-Irribarren
// Github : alejandrogzi
// Email  : alejandrxgzi@gmail.com

//! The `compare` command: run the C++ CUDA oracle and this implementation on
//! the same pair, then diff every `.segments` file byte for byte.

use crate::Fallible;
use crate::cli::{CompareArgs, RunArgs};
use crate::run::{Stats, run};
use std::path::Path;
use std::time::Instant;

pub(crate) fn compare(args: &CompareArgs) -> Fallible<()> {
    let workdir = match &args.workdir {
        Some(p) => p.clone(),
        None => std::env::temp_dir().join(format!("hspz-compare-{}", std::process::id())),
    };
    let cpp_dir = workdir.join("cpp");
    let rust_dir = workdir.join("rust");
    std::fs::create_dir_all(&cpp_dir)?;
    std::fs::create_dir_all(&rust_dir)?;

    eprintln!("running C++ CUDA reference ...");
    let (cpp_secs, cpp_stats) = run_kegalign(args, &cpp_dir)?;

    eprintln!("running Rust implementation ...");
    let run_args = RunArgs {
        reference: args.reference.clone(),
        query: args.query.clone(),
        output: rust_dir.clone(),
        strand: "both".into(),
        seed: args.tuning.seed.clone(),
        step: 1,
        notransition: false,
        xdrop: args.tuning.xdrop,
        hspthresh: args.tuning.hspthresh,
        noentropy: false,
        ambiguous: String::new(),
        scoring: args.tuning.scoring.clone(),
        target_prefix: String::new(),
        query_prefix: String::new(),
        wga_chunk_size: 250_000,
        lastz_interval_size: 10_000_000,
        seq_block_size: 500_000_000,
        // `compare` matches KegAlign's single-block layout, which has no separate query
        // target — leaving this None keeps the query side on `seq_block_size`.
        query_block_size: None,
        max_hits: 0,
        threads: 0,
        hsp_blocks: 0,
        no_pinned_seeds: false,
        no_persistent_seed_buffers: false,
        no_ref_prefetch: false,
        kegalign_bins: false,
        dump_plan: None,
        gpus: 1,
        no_async_stages: false,
        no_async_seed_copy: false,
        time: false,
        hit_stats: false,
        cpu_only: false,
        dump_raw: None,
        // `compare` is oracle parity only. The oracle writes a directory of
        // unsplit FASTA-derived segments, so -D/-Z are not offered here; their
        // correctness is self-parity, checked in `run`.
        diagonal_partition: false,
        tarball: None,
    };
    let start = Instant::now();
    let rust_stats = run(&run_args, 0.0, Instant::now())?;
    let rust_secs = start.elapsed().as_secs_f64();

    let diff = compare_segments(&cpp_dir, &rust_dir)?;

    println!("\n=== Seed + Filter comparison ===");
    println!("C++ CUDA runtime   {cpp_secs:8.2} s");
    println!("Rust CUDA runtime  {rust_secs:8.2} s");
    println!(
        "speed ratio        {:8.2}x (rust/cpp)",
        rust_secs / cpp_secs.max(1e-9)
    );
    println!(
        "\nC++  seeds={} hits={} hsps={}",
        cpp_stats.seeds, cpp_stats.seed_hits, cpp_stats.hsps
    );
    println!(
        "Rust seeds={} hits={} hsps={} (raw {})",
        rust_stats.seeds, rust_stats.seed_hits, rust_stats.hsps, rust_stats.raw_hsps
    );
    println!("\nHSP parity: {diff}");
    if !diff.starts_with("IDENTICAL") {
        std::process::exit(2);
    }
    Ok(())
}

fn run_kegalign(args: &CompareArgs, dir: &Path) -> Fallible<(f64, Stats)> {
    // The oracle occasionally aborts during context setup when runs are
    // launched back to back under ZLUDA. Retry once — but only on a failed
    // *launch*; a run that completes is never re-rolled.
    match run_kegalign_once(args, dir) {
        Ok(v) => Ok(v),
        Err(first) => {
            eprintln!("oracle failed, retrying once: {first}");
            run_kegalign_once(args, dir)
        }
    }
}

fn run_kegalign_once(args: &CompareArgs, dir: &Path) -> Fallible<(f64, Stats)> {
    let start = Instant::now();
    let out = std::process::Command::new(&args.kegalign)
        .current_dir(dir)
        .arg(&args.reference)
        .arg(&args.query)
        .arg(dir)
        .args(["--nogapped", "--debug"])
        .args(["--seed", &args.tuning.seed])
        .args(["--xdrop", &args.tuning.xdrop.to_string()])
        .args(["--hspthresh", &args.tuning.hspthresh.to_string()])
        .args(match &args.tuning.scoring {
            Some(p) => vec!["--scoring".to_string(), p.display().to_string()],
            None => vec![],
        })
        // Only override what the caller actually set. Passing an empty value would
        // *clear* the inherited variable, and clearing LD_LIBRARY_PATH breaks the
        // oracle on any host that needs it to find its CUDA runtime.
        .envs(
            [
                ("LD_PRELOAD", &args.ld_preload),
                ("LD_LIBRARY_PATH", &args.ld_library_path),
            ]
            .into_iter()
            .filter(|(_, v)| !v.is_empty()),
        )
        .output()
        .map_err(|e| format!("{}: {e}", args.kegalign.display()))?;
    let secs = start.elapsed().as_secs_f64();
    if !out.status.success() {
        return Err(format!(
            "kegalign exited {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }

    let stderr = String::from_utf8_lossy(&out.stderr);
    let field = |k: &str| -> u64 {
        stderr
            .lines()
            .find_map(|l| l.strip_prefix(k))
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0)
    };
    Ok((
        secs,
        Stats {
            seeds: field("#seeds:"),
            seed_hits: field("#seed hits:"),
            raw_hsps: 0,
            hsps: field("#HSPs:"),
        },
    ))
}

/// Byte-exact comparison of every `.segments` file the two runs produced.
fn compare_segments(cpp: &Path, rust: &Path) -> Fallible<String> {
    let list = |dir: &Path| -> Fallible<Vec<String>> {
        let mut v: Vec<String> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".segments"))
            .collect();
        v.sort();
        Ok(v)
    };
    let (a, b) = (list(cpp)?, list(rust)?);
    if a != b {
        return Ok(format!("DIFFERENT file sets: C++ {a:?} vs Rust {b:?}"));
    }

    let mut total = 0usize;
    for name in &a {
        let x = std::fs::read_to_string(cpp.join(name))?;
        let y = std::fs::read_to_string(rust.join(name))?;
        let (xl, yl): (Vec<_>, Vec<_>) = (x.lines().collect(), y.lines().collect());
        if xl.len() != yl.len() {
            return Ok(format!(
                "DIFFERENT {name}: {} vs {} lines",
                xl.len(),
                yl.len()
            ));
        }
        if let Some(i) = (0..xl.len()).find(|&i| xl[i] != yl[i]) {
            return Ok(format!(
                "DIFFERENT {name} line {}:\n  C++  {}\n  Rust {}",
                i + 1,
                xl[i],
                yl[i]
            ));
        }
        total += xl.len();
    }
    Ok(format!("IDENTICAL ({total} HSPs across {} files)", a.len()))
}
