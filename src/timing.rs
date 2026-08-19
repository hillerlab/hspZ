// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung
// Distributed under the terms of the GNU General Public License, Version 3.0.

// Author : Alejandro Gonzales-Irribarren
// Github : alejandrogzi
// Email  : alejandrxgzi@gmail.com

//! Wall-time accounting.
//!
//! Every phase is timed on the host clock and the numbers are required to add
//! up to the observed wall time. Kernel phases additionally carry their
//! CUDA-event duration, which is what separates real GPU work from launch and
//! synchronisation overhead.
//!
//! Timing and synchronisation are separated: `Stage::finish`
//! only records an end event and returns, and durations are resolved at the
//! next genuine sync (`resolve_pending`). Phases that ran concurrently with
//! another are marked `overlapped` — reported but excluded from the accounted
//! total, so the table keeps balancing against wall clock.

use std::time::Duration;

#[derive(Debug, Clone)]
struct Row {
    name: String,
    host_ms: f64,
    gpu_ms: f64,
    /// How many times the phase ran — chunked stages are entered once per chunk.
    calls: u32,
    has_gpu: bool,
    /// Ran concurrently with another phase, so it is reported but excluded from
    /// the accounted total — otherwise overlapped work would be counted twice
    /// and the table would stop adding up to wall time.
    overlapped: bool,
}

/// An ordered, accumulating set of named phases.
///
/// # Example
/// ```
/// let mut p = Phases::new();
/// p.add("parse", std::time::Duration::from_millis(12));
/// assert!(p.total_ms() >= 12.0);
/// assert!(p.report(100.0).contains("parse"));
/// ```
#[derive(Debug, Default, Clone)]
pub struct Phases {
    rows: Vec<Row>,
}

impl Phases {
    /// Summed CUDA-event time across every stage that has events.
    ///
    /// The executor reports this per worker. With 49 work units assigned by
    /// reference bin, 7 bins cannot split evenly across 2 workers (4/3) or 4 (2/2/2/1),
    /// and the sum over workers cannot show that — only the spread between them can.
    pub fn gpu_ms(&self) -> f64 {
        self.rows
            .iter()
            .filter(|r| r.has_gpu)
            .map(|r| r.gpu_ms)
            .sum()
    }

    pub fn new() -> Self {
        Self::default()
    }

    /// Records a host-only phase — wall time with no CUDA-event duration.
    pub fn add(&mut self, name: &str, host: Duration) {
        self.row(name, false).host_ms += host.as_secs_f64() * 1000.0;
    }

    /// Records a phase that ran on the GPU: `host` is the wall time the call
    /// occupied, `gpu_ms` what the CUDA events measured inside it.
    pub fn add_gpu(&mut self, name: &str, host: Duration, gpu_ms: f32) {
        let row = self.row(name, true);
        row.host_ms += host.as_secs_f64() * 1000.0;
        row.gpu_ms += gpu_ms as f64;
    }

    /// Records a phase whose duration is already known in milliseconds.
    pub fn add_ms(&mut self, name: &str, ms: f64) {
        self.row(name, false).host_ms += ms;
    }

    /// Records a phase that ran concurrently with other timed work. Shown in
    /// the table but excluded from `accounted`, so wall time still balances.
    pub fn add_overlapped(&mut self, name: &str, host: Duration) {
        let row = self.row(name, false);
        row.host_ms += host.as_secs_f64() * 1000.0;
        row.overlapped = true;
    }

    /// As [`add_overlapped`](Self::add_overlapped), for a duration already
    /// measured in milliseconds (a CUDA-event delta on another stream).
    pub fn add_overlapped_ms(&mut self, name: &str, ms: f64) {
        let row = self.row(name, false);
        row.host_ms += ms;
        row.overlapped = true;
    }

    fn row(&mut self, name: &str, has_gpu: bool) -> &mut Row {
        if let Some(i) = self.rows.iter().position(|r| r.name == name) {
            self.rows[i].calls += 1;
            self.rows[i].has_gpu |= has_gpu;
            return &mut self.rows[i];
        }
        self.rows.push(Row {
            name: name.to_string(),
            host_ms: 0.0,
            gpu_ms: 0.0,
            calls: 1,
            has_gpu,
            overlapped: false,
        });
        self.rows.last_mut().unwrap()
    }

    /// Appends another set of phases, preserving its order.
    pub fn merge(&mut self, other: &Phases) {
        for r in &other.rows {
            let dst = self.row(&r.name, r.has_gpu);
            dst.host_ms += r.host_ms;
            dst.gpu_ms += r.gpu_ms;
            dst.calls = r.calls;
            dst.overlapped = r.overlapped;
        }
    }

    /// How many times a named phase ran — the batch count, for any stage that
    /// runs once per batch.
    pub fn calls(&self, name: &str) -> u32 {
        self.rows
            .iter()
            .find(|r| r.name == name)
            .map_or(0, |r| r.calls)
    }

    pub fn total_ms(&self) -> f64 {
        self.rows
            .iter()
            .filter(|r| !r.overlapped)
            .map(|r| r.host_ms)
            .sum()
    }

    /// The phase table as a JSON object body, for `benchmark --json`
    /// Stage names are our own literals — no quotes or
    /// backslashes — so they need no escaping beyond what `json_key` does.
    pub fn json_stages(&self) -> String {
        let mut out = String::from("{");
        for (i, r) in self.rows.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!(
                "\"{}\":{{\"ms\":{:.4},\"gpu_ms\":{:.4},\"calls\":{},\"overlapped\":{}}}",
                json_key(&r.name),
                r.host_ms,
                r.gpu_ms,
                r.calls,
                r.overlapped
            ));
        }
        out.push('}');
        out
    }

    /// A table of absolute milliseconds and percentage of `wall_ms`, closing
    /// with whatever the phases failed to account for.
    pub fn report(&self, wall_ms: f64) -> String {
        let pct = |ms: f64| {
            if wall_ms > 0.0 {
                ms / wall_ms * 100.0
            } else {
                0.0
            }
        };
        let mut out = String::new();
        out.push_str(&format!(
            "  {:<26} {:>9} {:>7} {:>9} {:>7}\n",
            "phase", "ms", "%", "gpu ms", "calls"
        ));
        for r in &self.rows {
            let gpu = if r.has_gpu {
                format!("{:.2}", r.gpu_ms)
            } else {
                "-".into()
            };
            // Overlapped rows are informational; their percentage is of wall
            // time but they do not add into `accounted`.
            out.push_str(&format!(
                "  {:<26} {:>9.2} {:>6.1}% {:>9} {:>7}\n",
                if r.overlapped {
                    format!("({})", r.name)
                } else {
                    r.name.clone()
                },
                r.host_ms,
                pct(r.host_ms),
                gpu,
                r.calls
            ));
        }
        let accounted = self.total_ms();
        out.push_str(&format!(
            "  {:-<26} {:->9} {:->7} {:->9} {:->7}\n",
            "", "", "", "", ""
        ));
        out.push_str(&format!(
            "  {:<26} {:>9.2} {:>6.1}%\n",
            "accounted",
            accounted,
            pct(accounted)
        ));
        out.push_str(&format!(
            "  {:<26} {:>9.2} {:>6.1}%\n",
            "unaccounted",
            wall_ms - accounted,
            pct(wall_ms - accounted)
        ));
        out.push_str(&format!(
            "  {:<26} {:>9.2} {:>6.1}%\n",
            "total wall", wall_ms, 100.0
        ));
        // The metric that decides whether removing host barriers
        // helped. Summing the CUDA-event durations is exact per stage; stages that
        // genuinely overlap would double-count, but this pipeline is one stream.
        let gpu_ms: f64 = self.gpu_ms();
        out.push_str(&format!(
            "  {:<26} {:>9.2} {:>6.1}%\n",
            "gpu busy (events)",
            gpu_ms,
            pct(gpu_ms)
        ));
        out
    }
}

/// Escapes the characters a JSON string cannot carry raw. Phase names never
/// contain them today; this keeps that from becoming a silent invariant.
pub fn json_key(s: &str) -> String {
    s.chars()
        .flat_map(|c| match c {
            '"' => vec!['\\', '"'],
            '\\' => vec!['\\', '\\'],
            c if (c as u32) < 0x20 => vec![' '],
            c => vec![c],
        })
        .collect()
}

/// Milliseconds between process start and now, from `/proc/self/stat`.
///
/// Everything before `main` — dynamic linking against the cuda-oxide tree,
/// libc startup — is invisible to an in-process `Instant`, and it is not small.
///
/// Resolution is one USER_HZ tick, i.e. 10 ms — so this row of the table is
/// quantised and should not be read to the millisecond.
///
/// USER_HZ is hard-coded to 100, which is every mainstream Linux
/// build. If this ever needs to be right on an exotic kernel, read
/// `sysconf(_SC_CLK_TCK)` through libc.
pub fn since_process_start_ms() -> Option<f64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // Field 2 is the comm, parenthesised and free to contain spaces, so parse
    // after its final ')'. Field 22 (starttime) is then index 19.
    let after_comm = &stat[stat.rfind(')')? + 2..];
    let starttime: f64 = after_comm.split(' ').nth(19)?.parse().ok()?;
    let uptime: f64 = std::fs::read_to_string("/proc/uptime")
        .ok()?
        .split(' ')
        .next()?
        .parse()
        .ok()?;
    Some((uptime - starttime / 100.0) * 1000.0)
}

/// Peak resident set size in KiB, from `VmHWM` in `/proc/self/status`.
/// Zero on non-Linux, which is fine — the report column is Linux-only anyway.
pub fn peak_rss_kib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("VmHWM:"))
                .and_then(|v| v.split_whitespace().next()?.parse().ok())
        })
        .unwrap_or(0)
}

/// Available host memory in bytes, from the tightest cgroup v2 limit that governs
/// this process, falling back to `/proc/meminfo` `MemAvailable`.
///
/// cgroup v2 exposes `memory.max` (hard OOM-kill limit) and `memory.high` (soft
/// throttle); "available" is the tighter of the two minus `memory.current`, with
/// "max" meaning unlimited. `None` means no readable limit was found and the
/// caller should skip the host preflight rather than guess.
///
/// The limit must be looked up on the process's **own** cgroup and its ancestors,
/// not on the mount root: under Slurm the job lives in a nested cgroup, the root's
/// `memory.max` reads "max", and reading only the root silently turns the preflight
/// into a no-op on exactly the machines it exists to protect.
pub fn available_host_bytes() -> Option<u64> {
    // Test/CI override: deterministic, no platform dependence.
    if let Ok(v) = std::env::var("HSPZ_HOST_MEMORY_BYTES")
        && let Ok(bytes) = v.parse::<u64>()
    {
        return Some(bytes);
    }
    let proc = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
    // v2: one `0::<path>` line, unified hierarchy.
    let v2 = proc
        .lines()
        .find_map(|l| l.strip_prefix("0::"))
        .and_then(|rel| {
            cgroup_available(
                std::path::Path::new("/sys/fs/cgroup"),
                rel.trim(),
                &["memory.max", "memory.high"],
                "memory.current",
            )
        });
    // v1: a per-controller line, `N:memory:<path>`, under its own mount. node50 is
    // v1, and its Slurm limit lives at
    // /sys/fs/cgroup/memory/slurm/uid_*/job_*/step_*/task_*/memory.limit_in_bytes.
    let v1 = || {
        let rel = proc.lines().find_map(|l| {
            let mut f = l.splitn(3, ':');
            let (_, ctrl, path) = (f.next()?, f.next()?, f.next()?);
            ctrl.split(',')
                .any(|c| c == "memory")
                .then(|| path.to_string())
        })?;
        cgroup_available(
            std::path::Path::new("/sys/fs/cgroup/memory"),
            &rel,
            &["memory.limit_in_bytes"],
            "memory.usage_in_bytes",
        )
    };
    v2.or_else(v1).or_else(mem_available)
}

/// The tightest `limit - current` over a cgroup and its ancestors.
///
/// Split out from [`available_host_bytes`] so the parsing and the walk can be
/// tested against a synthetic tree rather than a live node, and
/// parameterised by file names so one walker serves cgroup v1 and v2.
fn cgroup_available(
    mount: &std::path::Path,
    rel: &str,
    limit_files: &[&str],
    current_file: &str,
) -> Option<u64> {
    // v1 spells "unlimited" as a huge number rather than "max"; anything within a
    // few pages of i64::MAX is that sentinel, not a real limit.
    const UNLIMITED: u64 = i64::MAX as u64 - 4096;
    let read = |dir: &std::path::Path, file: &str| -> Option<u64> {
        let s = std::fs::read_to_string(dir.join(file)).ok()?;
        match s.trim() {
            "max" => None,
            v => v.parse::<u64>().ok().filter(|&n| n < UNLIMITED),
        }
    };
    let mut dir = mount.join(rel.trim_start_matches('/'));
    let mut best: Option<u64> = None;
    loop {
        if let Some(current) = read(&dir, current_file)
            && let Some(limit) = limit_files.iter().filter_map(|f| read(&dir, f)).min()
        {
            let free = limit.saturating_sub(current);
            best = Some(best.map_or(free, |b: u64| b.min(free)));
        }
        if dir == mount {
            break;
        }
        match dir.parent() {
            Some(p) if p.starts_with(mount) => dir = p.to_path_buf(),
            _ => break,
        }
    }
    best
}

/// System-wide `MemAvailable`, the last resort when no cgroup limit applies.
fn mem_available() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The limit that governs a Slurm job is on a *nested* cgroup,
    /// so the walk must take the tightest `limit - current` over the whole chain.
    /// A root that reads "max" must not mask a job cgroup that reads 8 GiB.
    #[test]
    fn cgroup_walk_takes_the_tightest_limit_above_the_process() {
        use super::cgroup_available;
        let root = std::env::temp_dir().join(format!("hspz-cg-{}", std::process::id()));
        let job = root.join("job_1");
        let step = job.join("step_batch");
        std::fs::create_dir_all(&step).unwrap();
        let w = |dir: &std::path::Path, f: &str, v: &str| {
            std::fs::write(dir.join(f), v).unwrap();
        };
        // Mount root: unlimited. Job: 8 GiB with 1 GiB charged. Step: no limit.
        w(&root, "memory.max", "max");
        w(&root, "memory.current", "2147483648");
        w(&job, "memory.max", "8589934592");
        w(&job, "memory.current", "1073741824");
        w(&step, "memory.max", "max");
        w(&step, "memory.current", "536870912");
        let v2 = |rel: &str| {
            cgroup_available(&root, rel, &["memory.max", "memory.high"], "memory.current")
        };
        assert_eq!(v2("/job_1/step_batch"), Some(7516192768));

        // `memory.high` below `memory.max` governs, and the deepest cgroup wins
        // when it is the tightest.
        w(&step, "memory.high", "1073741824");
        assert_eq!(v2("/job_1/step_batch"), Some(536870912));

        // cgroup v1: different file names, and "unlimited" is a huge number rather
        // than the string "max". node50 is v1, so this is the layout that decides
        // whether the preflight sees the Slurm limit at all.
        w(&root, "memory.limit_in_bytes", "9223372036854771712");
        w(&root, "memory.usage_in_bytes", "2147483648");
        w(&job, "memory.limit_in_bytes", "8589934592");
        w(&job, "memory.usage_in_bytes", "1073741824");
        w(&step, "memory.limit_in_bytes", "9223372036854771712");
        w(&step, "memory.usage_in_bytes", "536870912");
        assert_eq!(
            cgroup_available(
                &root,
                "/job_1/step_batch",
                &["memory.limit_in_bytes"],
                "memory.usage_in_bytes"
            ),
            Some(7516192768)
        );

        // No limit anywhere: fall through so the caller can use MemAvailable.
        std::fs::write(job.join("memory.max"), "max").unwrap();
        std::fs::remove_file(step.join("memory.high")).unwrap();
        assert_eq!(v2("/job_1/step_batch"), None);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn phases_accumulate_by_name_and_keep_order() {
        let mut p = Phases::new();
        p.add("first", Duration::from_millis(10));
        p.add("second", Duration::from_millis(5));
        p.add("first", Duration::from_millis(10));
        assert_eq!(p.total_ms(), 25.0);
        let r = p.report(50.0);
        assert!(
            r.find("first").unwrap() < r.find("second").unwrap(),
            "insertion order kept"
        );
        assert!(r.contains("unaccounted"));
        // 20/50 = 40%
        assert!(r.contains("40.0%"), "{r}");
    }

    #[test]
    fn gpu_phases_track_host_and_device_separately() {
        let mut p = Phases::new();
        p.add_gpu("find_hits", Duration::from_millis(100), 47.0);
        assert_eq!(p.total_ms(), 100.0, "the wall-time column is host time");
        assert!(
            p.report(100.0).contains("47.00"),
            "device time reported alongside"
        );
    }

    #[test]
    fn merge_preserves_both_sets() {
        let (mut a, mut b) = (Phases::new(), Phases::new());
        a.add("host", Duration::from_millis(1));
        b.add_gpu("kernel", Duration::from_millis(2), 1.5);
        a.merge(&b);
        assert_eq!(a.total_ms(), 3.0);
        assert!(a.report(3.0).contains("kernel"));
    }

    #[test]
    fn process_start_is_plausible() {
        // A test binary can be younger than one 10 ms tick, so 0 is legitimate.
        let ms = since_process_start_ms().expect("linux /proc");
        assert!(
            (0.0..600_000.0).contains(&ms),
            "implausible process age: {ms}"
        );
    }
}
