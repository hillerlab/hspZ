// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung
// Distributed under the terms of the GNU General Public License, Version 3.0.

// Author : Alejandro Gonzales-Irribarren
// Github : alejandrogzi
// Email  : alejandrxgzi@gmail.com

//! On-disk reference index (`hspz index`, `hspz run --index DIR`).
//!
//! `hspz index` builds every reference bin's execution inputs once — the
//! [`SeedTable`] arrays exactly as they exist in RAM plus the encoded reference
//! — and publishes them under `DIR` atomically. `hspz run --index DIR` loads
//! those arrays in the existing one-bin-ahead prefetch slot instead of calling
//! `build_ref_bin`. Planning, upload, kernels and emit are unchanged, so output
//! is byte-identical whenever the resolved reference bins equal the index's.
//!
//! Directory layout (v1): text `MANIFEST` in [`plan::PlanManifest`] style,
//! `READY` proving the publish finished, and one `{id:03}` subdirectory per bin
//! with three raw little-endian array files (`index_table`, `pos_table`, `enc`)
//! and no headers.

use crate::Fallible;
use crate::cli::{IndexArgs, RunArgs};
use crate::plan::{self, Bin, PackedBin};
use crate::run::{build_ref_bin, record_meta, resolve_threads, sha256_file};
use crate::seed::{SeedTable, Shape};
use crate::sequence::{self, E_NT};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Version triple a v1 reader accepts, exactly.
pub(crate) const FORMAT_VERSION: u32 = 1;
pub(crate) const PACKING_VERSION: u32 = 1;
pub(crate) const SEED_BUILDER_VERSION: u32 = 1;
/// v1 refuses more bins than this (directory names are `{id:03}`).
const MAX_BINS: usize = 1000;
const READY_TEXT: &str = "hspz-index-ready 1\n";

/// One record's entry in the manifest: identity plus its placement inside its
/// bin's packed buffer, so the loader can rebuild `Chr` without the FASTA.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RecEntry {
    id: u32,
    ordinal: u32,
    len: u64,
    bin_id: u32,
    bin_start: u64,
    name: String,
}

/// One bin's manifest entry: membership plus per-array lengths and checksums.
#[derive(Debug, Clone, PartialEq, Eq)]
struct BinEntry {
    bin: Bin,
    block_len: u64,
    index_len: u64,
    pos_len: u64,
    enc_len: u64,
    index_crc: u64,
    pos_crc: u64,
    enc_crc: u64,
    recs: Vec<RecEntry>,
}

/// Parsed `MANIFEST`. Checksums are FNV-1a of the raw file bytes, same
/// polynomial as [`plan::records_hash`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IndexManifest {
    format_version: u32,
    packing_version: u32,
    seed_builder_version: u32,
    byte_order: String,
    hspz_version: String,
    features: String,
    executable_hash: u64,
    seed: String,
    seed_size: usize,
    kmer_size: usize,
    seed_pos: Vec<usize>,
    step: u32,
    kegalign_bins: bool,
    seq_block_size: u64,
    ref_records_hash: u64,
    ref_file_sha256: String,
    ref_file_bytes: u64,
    threads: usize,
    n_records: usize,
    bins: Vec<BinEntry>,
}

impl IndexManifest {
    /// One-line pre-schedule summary for the `index: using` ledger line.
    pub(crate) fn run_summary(&self) -> (usize, u64, &str, u32) {
        (self.bins.len(), self.seq_block_size, &self.seed, self.step)
    }
}

fn crc_u32s(v: &[u32]) -> u64 {
    let mut h = plan::FNV_OFFSET;
    for &w in v {
        h = plan::fnv1a(&w.to_le_bytes(), h);
    }
    h
}

fn crc_bytes(b: &[u8]) -> u64 {
    plan::fnv1a(b, plan::FNV_OFFSET)
}

fn bin_subdir(dir: &Path, id: u32) -> PathBuf {
    dir.join(format!("{id:03}"))
}

// ---------------------------------------------------------------------------
// Write path (`hspz index`)
// ---------------------------------------------------------------------------

/// Writes one `Vec<u32>` as bare little-endian bytes with an fsync, without
/// materialising a byte copy of the whole array.
fn write_u32s(path: &Path, v: &[u32]) -> Fallible<()> {
    let f = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
    let mut buf = [0u8; 1 << 20];
    // 1 MiB is a multiple of 4, so every chunk fills whole words.
    for chunk in v.chunks(buf.len() / 4) {
        for (i, &word) in chunk.iter().enumerate() {
            buf[4 * i..4 * i + 4].copy_from_slice(&word.to_le_bytes());
        }
        w.write_all(&buf[..chunk.len() * 4])?;
    }
    w.flush()?;
    let f = w.into_inner().map_err(|e| e.into_error())?;
    f.sync_all()?;
    Ok(())
}

fn write_bytes(path: &Path, b: &[u8]) -> Fallible<()> {
    let mut f = std::fs::File::create(path)?;
    f.write_all(b)?;
    f.sync_all()?;
    Ok(())
}

fn manifest_text(m: &IndexManifest) -> String {
    let mut s = String::new();
    s.push_str("hspz-index 1\n");
    s.push_str("magic HSPZIDX\n");
    s.push_str(&format!("format_version {}\n", m.format_version));
    s.push_str(&format!("packing_version {}\n", m.packing_version));
    s.push_str(&format!(
        "seed_builder_version {}\n",
        m.seed_builder_version
    ));
    s.push_str(&format!("byte_order {}\n", m.byte_order));
    s.push_str(&format!("hspz_version {}\n", m.hspz_version));
    s.push_str(&format!("features {}\n", m.features));
    s.push_str(&format!("executable_hash {:016x}\n", m.executable_hash));
    s.push_str(&format!("seed {}\n", m.seed));
    s.push_str(&format!("seed_size {}\n", m.seed_size));
    s.push_str(&format!("kmer_size {}\n", m.kmer_size));
    s.push_str(&format!("seed_pos {}\n", plan::join_csv(&m.seed_pos)));
    s.push_str(&format!("step {}\n", m.step));
    s.push_str(&format!("kegalign_bins {}\n", m.kegalign_bins as u8));
    s.push_str(&format!("seq_block_size {}\n", m.seq_block_size));
    s.push_str(&format!("n_ref_bins {}\n", m.bins.len()));
    s.push_str(&format!("n_records {}\n", m.n_records));
    s.push_str(&format!("ref_records_hash {:016x}\n", m.ref_records_hash));
    s.push_str(&format!("ref_file_sha256 {}\n", m.ref_file_sha256));
    s.push_str(&format!("ref_file_bytes {}\n", m.ref_file_bytes));
    s.push_str(&format!("threads {}\n", m.threads));
    for b in &m.bins {
        s.push_str(&format!(
            "bin R {} {} {}\n",
            b.bin.id,
            b.bin.total_bp,
            plan::join_csv(&b.bin.record_ids)
        ));
    }
    for b in &m.bins {
        for r in &b.recs {
            s.push_str(&format!(
                "rec {} {} {} {} {} {}\n",
                r.id, r.ordinal, r.len, r.bin_id, r.bin_start, r.name
            ));
        }
    }
    for b in &m.bins {
        let tag = format!("{:03}", b.bin.id);
        s.push_str(&format!(
            "checksum {tag}/index_table {} {:016x}\n",
            b.index_len, b.index_crc
        ));
        s.push_str(&format!(
            "checksum {tag}/pos_table {} {:016x}\n",
            b.pos_len, b.pos_crc
        ));
        s.push_str(&format!(
            "checksum {tag}/enc {} {:016x}\n",
            b.enc_len, b.enc_crc
        ));
    }
    s
}

/// `hspz index --reference R --index DIR [layout/seed]`: build every reference
/// bin's tables via the same [`build_ref_bin`] the executor uses and publish
/// them atomically (`{DIR}.tmp` → fsync → `READY` → rename onto `DIR`).
pub(crate) fn run(args: &IndexArgs) -> Fallible<()> {
    if args.target_prefix.as_ref().is_some_and(|s| !s.is_empty()) {
        return Err("hspZ index stores unprefixed names; pass --target-prefix on run".into());
    }
    if args.seq_block_size == 0 {
        return Err("hspZ index rejects -B 0; give --seq-block-size explicitly".into());
    }
    let shape = Shape::parse(&args.seed)?;
    let dir = &args.index;
    if dir.exists() {
        return Err(format!(
            "index directory {} already exists; delete it or pick another path",
            dir.display()
        )
        .into());
    }
    // Sibling `{DIR}.tmp`, never `/tmp/...`, so the publish rename does not
    // cross devices (NFS/Lustre home vs local scratch).
    let mut tmp_os = dir.as_os_str().to_owned();
    tmp_os.push(".tmp");
    let tmp = PathBuf::from(tmp_os);
    if tmp.exists() {
        if tmp.join("READY").exists() {
            return Err(format!(
                "index directory {} already exists; delete it or pick another path",
                tmp.display()
            )
            .into());
        }
        std::fs::remove_dir_all(&tmp)?;
    }

    let (_, records, _) = sequence::read_records(&args.reference)?;
    // The MANIFEST stores names as whitespace-separated tokens, so a name the
    // FASTA reader accepts but the MANIFEST cannot round-trip (empty, or holding
    // Unicode whitespace such as U+00A0) is refused here, before the build,
    // instead of by `run --index` after it.
    if let Some((name, _)) = records
        .iter()
        .find(|(n, _)| n.is_empty() || n.chars().any(char::is_whitespace))
    {
        return Err(format!(
            "index {}: record name {name:?} is empty or contains whitespace, which the \
             index MANIFEST cannot store; rename the record",
            dir.display()
        )
        .into());
    }
    let ref_meta = record_meta(&records);
    let target = args.seq_block_size as u64;
    let bins: Vec<Bin> = if args.kegalign_bins {
        plan::bin_records_sequential(&ref_meta, target)
    } else {
        plan::bin_records(&ref_meta, target)
    };
    // Bases are addressed with u32 (`Chr::len`, `pos_table`): a bin that does not
    // fit would publish wrapped positions that `run` can never use.
    for b in &bins {
        plan::packed_bin_len("reference", b).map_err(|e| {
            format!(
                "index {}: {e}; bases are addressed with u32, so a bin (and any single \
                 record) must stay below 4,294,967,295 bp",
                dir.display()
            )
        })?;
    }
    if bins.len() > MAX_BINS {
        return Err(format!(
            "index {}: {} reference bins exceeds the v1 limit of {MAX_BINS}",
            dir.display(),
            bins.len()
        )
        .into());
    }
    let threads = resolve_threads(args.threads);
    let ref_file_bytes = std::fs::metadata(&args.reference)
        .map(|m| m.len())
        .unwrap_or(0);
    // Provenance only: a missing `sha256sum` degrades to `-`, never an error.
    let ref_file_sha256 = match sha256_file(&args.reference) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "note: index {}: sha256sum unavailable ({e}); recording ref_file_sha256 as -",
                dir.display()
            );
            "-".to_string()
        }
    };

    let wall = Instant::now();
    let mut entries: Vec<BinEntry> = Vec::with_capacity(bins.len());
    let mut payload_bytes: u64 = 0;
    for rbin in &bins {
        let t = Instant::now();
        let (packed, table) = build_ref_bin(rbin, &records, "", &shape, args.step, threads);
        let build_ms = t.elapsed();
        let t = Instant::now();
        let sub = bin_subdir(&tmp, rbin.id);
        std::fs::create_dir_all(&sub)?;
        write_u32s(&sub.join("index_table"), &table.index_table)?;
        write_u32s(&sub.join("pos_table"), &table.pos_table)?;
        write_bytes(&sub.join("enc"), &packed.enc)?;
        let write_ms = t.elapsed();

        let index_len = table.index_table.len() as u64;
        let pos_len = table.pos_table.len() as u64;
        let enc_len = packed.enc.len() as u64;
        let bytes = index_len * 4 + pos_len * 4 + enc_len;
        payload_bytes += bytes;
        // `packed.chrs` is in `record_ids` order (the order `build` packed),
        // and the index stores unprefixed names, so the source record names
        // are authoritative while the starts come from the pack.
        let recs = rbin
            .record_ids
            .iter()
            .enumerate()
            .map(|(k, &rid)| {
                let (name, _) = &records[rid as usize];
                RecEntry {
                    id: rid,
                    ordinal: ref_meta[rid as usize].ordinal,
                    len: packed.chrs[k].len as u64,
                    bin_id: rbin.id,
                    bin_start: packed.chrs[k].start as u64,
                    name: name.clone(),
                }
            })
            .collect();
        entries.push(BinEntry {
            bin: rbin.clone(),
            block_len: packed.block_len as u64,
            index_len,
            pos_len,
            enc_len,
            index_crc: crc_u32s(&table.index_table),
            pos_crc: crc_u32s(&table.pos_table),
            enc_crc: crc_bytes(&packed.enc),
            recs,
        });
        if args.time {
            eprintln!(
                "index: bin {:03}: build {:.0} ms, write {:.0} ms, valid {} positions, {bytes} bytes",
                rbin.id,
                build_ms.as_secs_f64() * 1000.0,
                write_ms.as_secs_f64() * 1000.0,
                pos_len,
            );
        }
    }

    let manifest = IndexManifest {
        format_version: FORMAT_VERSION,
        packing_version: PACKING_VERSION,
        seed_builder_version: SEED_BUILDER_VERSION,
        byte_order: "little".to_string(),
        hspz_version: env!("CARGO_PKG_VERSION").into(),
        features: plan::compiled_features(),
        executable_hash: plan::executable_hash().unwrap_or(0),
        seed: args.seed.clone(),
        seed_size: shape.size,
        kmer_size: shape.kmer_size,
        seed_pos: shape.pos.clone(),
        step: args.step,
        kegalign_bins: args.kegalign_bins,
        seq_block_size: target,
        ref_records_hash: plan::records_hash(&records),
        ref_file_sha256,
        ref_file_bytes,
        threads,
        n_records: records.len(),
        bins: entries,
    };
    write_bytes(&tmp.join("MANIFEST"), manifest_text(&manifest).as_bytes())?;
    write_bytes(&tmp.join("READY"), READY_TEXT.as_bytes())?;
    // Publish: the directory entry itself must reach the disk before the
    // rename, or a crash can lose the whole `{DIR}.tmp`.
    std::fs::File::open(&tmp)?.sync_all()?;
    std::fs::rename(&tmp, dir)?;
    if args.time {
        eprintln!(
            "index: wrote {} bins, {payload_bytes} bytes payload in {:.0} ms wall",
            manifest.bins.len(),
            wall.elapsed().as_secs_f64() * 1000.0,
        );
    } else {
        eprintln!(
            "index: wrote {} bins ({} bytes) to {}",
            manifest.bins.len(),
            payload_bytes,
            dir.display()
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Manifest parsing
// ---------------------------------------------------------------------------

fn parse_u64(tok: &str, line: usize, what: &str, dir: &Path) -> Result<u64, String> {
    tok.parse::<u64>().map_err(|_| {
        format!(
            "index {} MANIFEST line {line}: bad {what} {tok:?}",
            dir.display()
        )
    })
}

fn parse_hex_u64(tok: &str, line: usize, what: &str, dir: &Path) -> Result<u64, String> {
    u64::from_str_radix(tok, 16).map_err(|_| {
        format!(
            "index {} MANIFEST line {line}: bad {what} {tok:?}",
            dir.display()
        )
    })
}

fn parse_csv_u32(tok: &str, line: usize, what: &str, dir: &Path) -> Result<Vec<u32>, String> {
    if tok.is_empty() {
        return Ok(Vec::new());
    }
    tok.split(',')
        .map(|p| {
            p.parse::<u32>().map_err(|_| {
                format!(
                    "index {} MANIFEST line {line}: bad {what} {tok:?}",
                    dir.display()
                )
            })
        })
        .collect()
}

fn parse_csv_usize(tok: &str, line: usize, what: &str, dir: &Path) -> Result<Vec<usize>, String> {
    if tok.is_empty() {
        return Ok(Vec::new());
    }
    tok.split(',')
        .map(|p| {
            p.parse::<usize>().map_err(|_| {
                format!(
                    "index {} MANIFEST line {line}: bad {what} {tok:?}",
                    dir.display()
                )
            })
        })
        .collect()
}

/// Reads and parses `DIR/MANIFEST`, enforcing versions and byte order. A
/// directory without `READY` is an unfinished publish, never an index.
pub(crate) fn load_manifest(dir: &Path) -> Fallible<IndexManifest> {
    if !dir.is_dir() {
        return Err(format!("index {} is not a directory", dir.display()).into());
    }
    match std::fs::read_to_string(dir.join("READY")) {
        Ok(s) if s.trim() == READY_TEXT.trim() => {}
        _ => {
            return Err(format!(
                "index {} is incomplete (missing READY); not a finished hspZ index",
                dir.display()
            )
            .into());
        }
    }
    let text = std::fs::read_to_string(dir.join("MANIFEST"))
        .map_err(|e| format!("index {} MANIFEST unreadable: {e}", dir.display()))?;
    parse_manifest(dir, &text).map_err(|e| e.into())
}

fn parse_manifest(dir: &Path, text: &str) -> Result<IndexManifest, String> {
    let err = |line: usize, m: &str| format!("index {} MANIFEST line {line}: {m}", dir.display());
    let mut scalars = std::collections::HashMap::<String, String>::new();
    let mut bin_lines: Vec<(usize, Vec<String>)> = Vec::new();
    let mut rec_lines: Vec<(usize, Vec<String>)> = Vec::new();
    let mut cks_lines: Vec<(usize, Vec<String>)> = Vec::new();
    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let toks: Vec<String> = line.split_whitespace().map(str::to_string).collect();
        let (key, rest) = (&toks[0], &toks[1..]);
        let line_no = lineno + 1;
        match key.as_str() {
            "bin" | "rec" | "checksum" => {
                let v = (line_no, rest.to_vec());
                match key.as_str() {
                    "bin" => bin_lines.push(v),
                    "rec" => rec_lines.push(v),
                    _ => cks_lines.push(v),
                }
            }
            _ => {
                if rest.len() != 1 {
                    return Err(err(line_no, &format!("key {key} wants one value")));
                }
                scalars.insert(key.clone(), rest[0].clone());
            }
        }
    }
    let get = |k: &str| -> Result<String, String> {
        scalars
            .get(k)
            .cloned()
            .ok_or_else(|| format!("index {} MANIFEST is missing {k}", dir.display()))
    };
    let one_u32 = |k: &str| -> Result<u32, String> {
        get(k)?
            .parse::<u32>()
            .map_err(|_| format!("index {} MANIFEST has bad {k}", dir.display()))
    };
    // Header tag.
    if get("hspz-index")? != "1" {
        return Err(format!(
            "index {} MANIFEST is not an hspz-index (bad hspz-index tag)",
            dir.display()
        ));
    }
    if get("magic")? != "HSPZIDX" {
        return Err(format!("index {} MANIFEST has bad magic", dir.display()));
    }
    let format_version = one_u32("format_version")?;
    let packing_version = one_u32("packing_version")?;
    let seed_builder_version = one_u32("seed_builder_version")?;
    for (name, v) in [
        ("format_version", format_version),
        ("packing_version", packing_version),
        ("seed_builder_version", seed_builder_version),
    ] {
        let want = match name {
            "format_version" => FORMAT_VERSION,
            "packing_version" => PACKING_VERSION,
            _ => SEED_BUILDER_VERSION,
        };
        if v != want {
            return Err(format!(
                "index {} {name} {v} is not supported (this binary reads \
                 format_version=1 packing_version=1 seed_builder_version=1)",
                dir.display()
            ));
        }
    }
    let byte_order = get("byte_order")?;
    if byte_order != "little" {
        return Err(format!(
            "index {} byte_order is '{byte_order}', expected little",
            dir.display()
        ));
    }
    let seed_size = parse_u64(&get("seed_size")?, 0, "seed_size", dir)? as usize;
    let kmer_size = parse_u64(&get("kmer_size")?, 0, "kmer_size", dir)? as usize;
    let seed_pos = parse_csv_usize(&get("seed_pos")?, 0, "seed_pos", dir)?;
    let step = parse_u64(&get("step")?, 0, "step", dir)? as u32;
    let kegalign_bins = match get("kegalign_bins")?.as_str() {
        "0" => false,
        "1" => true,
        other => {
            return Err(format!(
                "index {} MANIFEST has bad kegalign_bins {other:?}",
                dir.display()
            ));
        }
    };
    let seq_block_size = parse_u64(&get("seq_block_size")?, 0, "seq_block_size", dir)?;
    let n_ref_bins = parse_u64(&get("n_ref_bins")?, 0, "n_ref_bins", dir)? as usize;
    let n_records = parse_u64(&get("n_records")?, 0, "n_records", dir)? as usize;
    let ref_records_hash = parse_hex_u64(&get("ref_records_hash")?, 0, "ref_records_hash", dir)?;
    let ref_file_sha256 = get("ref_file_sha256")?;
    let ref_file_bytes = parse_u64(&get("ref_file_bytes")?, 0, "ref_file_bytes", dir)?;
    let threads = parse_u64(&get("threads")?, 0, "threads", dir)? as usize;
    let executable_hash = parse_hex_u64(&get("executable_hash")?, 0, "executable_hash", dir)?;

    let mut bins: Vec<BinEntry> = Vec::new();
    for (line_no, t) in &bin_lines {
        if t.len() != 4 || t[0] != "R" {
            return Err(err(
                *line_no,
                "bin line wants `bin R <id> <total_bp> <ids>`",
            ));
        }
        let id = parse_u64(&t[1], *line_no, "bin id", dir)? as u32;
        let total_bp = parse_u64(&t[2], *line_no, "bin bp", dir)?;
        let record_ids = parse_csv_u32(&t[3], *line_no, "bin record ids", dir)?;
        bins.push(BinEntry {
            bin: Bin {
                id,
                record_ids,
                total_bp,
            },
            block_len: 0,
            index_len: 0,
            pos_len: 0,
            enc_len: 0,
            index_crc: 0,
            pos_crc: 0,
            enc_crc: 0,
            recs: Vec::new(),
        });
    }
    if bins.len() != n_ref_bins {
        return Err(format!(
            "index {} MANIFEST declares n_ref_bins={n_ref_bins} but lists {} bins",
            dir.display(),
            bins.len()
        ));
    }
    fn bin_idx(bins: &[BinEntry], dir: &Path, id: u32, line_no: usize) -> Result<usize, String> {
        bins.iter().position(|b| b.bin.id == id).ok_or_else(|| {
            format!(
                "index {} MANIFEST line {line_no}: unknown bin {id}",
                dir.display()
            )
        })
    }
    for (line_no, t) in &rec_lines {
        if t.len() != 6 {
            return Err(err(
                *line_no,
                "rec line wants `rec <id> <ord> <len> <bin> <start> <name>`",
            ));
        }
        let id = parse_u64(&t[0], *line_no, "rec id", dir)? as u32;
        let ordinal = parse_u64(&t[1], *line_no, "rec ordinal", dir)? as u32;
        let len = parse_u64(&t[2], *line_no, "rec len", dir)?;
        let bin_id = parse_u64(&t[3], *line_no, "rec bin", dir)? as u32;
        let bin_start = parse_u64(&t[4], *line_no, "rec start", dir)?;
        let name = t[5].clone();
        let i = bin_idx(&bins, dir, bin_id, *line_no)?;
        bins[i].recs.push(RecEntry {
            id,
            ordinal,
            len,
            bin_id,
            bin_start,
            name,
        });
    }
    for (line_no, t) in &cks_lines {
        if t.len() != 3 {
            return Err(err(
                *line_no,
                "checksum line wants `checksum <bin>/<array> <len> <hex>`",
            ));
        }
        let (bindir, array) = t[0]
            .split_once('/')
            .ok_or_else(|| err(*line_no, "bad checksum path"))?;
        let id: u32 = bindir
            .parse()
            .map_err(|_| err(*line_no, "bad checksum bin"))?;
        let len = parse_u64(&t[1], *line_no, "checksum len", dir)?;
        let crc = parse_hex_u64(&t[2], *line_no, "checksum", dir)?;
        let i = bin_idx(&bins, dir, id, *line_no)?;
        match array {
            "index_table" => {
                bins[i].index_len = len;
                bins[i].index_crc = crc;
            }
            "pos_table" => {
                bins[i].pos_len = len;
                bins[i].pos_crc = crc;
            }
            "enc" => {
                bins[i].enc_len = len;
                bins[i].enc_crc = crc;
            }
            _ => return Err(err(*line_no, "unknown checksum array")),
        }
    }
    // Lengths come only from checksum lines: a bin without all three arrays
    // is not a bin the writer published.
    for b in &bins {
        let lens = [b.index_len, b.pos_len, b.enc_len];
        let crcs = [b.index_crc, b.pos_crc, b.enc_crc];
        // A zero length is legal for `pos_table` (no valid windows) but the
        // checksum lines must exist; `index_len` is always 4^k > 0.
        if b.index_len == 0 {
            return Err(format!(
                "index {} bin {}: MANIFEST is missing array checksums",
                dir.display(),
                b.bin.id
            ));
        }
        let _ = (lens, crcs);
    }
    // `block_len` is not stored per bin; it is `enc_len` by construction
    // (`enc = encode(buf[..block_len])`), and `load_bin` enforces it.
    for b in &mut bins {
        b.block_len = b.enc_len;
        b.recs.sort_by_key(|r| r.bin_start);
    }
    Ok(IndexManifest {
        format_version,
        packing_version,
        seed_builder_version,
        byte_order,
        hspz_version: get("hspz_version")?,
        features: get("features").unwrap_or_default(),
        executable_hash,
        seed: get("seed")?,
        seed_size,
        kmer_size,
        seed_pos,
        step,
        kegalign_bins,
        seq_block_size,
        ref_records_hash,
        ref_file_sha256,
        ref_file_bytes,
        threads,
        n_records,
        bins,
    })
}

// ---------------------------------------------------------------------------
// Compatibility (`run --index`)
// ---------------------------------------------------------------------------

/// Requires the resolved run (seed/step, layout, reference identity, exact bin
/// membership) to equal the index. `res_seq` is the resolved `--seq-block-size`
/// (the frozen manifest's on replay, where the CLI default is meaningless).
/// Pure so it is unit-testable without a GPU or an index directory.
pub(crate) fn check_run(
    dir: &Path,
    manifest: &IndexManifest,
    args: &RunArgs,
    res_seq: u64,
    ref_records: &[(String, Vec<u8>)],
    plan: &plan::Plan,
) -> Result<(), String> {
    for (name, v) in [
        ("format_version", manifest.format_version),
        ("packing_version", manifest.packing_version),
        ("seed_builder_version", manifest.seed_builder_version),
    ] {
        let want = match name {
            "format_version" => FORMAT_VERSION,
            "packing_version" => PACKING_VERSION,
            _ => SEED_BUILDER_VERSION,
        };
        if v != want {
            return Err(format!(
                "index {} {name} {v} is not supported (this binary reads \
                 format_version=1 packing_version=1 seed_builder_version=1)",
                dir.display()
            ));
        }
    }
    if manifest.byte_order != "little" {
        return Err(format!(
            "index {} byte_order is '{}', expected little",
            dir.display(),
            manifest.byte_order
        ));
    }
    let shape = Shape::parse(&args.seed)
        .map_err(|e| format!("index {}: cannot parse this run's seed: {e}", dir.display()))?;
    if manifest.seed != args.seed
        || manifest.kmer_size != shape.kmer_size
        || manifest.seed_size != shape.size
        || manifest.seed_pos != shape.pos
        || manifest.step != args.step
    {
        return Err(format!(
            "index {} seed is '{}' (k={} size={} pos={}) step {}; this run requested '{}' step {}",
            dir.display(),
            manifest.seed,
            manifest.kmer_size,
            manifest.seed_size,
            plan::join_csv(&manifest.seed_pos),
            manifest.step,
            args.seed,
            args.step,
        ));
    }
    if manifest.kegalign_bins != args.kegalign_bins || manifest.seq_block_size != res_seq {
        return Err(format!(
            "index {} was built with kegalign_bins={} --seq-block-size {}; this run has \
             kegalign_bins={} --seq-block-size {res_seq}",
            dir.display(),
            manifest.kegalign_bins as u8,
            manifest.seq_block_size,
            args.kegalign_bins as u8,
        ));
    }
    let got_hash = plan::records_hash(ref_records);
    if manifest.ref_records_hash != got_hash {
        return Err(format!(
            "index {} records_hash {:016x} != this reference {:016x} (reference content \
             differs; rebuild the index)",
            dir.display(),
            manifest.ref_records_hash,
            got_hash,
        ));
    }
    if manifest.bins.len() != plan.reference_bins.len() {
        return Err(format!(
            "index {} has R={} reference bins; this run planned R={} (plan_within_budget \
             shrank the layout, or -B/--kegalign-bins differ). Rebuild the index with \
             matching -B or use a device/budget that preserves the indexed bins",
            dir.display(),
            manifest.bins.len(),
            plan.reference_bins.len(),
        ));
    }
    for (entry, planned) in manifest.bins.iter().zip(plan.reference_bins.iter()) {
        if entry.bin.id != planned.id
            || entry.bin.record_ids != planned.record_ids
            || entry.bin.total_bp != planned.total_bp
        {
            return Err(format!(
                "index {} bin {}: record_ids/total_bp differ from this run's plan \
                 (index {:?} / {}; plan {:?} / {})",
                dir.display(),
                planned.id,
                entry.bin.record_ids,
                entry.bin.total_bp,
                planned.record_ids,
                planned.total_bp,
            ));
        }
    }
    for entry in &manifest.bins {
        check_bin_recs(dir, entry, ref_records)?;
    }
    Ok(())
}

/// Structural half of the `rec` binding (no FASTA): counts, pack-order ids,
/// and the `[0, block_len)` tiling. Runs in `load_bin`, before any upload.
fn check_rec_tiling(dir: &Path, entry: &BinEntry) -> Result<(), String> {
    let bin = &entry.bin;
    if entry.recs.len() != bin.record_ids.len() {
        return Err(format!(
            "index {} bin {}: MANIFEST lists {} recs but the bin has {} records",
            dir.display(),
            bin.id,
            entry.recs.len(),
            bin.record_ids.len(),
        ));
    }
    for (k, &rid) in bin.record_ids.iter().enumerate() {
        let r = &entry.recs[k];
        if r.id != rid {
            return Err(format!(
                "index {} bin {}: rec {k} id {} != pack order id {rid}",
                dir.display(),
                bin.id,
                r.id,
            ));
        }
        if r.bin_id != bin.id {
            return Err(format!(
                "index {} bin {}: rec id {rid} names bin {}",
                dir.display(),
                bin.id,
                r.bin_id,
            ));
        }
    }
    if entry.recs.is_empty() {
        if entry.block_len != 0 {
            return Err(format!(
                "index {} bin {}: empty recs but block_len {}",
                dir.display(),
                bin.id,
                entry.block_len,
            ));
        }
        return Ok(());
    }
    if entry.recs[0].bin_start != 0 {
        return Err(format!(
            "index {} bin {}: rec id {} first start {} != 0",
            dir.display(),
            bin.id,
            entry.recs[0].id,
            entry.recs[0].bin_start,
        ));
    }
    for w in entry.recs.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        if b.bin_start <= a.bin_start {
            return Err(format!(
                "index {} bin {}: rec id {} start {} is not after rec id {} start {}",
                dir.display(),
                bin.id,
                b.id,
                b.bin_start,
                a.id,
                a.bin_start,
            ));
        }
        if a.bin_start + a.len + 1 != b.bin_start {
            return Err(format!(
                "index {} bin {}: rec id {} (start {} len {}) does not tile rec id {} start {}",
                dir.display(),
                bin.id,
                a.id,
                a.bin_start,
                a.len,
                b.id,
                b.bin_start,
            ));
        }
    }
    let last = entry.recs.last().unwrap();
    if last.bin_start + last.len != entry.block_len {
        return Err(format!(
            "index {} bin {}: rec id {} end {} != block_len {}",
            dir.display(),
            bin.id,
            last.id,
            last.bin_start + last.len,
            entry.block_len,
        ));
    }
    Ok(())
}

/// Binds one bin's MANIFEST `rec` stanza to the parsed FASTA: unprefixed
/// names and lengths, on top of the structural tiling. Runs in `check_run`
/// for every bin, before any device upload.
fn check_bin_recs(
    dir: &Path,
    entry: &BinEntry,
    ref_records: &[(String, Vec<u8>)],
) -> Result<(), String> {
    check_rec_tiling(dir, entry)?;
    let bin = &entry.bin;
    for r in &entry.recs {
        let (want_name, want_seq) = ref_records.get(r.id as usize).ok_or_else(|| {
            format!(
                "index {} bin {}: rec id {} is past this reference ({} records)",
                dir.display(),
                bin.id,
                r.id,
                ref_records.len(),
            )
        })?;
        if r.name != *want_name {
            return Err(format!(
                "index {} bin {}: rec id {} name {:?} != reference {:?}",
                dir.display(),
                bin.id,
                r.id,
                r.name,
                want_name,
            ));
        }
        if r.len != want_seq.len() as u64 {
            return Err(format!(
                "index {} bin {}: rec id {} len {} != reference len {}",
                dir.display(),
                bin.id,
                r.id,
                r.len,
                want_seq.len(),
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Load path
// ---------------------------------------------------------------------------

/// Reads one array file: the metadata length must match the manifest, then the
/// bytes stream in with a running FNV-1a. Truncation is a length error,
/// bit-flips are checksum errors; neither reaches the device.
fn read_array(
    dir: &Path,
    id: u32,
    name: &str,
    expect_bytes: u64,
    want_crc: u64,
) -> Fallible<Vec<u8>> {
    let path = bin_subdir(dir, id).join(name);
    let meta = std::fs::metadata(&path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            format!("index {} bin {id}: missing {name}", dir.display())
        } else {
            format!("index {} bin {id}: cannot stat {name}: {e}", dir.display())
        }
    })?;
    if meta.len() != expect_bytes {
        return Err(format!(
            "index {} bin {id}: {name} length {} != expected {expect_bytes}",
            dir.display(),
            meta.len(),
        )
        .into());
    }
    let mut f = std::io::BufReader::new(
        std::fs::File::open(&path)
            .map_err(|e| format!("index {} bin {id}: cannot open {name}: {e}", dir.display()))?,
    );
    let mut out = Vec::new();
    // Bound the allocation before any byte is read: a corrupt manifest must
    // not drive a wild reserve.
    let n: usize = usize::try_from(expect_bytes).map_err(|_| {
        format!(
            "index {} bin {id}: n_records {} or block_len {} fails bounded-length checks",
            dir.display(),
            expect_bytes,
            expect_bytes,
        )
    })?;
    out.try_reserve_exact(n).map_err(|_| {
        format!(
            "index {} bin {id}: n_records {n} or block_len {n} fails bounded-length checks",
            dir.display()
        )
    })?;
    let mut crc = plan::FNV_OFFSET;
    let mut buf = [0u8; 1 << 20];
    let mut remaining = n;
    while remaining > 0 {
        let want = buf.len().min(remaining);
        f.read_exact(&mut buf[..want])
            .map_err(|e| format!("index {} bin {id}: cannot read {name}: {e}", dir.display()))?;
        crc = plan::fnv1a(&buf[..want], crc);
        out.extend_from_slice(&buf[..want]);
        remaining -= want;
    }
    if crc != want_crc {
        return Err(format!(
            "index {} bin {id}: checksum mismatch ({name} fnv1a {crc:016x} != {want_crc:016x})",
            dir.display()
        )
        .into());
    }
    Ok(out)
}

fn bytes_to_u32s(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Loads one reference bin's execution inputs: the same `(PackedBin,
/// SeedTable)` pair [`build_ref_bin`] returns. Only `enc`/`chrs`/`block_len`
/// are populated (`buf`/`rc`/`enc_rc` stay empty); the executor drops
/// everything but `chrs` after upload anyway.
pub(crate) fn load_bin(
    dir: &Path,
    manifest: &IndexManifest,
    rbin: &Bin,
    target_prefix: &str,
) -> Fallible<(PackedBin, SeedTable)> {
    let entry = manifest
        .bins
        .iter()
        .find(|b| b.bin.id == rbin.id)
        .ok_or_else(|| {
            format!(
                "index {} bin {}: record_ids/total_bp differ from this run's plan \
                 (index missing; plan {:?} / {})",
                dir.display(),
                rbin.id,
                rbin.record_ids,
                rbin.total_bp,
            )
        })?;
    if let Err(e) = check_rec_tiling(dir, entry) {
        return Err(e.into());
    }
    let k = manifest.kmer_size;
    let expect_nk: u64 = 4u64.checked_pow(k as u32).ok_or_else(|| {
        format!(
            "index {} bin {}: n_records {} or block_len {} fails bounded-length checks",
            dir.display(),
            rbin.id,
            entry.recs.len(),
            entry.block_len,
        )
    })?;
    if entry.index_len != expect_nk {
        return Err(format!(
            "index {} bin {}: index_table length {} != 4^{k} = {expect_nk}",
            dir.display(),
            rbin.id,
            entry.index_len,
        )
        .into());
    }
    if entry.block_len > u32::MAX as u64 {
        return Err(format!(
            "index {} bin {}: n_records {} or block_len {} fails bounded-length checks",
            dir.display(),
            rbin.id,
            entry.recs.len(),
            entry.block_len,
        )
        .into());
    }
    if entry.enc_len != entry.block_len {
        return Err(format!(
            "index {} bin {}: encoded length {} != block_len {}",
            dir.display(),
            rbin.id,
            entry.enc_len,
            entry.block_len,
        )
        .into());
    }
    let block_len = entry.block_len as u32;
    let index_bytes = entry.index_len.checked_mul(4).ok_or_else(|| {
        format!(
            "index {} bin {}: n_records {} or block_len {} fails bounded-length checks",
            dir.display(),
            rbin.id,
            entry.recs.len(),
            entry.block_len,
        )
    })?;
    let pos_bytes = entry.pos_len.checked_mul(4).ok_or_else(|| {
        format!(
            "index {} bin {}: n_records {} or block_len {} fails bounded-length checks",
            dir.display(),
            rbin.id,
            entry.recs.len(),
            entry.block_len,
        )
    })?;
    let index_table = bytes_to_u32s(&read_array(
        dir,
        rbin.id,
        "index_table",
        index_bytes,
        entry.index_crc,
    )?);
    let pos_table = bytes_to_u32s(&read_array(
        dir,
        rbin.id,
        "pos_table",
        pos_bytes,
        entry.pos_crc,
    )?);
    let enc = read_array(dir, rbin.id, "enc", entry.enc_len, entry.enc_crc)?;

    if let Some((&first, rest)) = index_table.split_first() {
        let mut prev = first;
        for (o, &end) in rest.iter().enumerate() {
            if end < prev {
                return Err(format!(
                    "index {} bin {}: index_table is not monotone cumulative ends at key {}",
                    dir.display(),
                    rbin.id,
                    o + 1,
                )
                .into());
            }
            prev = end;
        }
        if prev != pos_table.len() as u32 {
            return Err(format!(
                "index {} bin {}: index_table final end {prev} != pos_table length {}",
                dir.display(),
                rbin.id,
                pos_table.len(),
            )
            .into());
        }
    } else if !pos_table.is_empty() {
        return Err(format!(
            "index {} bin {}: index_table final end 0 != pos_table length {}",
            dir.display(),
            rbin.id,
            pos_table.len(),
        )
        .into());
    }
    for &p in &pos_table {
        if p >= block_len {
            return Err(format!(
                "index {} bin {}: position {p} >= block_len {}",
                dir.display(),
                rbin.id,
                block_len,
            )
            .into());
        }
    }
    for (i, &b) in enc.iter().enumerate() {
        if b > 7 {
            return Err(format!(
                "index {} bin {}: encoded symbol {b} at offset {i} is not in 0..=7",
                dir.display(),
                rbin.id,
            )
            .into());
        }
    }
    // Record separators: `enc[start-1] == E_NT` for every record after the
    // first, exactly where `pack` put the `&` join.
    let mut chrs = Vec::with_capacity(entry.recs.len());
    for r in &entry.recs {
        if r.bin_start + r.len > entry.block_len {
            return Err(format!(
                "index {} bin {}: n_records {} or block_len {} fails bounded-length checks",
                dir.display(),
                rbin.id,
                entry.recs.len(),
                entry.block_len,
            )
            .into());
        }
        if r.bin_start > 0 {
            let off = (r.bin_start - 1) as usize;
            if enc[off] != E_NT {
                return Err(format!(
                    "index {} bin {}: expected E_NT at record boundary offset {off}",
                    dir.display(),
                    rbin.id,
                )
                .into());
            }
        }
        chrs.push(crate::sequence::Chr {
            name: format!("{target_prefix}{}", r.name),
            start: r.bin_start as usize,
            len: r.len as u32,
        });
    }
    let packed = PackedBin {
        buf: Vec::new(),
        chrs,
        block_len: entry.block_len as usize,
        rc: Vec::new(),
        rc_chrs: Vec::new(),
        enc,
        enc_rc: Vec::new(),
    };
    Ok((
        packed,
        SeedTable {
            index_table,
            pos_table,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Cli, Command};
    use clap::Parser;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TMP_CTR: AtomicU64 = AtomicU64::new(0);

    /// Scratch directory unique to this test process run.
    fn scratch() -> PathBuf {
        let id = TMP_CTR.fetch_add(1, Ordering::SeqCst);
        std::env::temp_dir().join(format!("hspz-index-test-{}-{id}", std::process::id()))
    }

    fn write_fasta(dir: &Path, name: &str, records: &[(&str, &str)]) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let path = dir.join(name);
        let mut s = String::new();
        for (n, seq) in records {
            s.push_str(&format!(">{n}\n{seq}\n"));
        }
        std::fs::write(&path, s).unwrap();
        path
    }

    /// Small multi-record reference: mixed case, Ns, and enough length for
    /// real 12of19 windows at --step 1.
    fn fixture_records() -> Vec<(String, String)> {
        let a = "ACGT".repeat(1500);
        let mut b = "ACGT".repeat(1200);
        b.replace_range(100..140, &"n".repeat(40));
        let mut c = "ACGT".repeat(900);
        c.replace_range(0..60, &"acgt".repeat(15));
        vec![
            ("r0".to_string(), a),
            ("r1".to_string(), b),
            ("r2".to_string(), c),
        ]
    }

    fn index_args(reference: PathBuf, dir: PathBuf, seq_block_size: u32) -> IndexArgs {
        IndexArgs {
            reference,
            index: dir,
            seed: "12of19".into(),
            step: 1,
            seq_block_size,
            kegalign_bins: false,
            threads: 2,
            time: false,
            target_prefix: None,
        }
    }

    fn run_args_for(reference: PathBuf) -> RunArgs {
        match Cli::try_parse_from(["hspz", "run", "-r", "x", "-q", "y"])
            .unwrap()
            .command
        {
            Command::Run(mut a) => {
                a.reference = reference;
                a
            }
            _ => unreachable!(),
        }
    }

    /// Fixture inputs live outside the repository, so their location comes from
    /// the environment instead of a path baked into the source: `HSPZ_FIXTURE_CHR20`
    /// and `HSPZ_FIXTURE_R90` name directories holding them. A fixture test whose
    /// variable is unset skips.
    fn fixture_dir(var: &str) -> Option<PathBuf> {
        std::env::var_os(var).map(PathBuf::from)
    }

    /// One helper for the repeated scratch-dir + FASTA + `hspz index` triple.
    /// Returns the scratch base (for cleanup), the reference path, the index
    /// dir, the parsed manifest, and the parsed reference records.
    struct Fixture {
        base: PathBuf,
        rf: PathBuf,
        dir: PathBuf,
        man: IndexManifest,
        records: Vec<(String, Vec<u8>)>,
    }

    fn build_fixture(recs: &[(&str, &str)], seq_block_size: u32) -> Fixture {
        let base = scratch();
        let rf = write_fasta(&base, "ref.fa", recs);
        let dir = base.join("idx");
        run(&index_args(rf.clone(), dir.clone(), seq_block_size)).unwrap();
        let man = load_manifest(&dir).unwrap();
        let (_, records, _) = sequence::read_records(&rf).unwrap();
        Fixture {
            base,
            rf,
            dir,
            man,
            records,
        }
    }

    fn build_default_fixture(seq_block_size: u32) -> Fixture {
        let recs = fixture_records();
        let flat: Vec<(&str, &str)> = recs.iter().map(|(n, s)| (n.as_str(), s.as_str())).collect();
        build_fixture(&flat, seq_block_size)
    }

    /// MANIFEST text survives a write→parse round-trip field for field.
    #[test]
    fn manifest_round_trip() {
        let dir = PathBuf::from("/nonexistent");
        let m = IndexManifest {
            format_version: FORMAT_VERSION,
            packing_version: PACKING_VERSION,
            seed_builder_version: SEED_BUILDER_VERSION,
            byte_order: "little".into(),
            hspz_version: "0.1.0".into(),
            features: "a,b".into(),
            executable_hash: 0x1234_5678_9abc_def0,
            seed: "12of19".into(),
            seed_size: 19,
            kmer_size: 12,
            seed_pos: vec![0, 1, 2, 4, 7, 8, 11, 13, 15, 16, 17, 18],
            step: 1,
            kegalign_bins: false,
            seq_block_size: 10_000_000,
            ref_records_hash: 0xdead_beef_cafe_1234,
            ref_file_sha256: "abc".into(),
            ref_file_bytes: 42,
            threads: 4,
            n_records: 2,
            bins: vec![BinEntry {
                bin: Bin {
                    id: 0,
                    record_ids: vec![0, 1],
                    total_bp: 100,
                },
                block_len: 101,
                index_len: 16_777_216,
                pos_len: 50,
                enc_len: 101,
                index_crc: 1,
                pos_crc: 2,
                enc_crc: 3,
                recs: vec![
                    RecEntry {
                        id: 0,
                        ordinal: 0,
                        len: 60,
                        bin_id: 0,
                        bin_start: 0,
                        name: "chrA".into(),
                    },
                    RecEntry {
                        id: 1,
                        ordinal: 1,
                        len: 40,
                        bin_id: 0,
                        bin_start: 61,
                        name: "chrB".into(),
                    },
                ],
            }],
        };
        let back = parse_manifest(&dir, &manifest_text(&m)).unwrap();
        assert_eq!(back, m);
    }

    /// write→load is bit-identical to a fresh `build_ref_bin` for every bin:
    /// `index_table`, `pos_table`, `enc`, `block_len`, names and starts.
    #[test]
    fn write_load_bit_identity() {
        let f = build_default_fixture(8000);
        let (base, dir, man, records) = (f.base, f.dir, f.man, f.records);
        assert_eq!(
            man.bins.len(),
            2,
            "8000 bp target over ~15 kbp must bin as 2"
        );

        let shape = Shape::parse("12of19").unwrap();
        let ref_meta = record_meta(&records);
        let bins = plan::bin_records(&ref_meta, 8000);
        assert_eq!(bins.len(), man.bins.len());
        for rbin in &bins {
            let (want_packed, want_table) = build_ref_bin(rbin, &records, "", &shape, 1, 2);
            let (got_packed, got_table) = load_bin(&dir, &man, rbin, "").unwrap();
            assert_eq!(
                got_table.index_table, want_table.index_table,
                "bin {}",
                rbin.id
            );
            assert_eq!(got_table.pos_table, want_table.pos_table, "bin {}", rbin.id);
            assert_eq!(got_packed.enc, want_packed.enc, "bin {}", rbin.id);
            assert_eq!(
                got_packed.block_len, want_packed.block_len,
                "bin {}",
                rbin.id
            );
            assert_eq!(
                got_packed.chrs.len(),
                want_packed.chrs.len(),
                "bin {}",
                rbin.id
            );
            for (g, w) in got_packed.chrs.iter().zip(want_packed.chrs.iter()) {
                assert_eq!(g.name, w.name);
                assert_eq!(g.start, w.start);
                assert_eq!(g.len, w.len);
            }
            // A prefixed load renames but keeps coordinates.
            let (pre_packed, _) = load_bin(&dir, &man, rbin, "T_").unwrap();
            for (g, w) in pre_packed.chrs.iter().zip(want_packed.chrs.iter()) {
                assert_eq!(g.name, format!("T_{}", w.name));
                assert_eq!(g.start, w.start);
            }
        }
        std::fs::remove_dir_all(&base).unwrap();
    }

    /// Empty, tiny and single-record bins survive the round-trip (the seed
    /// builder's short-circuit paths).
    #[test]
    fn tiny_and_single_record_bins() {
        let base = scratch();
        let rf = write_fasta(&base, "tiny.fa", &[("only", "ACGTACGTACGT")]);
        let dir = base.join("idx");
        run(&index_args(rf.clone(), dir.clone(), 500_000_000)).unwrap();
        let man = load_manifest(&dir).unwrap();
        assert_eq!(man.bins.len(), 1);
        let (_, records, _) = sequence::read_records(&rf).unwrap();
        let shape = Shape::parse("12of19").unwrap();
        let ref_meta = record_meta(&records);
        let bins = plan::bin_records(&ref_meta, 500_000_000);
        let (want_packed, want_table) = build_ref_bin(&bins[0], &records, "", &shape, 1, 1);
        let (got_packed, got_table) = load_bin(&dir, &man, &bins[0], "").unwrap();
        assert_eq!(got_table.index_table, want_table.index_table);
        assert_eq!(got_table.pos_table, want_table.pos_table);
        assert_eq!(got_packed.enc, want_packed.enc);
        std::fs::remove_dir_all(&base).unwrap();
    }

    /// Truncating `pos_table` by 4 bytes is a length error, never a panic.
    #[test]
    fn truncated_pos_table_is_a_length_error() {
        let f = build_default_fixture(8000);
        let (base, dir, man) = (f.base, f.dir, f.man);
        let p = bin_subdir(&dir, 0).join("pos_table");
        let len = std::fs::metadata(&p).unwrap().len();
        let mut bytes = std::fs::read(&p).unwrap();
        bytes.truncate(bytes.len() - 4);
        std::fs::write(&p, &bytes).unwrap();
        let err = load_bin(&dir, &man, &man.bins[0].bin, "")
            .err()
            .expect("load must fail");
        let msg = err.to_string();
        assert!(msg.contains("length"), "{msg}");
        assert!(msg.contains(&format!("{len}")), "{msg}");
        std::fs::remove_dir_all(&base).unwrap();
    }

    /// Flipping one `enc` byte is a checksum error.
    #[test]
    fn flipped_enc_byte_is_a_checksum_error() {
        let f = build_default_fixture(8000);
        let (base, dir, man) = (f.base, f.dir, f.man);
        let p = bin_subdir(&dir, 0).join("enc");
        let mut bytes = std::fs::read(&p).unwrap();
        bytes[10] ^= 0x01;
        std::fs::write(&p, &bytes).unwrap();
        let err = load_bin(&dir, &man, &man.bins[0].bin, "")
            .err()
            .expect("load must fail");
        assert!(err.to_string().contains("checksum"), "{}", err.to_string());
        std::fs::remove_dir_all(&base).unwrap();
    }

    /// A publish without `READY` is incomplete, never an index.
    #[test]
    fn missing_ready_is_incomplete() {
        let f = build_default_fixture(8000);
        let (base, dir) = (f.base, f.dir);
        std::fs::remove_file(dir.join("READY")).unwrap();
        let err = load_manifest(&dir).err().expect("load must fail");
        assert!(
            err.to_string().contains("missing READY"),
            "{}",
            err.to_string()
        );
        std::fs::remove_dir_all(&base).unwrap();
    }

    /// `READY` with the wrong content is incomplete, not an index: existence
    /// alone does not publish.
    #[test]
    fn bad_ready_content_is_incomplete() {
        let f = build_default_fixture(8000);
        let (base, dir) = (f.base, f.dir);
        std::fs::write(dir.join("READY"), "stale\n").unwrap();
        let err = load_manifest(&dir).err().expect("load must fail");
        assert!(
            err.to_string().contains("missing READY"),
            "{}",
            err.to_string()
        );
        std::fs::remove_dir_all(&base).unwrap();
    }

    /// A flipped `rec` name does not match the parsed FASTA.
    #[test]
    fn rec_flipped_name_is_rejected() {
        let f = build_default_fixture(8000);
        let (base, rf, dir, records) = (f.base, f.rf, f.dir, f.records);
        let mut man = f.man;
        let rid = man.bins[0].recs[0].id;
        man.bins[0].recs[0].name = "wrong_name".into();
        let ref_meta = record_meta(&records);
        let plan = plan::plan_with(&ref_meta, &ref_meta, 8000, 8000, false);
        let args = run_args_for(rf);
        let err = check_run(&dir, &man, &args, 8000, &records, &plan)
            .err()
            .expect("check must fail");
        assert!(
            err.contains(&format!("bin {}", man.bins[0].bin.id)),
            "{err}"
        );
        assert!(err.contains(&format!("{rid}")), "{err}");
        assert!(err.contains("wrong_name"), "{err}");
        std::fs::remove_dir_all(&base).unwrap();
    }

    /// A shifted `rec` start breaks the `[0, block_len)` tiling.
    #[test]
    fn rec_shifted_start_is_rejected() {
        let f = build_default_fixture(8000);
        let (base, rf, dir, records) = (f.base, f.rf, f.dir, f.records);
        let mut man = f.man;
        // Shift a non-first rec when the fixture bins one that way;
        // otherwise break the first-start==0 rule on a single-rec bin.
        let bi = man.bins.iter().position(|b| b.recs.len() >= 2).unwrap_or(0);
        let bid = man.bins[bi].bin.id;
        if man.bins[bi].recs.len() >= 2 {
            man.bins[bi].recs[1].bin_start += 1;
        } else {
            man.bins[bi].recs[0].bin_start = 1;
        }
        let ref_meta = record_meta(&records);
        let plan = plan::plan_with(&ref_meta, &ref_meta, 8000, 8000, false);
        let args = run_args_for(rf);
        let err = check_run(&dir, &man, &args, 8000, &records, &plan)
            .err()
            .expect("check must fail");
        assert!(err.contains(&format!("bin {bid}")), "{err}");
        // The per-visit loader rejects the same manifest before any upload.
        let rbin = man.bins[bi].bin.clone();
        let lerr = load_bin(&dir, &man, &rbin, "")
            .err()
            .expect("load must fail");
        assert!(
            lerr.to_string().contains(&format!("bin {}", rbin.id)),
            "{}",
            lerr.to_string()
        );
        std::fs::remove_dir_all(&base).unwrap();
    }

    /// A dropped `rec` leaves the bin short of its `record_ids`.
    #[test]
    fn rec_dropped_record_is_rejected() {
        let f = build_default_fixture(8000);
        let (base, rf, dir, records) = (f.base, f.rf, f.dir, f.records);
        let mut man = f.man;
        man.bins[0].recs.remove(0);
        let ref_meta = record_meta(&records);
        let plan = plan::plan_with(&ref_meta, &ref_meta, 8000, 8000, false);
        let args = run_args_for(rf);
        let err = check_run(&dir, &man, &args, 8000, &records, &plan)
            .err()
            .expect("check must fail");
        assert!(
            err.contains(&format!("bin {}", man.bins[0].bin.id)),
            "{err}"
        );
        std::fs::remove_dir_all(&base).unwrap();
    }

    /// An extra `rec` exceeds the bin's `record_ids`.
    #[test]
    fn rec_extra_record_is_rejected() {
        let f = build_default_fixture(8000);
        let (base, rf, dir, records) = (f.base, f.rf, f.dir, f.records);
        let mut man = f.man;
        let dup = man.bins[0].recs[0].clone();
        man.bins[0].recs.push(dup);
        let ref_meta = record_meta(&records);
        let plan = plan::plan_with(&ref_meta, &ref_meta, 8000, 8000, false);
        let args = run_args_for(rf);
        let err = check_run(&dir, &man, &args, 8000, &records, &plan)
            .err()
            .expect("check must fail");
        assert!(
            err.contains(&format!("bin {}", man.bins[0].bin.id)),
            "{err}"
        );
        std::fs::remove_dir_all(&base).unwrap();
    }

    /// A same-size single-base edit changes `records_hash` and is rejected.
    #[test]
    fn stale_reference_same_size_is_rejected() {
        let f = build_default_fixture(8000);
        let (base, rf, dir, man) = (f.base, f.rf, f.dir, f.man);
        let recs = fixture_records();

        let mut stale: Vec<(String, String)> = recs;
        let first = std::mem::take(&mut stale[0].1);
        let mut bytes = first.into_bytes();
        let i = bytes.iter().position(|&b| b == b'A').unwrap();
        bytes[i] = b'C';
        stale[0].1 = String::from_utf8(bytes).unwrap();
        assert_eq!(stale[0].1.len(), records_len(&rf, 0));
        let stale_records: Vec<(String, Vec<u8>)> = stale
            .into_iter()
            .map(|(n, s)| (n, s.into_bytes()))
            .collect();
        let ref_meta = record_meta(&stale_records);
        let plan = plan::plan_with(&ref_meta, &ref_meta, 8000, 8000, false);
        let args = run_args_for(rf);
        let err = check_run(&dir, &man, &args, 8000, &stale_records, &plan)
            .err()
            .expect("load must fail");
        assert!(err.contains("records_hash"), "{err}");
        std::fs::remove_dir_all(&base).unwrap();
    }

    fn records_len(path: &Path, idx: usize) -> usize {
        let (_, records, _) = sequence::read_records(path).unwrap();
        records[idx].1.len()
    }

    /// A `-B` the index was not built with is a bin-membership error.
    #[test]
    fn block_size_mismatch_is_a_bin_error() {
        let f = build_default_fixture(8000);
        let (base, rf, dir, man, records) = (f.base, f.rf, f.dir, f.man, f.records);
        let ref_meta = record_meta(&records);
        // A much larger target collapses the two bins into one.
        let plan = plan::plan_with(&ref_meta, &ref_meta, 500_000_000, 500_000_000, false);
        assert_eq!(plan.reference_bins.len(), 1);
        let mut args = run_args_for(rf);
        args.seq_block_size = 500_000_000;
        let err = check_run(&dir, &man, &args, 500_000_000, &records, &plan)
            .err()
            .expect("load must fail");
        assert!(err.contains("--seq-block-size"), "{err}");
        // Same collapsed plan against the indexed target exercises the R-count
        // branch instead.
        let err = check_run(&dir, &man, &args, 8000, &records, &plan)
            .err()
            .expect("load must fail");
        assert!(err.contains("R="), "{err}");
        std::fs::remove_dir_all(&base).unwrap();
    }

    /// Seed and step mismatches name both sides.
    #[test]
    fn seed_and_step_mismatches() {
        let f = build_default_fixture(8000);
        let (base, rf, dir, man, records) = (f.base, f.rf, f.dir, f.man, f.records);
        let ref_meta = record_meta(&records);
        let plan = plan::plan_with(&ref_meta, &ref_meta, 8000, 8000, false);

        let mut args = run_args_for(rf.clone());
        args.seed = "14of22".into();
        let err = check_run(&dir, &man, &args, 8000, &records, &plan)
            .err()
            .expect("load must fail");
        assert!(err.contains("seed"), "{err}");

        let mut args = run_args_for(rf);
        args.step = 2;
        let err = check_run(&dir, &man, &args, 8000, &records, &plan)
            .err()
            .expect("load must fail");
        assert!(err.contains("step"), "{err}");
        std::fs::remove_dir_all(&base).unwrap();
    }

    /// Crafted corruptions trip the structural validators, in order.
    #[test]
    fn structural_validators() {
        let f = build_default_fixture(8000);
        let (base, rf, dir, man) = (f.base, f.rf, f.dir, f.man);
        let _ = rf;
        let rbin = man.bins[0].bin.clone();

        // Non-monotone index_table with intact checksums: rewrite the file and
        // re-stamp the manifest checksum (checksum passes, monotone must fail).
        let rewrite = |bin: u32, array: &str, f: &dyn Fn(&mut Vec<u32>)| -> IndexManifest {
            let p = bin_subdir(&dir, bin).join(array);
            let n = std::fs::metadata(&p).unwrap().len() as usize / 4;
            let bytes = std::fs::read(&p).unwrap();
            let mut v: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(v.len(), n);
            f(&mut v);
            let mut raw = Vec::with_capacity(v.len() * 4);
            for &w in &v {
                raw.extend_from_slice(&w.to_le_bytes());
            }
            std::fs::write(&p, &raw).unwrap();
            let mut m2 = load_manifest(&dir).unwrap();
            // Re-stamp so the failure is structural, not a checksum error.
            let e = m2.bins.iter_mut().find(|b| b.bin.id == bin).unwrap();
            if array == "index_table" {
                e.index_crc = crc_u32s(&v);
            } else {
                e.pos_crc = crc_u32s(&v);
            }
            m2
        };
        let p_index = bin_subdir(&dir, 0).join("index_table");
        let pristine_index = std::fs::read(&p_index).unwrap();
        let m2 = rewrite(0, "index_table", &|v: &mut Vec<u32>| {
            // Break monotonicity just past the first non-empty bucket.
            let i = v.iter().position(|&x| x > 0).unwrap() + 1;
            v[i] = v[i - 1] - 1;
        });
        let err = load_bin(&dir, &m2, &rbin, "")
            .err()
            .expect("load must fail");
        assert!(err.to_string().contains("monotone"), "{}", err.to_string());

        // A position past the block end: restore the pristine index_table file
        // first so this part reaches the position check.
        std::fs::write(&p_index, &pristine_index).unwrap();

        // A position past the block end (the pristine index_table above and
        // the on-disk manifest checksums both hold again).
        let m3 = rewrite(0, "pos_table", &|v: &mut Vec<u32>| {
            v[0] = u32::MAX;
        });
        let err = load_bin(&dir, &m3, &rbin, "")
            .err()
            .expect("load must fail");
        assert!(err.to_string().contains("position"), "{}", err.to_string());
        std::fs::remove_dir_all(&base).unwrap();
    }

    /// Illegal symbols and missing separators are rejected.
    #[test]
    fn symbol_and_separator_validators() {
        let base = scratch();
        let rf = write_fasta(
            &base,
            "ref.fa",
            &[("a", &"ACGT".repeat(2000)), ("b", &"ACGT".repeat(2000))],
        );
        let dir = base.join("idx");
        run(&index_args(rf, dir.clone(), 500_000_000)).unwrap();
        let man = load_manifest(&dir).unwrap();
        assert_eq!(man.bins.len(), 1);
        let rbin = man.bins[0].bin.clone();

        // Illegal enc symbol 9 with a re-stamped checksum.
        let p = bin_subdir(&dir, 0).join("enc");
        let mut bytes = std::fs::read(&p).unwrap();
        bytes[0] = 9;
        std::fs::write(&p, &bytes).unwrap();
        let mut m2 = load_manifest(&dir).unwrap();
        m2.bins[0].enc_crc = crc_bytes(&bytes);
        let err = load_bin(&dir, &m2, &rbin, "")
            .err()
            .expect("load must fail");
        assert!(err.to_string().contains("symbol"), "{}", err.to_string());

        // Zero the record-boundary separator instead (restoring the symbol
        // corrupted above first).
        let mut m3 = load_manifest(&dir).unwrap();
        let start = m3.bins[0].recs[1].bin_start as usize;
        let mut bytes = std::fs::read(&p).unwrap();
        bytes[0] = 0; // `a` starts with A
        assert_eq!(bytes[start - 1], E_NT);
        bytes[start - 1] = 0;
        std::fs::write(&p, &bytes).unwrap();
        m3.bins[0].enc_crc = crc_bytes(&bytes);
        let err = load_bin(&dir, &m3, &rbin, "")
            .err()
            .expect("load must fail");
        assert!(err.to_string().contains("E_NT"), "{}", err.to_string());
        std::fs::remove_dir_all(&base).unwrap();
    }

    /// CLI rejections: `-B 0`, an existing `DIR`, and `-T`.
    #[test]
    fn cli_rejections() {
        let base = scratch();
        let rf = write_fasta(&base, "ref.fa", &[("a", &"ACGT".repeat(2000))]);
        // -B 0.
        let err = run(&index_args(rf.clone(), base.join("i0"), 0))
            .err()
            .expect("load must fail");
        assert!(err.to_string().contains("-B 0"), "{}", err.to_string());
        // Existing DIR.
        let dir = base.join("idx");
        std::fs::create_dir_all(&dir).unwrap();
        let err = run(&index_args(rf.clone(), dir, 8000))
            .err()
            .expect("load must fail");
        assert!(
            err.to_string().contains("already exists"),
            "{}",
            err.to_string()
        );
        // -T passes clap (hidden) but the command rejects it with the design
        // message.
        let mut args = index_args(rf, base.join("iT"), 8000);
        args.target_prefix = Some("T_".into());
        let err = run(&args).err().expect("load must fail");
        assert!(
            err.to_string().contains("unprefixed"),
            "{}",
            err.to_string()
        );
        // And clap itself accepts the hidden spelling.
        let cli = Cli::try_parse_from(["hspz", "index", "-r", "r.fa", "--index", "d", "-T", "T_"])
            .unwrap();
        assert!(matches!(cli.command, Command::Index(_)));
        std::fs::remove_dir_all(&base).unwrap();
    }

    /// Names the MANIFEST cannot round-trip are refused before the build; the
    /// FASTA reader splits names only on ASCII whitespace, the MANIFEST parser
    /// on Unicode whitespace.
    #[test]
    fn unstorable_record_names_are_rejected_before_the_build() {
        let base = scratch();
        let seq = "ACGT".repeat(2000);
        for (i, name) in ["a\u{a0}b", ""].into_iter().enumerate() {
            let rf = write_fasta(
                &base,
                &format!("ref{i}.fa"),
                &[("ok", seq.as_str()), (name, seq.as_str())],
            );
            let dir = base.join(format!("idx{i}"));
            let err = run(&index_args(rf, dir.clone(), 8000))
                .err()
                .expect("index must refuse the name");
            assert!(err.to_string().contains("whitespace"), "{err}");
            assert!(!dir.exists(), "nothing may be published");
        }
        std::fs::remove_dir_all(&base).unwrap();
    }

    /// Fixture array identity over a real chromosome: every bin the chr20
    /// index holds must equal a fresh `build_ref_bin`. Needs the fixture;
    /// run explicitly in the round gate.
    #[test]
    #[ignore]
    fn chr20_index_arrays_match_fresh_build() {
        let base = scratch();
        let Some(dir) = fixture_dir("HSPZ_FIXTURE_CHR20") else {
            eprintln!("skipped: set HSPZ_FIXTURE_CHR20 to a directory holding ref.fa");
            return;
        };
        let rf = dir.join("ref.fa");
        let dir = base.join("chr20.idx");
        run(&index_args(rf.clone(), dir.clone(), 500_000_000)).unwrap();
        let man = load_manifest(&dir).unwrap();
        let (_, records, _) = sequence::read_records(&rf).unwrap();
        let shape = Shape::parse("12of19").unwrap();
        let ref_meta = record_meta(&records);
        let bins = plan::bin_records(&ref_meta, 500_000_000);
        assert_eq!(bins.len(), man.bins.len());
        for rbin in &bins {
            let (want_packed, want_table) = build_ref_bin(rbin, &records, "", &shape, 1, 0);
            let (got_packed, got_table) = load_bin(&dir, &man, rbin, "").unwrap();
            assert_eq!(
                got_table.index_table, want_table.index_table,
                "bin {}",
                rbin.id
            );
            assert_eq!(got_table.pos_table, want_table.pos_table, "bin {}", rbin.id);
            assert_eq!(got_packed.enc, want_packed.enc, "bin {}", rbin.id);
            assert_eq!(
                got_packed.block_len, want_packed.block_len,
                "bin {}",
                rbin.id
            );
        }
        std::fs::remove_dir_all(&base).unwrap();
    }

    /// Fixture array identity over the 5-bin reference at `-B 10000000`.
    #[test]
    #[ignore]
    fn ref5_index_arrays_match_fresh_build() {
        let base = scratch();
        let Some(dir) = fixture_dir("HSPZ_FIXTURE_R90") else {
            eprintln!("skipped: set HSPZ_FIXTURE_R90 to a directory holding ref5.fa");
            return;
        };
        let rf = dir.join("ref5.fa");
        let dir = base.join("ref5.idx");
        run(&index_args(rf.clone(), dir.clone(), 10_000_000)).unwrap();
        let man = load_manifest(&dir).unwrap();
        assert_eq!(man.bins.len(), 5);
        let (_, records, _) = sequence::read_records(&rf).unwrap();
        let shape = Shape::parse("12of19").unwrap();
        let ref_meta = record_meta(&records);
        let bins = plan::bin_records(&ref_meta, 10_000_000);
        assert_eq!(bins.len(), 5);
        for rbin in &bins {
            let (want_packed, want_table) = build_ref_bin(rbin, &records, "", &shape, 1, 0);
            let (got_packed, got_table) = load_bin(&dir, &man, rbin, "").unwrap();
            assert_eq!(
                got_table.index_table, want_table.index_table,
                "bin {}",
                rbin.id
            );
            assert_eq!(got_table.pos_table, want_table.pos_table, "bin {}", rbin.id);
            assert_eq!(got_packed.enc, want_packed.enc, "bin {}", rbin.id);
        }
        std::fs::remove_dir_all(&base).unwrap();
    }
}
