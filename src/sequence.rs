// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung
// Distributed under the terms of the GNU General Public License, Version 3.0.

// Author : Alejandro Gonzales-Irribarren
// Github : alejandrogzi
// Email  : alejandrxgzi@gmail.com

//! FASTA, FASTA.gz and 2bit readers, and the concatenated buffer layout
//! KegAlign feeds to the GPU.
//!
//! Formats are detected by magic bytes and decode to the same
//! packed representation, so the input format never changes the HSP hash.
//! KegAlign packs every record of a file into one buffer separated by a single
//! `&` byte, then hands the GPU a *block* of that buffer. The trailing `&` of
//! the last record is present in the buffer but excluded from the block length
//! (`main.cpp`: `seq_block_len--`), so the device never sees it.
//!
//! Coordinate conventions used throughout the port:
//!   * `Chr::start` is an offset into `Genome::buf` (block 0 starts at 0).
//!   * HSP coordinates out of the GPU are block-relative.
//!   * `.segments` coordinates are 1-based, chromosome-relative, inclusive at
//!     both ends (`segment_printer.cpp`).

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Sequence separator KegAlign inserts between records (`E_NT` on the device).
pub const SEP: u8 = b'&';

// Device alphabet (`parameters.h`). The substitution matrix is NUC x NUC.
// `A_NT`..`T_NT` are the four bases, `L_NT` soft-masked `acgt`, `N_NT` Ns,
// `X_NT` any other IUPAC code, `E_NT` the record separator, and `NUC` / `NUC2`
// the alphabet size and its square (the matrix dimension).
pub const A_NT: u8 = 0;
pub const C_NT: u8 = 1;
pub const G_NT: u8 = 2;
pub const T_NT: u8 = 3;
pub const L_NT: u8 = 4; // soft-masked acgt
pub const N_NT: u8 = 5;
pub const X_NT: u8 = 6; // any other IUPAC code
pub const E_NT: u8 = 7; // sequence separator
pub const NUC: usize = 8;
pub const NUC2: usize = NUC * NUC;

/// `ntcoding.cpp: NtChar2Int` — the *seeding* alphabet. Only uppercase ACGT is
/// codeable; everything else (including soft-masked acgt) reads as `N_NT` and
/// invalidates the k-mer.
pub fn nt_char_to_int(nt: u8) -> u8 {
    match nt {
        b'A' => A_NT,
        b'C' => C_NT,
        b'G' => G_NT,
        b'T' => T_NT,
        _ => N_NT,
    }
}

/// `seed_filter_interface.cu: compress_string` — the *scoring* alphabet, which
/// unlike the seeding one distinguishes soft-masked bases and separators.
///
/// # Example
/// ```
/// assert_eq!(encode(b"Acgt&"), vec![0, 4, 4, 4, 7]);
/// ```
pub fn encode(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .map(|&ch| match ch {
            b'A' => A_NT,
            b'C' => C_NT,
            b'G' => G_NT,
            b'T' => T_NT,
            b'a' | b'c' | b'g' | b't' => L_NT,
            b'n' | b'N' => N_NT,
            SEP => E_NT,
            _ => X_NT,
        })
        .collect()
}

/// `ntcoding.cpp: rev_comp` — 128-entry ASCII complement table, IUPAC aware.
/// Anything outside the table (or >127) becomes `N`, matching `RevComp()`.
const REV_COMP: &[u8; 128] = b"NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN\
NNNNNN&NNNNNNNNNNNNNNNNNNNNNNNNN\
NTVGHNNCDNNMNKNNNNYSAABWNRNNNNNN\
NtvghnncdnnmnknnnnysaabwnrnNNNNN";

#[derive(Debug, Clone)]
pub struct Chr {
    pub name: String,
    /// Offset of the first base within the owning buffer.
    pub start: usize,
    pub len: u32,
}

/// One FASTA file loaded into KegAlign's packed representation.
pub struct Genome {
    /// Records joined by `SEP`, including the trailing separator.
    pub buf: Vec<u8>,
    pub chrs: Vec<Chr>,
    /// Bytes of `buf` that make up block 0 — i.e. `buf.len()` minus the
    /// trailing separator.
    pub block_len: usize,
    /// Which reader produced this, for the input report.
    pub format: Format,
    /// On-disk size of the source file, so input throughput is reportable.
    pub bytes_read: u64,
}

/// Which reader a path needs.
///
/// Chosen from magic bytes, never from the extension: a `.fa` that is really
/// gzipped, or a `.2bit` named `.bin`, both still work. The extension only
/// improves error messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Fasta,
    FastaGz,
    TwoBit,
}

impl Format {
    /// `2bit` magic is `0x1A412743` in the file's own byte order; gzip is
    /// `1f 8b`; anything else is treated as plain FASTA.
    pub fn detect(path: &Path) -> Result<Self, String> {
        let mut head = [0u8; 4];
        let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let n = f
            .read(&mut head)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if n >= 4 {
            let be = u32::from_be_bytes(head);
            let le = u32::from_le_bytes(head);
            if be == 0x1A41_2743 || le == 0x1A41_2743 {
                return Ok(Format::TwoBit);
            }
        }
        if n >= 2 && head[..2] == [0x1f, 0x8b] {
            return Ok(Format::FastaGz);
        }
        Ok(Format::Fasta)
    }

    pub fn label(self) -> &'static str {
        match self {
            Format::Fasta => "fasta",
            Format::FastaGz => "fasta.gz",
            Format::TwoBit => "2bit",
        }
    }
}

/// Reads a 2bit file into the same `(name, bases)` records a FASTA yields.
///
/// Soft masks must survive, so `enable_softmask(true)` is mandatory — without
/// it every base comes back uppercase and the seeding alphabet silently changes
/// (lowercase kills a seed, uppercase does not).
/// Record order is the file's own index order, which `chrom_names()` preserves;
/// that order is load-bearing because the `&` separator, the chr table and
/// interval chunking all depend on it.
fn read_2bit(path: &Path) -> Result<Vec<(String, Vec<u8>)>, String> {
    let mut tb = twobit::TwoBitFile::open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .enable_softmask(true);
    let names = tb.chrom_names();
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let seq = tb
            .read_sequence(&name, ..)
            .map_err(|e| format!("{}: {name}: {e}", path.display()))?;
        out.push((name, seq.into_bytes()));
    }
    Ok(out)
}

/// Packs records into the single buffer the engine consumes: bases joined by
/// `SEP`, with a trailing separator present in the buffer but excluded from
/// `block_len`.
///
/// Extracted as a free function precisely so a bin and a whole genome cannot
/// drift apart: `Genome::load` and `PackedBin` both call this, so byte-identity
/// holds by construction rather than resting on a cross-implementation test.
///
/// Returns `(buf, chrs, block_len)`.
pub fn pack<'a>(
    records: impl IntoIterator<Item = (&'a str, &'a [u8])>,
    prefix: &str,
) -> (Vec<u8>, Vec<Chr>, usize) {
    let mut buf = Vec::new();
    let mut chrs = Vec::new();
    let mut block_len: usize = 0;
    for (name, seq) in records {
        chrs.push(Chr {
            name: format!("{prefix}{name}"),
            start: buf.len(),
            len: seq.len() as u32,
        });
        buf.extend_from_slice(seq);
        block_len += seq.len();
        buf.push(SEP);
        block_len += 1;
    }
    block_len = block_len.saturating_sub(1);
    (buf, chrs, block_len)
}

/// Reverse complement of a packed block, with the chromosome table remapped onto
/// it (`main.cpp`: `rc_q_chr_start = block_len - start - len`).
///
/// Free function for the same reason as [`pack`]: a bin's reverse complement must
/// be the identical algorithm, not a second one that agrees today.
pub fn reverse_complement(buf: &[u8], chrs: &[Chr], block_len: usize) -> (Vec<u8>, Vec<Chr>) {
    let n = block_len;
    let rc: Vec<u8> = (0..n)
        .map(|i| {
            let c = buf[n - 1 - i];
            if c < 128 { REV_COMP[c as usize] } else { b'N' }
        })
        .collect();
    let rc_chrs = chrs
        .iter()
        .rev()
        .map(|c| Chr {
            name: c.name.clone(),
            start: n - c.start - c.len as usize,
            len: c.len,
        })
        .collect();
    (rc, rc_chrs)
}

/// Reads a file into its raw `(name, bases)` records, format and byte count,
/// without packing or any block-size guard. The multi-block
/// executor needs the records themselves so it can bin them; [`Genome::load`]
/// is the single-block wrapper that packs and enforces the old guard.
// (format, records, bytes) is the reader's return, documented at the fn
#[allow(clippy::type_complexity)]
pub fn read_records(path: &Path) -> Result<(Format, Vec<(String, Vec<u8>)>, u64), String> {
    let format = Format::detect(path)?;
    let records = match format {
        Format::Fasta | Format::FastaGz => read_fasta(path)?,
        Format::TwoBit => read_2bit(path)?,
    };
    let bytes_read = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    Ok((format, records, bytes_read))
}

impl Genome {
    /// Loads FASTA, gzipped FASTA or 2bit — whichever the magic bytes say.
    ///
    /// Returns an error when the input would spill into a second block, which
    /// v1 does not implement — block/interval partitioning of the reference is
    /// part of the KegAlign runner, not Seed+Filter.
    pub fn load(path: &Path, prefix: &str, seq_block_size: u32) -> Result<Self, String> {
        let (format, records, bytes_read) = read_records(path)?;

        let (buf, chrs, block_len) = pack(
            records.iter().map(|(n, s)| (n.as_str(), s.as_slice())),
            prefix,
        );
        if block_len > seq_block_size as usize {
            // This guard stays until the multi-block executor is the selected
            // path; removing it earlier would swap a clear message for a host
            // OOM or a doomed single-block run.
            return Err(format!(
                "{} exceeds one sequence block ({} > {} bytes); multi-block input is out of \
                 scope for v1 — raise --seq_block_size",
                path.display(),
                block_len,
                seq_block_size
            ));
        }

        Ok(Genome {
            buf,
            chrs,
            block_len,
            format,
            bytes_read,
        })
    }

    /// FNV-1a hash of everything the core consumes from this genome: the record
    /// order, every name, every base including case, and the block length.
    ///
    /// This is what makes a reader bug attributable. If FASTA, gzipped FASTA and
    /// 2bit disagree, the hash differs *here*, before any GPU work, so a lost
    /// soft mask or a reordered record cannot hide behind an identical HSP count.
    pub fn digest(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |bytes: &[u8]| {
            for &b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
        };
        for c in &self.chrs {
            eat(c.name.as_bytes());
            eat(&c.start.to_le_bytes());
            eat(&c.len.to_le_bytes());
        }
        eat(&self.buf);
        eat(&self.block_len.to_le_bytes());
        h
    }

    /// Reverse complement of block 0, plus the chromosome table remapped onto
    /// it (`main.cpp`: `rc_q_chr_start = block_len - start - len`).
    pub fn reverse_complement(&self) -> (Vec<u8>, Vec<Chr>) {
        reverse_complement(&self.buf, &self.chrs, self.block_len)
    }
}

/// `segment_printer.cpp` resolves a block offset to a chromosome with
/// `upper_bound(starts, off) - 1`.
pub fn chr_at(chrs: &[Chr], off: usize) -> usize {
    chrs.partition_point(|c| c.start <= off) - 1
}

/// Splits a block into the LASTZ intervals the seeder iterates over
/// (`main.cpp`), as `(start, end)` pairs; `end` is inclusive for seeding.
pub fn intervals(block_len: usize, seed_size: usize, interval_size: u32) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    if block_len <= seed_size {
        return out;
    }
    let end_pos = (block_len - seed_size) as u32;
    let mut curr = 0u32;
    while curr < end_pos {
        out.push((curr, end_pos.min(curr + interval_size)));
        curr += interval_size;
    }
    out
}

/// Minimal FASTA reader matching `kseq`: name is the first whitespace-delimited
/// token after `>`, and all whitespace inside sequence lines is dropped.
///
/// Reads by block and splits on newlines in place. The obvious
/// `BufReader::lines()` version allocated a `String` per line — ~1.1 M of them
/// for an hg38 chromosome — and ran at ~350 MB/s; this one does no per-line
/// allocation. Input loading measured 2.3% of a cold run and 0% of warm (it
/// happens once, outside the benchmark loop), so an mmap specialization cannot
/// be earned here — this version is strictly *less* code than what it
/// replaces.
fn read_fasta(path: &Path) -> Result<Vec<(String, Vec<u8>)>, String> {
    let err = |e: std::io::Error| format!("{}: {e}", path.display());
    let file = File::open(path).map_err(err)?;
    let mut magic = [0u8; 2];
    let mut probe = BufReader::new(file);
    let is_gz = match probe.read_exact(&mut magic) {
        Ok(()) => magic == [0x1f, 0x8b],
        Err(_) => false,
    };
    let file = File::open(path).map_err(err)?;
    let mut reader: Box<dyn Read> = if is_gz {
        Box::new(flate2::read::MultiGzDecoder::new(file))
    } else {
        Box::new(file)
    };

    let mut records: Vec<(String, Vec<u8>)> = Vec::new();
    let mut chunk = vec![0u8; 1 << 20];
    // Bytes of the final, incomplete line of the previous chunk.
    let mut carry: Vec<u8> = Vec::new();
    loop {
        let n = reader.read(&mut chunk).map_err(err)?;
        if n == 0 {
            break;
        }
        let mut rest = &chunk[..n];
        // `position` is a scalar byte compare; `memchr` vectorises it. 3.09 GB of hg38 is
        // 60.6 M lines, and the scan alone was 1.76 s of the parse against 0.64 s here.
        while let Some(nl) = memchr::memchr(b'\n', rest) {
            let (line, after) = rest.split_at(nl);
            if carry.is_empty() {
                push_line(line, &mut records);
            } else {
                carry.extend_from_slice(line);
                push_line(&carry, &mut records);
                carry.clear();
            }
            rest = &after[1..];
        }
        carry.extend_from_slice(rest);
    }
    if !carry.is_empty() {
        push_line(&carry, &mut records);
    }
    Ok(records)
}

/// One FASTA line: a header starts a record, anything else appends bases with
/// whitespace stripped (`\r` included, so CRLF files behave).
fn push_line(line: &[u8], records: &mut Vec<(String, Vec<u8>)>) {
    if let Some((b'>', header)) = line.split_first() {
        let name = header
            .split(|b| b.is_ascii_whitespace())
            .find(|t| !t.is_empty())
            .unwrap_or(b"");
        records.push((String::from_utf8_lossy(name).into_owned(), Vec::new()));
    } else if let Some((_, seq)) = records.last_mut() {
        // A sequence line holds nothing but sequence once the newline is split off, so the
        // overwhelmingly common case is a straight memcpy. The filtering path stays for the
        // line that really does carry interior whitespace. It matters because `Filter`'s
        // `size_hint` is `(0, Some(n))`, so `extend` cannot reserve and pushes one byte at a
        // time with a capacity check each — 3.03 billion of them for hg38. Measured on the
        // real assembly: parse 6.07 s -> 3.51 s here, 2.75 s with the memchr scan
        // below, records byte-identical.
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.iter().any(|b| b.is_ascii_whitespace()) {
            seq.extend(line.iter().copied().filter(|b| !b.is_ascii_whitespace()));
        } else {
            seq.extend_from_slice(line);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rev_comp_table_matches_kegalign() {
        assert_eq!(REV_COMP.len(), 128);
        for (a, b) in [(b'A', b'T'), (b'C', b'G'), (b'G', b'C'), (b'T', b'A')] {
            assert_eq!(REV_COMP[a as usize], b);
        }
        assert_eq!(
            REV_COMP[SEP as usize], SEP,
            "separators must survive revcomp"
        );
        assert_eq!(REV_COMP[b'a' as usize], b't', "soft-masking is preserved");
        assert_eq!(REV_COMP[b'N' as usize], b'N');
    }

    #[test]
    fn interval_split_matches_main_cpp() {
        // 376676-byte block, seed 19 -> single interval ending at block_len-19.
        assert_eq!(intervals(376676, 19, 10_000_000), vec![(0, 376657)]);
        // Exact multiples must not emit a trailing empty interval.
        assert_eq!(intervals(1019, 19, 500), vec![(0, 500), (500, 1000)]);
    }

    #[test]
    fn chr_lookup_is_upper_bound_minus_one() {
        let chrs = vec![
            Chr {
                name: "a".into(),
                start: 0,
                len: 10,
            },
            Chr {
                name: "b".into(),
                start: 11,
                len: 10,
            },
        ];
        assert_eq!(chr_at(&chrs, 0), 0);
        assert_eq!(chr_at(&chrs, 10), 0); // the separator belongs to the left chr
        assert_eq!(chr_at(&chrs, 11), 1);
        assert_eq!(chr_at(&chrs, 20), 1);
    }

    /// The invariant: FASTA, FASTA.gz and 2bit must load to the
    /// identical SequenceSet (record order, names, case, Ns). The `.2bit` is
    /// hand-encoded against the UCSC spec, so the reader is tested against the
    /// real file format rather than the crate's own writer.
    #[test]
    fn three_formats_load_to_the_same_sequence_set() {
        use flate2::Compression;
        use flate2::write::GzEncoder;
        use std::io::Write;

        let records = vec![
            ("chrA".to_string(), b"ACGTacgtNNAACG".to_vec()),
            ("chrB".to_string(), b"ttttGCttttN".to_vec()),
        ];
        // Expected: soft mask -> lowercase, hard mask -> N, order preserved.
        let expected_buf: Vec<u8> = b"ACGTacgtNNAACG&ttttGCttttN&".to_vec();

        let dir = std::env::temp_dir().join(format!("hspz-fmt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let mut fa = String::new();
        for (n, seq) in &records {
            fa.push('>');
            fa.push_str(n);
            fa.push('\n');
            fa.push_str(&String::from_utf8_lossy(seq));
            fa.push('\n');
        }
        let fa_path = dir.join("x.fa");
        std::fs::write(&fa_path, &fa).unwrap();

        let gz_path = dir.join("x.fa.gz");
        {
            let mut enc = GzEncoder::new(
                std::fs::File::create(&gz_path).unwrap(),
                Compression::default(),
            );
            enc.write_all(fa.as_bytes()).unwrap();
            enc.finish().unwrap();
        }

        let tb_path = dir.join("x.2bit");
        write_2bit(&tb_path, &records);

        let g = Genome::load(&fa_path, "", 10_000_000).unwrap();
        let gz = Genome::load(&gz_path, "", 10_000_000).unwrap();
        let tb = Genome::load(&tb_path, "", 10_000_000).unwrap();
        assert_eq!(g.format, Format::Fasta);
        assert_eq!(gz.format, Format::FastaGz);
        assert_eq!(tb.format, Format::TwoBit);
        for (label, other) in [("gz", &gz), ("2bit", &tb)] {
            assert_eq!(g.buf, expected_buf, "{label}: plain FASTA buf");
            assert_eq!(other.buf, expected_buf, "{label}: buf differs from FASTA");
            assert_eq!(other.block_len, g.block_len, "{label}: block_len");
            assert_eq!(
                other.digest(),
                g.digest(),
                "{label}: SequenceSet digest differs"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Minimal 2bit writer for the tiny fixture: magic, version, count,
    /// reserved, length-prefixed index, then per-sequence records of
    /// name/len/N-blocks/mask-blocks/packed 2-bit bases. Matches the layout
    /// `twobit`'s reader implements (its index entries are length-prefixed,
    /// not null-terminated — verified against the crate's own test asset).
    fn write_2bit(path: &Path, records: &[(String, Vec<u8>)]) {
        use std::io::Write;

        let block_runs = |seq: &[u8], pred: fn(u8) -> bool| -> Vec<(u32, u32)> {
            let mut out = Vec::new();
            let mut i = 0;
            while i < seq.len() {
                if pred(seq[i]) {
                    let s = i;
                    while i < seq.len() && pred(seq[i]) {
                        i += 1;
                    }
                    out.push((s as u32, (i - s) as u32));
                } else {
                    i += 1;
                }
            }
            out
        };
        let n_blocks = |seq: &[u8]| block_runs(seq, |b| b == b'N' || b == b'n');
        let mask_blocks = |seq: &[u8]| block_runs(seq, |b| b.is_ascii_lowercase());

        // One record: seqLen u32 + nBlockCount u32 + (start,size) pairs +
        // mBlockCount u32 + (start,size) pairs + reserved u32 + bases.
        let rec_len = |_name: &str, seq: &[u8]| -> u32 {
            (4 + 4
                + 8 * n_blocks(seq).len()
                + 4
                + 8 * mask_blocks(seq).len()
                + 4
                + seq.len().div_ceil(4)) as u32
        };
        // The index follows the 16-byte header; each entry is name + NUL + offset.
        let mut offset: u32 = 16
            + records
                .iter()
                .map(|(n, _)| (n.len() + 1 + 4) as u32)
                .sum::<u32>();

        let mut file = Vec::new();
        fn u32le(v: u32, file: &mut Vec<u8>) {
            file.extend_from_slice(&v.to_le_bytes());
        }
        u32le(0x1A41_2743, &mut file);
        u32le(0, &mut file); // version
        u32le(records.len() as u32, &mut file);
        u32le(0, &mut file); // reserved
        for (name, seq) in records {
            // twobit's index entries are [u8 name-length][name][u32 offset] —
            // length-prefixed, no NUL (verified against the crate's own asset).
            file.push(name.len() as u8);
            file.extend_from_slice(name.as_bytes());
            u32le(offset, &mut file);
            offset += rec_len(name, seq);
        }
        for (name, seq) in records {
            // Real UCSC records start with the sequence length — no dnaSize, no
            // name inside the record (verified against hgdownload sacCer3.2bit).
            let _ = name;
            u32le(seq.len() as u32, &mut file);

            let nb = n_blocks(seq);
            u32le(nb.len() as u32, &mut file);
            // The spec stores all starts, then all sizes — two arrays, not
            // interleaved pairs (UCSC twoBit.c reads them this way).
            for (s, _) in &nb {
                u32le(*s, &mut file);
            }
            for (_, l) in &nb {
                u32le(*l, &mut file);
            }
            let mb = mask_blocks(seq);
            u32le(mb.len() as u32, &mut file);
            for (s, _) in &mb {
                u32le(*s, &mut file);
            }
            for (_, l) in &mb {
                u32le(*l, &mut file);
            }
            // 4-byte reserved field between the mask blocks and the DNA
            // (the reader consumes it and offsets the packed bases past it).
            u32le(0, &mut file);
            // 2 bits per base, first base in the top bits; N positions use the
            // T slot (0) and are overwritten from the N blocks on read.
            let mut byte = 0u8;
            let mut shift = 6;
            for &b in seq {
                let v = match b.to_ascii_uppercase() {
                    b'A' => 2,
                    b'C' => 1,
                    b'G' => 3,
                    _ => 0,
                };
                byte |= v << shift;
                if shift == 0 {
                    file.push(byte);
                    byte = 0;
                    shift = 6;
                } else {
                    shift -= 2;
                }
            }
            if shift != 6 {
                file.push(byte);
            }
        }
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&file).unwrap();
    }
}
