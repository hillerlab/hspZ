// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung
// Distributed under the terms of the GNU General Public License, Version 3.0.

// Author : Alejandro Gonzales-Irribarren
// Github : alejandrogzi
// Email  : alejandrxgzi@gmail.com

//! Where output goes: a directory of files, or one `.tar.gz`.
//!
//! Deliberately two implementations and no framework. `-Z` archives directly
//! from the formatted bytes — it never writes files and re-reads them to tar
//! them up, which is the whole point.

use crate::Fallible;
use flate2::Compression;
use flate2::write::GzEncoder;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

pub trait OutputSink {
    /// Emits one logical output file. `name` is the bare file name.
    fn write_entry(&mut self, name: &str, bytes: &[u8]) -> Fallible<()>;
    /// Flushes and closes. Must be called; `-Z` needs the tar trailer.
    fn finish(self: Box<Self>) -> Fallible<()>;
    /// Total bytes handed to the sink, for the output report.
    fn bytes_in(&self) -> u64;
    /// Bytes actually landed on disk, which differs from `bytes_in` once
    /// compression is involved.
    fn bytes_out(&self) -> Fallible<u64>;
}

pub struct DirectorySink {
    dir: PathBuf,
    bytes: u64,
}

impl DirectorySink {
    pub fn new(dir: &Path) -> Fallible<Self> {
        std::fs::create_dir_all(dir)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            bytes: 0,
        })
    }
}

impl OutputSink for DirectorySink {
    fn write_entry(&mut self, name: &str, bytes: &[u8]) -> Fallible<()> {
        let mut f = BufWriter::new(File::create(self.dir.join(name))?);
        f.write_all(bytes)?;
        f.flush()?;
        self.bytes += bytes.len() as u64;
        Ok(())
    }

    fn finish(self: Box<Self>) -> Fallible<()> {
        Ok(())
    }

    fn bytes_in(&self) -> u64 {
        self.bytes
    }

    fn bytes_out(&self) -> Fallible<u64> {
        Ok(self.bytes)
    }
}

pub struct TarGzSink {
    tar: tar::Builder<GzEncoder<BufWriter<File>>>,
    path: PathBuf,
    bytes: u64,
}

impl TarGzSink {
    /// Default compression, not maximum — this is an output sink, not an
    /// archival tool.
    pub fn new(path: &Path) -> Fallible<Self> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let gz = GzEncoder::new(BufWriter::new(File::create(path)?), Compression::default());
        Ok(Self {
            tar: tar::Builder::new(gz),
            path: path.to_path_buf(),
            bytes: 0,
        })
    }
}

impl OutputSink for TarGzSink {
    fn write_entry(&mut self, name: &str, bytes: &[u8]) -> Fallible<()> {
        let mut header = tar::Header::new_gnu();
        header.set_size(bytes.len() as u64);
        header.set_mode(0o644);
        // Fixed mtime so two runs of the same input produce the same archive;
        // a wall-clock timestamp would make `-Z` output non-reproducible.
        header.set_mtime(0);
        header.set_cksum();
        self.tar.append_data(&mut header, name, bytes)?;
        self.bytes += bytes.len() as u64;
        Ok(())
    }

    fn finish(self: Box<Self>) -> Fallible<()> {
        // `into_inner` writes the tar trailer, then the gzip trailer.
        self.tar.into_inner()?.finish()?.flush()?;
        Ok(())
    }

    fn bytes_in(&self) -> u64 {
        self.bytes
    }

    fn bytes_out(&self) -> Fallible<u64> {
        Ok(std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// After extraction, `-Z` output must be byte-identical to the directory
    /// output, and two archives of the same input must be identical.
    #[test]
    fn tar_round_trip_matches_directory_and_is_reproducible() {
        use std::io::Read;

        let dir = std::env::temp_dir().join(format!("hspz-sink-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // One large-ish entry forces multi-gzip-block output.
        let big = vec![b'x'; 200_000];
        let files: Vec<(String, Vec<u8>)> = vec![
            ("tmp1.block0.r0.plus.segments".into(), big),
            (
                "tmp2.block0.r0.minus.segments".into(),
                b"acgtacgt\n".to_vec(),
            ),
            ("empty.segments".into(), Vec::new()),
        ];

        // Directory output.
        let mut ds = DirectorySink::new(&dir).unwrap();
        for (n, b) in &files {
            ds.write_entry(n, b).unwrap();
        }
        Box::new(ds).finish().unwrap();

        // Tar output, twice, for reproducibility.
        let mut archives = Vec::new();
        for f in ["o1.tar.gz", "o2.tar.gz"] {
            let path = dir.join(f);
            let mut ts = TarGzSink::new(&path).unwrap();
            for (n, b) in &files {
                ts.write_entry(n, b).unwrap();
            }
            Box::new(ts).finish().unwrap();
            archives.push(path);
        }
        assert_eq!(
            std::fs::read(&archives[0]).unwrap(),
            std::fs::read(&archives[1]).unwrap(),
            "archives must be reproducible (fixed mtime)"
        );

        // Extract the first and diff against the directory output.
        let raw = std::fs::read(&archives[0]).unwrap();
        let gz = flate2::read::GzDecoder::new(&raw[..]);
        let mut ar = tar::Archive::new(gz);
        let mut seen = std::collections::BTreeMap::new();
        for entry in ar.entries().unwrap() {
            let mut e = entry.unwrap();
            let name = e.path().unwrap().to_string_lossy().into_owned();
            assert!(
                !name.starts_with("/") && !name.starts_with("./"),
                "tar entries must be bare filenames, got {name}"
            );
            let mut data = Vec::new();
            e.read_to_end(&mut data).unwrap();
            seen.insert(name, data);
        }
        for (n, b) in &files {
            assert_eq!(seen.get(n), Some(b), "extracted {n} differs from input");
            assert_eq!(
                std::fs::read(dir.join(n)).unwrap(),
                *b,
                "directory {n} differs from input"
            );
        }
        assert_eq!(seen.len(), files.len(), "tar entry set differs");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
