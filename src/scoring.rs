// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung
// Distributed under the terms of the GNU General Public License, Version 3.0.

// Author : Alejandro Gonzales-Irribarren
// Github : alejandrogzi
// Email  : alejandrxgzi@gmail.com

//! The substitution matrix, including LASTZ-format `--scoring` files.
//!
//! KegAlign runs the full LASTZ score-set reader but keeps only the 4x4 ACGT
//! block (`common/scoring.c`); every other setting in the file — gap penalties,
//! thresholds, seed, step — is parsed and discarded. The rows for soft-masked,
//! ambiguous and separator characters are always rebuilt from `main.cpp`'s own
//! hard-coded `bad_score`/`fill_score`, *not* from the file. This module
//! reproduces exactly that.

use crate::sequence::{E_NT, L_NT, N_NT, NUC, NUC2, X_NT};
use std::path::Path;

/// `main.cpp`'s defaults for everything outside the ACGT block. These are
/// deliberately independent of the scoring file's own settings.
const BAD: i32 = -1000;
const FILL: i32 = -100;

/// The blastz default, `main.cpp: tmp_sub_mat`.
const DEFAULT_ACGT: [[i32; 4]; 4] = [
    [91, -114, -31, -123],
    [-114, 100, -125, -31],
    [-31, -125, 100, -114],
    [-123, -31, -114, 91],
];

/// Builds the `NUC x NUC` matrix `find_hsps` indexes with the device alphabet.
///
/// # Example
/// ```
/// let m = build_sub_mat("", 910, None).unwrap();
/// assert_eq!(m.len(), 64); // NUC x NUC
/// assert_eq!(m[0], 91);    // the blastz A-A match score
/// ```
pub fn build_sub_mat(
    ambiguous: &str,
    xdrop: i32,
    scoring: Option<&Path>,
) -> Result<Vec<i32>, String> {
    let acgt = match scoring {
        Some(path) => load_scoring_matrix(path)?,
        None => DEFAULT_ACGT,
    };

    let fields: Vec<&str> = ambiguous.split(',').collect();
    let field = fields[0];
    let (reward, penalty) = if fields.len() == 3 {
        (
            fields[1]
                .parse::<i32>()
                .map_err(|e| format!("--ambiguous reward: {e}"))?,
            -fields[2]
                .parse::<i32>()
                .map_err(|e| format!("--ambiguous penalty: {e}"))?,
        )
    } else if ambiguous == "n" || ambiguous == "iupac" {
        (0, 0)
    } else {
        (-100, -100)
    };

    let mut m = vec![0i32; NUC2];
    let (l, n, x, e) = (L_NT as usize, N_NT as usize, X_NT as usize, E_NT as usize);
    for i in 0..l {
        for j in 0..l {
            m[i * NUC + j] = acgt[i][j];
        }
    }

    // Soft-masked bases never score.
    for i in 0..l {
        m[i * NUC + l] = BAD;
        m[l * NUC + i] = BAD;
    }
    m[l * NUC + l] = BAD;

    let n_score = if field == "n" || field == "iupac" {
        (penalty, reward)
    } else {
        (BAD, BAD)
    };
    for i in 0..n {
        m[i * NUC + n] = n_score.0;
        m[n * NUC + i] = n_score.0;
    }
    m[n * NUC + n] = n_score.1;

    if field == "iupac" {
        for i in 0..x {
            m[i * NUC + x] = penalty;
            m[x * NUC + i] = penalty;
        }
        m[x * NUC + x] = reward;
    } else {
        for i in 0..l {
            m[i * NUC + x] = FILL;
            m[x * NUC + i] = FILL;
        }
        for i in l..x {
            m[i * NUC + x] = BAD;
            m[x * NUC + i] = BAD;
        }
        m[x * NUC + x] = FILL;
    }

    // The separator must break any extension that reaches it.
    for i in 0..e {
        m[i * NUC + e] = -10 * xdrop;
        m[e * NUC + i] = -10 * xdrop;
    }
    m[e * NUC + e] = -10 * xdrop;

    Ok(m)
}

/// Reads a LASTZ score-set file and returns its ACGT block.
///
/// Supported, because it is what KegAlign's reader accepts and consumes:
/// `#` comments, `key = value` settings before the matrix, a column-label
/// header, and rows that either carry a label or (blastz style) omit it. Every
/// setting other than `bad_score` and `fill_score` is accepted and ignored,
/// matching KegAlign. Quantum alphabets and floating-point scores are rejected
/// rather than guessed at.
fn load_scoring_matrix(path: &Path) -> Result<[[i32; 4]; 4], String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let at = |n: usize| format!("{}: line {n}", path.display());

    let mut lines = text
        .lines()
        .enumerate()
        .map(|(i, l)| (i + 1, l.split('#').next().unwrap_or("").trim()))
        .filter(|(_, l)| !l.is_empty());

    // Phase 1: `key = value` settings, up to the first line without an '='.
    let mut bad_score = BAD;
    let mut fill_score = FILL;
    let mut bad_row = None;
    let mut bad_col = None;
    let mut header = None;
    for (n, line) in lines.by_ref() {
        let Some((key, value)) = line.split_once('=') else {
            header = Some((n, line));
            break;
        };
        let (key, value) = (key.trim(), value.trim());
        match key {
            // `<score>`, `<char>:<score>`, or `<row>:<col>:<score>`.
            "bad_score" => {
                let parts: Vec<&str> = value.split(':').collect();
                bad_score = parse_score(parts[parts.len() - 1], &at(n))?;
                match parts.len() {
                    1 => {}
                    2 => {
                        bad_row = parse_char_code(parts[0], &at(n))?;
                        bad_col = bad_row;
                    }
                    3 => {
                        bad_row = parse_char_code(parts[0], &at(n))?;
                        bad_col = parse_char_code(parts[1], &at(n))?;
                    }
                    _ => return Err(format!("malformed bad_score ({})", at(n))),
                }
            }
            "fill_score" => fill_score = parse_score(value, &at(n))?,
            // Everything else is real LASTZ configuration that KegAlign drops
            // on the floor; accept it so stock scoring files load.
            _ => {}
        }
    }

    // Phase 2: the matrix, starting at the column-label header.
    let (hn, header) = header.ok_or_else(|| format!("{}: no scoring matrix", path.display()))?;
    let col_chars: Vec<u8> = header
        .split_whitespace()
        .map(|f| {
            parse_char_code(f, &at(hn))?.ok_or_else(|| format!("empty column label ({})", at(hn)))
        })
        .collect::<Result<_, _>>()?;
    let num_cols = col_chars.len();
    if num_cols == 0 {
        return Err(format!("{}: no score columns", path.display()));
    }

    let mut sub = [[None; 128]; 128];
    let mut num_fields = None;
    let mut implicit_row = 0usize;
    for (n, line) in lines {
        let fields: Vec<&str> = line.split_whitespace().collect();
        match num_fields {
            None => {
                if fields.len() != num_cols && fields.len() != num_cols + 1 {
                    return Err(format!(
                        "wrong number of score columns: got {}, expected {num_cols} or {} ({})",
                        fields.len(),
                        num_cols + 1,
                        at(n)
                    ));
                }
                num_fields = Some(fields.len());
            }
            Some(k) if k != fields.len() => {
                return Err(format!("inconsistent number of score columns ({})", at(n)));
            }
            _ => {}
        }

        // Rows without a label take the next column label, blastz style.
        let (row_char, values) = if fields.len() == num_cols {
            if implicit_row >= num_cols {
                return Err(format!("too many score rows ({})", at(n)));
            }
            implicit_row += 1;
            (col_chars[implicit_row - 1], &fields[..])
        } else {
            let c = parse_char_code(fields[0], &at(n))?
                .ok_or_else(|| format!("invalid row character code ({})", at(n)))?;
            (c, &fields[1..])
        };

        for (ix, v) in values.iter().enumerate() {
            sub[row_char as usize][col_chars[ix] as usize] = Some(parse_score(v, &at(n))?);
        }
    }
    if num_fields.is_none() {
        return Err(format!("{}: contains no score rows", path.display()));
    }

    // Undefined pairs take fill_score; a nucleotide named as the bad row or
    // column overrides it. (LASTZ normally aims bad_score at X/N, which never
    // reaches this block.)
    let mut out = [[0i32; 4]; 4];
    for (x, r) in b"ACGT".iter().enumerate() {
        for (y, c) in b"ACGT".iter().enumerate() {
            out[x][y] = if bad_row == Some(*r) || bad_col == Some(*c) {
                bad_score
            } else {
                sub[*r as usize][*c as usize].unwrap_or(fill_score)
            };
        }
    }
    Ok(out)
}

/// A single character or a two-digit hex code (`00` is not a valid label).
fn parse_char_code(field: &str, at: &str) -> Result<Option<u8>, String> {
    if field.is_empty() {
        return Ok(None);
    }
    if field.contains('~') {
        return Err(format!(
            "quantum alphabet column `{field}` is not supported ({at}); KegAlign only uses the \
             ACGT block"
        ));
    }
    let bytes = field.as_bytes();
    let code = match bytes.len() {
        1 => bytes[0],
        2 => u8::from_str_radix(field, 16)
            .map_err(|_| format!("invalid character code `{field}` ({at})"))?,
        _ => return Err(format!("invalid character code `{field}` ({at})")),
    };
    if code == 0 || code >= 128 {
        return Err(format!("character code `{field}` out of range ({at})"));
    }
    Ok(Some(code))
}

fn parse_score(field: &str, at: &str) -> Result<i32, String> {
    field
        .parse::<i32>()
        .map_err(|_| format!("invalid score `{field}` ({at}); scores must be integers"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write(name: &str, body: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("hspz-score-{name}-{}", std::process::id()));
        std::fs::File::create(&p)
            .unwrap()
            .write_all(body.as_bytes())
            .unwrap();
        p
    }

    #[test]
    fn default_sub_mat_matches_kegalign() {
        let m = build_sub_mat("", 910, None).unwrap();
        assert_eq!(m.len(), NUC2);
        assert_eq!(m[0], 91, "A/A");
        assert_eq!(m[3], -123, "A/T");
        assert_eq!(m[2 * NUC + 2], 100, "G/G");
        assert_eq!(m[4 * NUC + 4], -1000, "lowercase never scores");
        assert_eq!(m[5 * NUC + 5], -1000, "N/N without --ambiguous");
        assert_eq!(m[6 * NUC + 6], -100, "X/X fill score");
        assert_eq!(m[7 * NUC + 7], -9100, "separator = -10 * xdrop");
        assert_eq!(m[7], -9100, "any base against a separator");
    }

    #[test]
    fn ambiguous_n_opens_up_the_n_row() {
        let m = build_sub_mat("n", 910, None).unwrap();
        assert_eq!(m[5 * NUC + 5], 0, "N/N rewarded");
        assert_eq!(m[5], 0, "A/N penalty");
        assert_eq!(m[6 * NUC + 6], -100, "X still fill without iupac");
    }

    #[test]
    fn ambiguous_triplet_sets_reward_and_penalty() {
        let m = build_sub_mat("iupac,5,7", 910, None).unwrap();
        assert_eq!(m[5 * NUC + 5], 5);
        assert_eq!(m[5], -7);
        assert_eq!(m[6 * NUC + 6], 5, "iupac extends the treatment to X");
    }

    #[test]
    fn reads_the_documented_lastz_example() {
        let p = write(
            "example",
            "# This matches the default scoring set for blastz\n\
             bad_score          = X:-1000  # used for sub['X'][*] and sub[*]['X']\n\
             fill_score         = -100     # used when sub[*][*] not defined\n\
             gap_open_penalty   =   30\n\
             gap_extend_penalty =  400\n\
             \n\
                  A     C     G     T\n\
             A   91  -114   -31  -123\n\
             C -114   100  -125   -31\n\
             G  -31  -125   100  -114\n\
             T -123   -31  -114    91\n",
        );
        assert_eq!(load_scoring_matrix(&p).unwrap(), DEFAULT_ACGT);
        // ... and the full matrix is then identical to the built-in default.
        assert_eq!(
            build_sub_mat("", 910, Some(&p)).unwrap(),
            build_sub_mat("", 910, None).unwrap()
        );
    }

    #[test]
    fn reads_blastz_style_rows_without_labels() {
        let p = write(
            "blastz",
            "     A     C     G     T\n\
              91  -114   -31  -123\n\
             -114   100  -125   -31\n\
              -31  -125   100  -114\n\
             -123   -31  -114    91\n",
        );
        assert_eq!(load_scoring_matrix(&p).unwrap(), DEFAULT_ACGT);
    }

    #[test]
    fn undefined_pairs_take_fill_score() {
        // Only the A row/column is given; everything else falls back.
        let p = write("partial", "fill_score = -7\n   A\nA  5\n");
        let m = load_scoring_matrix(&p).unwrap();
        assert_eq!(m[0][0], 5);
        assert_eq!(m[1][1], -7, "C/C was never defined");
        assert_eq!(m[3][0], -7);
    }

    #[test]
    fn rejects_malformed_matrices_clearly() {
        let bad = [
            (
                "ragged",
                "  A  C\nA 1 2\nC 3\n",
                "inconsistent number of score columns",
            ),
            (
                "wide",
                "  A  C\nA 1 2 3 4\n",
                "wrong number of score columns",
            ),
            ("float", "  A\nA 1.5\n", "scores must be integers"),
            ("quantum", "  A~T\nA 1\n", "quantum alphabet"),
            ("empty", "fill_score = -1\n", "no scoring matrix"),
            ("norows", "  A  C\n", "contains no score rows"),
        ];
        for (name, body, want) in bad {
            let err = load_scoring_matrix(&write(name, body)).unwrap_err();
            assert!(err.contains(want), "{name}: expected {want:?}, got {err:?}");
        }
    }

    #[test]
    fn bad_score_aimed_at_a_nucleotide_overrides_the_block() {
        let p = write("badnt", "bad_score = A:-500\n  A  C\nA 1 2\nC 3 4\n");
        let m = load_scoring_matrix(&p).unwrap();
        assert_eq!(m[0][0], -500);
        assert_eq!(m[0][1], -500, "whole A row");
        assert_eq!(m[1][0], -500, "whole A column");
        assert_eq!(m[1][1], 4, "C/C untouched");
    }
}
