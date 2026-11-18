// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung
// Distributed under the terms of the GNU General Public License, Version 3.0.

// Author : Alejandro Gonzales-Irribarren
// Github : alejandrogzi
// Email  : alejandrxgzi@gmail.com

//! The lane-local upper bound in front of the score gate (`gate-sieve`).
//!
//! The score gate scores one hit per warp. Almost every hit fails it, so each
//! lane first bounds the score of *its own* hit from 2-bit packed sequence, and
//! the warp runs the exact gate only over the hits whose bound reaches the
//! threshold or could not be certified. A rejected hit has an exact score below
//! the threshold, so the flags the gate writes do not change.
//!
//! # The bound
//!
//! One side of a hit is the walk `k = 1, 2, ...` away from the hit with scores
//! `s_k`, prefix sums `P_k` and running maximum `M_k = max(0, P_1..P_k)`; the
//! gate stops it at the first `k` with `M_k - P_k > xdrop` and returns the
//! maximum before that base, or the maximum at the sequence end.
//!
//! The bound replaces `s_k` by a class score `u_k >= s_k` that depends only on
//! the 2-bit codes: match, transition or transversion ([`Classes`]). Then
//!
//! * the running maximum under `u` is no smaller than under `s`, and the
//!   drawdown `M - P` is no larger, so the X-drop under `u` fires no earlier;
//! * a query base that is not an uppercase `ACGT` is *terminal*: its score is
//!   below `-xdrop`, so the exact walk stops at or before it;
//! * over blocks of 8 bases, with `B_j` the `u` prefix sum at the end of block
//!   `j` and `m_j` its matches, no prefix inside block `j` exceeds
//!   `B_{j-1} + um * m_j` (matches first; needs `uti, utv <= 0 <= um`), and
//!   `max_i B_i - B_j > xdrop` certifies that the `u` walk has dropped by the
//!   end of block `j`.
//!
//! So with `J` the first block that holds a terminal base or certifies a drop,
//! the exact side is at most `best = max(0, max_{j <= J} B_{j-1} + um * m_j)`.
//! A lane rejects its hit iff both sides have such a `J` inside their window
//! (64 bases to the right, 96 to the left) and `best_right + best_left < t_min`. A side
//! without one *abstains* and the hit goes to the exact gate.
//!
//! Reference bases past the end of the sequence read as padding. They only
//! lengthen the `u` walk, which cannot lower `best`; the query padding is
//! terminal, like the query end is for the exact walk.
//!
//! [`classes`] decides when all of this holds for a scoring matrix, and
//! switches the sieve off when it does not.

#[cfg(any(test, feature = "gate-sieve"))]
use crate::sequence::{A_NT, C_NT, E_NT, L_NT, N_NT, NUC, T_NT, X_NT};

/// 32-base groups of padding in front of and behind every packed plane, so a
/// lane can read its six reference and seven query groups without a bounds test.
pub const PADG: u32 = 4;
/// Bases the bound reads to the right of a hit (the hit position included).
#[cfg(feature = "gate-sieve")]
pub const RIGHT: u32 = 64;
/// Bases the bound reads to the left of a hit.
pub const LEFT: u32 = 96;
/// 2-bit code every reference symbol outside `ACGT` is packed as. Any code is
/// sound ([`classes`] takes the class maxima over the pairs it creates); `C`
/// meets a mammalian query as a match least often.
#[cfg(any(test, feature = "gate-sieve"))]
pub const REF_FILL: u32 = C_NT as u32;
/// Largest `xdrop` and class score the bound's 32-bit arithmetic is used for.
/// 96 bases of scores this size stay below 2^27.
#[cfg(any(test, feature = "gate-sieve"))]
const MAX_ABS: i32 = 1 << 20;

/// Packed groups for a sequence of `len` bases.
#[cfg(any(test, feature = "gate-sieve"))]
pub fn groups(len: usize) -> usize {
    len / 32 + 1 + 2 * PADG as usize
}

/// Class scores of the bound: an upper bound on the score of every base pair
/// whose 2-bit codes are equal (`um`), a transition apart (`uti`) or a
/// transversion apart (`utv`).
#[cfg(any(test, feature = "gate-sieve"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Classes {
    pub um: i32,
    pub uti: i32,
    pub utv: i32,
}

#[cfg(any(test, feature = "gate-sieve"))]
impl Classes {
    /// The constants the kernel takes: `um`, `um - utv`, `uti - utv` and
    /// `8 * utv`, so a block's sum is `a*m + b*t + bias` for `m` matches and `t`
    /// transitions among its 8 bases.
    pub fn kernel_args(&self) -> (i32, i32, i32, i32) {
        (
            self.um,
            self.um - self.utv,
            self.uti - self.utv,
            8 * self.utv,
        )
    }
}

/// The class scores for a resolved substitution matrix, or why the sieve must
/// stay off.
///
/// The reference is packed without a mask plane: every symbol outside `ACGT`
/// (soft-masked, `N`, other IUPAC codes, the record separator) takes the code
/// [`REF_FILL`], so each class maximum runs over those pairs too. A query base
/// outside `ACGT` ends the bound's walk, which is right only if it ends the
/// exact walk: every score against it must be below `-xdrop`.
///
/// Soft-masked bases, `N` and separators are assumed present in both
/// sequences. Other IUPAC codes are rare and change the answer (they score
/// -100 by default, above the worst transversion and far from terminal), so the
/// caller says whether each sequence holds one.
#[cfg(any(test, feature = "gate-sieve"))]
pub fn classes(
    sub_mat: &[i32],
    xdrop: i32,
    ref_has_x: bool,
    qry_has_x: bool,
) -> Result<Classes, String> {
    if !(0..=MAX_ABS).contains(&xdrop) {
        return Err(format!("xdrop {xdrop} is outside 0..={MAX_ABS}"));
    }
    let score = |r: u8, q: u8| sub_mat[r as usize * NUC + q as usize];
    let present = |sym: u8, has_x: bool| sym != X_NT || has_x;
    let name = |sym: u8| match sym {
        L_NT => "soft-masked",
        N_NT => "N",
        X_NT => "IUPAC",
        _ => "separator",
    };

    // Every score the exact walk can add before it stops must leave its 32-bit
    // sums exact inside the window.
    for r in A_NT..=E_NT {
        for q in A_NT..=E_NT {
            if present(r, ref_has_x)
                && present(q, qry_has_x)
                && !(-(1 << 30)..=MAX_ABS).contains(&score(r, q))
            {
                return Err(format!(
                    "score {} is outside the 32-bit safe range",
                    score(r, q)
                ));
            }
        }
    }

    // A query base outside ACGT must stop the exact walk.
    for q in L_NT..=E_NT {
        if !present(q, qry_has_x) {
            continue;
        }
        for r in A_NT..=E_NT {
            if present(r, ref_has_x) && score(r, q) >= -xdrop {
                return Err(format!(
                    "a {} query base scores {} and does not end an extension at xdrop {xdrop}",
                    name(q),
                    score(r, q)
                ));
            }
        }
    }

    let (mut um, mut uti, mut utv) = (i32::MIN, i32::MIN, i32::MIN);
    for r in A_NT..=E_NT {
        if !present(r, ref_has_x) {
            continue;
        }
        let code = if r <= T_NT { r as u32 } else { REF_FILL };
        for q in A_NT..=T_NT {
            let s = score(r, q);
            match code ^ q as u32 {
                0 => um = um.max(s),
                2 => uti = uti.max(s),
                _ => utv = utv.max(s),
            }
        }
    }
    if um < 0 || uti > 0 || utv > 0 {
        return Err(format!(
            "class scores match {um}, transition {uti}, transversion {utv} are not ordered around 0"
        ));
    }
    // A random pair is a match 1/4, a transition 1/4 and a transversion 1/2 of
    // the time. Without a negative drift the bound certifies no drop.
    if i64::from(um) + i64::from(uti) + 2 * i64::from(utv) >= 0 {
        return Err(format!(
            "class scores match {um}, transition {uti}, transversion {utv} have no negative drift"
        ));
    }
    Ok(Classes { um, uti, utv })
}

/// Host mirror of the `pack_planes` kernel: the two code planes of every
/// 32-base group (low word = high code bit, high word = low code bit) and the
/// plane of bases that are not an uppercase `ACGT`. Symbols outside `ACGT` take
/// the code `fill`; everything outside the sequence is code 0 and bad.
#[cfg(any(test, feature = "gate-sieve"))]
pub fn pack(enc: &[u8], fill: u32) -> (Vec<u64>, Vec<u32>) {
    let n = groups(enc.len());
    let mut pk = vec![0u64; n];
    let mut bad = vec![u32::MAX; n];
    for (i, &sym) in enc.iter().enumerate() {
        let (g, b) = (i / 32 + PADG as usize, i % 32);
        let code = if sym <= T_NT {
            bad[g] &= !(1 << b);
            sym as u64
        } else {
            fill as u64
        };
        pk[g] |= ((code >> 1) & 1) << b | (code & 1) << (32 + b);
    }
    (pk, bad)
}

/// The low word of `hi:lo >> s`, for `s` in `0..32`.
#[inline(always)]
pub fn funnel(lo: u32, hi: u32, s: u32) -> u32 {
    (((lo as u64) | ((hi as u64) << 32)) >> s) as u32
}

/// Mismatch, transition and valid planes of one reference group against the
/// query group that the hit's diagonal lays over it.
#[inline(always)]
fn planes(r: u64, qp: u64, qn: u64, bp: u32, bn: u32, qs: u32) -> (u32, u32, u32) {
    let qh = funnel(qp as u32, qn as u32, qs);
    let ql = funnel((qp >> 32) as u32, (qn >> 32) as u32, qs);
    let xh = (r as u32) ^ qh;
    let xl = ((r >> 32) as u32) ^ ql;
    (xh | xl, xh & !xl, !funnel(bp, bn, qs))
}

/// One block of 8 bases at bit `sh` of a stream word.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn block(
    st: (i32, i32, i32, bool),
    mv: u32,
    tvv: u32,
    tv: u32,
    sh: u32,
    um: i32,
    a: i32,
    b: i32,
    bias: i32,
    xdrop: i32,
) -> (i32, i32, i32, bool) {
    let (sum, peak, best, done) = st;
    let m = ((mv >> sh) & 0xff).count_ones() as i32;
    let t = ((tvv >> sh) & 0xff).count_ones() as i32;
    let terminal = (tv >> sh) & 0xff != 0xff;
    let cand = sum.wrapping_add(um.wrapping_mul(m));
    let best = if !done && cand > best { cand } else { best };
    let next = sum
        .wrapping_add(a.wrapping_mul(m))
        .wrapping_add(b.wrapping_mul(t))
        .wrapping_add(bias);
    let peak = if next > peak { next } else { peak };
    let done = done || terminal || peak.wrapping_sub(next) > xdrop;
    (next, peak, best, done)
}

/// One stream word (32 bases in extension order from bit 0) of one side.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn word(
    st: (i32, i32, i32, bool),
    alive: bool,
    mm: u32,
    tr: u32,
    vd: u32,
    um: i32,
    a: i32,
    b: i32,
    bias: i32,
    xdrop: i32,
) -> ((i32, i32, i32, bool), bool) {
    // The valid bases before this word's first terminal base. Nothing is valid
    // once an earlier word held one.
    let tv = if alive { vd & !vd.wrapping_add(1) } else { 0 };
    let mv = !mm & tv;
    let tvv = tr & tv;
    let st = block(st, mv, tvv, tv, 0, um, a, b, bias, xdrop);
    let st = block(st, mv, tvv, tv, 8, um, a, b, bias, xdrop);
    let st = block(st, mv, tvv, tv, 16, um, a, b, bias, xdrop);
    let st = block(st, mv, tvv, tv, 24, um, a, b, bias, xdrop);
    (st, alive && tv == u32::MAX)
}

/// The bound from the six reference groups around the hit, already loaded.
///
/// `o` is the hit's bit inside reference group `r3`; `qry_pk`/`qry_bad` point
/// at the query group aligned with `r0` and `qs` is the shift between them.
/// Returns whether each side is certified and its bound: `(right certified,
/// left certified, right best, left best)`.
///
/// # Safety
/// `qry_pk` and `qry_bad` must be readable for seven elements.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn bound_words(
    r0: u64,
    r1: u64,
    r2: u64,
    r3: u64,
    r4: u64,
    r5: u64,
    qry_pk: *const u64,
    qry_bad: *const u32,
    o: u32,
    qs: u32,
    um: i32,
    a: i32,
    b: i32,
    bias: i32,
    xdrop: i32,
) -> (bool, bool, i32, i32) {
    let (q0, q1, q2, q3, q4, q5, q6) = unsafe {
        (
            *qry_pk,
            *qry_pk.add(1),
            *qry_pk.add(2),
            *qry_pk.add(3),
            *qry_pk.add(4),
            *qry_pk.add(5),
            *qry_pk.add(6),
        )
    };
    let (b0, b1, b2, b3, b4, b5, b6) = unsafe {
        (
            *qry_bad,
            *qry_bad.add(1),
            *qry_bad.add(2),
            *qry_bad.add(3),
            *qry_bad.add(4),
            *qry_bad.add(5),
            *qry_bad.add(6),
        )
    };
    let (m0, t0, v0) = planes(r0, q0, q1, b0, b1, qs);
    let (m1, t1, v1) = planes(r1, q1, q2, b1, b2, qs);
    let (m2, t2, v2) = planes(r2, q2, q3, b2, b3, qs);
    let (m3, t3, v3) = planes(r3, q3, q4, b3, b4, qs);
    let (m4, t4, v4) = planes(r4, q4, q5, b4, b5, qs);
    let (m5, t5, v5) = planes(r5, q5, q6, b5, b6, qs);

    // Right: from bit `o` of group 3 upward.
    let st = (0i32, 0i32, 0i32, false);
    let (st, alive) = word(
        st,
        true,
        funnel(m3, m4, o),
        funnel(t3, t4, o),
        funnel(v3, v4, o),
        um,
        a,
        b,
        bias,
        xdrop,
    );
    let (right, _) = word(
        st,
        alive,
        funnel(m4, m5, o),
        funnel(t4, t5, o),
        funnel(v4, v5, o),
        um,
        a,
        b,
        bias,
        xdrop,
    );

    // Left: from the bit below `o` of group 3 downward, reversed so that the
    // extension order ascends from bit 0.
    let st = (0i32, 0i32, 0i32, false);
    let (st, alive) = word(
        st,
        true,
        funnel(m2, m3, o).reverse_bits(),
        funnel(t2, t3, o).reverse_bits(),
        funnel(v2, v3, o).reverse_bits(),
        um,
        a,
        b,
        bias,
        xdrop,
    );
    let (st, alive) = word(
        st,
        alive,
        funnel(m1, m2, o).reverse_bits(),
        funnel(t1, t2, o).reverse_bits(),
        funnel(v1, v2, o).reverse_bits(),
        um,
        a,
        b,
        bias,
        xdrop,
    );
    let (left, _) = word(
        st,
        alive,
        funnel(m0, m1, o).reverse_bits(),
        funnel(t0, t1, o).reverse_bits(),
        funnel(v0, v1, o).reverse_bits(),
        um,
        a,
        b,
        bias,
        xdrop,
    );
    (right.3, left.3, right.2, left.2)
}

/// Where a hit's packed groups are: the first of its six reference groups, the
/// first of its seven query groups, the hit's bit inside reference group 3 and
/// the shift between the two sequences.
#[inline(always)]
pub fn locate(ref_loc: u32, query_loc: u32) -> (u32, u32, u32, u32) {
    let rb = ref_loc.wrapping_add(32 * PADG);
    let o = rb & 31;
    let qb = query_loc.wrapping_add(32 * PADG - LEFT).wrapping_sub(o);
    ((rb >> 5) - 3, qb >> 5, o, qb & 31)
}

/// The bound for the hit at `(ref_loc, query_loc)`, with every lane loading its
/// own six reference groups.
///
/// # Safety
/// The three planes must be packed as [`pack`] does, for sequences that hold
/// `ref_loc` and `query_loc` (a position equal to the length is allowed).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn bound_at(
    ref_pk: *const u64,
    qry_pk: *const u64,
    qry_bad: *const u32,
    ref_loc: u32,
    query_loc: u32,
    um: i32,
    a: i32,
    b: i32,
    bias: i32,
    xdrop: i32,
) -> (bool, bool, i32, i32) {
    let (rg, qg, o, qs) = locate(ref_loc, query_loc);
    unsafe {
        let r = ref_pk.add(rg as usize);
        bound_words(
            *r,
            *r.add(1),
            *r.add(2),
            *r.add(3),
            *r.add(4),
            *r.add(5),
            qry_pk.add(qg as usize),
            qry_bad.add(qg as usize),
            o,
            qs,
            um,
            a,
            b,
            bias,
            xdrop,
        )
    }
}

/// Whether the bound sends a hit to the exact gate.
#[inline(always)]
pub fn passes(bound: (bool, bool, i32, i32), t_min: i32) -> bool {
    let (right_done, left_done, right, left) = bound;
    !right_done || !left_done || right.wrapping_add(left) >= t_min
}

/// Host model of one lane: whether the hit is a candidate for the exact gate.
#[cfg(feature = "gate-sieve")]
#[allow(clippy::too_many_arguments)]
pub fn candidate(
    ref_pk: &[u64],
    qry_pk: &[u64],
    qry_bad: &[u32],
    ref_loc: u32,
    query_loc: u32,
    classes: &Classes,
    xdrop: i32,
    t_min: i32,
) -> bool {
    passes(
        lane_bound(ref_pk, qry_pk, qry_bad, ref_loc, query_loc, classes, xdrop),
        t_min,
    )
}

/// Host model of one lane's bound, bounds-checked.
#[cfg(any(test, feature = "gate-sieve"))]
pub fn lane_bound(
    ref_pk: &[u64],
    qry_pk: &[u64],
    qry_bad: &[u32],
    ref_loc: u32,
    query_loc: u32,
    classes: &Classes,
    xdrop: i32,
) -> (bool, bool, i32, i32) {
    let (rg, qg, _, _) = locate(ref_loc, query_loc);
    assert!(
        rg as usize + 6 <= ref_pk.len(),
        "reference groups out of range"
    );
    assert!(
        qg as usize + 7 <= qry_pk.len() && qry_pk.len() == qry_bad.len(),
        "query groups out of range"
    );
    let (um, a, b, bias) = classes.kernel_args();
    // SAFETY: the six reference and seven query groups were just checked.
    unsafe {
        bound_at(
            ref_pk.as_ptr(),
            qry_pk.as_ptr(),
            qry_bad.as_ptr(),
            ref_loc,
            query_loc,
            um,
            a,
            b,
            bias,
            xdrop,
        )
    }
}

/// The exact gate's score of one side and whether an X-drop ended it (rather
/// than a sequence end). `step(k)` is the score of the `k`-th base away from
/// the hit, `n` the bases the side has.
#[cfg(any(test, feature = "gate-sieve"))]
fn exact_side(n: usize, xdrop: i32, step: impl Fn(usize) -> i32) -> (i32, bool) {
    let (mut sum, mut peak) = (0i32, 0i32);
    for k in 0..n {
        sum = sum.wrapping_add(step(k));
        peak = peak.max(sum);
        if peak.wrapping_sub(sum) > xdrop {
            return (peak, true);
        }
    }
    (peak, false)
}

/// Host model of the exact score gate: `(right, left)` side scores of the hit
/// at `(ref_loc, query_loc)` over device-alphabet sequences. The gate keeps the
/// hit iff their sum reaches `keep_threshold(hspthresh)`.
#[cfg(any(test, feature = "gate-sieve"))]
pub fn exact_sides(
    ref_enc: &[u8],
    qry_enc: &[u8],
    ref_loc: u32,
    query_loc: u32,
    sub_mat: &[i32],
    xdrop: i32,
) -> (i32, i32) {
    let (r, q) = (ref_loc as usize, query_loc as usize);
    let score = |rp: usize, qp: usize| sub_mat[ref_enc[rp] as usize * NUC + qry_enc[qp] as usize];
    let n_right = ref_enc
        .len()
        .saturating_sub(r)
        .min(qry_enc.len().saturating_sub(q));
    let (right, _) = exact_side(n_right, xdrop, |k| score(r + k, q + k));
    let (left, _) = exact_side(r.min(q), xdrop, |k| score(r - 1 - k, q - 1 - k));
    (right, left)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scoring::build_sub_mat;
    use crate::sequence::encode;

    fn default_classes() -> Classes {
        classes(&build_sub_mat("", 910, None).unwrap(), 910, false, false).unwrap()
    }

    #[test]
    fn default_scoring_gives_the_blastz_class_maxima() {
        assert_eq!(
            default_classes(),
            Classes {
                um: 100,
                uti: -31,
                utv: -114
            }
        );
        assert_eq!(default_classes().kernel_args(), (100, 214, 83, -912));
    }

    /// An IUPAC code other than `N` scores -100 against a base: above the worst
    /// transversion, and nowhere near terminal.
    #[test]
    fn an_iupac_code_loosens_the_reference_and_switches_the_query_off() {
        let m = build_sub_mat("", 910, None).unwrap();
        assert_eq!(
            classes(&m, 910, true, false).unwrap(),
            Classes {
                um: 100,
                uti: -31,
                utv: -100
            }
        );
        let why = classes(&m, 910, false, true).unwrap_err();
        assert!(why.contains("IUPAC query base scores -100"), "{why}");
        // Below 100 it does end an extension.
        assert!(classes(&build_sub_mat("", 99, None).unwrap(), 99, true, true).is_ok());
    }

    /// A soft-masked or `N` base scores -1000: terminal only below that xdrop.
    #[test]
    fn the_sieve_is_off_once_a_masked_base_no_longer_ends_an_extension() {
        assert!(classes(&build_sub_mat("", 999, None).unwrap(), 999, false, false).is_ok());
        for xdrop in [1000, 1200] {
            let why = classes(
                &build_sub_mat("", xdrop, None).unwrap(),
                xdrop,
                false,
                false,
            )
            .unwrap_err();
            assert!(why.contains("soft-masked query base scores -1000"), "{why}");
        }
        assert!(classes(&build_sub_mat("", 910, None).unwrap(), -1, false, false).is_err());
    }

    /// `--ambiguous n` and `iupac` score `N` at 0 or better, so it neither ends
    /// an extension nor stays under a class maximum.
    #[test]
    fn the_sieve_is_off_when_n_is_scored() {
        for ambiguous in ["n", "iupac", "n,10,20", "iupac,200,10"] {
            let m = build_sub_mat(ambiguous, 910, None).unwrap();
            assert!(classes(&m, 910, false, false).is_err(), "{ambiguous}");
        }
    }

    /// The class maxima come from the matrix in use, not from blastz.
    #[test]
    fn class_maxima_follow_a_custom_matrix() {
        let mut m = build_sub_mat("", 910, None).unwrap();
        for (r, q) in [(0usize, 2usize), (2, 0), (1, 3), (3, 1)] {
            m[r * NUC + q] = -20;
        }
        assert_eq!(
            classes(&m, 910, false, false).unwrap(),
            Classes {
                um: 100,
                uti: -20,
                utv: -114
            }
        );
        // A rewarded transition, and a matrix that does not drift down.
        m[2] = 5;
        assert!(classes(&m, 910, false, false).is_err());
        let mut flat = build_sub_mat("", 910, None).unwrap();
        for r in 0..4 {
            for q in 0..4 {
                if r != q {
                    flat[r * NUC + q] = -10;
                }
            }
        }
        assert!(
            classes(&flat, 910, false, false)
                .unwrap_err()
                .contains("no negative drift")
        );
    }

    #[test]
    fn pack_lays_out_codes_bad_bits_and_padding() {
        let enc = encode(b"ACGTacgtNR&");
        let (pk, bad) = pack(&enc, REF_FILL);
        assert_eq!(pk.len(), groups(enc.len()));
        let g = PADG as usize;
        // A C G T, then seven symbols outside ACGT with the fill code C (01).
        let hi = 0b0000_0000_1100u64;
        let lo = 0b0111_1111_1010u64;
        assert_eq!(pk[g], hi | lo << 32);
        assert_eq!(bad[g], !0b1111);
        for i in (0..g).chain(g + 1..pk.len()) {
            assert_eq!((pk[i], bad[i]), (0, u32::MAX), "group {i}");
        }
    }

    #[test]
    fn funnel_reads_across_the_word_boundary() {
        assert_eq!(funnel(0x1234_5678, 0x9abc_def0, 0), 0x1234_5678);
        assert_eq!(funnel(0x1234_5678, 0x9abc_def0, 4), 0x0123_4567);
        assert_eq!(funnel(0x8000_0000, 0x0000_0001, 31), 0x0000_0003);
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            (self.next() >> 11) % n
        }
        fn chance(&mut self, per_mille: u64) -> bool {
            self.below(1000) < per_mille
        }
    }

    /// Soft-masked runs, `N` runs, a separator and, when asked, IUPAC codes.
    fn decorate(seq: &mut [u8], rng: &mut Rng, iupac: bool) {
        let n = seq.len() as u64;
        if n == 0 {
            return;
        }
        for _ in 0..rng.below(4) {
            let (at, len) = (rng.below(n) as usize, 1 + rng.below(40) as usize);
            for c in seq.iter_mut().skip(at).take(len) {
                *c = c.to_ascii_lowercase();
            }
        }
        for _ in 0..rng.below(3) {
            let (at, len) = (rng.below(n) as usize, 1 + rng.below(12) as usize);
            for c in seq.iter_mut().skip(at).take(len) {
                *c = b'N';
            }
        }
        if rng.chance(300) {
            seq[rng.below(n) as usize] = b'&';
        }
        if iupac {
            for _ in 0..1 + rng.below(3) {
                seq[rng.below(n) as usize] = b"RYKMSW"[rng.below(6) as usize];
            }
        }
    }

    /// The matrices and xdrops the property runs over, each with whether the
    /// reference and the query may hold IUPAC codes.
    fn regimes() -> Vec<(Vec<i32>, i32, bool, bool)> {
        let mut out = Vec::new();
        for xdrop in [910, 600, 340, 99, 30] {
            let m = build_sub_mat("", xdrop, None).unwrap();
            out.push((m.clone(), xdrop, false, false));
            out.push((m.clone(), xdrop, true, false));
            if xdrop < 100 {
                out.push((m, xdrop, true, true));
            }
        }
        // Milder transitions and a harsher match, as a custom matrix would give.
        let mut m = build_sub_mat("", 910, None).unwrap();
        for (r, q) in [(0usize, 2usize), (2, 0), (1, 3), (3, 1)] {
            m[r * NUC + q] = -5;
        }
        m[0] = 60;
        out.push((m, 910, false, false));
        out
    }

    #[derive(Default, Debug)]
    struct Tally {
        hits: u64,
        /// Hits with both sides certified.
        certified: u64,
        /// Hits the bound rejects at a threshold of 3000.
        rejected: u64,
        /// A certified side below the exact side, or a kept hit rejected.
        violations: u64,
    }

    /// Random hits over random sequence pairs that share a diverged stretch.
    /// The property: a certified side of the bound is never below the exact
    /// side, and a hit the exact gate keeps is always a candidate. `tamper`
    /// replaces the class scores, to show that the run notices a wrong bound.
    fn run_property(cases: u64, seed: u64, tamper: Option<Classes>) -> Tally {
        let mut rng = Rng(seed);
        let regimes = regimes();
        let mut tally = Tally::default();
        for case in 0..cases {
            let (sub, xdrop, ref_x, qry_x) = &regimes[(case % regimes.len() as u64) as usize];
            let cls = classes(sub, *xdrop, *ref_x, *qry_x).expect("regime keeps the sieve on");
            let cls = tamper.unwrap_or(cls);

            // A reference, and a query that copies a stretch of it at some
            // divergence so that extensions of every length occur.
            let rlen = rng.below(420) as usize;
            let qlen = rng.below(420) as usize;
            let base = |rng: &mut Rng| b"ACGT"[rng.below(4) as usize];
            let mut r: Vec<u8> = (0..rlen).map(|_| base(&mut rng)).collect();
            let mut q: Vec<u8> = (0..qlen).map(|_| base(&mut rng)).collect();
            let divergence = [0, 20, 80, 250, 500][rng.below(5) as usize];
            let shift = rng.below(60) as usize;
            for i in 0..qlen {
                if i + shift < rlen && !rng.chance(divergence) {
                    q[i] = r[i + shift];
                }
            }
            decorate(&mut r, &mut rng, *ref_x);
            decorate(&mut q, &mut rng, *qry_x);
            let (renc, qenc) = (encode(&r), encode(&q));
            let (rpk, _) = pack(&renc, REF_FILL);
            let (qpk, qbad) = pack(&qenc, REF_FILL);

            for _ in 0..24 {
                // On the shared diagonal most of the time, anywhere otherwise;
                // both ends included.
                let ql = rng.below(qlen as u64 + 1) as u32;
                let rl = if rng.chance(800) && ql as usize + shift <= rlen {
                    ql + shift as u32
                } else {
                    rng.below(rlen as u64 + 1) as u32
                };
                let (dr, dl, br, bl) = lane_bound(&rpk, &qpk, &qbad, rl, ql, &cls, *xdrop);
                let (er, el) = exact_sides(&renc, &qenc, rl, ql, sub, *xdrop);
                tally.hits += 1;
                tally.certified += (dr && dl) as u64;
                let mut wrong = (dr && br < er) || (dl && bl < el);
                for t_min in [er + el, 3000, 2400, 1] {
                    let cand = passes((dr, dl, br, bl), t_min);
                    wrong |= er + el >= t_min && !cand;
                    tally.rejected += (t_min == 3000 && !cand) as u64;
                }
                if wrong {
                    tally.violations += 1;
                    if tamper.is_none() {
                        panic!(
                            "bound ({dr}, {dl}, {br}, {bl}) under the exact sides ({er}, {el}) \
                             at ({rl}, {ql}), xdrop {xdrop}, case {case}"
                        );
                    }
                }
            }
        }
        tally
    }

    #[test]
    fn a_certified_bound_is_never_below_the_exact_gate() {
        let tally = run_property(40_000, 0x9e37_79b9_7f4a_7c15, None);
        // The property is vacuous unless most sides certify and most hits are
        // rejected, as they are on real sequence.
        assert!(tally.certified * 10 > tally.hits * 8, "{tally:?}");
        assert!(tally.rejected * 10 > tally.hits * 6, "{tally:?}");
    }

    /// The same run notices class scores that are too low: a match scored as
    /// the A/T diagonal (91, below C/C and G/G), and blastz transversions where
    /// the matrix in use is milder.
    #[test]
    fn the_property_run_catches_a_bound_that_is_too_low() {
        for wrong in [
            Classes {
                um: 91,
                uti: -31,
                utv: -114,
            },
            Classes {
                um: 100,
                uti: -31,
                utv: -125,
            },
            Classes {
                um: 100,
                uti: -123,
                utv: -114,
            },
        ] {
            let tally = run_property(4_000, 7, Some(wrong));
            assert!(tally.violations > 0, "{wrong:?} went unnoticed: {tally:?}");
        }
    }

    /// The long form of the property run: more than 10^7 hits. `cargo oxide
    /// test -- --release --ignored bound_holds_on_ten_million_hits -- --nocapture`.
    #[test]
    #[ignore = "long: 12 million hits"]
    fn bound_holds_on_ten_million_hits() {
        let tally = run_property(500_000, 20_261_118, None);
        eprintln!("{tally:?}");
        assert!(tally.hits >= 10_000_000 && tally.violations == 0);
    }
}
