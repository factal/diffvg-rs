//! PCG32 random number helpers for CubeCL kernels.

use cubecl::prelude::*;
use crate::gpu::constants::*;

/// Multiply two u32 values and return the full 64-bit product as [lo, hi].
#[cube]
pub(super) fn mul32_full(a: u32, b: u32) -> Line<u32> {
    let mask = u32::new(0xffff);
    let a0 = a & mask;
    let a1 = a >> 16;
    let b0 = b & mask;
    let b1 = b >> 16;

    let p0 = a0 * b0;
    let p1 = a0 * b1;
    let p2 = a1 * b0;
    let p3 = a1 * b1;

    let mid_sum = (p1 & mask) + (p2 & mask);
    let mid_low = mid_sum & mask;
    let carry = mid_sum >> 16;
    let mid_high = (p1 >> 16) + (p2 >> 16) + carry;

    let p0_low = p0 & mask;
    let p0_high = p0 >> 16;
    let high_sum = p0_high + mid_low;
    let lo = p0_low | ((high_sum & mask) << 16);
    let carry2 = high_sum >> 16;
    let hi = p3 + mid_high + carry2;

    let mut out = Line::empty(2usize);
    out[0] = lo;
    out[1] = hi;
    out
}

/// Multiply two 64-bit values (lo, hi) and return the low 64 bits as [lo, hi].
#[cube]
pub(super) fn mul64_low(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> Line<u32> {
    let prod0 = mul32_full(a_lo, b_lo);
    let sum_low = a_lo * b_hi + a_hi * b_lo;
    let hi = prod0[1] + sum_low;
    let mut out = Line::empty(2usize);
    out[0] = prod0[0];
    out[1] = hi;
    out
}

/// Add two 64-bit values (lo, hi) and return the sum as [lo, hi].
#[cube]
pub(super) fn add64(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> Line<u32> {
    let mask = u32::new(0xffff);
    let a_low = a_lo & mask;
    let a_high = a_lo >> 16;
    let b_low = b_lo & mask;
    let b_high = b_lo >> 16;
    let sum_low = a_low + b_low;
    let carry = sum_low >> 16;
    let sum_high = a_high + b_high + carry;
    let lo = (sum_low & mask) | ((sum_high & mask) << 16);
    let carry2 = sum_high >> 16;
    let hi = a_hi + b_hi + carry2;
    let mut out = Line::empty(2usize);
    out[0] = lo;
    out[1] = hi;
    out
}

/// Logical right shift of a 64-bit value (lo, hi) by shift bits.
#[cube]
pub(super) fn shr64(lo: u32, hi: u32, shift: u32) -> Line<u32> {
    let mut out = Line::empty(2usize);
    let mask = u32::new(0xffff_ffffu32 as i64);
    let s = shift & u32::new(31);
    let is_high = (shift >> 5) & u32::new(1);
    let mask_high = mask * is_high;
    let mask_low = mask * (is_high ^ u32::new(1));

    let hi_shift = hi >> s;
    let mut hi_part = u32::new(0);
    if s != u32::new(0) {
        let lshift = ((s ^ u32::new(31)) + u32::new(1)) & u32::new(31);
        hi_part = hi << lshift;
    }

    let lo_low = (lo >> s) | hi_part;
    let lo_high = hi_shift;

    out[0] = (lo_low & mask_low) | (lo_high & mask_high);
    out[1] = hi_shift & mask_low;
    out
}

/// Return the low 32 bits of a shifted 64-bit value.
#[cube]
pub(super) fn shr64_to_u32(lo: u32, hi: u32, shift: u32) -> u32 {
    let shifted = shr64(lo, hi, shift);
    shifted[0]
}

/// Advance PCG32 state and return [rand_u32, state_lo, state_hi, 0].
#[cube]
pub(super) fn pcg32_next(state_lo: u32, state_hi: u32, inc_lo: u32, inc_hi: u32) -> Line<u32> {
    let old_lo = state_lo;
    let old_hi = state_hi;

    let mult = mul64_low(old_lo, old_hi, PCG_MULT_LO, PCG_MULT_HI);
    let inc_lo = inc_lo | u32::new(1);
    let new_state = add64(mult[0], mult[1], inc_lo, inc_hi);

    let shifted = shr64(old_lo, old_hi, 18);
    let xor_lo = shifted[0] ^ old_lo;
    let xor_hi = shifted[1] ^ old_hi;
    let xorshifted = shr64_to_u32(xor_lo, xor_hi, 27);
    let rot = shr64_to_u32(old_lo, old_hi, 59) & 31;
    let rot_l = ((rot ^ u32::new(31)) + u32::new(1)) & u32::new(31);
    let out_val = (xorshifted >> rot) | (xorshifted << rot_l);

    let mut out = Line::empty(4usize);
    out[0] = out_val;
    out[1] = new_state[0];
    out[2] = new_state[1];
    out[3] = u32::new(0);
    out
}

/// Initialize PCG32 from an index and seed, returning [state_lo, state_hi, inc_lo, inc_hi].
#[cube]
pub(super) fn pcg32_init(idx: u32, seed: u32) -> Line<u32> {
    let base = idx + u32::new(1);
    let inc_lo = (base << 1) | u32::new(1);
    let inc_hi = base >> 31;
    let mut state_lo = u32::new(0);
    let mut state_hi = u32::new(0);

    let step0 = pcg32_next(state_lo, state_hi, inc_lo, inc_hi);
    state_lo = step0[1];
    state_hi = step0[2];

    let seed_add = add64(PCG_INIT_LO, PCG_INIT_HI, seed, u32::new(0));
    let seeded = add64(state_lo, state_hi, seed_add[0], seed_add[1]);
    state_lo = seeded[0];
    state_hi = seeded[1];

    let step1 = pcg32_next(state_lo, state_hi, inc_lo, inc_hi);
    state_lo = step1[1];
    state_hi = step1[2];

    let mut out = Line::empty(4usize);
    out[0] = state_lo;
    out[1] = state_hi;
    out[2] = inc_lo;
    out[3] = inc_hi;
    out
}

/// Convert a u32 to a float in [0, 1) using a 23-bit mantissa.
#[cube]
pub(super) fn pcg32_f32(x: u32) -> f32 {
    let mantissa = x >> 9;
    f32::cast_from(mantissa) * f32::new(1.0 / 8_388_608.0)
}

/// Initialize stratified jitter for a sample, returning [rx, ry].
#[cube]
pub(super) fn jitter_xy(
    x: u32,
    y: u32,
    sx: u32,
    sy: u32,
    width: u32,
    samples_x: u32,
    samples_y: u32,
    seed: u32,
    jitter: u32,
) -> Line<f32> {
    let half = f32::new(0.5);
    let mut out = Line::empty(2usize);
    out[0] = half;
    out[1] = half;
    if jitter != u32::new(0) {
        let canonical_idx = ((y * width + x) * samples_y + sy) * samples_x + sx;
        let rng = pcg32_init(canonical_idx, seed);
        let mut state_lo = rng[0];
        let mut state_hi = rng[1];
        let inc_lo = rng[2];
        let inc_hi = rng[3];
        let step0 = pcg32_next(state_lo, state_hi, inc_lo, inc_hi);
        state_lo = step0[1];
        state_hi = step0[2];
        out[0] = pcg32_f32(step0[0]);
        let step1 = pcg32_next(state_lo, state_hi, inc_lo, inc_hi);
        out[1] = pcg32_f32(step1[0]);
    }
    out
}
