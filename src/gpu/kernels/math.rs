//! GPU math helpers shared by CubeCL kernels.

use cubecl::prelude::*;

/// Smoothstep on the unit interval for signed distances in [-1, 1].
#[cube]
pub(super) fn smoothstep_unit(d: f32) -> f32 {
    let t = clamp01((d + f32::new(1.0)) * f32::new(0.5));
    t * t * (f32::new(3.0) - f32::new(2.0) * t)
}

/// Clamp a value to the inclusive range [0, 1].
#[cube]
pub(super) fn clamp01(v: f32) -> f32 {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    if v < zero {
        zero
    } else if v > one {
        one
    } else {
        v
    }
}

/// Return the smaller of two f32 values.
#[cube]
pub(super) fn min_f32(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// Return the larger of two f32 values.
#[cube]
pub(super) fn max_f32(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// Absolute value for f32 without relying on std.
#[cube]
pub(super) fn abs_f32(a: f32) -> f32 {
    let zero = f32::new(0.0);
    if a < zero { -a } else { a }
}

/// Signed distance to an axis-aligned rectangle (negative inside).
#[cube]
pub(super) fn rect_signed_distance(
    px: f32,
    py: f32,
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
) -> f32 {
    let zero = f32::new(0.0);
    let dx = max_f32(max_f32(min_x - px, zero), px - max_x);
    let dy = max_f32(max_f32(min_y - py, zero), py - max_y);
    let outside = (dx * dx + dy * dy).sqrt();
    let inside = min_f32(
        min_f32(px - min_x, max_x - px),
        min_f32(py - min_y, max_y - py),
    );
    if outside > zero { outside } else { -inside }
}

/// Clamp a value to the inclusive range [min_v, max_v].
#[cube]
pub(super) fn clamp_f32(v: f32, min_v: f32, max_v: f32) -> f32 {
    if v < min_v {
        min_v
    } else if v > max_v {
        max_v
    } else {
        v
    }
}

/// Dot product of two 2D vectors.
#[cube]
pub(super) fn vec2_dot(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    ax * bx + ay * by
}

/// Length of a 2D vector.
#[cube]
pub(super) fn vec2_length(ax: f32, ay: f32) -> f32 {
    let l_sq = vec2_dot(ax, ay, ax, ay);
    l_sq.sqrt()
}

/// Normalize a 2D vector, returning [x, y] or zeros if length is 0.
#[cube]
pub(super) fn vec2_normalize(ax: f32, ay: f32) -> Line<f32> {
    let mut out = Line::empty(2usize);
    let len = vec2_length(ax, ay);
    if len > f32::new(0.0) {
        out[0] = ax / len;
        out[1] = ay / len;
    } else {
        out[0] = f32::new(0.0);
        out[1] = f32::new(0.0);
    }
    out
}

/// Apply a 2x3 affine transform to a point, returning [x, y].
#[cube]
pub(super) fn xform_pt_affine(m00: f32, m01: f32, m02: f32, m10: f32, m11: f32, m12: f32, px: f32, py: f32) -> Line<f32> {
    let mut out = Line::empty(2usize);
    out[0] = m00 * px + m01 * py + m02;
    out[1] = m10 * px + m11 * py + m12;
    out
}

/// Backprop for vec2 length; returns [d_ax, d_ay] given upstream d_l.
#[cube]
pub(super) fn d_length_vec2(ax: f32, ay: f32, d_l: f32) -> Line<f32> {
    let mut out = Line::empty(2usize);
    let l_sq = ax * ax + ay * ay;
    let l = l_sq.sqrt();
    if l == f32::new(0.0) {
        out[0] = f32::new(0.0);
        out[1] = f32::new(0.0);
    } else {
        let d_l_sq = f32::new(0.5) * d_l / l;
        out[0] = ax * f32::new(2.0) * d_l_sq;
        out[1] = ay * f32::new(2.0) * d_l_sq;
    }
    out
}

/// Backprop for distance between two points; accumulates into d_a and d_b.
#[cube]
pub(super) fn d_distance(ax: f32, ay: f32, bx: f32, by: f32, d_out: f32, d_a: &mut Line<f32>, d_b: &mut Line<f32>) {
    let dx = bx - ax;
    let dy = by - ay;
    let d_len = d_length_vec2(dx, dy, d_out);
    d_a[0] -= d_len[0];
    d_a[1] -= d_len[1];
    d_b[0] += d_len[0];
    d_b[1] += d_len[1];
}

/// Backprop for vec2 normalization, returning [d_ax, d_ay].
#[cube]
pub(super) fn d_normalize_vec2(ax: f32, ay: f32, d_nx: f32, d_ny: f32) -> Line<f32> {
    let mut out = Line::empty(2usize);
    let len = vec2_length(ax, ay);
    if len == f32::new(0.0) {
        out[0] = f32::new(0.0);
        out[1] = f32::new(0.0);
    } else {
        let nx = ax / len;
        let ny = ay / len;
        out[0] = d_nx / len;
        out[1] = d_ny / len;
        let d_l = -(d_nx * nx + d_ny * ny) / len;
        let d_len = d_length_vec2(ax, ay, d_l);
        out[0] += d_len[0];
        out[1] += d_len[1];
    }
    out
}

/// Backprop for smoothstep_unit with respect to input d.
#[cube]
pub(super) fn d_smoothstep_unit(d: f32, d_ret: f32) -> f32 {
    let mut out = f32::new(0.0);
    if d >= f32::new(-1.0) && d <= f32::new(1.0) {
        let t = (d + f32::new(1.0)) * f32::new(0.5);
        let d_t = d_ret * (f32::new(6.0) * t - f32::new(6.0) * t * t);
        out = d_t * f32::new(0.5);
    }
    out
}
