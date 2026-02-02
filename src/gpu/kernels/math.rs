use cubecl::prelude::*;

#[cube]
pub(super) fn smoothstep_unit(d: f32) -> f32 {
    let t = clamp01((d + f32::new(1.0)) * f32::new(0.5));
    t * t * (f32::new(3.0) - f32::new(2.0) * t)
}

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

#[cube]
pub(super) fn min_f32(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

#[cube]
pub(super) fn max_f32(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

#[cube]
pub(super) fn abs_f32(a: f32) -> f32 {
    let zero = f32::new(0.0);
    if a < zero { -a } else { a }
}

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

#[cube]
pub(super) fn vec2_dot(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    ax * bx + ay * by
}

#[cube]
pub(super) fn vec2_length(ax: f32, ay: f32) -> f32 {
    let l_sq = vec2_dot(ax, ay, ax, ay);
    l_sq.sqrt()
}

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

#[cube]
pub(super) fn xform_pt_affine(m00: f32, m01: f32, m02: f32, m10: f32, m11: f32, m12: f32, px: f32, py: f32) -> Line<f32> {
    let mut out = Line::empty(2usize);
    out[0] = m00 * px + m01 * py + m02;
    out[1] = m10 * px + m11 * py + m12;
    out
}

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
