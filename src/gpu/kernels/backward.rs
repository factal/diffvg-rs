use cubecl::prelude::*;
use crate::gpu::constants::*;
use super::{math::*, rng::*, sampling::*, forward::accumulate_group_shapes, distance::compute_distance_group};

const MAX_FRAGMENTS: usize = 256;
const MAX_PREFILTER_FRAGMENTS: usize = 64;
const MAT3_STRIDE: usize = 4;
const MAT3_SIZE: usize = 12;
const AFFINE_SIZE: usize = 8;

/// Backprop for affine point transform; accumulates grads for matrix and point.
#[cube]
pub(super) fn d_xform_pt_affine(
    m00: f32,
    m01: f32,
    m02: f32,
    m10: f32,
    m11: f32,
    m12: f32,
    px: f32,
    py: f32,
    d_out_x: f32,
    d_out_y: f32,
    d_m: &mut Line<f32>,
    d_pt: &mut Line<f32>,
) {
    d_m[0] += d_out_x * px;
    d_m[1] += d_out_x * py;
    d_m[2] += d_out_x;
    d_m[3] += d_out_y * px;
    d_m[4] += d_out_y * py;
    d_m[5] += d_out_y;
    d_m[6] += f32::new(0.0);
    d_m[7] += f32::new(0.0);
    d_pt[0] += d_out_x * m00 + d_out_y * m10;
    d_pt[1] += d_out_x * m01 + d_out_y * m11;
}

/// Build a packed 3x3 matrix line (stride 4) from affine coefficients.
#[cube]
pub(super) fn mat3_from_affine(m00: f32, m01: f32, m02: f32, m10: f32, m11: f32, m12: f32) -> Line<f32> {
    let mut out = Line::empty(MAT3_SIZE);
    out[0] = m00;
    out[1] = m01;
    out[2] = m02;
    out[3] = f32::new(0.0);
    out[4] = m10;
    out[5] = m11;
    out[6] = m12;
    out[7] = f32::new(0.0);
    out[8] = f32::new(0.0);
    out[9] = f32::new(0.0);
    out[10] = f32::new(1.0);
    out[11] = f32::new(0.0);
    out
}

/// Transpose a packed 3x3 matrix line.
#[cube]
pub(super) fn mat3_transpose(m: Line<f32>) -> Line<f32> {
    let mut out = Line::empty(MAT3_SIZE);
    out[0] = m[0];
    out[1] = m[4];
    out[2] = m[8];
    out[3] = f32::new(0.0);
    out[4] = m[1];
    out[5] = m[5];
    out[6] = m[9];
    out[7] = f32::new(0.0);
    out[8] = m[2];
    out[9] = m[6];
    out[10] = m[10];
    out[11] = f32::new(0.0);
    out
}

/// Multiply two packed 3x3 matrices.
#[cube]
pub(super) fn mat3_mul(a: Line<f32>, b: Line<f32>) -> Line<f32> {
    let mut out = Line::empty(MAT3_SIZE);
    let a00 = a[0];
    let a01 = a[1];
    let a02 = a[2];
    let a10 = a[4];
    let a11 = a[5];
    let a12 = a[6];
    let a20 = a[8];
    let a21 = a[9];
    let a22 = a[10];

    let b00 = b[0];
    let b01 = b[1];
    let b02 = b[2];
    let b10 = b[4];
    let b11 = b[5];
    let b12 = b[6];
    let b20 = b[8];
    let b21 = b[9];
    let b22 = b[10];

    out[0] = a00 * b00 + a01 * b10 + a02 * b20;
    out[1] = a00 * b01 + a01 * b11 + a02 * b21;
    out[2] = a00 * b02 + a01 * b12 + a02 * b22;
    out[3] = f32::new(0.0);

    out[4] = a10 * b00 + a11 * b10 + a12 * b20;
    out[5] = a10 * b01 + a11 * b11 + a12 * b21;
    out[6] = a10 * b02 + a11 * b12 + a12 * b22;
    out[7] = f32::new(0.0);

    out[8] = a20 * b00 + a21 * b10 + a22 * b20;
    out[9] = a20 * b01 + a21 * b11 + a22 * b21;
    out[10] = a20 * b02 + a21 * b12 + a22 * b22;
    out[11] = f32::new(0.0);
    out
}

/// Scale a packed 3x3 matrix by a scalar.
#[cube]
pub(super) fn mat3_scale(m: Line<f32>, s: f32) -> Line<f32> {
    let mut out = Line::empty(MAT3_SIZE);
    out[0] = m[0] * s;
    out[1] = m[1] * s;
    out[2] = m[2] * s;
    out[3] = f32::new(0.0);
    out[4] = m[4] * s;
    out[5] = m[5] * s;
    out[6] = m[6] * s;
    out[7] = f32::new(0.0);
    out[8] = m[8] * s;
    out[9] = m[9] * s;
    out[10] = m[10] * s;
    out[11] = f32::new(0.0);
    out
}

/// Atomically add the 3x3 entries into an output buffer.
#[cube]
pub(super) fn atomic_add_mat3(out: &mut Array<Atomic<f32>>, base: usize, m: Line<f32>) {
    out[base].fetch_add(m[0]);
    out[base + 1].fetch_add(m[1]);
    out[base + 2].fetch_add(m[2]);
    out[base + 3].fetch_add(m[4]);
    out[base + 4].fetch_add(m[5]);
    out[base + 5].fetch_add(m[6]);
    out[base + 6].fetch_add(m[8]);
    out[base + 7].fetch_add(m[9]);
    out[base + 8].fetch_add(m[10]);
}

/// Expand affine gradient (2x3) into a packed 3x3 gradient line.
#[cube]
pub(super) fn affine_grad_to_mat3(affine: Line<f32>) -> Line<f32> {
    let mut out = Line::empty(MAT3_SIZE);
    out[0] = affine[0];
    out[1] = affine[1];
    out[2] = affine[2];
    out[3] = f32::new(0.0);
    out[4] = affine[3];
    out[5] = affine[4];
    out[6] = affine[5];
    out[7] = f32::new(0.0);
    out[8] = f32::new(0.0);
    out[9] = f32::new(0.0);
    out[10] = f32::new(0.0);
    out[11] = f32::new(0.0);
    let pad = affine[6] + affine[7];
    out[11] += pad * f32::new(0.0);
    out
}

/// Atomically add a vec2 into an output buffer.
#[cube]
pub(super) fn atomic_add_vec2(out: &mut Array<Atomic<f32>>, base: usize, x: f32, y: f32) {
    out[base].fetch_add(x);
    out[base + 1].fetch_add(y);
}

/// Atomically add a vec4 into an output buffer.
#[cube]
pub(super) fn atomic_add_vec4(out: &mut Array<Atomic<f32>>, base: usize, x: f32, y: f32, z: f32, w: f32) {
    out[base].fetch_add(x);
    out[base + 1].fetch_add(y);
    out[base + 2].fetch_add(z);
    out[base + 3].fetch_add(w);
}

/// Accumulate per-pixel translation gradient into the translation buffer.
#[cube]
pub(super) fn add_translation(d_translation: &mut Array<Atomic<f32>>, pixel_index: u32, dx: f32, dy: f32) {
    let base = (pixel_index * u32::new(2)) as usize;
    d_translation[base].fetch_add(dx);
    d_translation[base + 1].fetch_add(dy);
}

/// Accumulate background gradient into image or solid background based on flag.
#[cube]
pub(super) fn accumulate_background_grad(
    d_background: &mut Array<Atomic<f32>>,
    d_background_image: &mut Array<Atomic<f32>>,
    has_background_image: u32,
    pixel_index: u32,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
) {
    if has_background_image != u32::new(0) {
        let base = (pixel_index * u32::new(4)) as usize;
        atomic_add_vec4(d_background_image, base, r, g, b, a);
    } else {
        atomic_add_vec4(d_background, 0usize, r, g, b, a);
    }
}

/// Map a point index with clamping or wrapping for closed paths.
#[cube]
pub(super) fn path_point_index(idx: u32, count: u32, is_closed: u32) -> u32 {
    let mut out = u32::new(0);
    if count == u32::new(0) {
        out = u32::new(0);
    } else if is_closed != u32::new(0) {
        out = idx % count;
    } else if idx >= count {
        out = count - u32::new(1);
    } else {
        out = idx;
    }
    out
}

/// Convert a base segment id to an absolute point index using control counts.
#[cube]
pub(super) fn path_point_id(path_num_controls: &Array<u32>, ctrl_offset: u32, base_point_id: u32) -> u32 {
    let mut point_id = u32::new(0);
    let mut i = u32::new(0);
    while i < base_point_id {
        let controls = path_num_controls[(ctrl_offset + i) as usize];
        if controls == u32::new(0) {
            point_id += u32::new(1);
        } else if controls == u32::new(1) {
            point_id += u32::new(2);
        } else if controls == u32::new(2) {
            point_id += u32::new(3);
        }
        i += u32::new(1);
    }
    point_id
}

/// Load an (x, y) point from packed path data.
#[cube]
pub(super) fn load_path_point(
    path_points: &Array<f32>,
    point_offset: u32,
    point_index: u32,
) -> Line<f32> {
    let mut out = Line::empty(2usize);
    let base = ((point_offset + point_index) * u32::new(2)) as usize;
    out[0] = path_points[base];
    out[1] = path_points[base + 1];
    out
}

/// Atomically add a gradient contribution to a path point.
#[cube]
pub(super) fn add_path_point_grad(
    d_path_points: &mut Array<Atomic<f32>>,
    point_offset: u32,
    point_index: u32,
    dx: f32,
    dy: f32,
) {
    let base = ((point_offset + point_index) * u32::new(2)) as usize;
    d_path_points[base].fetch_add(dx);
    d_path_points[base + 1].fetch_add(dy);
}

/// Distance from point to a segment, used for rectangle edges.
#[cube]
pub(super) fn rect_dist_to_seg(px: f32, py: f32, p0x: f32, p0y: f32, p1x: f32, p1y: f32) -> f32 {
    let vx = p1x - p0x;
    let vy = p1y - p0y;
    let t = (px - p0x) * vx + (py - p0y) * vy;
    let denom = vx * vx + vy * vy;
    let mut tt = f32::new(0.0);
    if denom > f32::new(0.0) {
        tt = t / denom;
    }
    if tt < f32::new(0.0) {
        vec2_length(p0x - px, p0y - py)
    } else if tt > f32::new(1.0) {
        vec2_length(p1x - px, p1y - py)
    } else {
        let cx = p0x + vx * tt;
        let cy = p0y + vy * tt;
        vec2_length(cx - px, cy - py)
    }
}

/// Backprop closest-point gradients for a rectangle edge segment.
#[cube]
pub(super) fn rect_update_seg(
    px: f32,
    py: f32,
    p0x: f32,
    p0y: f32,
    p1x: f32,
    p1y: f32,
    d_closest_x: f32,
    d_closest_y: f32,
    d_p0: &mut Line<f32>,
    d_p1: &mut Line<f32>,
    d_pt: &mut Line<f32>,
) {
    let vx = p1x - p0x;
    let vy = p1y - p0y;
    let t = (px - p0x) * vx + (py - p0y) * vy;
    let denom = vx * vx + vy * vy;
    let mut tt = f32::new(0.0);
    if denom > f32::new(0.0) {
        tt = t / denom;
    }
    if tt < f32::new(0.0) {
        d_p0[0] += d_closest_x;
        d_p0[1] += d_closest_y;
    } else if tt > f32::new(1.0) {
        d_p1[0] += d_closest_x;
        d_p1[1] += d_closest_y;
    } else {
        d_p0[0] += d_closest_x * (f32::new(1.0) - tt);
        d_p0[1] += d_closest_y * (f32::new(1.0) - tt);
        d_p1[0] += d_closest_x * tt;
        d_p1[1] += d_closest_y * tt;

        let d_t = d_closest_x * vx + d_closest_y * vy;
        let d_num = if denom > f32::new(0.0) { d_t / denom } else { f32::new(0.0) };
        let d_den = if denom > f32::new(0.0) { d_t * (-tt) / denom } else { f32::new(0.0) };
        d_pt[0] += vx * d_num;
        d_pt[1] += vy * d_num;
        d_p1[0] += (px - p0x) * d_num;
        d_p1[1] += (py - p0y) * d_num;
        d_p0[0] += (p0x - p1x + p0x - px) * d_num;
        d_p0[1] += (p0y - p1y + p0y - py) * d_num;
        d_p1[0] += (p1x - p0x) * (f32::new(2.0) * d_den);
        d_p1[1] += (p1y - p0y) * (f32::new(2.0) * d_den);
        d_p0[0] += (p0x - p1x) * (f32::new(2.0) * d_den);
        d_p0[1] += (p0y - p1y) * (f32::new(2.0) * d_den);
    }
}

/// Backprop closest-point gradients for a rectangle boundary.
#[cube]
pub(super) fn d_closest_point_rect(
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    px: f32,
    py: f32,
    d_closest_x: f32,
    d_closest_y: f32,
    d_min: &mut Line<f32>,
    d_max: &mut Line<f32>,
    d_pt: &mut Line<f32>,
) {
    let lt_x = min_x;
    let lt_y = min_y;
    let rt_x = max_x;
    let rt_y = min_y;
    let lb_x = min_x;
    let lb_y = max_y;
    let rb_x = max_x;
    let rb_y = max_y;

    let mut min_id = u32::new(0);
    let mut min_dist = rect_dist_to_seg(px, py, lt_x, lt_y, lb_x, lb_y);
    let top_dist = rect_dist_to_seg(px, py, lt_x, lt_y, rt_x, rt_y);
    let right_dist = rect_dist_to_seg(px, py, rt_x, rt_y, rb_x, rb_y);
    let bottom_dist = rect_dist_to_seg(px, py, lb_x, lb_y, rb_x, rb_y);
    if top_dist < min_dist {
        min_dist = top_dist;
        min_id = u32::new(1);
    }
    if right_dist < min_dist {
        min_dist = right_dist;
        min_id = u32::new(2);
    }
    if bottom_dist < min_dist {
        min_dist = bottom_dist;
        min_id = u32::new(3);
    }

    let mut d_lt = Line::empty(2usize);
    let mut d_rt = Line::empty(2usize);
    let mut d_lb = Line::empty(2usize);
    let mut d_rb = Line::empty(2usize);

    if min_id == u32::new(0) {
        rect_update_seg(px, py, lt_x, lt_y, lb_x, lb_y, d_closest_x, d_closest_y, &mut d_lt, &mut d_lb, d_pt);
    } else if min_id == u32::new(1) {
        rect_update_seg(px, py, lt_x, lt_y, rt_x, rt_y, d_closest_x, d_closest_y, &mut d_lt, &mut d_rt, d_pt);
    } else if min_id == u32::new(2) {
        rect_update_seg(px, py, rt_x, rt_y, rb_x, rb_y, d_closest_x, d_closest_y, &mut d_rt, &mut d_rb, d_pt);
    } else {
        rect_update_seg(px, py, lb_x, lb_y, rb_x, rb_y, d_closest_x, d_closest_y, &mut d_lb, &mut d_rb, d_pt);
    }

    d_min[0] += d_lt[0];
    d_min[1] += d_lt[1];
    d_max[0] += d_rt[0];
    d_min[1] += d_rt[1];
    d_min[0] += d_lb[0];
    d_max[1] += d_lb[1];
    d_max[0] += d_rb[0];
    d_max[1] += d_rb[1];
}

/// Backprop closest-point gradients for an ellipse, including degenerate radii.
#[cube]
pub(super) fn d_closest_point_ellipse(
    cx: f32,
    cy: f32,
    rx_in: f32,
    ry_in: f32,
    px: f32,
    py: f32,
    d_closest_x: f32,
    d_closest_y: f32,
    d_center: &mut Line<f32>,
    d_radius: &mut Line<f32>,
    d_pt: &mut Line<f32>,
) {
    let rx = rx_in.abs();
    let ry = ry_in.abs();
    let eps = f32::new(1.0e-6);
    let local_x = px - cx;
    let local_y = py - cy;
    let mut done = false;
    if rx < eps && ry < eps {
        d_center[0] += d_closest_x;
        d_center[1] += d_closest_y;
        done = true;
    }
    if !done && rx < eps {
        let hit = if local_y >= -ry && local_y <= ry { f32::new(1.0) } else { f32::new(0.0) };
        d_center[0] += d_closest_x;
        d_center[1] += d_closest_y;
        d_pt[1] += d_closest_y * hit;
        if local_y > ry {
            d_radius[1] += d_closest_y;
        } else if local_y < -ry {
            d_radius[1] -= d_closest_y;
        }
        done = true;
    }
    if !done && ry < eps {
        let hit = if local_x >= -rx && local_x <= rx { f32::new(1.0) } else { f32::new(0.0) };
        d_center[0] += d_closest_x;
        d_center[1] += d_closest_y;
        d_pt[0] += d_closest_x * hit;
        if local_x > rx {
            d_radius[0] += d_closest_x;
        } else if local_x < -rx {
            d_radius[0] -= d_closest_x;
        }
        done = true;
    }

    if !done {
        let sign_x = if local_x < f32::new(0.0) { f32::new(-1.0) } else { f32::new(1.0) };
        let sign_y = if local_y < f32::new(0.0) { f32::new(-1.0) } else { f32::new(1.0) };
        let x = abs_f32(local_x);
        let y = abs_f32(local_y);
        let mut t = (y * rx).atan2(x * ry);
        let mut g_t = f32::new(0.0);
        let mut i = u32::new(0);
        while i < u32::new(20) {
            let s = t.sin();
            let c = t.cos();
            let g = rx * x * s - ry * y * c + (ry * ry - rx * rx) * s * c;
            g_t = rx * x * c + ry * y * s + (ry * ry - rx * rx) * (c * c - s * s);
            if abs_f32(g_t) < f32::new(1.0e-12) {
                break;
            }
            let next = clamp_f32(t - g / g_t, f32::new(0.0), f32::new(1.57079633));
            if abs_f32(next - t) < f32::new(1.0e-6) {
                t = next;
                break;
            }
            t = next;
            i += u32::new(1);
        }
        let s = t.sin();
        let c = t.cos();

        let mut d_t = f32::new(0.0);
        d_radius[0] += d_closest_x * sign_x * c;
        d_t += d_closest_x * sign_x * (-rx * s);
        d_radius[1] += d_closest_y * sign_y * s;
        d_t += d_closest_y * sign_y * (ry * c);

        let g_a = x * s - f32::new(2.0) * rx * s * c;
        let g_b = -y * c + f32::new(2.0) * ry * s * c;
        let g_x = rx * s;
        let g_y = -ry * c;
        if abs_f32(g_t) > f32::new(1.0e-12) {
            let inv = -d_t / g_t;
            d_radius[0] += inv * g_a;
            d_radius[1] += inv * g_b;
            let d_x = inv * g_x;
            let d_y = inv * g_y;
            d_pt[0] += d_x * sign_x;
            d_pt[1] += d_y * sign_y;
            d_center[0] -= d_x * sign_x;
            d_center[1] -= d_y * sign_y;
        }
    }
}

/// Backprop closest-point gradients for a path segment (line, quad, cubic).
#[cube]
pub(super) fn d_closest_point_path(
    path_points: &Array<f32>,
    path_num_controls: &Array<u32>,
    point_offset: u32,
    point_count: u32,
    ctrl_offset: u32,
    is_closed: u32,
    base_point_id: u32,
    t: f32,
    pt_x: f32,
    pt_y: f32,
    d_closest_x: f32,
    d_closest_y: f32,
    d_path_points: &mut Array<Atomic<f32>>,
    d_pt: &mut Line<f32>,
) {
    if point_count != u32::new(0) {
        let point_id = path_point_id(path_num_controls, ctrl_offset, base_point_id);
        let num_controls = path_num_controls[(ctrl_offset + base_point_id) as usize];
        if num_controls == u32::new(0) {
            let i0 = point_id;
            let i1 = path_point_index(point_id + u32::new(1), point_count, is_closed);
            let p0 = load_path_point(path_points, point_offset, i0);
            let p1 = load_path_point(path_points, point_offset, i1);
            let vx = p1[0] - p0[0];
            let vy = p1[1] - p0[1];
            let denom = vx * vx + vy * vy;
            let mut tt = f32::new(0.0);
            if denom > f32::new(0.0) {
                tt = ((pt_x - p0[0]) * vx + (pt_y - p0[1]) * vy) / denom;
            }
            if tt < f32::new(0.0) {
                add_path_point_grad(d_path_points, point_offset, i0, d_closest_x, d_closest_y);
            } else if tt > f32::new(1.0) {
                add_path_point_grad(d_path_points, point_offset, i1, d_closest_x, d_closest_y);
            } else {
                add_path_point_grad(
                    d_path_points,
                    point_offset,
                    i0,
                    d_closest_x * (f32::new(1.0) - tt),
                    d_closest_y * (f32::new(1.0) - tt),
                );
                add_path_point_grad(
                    d_path_points,
                    point_offset,
                    i1,
                    d_closest_x * tt,
                    d_closest_y * tt,
                );
            }
        } else if num_controls == u32::new(1) {
            let i0 = point_id;
            let i1 = point_id + u32::new(1);
            let i2 = path_point_index(point_id + u32::new(2), point_count, is_closed);
            let p0 = load_path_point(path_points, point_offset, i0);
            let p1 = load_path_point(path_points, point_offset, i1);
            let p2 = load_path_point(path_points, point_offset, i2);
            let tt = f32::new(1.0) - t;
            let mut d_p0 = Line::empty(2usize);
            let mut d_p1 = Line::empty(2usize);
            let mut d_p2 = Line::empty(2usize);
            if t == f32::new(0.0) {
                d_p0[0] += d_closest_x;
                d_p0[1] += d_closest_y;
            } else if t == f32::new(1.0) {
                d_p2[0] += d_closest_x;
                d_p2[1] += d_closest_y;
            } else {
                let ax = p0[0] - f32::new(2.0) * p1[0] + p2[0];
                let ay = p0[1] - f32::new(2.0) * p1[1] + p2[1];
                let bx = p1[0] - p0[0];
                let by = p1[1] - p0[1];
                let A = ax * ax + ay * ay;
                let B = f32::new(3.0) * (ax * bx + ay * by);
                let C = f32::new(2.0) * (bx * bx + by * by) + (ax * (p0[0] - pt_x) + ay * (p0[1] - pt_y));
                let d_tt = f32::new(2.0) * tt * (d_closest_x * p0[0] + d_closest_y * p0[1])
                    + f32::new(2.0) * t * (d_closest_x * p1[0] + d_closest_y * p1[1]);
                let d_t = -d_tt
                    + f32::new(2.0) * tt * (d_closest_x * p1[0] + d_closest_y * p1[1])
                    + f32::new(2.0) * t * (d_closest_x * p2[0] + d_closest_y * p2[1]);
                d_p0[0] += d_closest_x * (tt * tt);
                d_p0[1] += d_closest_y * (tt * tt);
                d_p1[0] += d_closest_x * (f32::new(2.0) * tt * t);
                d_p1[1] += d_closest_y * (f32::new(2.0) * tt * t);
                d_p2[0] += d_closest_x * (t * t);
                d_p2[1] += d_closest_y * (t * t);
                let poly_deriv_t = f32::new(3.0) * A * t * t + f32::new(2.0) * B * t + C;
                if abs_f32(poly_deriv_t) > f32::new(1.0e-6) {
                    let d_A = -(d_t / poly_deriv_t) * t * t * t;
                    let d_B = -(d_t / poly_deriv_t) * t * t;
                    let d_C = -(d_t / poly_deriv_t) * t;
                    let d_D = -(d_t / poly_deriv_t);

                    d_p0[0] += ax * (f32::new(2.0) * d_A)
                        + (bx - ax) * (f32::new(3.0) * d_B)
                        + (bx * f32::new(-4.0)) * d_C
                        + (p0[0] - pt_x + ax) * d_C
                        + (bx - (p0[0] - pt_x)) * (f32::new(2.0) * d_D);
                    d_p0[1] += ay * (f32::new(2.0) * d_A)
                        + (by - ay) * (f32::new(3.0) * d_B)
                        + (by * f32::new(-4.0)) * d_C
                        + (p0[1] - pt_y + ay) * d_C
                        + (by - (p0[1] - pt_y)) * (f32::new(2.0) * d_D);

                    d_p1[0] += ax * (f32::new(-4.0) * d_A)
                        + (ax + bx * f32::new(-2.0)) * (f32::new(3.0) * d_B)
                        + (bx * f32::new(4.0)) * d_C
                        + (p0[0] - pt_x) * (f32::new(-2.0) * d_C)
                        + (p0[0] - pt_x) * d_D;
                    d_p1[1] += ay * (f32::new(-4.0) * d_A)
                        + (ay + by * f32::new(-2.0)) * (f32::new(3.0) * d_B)
                        + (by * f32::new(4.0)) * d_C
                        + (p0[1] - pt_y) * (f32::new(-2.0) * d_C)
                        + (p0[1] - pt_y) * d_D;

                    d_p2[0] += ax * (f32::new(2.0) * d_A) + bx * (f32::new(3.0) * d_B) + (p0[0] - pt_x) * d_C;
                    d_p2[1] += ay * (f32::new(2.0) * d_A) + by * (f32::new(3.0) * d_B) + (p0[1] - pt_y) * d_C;

                    d_pt[0] += ax * (-d_C) + bx * d_D;
                    d_pt[1] += ay * (-d_C) + by * d_D;
                }
            }
            add_path_point_grad(d_path_points, point_offset, i0, d_p0[0], d_p0[1]);
            add_path_point_grad(d_path_points, point_offset, i1, d_p1[0], d_p1[1]);
            add_path_point_grad(d_path_points, point_offset, i2, d_p2[0], d_p2[1]);
        } else if num_controls == u32::new(2) {
            let i0 = point_id;
            let i1 = point_id + u32::new(1);
            let i2 = point_id + u32::new(2);
            let i3 = path_point_index(point_id + u32::new(3), point_count, is_closed);
            let p0 = load_path_point(path_points, point_offset, i0);
            let p1 = load_path_point(path_points, point_offset, i1);
            let p2 = load_path_point(path_points, point_offset, i2);
            let p3 = load_path_point(path_points, point_offset, i3);
            let tt = f32::new(1.0) - t;
            let mut d_p0 = Line::empty(2usize);
            let mut d_p1 = Line::empty(2usize);
            let mut d_p2 = Line::empty(2usize);
            let mut d_p3 = Line::empty(2usize);
            if t == f32::new(0.0) {
                d_p0[0] += d_closest_x;
                d_p0[1] += d_closest_y;
            } else if t == f32::new(1.0) {
                d_p3[0] += d_closest_x;
                d_p3[1] += d_closest_y;
            } else {
                let ax = -p0[0] + f32::new(3.0) * p1[0] - f32::new(3.0) * p2[0] + p3[0];
                let ay = -p0[1] + f32::new(3.0) * p1[1] - f32::new(3.0) * p2[1] + p3[1];
                let bx = f32::new(3.0) * p0[0] - f32::new(6.0) * p1[0] + f32::new(3.0) * p2[0];
                let by = f32::new(3.0) * p0[1] - f32::new(6.0) * p1[1] + f32::new(3.0) * p2[1];
                let cx = -f32::new(3.0) * p0[0] + f32::new(3.0) * p1[0];
                let cy = -f32::new(3.0) * p0[1] + f32::new(3.0) * p1[1];
                let A = f32::new(3.0) * (ax * ax + ay * ay);
                if abs_f32(A) >= f32::new(1.0e-10) {
                    let B = f32::new(5.0) * (ax * bx + ay * by);
                    let C = f32::new(4.0) * (ax * cx + ay * cy) + f32::new(2.0) * (bx * bx + by * by);
                    let D = f32::new(3.0) * ((bx * cx + by * cy) + ax * (p0[0] - pt_x) + ay * (p0[1] - pt_y));
                    let E = cx * cx + cy * cy + f32::new(2.0) * ((p0[0] - pt_x) * bx + (p0[1] - pt_y) * by);
                    let F = (p0[0] - pt_x) * cx + (p0[1] - pt_y) * cy;
                    let B = B / A;
                    let C = C / A;
                    let D = D / A;
                    let E = E / A;
                    let F = F / A;

                    let d_tt = f32::new(3.0) * tt * tt * (d_closest_x * p0[0] + d_closest_y * p0[1])
                        + f32::new(6.0) * tt * t * (d_closest_x * p1[0] + d_closest_y * p1[1])
                        + f32::new(3.0) * t * t * (d_closest_x * p2[0] + d_closest_y * p2[1]);
                    let d_t = -d_tt
                        + f32::new(3.0) * tt * tt * (d_closest_x * p1[0] + d_closest_y * p1[1])
                        + f32::new(6.0) * tt * t * (d_closest_x * p2[0] + d_closest_y * p2[1])
                        + f32::new(3.0) * t * t * (d_closest_x * p3[0] + d_closest_y * p3[1]);

                    d_p0[0] += d_closest_x * (tt * tt * tt);
                    d_p0[1] += d_closest_y * (tt * tt * tt);
                    d_p1[0] += d_closest_x * (f32::new(3.0) * tt * tt * t);
                    d_p1[1] += d_closest_y * (f32::new(3.0) * tt * tt * t);
                    d_p2[0] += d_closest_x * (f32::new(3.0) * tt * t * t);
                    d_p2[1] += d_closest_y * (f32::new(3.0) * tt * t * t);
                    d_p3[0] += d_closest_x * (t * t * t);
                    d_p3[1] += d_closest_y * (t * t * t);

                    let poly_deriv_t = f32::new(5.0) * t * t * t * t
                        + f32::new(4.0) * B * t * t * t
                        + f32::new(3.0) * C * t * t
                        + f32::new(2.0) * D * t
                        + E;
                    if abs_f32(poly_deriv_t) > f32::new(1.0e-10) {
                        let mut d_B = -(d_t / poly_deriv_t) * t * t * t * t;
                        let mut d_C = -(d_t / poly_deriv_t) * t * t * t;
                        let mut d_D = -(d_t / poly_deriv_t) * t * t;
                        let mut d_E = -(d_t / poly_deriv_t) * t;
                        let mut d_F = -(d_t / poly_deriv_t);
                        let mut d_A = -d_B * B / A - d_C * C / A - d_D * D / A - d_E * E / A - d_F * F / A;
                        d_B /= A;
                        d_C /= A;
                        d_D /= A;
                        d_E /= A;
                        d_F /= A;

                        d_p0[0] += ax * (f32::new(3.0) * f32::new(-1.0) * f32::new(2.0) * d_A);
                        d_p0[1] += ay * (f32::new(3.0) * f32::new(-1.0) * f32::new(2.0) * d_A);
                        d_p1[0] += ax * (f32::new(3.0) * f32::new(3.0) * f32::new(2.0) * d_A);
                        d_p1[1] += ay * (f32::new(3.0) * f32::new(3.0) * f32::new(2.0) * d_A);
                        d_p2[0] += ax * (f32::new(3.0) * f32::new(-3.0) * f32::new(2.0) * d_A);
                        d_p2[1] += ay * (f32::new(3.0) * f32::new(-3.0) * f32::new(2.0) * d_A);
                        d_p3[0] += ax * (f32::new(3.0) * f32::new(1.0) * f32::new(2.0) * d_A);
                        d_p3[1] += ay * (f32::new(3.0) * f32::new(1.0) * f32::new(2.0) * d_A);

                        d_p0[0] += (bx * f32::new(-1.0) + ax * f32::new(3.0)) * (f32::new(5.0) * d_B);
                        d_p0[1] += (by * f32::new(-1.0) + ay * f32::new(3.0)) * (f32::new(5.0) * d_B);
                        d_p1[0] += (bx * f32::new(3.0) + ax * f32::new(-6.0)) * (f32::new(5.0) * d_B);
                        d_p1[1] += (by * f32::new(3.0) + ay * f32::new(-6.0)) * (f32::new(5.0) * d_B);
                        d_p2[0] += (bx * f32::new(-3.0) + ax * f32::new(3.0)) * (f32::new(5.0) * d_B);
                        d_p2[1] += (by * f32::new(-3.0) + ay * f32::new(3.0)) * (f32::new(5.0) * d_B);
                        d_p3[0] += bx * (f32::new(5.0) * d_B);
                        d_p3[1] += by * (f32::new(5.0) * d_B);

                        d_p0[0] += (cx * f32::new(-1.0) + ax * f32::new(-3.0)) * (f32::new(4.0) * d_C)
                            + bx * (f32::new(3.0) * f32::new(2.0) * d_C);
                        d_p0[1] += (cy * f32::new(-1.0) + ay * f32::new(-3.0)) * (f32::new(4.0) * d_C)
                            + by * (f32::new(3.0) * f32::new(2.0) * d_C);
                        d_p1[0] += (cx * f32::new(3.0) + ax * f32::new(3.0)) * (f32::new(4.0) * d_C)
                            + bx * (f32::new(-6.0) * f32::new(2.0) * d_C);
                        d_p1[1] += (cy * f32::new(3.0) + ay * f32::new(3.0)) * (f32::new(4.0) * d_C)
                            + by * (f32::new(-6.0) * f32::new(2.0) * d_C);
                        d_p2[0] += cx * (f32::new(-3.0) * d_C * f32::new(4.0))
                            + bx * (f32::new(3.0) * f32::new(2.0) * d_C);
                        d_p2[1] += cy * (f32::new(-3.0) * d_C * f32::new(4.0))
                            + by * (f32::new(3.0) * f32::new(2.0) * d_C);
                        d_p3[0] += cx * (f32::new(4.0) * d_C);
                        d_p3[1] += cy * (f32::new(4.0) * d_C);

                        d_p0[0] += (cx * f32::new(3.0) + bx * f32::new(-3.0)) * (f32::new(3.0) * d_D)
                            + (ax - (p0[0] - pt_x)) * (f32::new(3.0) * d_D);
                        d_p0[1] += (cy * f32::new(3.0) + by * f32::new(-3.0)) * (f32::new(3.0) * d_D)
                            + (ay - (p0[1] - pt_y)) * (f32::new(3.0) * d_D);
                        d_p1[0] += (cx * f32::new(-6.0) + bx * f32::new(3.0)) * (f32::new(3.0) * d_D)
                            + (p0[0] - pt_x) * (f32::new(3.0) * f32::new(3.0) * d_D);
                        d_p1[1] += (cy * f32::new(-6.0) + by * f32::new(3.0)) * (f32::new(3.0) * d_D)
                            + (p0[1] - pt_y) * (f32::new(3.0) * f32::new(3.0) * d_D);
                        d_p2[0] += cx * (f32::new(3.0) * f32::new(3.0) * d_D)
                            + (p0[0] - pt_x) * (f32::new(-3.0) * f32::new(3.0) * d_D);
                        d_p2[1] += cy * (f32::new(3.0) * f32::new(3.0) * d_D)
                            + (p0[1] - pt_y) * (f32::new(-3.0) * f32::new(3.0) * d_D);
                        d_pt[0] += ax * (f32::new(-1.0) * f32::new(3.0) * d_D);
                        d_pt[1] += ay * (f32::new(-1.0) * f32::new(3.0) * d_D);

                        d_p0[0] += cx * (f32::new(-3.0) * f32::new(2.0) * d_E)
                            + (bx + (p0[0] - pt_x) * f32::new(3.0)) * (f32::new(2.0) * d_E);
                        d_p0[1] += cy * (f32::new(-3.0) * f32::new(2.0) * d_E)
                            + (by + (p0[1] - pt_y) * f32::new(3.0)) * (f32::new(2.0) * d_E);
                        d_p1[0] += cx * (f32::new(3.0) * f32::new(2.0) * d_E)
                            + (p0[0] - pt_x) * (f32::new(-6.0) * f32::new(2.0) * d_E);
                        d_p1[1] += cy * (f32::new(3.0) * f32::new(2.0) * d_E)
                            + (p0[1] - pt_y) * (f32::new(-6.0) * f32::new(2.0) * d_E);
                        d_p2[0] += (p0[0] - pt_x) * (f32::new(3.0) * f32::new(2.0) * d_E);
                        d_p2[1] += (p0[1] - pt_y) * (f32::new(3.0) * f32::new(2.0) * d_E);
                        d_pt[0] += bx * (f32::new(-1.0) * f32::new(2.0) * d_E);
                        d_pt[1] += by * (f32::new(-1.0) * f32::new(2.0) * d_E);

                        d_p0[0] += cx * d_F + (p0[0] - pt_x) * (f32::new(-3.0) * d_F);
                        d_p0[1] += cy * d_F + (p0[1] - pt_y) * (f32::new(-3.0) * d_F);
                        d_p1[0] += (p0[0] - pt_x) * (f32::new(3.0) * d_F);
                        d_p1[1] += (p0[1] - pt_y) * (f32::new(3.0) * d_F);
                        d_pt[0] += cx * (f32::new(-1.0) * d_F);
                        d_pt[1] += cy * (f32::new(-1.0) * d_F);
                    }
                }
            }
            add_path_point_grad(d_path_points, point_offset, i0, d_p0[0], d_p0[1]);
            add_path_point_grad(d_path_points, point_offset, i1, d_p1[0], d_p1[1]);
            add_path_point_grad(d_path_points, point_offset, i2, d_p2[0], d_p2[1]);
            add_path_point_grad(d_path_points, point_offset, i3, d_p3[0], d_p3[1]);
        }
    }
}

/// Dispatch closest-point gradients based on shape kind.
#[cube]
pub(super) fn d_closest_point(
    shape_data: &Array<f32>,
    path_points: &Array<f32>,
    path_num_controls: &Array<u32>,
    shape_path_offsets: &Array<u32>,
    shape_path_point_counts: &Array<u32>,
    shape_path_ctrl_offsets: &Array<u32>,
    shape_path_is_closed: &Array<u32>,
    shape_index: u32,
    pt_x: f32,
    pt_y: f32,
    d_closest_x: f32,
    d_closest_y: f32,
    base_point_id: u32,
    t: f32,
    d_shape_params: &mut Array<Atomic<f32>>,
    d_path_points: &mut Array<Atomic<f32>>,
    d_pt: &mut Line<f32>,
) {
    let base = (shape_index * SHAPE_STRIDE) as usize;
    let kind = shape_data[base] as u32;
    let p0 = shape_data[base + 4];
    let p1 = shape_data[base + 5];
    let p2 = shape_data[base + 6];
    let p3 = shape_data[base + 7];

    if kind == SHAPE_KIND_CIRCLE {
        let vx = pt_x - p0;
        let vy = pt_y - p1;
        let n = vec2_normalize(vx, vy);
        let d_radius = d_closest_x * n[0] + d_closest_y * n[1];
        let d_nx = d_closest_x * p2;
        let d_ny = d_closest_y * p2;
        let d_v = d_normalize_vec2(vx, vy, d_nx, d_ny);
        let d_center_x = d_closest_x - d_v[0];
        let d_center_y = d_closest_y - d_v[1];
        d_pt[0] += d_v[0];
        d_pt[1] += d_v[1];
        let base = (shape_index * u32::new(8)) as usize;
        d_shape_params[base].fetch_add(d_center_x);
        d_shape_params[base + 1].fetch_add(d_center_y);
        d_shape_params[base + 2].fetch_add(d_radius);
    } else if kind == SHAPE_KIND_ELLIPSE {
        let mut d_center = Line::empty(2usize);
        let mut d_radius = Line::empty(2usize);
        d_closest_point_ellipse(
            p0,
            p1,
            p2,
            p3,
            pt_x,
            pt_y,
            d_closest_x,
            d_closest_y,
            &mut d_center,
            &mut d_radius,
            d_pt,
        );
        let base = (shape_index * u32::new(8)) as usize;
        d_shape_params[base].fetch_add(d_center[0]);
        d_shape_params[base + 1].fetch_add(d_center[1]);
        d_shape_params[base + 2].fetch_add(d_radius[0]);
        d_shape_params[base + 3].fetch_add(d_radius[1]);
    } else if kind == SHAPE_KIND_RECT {
        let mut d_min = Line::empty(2usize);
        let mut d_max = Line::empty(2usize);
        d_closest_point_rect(
            p0,
            p1,
            p2,
            p3,
            pt_x,
            pt_y,
            d_closest_x,
            d_closest_y,
            &mut d_min,
            &mut d_max,
            d_pt,
        );
        let base = (shape_index * u32::new(8)) as usize;
        d_shape_params[base].fetch_add(d_min[0]);
        d_shape_params[base + 1].fetch_add(d_min[1]);
        d_shape_params[base + 2].fetch_add(d_max[0]);
        d_shape_params[base + 3].fetch_add(d_max[1]);
    } else if kind == SHAPE_KIND_PATH {
        let point_offset = shape_path_offsets[shape_index as usize];
        let point_count = shape_path_point_counts[shape_index as usize];
        let ctrl_offset = shape_path_ctrl_offsets[shape_index as usize];
        let is_closed = shape_path_is_closed[shape_index as usize];
        d_closest_point_path(
            path_points,
            path_num_controls,
            point_offset,
            point_count,
            ctrl_offset,
            is_closed,
            base_point_id,
            t,
            pt_x,
            pt_y,
            d_closest_x,
            d_closest_y,
            d_path_points,
            d_pt,
        );
    }
}

/// Backprop distance from point to shape, including group and shape transforms.
#[cube]
pub(super) fn d_compute_distance(
    shape_data: &Array<f32>,
    shape_xform: &Array<f32>,
    shape_transform: &Array<f32>,
    group_xform: &Array<f32>,
    group_shape_xform: &Array<f32>,
    path_points: &Array<f32>,
    path_num_controls: &Array<u32>,
    shape_path_offsets: &Array<u32>,
    shape_path_point_counts: &Array<u32>,
    shape_path_ctrl_offsets: &Array<u32>,
    shape_path_is_closed: &Array<u32>,
    shape_index: u32,
    group_id: u32,
    pt_x: f32,
    pt_y: f32,
    local_closest_x: f32,
    local_closest_y: f32,
    base_point_id: u32,
    t: f32,
    d_dist: f32,
    d_shape_params: &mut Array<Atomic<f32>>,
    d_shape_points: &mut Array<Atomic<f32>>,
    d_shape_transform: &mut Array<Atomic<f32>>,
    d_group_transform: &mut Array<Atomic<f32>>,
    d_translation: &mut Array<Atomic<f32>>,
    translation_flag: u32,
    pixel_index: u32,
) {
    let group_base = (group_id * XFORM_STRIDE) as usize;
    let g_m00 = group_xform[group_base];
    let g_m01 = group_xform[group_base + 1];
    let g_m02 = group_xform[group_base + 2];
    let g_m10 = group_xform[group_base + 3];
    let g_m11 = group_xform[group_base + 4];
    let g_m12 = group_xform[group_base + 5];

    let shape_base = (shape_index * XFORM_STRIDE) as usize;
    let s_inv00 = shape_xform[shape_base];
    let s_inv01 = shape_xform[shape_base + 1];
    let s_inv02 = shape_xform[shape_base + 2];
    let s_inv10 = shape_xform[shape_base + 3];
    let s_inv11 = shape_xform[shape_base + 4];
    let s_inv12 = shape_xform[shape_base + 5];

    let s_m00 = shape_transform[shape_base];
    let s_m01 = shape_transform[shape_base + 1];
    let s_m02 = shape_transform[shape_base + 2];
    let s_m10 = shape_transform[shape_base + 3];
    let s_m11 = shape_transform[shape_base + 4];
    let s_m12 = shape_transform[shape_base + 5];

    let gs_base = (group_id * XFORM_STRIDE) as usize;
    let gs_m00 = group_shape_xform[gs_base];
    let gs_m01 = group_shape_xform[gs_base + 1];
    let gs_m02 = group_shape_xform[gs_base + 2];
    let gs_m10 = group_shape_xform[gs_base + 3];
    let gs_m11 = group_shape_xform[gs_base + 4];
    let gs_m12 = group_shape_xform[gs_base + 5];

    let local_pt_group = xform_pt_affine(g_m00, g_m01, g_m02, g_m10, g_m11, g_m12, pt_x, pt_y);
    let local_pt_shape = xform_pt_affine(
        s_inv00,
        s_inv01,
        s_inv02,
        s_inv10,
        s_inv11,
        s_inv12,
        local_pt_group[0],
        local_pt_group[1],
    );

    let local_closest_group = xform_pt_affine(
        s_m00,
        s_m01,
        s_m02,
        s_m10,
        s_m11,
        s_m12,
        local_closest_x,
        local_closest_y,
    );
    let closest_canvas = xform_pt_affine(
        gs_m00,
        gs_m01,
        gs_m02,
        gs_m10,
        gs_m11,
        gs_m12,
        local_closest_group[0],
        local_closest_group[1],
    );
    let diff_x = pt_x - closest_canvas[0];
    let diff_y = pt_y - closest_canvas[1];
    if vec2_dot(diff_x, diff_y, diff_x, diff_y) >= f32::new(1.0e-10) {
        let mut d_closest_canvas = Line::empty(2usize);
        let mut d_pt = Line::empty(2usize);
        d_distance(
            closest_canvas[0],
            closest_canvas[1],
            pt_x,
            pt_y,
            d_dist,
            &mut d_closest_canvas,
            &mut d_pt,
        );

    let mut d_shape_to_canvas_affine = Line::empty(AFFINE_SIZE);
    let mut d_local_closest_group = Line::empty(2usize);
    d_xform_pt_affine(
        gs_m00,
        gs_m01,
        gs_m02,
        gs_m10,
        gs_m11,
        gs_m12,
        local_closest_group[0],
        local_closest_group[1],
        d_closest_canvas[0],
        d_closest_canvas[1],
        &mut d_shape_to_canvas_affine,
        &mut d_local_closest_group,
    );
    let d_shape_to_canvas = affine_grad_to_mat3(d_shape_to_canvas_affine);

    let mut d_shape_transform_affine = Line::empty(AFFINE_SIZE);
    let mut d_local_closest_shape = Line::empty(2usize);
    d_xform_pt_affine(
        s_m00,
        s_m01,
        s_m02,
        s_m10,
        s_m11,
        s_m12,
        local_closest_x,
        local_closest_y,
        d_local_closest_group[0],
        d_local_closest_group[1],
        &mut d_shape_transform_affine,
        &mut d_local_closest_shape,
    );
    let d_shape_transform_local = affine_grad_to_mat3(d_shape_transform_affine);

    let mut d_local_pt_shape = Line::empty(2usize);
    d_closest_point(
        shape_data,
        path_points,
        path_num_controls,
        shape_path_offsets,
        shape_path_point_counts,
        shape_path_ctrl_offsets,
        shape_path_is_closed,
        shape_index,
        local_pt_shape[0],
        local_pt_shape[1],
        d_local_closest_shape[0],
        d_local_closest_shape[1],
        base_point_id,
        t,
        d_shape_params,
        d_shape_points,
        &mut d_local_pt_shape,
    );

    let mut d_shape_inv_affine = Line::empty(AFFINE_SIZE);
    let mut d_local_pt_group = Line::empty(2usize);
    d_xform_pt_affine(
        s_inv00,
        s_inv01,
        s_inv02,
        s_inv10,
        s_inv11,
        s_inv12,
        local_pt_group[0],
        local_pt_group[1],
        d_local_pt_shape[0],
        d_local_pt_shape[1],
        &mut d_shape_inv_affine,
        &mut d_local_pt_group,
    );
    let d_shape_inv = affine_grad_to_mat3(d_shape_inv_affine);

    let mut d_canvas_to_shape_affine = Line::empty(AFFINE_SIZE);
    let mut d_pt_extra = Line::empty(2usize);
    d_xform_pt_affine(
        g_m00,
        g_m01,
        g_m02,
        g_m10,
        g_m11,
        g_m12,
        pt_x,
        pt_y,
        d_local_pt_group[0],
        d_local_pt_group[1],
        &mut d_canvas_to_shape_affine,
        &mut d_pt_extra,
    );
    let d_canvas_to_shape = affine_grad_to_mat3(d_canvas_to_shape_affine);
    d_pt[0] += d_pt_extra[0];
    d_pt[1] += d_pt_extra[1];

    let c2s = mat3_from_affine(g_m00, g_m01, g_m02, g_m10, g_m11, g_m12);
    let tc2s = mat3_transpose(c2s);
    let corr_group = mat3_mul(mat3_mul(mat3_scale(tc2s, f32::new(-1.0)), d_canvas_to_shape), tc2s);

    let s_inv = mat3_from_affine(s_inv00, s_inv01, s_inv02, s_inv10, s_inv11, s_inv12);
    let ts_inv = mat3_transpose(s_inv);
    let corr_shape = mat3_mul(mat3_mul(mat3_scale(ts_inv, f32::new(-1.0)), d_shape_inv), ts_inv);

    let g_base = (group_id * u32::new(9)) as usize;
    atomic_add_mat3(d_group_transform, g_base, d_shape_to_canvas);
    atomic_add_mat3(d_group_transform, g_base, corr_group);

    let s_base = (shape_index * u32::new(9)) as usize;
    atomic_add_mat3(d_shape_transform, s_base, d_shape_transform_local);
    atomic_add_mat3(d_shape_transform, s_base, corr_shape);

        if translation_flag != u32::new(0) {
            add_translation(d_translation, pixel_index, -d_pt[0], -d_pt[1]);
        }
    }
}

/// Gather filtered d_render_image at a sample position with weight normalization.
#[cube]
pub(super) fn gather_d_color(
    filter_type: u32,
    radius: f32,
    d_render_image: &Array<f32>,
    weight_image: &Array<Atomic<f32>>,
    width: u32,
    height: u32,
    px: f32,
    py: f32,
) -> Line<f32> {
    let mut out = Line::empty(4usize);
    out[0] = f32::new(0.0);
    out[1] = f32::new(0.0);
    out[2] = f32::new(0.0);
    out[3] = f32::new(0.0);

    let x = px as i32;
    let y = py as i32;
    let ri = radius.ceil() as i32;
    let mut dy = -ri;
    while dy <= ri {
        let mut dx = -ri;
        while dx <= ri {
            let xx = x + dx;
            let yy = y + dy;
            if xx >= 0 && yy >= 0 && xx < width as i32 && yy < height as i32 {
                let xc = f32::cast_from(xx) + f32::new(0.5);
                let yc = f32::cast_from(yy) + f32::new(0.5);
                let w = filter_weight(filter_type, xc - px, yc - py, radius);
                if w > f32::new(0.0) {
                    let base = (yy as u32 * width + xx as u32) as usize;
                    let weight_sum = weight_image[base].load();
                    if weight_sum > f32::new(0.0) {
                        let idx4 = base * 4;
                        let scale = w / weight_sum;
                        out[0] += scale * d_render_image[idx4];
                        out[1] += scale * d_render_image[idx4 + 1];
                        out[2] += scale * d_render_image[idx4 + 2];
                        out[3] += scale * d_render_image[idx4 + 3];
                    }
                }
            }
            dx += 1;
        }
        dy += 1;
    }
    out
}

/// Backprop filter weight with respect to the filter radius.
#[cube]
pub(super) fn d_compute_filter_weight(
    filter_type: u32,
    radius: f32,
    dx: f32,
    dy: f32,
    d_return: f32,
    d_filter_radius: &mut Array<Atomic<f32>>,
) {
    if filter_type == FILTER_BOX {
        let denom = f32::new(2.0) * radius;
        if denom != f32::new(0.0) {
            let d_r = d_return * f32::new(-2.0) * denom / (denom * denom * denom);
            d_filter_radius[0].fetch_add(d_r);
        }
    } else if filter_type == FILTER_TENT {
        let fx = radius - abs_f32(dx);
        let fy = radius - abs_f32(dy);
        let norm = f32::new(1.0) / (radius * radius);
        let d_fx = d_return * fy * norm;
        let d_fy = d_return * fx * norm;
        let d_norm = d_return * fx * fy;
        if radius != f32::new(0.0) {
            let d_r = d_fx + d_fy + (f32::new(-4.0) * d_norm) / (radius * radius * radius * radius * radius);
            d_filter_radius[0].fetch_add(d_r);
        }
    } else if filter_type == FILTER_RADIAL_PARABOLIC {
        let r3 = radius * radius * radius;
        if r3 != f32::new(0.0) {
            let d_r = -(f32::new(2.0) * dx * dx + f32::new(2.0) * dy * dy) / r3;
            d_filter_radius[0].fetch_add(d_r * d_return);
        }
    } else if filter_type == FILTER_HANN {
        let ndx = (dx / (f32::new(2.0) * radius)) + f32::new(0.5);
        let ndy = (dy / (f32::new(2.0) * radius)) + f32::new(0.5);
        let fx = f32::new(0.5) * (f32::new(1.0) - (f32::new(2.0) * f32::new(3.14159265) * ndx).cos());
        let fy = f32::new(0.5) * (f32::new(1.0) - (f32::new(2.0) * f32::new(3.14159265) * ndy).cos());
        let norm = f32::new(1.0) / (radius * radius);
        let d_fx = d_return * fy * norm;
        let d_fy = d_return * fx * norm;
        let d_norm = d_return * fx * fy;
        let d_ndx = d_fx * f32::new(0.5)
            * (f32::new(2.0) * f32::new(3.14159265) * ndx).sin()
            * (f32::new(2.0) * f32::new(3.14159265));
        let d_ndy = d_fy * f32::new(0.5)
            * (f32::new(2.0) * f32::new(3.14159265) * ndy).sin()
            * (f32::new(2.0) * f32::new(3.14159265));
        let denom = f32::new(2.0) * radius;
        let denom_sq = denom * denom;
        let d_r = d_ndx * (f32::new(-2.0) * dx / denom_sq)
            + d_ndy * (f32::new(-2.0) * dy / denom_sq)
            + (f32::new(-2.0) * d_norm) / (radius * radius * radius);
        d_filter_radius[0].fetch_add(d_r);
    }
}

/// Accumulate filter-radius gradients from filtered color reconstruction.
#[cube]
pub(super) fn accumulate_filter_gradient(
    filter_type: u32,
    radius: f32,
    color: Line<f32>,
    d_render_image: &Array<f32>,
    weight_image: &Array<Atomic<f32>>,
    width: u32,
    height: u32,
    px: f32,
    py: f32,
    d_filter_radius: &mut Array<Atomic<f32>>,
) {
    let x = px as i32;
    let y = py as i32;
    let ri = radius.ceil() as i32;
    let mut dy = -ri;
    while dy <= ri {
        let mut dx = -ri;
        while dx <= ri {
            let xx = x + dx;
            let yy = y + dy;
            if xx >= 0 && yy >= 0 && xx < width as i32 && yy < height as i32 {
                let base = (yy as u32 * width + xx as u32) as usize;
                let weight_sum = weight_image[base].load();
                if weight_sum > f32::new(0.0) {
                    let xc = f32::cast_from(xx) + f32::new(0.5);
                    let yc = f32::cast_from(yy) + f32::new(0.5);
                    let w = filter_weight(filter_type, xc - px, yc - py, radius);
                    if w > f32::new(0.0) {
                        let idx4 = base * 4;
                        let d_pixel0 = d_render_image[idx4];
                        let d_pixel1 = d_render_image[idx4 + 1];
                        let d_pixel2 = d_render_image[idx4 + 2];
                        let d_pixel3 = d_render_image[idx4 + 3];
                        let dot = d_pixel0 * color[0]
                            + d_pixel1 * color[1]
                            + d_pixel2 * color[2]
                            + d_pixel3 * color[3];
                        let denom = weight_sum * weight_sum;
                        let d_weight = if denom > f32::new(0.0) {
                            (dot * weight_sum - w * dot * (weight_sum - w)) / denom
                        } else {
                            f32::new(0.0)
                        };
                        if d_weight != f32::new(0.0) {
                            d_compute_filter_weight(
                                filter_type,
                                radius,
                                xc - px,
                                yc - py,
                                d_weight,
                                d_filter_radius,
                            );
                        }
                    }
                }
            }
            dx += 1;
        }
        dy += 1;
    }
}

/// Backprop paint parameters (solid or gradient) from a color gradient.
#[cube]
pub(super) fn d_sample_paint(
    paint_kind: u32,
    gradient_index: u32,
    solid_r: f32,
    solid_g: f32,
    solid_b: f32,
    solid_a: f32,
    gradient_data: &Array<f32>,
    stop_offsets: &Array<f32>,
    stop_colors: &Array<f32>,
    px: f32,
    py: f32,
    d_color: Line<f32>,
    solid_offset: u32,
    d_group_data: &mut Array<Atomic<f32>>,
    d_gradient_data: &mut Array<Atomic<f32>>,
    d_stop_offsets: &mut Array<Atomic<f32>>,
    d_stop_colors: &mut Array<Atomic<f32>>,
    translation_flag: u32,
    d_translation: &mut Array<Atomic<f32>>,
    pixel_index: u32,
) {
    let mut handled = false;
    if paint_kind == PAINT_SOLID {
        let base = solid_offset as usize;
        d_group_data[base].fetch_add(d_color[0]);
        d_group_data[base + 1].fetch_add(d_color[1]);
        d_group_data[base + 2].fetch_add(d_color[2]);
        d_group_data[base + 3].fetch_add(d_color[3]);
        handled = true;
    }
    if !handled && (paint_kind == PAINT_LINEAR || paint_kind == PAINT_RADIAL) {
        let grad_base = (gradient_index * GRADIENT_STRIDE) as usize;
        let grad_type = gradient_data[grad_base] as u32;
        let p0 = gradient_data[grad_base + 1];
        let p1 = gradient_data[grad_base + 2];
        let p2 = gradient_data[grad_base + 3];
        let p3 = gradient_data[grad_base + 4];
        let stop_offset = gradient_data[grad_base + 5] as u32;
        let stop_count = gradient_data[grad_base + 6] as u32;
        if stop_count != u32::new(0) {
            if grad_type == u32::new(0) {
                let vx = p2 - p0;
                let vy = p3 - p1;
                let denom = max_f32(vx * vx + vy * vy, f32::new(1.0e-3));
                let t = ((px - p0) * vx + (py - p1) * vy) / denom;
                let mut done = false;
                if t < stop_offsets[stop_offset as usize] {
                    let base = (stop_offset * u32::new(4)) as usize;
                    d_stop_colors[base].fetch_add(d_color[0]);
                    d_stop_colors[base + 1].fetch_add(d_color[1]);
                    d_stop_colors[base + 2].fetch_add(d_color[2]);
                    d_stop_colors[base + 3].fetch_add(d_color[3]);
                    done = true;
                } else {
                    let mut i = u32::new(0);
                    while i + u32::new(1) < stop_count {
                        let curr = stop_offsets[(stop_offset + i) as usize];
                        let next = stop_offsets[(stop_offset + i + u32::new(1)) as usize];
                        if t >= curr && t < next && !done {
                            let tt = (t - curr) / max_f32(next - curr, f32::new(1.0e-5));
                            let color_curr_base = ((stop_offset + i) * u32::new(4)) as usize;
                            let color_next_base = ((stop_offset + i + u32::new(1)) * u32::new(4)) as usize;
                            let c0r = stop_colors[color_curr_base];
                            let c0g = stop_colors[color_curr_base + 1];
                            let c0b = stop_colors[color_curr_base + 2];
                            let c0a = stop_colors[color_curr_base + 3];
                            let c1r = stop_colors[color_next_base];
                            let c1g = stop_colors[color_next_base + 1];
                            let c1b = stop_colors[color_next_base + 2];
                            let c1a = stop_colors[color_next_base + 3];

                            d_stop_colors[color_curr_base].fetch_add(d_color[0] * (f32::new(1.0) - tt));
                            d_stop_colors[color_curr_base + 1].fetch_add(d_color[1] * (f32::new(1.0) - tt));
                            d_stop_colors[color_curr_base + 2].fetch_add(d_color[2] * (f32::new(1.0) - tt));
                            d_stop_colors[color_curr_base + 3].fetch_add(d_color[3] * (f32::new(1.0) - tt));
                            d_stop_colors[color_next_base].fetch_add(d_color[0] * tt);
                            d_stop_colors[color_next_base + 1].fetch_add(d_color[1] * tt);
                            d_stop_colors[color_next_base + 2].fetch_add(d_color[2] * tt);
                            d_stop_colors[color_next_base + 3].fetch_add(d_color[3] * tt);

                            let diff_r = c1r - c0r;
                            let diff_g = c1g - c0g;
                            let diff_b = c1b - c0b;
                            let diff_a = c1a - c0a;
                            let d_tt = d_color[0] * diff_r + d_color[1] * diff_g + d_color[2] * diff_b + d_color[3] * diff_a;
                            let denom_offset = next - curr;
                            if abs_f32(denom_offset) > f32::new(0.0) {
                                let d_offset_next = -d_tt * tt / denom_offset;
                                let d_offset_curr = d_tt * (tt - f32::new(1.0)) / denom_offset;
                                d_stop_offsets[(stop_offset + i) as usize].fetch_add(d_offset_curr);
                                d_stop_offsets[(stop_offset + i + u32::new(1)) as usize].fetch_add(d_offset_next);
                            }
                            let d_t = d_tt / denom_offset;
                            let d_beg_x = d_t * (-(px - p0) - (p2 - p0)) / denom;
                            let d_beg_y = d_t * (-(py - p1) - (p3 - p1)) / denom;
                            let d_end_x = d_t * (px - p0) / denom;
                            let d_end_y = d_t * (py - p1) / denom;
                            let d_l = -d_t * t / denom;
                            if vec2_dot(vx, vy, vx, vy) > f32::new(1.0e-3) {
                                d_gradient_data[grad_base + 1].fetch_add(d_beg_x + f32::new(2.0) * d_l * (p0 - p2));
                                d_gradient_data[grad_base + 2].fetch_add(d_beg_y + f32::new(2.0) * d_l * (p1 - p3));
                                d_gradient_data[grad_base + 3].fetch_add(d_end_x + f32::new(2.0) * d_l * (p2 - p0));
                                d_gradient_data[grad_base + 4].fetch_add(d_end_y + f32::new(2.0) * d_l * (p3 - p1));
                            } else {
                                d_gradient_data[grad_base + 1].fetch_add(d_beg_x);
                                d_gradient_data[grad_base + 2].fetch_add(d_beg_y);
                                d_gradient_data[grad_base + 3].fetch_add(d_end_x);
                                d_gradient_data[grad_base + 4].fetch_add(d_end_y);
                            }
                            if translation_flag != u32::new(0) {
                                add_translation(d_translation, pixel_index, d_beg_x + d_end_x, d_beg_y + d_end_y);
                            }
                            done = true;
                        }
                        i += u32::new(1);
                    }
                }
                if !done {
                    let last_base = ((stop_offset + stop_count - u32::new(1)) * u32::new(4)) as usize;
                    d_stop_colors[last_base].fetch_add(d_color[0]);
                    d_stop_colors[last_base + 1].fetch_add(d_color[1]);
                    d_stop_colors[last_base + 2].fetch_add(d_color[2]);
                    d_stop_colors[last_base + 3].fetch_add(d_color[3]);
                }
            } else {
                let offset_x = px - p0;
                let offset_y = py - p1;
                let norm_x = offset_x / p2;
                let norm_y = offset_y / p3;
                let t = vec2_length(norm_x, norm_y);
                let mut done = false;
                if t < stop_offsets[stop_offset as usize] {
                    let base = (stop_offset * u32::new(4)) as usize;
                    d_stop_colors[base].fetch_add(d_color[0]);
                    d_stop_colors[base + 1].fetch_add(d_color[1]);
                    d_stop_colors[base + 2].fetch_add(d_color[2]);
                    d_stop_colors[base + 3].fetch_add(d_color[3]);
                    done = true;
                } else {
                    let mut i = u32::new(0);
                    while i + u32::new(1) < stop_count {
                        let curr = stop_offsets[(stop_offset + i) as usize];
                        let next = stop_offsets[(stop_offset + i + u32::new(1)) as usize];
                        if t >= curr && t < next && !done {
                            let tt = (t - curr) / max_f32(next - curr, f32::new(1.0e-5));
                            let color_curr_base = ((stop_offset + i) * u32::new(4)) as usize;
                            let color_next_base = ((stop_offset + i + u32::new(1)) * u32::new(4)) as usize;
                            let c0r = stop_colors[color_curr_base];
                            let c0g = stop_colors[color_curr_base + 1];
                            let c0b = stop_colors[color_curr_base + 2];
                            let c0a = stop_colors[color_curr_base + 3];
                            let c1r = stop_colors[color_next_base];
                            let c1g = stop_colors[color_next_base + 1];
                            let c1b = stop_colors[color_next_base + 2];
                            let c1a = stop_colors[color_next_base + 3];

                            d_stop_colors[color_curr_base].fetch_add(d_color[0] * (f32::new(1.0) - tt));
                            d_stop_colors[color_curr_base + 1].fetch_add(d_color[1] * (f32::new(1.0) - tt));
                            d_stop_colors[color_curr_base + 2].fetch_add(d_color[2] * (f32::new(1.0) - tt));
                            d_stop_colors[color_curr_base + 3].fetch_add(d_color[3] * (f32::new(1.0) - tt));
                            d_stop_colors[color_next_base].fetch_add(d_color[0] * tt);
                            d_stop_colors[color_next_base + 1].fetch_add(d_color[1] * tt);
                            d_stop_colors[color_next_base + 2].fetch_add(d_color[2] * tt);
                            d_stop_colors[color_next_base + 3].fetch_add(d_color[3] * tt);

                            let diff_r = c1r - c0r;
                            let diff_g = c1g - c0g;
                            let diff_b = c1b - c0b;
                            let diff_a = c1a - c0a;
                            let d_tt = d_color[0] * diff_r + d_color[1] * diff_g + d_color[2] * diff_b + d_color[3] * diff_a;
                            let denom_offset = next - curr;
                            if abs_f32(denom_offset) > f32::new(0.0) {
                                let d_offset_next = -d_tt * tt / denom_offset;
                                let d_offset_curr = d_tt * (tt - f32::new(1.0)) / denom_offset;
                                d_stop_offsets[(stop_offset + i) as usize].fetch_add(d_offset_curr);
                                d_stop_offsets[(stop_offset + i + u32::new(1)) as usize].fetch_add(d_offset_next);
                            }
                            let d_t = d_tt / denom_offset;
                            let d_norm = d_length_vec2(norm_x, norm_y, d_t);
                            let d_offset_x = d_norm[0] / p2;
                            let d_offset_y = d_norm[1] / p3;
                            let d_radius_x = -d_norm[0] * offset_x / (p2 * p2);
                            let d_radius_y = -d_norm[1] * offset_y / (p3 * p3);
                            let d_center_x = -d_offset_x;
                            let d_center_y = -d_offset_y;
                            d_gradient_data[grad_base + 1].fetch_add(d_center_x);
                            d_gradient_data[grad_base + 2].fetch_add(d_center_y);
                            d_gradient_data[grad_base + 3].fetch_add(d_radius_x);
                            d_gradient_data[grad_base + 4].fetch_add(d_radius_y);
                            if translation_flag != u32::new(0) {
                                add_translation(d_translation, pixel_index, d_center_x, d_center_y);
                            }
                            done = true;
                        }
                        i += u32::new(1);
                    }
                }
                if !done {
                    let last_base = ((stop_offset + stop_count - u32::new(1)) * u32::new(4)) as usize;
                    d_stop_colors[last_base].fetch_add(d_color[0]);
                    d_stop_colors[last_base + 1].fetch_add(d_color[1]);
                    d_stop_colors[last_base + 2].fetch_add(d_color[2]);
                    d_stop_colors[last_base + 3].fetch_add(d_color[3]);
                }
            }
        }
    }
    let _ = solid_r;
    let _ = solid_g;
    let _ = solid_b;
    let _ = solid_a;
}

/// Check if a point is inside a group fill using BVH and fill rule.
#[cube]
pub(super) fn fill_inside_group(
    shape_data: &Array<f32>,
    segment_data: &Array<f32>,
    shape_bounds: &Array<f32>,
    group_data: &Array<f32>,
    group_xform: &Array<f32>,
    group_shapes: &Array<f32>,
    shape_xform: &Array<f32>,
    curve_data: &Array<f32>,
    group_bvh_bounds: &Array<f32>,
    group_bvh_nodes: &Array<u32>,
    group_bvh_indices: &Array<u32>,
    group_bvh_meta: &Array<u32>,
    path_bvh_bounds: &Array<f32>,
    path_bvh_nodes: &Array<u32>,
    path_bvh_indices: &Array<u32>,
    path_bvh_meta: &Array<u32>,
    group_id: u32,
    px: f32,
    py: f32,
    fill_kind: u32,
    fill_rule: u32,
) -> u32 {
    let mut result = u32::new(0);
    if fill_kind != PAINT_NONE {
        let big = f32::new(1.0e20);
        let local_pt = {
            let base = (group_id * XFORM_STRIDE) as usize;
            xform_pt_affine(
                group_xform[base],
                group_xform[base + 1],
                group_xform[base + 2],
                group_xform[base + 3],
                group_xform[base + 4],
                group_xform[base + 5],
                px,
                py,
            )
        };
        let mut fill_min_dist = big;
        let mut fill_winding = f32::new(0.0);
        let mut fill_crossings = f32::new(0.0);
        let mut stroke_min_dist = big;
        let mut stroke_min_radius = f32::new(0.0);
        let mut stroke_hit = f32::new(0.0);
        accumulate_group_shapes(
            shape_data,
            segment_data,
            shape_bounds,
            group_data,
            group_shapes,
            shape_xform,
            curve_data,
            group_id,
            local_pt[0],
            local_pt[1],
            fill_kind,
            PAINT_NONE,
            fill_rule,
            u32::new(0),
            group_bvh_bounds,
            group_bvh_nodes,
            group_bvh_indices,
            group_bvh_meta,
            path_bvh_bounds,
            path_bvh_nodes,
            path_bvh_indices,
            path_bvh_meta,
            &mut fill_min_dist,
            &mut fill_winding,
            &mut fill_crossings,
            &mut stroke_min_dist,
            &mut stroke_min_radius,
            &mut stroke_hit,
        );
        if fill_rule == u32::new(1) {
            if fill_crossings > f32::new(0.0) {
                result = u32::new(1);
            }
        } else if fill_winding != f32::new(0.0) {
            result = u32::new(1);
        }
    }
    result
}

/// Backward color sampling without prefiltering; accumulates paint/background gradients.
#[cube]
pub(super) fn sample_color(
    shape_data: &Array<f32>,
    segment_data: &Array<f32>,
    shape_bounds: &Array<f32>,
    group_data: &Array<f32>,
    group_xform: &Array<f32>,
    group_shapes: &Array<f32>,
    shape_xform: &Array<f32>,
    curve_data: &Array<f32>,
    gradient_data: &Array<f32>,
    stop_offsets: &Array<f32>,
    stop_colors: &Array<f32>,
    group_bvh_bounds: &Array<f32>,
    group_bvh_nodes: &Array<u32>,
    group_bvh_indices: &Array<u32>,
    group_bvh_meta: &Array<u32>,
    path_bvh_bounds: &Array<f32>,
    path_bvh_nodes: &Array<u32>,
    path_bvh_indices: &Array<u32>,
    path_bvh_meta: &Array<u32>,
    num_groups: u32,
    px: f32,
    py: f32,
    background: Line<f32>,
    d_color: Line<f32>,
    has_background_image: u32,
    translation_flag: u32,
    pixel_index: u32,
    d_group_data: &mut Array<Atomic<f32>>,
    d_gradient_data: &mut Array<Atomic<f32>>,
    d_stop_offsets: &mut Array<Atomic<f32>>,
    d_stop_colors: &mut Array<Atomic<f32>>,
    d_background: &mut Array<Atomic<f32>>,
    d_background_image: &mut Array<Atomic<f32>>,
    d_translation: &mut Array<Atomic<f32>>,
) -> Line<f32> {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let big = f32::new(1.0e20);

    let mut frag_color_r: Array<f32> = Array::new(MAX_FRAGMENTS);
    let mut frag_color_g: Array<f32> = Array::new(MAX_FRAGMENTS);
    let mut frag_color_b: Array<f32> = Array::new(MAX_FRAGMENTS);
    let mut frag_alpha: Array<f32> = Array::new(MAX_FRAGMENTS);
    let mut frag_group: Array<u32> = Array::new(MAX_FRAGMENTS);
    let mut frag_is_stroke: Array<u32> = Array::new(MAX_FRAGMENTS);
    let mut frag_count = u32::new(0);

    let mut group_id = u32::new(0);
    while group_id < num_groups {
        let group_base = (group_id * GROUP_STRIDE) as usize;
        let fill_kind = group_data[group_base + 2] as u32;
        let fill_index = group_data[group_base + 3] as u32;
        let stroke_kind = group_data[group_base + 4] as u32;
        let stroke_index = group_data[group_base + 5] as u32;
        let fill_rule = group_data[group_base + 7] as u32;

        let local_pt = {
            let base = (group_id * XFORM_STRIDE) as usize;
            xform_pt_affine(
                group_xform[base],
                group_xform[base + 1],
                group_xform[base + 2],
                group_xform[base + 3],
                group_xform[base + 4],
                group_xform[base + 5],
                px,
                py,
            )
        };

        let mut fill_min_dist = big;
        let mut fill_winding = zero;
        let mut fill_crossings = zero;
        let mut stroke_min_dist = big;
        let mut stroke_min_radius = zero;
        let mut stroke_hit = zero;
        if fill_kind != PAINT_NONE || stroke_kind != PAINT_NONE {
            accumulate_group_shapes(
                shape_data,
                segment_data,
                shape_bounds,
                group_data,
                group_shapes,
                shape_xform,
                curve_data,
                group_id,
                local_pt[0],
                local_pt[1],
                fill_kind,
                stroke_kind,
                fill_rule,
                u32::new(0),
                group_bvh_bounds,
                group_bvh_nodes,
                group_bvh_indices,
                group_bvh_meta,
                path_bvh_bounds,
                path_bvh_nodes,
                path_bvh_indices,
                path_bvh_meta,
                &mut fill_min_dist,
                &mut fill_winding,
                &mut fill_crossings,
                &mut stroke_min_dist,
                &mut stroke_min_radius,
                &mut stroke_hit,
            );
        }

        if stroke_kind != PAINT_NONE && stroke_hit > zero {
            if frag_count < u32::new(MAX_FRAGMENTS as i64) {
                let color = paint_color(
                    stroke_kind,
                    stroke_index,
                    group_data[group_base + 12],
                    group_data[group_base + 13],
                    group_data[group_base + 14],
                    group_data[group_base + 15],
                    gradient_data,
                    stop_offsets,
                    stop_colors,
                    px,
                    py,
                );
                let idx = frag_count as usize;
                frag_color_r[idx] = color[0];
                frag_color_g[idx] = color[1];
                frag_color_b[idx] = color[2];
                frag_alpha[idx] = color[3];
                frag_group[idx] = group_id;
                frag_is_stroke[idx] = u32::new(1);
                frag_count += u32::new(1);
            }
        }

        if fill_kind != PAINT_NONE {
            let inside = if fill_rule == u32::new(1) {
                fill_crossings > zero
            } else {
                fill_winding != zero
            };
            if inside {
                if frag_count < u32::new(MAX_FRAGMENTS as i64) {
                    let color = paint_color(
                        fill_kind,
                        fill_index,
                        group_data[group_base + 8],
                        group_data[group_base + 9],
                        group_data[group_base + 10],
                        group_data[group_base + 11],
                        gradient_data,
                        stop_offsets,
                        stop_colors,
                        px,
                        py,
                    );
                    let idx = frag_count as usize;
                    frag_color_r[idx] = color[0];
                    frag_color_g[idx] = color[1];
                    frag_color_b[idx] = color[2];
                    frag_alpha[idx] = color[3];
                    frag_group[idx] = group_id;
                    frag_is_stroke[idx] = u32::new(0);
                    frag_count += u32::new(1);
                }
            }
        }

        group_id += u32::new(1);
    }

    let mut out = Line::empty(4usize);
    if frag_count == u32::new(0) {
        out[0] = background[0];
        out[1] = background[1];
        out[2] = background[2];
        out[3] = background[3];
        accumulate_background_grad(
            d_background,
            d_background_image,
            has_background_image,
            pixel_index,
            d_color[0],
            d_color[1],
            d_color[2],
            d_color[3],
        );
    } else {
        let mut accum_r: Array<f32> = Array::new(MAX_FRAGMENTS);
        let mut accum_g: Array<f32> = Array::new(MAX_FRAGMENTS);
        let mut accum_b: Array<f32> = Array::new(MAX_FRAGMENTS);
        let mut accum_a: Array<f32> = Array::new(MAX_FRAGMENTS);

        let mut prev_r = background[0];
        let mut prev_g = background[1];
        let mut prev_b = background[2];
        let mut prev_a = background[3];

        let mut i = u32::new(0);
        while i < frag_count {
            let idx = i as usize;
            let new_r = frag_color_r[idx];
            let new_g = frag_color_g[idx];
            let new_b = frag_color_b[idx];
            let new_a = frag_alpha[idx];
            let blended_r = prev_r * (one - new_a) + new_r * new_a;
            let blended_g = prev_g * (one - new_a) + new_g * new_a;
            let blended_b = prev_b * (one - new_a) + new_b * new_a;
            let blended_a = prev_a * (one - new_a) + new_a;
            accum_r[idx] = blended_r;
            accum_g[idx] = blended_g;
            accum_b[idx] = blended_b;
            accum_a[idx] = blended_a;
            prev_r = blended_r;
            prev_g = blended_g;
            prev_b = blended_b;
            prev_a = blended_a;
            i += u32::new(1);
        }

        let last_idx = (frag_count - u32::new(1)) as usize;
        let mut final_r = accum_r[last_idx];
        let mut final_g = accum_g[last_idx];
        let mut final_b = accum_b[last_idx];
        let final_a = accum_a[last_idx];
        if final_a > f32::new(1.0e-6) {
            let inv = one / final_a;
            final_r *= inv;
            final_g *= inv;
            final_b *= inv;
        }

        out[0] = final_r;
        out[1] = final_g;
        out[2] = final_b;
        out[3] = final_a;

        let mut d_curr_r = d_color[0];
        let mut d_curr_g = d_color[1];
        let mut d_curr_b = d_color[2];
        let mut d_curr_a = d_color[3];
        if final_a > f32::new(1.0e-6) {
            let d_final_r = d_curr_r;
            let d_final_g = d_curr_g;
            let d_final_b = d_curr_b;
            let inv = one / final_a;
            d_curr_r *= inv;
            d_curr_g *= inv;
            d_curr_b *= inv;
            d_curr_a -= (d_final_r * final_r + d_final_g * final_g + d_final_b * final_b) * inv;
        }

        let mut ri = frag_count;
        while ri > u32::new(0) {
            ri -= u32::new(1);
            let idx = ri as usize;
            let prev_alpha = if ri > u32::new(0) {
                accum_a[(ri - u32::new(1)) as usize]
            } else {
                background[3]
            };
            let prev_r = if ri > u32::new(0) {
                accum_r[(ri - u32::new(1)) as usize]
            } else {
                background[0]
            };
            let prev_g = if ri > u32::new(0) {
                accum_g[(ri - u32::new(1)) as usize]
            } else {
                background[1]
            };
            let prev_b = if ri > u32::new(0) {
                accum_b[(ri - u32::new(1)) as usize]
            } else {
                background[2]
            };

            let frag_r = frag_color_r[idx];
            let frag_g = frag_color_g[idx];
            let frag_b = frag_color_b[idx];
            let frag_a = frag_alpha[idx];

            let d_prev_a = d_curr_a * (one - frag_a);
            let mut d_alpha_i = d_curr_a * (one - prev_alpha);
            d_alpha_i += d_curr_r * (frag_r - prev_r)
                + d_curr_g * (frag_g - prev_g)
                + d_curr_b * (frag_b - prev_b);
            let d_prev_r = d_curr_r * (one - frag_a);
            let d_prev_g = d_curr_g * (one - frag_a);
            let d_prev_b = d_curr_b * (one - frag_a);
            let d_color_i_r = d_curr_r * frag_a;
            let d_color_i_g = d_curr_g * frag_a;
            let d_color_i_b = d_curr_b * frag_a;

            let group_id = frag_group[idx];
            let group_base = (group_id * GROUP_STRIDE) as usize;
            if frag_is_stroke[idx] != u32::new(0) {
                let stroke_kind = group_data[group_base + 4] as u32;
                let stroke_index = group_data[group_base + 5] as u32;
                d_sample_paint(
                    stroke_kind,
                    stroke_index,
                    group_data[group_base + 12],
                    group_data[group_base + 13],
                    group_data[group_base + 14],
                    group_data[group_base + 15],
                    gradient_data,
                    stop_offsets,
                    stop_colors,
                    px,
                    py,
                    {
                        let mut tmp = Line::empty(4usize);
                        tmp[0] = d_color_i_r;
                        tmp[1] = d_color_i_g;
                        tmp[2] = d_color_i_b;
                        tmp[3] = d_alpha_i;
                        tmp
                    },
                    (group_base + 12) as u32,
                    d_group_data,
                    d_gradient_data,
                    d_stop_offsets,
                    d_stop_colors,
                    translation_flag,
                    d_translation,
                    pixel_index,
                );
            } else {
                let fill_kind = group_data[group_base + 2] as u32;
                let fill_index = group_data[group_base + 3] as u32;
                d_sample_paint(
                    fill_kind,
                    fill_index,
                    group_data[group_base + 8],
                    group_data[group_base + 9],
                    group_data[group_base + 10],
                    group_data[group_base + 11],
                    gradient_data,
                    stop_offsets,
                    stop_colors,
                    px,
                    py,
                    {
                        let mut tmp = Line::empty(4usize);
                        tmp[0] = d_color_i_r;
                        tmp[1] = d_color_i_g;
                        tmp[2] = d_color_i_b;
                        tmp[3] = d_alpha_i;
                        tmp
                    },
                    (group_base + 8) as u32,
                    d_group_data,
                    d_gradient_data,
                    d_stop_offsets,
                    d_stop_colors,
                    translation_flag,
                    d_translation,
                    pixel_index,
                );
            }

            d_curr_r = d_prev_r;
            d_curr_g = d_prev_g;
            d_curr_b = d_prev_b;
            d_curr_a = d_prev_a;
        }

        accumulate_background_grad(
            d_background,
            d_background_image,
            has_background_image,
            pixel_index,
            d_curr_r,
            d_curr_g,
            d_curr_b,
            d_curr_a,
        );
    }

    out
}

/// Backward color sampling with prefiltering and distance-based coverage.
#[cube]
pub(super) fn sample_color_prefiltered(
    shape_data: &Array<f32>,
    segment_data: &Array<f32>,
    shape_bounds: &Array<f32>,
    group_data: &Array<f32>,
    group_xform: &Array<f32>,
    group_shape_xform: &Array<f32>,
    group_shapes: &Array<f32>,
    shape_xform: &Array<f32>,
    shape_transform: &Array<f32>,
    curve_data: &Array<f32>,
    gradient_data: &Array<f32>,
    stop_offsets: &Array<f32>,
    stop_colors: &Array<f32>,
    group_bvh_bounds: &Array<f32>,
    group_bvh_nodes: &Array<u32>,
    group_bvh_indices: &Array<u32>,
    group_bvh_meta: &Array<u32>,
    path_bvh_bounds: &Array<f32>,
    path_bvh_nodes: &Array<u32>,
    path_bvh_indices: &Array<u32>,
    path_bvh_meta: &Array<u32>,
    path_points: &Array<f32>,
    path_num_controls: &Array<u32>,
    shape_path_offsets: &Array<u32>,
    shape_path_point_counts: &Array<u32>,
    shape_path_ctrl_offsets: &Array<u32>,
    shape_path_is_closed: &Array<u32>,
    num_groups: u32,
    px: f32,
    py: f32,
    background: Line<f32>,
    d_color: Line<f32>,
    has_background_image: u32,
    translation_flag: u32,
    pixel_index: u32,
    d_shape_params: &mut Array<Atomic<f32>>,
    d_shape_points: &mut Array<Atomic<f32>>,
    d_shape_stroke_width: &mut Array<Atomic<f32>>,
    d_shape_transform: &mut Array<Atomic<f32>>,
    d_group_transform: &mut Array<Atomic<f32>>,
    d_group_data: &mut Array<Atomic<f32>>,
    d_gradient_data: &mut Array<Atomic<f32>>,
    d_stop_offsets: &mut Array<Atomic<f32>>,
    d_stop_colors: &mut Array<Atomic<f32>>,
    d_background: &mut Array<Atomic<f32>>,
    d_background_image: &mut Array<Atomic<f32>>,
    d_translation: &mut Array<Atomic<f32>>,
) -> Line<f32> {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let big = f32::new(1.0e20);

    let mut frag_color_r: Array<f32> = Array::new(MAX_PREFILTER_FRAGMENTS);
    let mut frag_color_g: Array<f32> = Array::new(MAX_PREFILTER_FRAGMENTS);
    let mut frag_color_b: Array<f32> = Array::new(MAX_PREFILTER_FRAGMENTS);
    let mut frag_alpha: Array<f32> = Array::new(MAX_PREFILTER_FRAGMENTS);
    let mut frag_group: Array<u32> = Array::new(MAX_PREFILTER_FRAGMENTS);
    let mut frag_is_stroke: Array<u32> = Array::new(MAX_PREFILTER_FRAGMENTS);
    let mut frag_shape: Array<u32> = Array::new(MAX_PREFILTER_FRAGMENTS);
    let mut frag_distance: Array<f32> = Array::new(MAX_PREFILTER_FRAGMENTS);
    let mut frag_local_x: Array<f32> = Array::new(MAX_PREFILTER_FRAGMENTS);
    let mut frag_local_y: Array<f32> = Array::new(MAX_PREFILTER_FRAGMENTS);
    let mut frag_base_point: Array<u32> = Array::new(MAX_PREFILTER_FRAGMENTS);
    let mut frag_t: Array<f32> = Array::new(MAX_PREFILTER_FRAGMENTS);
    let mut frag_within: Array<u32> = Array::new(MAX_PREFILTER_FRAGMENTS);
    let mut frag_count = u32::new(0);

    let mut group_id = u32::new(0);
    while group_id < num_groups {
        let group_base = (group_id * GROUP_STRIDE) as usize;
        let fill_kind = group_data[group_base + 2] as u32;
        let fill_index = group_data[group_base + 3] as u32;
        let stroke_kind = group_data[group_base + 4] as u32;
        let stroke_index = group_data[group_base + 5] as u32;
        let fill_rule = group_data[group_base + 7] as u32;

        if stroke_kind != PAINT_NONE {
            if frag_count < u32::new(MAX_PREFILTER_FRAGMENTS as i64) {
                let mut local = Line::empty(2usize);
                let mut dist = big;
                let mut shape_id = u32::new(0);
                let mut base_point = u32::new(0);
                let mut t = zero;
                let found = compute_distance_group(
                    shape_data,
                    shape_xform,
                    shape_transform,
                    group_xform,
                    group_shape_xform,
                    group_data,
                    group_shapes,
                    curve_data,
                    group_bvh_bounds,
                    group_bvh_nodes,
                    group_bvh_indices,
                    group_bvh_meta,
                    group_id,
                    px,
                    py,
                    big,
                    &mut local,
                    &mut dist,
                    &mut shape_id,
                    &mut base_point,
                    &mut t,
                );
                if found != u32::new(0) {
                    let shape_base = (shape_id * SHAPE_STRIDE) as usize;
                    let stroke_width = shape_data[shape_base + 3];
                    let abs_d = abs_f32(dist);
                    let abs_plus = abs_d + stroke_width;
                    let abs_minus = abs_d - stroke_width;
                    let w = smoothstep_unit(abs_plus) - smoothstep_unit(abs_minus);
                    if w > zero {
                        let color = paint_color(
                            stroke_kind,
                            stroke_index,
                            group_data[group_base + 12],
                            group_data[group_base + 13],
                            group_data[group_base + 14],
                            group_data[group_base + 15],
                            gradient_data,
                            stop_offsets,
                            stop_colors,
                            px,
                            py,
                        );
                        let idx = frag_count as usize;
                        frag_color_r[idx] = color[0];
                        frag_color_g[idx] = color[1];
                        frag_color_b[idx] = color[2];
                        frag_alpha[idx] = color[3] * w;
                        frag_group[idx] = group_id;
                        frag_is_stroke[idx] = u32::new(1);
                        frag_shape[idx] = shape_id;
                        frag_distance[idx] = dist;
                        frag_local_x[idx] = local[0];
                        frag_local_y[idx] = local[1];
                        frag_base_point[idx] = base_point;
                        frag_t[idx] = t;
                        frag_within[idx] = u32::new(1);
                        frag_count += u32::new(1);
                    }
                }
            }
        }

        if fill_kind != PAINT_NONE {
            if frag_count < u32::new(MAX_PREFILTER_FRAGMENTS as i64) {
                let mut local = Line::empty(2usize);
                let mut dist = big;
                let mut shape_id = u32::new(0);
                let mut base_point = u32::new(0);
                let mut t = zero;
                let found = compute_distance_group(
                    shape_data,
                    shape_xform,
                    shape_transform,
                    group_xform,
                    group_shape_xform,
                    group_data,
                    group_shapes,
                    curve_data,
                    group_bvh_bounds,
                    group_bvh_nodes,
                    group_bvh_indices,
                    group_bvh_meta,
                    group_id,
                    px,
                    py,
                    one,
                    &mut local,
                    &mut dist,
                    &mut shape_id,
                    &mut base_point,
                    &mut t,
                );
                let inside = fill_inside_group(
                    shape_data,
                    segment_data,
                    shape_bounds,
                    group_data,
                    group_xform,
                    group_shapes,
                    shape_xform,
                    curve_data,
                    group_bvh_bounds,
                    group_bvh_nodes,
                    group_bvh_indices,
                    group_bvh_meta,
                    path_bvh_bounds,
                    path_bvh_nodes,
                    path_bvh_indices,
                    path_bvh_meta,
                    group_id,
                    px,
                    py,
                    fill_kind,
                    fill_rule,
                );
                if found != u32::new(0) || inside != u32::new(0) {
                    if found == u32::new(0) {
                        dist = one;
                        shape_id = u32::new(0);
                        local[0] = zero;
                        local[1] = zero;
                        base_point = u32::new(0);
                        t = zero;
                    }
                    if inside == u32::new(0) {
                        dist = -dist;
                    }
                    let w = smoothstep_unit(dist);
                    if w > zero {
                        let color = paint_color(
                            fill_kind,
                            fill_index,
                            group_data[group_base + 8],
                            group_data[group_base + 9],
                            group_data[group_base + 10],
                            group_data[group_base + 11],
                            gradient_data,
                            stop_offsets,
                            stop_colors,
                            px,
                            py,
                        );
                        let idx = frag_count as usize;
                        frag_color_r[idx] = color[0];
                        frag_color_g[idx] = color[1];
                        frag_color_b[idx] = color[2];
                        frag_alpha[idx] = color[3] * w;
                        frag_group[idx] = group_id;
                        frag_is_stroke[idx] = u32::new(0);
                        frag_shape[idx] = shape_id;
                        frag_distance[idx] = dist;
                        frag_local_x[idx] = local[0];
                        frag_local_y[idx] = local[1];
                        frag_base_point[idx] = base_point;
                        frag_t[idx] = t;
                        frag_within[idx] = found;
                        frag_count += u32::new(1);
                    }
                }
            }
        }

        group_id += u32::new(1);
    }

    let mut out = Line::empty(4usize);
    if frag_count == u32::new(0) {
        out[0] = background[0];
        out[1] = background[1];
        out[2] = background[2];
        out[3] = background[3];
        accumulate_background_grad(
            d_background,
            d_background_image,
            has_background_image,
            pixel_index,
            d_color[0],
            d_color[1],
            d_color[2],
            d_color[3],
        );
    } else {
        let mut accum_r: Array<f32> = Array::new(MAX_PREFILTER_FRAGMENTS);
        let mut accum_g: Array<f32> = Array::new(MAX_PREFILTER_FRAGMENTS);
        let mut accum_b: Array<f32> = Array::new(MAX_PREFILTER_FRAGMENTS);
        let mut accum_a: Array<f32> = Array::new(MAX_PREFILTER_FRAGMENTS);

        let mut prev_r = background[0];
        let mut prev_g = background[1];
        let mut prev_b = background[2];
        let mut prev_a = background[3];

        let mut i = u32::new(0);
        while i < frag_count {
            let idx = i as usize;
            let new_r = frag_color_r[idx];
            let new_g = frag_color_g[idx];
            let new_b = frag_color_b[idx];
            let new_a = frag_alpha[idx];
            let blended_r = prev_r * (one - new_a) + new_r * new_a;
            let blended_g = prev_g * (one - new_a) + new_g * new_a;
            let blended_b = prev_b * (one - new_a) + new_b * new_a;
            let blended_a = prev_a * (one - new_a) + new_a;
            accum_r[idx] = blended_r;
            accum_g[idx] = blended_g;
            accum_b[idx] = blended_b;
            accum_a[idx] = blended_a;
            prev_r = blended_r;
            prev_g = blended_g;
            prev_b = blended_b;
            prev_a = blended_a;
            i += u32::new(1);
        }

        let last_idx = (frag_count - u32::new(1)) as usize;
        let mut final_r = accum_r[last_idx];
        let mut final_g = accum_g[last_idx];
        let mut final_b = accum_b[last_idx];
        let final_a = accum_a[last_idx];
        if final_a > f32::new(1.0e-6) {
            let inv = one / final_a;
            final_r *= inv;
            final_g *= inv;
            final_b *= inv;
        }

        out[0] = final_r;
        out[1] = final_g;
        out[2] = final_b;
        out[3] = final_a;

        let mut d_curr_r = d_color[0];
        let mut d_curr_g = d_color[1];
        let mut d_curr_b = d_color[2];
        let mut d_curr_a = d_color[3];
        if final_a > f32::new(1.0e-6) {
            let d_final_r = d_curr_r;
            let d_final_g = d_curr_g;
            let d_final_b = d_curr_b;
            let inv = one / final_a;
            d_curr_r *= inv;
            d_curr_g *= inv;
            d_curr_b *= inv;
            d_curr_a -= (d_final_r * final_r + d_final_g * final_g + d_final_b * final_b) * inv;
        }

        let mut ri = frag_count;
        while ri > u32::new(0) {
            ri -= u32::new(1);
            let idx = ri as usize;
            let prev_alpha = if ri > u32::new(0) {
                accum_a[(ri - u32::new(1)) as usize]
            } else {
                background[3]
            };
            let prev_r = if ri > u32::new(0) {
                accum_r[(ri - u32::new(1)) as usize]
            } else {
                background[0]
            };
            let prev_g = if ri > u32::new(0) {
                accum_g[(ri - u32::new(1)) as usize]
            } else {
                background[1]
            };
            let prev_b = if ri > u32::new(0) {
                accum_b[(ri - u32::new(1)) as usize]
            } else {
                background[2]
            };

            let frag_r = frag_color_r[idx];
            let frag_g = frag_color_g[idx];
            let frag_b = frag_color_b[idx];
            let frag_a = frag_alpha[idx];

            let d_prev_a = d_curr_a * (one - frag_a);
            let mut d_alpha_i = d_curr_a * (one - prev_alpha);
            d_alpha_i += d_curr_r * (frag_r - prev_r)
                + d_curr_g * (frag_g - prev_g)
                + d_curr_b * (frag_b - prev_b);
            let d_prev_r = d_curr_r * (one - frag_a);
            let d_prev_g = d_curr_g * (one - frag_a);
            let d_prev_b = d_curr_b * (one - frag_a);
            let d_color_i_r = d_curr_r * frag_a;
            let d_color_i_g = d_curr_g * frag_a;
            let d_color_i_b = d_curr_b * frag_a;

            let group_id = frag_group[idx];
            let group_base = (group_id * GROUP_STRIDE) as usize;

            if frag_is_stroke[idx] != u32::new(0) {
                let shape_id = frag_shape[idx];
                let shape_base = (shape_id * SHAPE_STRIDE) as usize;
                let stroke_width = shape_data[shape_base + 3];
                let dist = frag_distance[idx];
                let abs_d = abs_f32(dist);
                let abs_plus = abs_d + stroke_width;
                let abs_minus = abs_d - stroke_width;
                let w = smoothstep_unit(abs_plus) - smoothstep_unit(abs_minus);
                if w != zero {
                    let d_w = if w > zero { (frag_a / w) * d_alpha_i } else { zero };
                    let d_alpha_i = d_alpha_i * w;
                    let stroke_kind = group_data[group_base + 4] as u32;
                    let stroke_index = group_data[group_base + 5] as u32;
                    d_sample_paint(
                        stroke_kind,
                        stroke_index,
                        group_data[group_base + 12],
                        group_data[group_base + 13],
                        group_data[group_base + 14],
                        group_data[group_base + 15],
                        gradient_data,
                        stop_offsets,
                        stop_colors,
                        px,
                        py,
                        {
                            let mut tmp = Line::empty(4usize);
                            tmp[0] = d_color_i_r;
                            tmp[1] = d_color_i_g;
                            tmp[2] = d_color_i_b;
                            tmp[3] = d_alpha_i;
                            tmp
                        },
                        (group_base + 12) as u32,
                        d_group_data,
                        d_gradient_data,
                        d_stop_offsets,
                        d_stop_colors,
                        translation_flag,
                        d_translation,
                        pixel_index,
                    );

                    let d_abs_plus = d_smoothstep_unit(abs_plus, d_w);
                    let d_abs_minus = -d_smoothstep_unit(abs_minus, d_w);
                    let mut d_d = d_abs_plus + d_abs_minus;
                    if dist < zero {
                        d_d = -d_d;
                    }
                    let d_stroke_width = d_abs_plus - d_abs_minus;
                    d_shape_stroke_width[shape_id as usize].fetch_add(d_stroke_width);

                    if abs_f32(d_d) > f32::new(1.0e-10) {
                        d_compute_distance(
                            shape_data,
                            shape_xform,
                            shape_transform,
                            group_xform,
                            group_shape_xform,
                            path_points,
                            path_num_controls,
                            shape_path_offsets,
                            shape_path_point_counts,
                            shape_path_ctrl_offsets,
                            shape_path_is_closed,
                            shape_id,
                            group_id,
                            px,
                            py,
                            frag_local_x[idx],
                            frag_local_y[idx],
                            frag_base_point[idx],
                            frag_t[idx],
                            d_d,
                            d_shape_params,
                            d_shape_points,
                            d_shape_transform,
                            d_group_transform,
                            d_translation,
                            translation_flag,
                            pixel_index,
                        );
                    }
                }
            } else {
                let dist = frag_distance[idx];
                let w = smoothstep_unit(dist);
                if w != zero {
                    let d_w = if w > zero { (frag_a / w) * d_alpha_i } else { zero };
                    let d_alpha_i = d_alpha_i * w;
                    let fill_kind = group_data[group_base + 2] as u32;
                    let fill_index = group_data[group_base + 3] as u32;
                    d_sample_paint(
                        fill_kind,
                        fill_index,
                        group_data[group_base + 8],
                        group_data[group_base + 9],
                        group_data[group_base + 10],
                        group_data[group_base + 11],
                        gradient_data,
                        stop_offsets,
                        stop_colors,
                        px,
                        py,
                        {
                            let mut tmp = Line::empty(4usize);
                            tmp[0] = d_color_i_r;
                            tmp[1] = d_color_i_g;
                            tmp[2] = d_color_i_b;
                            tmp[3] = d_alpha_i;
                            tmp
                        },
                        (group_base + 8) as u32,
                        d_group_data,
                        d_gradient_data,
                        d_stop_offsets,
                        d_stop_colors,
                        translation_flag,
                        d_translation,
                        pixel_index,
                    );

                    let mut d_d = d_smoothstep_unit(dist, d_w);
                    if dist < zero {
                        d_d = -d_d;
                    }
                    if abs_f32(d_d) > f32::new(1.0e-10) && frag_within[idx] != u32::new(0) {
                        let shape_id = frag_shape[idx];
                        d_compute_distance(
                            shape_data,
                            shape_xform,
                            shape_transform,
                            group_xform,
                            group_shape_xform,
                            path_points,
                            path_num_controls,
                            shape_path_offsets,
                            shape_path_point_counts,
                            shape_path_ctrl_offsets,
                            shape_path_is_closed,
                            shape_id,
                            group_id,
                            px,
                            py,
                            frag_local_x[idx],
                            frag_local_y[idx],
                            frag_base_point[idx],
                            frag_t[idx],
                            d_d,
                            d_shape_params,
                            d_shape_points,
                            d_shape_transform,
                            d_group_transform,
                            d_translation,
                            translation_flag,
                            pixel_index,
                        );
                    }
                }
            }

            d_curr_r = d_prev_r;
            d_curr_g = d_prev_g;
            d_curr_b = d_prev_b;
            d_curr_a = d_prev_a;
        }

        accumulate_background_grad(
            d_background,
            d_background_image,
            has_background_image,
            pixel_index,
            d_curr_r,
            d_curr_g,
            d_curr_b,
            d_curr_a,
        );
    }

    out
}

/// Backward SDF sampling; picks closest shape and propagates signed distance.
#[cube]
pub(super) fn sample_distance_sdf(
    shape_data: &Array<f32>,
    shape_xform: &Array<f32>,
    shape_transform: &Array<f32>,
    group_xform: &Array<f32>,
    group_shape_xform: &Array<f32>,
    group_data: &Array<f32>,
    group_shapes: &Array<f32>,
    curve_data: &Array<f32>,
    group_bvh_bounds: &Array<f32>,
    group_bvh_nodes: &Array<u32>,
    group_bvh_indices: &Array<u32>,
    group_bvh_meta: &Array<u32>,
    shape_bounds: &Array<f32>,
    segment_data: &Array<f32>,
    path_bvh_bounds: &Array<f32>,
    path_bvh_nodes: &Array<u32>,
    path_bvh_indices: &Array<u32>,
    path_bvh_meta: &Array<u32>,
    path_points: &Array<f32>,
    path_num_controls: &Array<u32>,
    shape_path_offsets: &Array<u32>,
    shape_path_point_counts: &Array<u32>,
    shape_path_ctrl_offsets: &Array<u32>,
    shape_path_is_closed: &Array<u32>,
    num_groups: u32,
    px: f32,
    py: f32,
    d_dist: f32,
    translation_flag: u32,
    pixel_index: u32,
    d_shape_params: &mut Array<Atomic<f32>>,
    d_shape_points: &mut Array<Atomic<f32>>,
    d_shape_transform: &mut Array<Atomic<f32>>,
    d_group_transform: &mut Array<Atomic<f32>>,
    d_translation: &mut Array<Atomic<f32>>,
) {
    let big = f32::new(1.0e20);
    let mut best_dist = big;
    let mut best_shape = u32::new(0);
    let mut best_group = u32::new(0);
    let mut best_local = Line::empty(2usize);
    let mut best_base = u32::new(0);
    let mut best_t = f32::new(0.0);
    let mut found = u32::new(0);

    let mut group_id = num_groups;
    while group_id > u32::new(0) {
        group_id -= u32::new(1);
        let mut local = Line::empty(2usize);
        let mut dist = best_dist;
        let mut shape_id = u32::new(0);
        let mut base_point = u32::new(0);
        let mut t = f32::new(0.0);
        let hit = compute_distance_group(
            shape_data,
            shape_xform,
            shape_transform,
            group_xform,
            group_shape_xform,
            group_data,
            group_shapes,
            curve_data,
            group_bvh_bounds,
            group_bvh_nodes,
            group_bvh_indices,
            group_bvh_meta,
            group_id,
            px,
            py,
            best_dist,
            &mut local,
            &mut dist,
            &mut shape_id,
            &mut base_point,
            &mut t,
        );
        if hit != u32::new(0) && dist < best_dist {
            best_dist = dist;
            best_shape = shape_id;
            best_group = group_id;
            best_local[0] = local[0];
            best_local[1] = local[1];
            best_base = base_point;
            best_t = t;
            found = u32::new(1);
        }
    }

    if found != u32::new(0) {
        let group_base = (best_group * GROUP_STRIDE) as usize;
        let fill_kind = group_data[group_base + 2] as u32;
        let fill_rule = group_data[group_base + 7] as u32;
        let inside = if fill_kind != PAINT_NONE {
            fill_inside_group(
                shape_data,
                segment_data,
                shape_bounds,
                group_data,
                group_xform,
                group_shapes,
                shape_xform,
                curve_data,
                group_bvh_bounds,
                group_bvh_nodes,
                group_bvh_indices,
                group_bvh_meta,
                path_bvh_bounds,
                path_bvh_nodes,
                path_bvh_indices,
                path_bvh_meta,
                best_group,
                px,
                py,
                fill_kind,
                fill_rule,
            )
        } else {
            u32::new(0)
        };

        let d_abs = if inside != u32::new(0) { -d_dist } else { d_dist };
        d_compute_distance(
            shape_data,
            shape_xform,
            shape_transform,
            group_xform,
            group_shape_xform,
            path_points,
            path_num_controls,
            shape_path_offsets,
            shape_path_point_counts,
            shape_path_ctrl_offsets,
            shape_path_is_closed,
            best_shape,
            best_group,
            px,
            py,
            best_local[0],
            best_local[1],
            best_base,
            best_t,
            d_abs,
            d_shape_params,
            d_shape_points,
            d_shape_transform,
            d_group_transform,
            d_translation,
            translation_flag,
            pixel_index,
        );
    }
}

/// Kernel for color backward pass without prefiltering.
#[cube(launch_unchecked)]
pub(crate) fn render_backward_color_kernel(
    shape_data: &Array<f32>,
    segment_data: &Array<f32>,
    shape_bounds: &Array<f32>,
    group_data: &Array<f32>,
    group_xform: &Array<f32>,
    group_shapes: &Array<f32>,
    shape_xform: &Array<f32>,
    curve_data: &Array<f32>,
    gradient_data: &Array<f32>,
    stop_offsets: &Array<f32>,
    stop_colors: &Array<f32>,
    group_bvh_bounds: &Array<f32>,
    group_bvh_nodes: &Array<u32>,
    group_bvh_indices: &Array<u32>,
    group_bvh_meta: &Array<u32>,
    path_bvh_bounds: &Array<f32>,
    path_bvh_nodes: &Array<u32>,
    path_bvh_indices: &Array<u32>,
    path_bvh_meta: &Array<u32>,
    width: u32,
    height: u32,
    num_groups: u32,
    samples_x: u32,
    samples_y: u32,
    seed: u32,
    jitter: u32,
    filter_type: u32,
    filter_radius: f32,
    background_image: &Array<f32>,
    has_background_image: u32,
    background_r: f32,
    background_g: f32,
    background_b: f32,
    background_a: f32,
    weight_image: &Array<Atomic<f32>>,
    d_render_image: &Array<f32>,
    translation_flag: u32,
    d_group_data: &mut Array<Atomic<f32>>,
    d_gradient_data: &mut Array<Atomic<f32>>,
    d_stop_offsets: &mut Array<Atomic<f32>>,
    d_stop_colors: &mut Array<Atomic<f32>>,
    d_filter_radius: &mut Array<Atomic<f32>>,
    d_background: &mut Array<Atomic<f32>>,
    d_background_image: &mut Array<Atomic<f32>>,
    d_translation: &mut Array<Atomic<f32>>,
) {
    let idx = ABSOLUTE_POS;
    let samples_per_pixel = samples_x * samples_y;
    if samples_per_pixel == u32::new(0) {
        terminate!();
    }
    let total_samples = width * height * samples_per_pixel;
    if idx >= total_samples as usize {
        terminate!();
    }

    let idx_u32 = idx as u32;
    let sx = idx_u32 % samples_x;
    let sy = (idx_u32 / samples_x) % samples_y;
    let x = (idx_u32 / (samples_x * samples_y)) % width;
    let y = idx_u32 / (samples_x * samples_y * width);

    let inv_sx = f32::new(1.0) / f32::cast_from(samples_x);
    let inv_sy = f32::new(1.0) / f32::cast_from(samples_y);

    let half = f32::new(0.5);
    let mut rx = half;
    let mut ry = half;
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
        rx = pcg32_f32(step0[0]);
        let step1 = pcg32_next(state_lo, state_hi, inc_lo, inc_hi);
        ry = pcg32_f32(step1[0]);
    }

    let px = f32::cast_from(x) + (f32::cast_from(sx) + rx) * inv_sx;
    let py = f32::cast_from(y) + (f32::cast_from(sy) + ry) * inv_sy;
    let pixel_index = y * width + x;

    let mut bg = Line::empty(4usize);
    if has_background_image != u32::new(0) {
        let idx4 = (pixel_index as usize) * 4;
        bg[0] = background_image[idx4];
        bg[1] = background_image[idx4 + 1];
        bg[2] = background_image[idx4 + 2];
        bg[3] = background_image[idx4 + 3];
    } else {
        bg[0] = background_r;
        bg[1] = background_g;
        bg[2] = background_b;
        bg[3] = background_a;
    }

    let d_color = gather_d_color(
        filter_type,
        filter_radius,
        d_render_image,
        weight_image,
        width,
        height,
        px,
        py,
    );
    let color = sample_color(
        shape_data,
        segment_data,
        shape_bounds,
        group_data,
        group_xform,
        group_shapes,
        shape_xform,
        curve_data,
        gradient_data,
        stop_offsets,
        stop_colors,
        group_bvh_bounds,
        group_bvh_nodes,
        group_bvh_indices,
        group_bvh_meta,
        path_bvh_bounds,
        path_bvh_nodes,
        path_bvh_indices,
        path_bvh_meta,
        num_groups,
        px,
        py,
        bg,
        d_color,
        has_background_image,
        translation_flag,
        pixel_index,
        d_group_data,
        d_gradient_data,
        d_stop_offsets,
        d_stop_colors,
        d_background,
        d_background_image,
        d_translation,
    );
    accumulate_filter_gradient(
        filter_type,
        filter_radius,
        color,
        d_render_image,
        weight_image,
        width,
        height,
        px,
        py,
        d_filter_radius,
    );
}

/// Kernel for color backward pass with prefiltering and shape gradients.
#[cube(launch_unchecked)]
pub(crate) fn render_backward_color_prefilter_kernel(
    shape_data: &Array<f32>,
    segment_data: &Array<f32>,
    shape_bounds: &Array<f32>,
    group_data: &Array<f32>,
    group_xform: &Array<f32>,
    group_shape_xform: &Array<f32>,
    group_shapes: &Array<f32>,
    shape_xform: &Array<f32>,
    shape_transform: &Array<f32>,
    curve_data: &Array<f32>,
    gradient_data: &Array<f32>,
    stop_offsets: &Array<f32>,
    stop_colors: &Array<f32>,
    group_bvh_bounds: &Array<f32>,
    group_bvh_nodes: &Array<u32>,
    group_bvh_indices: &Array<u32>,
    group_bvh_meta: &Array<u32>,
    path_bvh_bounds: &Array<f32>,
    path_bvh_nodes: &Array<u32>,
    path_bvh_indices: &Array<u32>,
    path_bvh_meta: &Array<u32>,
    path_points: &Array<f32>,
    path_num_controls: &Array<u32>,
    shape_path_offsets: &Array<u32>,
    shape_path_point_counts: &Array<u32>,
    shape_path_ctrl_offsets: &Array<u32>,
    shape_path_is_closed: &Array<u32>,
    width: u32,
    height: u32,
    num_groups: u32,
    samples_x: u32,
    samples_y: u32,
    seed: u32,
    jitter: u32,
    filter_type: u32,
    filter_radius: f32,
    background_image: &Array<f32>,
    has_background_image: u32,
    background_r: f32,
    background_g: f32,
    background_b: f32,
    background_a: f32,
    weight_image: &Array<Atomic<f32>>,
    d_render_image: &Array<f32>,
    translation_flag: u32,
    d_shape_params: &mut Array<Atomic<f32>>,
    d_shape_points: &mut Array<Atomic<f32>>,
    d_shape_stroke_width: &mut Array<Atomic<f32>>,
    d_shape_transform: &mut Array<Atomic<f32>>,
    d_group_transform: &mut Array<Atomic<f32>>,
    d_group_data: &mut Array<Atomic<f32>>,
    d_gradient_data: &mut Array<Atomic<f32>>,
    d_stop_offsets: &mut Array<Atomic<f32>>,
    d_stop_colors: &mut Array<Atomic<f32>>,
    d_filter_radius: &mut Array<Atomic<f32>>,
    d_background: &mut Array<Atomic<f32>>,
    d_background_image: &mut Array<Atomic<f32>>,
    d_translation: &mut Array<Atomic<f32>>,
) {
    let idx = ABSOLUTE_POS;
    let samples_per_pixel = samples_x * samples_y;
    if samples_per_pixel == u32::new(0) {
        terminate!();
    }
    let total_samples = width * height * samples_per_pixel;
    if idx >= total_samples as usize {
        terminate!();
    }

    let idx_u32 = idx as u32;
    let sx = idx_u32 % samples_x;
    let sy = (idx_u32 / samples_x) % samples_y;
    let x = (idx_u32 / (samples_x * samples_y)) % width;
    let y = idx_u32 / (samples_x * samples_y * width);

    let inv_sx = f32::new(1.0) / f32::cast_from(samples_x);
    let inv_sy = f32::new(1.0) / f32::cast_from(samples_y);

    let half = f32::new(0.5);
    let mut rx = half;
    let mut ry = half;
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
        rx = pcg32_f32(step0[0]);
        let step1 = pcg32_next(state_lo, state_hi, inc_lo, inc_hi);
        ry = pcg32_f32(step1[0]);
    }

    let px = f32::cast_from(x) + (f32::cast_from(sx) + rx) * inv_sx;
    let py = f32::cast_from(y) + (f32::cast_from(sy) + ry) * inv_sy;
    let pixel_index = y * width + x;

    let mut bg = Line::empty(4usize);
    if has_background_image != u32::new(0) {
        let idx4 = (pixel_index as usize) * 4;
        bg[0] = background_image[idx4];
        bg[1] = background_image[idx4 + 1];
        bg[2] = background_image[idx4 + 2];
        bg[3] = background_image[idx4 + 3];
    } else {
        bg[0] = background_r;
        bg[1] = background_g;
        bg[2] = background_b;
        bg[3] = background_a;
    }

    let d_color = gather_d_color(
        filter_type,
        filter_radius,
        d_render_image,
        weight_image,
        width,
        height,
        px,
        py,
    );
    let color = sample_color_prefiltered(
        shape_data,
        segment_data,
        shape_bounds,
        group_data,
        group_xform,
        group_shape_xform,
        group_shapes,
        shape_xform,
        shape_transform,
        curve_data,
        gradient_data,
        stop_offsets,
        stop_colors,
        group_bvh_bounds,
        group_bvh_nodes,
        group_bvh_indices,
        group_bvh_meta,
        path_bvh_bounds,
        path_bvh_nodes,
        path_bvh_indices,
        path_bvh_meta,
        path_points,
        path_num_controls,
        shape_path_offsets,
        shape_path_point_counts,
        shape_path_ctrl_offsets,
        shape_path_is_closed,
        num_groups,
        px,
        py,
        bg,
        d_color,
        has_background_image,
        translation_flag,
        pixel_index,
        d_shape_params,
        d_shape_points,
        d_shape_stroke_width,
        d_shape_transform,
        d_group_transform,
        d_group_data,
        d_gradient_data,
        d_stop_offsets,
        d_stop_colors,
        d_background,
        d_background_image,
        d_translation,
    );
    accumulate_filter_gradient(
        filter_type,
        filter_radius,
        color,
        d_render_image,
        weight_image,
        width,
        height,
        px,
        py,
        d_filter_radius,
    );
}

/// Kernel for SDF backward pass over samples.
#[cube(launch_unchecked)]
pub(crate) fn render_backward_sdf_kernel(
    shape_data: &Array<f32>,
    segment_data: &Array<f32>,
    shape_bounds: &Array<f32>,
    group_data: &Array<f32>,
    group_xform: &Array<f32>,
    group_shape_xform: &Array<f32>,
    group_shapes: &Array<f32>,
    shape_xform: &Array<f32>,
    shape_transform: &Array<f32>,
    curve_data: &Array<f32>,
    group_bvh_bounds: &Array<f32>,
    group_bvh_nodes: &Array<u32>,
    group_bvh_indices: &Array<u32>,
    group_bvh_meta: &Array<u32>,
    path_bvh_bounds: &Array<f32>,
    path_bvh_nodes: &Array<u32>,
    path_bvh_indices: &Array<u32>,
    path_bvh_meta: &Array<u32>,
    path_points: &Array<f32>,
    path_num_controls: &Array<u32>,
    shape_path_offsets: &Array<u32>,
    shape_path_point_counts: &Array<u32>,
    shape_path_ctrl_offsets: &Array<u32>,
    shape_path_is_closed: &Array<u32>,
    width: u32,
    height: u32,
    num_groups: u32,
    samples_x: u32,
    samples_y: u32,
    seed: u32,
    jitter: u32,
    d_sdf_image: &Array<f32>,
    translation_flag: u32,
    d_shape_params: &mut Array<Atomic<f32>>,
    d_shape_points: &mut Array<Atomic<f32>>,
    d_shape_transform: &mut Array<Atomic<f32>>,
    d_group_transform: &mut Array<Atomic<f32>>,
    d_translation: &mut Array<Atomic<f32>>,
) {
    let idx = ABSOLUTE_POS;
    let samples_per_pixel = samples_x * samples_y;
    if samples_per_pixel == u32::new(0) {
        terminate!();
    }
    let total_samples = width * height * samples_per_pixel;
    if idx >= total_samples as usize {
        terminate!();
    }

    let idx_u32 = idx as u32;
    let sx = idx_u32 % samples_x;
    let sy = (idx_u32 / samples_x) % samples_y;
    let x = (idx_u32 / (samples_x * samples_y)) % width;
    let y = idx_u32 / (samples_x * samples_y * width);

    let inv_sx = f32::new(1.0) / f32::cast_from(samples_x);
    let inv_sy = f32::new(1.0) / f32::cast_from(samples_y);

    let half = f32::new(0.5);
    let mut rx = half;
    let mut ry = half;
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
        rx = pcg32_f32(step0[0]);
        let step1 = pcg32_next(state_lo, state_hi, inc_lo, inc_hi);
        ry = pcg32_f32(step1[0]);
    }

    let px = f32::cast_from(x) + (f32::cast_from(sx) + rx) * inv_sx;
    let py = f32::cast_from(y) + (f32::cast_from(sy) + ry) * inv_sy;
    let pixel_index = y * width + x;

    let inv_samples = if samples_per_pixel > u32::new(0) {
        f32::new(1.0) / f32::cast_from(samples_per_pixel)
    } else {
        f32::new(1.0)
    };
    let d_dist = d_sdf_image[pixel_index as usize] * inv_samples;
    if d_dist != f32::new(0.0) {
        sample_distance_sdf(
            shape_data,
            shape_xform,
            shape_transform,
            group_xform,
            group_shape_xform,
            group_data,
            group_shapes,
            curve_data,
            group_bvh_bounds,
            group_bvh_nodes,
            group_bvh_indices,
            group_bvh_meta,
            shape_bounds,
            segment_data,
            path_bvh_bounds,
            path_bvh_nodes,
            path_bvh_indices,
            path_bvh_meta,
            path_points,
            path_num_controls,
            shape_path_offsets,
            shape_path_point_counts,
            shape_path_ctrl_offsets,
            shape_path_is_closed,
            num_groups,
            px,
            py,
            d_dist,
            translation_flag,
            pixel_index,
            d_shape_params,
            d_shape_points,
            d_shape_transform,
            d_group_transform,
            d_translation,
        );
    }
}

/// Combined kernel handling color and/or SDF gradients based on flags.
#[cube(launch_unchecked)]
pub(crate) fn render_backward_kernel(
    shape_data: &Array<f32>,
    segment_data: &Array<f32>,
    shape_bounds: &Array<f32>,
    group_data: &Array<f32>,
    group_xform: &Array<f32>,
    group_shape_xform: &Array<f32>,
    group_shapes: &Array<f32>,
    shape_xform: &Array<f32>,
    shape_transform: &Array<f32>,
    curve_data: &Array<f32>,
    gradient_data: &Array<f32>,
    stop_offsets: &Array<f32>,
    stop_colors: &Array<f32>,
    group_bvh_bounds: &Array<f32>,
    group_bvh_nodes: &Array<u32>,
    group_bvh_indices: &Array<u32>,
    group_bvh_meta: &Array<u32>,
    path_bvh_bounds: &Array<f32>,
    path_bvh_nodes: &Array<u32>,
    path_bvh_indices: &Array<u32>,
    path_bvh_meta: &Array<u32>,
    path_points: &Array<f32>,
    path_num_controls: &Array<u32>,
    _path_thickness: &Array<f32>,
    shape_path_offsets: &Array<u32>,
    shape_path_point_counts: &Array<u32>,
    shape_path_ctrl_offsets: &Array<u32>,
    _shape_path_ctrl_counts: &Array<u32>,
    _shape_path_thickness_offsets: &Array<u32>,
    _shape_path_thickness_counts: &Array<u32>,
    shape_path_is_closed: &Array<u32>,
    width: u32,
    height: u32,
    num_groups: u32,
    samples_x: u32,
    samples_y: u32,
    seed: u32,
    jitter: u32,
    use_prefiltering: u32,
    filter_type: u32,
    filter_radius: f32,
    _filter_radius_i: u32,
    background_image: &Array<f32>,
    has_background_image: u32,
    background_r: f32,
    background_g: f32,
    background_b: f32,
    background_a: f32,
    weight_image: &Array<Atomic<f32>>,
    d_render_image: &Array<f32>,
    d_sdf_image: &Array<f32>,
    render_grad_flag: u32,
    sdf_grad_flag: u32,
    translation_flag: u32,
    d_shape_params: &mut Array<Atomic<f32>>,
    d_shape_points: &mut Array<Atomic<f32>>,
    _d_shape_thickness: &mut Array<Atomic<f32>>,
    d_shape_stroke_width: &mut Array<Atomic<f32>>,
    d_shape_transform: &mut Array<Atomic<f32>>,
    d_group_transform: &mut Array<Atomic<f32>>,
    d_group_data: &mut Array<Atomic<f32>>,
    d_gradient_data: &mut Array<Atomic<f32>>,
    d_stop_offsets: &mut Array<Atomic<f32>>,
    d_stop_colors: &mut Array<Atomic<f32>>,
    d_filter_radius: &mut Array<Atomic<f32>>,
    d_background: &mut Array<Atomic<f32>>,
    d_background_image: &mut Array<Atomic<f32>>,
    d_translation: &mut Array<Atomic<f32>>,
) {
    let idx = ABSOLUTE_POS;
    let samples_per_pixel = samples_x * samples_y;
    if samples_per_pixel == u32::new(0) {
        terminate!();
    }
    let total_samples = width * height * samples_per_pixel;
    if idx >= total_samples as usize {
        terminate!();
    }

    let idx_u32 = idx as u32;
    let sx = idx_u32 % samples_x;
    let sy = (idx_u32 / samples_x) % samples_y;
    let x = (idx_u32 / (samples_x * samples_y)) % width;
    let y = idx_u32 / (samples_x * samples_y * width);

    let inv_sx = f32::new(1.0) / f32::cast_from(samples_x);
    let inv_sy = f32::new(1.0) / f32::cast_from(samples_y);

    let half = f32::new(0.5);
    let mut rx = half;
    let mut ry = half;
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
        rx = pcg32_f32(step0[0]);
        let step1 = pcg32_next(state_lo, state_hi, inc_lo, inc_hi);
        ry = pcg32_f32(step1[0]);
    }

    let px = f32::cast_from(x) + (f32::cast_from(sx) + rx) * inv_sx;
    let py = f32::cast_from(y) + (f32::cast_from(sy) + ry) * inv_sy;
    let pixel_index = y * width + x;

    let mut bg = Line::empty(4usize);
    if has_background_image != u32::new(0) {
        let idx4 = (pixel_index as usize) * 4;
        bg[0] = background_image[idx4];
        bg[1] = background_image[idx4 + 1];
        bg[2] = background_image[idx4 + 2];
        bg[3] = background_image[idx4 + 3];
    } else {
        bg[0] = background_r;
        bg[1] = background_g;
        bg[2] = background_b;
        bg[3] = background_a;
    }

    if render_grad_flag != u32::new(0) {
        let d_color = gather_d_color(
            filter_type,
            filter_radius,
            d_render_image,
            weight_image,
            width,
            height,
            px,
            py,
        );
        let color = if use_prefiltering != u32::new(0) {
            sample_color_prefiltered(
                shape_data,
                segment_data,
                shape_bounds,
                group_data,
                group_xform,
                group_shape_xform,
                group_shapes,
                shape_xform,
                shape_transform,
                curve_data,
                gradient_data,
                stop_offsets,
                stop_colors,
                group_bvh_bounds,
                group_bvh_nodes,
                group_bvh_indices,
                group_bvh_meta,
                path_bvh_bounds,
                path_bvh_nodes,
                path_bvh_indices,
                path_bvh_meta,
                path_points,
                path_num_controls,
                shape_path_offsets,
                shape_path_point_counts,
                shape_path_ctrl_offsets,
                shape_path_is_closed,
                num_groups,
                px,
                py,
                bg,
                d_color,
                has_background_image,
                translation_flag,
                pixel_index,
                d_shape_params,
                d_shape_points,
                d_shape_stroke_width,
                d_shape_transform,
                d_group_transform,
                d_group_data,
                d_gradient_data,
                d_stop_offsets,
                d_stop_colors,
                d_background,
                d_background_image,
                d_translation,
            )
        } else {
            sample_color(
                shape_data,
                segment_data,
                shape_bounds,
                group_data,
                group_xform,
                group_shapes,
                shape_xform,
                curve_data,
                gradient_data,
                stop_offsets,
                stop_colors,
                group_bvh_bounds,
                group_bvh_nodes,
                group_bvh_indices,
                group_bvh_meta,
                path_bvh_bounds,
                path_bvh_nodes,
                path_bvh_indices,
                path_bvh_meta,
                num_groups,
                px,
                py,
                bg,
                d_color,
                has_background_image,
                translation_flag,
                pixel_index,
                d_group_data,
                d_gradient_data,
                d_stop_offsets,
                d_stop_colors,
                d_background,
                d_background_image,
                d_translation,
            )
        };
        accumulate_filter_gradient(
            filter_type,
            filter_radius,
            color,
            d_render_image,
            weight_image,
            width,
            height,
            px,
            py,
            d_filter_radius,
        );
    }

    if sdf_grad_flag != u32::new(0) {
        let d_dist = d_sdf_image[pixel_index as usize];
        sample_distance_sdf(
            shape_data,
            shape_xform,
            shape_transform,
            group_xform,
            group_shape_xform,
            group_data,
            group_shapes,
            curve_data,
            group_bvh_bounds,
            group_bvh_nodes,
            group_bvh_indices,
            group_bvh_meta,
            shape_bounds,
            segment_data,
            path_bvh_bounds,
            path_bvh_nodes,
            path_bvh_indices,
            path_bvh_meta,
            path_points,
            path_num_controls,
            shape_path_offsets,
            shape_path_point_counts,
            shape_path_ctrl_offsets,
            shape_path_is_closed,
            num_groups,
            px,
            py,
            d_dist,
            translation_flag,
            pixel_index,
            d_shape_params,
            d_shape_points,
            d_shape_transform,
            d_group_transform,
            d_translation,
        );
    }
}
