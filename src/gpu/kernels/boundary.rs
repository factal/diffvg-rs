//! GPU boundary sampling for non-prefiltered backward pass.

use cubecl::prelude::*;
use crate::gpu::constants::*;
use super::{
    math::*,
    rng::*,
    sampling::paint_color,
    forward::{accumulate_group_shapes, accumulate_shape_in_group},
    backward::{
        d_xform_pt_affine,
        affine_grad_to_mat3,
        mat3_from_affine,
        mat3_transpose,
        mat3_mul,
        atomic_add_mat3,
        add_translation,
        gather_d_color,
        path_point_index,
        load_path_point,
        add_path_point_grad,
    },
};

/// Upper bound on per-pixel fragments stored during edge compositing.
const MAX_FRAGMENTS: usize = 256;

/// Transform a normal by the transpose of a 2x2 affine matrix and renormalize.
#[cube]
fn xform_normal_affine(m00: f32, m01: f32, m10: f32, m11: f32, nx: f32, ny: f32) -> Line<f32> {
    let x = m00 * nx + m10 * ny;
    let y = m01 * nx + m11 * ny;
    vec2_normalize(x, y)
}

/// Sample a CDF, returning the selected index and local t in [0, 1].
#[cube]
fn sample_cdf(cdf: &Array<f32>, offset: u32, count: u32, u: f32, out_t: &mut f32) -> u32 {
    let mut idx = u32::new(0);
    if count == u32::new(0) {
        *out_t = f32::new(0.0);
    } else {
        while idx < count && u > cdf[(offset + idx) as usize] {
            idx += u32::new(1);
        }
        if idx >= count {
            idx = count - u32::new(1);
        }
        let prev = if idx == u32::new(0) {
            f32::new(0.0)
        } else {
            cdf[(offset + idx - u32::new(1)) as usize]
        };
        let curr = cdf[(offset + idx) as usize];
        let denom = max_f32(curr - prev, f32::new(1.0e-6));
        let mut t = (u - prev) / denom;
        if t < f32::new(0.0) {
            t = f32::new(0.0);
        } else if t > f32::new(1.0) {
            t = f32::new(1.0);
        }
        *out_t = t;
    }
    idx
}

/// Fetch a per-point stroke radius, handling open/closed paths and missing thickness data.
#[cube]
fn load_path_radius(
    path_thickness: &Array<f32>,
    thickness_offset: u32,
    thickness_count: u32,
    point_index: u32,
    point_count: u32,
    is_closed: u32,
    default_radius: f32,
) -> f32 {
    let mut out = default_radius;
    if thickness_count != u32::new(0) && point_count != u32::new(0) {
        let idx = if is_closed != u32::new(0) {
            point_index % point_count
        } else if point_index >= point_count {
            point_count - u32::new(1)
        } else {
            point_index
        };
        let t_idx = thickness_offset + idx;
        if t_idx < thickness_offset + thickness_count {
            out = path_thickness[t_idx as usize];
        }
    }
    out
}

/// Sample a path boundary point and normal using the path CDF and stroke settings.
/// Updates `base_point_id`, `point_id`, and `out_t` for gradient accumulation.
#[cube]
fn sample_boundary_path(
    path_points: &Array<f32>,
    path_num_controls: &Array<u32>,
    path_thickness: &Array<f32>,
    point_offset: u32,
    point_count: u32,
    ctrl_offset: u32,
    ctrl_count: u32,
    thickness_offset: u32,
    thickness_count: u32,
    is_closed: u32,
    path_cdf: &Array<f32>,
    path_pmf: &Array<f32>,
    path_point_ids: &Array<u32>,
    path_cdf_offset: u32,
    path_cdf_count: u32,
    path_point_offset: u32,
    path_length: f32,
    mut t: f32,
    stroke_direction: f32,
    stroke_radius: f32,
    normal: &mut Line<f32>,
    pdf: &mut f32,
    base_point_id: &mut u32,
    point_id: &mut u32,
    out_t: &mut f32,
) -> Line<f32> {
    let zero = f32::new(0.0);
    let two = f32::new(2.0);
    let pi = f32::new(3.14159265);
    let two_pi = two * pi;

    let mut out = Line::empty(2usize);
    let mut done = u32::new(0);

    if point_count == u32::new(0) || ctrl_count == u32::new(0) || path_cdf_count == u32::new(0) {
        *pdf = zero;
        done = u32::new(1);
    }

    if done == u32::new(0) && stroke_direction != zero && is_closed == u32::new(0) {
        let mut cap_length = zero;
        if thickness_count > u32::new(0) {
            let r0 = load_path_radius(
                path_thickness,
                thickness_offset,
                thickness_count,
                u32::new(0),
                point_count,
                is_closed,
                stroke_radius,
            );
            let r1 = load_path_radius(
                path_thickness,
                thickness_offset,
                thickness_count,
                point_count - u32::new(1),
                point_count,
                is_closed,
                stroke_radius,
            );
            cap_length = pi * (r0 + r1);
        } else {
            cap_length = two_pi * stroke_radius;
        }
        let denom = cap_length + path_length;
        if denom > zero {
            let cap_prob = cap_length / denom;
            if t < cap_prob {
                t = t / cap_prob;
                *pdf *= cap_prob;
                let mut r0 = stroke_radius;
                let mut r1 = stroke_radius;
                if thickness_count > u32::new(0) {
                    r0 = load_path_radius(
                        path_thickness,
                        thickness_offset,
                        thickness_count,
                        u32::new(0),
                        point_count,
                        is_closed,
                        stroke_radius,
                    );
                    r1 = load_path_radius(
                        path_thickness,
                        thickness_offset,
                        thickness_count,
                        point_count - u32::new(1),
                        point_count,
                        is_closed,
                        stroke_radius,
                    );
                }
                let angle = two_pi * t;
                if stroke_direction < zero {
                    let p0 = load_path_point(path_points, point_offset, u32::new(0));
                    let offset_x = r0 * angle.cos();
                    let offset_y = r0 * angle.sin();
                    let n = vec2_normalize(offset_x, offset_y);
                    normal[0] = n[0];
                    normal[1] = n[1];
                    *pdf /= two_pi * r0;
                    *base_point_id = u32::new(0);
                    *point_id = u32::new(0);
                    *out_t = zero;
                    out[0] = p0[0] + offset_x;
                    out[1] = p0[1] + offset_y;
                } else {
                    let last_point = point_count - u32::new(1);
                    let p0 = load_path_point(path_points, point_offset, last_point);
                    let offset_x = r1 * angle.cos();
                    let offset_y = r1 * angle.sin();
                    let n = vec2_normalize(offset_x, offset_y);
                    normal[0] = n[0];
                    normal[1] = n[1];
                    *pdf /= two_pi * r1;
                    *base_point_id = ctrl_count - u32::new(1);
                    let ctrl = path_num_controls[(ctrl_offset + *base_point_id) as usize];
                    let mut pid = point_count;
                    if pid >= u32::new(2) + ctrl {
                        pid = point_count - u32::new(2) - ctrl;
                    } else {
                        pid = u32::new(0);
                    }
                    *point_id = pid;
                    *out_t = f32::new(1.0);
                    out[0] = p0[0] + offset_x;
                    out[1] = p0[1] + offset_y;
                }
                done = u32::new(1);
            } else {
                t = (t - cap_prob) / (f32::new(1.0) - cap_prob);
                *pdf *= f32::new(1.0) - cap_prob;
            }
        }
    }

    if done == u32::new(0) {
        let mut local_t = f32::new(0.0);
        let sample_id = sample_cdf(path_cdf, path_cdf_offset, path_cdf_count, t, &mut local_t);
        let pid = path_point_ids[(path_point_offset + sample_id) as usize];
        *base_point_id = sample_id;
        *point_id = pid;
        *out_t = local_t;
        if local_t < f32::new(-1.0e-3) || local_t > f32::new(1.0) + f32::new(1.0e-3) {
            *pdf = zero;
            done = u32::new(1);
        }

        if done == u32::new(0) {
            let ctrl = path_num_controls[(ctrl_offset + sample_id) as usize];
            if ctrl == u32::new(0) {
                let i0 = pid;
                let i1 = path_point_index(pid + u32::new(1), point_count, is_closed);
                let p0 = load_path_point(path_points, point_offset, i0);
                let p1 = load_path_point(path_points, point_offset, i1);
                let vx = p1[0] - p0[0];
                let vy = p1[1] - p0[1];
                let tan_len = vec2_length(vx, vy);
                if tan_len == zero {
                    *pdf = zero;
                } else {
                    normal[0] = -vy / tan_len;
                    normal[1] = vx / tan_len;
                    *pdf *= path_pmf[(path_cdf_offset + sample_id) as usize] / tan_len;
                    out[0] = p0[0] + local_t * vx;
                    out[1] = p0[1] + local_t * vy;
                    if stroke_direction != zero {
                        let r0 = load_path_radius(
                            path_thickness,
                            thickness_offset,
                            thickness_count,
                            i0,
                            point_count,
                            is_closed,
                            stroke_radius,
                        );
                        let r1 = load_path_radius(
                            path_thickness,
                            thickness_offset,
                            thickness_count,
                            i1,
                            point_count,
                            is_closed,
                            stroke_radius,
                        );
                        let r = r0 + local_t * (r1 - r0);
                        out[0] += stroke_direction * r * normal[0];
                        out[1] += stroke_direction * r * normal[1];
                        if stroke_direction < zero {
                            normal[0] = -normal[0];
                            normal[1] = -normal[1];
                        }
                    }
                }
            } else if ctrl == u32::new(1) {
                let i0 = pid;
                let i1 = pid + u32::new(1);
                let i2 = path_point_index(pid + u32::new(2), point_count, is_closed);
                let p0 = load_path_point(path_points, point_offset, i0);
                let p1 = load_path_point(path_points, point_offset, i1);
                let p2 = load_path_point(path_points, point_offset, i2);
                let tt = f32::new(1.0) - local_t;
                let eval_x = p0[0] * (tt * tt) + p1[0] * (two * tt * local_t) + p2[0] * (local_t * local_t);
                let eval_y = p0[1] * (tt * tt) + p1[1] * (two * tt * local_t) + p2[1] * (local_t * local_t);
                let tan_x = (p1[0] - p0[0]) * (two * tt) + (p2[0] - p1[0]) * (two * local_t);
                let tan_y = (p1[1] - p0[1]) * (two * tt) + (p2[1] - p1[1]) * (two * local_t);
                let tan_len = vec2_length(tan_x, tan_y);
                if tan_len == zero {
                    *pdf = zero;
                } else {
                    normal[0] = -tan_y / tan_len;
                    normal[1] = tan_x / tan_len;
                    *pdf *= path_pmf[(path_cdf_offset + sample_id) as usize] / tan_len;
                    out[0] = eval_x;
                    out[1] = eval_y;
                    if stroke_direction != zero {
                        let r0 = load_path_radius(
                            path_thickness,
                            thickness_offset,
                            thickness_count,
                            i0,
                            point_count,
                            is_closed,
                            stroke_radius,
                        );
                        let r1 = load_path_radius(
                            path_thickness,
                            thickness_offset,
                            thickness_count,
                            i1,
                            point_count,
                            is_closed,
                            stroke_radius,
                        );
                        let r2 = load_path_radius(
                            path_thickness,
                            thickness_offset,
                            thickness_count,
                            i2,
                            point_count,
                            is_closed,
                            stroke_radius,
                        );
                        let r = r0 * (tt * tt) + r1 * (two * tt * local_t) + r2 * (local_t * local_t);
                        out[0] += stroke_direction * r * normal[0];
                        out[1] += stroke_direction * r * normal[1];
                        if stroke_direction < zero {
                            normal[0] = -normal[0];
                            normal[1] = -normal[1];
                        }
                    }
                }
            } else if ctrl == u32::new(2) {
                let i0 = pid;
                let i1 = pid + u32::new(1);
                let i2 = pid + u32::new(2);
                let i3 = path_point_index(pid + u32::new(3), point_count, is_closed);
                let p0 = load_path_point(path_points, point_offset, i0);
                let p1 = load_path_point(path_points, point_offset, i1);
                let p2 = load_path_point(path_points, point_offset, i2);
                let p3 = load_path_point(path_points, point_offset, i3);
                let tt = f32::new(1.0) - local_t;
                let eval_x = p0[0] * (tt * tt * tt)
                    + p1[0] * (f32::new(3.0) * tt * tt * local_t)
                    + p2[0] * (f32::new(3.0) * tt * local_t * local_t)
                    + p3[0] * (local_t * local_t * local_t);
                let eval_y = p0[1] * (tt * tt * tt)
                    + p1[1] * (f32::new(3.0) * tt * tt * local_t)
                    + p2[1] * (f32::new(3.0) * tt * local_t * local_t)
                    + p3[1] * (local_t * local_t * local_t);
                let tan_x = (p1[0] - p0[0]) * (f32::new(3.0) * tt * tt)
                    + (p2[0] - p1[0]) * (f32::new(6.0) * tt * local_t)
                    + (p3[0] - p2[0]) * (f32::new(3.0) * local_t * local_t);
                let tan_y = (p1[1] - p0[1]) * (f32::new(3.0) * tt * tt)
                    + (p2[1] - p1[1]) * (f32::new(6.0) * tt * local_t)
                    + (p3[1] - p2[1]) * (f32::new(3.0) * local_t * local_t);
                let tan_len = vec2_length(tan_x, tan_y);
                if tan_len == zero {
                    *pdf = zero;
                } else {
                    normal[0] = -tan_y / tan_len;
                    normal[1] = tan_x / tan_len;
                    *pdf *= path_pmf[(path_cdf_offset + sample_id) as usize] / tan_len;
                    out[0] = eval_x;
                    out[1] = eval_y;
                    if stroke_direction != zero {
                        let r0 = load_path_radius(
                            path_thickness,
                            thickness_offset,
                            thickness_count,
                            i0,
                            point_count,
                            is_closed,
                            stroke_radius,
                        );
                        let r1 = load_path_radius(
                            path_thickness,
                            thickness_offset,
                            thickness_count,
                            i1,
                            point_count,
                            is_closed,
                            stroke_radius,
                        );
                        let r2 = load_path_radius(
                            path_thickness,
                            thickness_offset,
                            thickness_count,
                            i2,
                            point_count,
                            is_closed,
                            stroke_radius,
                        );
                        let r3 = load_path_radius(
                            path_thickness,
                            thickness_offset,
                            thickness_count,
                            i3,
                            point_count,
                            is_closed,
                            stroke_radius,
                        );
                        let r = r0 * (tt * tt * tt)
                            + r1 * (f32::new(3.0) * tt * tt * local_t)
                            + r2 * (f32::new(3.0) * tt * local_t * local_t)
                            + r3 * (local_t * local_t * local_t);
                        out[0] += stroke_direction * r * normal[0];
                        out[1] += stroke_direction * r * normal[1];
                        if stroke_direction < zero {
                            normal[0] = -normal[0];
                            normal[1] = -normal[1];
                        }
                    }
                }
            } else {
                *pdf = zero;
            }
        }
    }

    out
}

/// Sample a boundary point on a shape (fill or stroke), returning local position and normal.
/// Sets `pdf`, `is_stroke`, and path identifiers for backprop.
#[cube]
fn sample_boundary_point(
    shape_data: &Array<f32>,
    path_points: &Array<f32>,
    path_num_controls: &Array<u32>,
    path_thickness: &Array<f32>,
    shape_path_offsets: &Array<u32>,
    shape_path_point_counts: &Array<u32>,
    shape_path_ctrl_offsets: &Array<u32>,
    shape_path_ctrl_counts: &Array<u32>,
    shape_path_thickness_offsets: &Array<u32>,
    shape_path_thickness_counts: &Array<u32>,
    shape_path_is_closed: &Array<u32>,
    path_cdf: &Array<f32>,
    path_pmf: &Array<f32>,
    path_point_ids: &Array<u32>,
    path_cdf_offsets: &Array<u32>,
    path_cdf_counts: &Array<u32>,
    path_point_offsets: &Array<u32>,
    shape_lengths: &Array<f32>,
    shape_id: u32,
    fill_kind: u32,
    stroke_kind: u32,
    t: f32,
    normal: &mut Line<f32>,
    pdf: &mut f32,
    is_stroke: &mut u32,
    base_point_id: &mut u32,
    point_id: &mut u32,
    out_t: &mut f32,
) -> Line<f32> {
    let zero = f32::new(0.0);
    let half = f32::new(0.5);
    let two = f32::new(2.0);
    let pi = f32::new(3.14159265);
    let two_pi = two * pi;

    *pdf = f32::new(1.0);
    let mut local_t = t;
    *is_stroke = u32::new(0);
    if fill_kind != PAINT_NONE && stroke_kind != PAINT_NONE {
        if local_t < half {
            local_t = local_t * two;
            *pdf *= half;
        } else {
            *is_stroke = u32::new(1);
            local_t = two * (local_t - half);
            *pdf *= half;
        }
    } else if stroke_kind != PAINT_NONE {
        *is_stroke = u32::new(1);
    }

    let mut stroke_direction = zero;
    if *is_stroke != u32::new(0) {
        if local_t < half {
            stroke_direction = f32::new(-1.0);
            local_t = local_t * two;
            *pdf *= half;
        } else {
            stroke_direction = f32::new(1.0);
            local_t = two * (local_t - half);
            *pdf *= half;
        }
    }

    let base = (shape_id * SHAPE_STRIDE) as usize;
    let kind = shape_data[base] as u32;
    let stroke_width = shape_data[base + 3];
    let p0 = shape_data[base + 4];
    let p1 = shape_data[base + 5];
    let p2 = shape_data[base + 6];
    let p3 = shape_data[base + 7];

    let mut out = Line::empty(2usize);
    if kind == SHAPE_KIND_CIRCLE {
        let r = p2.abs();
        if r <= zero {
            *pdf = zero;
        } else {
            let angle = two_pi * local_t;
            let offset_x = r * angle.cos();
            let offset_y = r * angle.sin();
            let n = vec2_normalize(offset_x, offset_y);
            normal[0] = n[0];
            normal[1] = n[1];
            *pdf /= two_pi * r;
            out[0] = p0 + offset_x;
            out[1] = p1 + offset_y;
            if stroke_direction != zero {
                out[0] += stroke_direction * stroke_width * normal[0];
                out[1] += stroke_direction * stroke_width * normal[1];
                if stroke_direction < zero {
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                }
            }
        }
    } else if kind == SHAPE_KIND_ELLIPSE {
        let rx = p2.abs();
        let ry = p3.abs();
        if rx <= zero || ry <= zero {
            *pdf = zero;
        } else {
            let angle = two_pi * local_t;
            let s = angle.sin();
            let c = angle.cos();
            let offset_x = rx * c;
            let offset_y = ry * s;
            let dxdt = -rx * s * two_pi;
            let dydt = ry * c * two_pi;
            let n = vec2_normalize(dydt, -dxdt);
            normal[0] = n[0];
            normal[1] = n[1];
            *pdf /= vec2_length(dxdt, dydt);
            out[0] = p0 + offset_x;
            out[1] = p1 + offset_y;
            if stroke_direction != zero {
                out[0] += stroke_direction * stroke_width * normal[0];
                out[1] += stroke_direction * stroke_width * normal[1];
                if stroke_direction < zero {
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                }
            }
        }
    } else if kind == SHAPE_KIND_RECT {
        let min_x = p0;
        let min_y = p1;
        let max_x = p2;
        let max_y = p3;
        let w = max_x - min_x;
        let h = max_y - min_y;
        if w == zero && h == zero {
            *pdf = zero;
        } else {
            *pdf /= two * (w + h);
            if local_t <= w / (w + h) {
                local_t = local_t * (w + h) / w;
                if local_t < half {
                    normal[0] = zero;
                    normal[1] = f32::new(-1.0);
                    out[0] = min_x + two * local_t * (max_x - min_x);
                    out[1] = min_y;
                } else {
                    normal[0] = zero;
                    normal[1] = f32::new(1.0);
                    out[0] = min_x + two * (local_t - half) * (max_x - min_x);
                    out[1] = max_y;
                }
            } else {
                local_t = (local_t - w / (w + h)) * (w + h) / h;
                if local_t < half {
                    normal[0] = f32::new(-1.0);
                    normal[1] = zero;
                    out[0] = min_x;
                    out[1] = min_y + two * local_t * (max_y - min_y);
                } else {
                    normal[0] = f32::new(1.0);
                    normal[1] = zero;
                    out[0] = max_x;
                    out[1] = min_y + two * (local_t - half) * (max_y - min_y);
                }
            }
            if stroke_direction != zero {
                out[0] += stroke_direction * stroke_width * normal[0];
                out[1] += stroke_direction * stroke_width * normal[1];
                if stroke_direction < zero {
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                }
            }
        }
    } else if kind == SHAPE_KIND_PATH {
        let point_offset = shape_path_offsets[shape_id as usize];
        let point_count = shape_path_point_counts[shape_id as usize];
        let ctrl_offset = shape_path_ctrl_offsets[shape_id as usize];
        let ctrl_count = shape_path_ctrl_counts[shape_id as usize];
        let thickness_offset = shape_path_thickness_offsets[shape_id as usize];
        let thickness_count = shape_path_thickness_counts[shape_id as usize];
        let is_closed = shape_path_is_closed[shape_id as usize];
        let path_length = shape_lengths[shape_id as usize];
        let cdf_offset = path_cdf_offsets[shape_id as usize];
        let cdf_count = path_cdf_counts[shape_id as usize];
        let point_offset_map = path_point_offsets[shape_id as usize];
        out = sample_boundary_path(
            path_points,
            path_num_controls,
            path_thickness,
            point_offset,
            point_count,
            ctrl_offset,
            ctrl_count,
            thickness_offset,
            thickness_count,
            is_closed,
            path_cdf,
            path_pmf,
            path_point_ids,
            cdf_offset,
            cdf_count,
            point_offset_map,
            path_length,
            local_t,
            stroke_direction,
            stroke_width,
            normal,
            pdf,
            base_point_id,
            point_id,
            out_t,
        );
    } else {
        *pdf = zero;
    }

    out
}

/// Composite the color at a normalized point by evaluating group fills and strokes.
/// Sets `edge_hit` if the boundary shape contributes a visible fragment.
#[cube]
fn sample_color_edge(
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
    width: u32,
    height: u32,
    px_norm: f32,
    py_norm: f32,
    background: Line<f32>,
    boundary_group_id: u32,
    boundary_shape_id: u32,
    edge_hit: &mut u32,
) -> Line<f32> {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let big = f32::new(1.0e20);

    let px = px_norm * f32::cast_from(width);
    let py = py_norm * f32::cast_from(height);

    let mut frag_color_r: Array<f32> = Array::new(MAX_FRAGMENTS);
    let mut frag_color_g: Array<f32> = Array::new(MAX_FRAGMENTS);
    let mut frag_color_b: Array<f32> = Array::new(MAX_FRAGMENTS);
    let mut frag_alpha: Array<f32> = Array::new(MAX_FRAGMENTS);
    let mut frag_edge: Array<u32> = Array::new(MAX_FRAGMENTS);
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

        let mut edge_fill_hit = u32::new(0);
        let mut edge_stroke_hit = u32::new(0);
        if group_id == boundary_group_id && (fill_kind != PAINT_NONE || stroke_kind != PAINT_NONE) {
            let mut ef_min_dist = big;
            let mut ef_winding = zero;
            let mut ef_crossings = zero;
            let mut es_min_dist = big;
            let mut es_min_radius = zero;
            let mut es_hit = zero;
            accumulate_shape_in_group(
                shape_data,
                segment_data,
                shape_bounds,
                shape_xform,
                curve_data,
                path_bvh_bounds,
                path_bvh_nodes,
                path_bvh_indices,
                path_bvh_meta,
                boundary_shape_id,
                local_pt[0],
                local_pt[1],
                fill_kind,
                stroke_kind,
                fill_rule,
                u32::new(0),
                &mut ef_min_dist,
                &mut ef_winding,
                &mut ef_crossings,
                &mut es_min_dist,
                &mut es_min_radius,
                &mut es_hit,
            );
            if stroke_kind != PAINT_NONE && es_hit > zero {
                edge_stroke_hit = u32::new(1);
            }
            if fill_kind != PAINT_NONE {
                let inside = if fill_rule == u32::new(1) {
                    ef_crossings > zero
                } else {
                    ef_winding != zero
                };
                if inside {
                    edge_fill_hit = u32::new(1);
                }
            }
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
                frag_edge[idx] = edge_stroke_hit;
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
                    frag_edge[idx] = edge_fill_hit;
                    frag_count += u32::new(1);
                }
            }
        }

        group_id += u32::new(1);
    }

    let mut out = Line::empty(4usize);
    *edge_hit = u32::new(0);
    if frag_count == u32::new(0) {
        out[0] = background[0];
        out[1] = background[1];
        out[2] = background[2];
        out[3] = background[3];
    } else {
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
            if new_a >= one && *edge_hit != u32::new(0) {
                *edge_hit = u32::new(0);
            }
            if frag_edge[idx] != u32::new(0) {
                *edge_hit = u32::new(1);
            }
            prev_r = prev_r * (one - new_a) + new_r * new_a;
            prev_g = prev_g * (one - new_a) + new_g * new_a;
            prev_b = prev_b * (one - new_a) + new_b * new_a;
            prev_a = prev_a * (one - new_a) + new_a;
            i += u32::new(1);
        }

        if prev_a > f32::new(1.0e-6) {
            let inv = one / prev_a;
            prev_r *= inv;
            prev_g *= inv;
            prev_b *= inv;
        }
        out[0] = prev_r;
        out[1] = prev_g;
        out[2] = prev_b;
        out[3] = prev_a;
    }

    out
}

/// Accumulate a boundary sample contribution into shape parameters and transforms.
#[cube]
fn accumulate_boundary_gradient(
    shape_data: &Array<f32>,
    path_num_controls: &Array<u32>,
    shape_path_offsets: &Array<u32>,
    shape_path_point_counts: &Array<u32>,
    shape_path_ctrl_offsets: &Array<u32>,
    shape_path_thickness_offsets: &Array<u32>,
    shape_path_thickness_counts: &Array<u32>,
    shape_path_is_closed: &Array<u32>,
    shape_index: u32,
    contrib: f32,
    t: f32,
    normal: Line<f32>,
    is_stroke: u32,
    base_point_id: u32,
    point_id: u32,
    path_t: f32,
    d_shape_params: &mut Array<Atomic<f32>>,
    d_shape_points: &mut Array<Atomic<f32>>,
    d_shape_thickness: &mut Array<Atomic<f32>>,
    d_shape_stroke_width: &mut Array<Atomic<f32>>,
    d_shape_transform: &mut Array<Atomic<f32>>,
    d_group_transform: &mut Array<Atomic<f32>>,
    shape_transform: &Array<f32>,
    group_shape_xform: &Array<f32>,
    group_id: u32,
    local_boundary_pt: Line<f32>,
) {
    if contrib != f32::new(0.0) {

        if is_stroke != u32::new(0) {
            let thickness_offset = shape_path_thickness_offsets[shape_index as usize];
            let thickness_count = shape_path_thickness_counts[shape_index as usize];
            let point_count = shape_path_point_counts[shape_index as usize];
            let is_closed = shape_path_is_closed[shape_index as usize];
            if thickness_count > u32::new(0) && point_count > u32::new(0) {
                let ctrl_offset = shape_path_ctrl_offsets[shape_index as usize];
                let ctrl = path_num_controls[(ctrl_offset + base_point_id) as usize];
                if ctrl == u32::new(0) {
                    let i0 = point_id;
                    let i1 = path_point_index(point_id + u32::new(1), point_count, is_closed);
                    let base = thickness_offset as usize;
                    d_shape_thickness[base + i0 as usize].fetch_add((f32::new(1.0) - path_t) * contrib);
                    d_shape_thickness[base + i1 as usize].fetch_add(path_t * contrib);
                } else if ctrl == u32::new(1) {
                    let i0 = point_id;
                    let i1 = point_id + u32::new(1);
                    let i2 = path_point_index(point_id + u32::new(2), point_count, is_closed);
                    let tt = f32::new(1.0) - path_t;
                    let base = thickness_offset as usize;
                    d_shape_thickness[base + i0 as usize].fetch_add(tt * tt * contrib);
                    d_shape_thickness[base + i1 as usize].fetch_add(f32::new(2.0) * tt * path_t * contrib);
                    d_shape_thickness[base + i2 as usize].fetch_add(path_t * path_t * contrib);
                } else if ctrl == u32::new(2) {
                    let i0 = point_id;
                    let i1 = point_id + u32::new(1);
                    let i2 = point_id + u32::new(2);
                    let i3 = path_point_index(point_id + u32::new(3), point_count, is_closed);
                    let tt = f32::new(1.0) - path_t;
                    let base = thickness_offset as usize;
                    d_shape_thickness[base + i0 as usize].fetch_add(tt * tt * tt * contrib);
                    d_shape_thickness[base + i1 as usize]
                        .fetch_add(f32::new(3.0) * tt * tt * path_t * contrib);
                    d_shape_thickness[base + i2 as usize]
                        .fetch_add(f32::new(3.0) * tt * path_t * path_t * contrib);
                    d_shape_thickness[base + i3 as usize].fetch_add(path_t * path_t * path_t * contrib);
                }
            } else {
                d_shape_stroke_width[shape_index as usize].fetch_add(contrib);
            }
        }

        let base = (shape_index * SHAPE_STRIDE) as usize;
        let kind = shape_data[base] as u32;
        if kind == SHAPE_KIND_CIRCLE {
            let param_base = shape_index * u32::new(8);
            d_shape_params[param_base as usize].fetch_add(normal[0] * contrib);
            d_shape_params[param_base as usize + 1].fetch_add(normal[1] * contrib);
            d_shape_params[param_base as usize + 2].fetch_add(contrib);
        } else if kind == SHAPE_KIND_ELLIPSE {
            let param_base = shape_index * u32::new(8);
            d_shape_params[param_base as usize].fetch_add(normal[0] * contrib);
            d_shape_params[param_base as usize + 1].fetch_add(normal[1] * contrib);
            let angle = f32::new(6.28318530) * t;
            d_shape_params[param_base as usize + 2].fetch_add(angle.cos() * normal[0] * contrib);
            d_shape_params[param_base as usize + 3].fetch_add(angle.sin() * normal[1] * contrib);
        } else if kind == SHAPE_KIND_PATH {
            let point_offset = shape_path_offsets[shape_index as usize];
            let point_count = shape_path_point_counts[shape_index as usize];
            let is_closed = shape_path_is_closed[shape_index as usize];
            let ctrl_offset = shape_path_ctrl_offsets[shape_index as usize];
            let ctrl = path_num_controls[(ctrl_offset + base_point_id) as usize];
            if ctrl == u32::new(0) {
                let i0 = point_id;
                let i1 = path_point_index(point_id + u32::new(1), point_count, is_closed);
                add_path_point_grad(
                    d_shape_points,
                    point_offset,
                    i0,
                    (f32::new(1.0) - path_t) * normal[0] * contrib,
                    (f32::new(1.0) - path_t) * normal[1] * contrib,
                );
                add_path_point_grad(
                    d_shape_points,
                    point_offset,
                    i1,
                    path_t * normal[0] * contrib,
                    path_t * normal[1] * contrib,
                );
            } else if ctrl == u32::new(1) {
                let i0 = point_id;
                let i1 = point_id + u32::new(1);
                let i2 = path_point_index(point_id + u32::new(2), point_count, is_closed);
                let tt = f32::new(1.0) - path_t;
                add_path_point_grad(
                    d_shape_points,
                    point_offset,
                    i0,
                    tt * tt * normal[0] * contrib,
                    tt * tt * normal[1] * contrib,
                );
                add_path_point_grad(
                    d_shape_points,
                    point_offset,
                    i1,
                    f32::new(2.0) * tt * path_t * normal[0] * contrib,
                    f32::new(2.0) * tt * path_t * normal[1] * contrib,
                );
                add_path_point_grad(
                    d_shape_points,
                    point_offset,
                    i2,
                    path_t * path_t * normal[0] * contrib,
                    path_t * path_t * normal[1] * contrib,
                );
            } else if ctrl == u32::new(2) {
                let i0 = point_id;
                let i1 = point_id + u32::new(1);
                let i2 = point_id + u32::new(2);
                let i3 = path_point_index(point_id + u32::new(3), point_count, is_closed);
                let tt = f32::new(1.0) - path_t;
                add_path_point_grad(
                    d_shape_points,
                    point_offset,
                    i0,
                    tt * tt * tt * normal[0] * contrib,
                    tt * tt * tt * normal[1] * contrib,
                );
                add_path_point_grad(
                    d_shape_points,
                    point_offset,
                    i1,
                    f32::new(3.0) * tt * tt * path_t * normal[0] * contrib,
                    f32::new(3.0) * tt * tt * path_t * normal[1] * contrib,
                );
                add_path_point_grad(
                    d_shape_points,
                    point_offset,
                    i2,
                    f32::new(3.0) * tt * path_t * path_t * normal[0] * contrib,
                    f32::new(3.0) * tt * path_t * path_t * normal[1] * contrib,
                );
                add_path_point_grad(
                    d_shape_points,
                    point_offset,
                    i3,
                    path_t * path_t * path_t * normal[0] * contrib,
                    path_t * path_t * path_t * normal[1] * contrib,
                );
            }
        } else if kind == SHAPE_KIND_RECT {
            let param_base = shape_index * u32::new(8);
            if normal[0] == f32::new(-1.0) && normal[1] == f32::new(0.0) {
                d_shape_params[param_base as usize].fetch_add(-contrib);
            } else if normal[0] == f32::new(1.0) && normal[1] == f32::new(0.0) {
                d_shape_params[param_base as usize + 2].fetch_add(contrib);
            } else if normal[0] == f32::new(0.0) && normal[1] == f32::new(-1.0) {
                d_shape_params[param_base as usize + 1].fetch_add(-contrib);
            } else if normal[0] == f32::new(0.0) && normal[1] == f32::new(1.0) {
                d_shape_params[param_base as usize + 3].fetch_add(contrib);
            }
        }

        let shape_base = (shape_index * XFORM_STRIDE) as usize;
        let s_m00 = shape_transform[shape_base];
        let s_m01 = shape_transform[shape_base + 1];
        let s_m02 = shape_transform[shape_base + 2];
        let s_m10 = shape_transform[shape_base + 3];
        let s_m11 = shape_transform[shape_base + 4];
        let s_m12 = shape_transform[shape_base + 5];

        let gs_base = (group_id * XFORM_STRIDE) as usize;
        let g_m00 = group_shape_xform[gs_base];
        let g_m01 = group_shape_xform[gs_base + 1];
        let g_m02 = group_shape_xform[gs_base + 2];
        let g_m10 = group_shape_xform[gs_base + 3];
        let g_m11 = group_shape_xform[gs_base + 4];
        let g_m12 = group_shape_xform[gs_base + 5];

        let shape_to_canvas_m00 = g_m00 * s_m00 + g_m01 * s_m10;
        let shape_to_canvas_m01 = g_m00 * s_m01 + g_m01 * s_m11;
        let shape_to_canvas_m02 = g_m00 * s_m02 + g_m01 * s_m12 + g_m02;
        let shape_to_canvas_m10 = g_m10 * s_m00 + g_m11 * s_m10;
        let shape_to_canvas_m11 = g_m10 * s_m01 + g_m11 * s_m11;
        let shape_to_canvas_m12 = g_m10 * s_m02 + g_m11 * s_m12 + g_m12;

        let mut d_shape_to_canvas_affine = Line::empty(8usize);
        let mut d_local_boundary_pt = Line::empty(2usize);
        d_xform_pt_affine(
            shape_to_canvas_m00,
            shape_to_canvas_m01,
            shape_to_canvas_m02,
            shape_to_canvas_m10,
            shape_to_canvas_m11,
            shape_to_canvas_m12,
            local_boundary_pt[0],
            local_boundary_pt[1],
            normal[0] * contrib,
            normal[1] * contrib,
            &mut d_shape_to_canvas_affine,
            &mut d_local_boundary_pt,
        );
        let d_shape_to_canvas = affine_grad_to_mat3(d_shape_to_canvas_affine);

        let shape_mat = mat3_from_affine(s_m00, s_m01, s_m02, s_m10, s_m11, s_m12);
        let group_mat = mat3_from_affine(g_m00, g_m01, g_m02, g_m10, g_m11, g_m12);
        let d_group_mat = mat3_mul(d_shape_to_canvas, mat3_transpose(shape_mat));
        let d_shape_mat = mat3_mul(mat3_transpose(group_mat), d_shape_to_canvas);

        let g_base = (group_id * u32::new(9)) as usize;
        atomic_add_mat3(d_group_transform, g_base, d_group_mat);
        let s_base = (shape_index * u32::new(9)) as usize;
        atomic_add_mat3(d_shape_transform, s_base, d_shape_mat);
        }
}
/// Boundary sampling kernel for non-prefiltered gradients.
#[cube(launch_unchecked)]
pub(crate) fn boundary_sampling_kernel(
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
    path_thickness: &Array<f32>,
    shape_path_offsets: &Array<u32>,
    shape_path_point_counts: &Array<u32>,
    shape_path_ctrl_offsets: &Array<u32>,
    shape_path_ctrl_counts: &Array<u32>,
    shape_path_thickness_offsets: &Array<u32>,
    shape_path_thickness_counts: &Array<u32>,
    shape_path_is_closed: &Array<u32>,
    shape_lengths: &Array<f32>,
    shape_cdf: &Array<f32>,
    shape_pmf: &Array<f32>,
    shape_ids: &Array<u32>,
    group_ids: &Array<u32>,
    path_cdf: &Array<f32>,
    path_pmf: &Array<f32>,
    path_point_ids: &Array<u32>,
    path_cdf_offsets: &Array<u32>,
    path_cdf_counts: &Array<u32>,
    path_point_offsets: &Array<u32>,
    shape_sample_count: u32,
    width: u32,
    height: u32,
    num_groups: u32,
    samples_x: u32,
    samples_y: u32,
    seed: u32,
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
    d_shape_thickness: &mut Array<Atomic<f32>>,
    d_shape_stroke_width: &mut Array<Atomic<f32>>,
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

    if shape_sample_count == u32::new(0) {
        terminate!();
    }

    let rng = pcg32_init(idx as u32, seed);
    let mut state_lo = rng[0];
    let mut state_hi = rng[1];
    let inc_lo = rng[2];
    let inc_hi = rng[3];
    let step0 = pcg32_next(state_lo, state_hi, inc_lo, inc_hi);
    state_lo = step0[1];
    state_hi = step0[2];
    let u = pcg32_f32(step0[0]);
    let step1 = pcg32_next(state_lo, state_hi, inc_lo, inc_hi);
    let t = pcg32_f32(step1[0]);

    let mut dummy_t = f32::new(0.0);
    let sample_id = sample_cdf(shape_cdf, u32::new(0), shape_sample_count, u, &mut dummy_t);
    let shape_id = shape_ids[sample_id as usize];
    let group_id = group_ids[sample_id as usize];
    if group_id >= num_groups {
        terminate!();
    }
    let shape_p = shape_pmf[sample_id as usize];
    if shape_p <= f32::new(0.0) {
        terminate!();
    }

    let group_base = (group_id * GROUP_STRIDE) as usize;
    let fill_kind = group_data[group_base + 2] as u32;
    let stroke_kind = group_data[group_base + 4] as u32;
    if fill_kind == PAINT_NONE && stroke_kind == PAINT_NONE {
        terminate!();
    }

    let mut normal_local = Line::empty(2usize);
    let mut boundary_pdf = f32::new(0.0);
    let mut is_stroke = u32::new(0);
    let mut base_point_id = u32::new(0);
    let mut point_id = u32::new(0);
    let mut path_t = f32::new(0.0);
    let local_boundary_pt = sample_boundary_point(
        shape_data,
        path_points,
        path_num_controls,
        path_thickness,
        shape_path_offsets,
        shape_path_point_counts,
        shape_path_ctrl_offsets,
        shape_path_ctrl_counts,
        shape_path_thickness_offsets,
        shape_path_thickness_counts,
        shape_path_is_closed,
        path_cdf,
        path_pmf,
        path_point_ids,
        path_cdf_offsets,
        path_cdf_counts,
        path_point_offsets,
        shape_lengths,
        shape_id,
        fill_kind,
        stroke_kind,
        t,
        &mut normal_local,
        &mut boundary_pdf,
        &mut is_stroke,
        &mut base_point_id,
        &mut point_id,
        &mut path_t,
    );
    if boundary_pdf <= f32::new(0.0) {
        terminate!();
    }

    let shape_base = (shape_id * XFORM_STRIDE) as usize;
    let s_inv00 = shape_xform[shape_base];
    let s_inv01 = shape_xform[shape_base + 1];
    let s_inv10 = shape_xform[shape_base + 3];
    let s_inv11 = shape_xform[shape_base + 4];

    let group_base_xf = (group_id * XFORM_STRIDE) as usize;
    let g_inv00 = group_xform[group_base_xf];
    let g_inv01 = group_xform[group_base_xf + 1];
    let g_inv10 = group_xform[group_base_xf + 3];
    let g_inv11 = group_xform[group_base_xf + 4];

    let c00 = s_inv00 * g_inv00 + s_inv01 * g_inv10;
    let c01 = s_inv00 * g_inv01 + s_inv01 * g_inv11;
    let c10 = s_inv10 * g_inv00 + s_inv11 * g_inv10;
    let c11 = s_inv10 * g_inv01 + s_inv11 * g_inv11;
    let normal_canvas = xform_normal_affine(c00, c01, c10, c11, normal_local[0], normal_local[1]);

    let s_m00 = shape_transform[shape_base];
    let s_m01 = shape_transform[shape_base + 1];
    let s_m02 = shape_transform[shape_base + 2];
    let s_m10 = shape_transform[shape_base + 3];
    let s_m11 = shape_transform[shape_base + 4];
    let s_m12 = shape_transform[shape_base + 5];

    let gs_base = (group_id * XFORM_STRIDE) as usize;
    let g_m00 = group_shape_xform[gs_base];
    let g_m01 = group_shape_xform[gs_base + 1];
    let g_m02 = group_shape_xform[gs_base + 2];
    let g_m10 = group_shape_xform[gs_base + 3];
    let g_m11 = group_shape_xform[gs_base + 4];
    let g_m12 = group_shape_xform[gs_base + 5];

    let local_group = xform_pt_affine(
        s_m00,
        s_m01,
        s_m02,
        s_m10,
        s_m11,
        s_m12,
        local_boundary_pt[0],
        local_boundary_pt[1],
    );
    let boundary_pt_canvas = xform_pt_affine(
        g_m00,
        g_m01,
        g_m02,
        g_m10,
        g_m11,
        g_m12,
        local_group[0],
        local_group[1],
    );

    let width_f = f32::cast_from(width);
    let height_f = f32::cast_from(height);
    let boundary_pt_norm_x = boundary_pt_canvas[0] / width_f;
    let boundary_pt_norm_y = boundary_pt_canvas[1] / height_f;
    let bx = (boundary_pt_norm_x * width_f) as i32;
    let by = (boundary_pt_norm_y * height_f) as i32;
    if bx < 0 || by < 0 || bx >= width as i32 || by >= height as i32 {
        terminate!();
    }
    let pixel_index = (by as u32) * width + (bx as u32);

    let mut background = Line::empty(4usize);
    if has_background_image != u32::new(0) {
        let idx4 = (pixel_index as usize) * 4;
        background[0] = background_image[idx4];
        background[1] = background_image[idx4 + 1];
        background[2] = background_image[idx4 + 2];
        background[3] = background_image[idx4 + 3];
    } else {
        background[0] = background_r;
        background[1] = background_g;
        background[2] = background_b;
        background[3] = background_a;
    }

    let eps = f32::new(1.0e-4);
    let inside_pt_x = boundary_pt_norm_x - normal_canvas[0] * eps;
    let inside_pt_y = boundary_pt_norm_y - normal_canvas[1] * eps;
    let outside_pt_x = boundary_pt_norm_x + normal_canvas[0] * eps;
    let outside_pt_y = boundary_pt_norm_y + normal_canvas[1] * eps;

    let mut inside_hit = u32::new(0);
    let mut outside_hit = u32::new(0);
    let mut color_inside = sample_color_edge(
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
        width,
        height,
        inside_pt_x,
        inside_pt_y,
        background,
        group_id,
        shape_id,
        &mut inside_hit,
    );
    let mut color_outside = sample_color_edge(
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
        width,
        height,
        outside_pt_x,
        outside_pt_y,
        background,
        group_id,
        shape_id,
        &mut outside_hit,
    );

    if inside_hit == u32::new(0) && outside_hit == u32::new(0) {
        terminate!();
    }

    let mut normal_use = normal_canvas;
    if inside_hit == u32::new(0) {
        normal_use[0] = -normal_use[0];
        normal_use[1] = -normal_use[1];
        let tmp = color_inside;
        color_inside = color_outside;
        color_outside = tmp;
    }

    let mut d_color = {
        let mut tmp = Line::empty(4usize);
        let sboundary_x = boundary_pt_norm_x * width_f;
        let sboundary_y = boundary_pt_norm_y * height_f;
        let dc = gather_d_color(
            filter_type,
            filter_radius,
            d_render_image,
            weight_image,
            width,
            height,
            sboundary_x,
            sboundary_y,
        );
        tmp[0] = dc[0];
        tmp[1] = dc[1];
        tmp[2] = dc[2];
        tmp[3] = dc[3];
        tmp
    };

    let norm = width_f * height_f;
    if norm > f32::new(0.0) {
        d_color[0] = d_color[0] / norm;
        d_color[1] = d_color[1] / norm;
        d_color[2] = d_color[2] / norm;
        d_color[3] = d_color[3] / norm;
    }

    let diff_r = color_inside[0] - color_outside[0];
    let diff_g = color_inside[1] - color_outside[1];
    let diff_b = color_inside[2] - color_outside[2];
    let diff_a = color_inside[3] - color_outside[3];
    let contrib = (diff_r * d_color[0] + diff_g * d_color[1] + diff_b * d_color[2] + diff_a * d_color[3])
        / (shape_p * boundary_pdf);

    accumulate_boundary_gradient(
        shape_data,
        path_num_controls,
        shape_path_offsets,
        shape_path_point_counts,
        shape_path_ctrl_offsets,
        shape_path_thickness_offsets,
        shape_path_thickness_counts,
        shape_path_is_closed,
        shape_id,
        contrib,
        t,
        normal_use,
        is_stroke,
        base_point_id,
        point_id,
        path_t,
        d_shape_params,
        d_shape_points,
        d_shape_thickness,
        d_shape_stroke_width,
        d_shape_transform,
        d_group_transform,
        shape_transform,
        group_shape_xform,
        group_id,
        local_boundary_pt,
    );

    if translation_flag != u32::new(0) {
        add_translation(d_translation, pixel_index, normal_use[0] * contrib, normal_use[1] * contrib);
    }
}
