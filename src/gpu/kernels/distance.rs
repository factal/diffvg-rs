//! GPU distance and closest-point helpers for shape queries.

use cubecl::prelude::*;
use crate::gpu::constants::*;
use super::{math::*, curves::*};

/// Compute Euclidean distance from (px, py) to an axis-aligned bounds.
/// Returns 0 when the point is inside or on the bounds.
#[cube]
pub(super) fn bounds_distance(min_x: f32, min_y: f32, max_x: f32, max_y: f32, px: f32, py: f32) -> f32 {
    let zero = f32::new(0.0);
    let dx = max_f32(max_f32(min_x - px, zero), px - max_x);
    let dy = max_f32(max_f32(min_y - py, zero), py - max_y);
    (dx * dx + dy * dy).sqrt()
}

/// Check whether (px, py) lies inside or on an axis-aligned bounds.
#[cube]
pub(super) fn bounds_contains(min_x: f32, min_y: f32, max_x: f32, max_y: f32, px: f32, py: f32) -> bool {
    px >= min_x && px <= max_x && py >= min_y && py <= max_y
}

/// Conservative rightward-ray test against bounds for winding queries.
/// Ignores `min_x`, only checks the y-range and `max_x`.
#[cube]
pub(super) fn ray_intersects_bounds(_min_x: f32, min_y: f32, max_x: f32, max_y: f32, px: f32, py: f32) -> bool {
    !(py < min_y || py > max_y || px > max_x)
}

/// Closest point on a circle boundary to (px, py).
/// If the point is at the center, picks (cx + radius, cy).
#[cube]
pub(super) fn closest_point_circle(cx: f32, cy: f32, radius: f32, px: f32, py: f32) -> Line<f32> {
    let mut out = Line::empty(2usize);
    let dx = px - cx;
    let dy = py - cy;
    let len = vec2_length(dx, dy);
    if len <= f32::new(1.0e-8) {
        out[0] = cx + radius;
        out[1] = cy;
    } else {
        let inv = radius / len;
        out[0] = cx + dx * inv;
        out[1] = cy + dy * inv;
    }
    out
}

/// Closest point on an axis-aligned rectangle boundary to (px, py).
/// Inside points snap to the nearest edge; outside points clamp to bounds.
#[cube]
pub(super) fn closest_point_rect(min_x: f32, min_y: f32, max_x: f32, max_y: f32, px: f32, py: f32) -> Line<f32> {
    let mut out = Line::empty(2usize);
    let inside = px >= min_x && px <= max_x && py >= min_y && py <= max_y;
    if !inside {
        out[0] = clamp_f32(px, min_x, max_x);
        out[1] = clamp_f32(py, min_y, max_y);
    } else {
        let dl = px - min_x;
        let dr = max_x - px;
        let db = py - min_y;
        let dt = max_y - py;
        if dl <= dr && dl <= db && dl <= dt {
            out[0] = min_x;
            out[1] = py;
        } else if dr <= db && dr <= dt {
            out[0] = max_x;
            out[1] = py;
        } else if db <= dt {
            out[0] = px;
            out[1] = min_y;
        } else {
            out[0] = px;
            out[1] = max_y;
        }
    }
    out
}

/// Closest point on an axis-aligned ellipse boundary to (px, py).
/// Handles degenerate radii and uses Newton refinement in the first quadrant.
#[cube]
pub(super) fn closest_point_ellipse(cx: f32, cy: f32, rx_in: f32, ry_in: f32, px: f32, py: f32) -> Line<f32> {
    let mut out = Line::empty(2usize);
    let rx = rx_in.abs();
    let ry = ry_in.abs();
    let eps = f32::new(1.0e-6);
    if rx < eps && ry < eps {
        out[0] = cx;
        out[1] = cy;
    } else if rx < eps {
        let y = clamp_f32(py - cy, -ry, ry);
        out[0] = cx;
        out[1] = cy + y;
    } else if ry < eps {
        let x = clamp_f32(px - cx, -rx, rx);
        out[0] = cx + x;
        out[1] = cy;
    } else {
        let mut dx = px - cx;
        let mut dy = py - cy;
        if abs_f32(dx) < eps && abs_f32(dy) < eps {
            out[0] = cx + rx;
            out[1] = cy;
        } else {
            let sign_x = if dx < f32::new(0.0) { f32::new(-1.0) } else { f32::new(1.0) };
            let sign_y = if dy < f32::new(0.0) { f32::new(-1.0) } else { f32::new(1.0) };
            dx = abs_f32(dx);
            dy = abs_f32(dy);

            let mut t = (dy * rx).atan2(dx * ry);
            let mut i = u32::new(0);
            while i < u32::new(20) {
                let s = t.sin();
                let c = t.cos();
                let g = rx * dx * s - ry * dy * c + (ry * ry - rx * rx) * s * c;
                let g_t = rx * dx * c + ry * dy * s + (ry * ry - rx * rx) * (c * c - s * s);
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
            out[0] = cx + sign_x * rx * c;
            out[1] = cy + sign_y * ry * s;
        }
    }
    out
}

/// Scan all curves in a path to find the closest point to (px, py).
/// Writes local position, segment index, and param t; returns 1 if found.
#[cube]
pub(super) fn closest_point_path(
    curve_data: &Array<f32>,
    curve_offset: u32,
    curve_count: u32,
    px: f32,
    py: f32,
    use_distance_approx: bool,
    out_local: &mut Line<f32>,
    out_base: &mut u32,
    out_t: &mut f32,
) -> u32 {
    let big = f32::new(1.0e20);
    let mut best_dist = big;
    let mut best_t = f32::new(0.0);
    let mut best_base = u32::new(0);
    let mut best_x = f32::new(0.0);
    let mut best_y = f32::new(0.0);

    let mut s = u32::new(0);
    while s < curve_count {
        let seg_base = ((curve_offset + s) * CURVE_STRIDE) as usize;
        let seg_kind = curve_data[seg_base] as u32;
        let x0 = curve_data[seg_base + 1];
        let y0 = curve_data[seg_base + 2];
        let x1 = curve_data[seg_base + 3];
        let y1 = curve_data[seg_base + 4];
        let x2 = curve_data[seg_base + 5];
        let y2 = curve_data[seg_base + 6];
        let x3 = curve_data[seg_base + 7];
        let y3 = curve_data[seg_base + 8];

        let mut dist = big;
        let mut t = f32::new(0.0);
        if seg_kind == u32::new(0) {
            let dist_t = distance_to_segment_with_t(px, py, x0, y0, x1, y1);
            dist = dist_t[0];
            t = dist_t[1];
        } else if seg_kind == u32::new(1) {
            let dist_t = closest_point_quadratic_with_t(
                px, py, x0, y0, x1, y1, x2, y2, use_distance_approx,
            );
            dist = dist_t[0];
            t = dist_t[1];
        } else {
            let dist_t = closest_point_cubic_with_t(
                px, py, x0, y0, x1, y1, x2, y2, x3, y3, use_distance_approx,
            );
            dist = dist_t[0];
            t = dist_t[1];
        }

        if dist < best_dist {
            best_dist = dist;
            best_t = t;
            best_base = s;
            if seg_kind == u32::new(0) {
                best_x = x0 + t * (x1 - x0);
                best_y = y0 + t * (y1 - y0);
            } else if seg_kind == u32::new(1) {
                let tt = f32::new(1.0) - t;
                best_x = tt * tt * x0 + f32::new(2.0) * tt * t * x1 + t * t * x2;
                best_y = tt * tt * y0 + f32::new(2.0) * tt * t * y1 + t * t * y2;
            } else {
                let tt = f32::new(1.0) - t;
                best_x = tt * tt * tt * x0
                    + f32::new(3.0) * tt * tt * t * x1
                    + f32::new(3.0) * tt * t * t * x2
                    + t * t * t * x3;
                best_y = tt * tt * tt * y0
                    + f32::new(3.0) * tt * tt * t * y1
                    + f32::new(3.0) * tt * t * t * y2
                    + t * t * t * y3;
            }
        }

        s += u32::new(1);
    }

    let mut result = u32::new(0);
    if best_dist < big {
        out_local[0] = best_x;
        out_local[1] = best_y;
        *out_base = best_base;
        *out_t = best_t;
        result = u32::new(1);
    }
    result
}

/// Dispatch to shape-specific closest-point routines in shape space.
/// Writes local position, segment index, and param t; returns 1 if valid.
#[cube]
pub(super) fn closest_point_shape(
    shape_data: &Array<f32>,
    curve_data: &Array<f32>,
    shape_index: u32,
    shape_px: f32,
    shape_py: f32,
    out_local: &mut Line<f32>,
    out_base: &mut u32,
    out_t: &mut f32,
) -> u32 {
    let base = (shape_index * SHAPE_STRIDE) as usize;
    let kind = shape_data[base] as u32;
    let curve_offset = shape_data[base + 12] as u32;
    let curve_count = shape_data[base + 13] as u32;
    let use_distance_approx = shape_data[base + 14] > f32::new(0.5);
    let p0 = shape_data[base + 4];
    let p1 = shape_data[base + 5];
    let p2 = shape_data[base + 6];
    let p3 = shape_data[base + 7];

    let mut result = u32::new(0);
    if kind == SHAPE_KIND_CIRCLE {
        let cp = closest_point_circle(p0, p1, p2.abs(), shape_px, shape_py);
        out_local[0] = cp[0];
        out_local[1] = cp[1];
        *out_base = u32::new(0);
        *out_t = f32::new(0.0);
        result = u32::new(1);
    } else if kind == SHAPE_KIND_ELLIPSE {
        let cp = closest_point_ellipse(p0, p1, p2.abs(), p3.abs(), shape_px, shape_py);
        out_local[0] = cp[0];
        out_local[1] = cp[1];
        *out_base = u32::new(0);
        *out_t = f32::new(0.0);
        result = u32::new(1);
    } else if kind == SHAPE_KIND_RECT {
        let cp = closest_point_rect(p0, p1, p2, p3, shape_px, shape_py);
        out_local[0] = cp[0];
        out_local[1] = cp[1];
        *out_base = u32::new(0);
        *out_t = f32::new(0.0);
        result = u32::new(1);
    } else if kind == SHAPE_KIND_PATH {
        result = closest_point_path(
            curve_data,
            curve_offset,
            curve_count,
            shape_px,
            shape_py,
            use_distance_approx,
            out_local,
            out_base,
            out_t,
        );
    }
    result
}

/// Find the closest point in a group to a canvas-space sample.
/// Uses group/shape transforms and optional BVH traversal.
/// Outputs local point, distance, shape id, segment index, and param t.
/// Search radius is capped by `max_radius`; returns 0 if no hit within it.
#[cube]
pub(super) fn compute_distance_group(
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
    group_id: u32,
    px: f32,
    py: f32,
    max_radius: f32,
    out_local: &mut Line<f32>,
    out_dist: &mut f32,
    out_shape: &mut u32,
    out_base: &mut u32,
    out_t: &mut f32,
) -> u32 {
    let big = f32::new(1.0e20);
    let mut best_dist = max_radius;
    let mut best_shape = u32::new(0);
    let mut best_base = u32::new(0);
    let mut best_t = f32::new(0.0);
    let mut best_x = f32::new(0.0);
    let mut best_y = f32::new(0.0);
    let mut found = u32::new(0);

    let group_base = (group_id * XFORM_STRIDE) as usize;
    let local_pt = xform_pt_affine(
        group_xform[group_base],
        group_xform[group_base + 1],
        group_xform[group_base + 2],
        group_xform[group_base + 3],
        group_xform[group_base + 4],
        group_xform[group_base + 5],
        px,
        py,
    );

    let meta_base = (group_id * BVH_META_STRIDE) as usize;
    let node_count = group_bvh_meta[meta_base + 1];
    let index_count = group_bvh_meta[meta_base + 3];
    if node_count > u32::new(0) && index_count > u32::new(0) {
        let node_offset = group_bvh_meta[meta_base];
        let index_offset = group_bvh_meta[meta_base + 2];
        let mut node_id = u32::new(0);
        while node_id != BVH_NONE {
            let node_base = ((node_offset + node_id) * BVH_NODE_STRIDE) as usize;
            let min_x = group_bvh_bounds[node_base];
            let min_y = group_bvh_bounds[node_base + 1];
            let max_x = group_bvh_bounds[node_base + 2];
            let max_y = group_bvh_bounds[node_base + 3];
            let skip = group_bvh_nodes[node_base + 1];
            if bounds_distance(min_x, min_y, max_x, max_y, local_pt[0], local_pt[1]) > best_dist {
                node_id = skip;
            } else {
                let left = group_bvh_nodes[node_base];
                let start = group_bvh_nodes[node_base + 2];
                let count = group_bvh_nodes[node_base + 3];
                if count > u32::new(0) {
                    let mut i = u32::new(0);
                    while i < count {
                        let shape_index = group_bvh_indices[(index_offset + start + i) as usize];
                        let shape_xform_base = (shape_index * XFORM_STRIDE) as usize;
                        let shape_px = shape_xform[shape_xform_base] * local_pt[0]
                            + shape_xform[shape_xform_base + 1] * local_pt[1]
                            + shape_xform[shape_xform_base + 2];
                        let shape_py = shape_xform[shape_xform_base + 3] * local_pt[0]
                            + shape_xform[shape_xform_base + 4] * local_pt[1]
                            + shape_xform[shape_xform_base + 5];

                        let mut local_closest = Line::empty(2usize);
                        let mut base_point = u32::new(0);
                        let mut t = f32::new(0.0);
                        if closest_point_shape(
                            shape_data,
                            curve_data,
                            shape_index,
                            shape_px,
                            shape_py,
                            &mut local_closest,
                            &mut base_point,
                            &mut t,
                        ) != u32::new(0)
                        {
                            let shape_t_base = (shape_index * XFORM_STRIDE) as usize;
                            let local_group = xform_pt_affine(
                                shape_transform[shape_t_base],
                                shape_transform[shape_t_base + 1],
                                shape_transform[shape_t_base + 2],
                                shape_transform[shape_t_base + 3],
                                shape_transform[shape_t_base + 4],
                                shape_transform[shape_t_base + 5],
                                local_closest[0],
                                local_closest[1],
                            );
                            let gs_base = (group_id * XFORM_STRIDE) as usize;
                            let closest_canvas = xform_pt_affine(
                                group_shape_xform[gs_base],
                                group_shape_xform[gs_base + 1],
                                group_shape_xform[gs_base + 2],
                                group_shape_xform[gs_base + 3],
                                group_shape_xform[gs_base + 4],
                                group_shape_xform[gs_base + 5],
                                local_group[0],
                                local_group[1],
                            );
                            let dx = closest_canvas[0] - px;
                            let dy = closest_canvas[1] - py;
                            let dist = vec2_length(dx, dy);
                            if dist < best_dist {
                                best_dist = dist;
                                best_shape = shape_index;
                                best_base = base_point;
                                best_t = t;
                                best_x = local_closest[0];
                                best_y = local_closest[1];
                                found = u32::new(1);
                            }
                        }
                        i += u32::new(1);
                    }
                    node_id = skip;
                } else {
                    node_id = left;
                }
            }
        }
    } else {
        let group_base = (group_id * GROUP_STRIDE) as usize;
        let shape_offset = group_data[group_base] as u32;
        let shape_count = group_data[group_base + 1] as u32;
        let mut i = u32::new(0);
        while i < shape_count {
            let shape_index = group_shapes[(shape_offset + i) as usize] as u32;
            let shape_xform_base = (shape_index * XFORM_STRIDE) as usize;
            let shape_px = shape_xform[shape_xform_base] * local_pt[0]
                + shape_xform[shape_xform_base + 1] * local_pt[1]
                + shape_xform[shape_xform_base + 2];
            let shape_py = shape_xform[shape_xform_base + 3] * local_pt[0]
                + shape_xform[shape_xform_base + 4] * local_pt[1]
                + shape_xform[shape_xform_base + 5];

            let mut local_closest = Line::empty(2usize);
            let mut base_point = u32::new(0);
            let mut t = f32::new(0.0);
            if closest_point_shape(
                shape_data,
                curve_data,
                shape_index,
                shape_px,
                shape_py,
                &mut local_closest,
                &mut base_point,
                &mut t,
            ) != u32::new(0)
            {
                let shape_t_base = (shape_index * XFORM_STRIDE) as usize;
                let local_group = xform_pt_affine(
                    shape_transform[shape_t_base],
                    shape_transform[shape_t_base + 1],
                    shape_transform[shape_t_base + 2],
                    shape_transform[shape_t_base + 3],
                    shape_transform[shape_t_base + 4],
                    shape_transform[shape_t_base + 5],
                    local_closest[0],
                    local_closest[1],
                );
                let gs_base = (group_id * XFORM_STRIDE) as usize;
                let closest_canvas = xform_pt_affine(
                    group_shape_xform[gs_base],
                    group_shape_xform[gs_base + 1],
                    group_shape_xform[gs_base + 2],
                    group_shape_xform[gs_base + 3],
                    group_shape_xform[gs_base + 4],
                    group_shape_xform[gs_base + 5],
                    local_group[0],
                    local_group[1],
                );
                let dx = closest_canvas[0] - px;
                let dy = closest_canvas[1] - py;
                let dist = vec2_length(dx, dy);
                if dist < best_dist {
                    best_dist = dist;
                    best_shape = shape_index;
                    best_base = base_point;
                    best_t = t;
                    best_x = local_closest[0];
                    best_y = local_closest[1];
                    found = u32::new(1);
                }
            }
            i += u32::new(1);
        }
    }

    if found != u32::new(0) {
        out_local[0] = best_x;
        out_local[1] = best_y;
        *out_dist = best_dist;
        *out_shape = best_shape;
        *out_base = best_base;
        *out_t = best_t;
    } else {
        *out_dist = if max_radius < big { max_radius } else { big };
        *out_shape = u32::new(0);
        *out_base = u32::new(0);
        *out_t = f32::new(0.0);
    }
    found
}
