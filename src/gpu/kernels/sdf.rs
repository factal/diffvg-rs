//! GPU kernels for signed distance field evaluation.

use cubecl::prelude::*;
use crate::gpu::constants::*;
use super::{rng::*, distance::compute_distance_group, backward::fill_inside_group};

/// Compute signed distance at a single point in canvas space.
#[cube]
fn compute_signed_distance(
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
    num_groups: u32,
    px: f32,
    py: f32,
) -> f32 {
    let big = f32::new(1.0e20);
    let mut best_dist = big;
    let mut best_group = u32::new(0);
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
            best_group = group_id;
            found = u32::new(1);
        }
    }

    if found == u32::new(0) {
        f32::new(0.0)
    } else {
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

        if inside != u32::new(0) {
            -best_dist
        } else {
            best_dist
        }
    }
}

/// Render per-pixel signed distance field by averaging sample distances.
#[cube(launch_unchecked)]
pub(crate) fn render_sdf_kernel(
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
    width: u32,
    height: u32,
    num_groups: u32,
    samples_x: u32,
    samples_y: u32,
    seed: u32,
    jitter: u32,
    output: &mut Array<f32>,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;
    if x >= width || y >= height {
        terminate!();
    }
    let samples_per_pixel = samples_x * samples_y;
    if samples_per_pixel == u32::new(0) {
        terminate!();
    }

    let inv_sx = f32::new(1.0) / f32::cast_from(samples_x);
    let inv_sy = f32::new(1.0) / f32::cast_from(samples_y);
    let half = f32::new(0.5);
    let mut accum = f32::new(0.0);

    let mut sy = u32::new(0);
    while sy < samples_y {
        let mut sx = u32::new(0);
        while sx < samples_x {
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
            let dist = compute_signed_distance(
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
            );
            accum += dist;
            sx += u32::new(1);
        }
        sy += u32::new(1);
    }

    let inv_samples = f32::new(1.0) / f32::cast_from(samples_per_pixel);
    let idx = (y * width + x) as usize;
    output[idx] = accum * inv_samples;
}

/// Evaluate SDF values at arbitrary positions (canvas space).
#[cube(launch_unchecked)]
pub(crate) fn eval_positions_kernel(
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
    positions: &Array<f32>,
    position_count: u32,
    num_groups: u32,
    output: &mut Array<f32>,
) {
    let idx = ABSOLUTE_POS;
    if idx >= position_count as usize {
        terminate!();
    }
    let base = idx * 2;
    let px = positions[base];
    let py = positions[base + 1];
    let dist = compute_signed_distance(
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
    );
    output[idx] = dist;
}
