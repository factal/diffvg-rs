use cubecl::prelude::*;
use crate::gpu::constants::*;
use super::{math::*, rng::*, sampling::*, curves::*, distance::*};

/// Accumulate filter weights per sample using fixed-point atomics.
#[cube(launch_unchecked)]
pub(crate) fn rasterize_weights(
    width: u32,
    height: u32,
    filter_type: u32,
    filter_radius: f32,
    filter_radius_i: u32,
    samples_x: u32,
    samples_y: u32,
    seed: u32,
    jitter: u32,
    weight_accum: &mut Array<Atomic<u32>>,
) {
    let idx = ABSOLUTE_POS;
    let samples_per_pixel = samples_x * samples_y;
    if samples_per_pixel == u32::new(0) {
        terminate!();
    }

    let total_samples = width * height * samples_x * samples_y;
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
    if jitter != 0 {
        // PCG-based jitter for stratified sampling.
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

    let ri = filter_radius_i as i32;
    let scale = f32::new(ACCUM_SCALE);
    let half = f32::new(0.5);
    let zero = f32::new(0.0);

    for dy in -ri..=ri {
        for dx in -ri..=ri {
            let xx = x as i32 + dx;
            let yy = y as i32 + dy;
            if xx >= 0 && yy >= 0 && xx < width as i32 && yy < height as i32 {
                let xc = f32::cast_from(xx) + half;
                let yc = f32::cast_from(yy) + half;
                let w = filter_weight(filter_type, xc - px, yc - py, filter_radius);
                if w > zero {
                    let w_scaled = w * scale;
                    let w_fixed = (w_scaled + half) as u32;
                    if w_fixed > 0 {
                        let base = (yy as u32 * width + xx as u32) as usize;
                        weight_accum[base].fetch_add(w_fixed);
                    }
                }
            }
        }
    }
}

/// Render samples into a splat buffer using tile binning.
#[cube(launch_unchecked)]
pub(crate) fn rasterize_splat(
    shape_data: &Array<f32>,
    segment_data: &Array<f32>,
    shape_bounds: &Array<f32>,
    group_data: &Array<f32>,
    group_xform: &Array<f32>,
    group_inv_scale: &Array<f32>,
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
    tile_offsets: &Array<u32>,
    tile_entries: &Array<u32>,
    tile_order: &Array<u32>,
    tile_count_x: u32,
    tile_count_y: u32,
    tile_size: u32,
    width: u32,
    height: u32,
    _num_groups: u32,
    background_image: &Array<f32>,
    has_background_image: u32,
    background_r: f32,
    background_g: f32,
    background_b: f32,
    background_a: f32,
    filter_type: u32,
    filter_radius: f32,
    filter_radius_i: u32,
    use_prefiltering: u32,
    samples_x: u32,
    samples_y: u32,
    seed: u32,
    jitter: u32,
    weight_accum: &Array<Atomic<u32>>,
    color_accum: &mut Array<Atomic<u32>>,
) {
    let idx = ABSOLUTE_POS;
    let samples_per_pixel = samples_x * samples_y;
    if samples_per_pixel == u32::new(0)
        || tile_size == u32::new(0)
        || tile_count_x == u32::new(0)
        || tile_count_y == u32::new(0)
    {
        terminate!();
    }

    let tile_pixels = tile_size * tile_size;
    let tile_samples = tile_pixels * samples_per_pixel;
    let total_samples = tile_samples * tile_count_x * tile_count_y;
    if idx >= total_samples as usize {
        terminate!();
    }

    let idx_u32 = idx as u32;
    let tile_slot = idx_u32 / tile_samples;
    let local_idx = idx_u32 - tile_slot * tile_samples;
    let tile_id = tile_order[tile_slot as usize];
    let pixel_index = local_idx / samples_per_pixel;
    let sample_index = local_idx - pixel_index * samples_per_pixel;

    let lx = pixel_index % tile_size;
    let ly = pixel_index / tile_size;
    let tile_x = tile_id % tile_count_x;
    let tile_y = tile_id / tile_count_x;
    let x = tile_x * tile_size + lx;
    let y = tile_y * tile_size + ly;

    if x >= width || y >= height {
        terminate!();
    }

    let sx = sample_index % samples_x;
    let sy = sample_index / samples_x;

    let inv_sx = f32::new(1.0) / f32::cast_from(samples_x);
    let inv_sy = f32::new(1.0) / f32::cast_from(samples_y);

    let half = f32::new(0.5);
    let mut rx = half;
    let mut ry = half;
    if jitter != 0 {
        // Deterministic jitter per sample for reproducible rendering.
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

    let mut bg_r = background_r;
    let mut bg_g = background_g;
    let mut bg_b = background_b;
    let mut bg_a = background_a;
    if has_background_image != u32::new(0) {
        let idx4 = ((y * width + x) as usize) * 4;
        bg_r = background_image[idx4];
        bg_g = background_image[idx4 + 1];
        bg_b = background_image[idx4 + 2];
        bg_a = background_image[idx4 + 3];
    }

    let color = eval_scene_tiled(
        shape_data,
        segment_data,
        shape_bounds,
        group_data,
        group_xform,
        group_inv_scale,
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
        tile_offsets,
        tile_entries,
        tile_id,
        px,
        py,
        bg_r,
        bg_g,
        bg_b,
        bg_a,
        use_prefiltering,
    );

    let ri = filter_radius_i as i32;
    let scale = f32::new(ACCUM_SCALE);
    let half = f32::new(0.5);
    let zero = f32::new(0.0);

    for dy in -ri..=ri {
        for dx in -ri..=ri {
            let xx = x as i32 + dx;
            let yy = y as i32 + dy;
            if xx >= 0 && yy >= 0 && xx < width as i32 && yy < height as i32 {
                let base = (yy as u32 * width + xx as u32) as usize;
                let weight_fixed = weight_accum[base].load();
                if weight_fixed > 0 {
                    let xc = f32::cast_from(xx) + half;
                    let yc = f32::cast_from(yy) + half;
                    let w = filter_weight(filter_type, xc - px, yc - py, filter_radius);
                    if w > zero {
                        let inv_weight = scale / f32::cast_from(weight_fixed);
                        let w_norm = w * inv_weight;
                        let c0 = ((color[0] * w_norm * scale) + half) as u32;
                        let c1 = ((color[1] * w_norm * scale) + half) as u32;
                        let c2 = ((color[2] * w_norm * scale) + half) as u32;
                        let c3 = ((color[3] * w_norm * scale) + half) as u32;
                        let idx4 = base * 4;
                        color_accum[idx4].fetch_add(c0);
                        color_accum[idx4 + 1].fetch_add(c1);
                        color_accum[idx4 + 2].fetch_add(c2);
                        color_accum[idx4 + 3].fetch_add(c3);
                    }
                }
            }
        }
    }
}

/// Resolve fixed-point splat buffers into the final RGBA output.
#[cube(launch_unchecked)]
pub(crate) fn resolve_splat(
    weight_accum: &Array<Atomic<u32>>,
    color_accum: &Array<Atomic<u32>>,
    width: u32,
    height: u32,
    background_r: f32,
    background_g: f32,
    background_b: f32,
    background_a: f32,
    output: &mut Array<f32>,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;
    if x >= width || y >= height {
        terminate!();
    }
    let idx = (y * width + x) as usize;
    let weight = weight_accum[idx].load();
    let mut r = background_r;
    let mut g = background_g;
    let mut b = background_b;
    let mut a = background_a;
    if weight > 0 {
        let inv = f32::new(1.0) / f32::new(ACCUM_SCALE);
        let base = idx * 4;
        r = f32::cast_from(color_accum[base].load()) * inv;
        g = f32::cast_from(color_accum[base + 1].load()) * inv;
        b = f32::cast_from(color_accum[base + 2].load()) * inv;
        a = f32::cast_from(color_accum[base + 3].load()) * inv;
    }
    if a > f32::new(1.0e-8) {
        r = r / a;
        g = g / a;
        b = b / a;
    }
    let out = idx * 4;
    output[out] = r;
    output[out + 1] = g;
    output[out + 2] = b;
    output[out + 3] = a;
}

/// Resolve float splat buffers into the final RGBA output.
#[cube(launch_unchecked)]
pub(crate) fn resolve_splat_f32(
    weight_accum: &Array<Atomic<f32>>,
    color_accum: &Array<Atomic<f32>>,
    width: u32,
    height: u32,
    background_r: f32,
    background_g: f32,
    background_b: f32,
    background_a: f32,
    output: &mut Array<f32>,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;
    if x >= width || y >= height {
        terminate!();
    }
    let idx = (y * width + x) as usize;
    let weight = weight_accum[idx].load();
    let mut r = background_r;
    let mut g = background_g;
    let mut b = background_b;
    let mut a = background_a;
    if weight > f32::new(0.0) {
        let base = idx * 4;
        r = color_accum[base].load();
        g = color_accum[base + 1].load();
        b = color_accum[base + 2].load();
        a = color_accum[base + 3].load();
    }
    if a > f32::new(1.0e-8) {
        r = r / a;
        g = g / a;
        b = b / a;
    }
    let out = idx * 4;
    output[out] = r;
    output[out + 1] = g;
    output[out + 2] = b;
    output[out + 3] = a;
}

#[cube]
pub(super) fn blend_group(
    out: &mut Line<f32>,
    fill_kind: u32,
    stroke_kind: u32,
    fill_rule: u32,
    fill_color: Line<f32>,
    stroke_color: Line<f32>,
    fill_min_dist: f32,
    fill_winding: f32,
    fill_crossings: f32,
    stroke_min_dist: f32,
    stroke_min_radius: f32,
    stroke_hit: f32,
    use_prefiltering: u32,
) {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let big = f32::new(1.0e20);
    let use_prefiltering = use_prefiltering != u32::new(0);

    // Stroke is composited before fill within the same group.
    if stroke_kind != PAINT_NONE {
        let mut coverage = zero;
        if use_prefiltering {
            if stroke_min_dist < big && stroke_min_radius > zero {
                coverage = smoothstep_unit(stroke_min_dist + stroke_min_radius)
                    - smoothstep_unit(stroke_min_dist - stroke_min_radius);
            }
        } else if stroke_hit > zero {
            coverage = one;
        }
        if coverage > zero {
            let alpha = coverage * stroke_color[3];
            let inv = one - alpha;
            out[0] = stroke_color[0] * alpha + out[0] * inv;
            out[1] = stroke_color[1] * alpha + out[1] * inv;
            out[2] = stroke_color[2] * alpha + out[2] * inv;
            out[3] = alpha + out[3] * inv;
        }
    }

    if fill_kind != PAINT_NONE {
        let inside = if fill_rule == u32::new(1) {
            fill_crossings > zero
        } else {
            fill_winding != zero
        };
        let mut coverage = zero;
        if use_prefiltering {
            if fill_min_dist < big {
                let signed = if inside { fill_min_dist } else { -fill_min_dist };
                coverage = smoothstep_unit(signed);
            }
        } else if inside {
            coverage = one;
        }
        if coverage > zero {
            let alpha = coverage * fill_color[3];
            let inv = one - alpha;
            out[0] = fill_color[0] * alpha + out[0] * inv;
            out[1] = fill_color[1] * alpha + out[1] * inv;
            out[2] = fill_color[2] * alpha + out[2] * inv;
            out[3] = alpha + out[3] * inv;
        }
    }
}

#[cube]
pub(super) fn accumulate_shape_fill(
    shape_data: &Array<f32>,
    segment_data: &Array<f32>,
    curve_data: &Array<f32>,
    path_bvh_bounds: &Array<f32>,
    path_bvh_nodes: &Array<u32>,
    path_bvh_indices: &Array<u32>,
    path_bvh_meta: &Array<u32>,
    shape_index: u32,
    fill_rule: u32,
    px: f32,
    py: f32,
    min_dist: &mut f32,
    winding: &mut f32,
    crossings: &mut f32,
) {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let big = f32::new(1.0e20);
    let base = (shape_index * SHAPE_STRIDE) as usize;
    let kind = shape_data[base] as u32;
    let seg_offset = shape_data[base + 1] as u32;
    let seg_count = shape_data[base + 2] as u32;
    let curve_offset = shape_data[base + 12] as u32;
    let curve_count = shape_data[base + 13] as u32;
    let use_distance_approx = shape_data[base + 14] > f32::new(0.5);
    let p0 = shape_data[base + 4];
    let p1 = shape_data[base + 5];
    let p2 = shape_data[base + 6];
    let p3 = shape_data[base + 7];

    if kind == SHAPE_KIND_CIRCLE {
        let dx = px - p0;
        let dy = py - p1;
        let radius = p2;
        let dist = (dx * dx + dy * dy).sqrt() - radius;
        let abs_dist = abs_f32(dist);
        if abs_dist < *min_dist {
            *min_dist = abs_dist;
        }
        if dist <= zero {
            if fill_rule == u32::new(1) {
                *crossings = one - *crossings;
            } else {
                *winding += one;
            }
        }
    } else if kind == SHAPE_KIND_ELLIPSE {
        let dx = (px - p0) / max_f32(p2.abs(), f32::new(1.0e-3));
        let dy = (py - p1) / max_f32(p3.abs(), f32::new(1.0e-3));
        let len = (dx * dx + dy * dy).sqrt();
        let scale = min_f32(p2.abs(), p3.abs());
        let dist = (len - one) * scale;
        let abs_dist = abs_f32(dist);
        if abs_dist < *min_dist {
            *min_dist = abs_dist;
        }
        if dist <= zero {
            if fill_rule == u32::new(1) {
                *crossings = one - *crossings;
            } else {
                *winding += one;
            }
        }
    } else if kind == SHAPE_KIND_RECT {
        let min_x = p0;
        let min_y = p1;
        let max_x = p2;
        let max_y = p3;
        let dx = max_f32(max_f32(min_x - px, zero), px - max_x);
        let dy = max_f32(max_f32(min_y - py, zero), py - max_y);
        let outside = (dx * dx + dy * dy).sqrt();
        let inside = min_f32(
            min_f32(px - min_x, max_x - px),
            min_f32(py - min_y, max_y - py),
        );
        let dist = if outside > zero { outside } else { -inside };
        let abs_dist = abs_f32(dist);
        if abs_dist < *min_dist {
            *min_dist = abs_dist;
        }
        if dist <= zero {
            if fill_rule == u32::new(1) {
                *crossings = one - *crossings;
            } else {
                *winding += one;
            }
        }
    } else if kind == SHAPE_KIND_PATH {
        let meta_base = (shape_index * BVH_META_STRIDE) as usize;
        let node_count = path_bvh_meta[meta_base + 1];
        let mut local_min = big;
        let mut local_winding = zero;
        let mut local_crossings = zero;
        if node_count > u32::new(0) {
            let node_offset = path_bvh_meta[meta_base];
            let index_offset = path_bvh_meta[meta_base + 2];
            accumulate_path_fill_bvh(
                curve_data,
                path_bvh_bounds,
                path_bvh_nodes,
                path_bvh_indices,
                node_offset,
                index_offset,
                px,
                py,
                fill_rule,
                use_distance_approx,
                &mut local_min,
                &mut local_winding,
                &mut local_crossings,
            );
        } else if curve_count > 0 {
            accumulate_path_fill_full(
                curve_data,
                curve_offset,
                curve_count,
                px,
                py,
                fill_rule,
                use_distance_approx,
                &mut local_min,
                &mut local_winding,
                &mut local_crossings,
            );
        } else if seg_count > 0 {
            let mut s = u32::new(0);
            while s < seg_count {
                let seg_base = ((seg_offset + s) * SEGMENT_STRIDE) as usize;
                let x0 = segment_data[seg_base];
                let y0 = segment_data[seg_base + 1];
                let x1 = segment_data[seg_base + 2];
                let y1 = segment_data[seg_base + 3];
                let dist = distance_to_segment(px, py, x0, y0, x1, y1);
                if dist < *min_dist {
                    *min_dist = dist;
                }
                winding_and_crossings_line(
                    px,
                    py,
                    x0,
                    y0,
                    x1,
                    y1,
                    fill_rule,
                    winding,
                    crossings,
                );
                s += u32::new(1);
            }
        }
        if local_min < *min_dist {
            *min_dist = local_min;
        }
        if fill_rule == u32::new(1) {
            if local_crossings > zero {
                *crossings = one - *crossings;
            }
        } else {
            *winding += local_winding;
        }
    }
}

#[cube]
pub(super) fn accumulate_shape_stroke(
    shape_data: &Array<f32>,
    segment_data: &Array<f32>,
    curve_data: &Array<f32>,
    path_bvh_bounds: &Array<f32>,
    path_bvh_nodes: &Array<u32>,
    path_bvh_indices: &Array<u32>,
    path_bvh_meta: &Array<u32>,
    shape_index: u32,
    px: f32,
    py: f32,
    use_prefiltering: u32,
    min_dist: &mut f32,
    min_radius: &mut f32,
    hit: &mut f32,
) {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let big = f32::new(1.0e20);
    let base = (shape_index * SHAPE_STRIDE) as usize;
    let kind = shape_data[base] as u32;
    let seg_offset = shape_data[base + 1] as u32;
    let seg_count = shape_data[base + 2] as u32;
    let stroke_width = shape_data[base + 3];
    let curve_offset = shape_data[base + 12] as u32;
    let curve_count = shape_data[base + 13] as u32;
    let use_distance_approx = shape_data[base + 14] > f32::new(0.5);
    let p0 = shape_data[base + 4];
    let p1 = shape_data[base + 5];
    let p2 = shape_data[base + 6];
    let p3 = shape_data[base + 7];

    if !(stroke_width <= zero && curve_count == 0 && seg_count == 0) {
        if kind == SHAPE_KIND_CIRCLE {
            let dx = px - p0;
            let dy = py - p1;
            let radius = p2;
            let dist = abs_f32((dx * dx + dy * dy).sqrt() - radius);
            if use_prefiltering != u32::new(0) {
                if dist < *min_dist {
                    *min_dist = dist;
                    *min_radius = stroke_width;
                }
            } else if dist < stroke_width {
                *hit = one;
            }
        } else if kind == SHAPE_KIND_ELLIPSE {
            let dx = (px - p0) / max_f32(p2.abs(), f32::new(1.0e-3));
            let dy = (py - p1) / max_f32(p3.abs(), f32::new(1.0e-3));
            let len = (dx * dx + dy * dy).sqrt();
            let scale = min_f32(p2.abs(), p3.abs());
            let dist = abs_f32((len - one) * scale);
            if use_prefiltering != u32::new(0) {
                if dist < *min_dist {
                    *min_dist = dist;
                    *min_radius = stroke_width;
                }
            } else if dist < stroke_width {
                *hit = one;
            }
        } else if kind == SHAPE_KIND_RECT {
            let min_x = p0;
            let min_y = p1;
            let max_x = p2;
            let max_y = p3;
            let dx = max_f32(max_f32(min_x - px, zero), px - max_x);
            let dy = max_f32(max_f32(min_y - py, zero), py - max_y);
            let outside = (dx * dx + dy * dy).sqrt();
            let inside = min_f32(
                min_f32(px - min_x, max_x - px),
                min_f32(py - min_y, max_y - py),
            );
            let dist = abs_f32(if outside > zero { outside } else { -inside });
            if use_prefiltering != u32::new(0) {
                if dist < *min_dist {
                    *min_dist = dist;
                    *min_radius = stroke_width;
                }
            } else if dist < stroke_width {
                *hit = one;
            }
        } else if kind == SHAPE_KIND_PATH {
            let meta_base = (shape_index * BVH_META_STRIDE) as usize;
            let node_count = path_bvh_meta[meta_base + 1];
            let mut local_min = big;
            let mut local_radius = zero;
            let mut local_hit = zero;
            if node_count > u32::new(0) {
                let node_offset = path_bvh_meta[meta_base];
                let index_offset = path_bvh_meta[meta_base + 2];
                accumulate_path_stroke_bvh(
                    curve_data,
                    path_bvh_bounds,
                    path_bvh_nodes,
                    path_bvh_indices,
                    node_offset,
                    index_offset,
                    px,
                    py,
                    use_distance_approx,
                    use_prefiltering,
                    &mut local_min,
                    &mut local_radius,
                    &mut local_hit,
                );
            } else if curve_count > 0 {
                accumulate_path_stroke_full(
                    curve_data,
                    curve_offset,
                    curve_count,
                    px,
                    py,
                    use_distance_approx,
                    use_prefiltering,
                    &mut local_min,
                    &mut local_radius,
                    &mut local_hit,
                );
            } else if seg_count > 0 {
                let mut s = u32::new(0);
                while s < seg_count {
                    let seg_base = ((seg_offset + s) * SEGMENT_STRIDE) as usize;
                    let x0 = segment_data[seg_base];
                    let y0 = segment_data[seg_base + 1];
                    let x1 = segment_data[seg_base + 2];
                    let y1 = segment_data[seg_base + 3];
                    let r0 = segment_data[seg_base + 4];
                    let r1 = segment_data[seg_base + 5];

                    let dist_t = distance_to_segment_with_t(px, py, x0, y0, x1, y1);
                    let dist = dist_t[0];
                    let t = dist_t[1];
                    let radius = r0 + t * (r1 - r0);

                    if use_prefiltering != u32::new(0) {
                        if dist < *min_dist {
                            *min_dist = dist;
                            *min_radius = radius;
                        }
                    } else if dist < radius {
                        *hit = one;
                    }
                    s += u32::new(1);
                }
            }
            if use_prefiltering != u32::new(0) {
                if local_min < *min_dist {
                    *min_dist = local_min;
                    *min_radius = local_radius;
                }
            } else if local_hit > zero {
                *hit = one;
            }
        }
    }
}

#[cube]
pub(super) fn accumulate_path_fill_full(
    curve_data: &Array<f32>,
    curve_offset: u32,
    curve_count: u32,
    px: f32,
    py: f32,
    fill_rule: u32,
    use_distance_approx: bool,
    min_dist: &mut f32,
    winding: &mut f32,
    crossings: &mut f32,
) {
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

        let dist = if seg_kind == 0 {
            distance_to_segment(px, py, x0, y0, x1, y1)
        } else if seg_kind == 1 {
            distance_to_quadratic(px, py, x0, y0, x1, y1, x2, y2, use_distance_approx)
        } else {
            distance_to_cubic(px, py, x0, y0, x1, y1, x2, y2, x3, y3, use_distance_approx)
        };
        if dist < *min_dist {
            *min_dist = dist;
        }

        if seg_kind == 0 {
            winding_and_crossings_line(
                px,
                py,
                x0,
                y0,
                x1,
                y1,
                fill_rule,
                winding,
                crossings,
            );
        } else if seg_kind == 1 {
            winding_and_crossings_quadratic(
                px,
                py,
                x0,
                y0,
                x1,
                y1,
                x2,
                y2,
                fill_rule,
                winding,
                crossings,
            );
        } else {
            winding_and_crossings_cubic(
                px,
                py,
                x0,
                y0,
                x1,
                y1,
                x2,
                y2,
                x3,
                y3,
                fill_rule,
                winding,
                crossings,
            );
        }
        s += u32::new(1);
    }
}

#[cube]
pub(super) fn accumulate_path_stroke_full(
    curve_data: &Array<f32>,
    curve_offset: u32,
    curve_count: u32,
    px: f32,
    py: f32,
    use_distance_approx: bool,
    use_prefiltering: u32,
    min_dist: &mut f32,
    min_radius: &mut f32,
    hit: &mut f32,
) {
    let one = f32::new(1.0);
    let mut done = u32::new(0);
    let mut s = u32::new(0);
    while s < curve_count {
        if done == u32::new(0) {
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
            let r0 = curve_data[seg_base + 9];
            let r1 = curve_data[seg_base + 10];
            let r2 = curve_data[seg_base + 11];
            let r3 = curve_data[seg_base + 12];

            let dist_t = if seg_kind == 0 {
                distance_to_segment_with_t(px, py, x0, y0, x1, y1)
            } else if seg_kind == 1 {
                closest_point_quadratic_with_t(
                    px, py, x0, y0, x1, y1, x2, y2, use_distance_approx,
                )
            } else {
                closest_point_cubic_with_t(
                    px, py, x0, y0, x1, y1, x2, y2, x3, y3, use_distance_approx,
                )
            };
            let dist = dist_t[0];
            let t = dist_t[1];
            let radius = if seg_kind == 0 {
                r0 + t * (r1 - r0)
            } else if seg_kind == 1 {
                let tt = one - t;
                tt * tt * r0 + f32::new(2.0) * tt * t * r1 + t * t * r2
            } else {
                let tt = one - t;
                tt * tt * tt * r0
                    + f32::new(3.0) * tt * tt * t * r1
                    + f32::new(3.0) * tt * t * t * r2
                    + t * t * t * r3
            };

            if use_prefiltering != u32::new(0) {
                if dist < *min_dist {
                    *min_dist = dist;
                    *min_radius = radius;
                }
            } else if dist < radius {
                *hit = one;
                done = u32::new(1);
            }
        }
        s += u32::new(1);
    }
}

#[cube]
pub(super) fn accumulate_path_fill_bvh(
    curve_data: &Array<f32>,
    path_bvh_bounds: &Array<f32>,
    path_bvh_nodes: &Array<u32>,
    path_bvh_indices: &Array<u32>,
    node_offset: u32,
    index_offset: u32,
    px: f32,
    py: f32,
    fill_rule: u32,
    use_distance_approx: bool,
    min_dist: &mut f32,
    winding: &mut f32,
    crossings: &mut f32,
) {
    let mut node_id = u32::new(0);
    while node_id != BVH_NONE {
        let node_base = ((node_offset + node_id) * BVH_NODE_STRIDE) as usize;
        let min_x = path_bvh_bounds[node_base];
        let min_y = path_bvh_bounds[node_base + 1];
        let max_x = path_bvh_bounds[node_base + 2];
        let max_y = path_bvh_bounds[node_base + 3];
        let dist = bounds_distance(min_x, min_y, max_x, max_y, px, py);
        if dist <= *min_dist {
            let left = path_bvh_nodes[node_base];
            let skip = path_bvh_nodes[node_base + 1];
            let start = path_bvh_nodes[node_base + 2];
            let count = path_bvh_nodes[node_base + 3];
            if count > u32::new(0) {
                let mut i = u32::new(0);
                while i < count {
                    let seg_index = path_bvh_indices[(index_offset + start + i) as usize];
                    let seg_base = (seg_index * CURVE_STRIDE) as usize;
                    let seg_kind = curve_data[seg_base] as u32;
                    let x0 = curve_data[seg_base + 1];
                    let y0 = curve_data[seg_base + 2];
                    let x1 = curve_data[seg_base + 3];
                    let y1 = curve_data[seg_base + 4];
                    let x2 = curve_data[seg_base + 5];
                    let y2 = curve_data[seg_base + 6];
                    let x3 = curve_data[seg_base + 7];
                    let y3 = curve_data[seg_base + 8];
                    let seg_dist = if seg_kind == 0 {
                        distance_to_segment(px, py, x0, y0, x1, y1)
                    } else if seg_kind == 1 {
                        distance_to_quadratic(px, py, x0, y0, x1, y1, x2, y2, use_distance_approx)
                    } else {
                        distance_to_cubic(px, py, x0, y0, x1, y1, x2, y2, x3, y3, use_distance_approx)
                    };
                    if seg_dist < *min_dist {
                        *min_dist = seg_dist;
                    }
                    i += u32::new(1);
                }
                node_id = skip;
            } else {
                node_id = left;
            }
        } else {
            let skip = path_bvh_nodes[node_base + 1];
            node_id = skip;
        }
    }
    node_id = u32::new(0);
    while node_id != BVH_NONE {
        let node_base = ((node_offset + node_id) * BVH_NODE_STRIDE) as usize;
        let min_x = path_bvh_bounds[node_base];
        let min_y = path_bvh_bounds[node_base + 1];
        let max_x = path_bvh_bounds[node_base + 2];
        let max_y = path_bvh_bounds[node_base + 3];
        if ray_intersects_bounds(min_x, min_y, max_x, max_y, px, py) {
            let left = path_bvh_nodes[node_base];
            let skip = path_bvh_nodes[node_base + 1];
            let start = path_bvh_nodes[node_base + 2];
            let count = path_bvh_nodes[node_base + 3];
            if count > u32::new(0) {
                let mut i = u32::new(0);
                while i < count {
                    let seg_index = path_bvh_indices[(index_offset + start + i) as usize];
                    let seg_base = (seg_index * CURVE_STRIDE) as usize;
                    let seg_kind = curve_data[seg_base] as u32;
                    let x0 = curve_data[seg_base + 1];
                    let y0 = curve_data[seg_base + 2];
                    let x1 = curve_data[seg_base + 3];
                    let y1 = curve_data[seg_base + 4];
                    let x2 = curve_data[seg_base + 5];
                    let y2 = curve_data[seg_base + 6];
                    let x3 = curve_data[seg_base + 7];
                    let y3 = curve_data[seg_base + 8];
                    if seg_kind == 0 {
                        winding_and_crossings_line(
                            px,
                            py,
                            x0,
                            y0,
                            x1,
                            y1,
                            fill_rule,
                            winding,
                            crossings,
                        );
                    } else if seg_kind == 1 {
                        winding_and_crossings_quadratic(
                            px,
                            py,
                            x0,
                            y0,
                            x1,
                            y1,
                            x2,
                            y2,
                            fill_rule,
                            winding,
                            crossings,
                        );
                    } else {
                        winding_and_crossings_cubic(
                            px,
                            py,
                            x0,
                            y0,
                            x1,
                            y1,
                            x2,
                            y2,
                            x3,
                            y3,
                            fill_rule,
                            winding,
                            crossings,
                        );
                    }
                    i += u32::new(1);
                }
                node_id = skip;
            } else {
                node_id = left;
            }
        } else {
            let skip = path_bvh_nodes[node_base + 1];
            node_id = skip;
        }
    }
}

#[cube]
pub(super) fn accumulate_path_stroke_bvh(
    curve_data: &Array<f32>,
    path_bvh_bounds: &Array<f32>,
    path_bvh_nodes: &Array<u32>,
    path_bvh_indices: &Array<u32>,
    node_offset: u32,
    index_offset: u32,
    px: f32,
    py: f32,
    use_distance_approx: bool,
    use_prefiltering: u32,
    min_dist: &mut f32,
    min_radius: &mut f32,
    hit: &mut f32,
) {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let mut node_id = u32::new(0);
    let mut done = u32::new(0);
    while node_id != BVH_NONE && done == u32::new(0) {
        let node_base = ((node_offset + node_id) * BVH_NODE_STRIDE) as usize;
        let min_x = path_bvh_bounds[node_base];
        let min_y = path_bvh_bounds[node_base + 1];
        let max_x = path_bvh_bounds[node_base + 2];
        let max_y = path_bvh_bounds[node_base + 3];
        let mut intersects = u32::new(1);
        if use_prefiltering != u32::new(0) {
            if bounds_distance(min_x, min_y, max_x, max_y, px, py) > *min_dist {
                intersects = u32::new(0);
            }
        } else if !bounds_contains(min_x, min_y, max_x, max_y, px, py) {
            intersects = u32::new(0);
        }
        let skip = path_bvh_nodes[node_base + 1];
        if intersects != u32::new(0) {
            let left = path_bvh_nodes[node_base];
            let start = path_bvh_nodes[node_base + 2];
            let count = path_bvh_nodes[node_base + 3];
            if count > u32::new(0) {
                let mut i = u32::new(0);
                while i < count {
                    if done == u32::new(0) {
                        let seg_index = path_bvh_indices[(index_offset + start + i) as usize];
                        let seg_base = (seg_index * CURVE_STRIDE) as usize;
                        let seg_kind = curve_data[seg_base] as u32;
                        let x0 = curve_data[seg_base + 1];
                        let y0 = curve_data[seg_base + 2];
                        let x1 = curve_data[seg_base + 3];
                        let y1 = curve_data[seg_base + 4];
                        let x2 = curve_data[seg_base + 5];
                        let y2 = curve_data[seg_base + 6];
                        let x3 = curve_data[seg_base + 7];
                        let y3 = curve_data[seg_base + 8];
                        let r0 = curve_data[seg_base + 9];
                        let r1 = curve_data[seg_base + 10];
                        let r2 = curve_data[seg_base + 11];
                        let r3 = curve_data[seg_base + 12];
                        let dist_t = if seg_kind == 0 {
                            distance_to_segment_with_t(px, py, x0, y0, x1, y1)
                        } else if seg_kind == 1 {
                            closest_point_quadratic_with_t(
                                px, py, x0, y0, x1, y1, x2, y2, use_distance_approx,
                            )
                        } else {
                            closest_point_cubic_with_t(
                                px, py, x0, y0, x1, y1, x2, y2, x3, y3, use_distance_approx,
                            )
                        };
                        let dist = dist_t[0];
                        let t = dist_t[1];
                        let radius = if seg_kind == 0 {
                            r0 + t * (r1 - r0)
                        } else if seg_kind == 1 {
                            let tt = one - t;
                            tt * tt * r0 + f32::new(2.0) * tt * t * r1 + t * t * r2
                        } else {
                            let tt = one - t;
                            tt * tt * tt * r0
                                + f32::new(3.0) * tt * tt * t * r1
                                + f32::new(3.0) * tt * t * t * r2
                                + t * t * t * r3
                        };
                        if use_prefiltering != u32::new(0) {
                            if dist < *min_dist {
                                *min_dist = dist;
                                *min_radius = radius;
                            }
                        } else if dist < radius {
                            *hit = one;
                            done = u32::new(1);
                        }
                    }
                    i += u32::new(1);
                }
                node_id = skip;
            } else {
                node_id = left;
            }
        } else {
            node_id = skip;
        }
        if use_prefiltering == u32::new(0) && *hit > zero {
            done = u32::new(1);
        }
    }
}

#[cube]
pub(super) fn eval_scene_tiled(
    shape_data: &Array<f32>,
    segment_data: &Array<f32>,
    shape_bounds: &Array<f32>,
    group_data: &Array<f32>,
    group_xform: &Array<f32>,
    group_inv_scale: &Array<f32>,
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
    tile_offsets: &Array<u32>,
    tile_entries: &Array<u32>,
    tile_id: u32,
    px: f32,
    py: f32,
    background_r: f32,
    background_g: f32,
    background_b: f32,
    background_a: f32,
    use_prefiltering: u32,
) -> Line<f32> {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let big = f32::new(1.0e20);

    let mut out = Line::empty(4usize);
    out[0] = background_r * background_a;
    out[1] = background_g * background_a;
    out[2] = background_b * background_a;
    out[3] = background_a;

    let start = tile_offsets[tile_id as usize];
    let end = tile_offsets[(tile_id + 1) as usize];

    let mut entry = start;
    while entry < end {
        let base = (entry * TILE_ENTRY_STRIDE) as usize;
        let group_id = tile_entries[base];
        let group_base = (group_id * GROUP_STRIDE) as usize;
        let fill_kind = group_data[group_base + 2] as u32;
        let stroke_kind = group_data[group_base + 4] as u32;
        let fill_index = group_data[group_base + 3] as u32;
        let stroke_index = group_data[group_base + 5] as u32;
        let fill_rule = group_data[group_base + 7] as u32;

        let xform_base = (group_id * XFORM_STRIDE) as usize;
        let local_px = group_xform[xform_base] * px
            + group_xform[xform_base + 1] * py
            + group_xform[xform_base + 2];
        let local_py = group_xform[xform_base + 3] * px
            + group_xform[xform_base + 4] * py
            + group_xform[xform_base + 5];
        let inv_scale = group_inv_scale[group_id as usize];
        let group_scale = if inv_scale > zero { one / inv_scale } else { one };

        let fill_color = paint_color(
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

        let stroke_color = paint_color(
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
                local_px,
                local_py,
                fill_kind,
                stroke_kind,
                fill_rule,
                use_prefiltering,
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

        let scaled_fill_dist = fill_min_dist * group_scale;
        let scaled_stroke_dist = stroke_min_dist * group_scale;
        let scaled_stroke_radius = stroke_min_radius * group_scale;
        blend_group(
            &mut out,
            fill_kind,
            stroke_kind,
            fill_rule,
            fill_color,
            stroke_color,
            scaled_fill_dist,
            fill_winding,
            fill_crossings,
            scaled_stroke_dist,
            scaled_stroke_radius,
            stroke_hit,
            use_prefiltering,
        );

        entry += u32::new(1);
    }

    out
}

#[cube]
pub(super) fn accumulate_group_shapes(
    shape_data: &Array<f32>,
    segment_data: &Array<f32>,
    shape_bounds: &Array<f32>,
    group_data: &Array<f32>,
    group_shapes: &Array<f32>,
    shape_xform: &Array<f32>,
    curve_data: &Array<f32>,
    group_id: u32,
    local_px: f32,
    local_py: f32,
    fill_kind: u32,
    stroke_kind: u32,
    fill_rule: u32,
    use_prefiltering: u32,
    group_bvh_bounds: &Array<f32>,
    group_bvh_nodes: &Array<u32>,
    group_bvh_indices: &Array<u32>,
    group_bvh_meta: &Array<u32>,
    path_bvh_bounds: &Array<f32>,
    path_bvh_nodes: &Array<u32>,
    path_bvh_indices: &Array<u32>,
    path_bvh_meta: &Array<u32>,
    fill_min_dist: &mut f32,
    fill_winding: &mut f32,
    fill_crossings: &mut f32,
    stroke_min_dist: &mut f32,
    stroke_min_radius: &mut f32,
    stroke_hit: &mut f32,
) {
    let meta_base = (group_id * BVH_META_STRIDE) as usize;
    let node_offset = group_bvh_meta[meta_base];
    let node_count = group_bvh_meta[meta_base + 1];
    let index_offset = group_bvh_meta[meta_base + 2];
    let index_count = group_bvh_meta[meta_base + 3];

    if node_count > u32::new(0) && index_count > u32::new(0) {
        let mut node_id = u32::new(0);
        while node_id != BVH_NONE {
            let node_base = ((node_offset + node_id) * BVH_NODE_STRIDE) as usize;
            let min_x = group_bvh_bounds[node_base];
            let min_y = group_bvh_bounds[node_base + 1];
            let max_x = group_bvh_bounds[node_base + 2];
            let max_y = group_bvh_bounds[node_base + 3];
            let skip = group_bvh_nodes[node_base + 1];
            if !(local_px < min_x || local_px > max_x || local_py < min_y || local_py > max_y) {
                let left = group_bvh_nodes[node_base];
                let start = group_bvh_nodes[node_base + 2];
                let count = group_bvh_nodes[node_base + 3];
                if count > u32::new(0) {
                    let mut i = u32::new(0);
                    while i < count {
                        let shape_index = group_bvh_indices[(index_offset + start + i) as usize];
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
                            shape_index,
                            local_px,
                            local_py,
                            fill_kind,
                            stroke_kind,
                            fill_rule,
                            use_prefiltering,
                            fill_min_dist,
                            fill_winding,
                            fill_crossings,
                            stroke_min_dist,
                            stroke_min_radius,
                            stroke_hit,
                        );
                        i += u32::new(1);
                    }
                    node_id = skip;
                } else {
                    node_id = left;
                }
            } else {
                node_id = skip;
            }
        }
    } else {
        let group_base = (group_id * GROUP_STRIDE) as usize;
        let shape_offset = group_data[group_base] as u32;
        let shape_count = group_data[group_base + 1] as u32;
        let mut i = u32::new(0);
        while i < shape_count {
            let shape_index = group_shapes[(shape_offset + i) as usize] as u32;
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
                shape_index,
                local_px,
                local_py,
                fill_kind,
                stroke_kind,
                fill_rule,
                use_prefiltering,
                fill_min_dist,
                fill_winding,
                fill_crossings,
                stroke_min_dist,
                stroke_min_radius,
                stroke_hit,
            );
            i += u32::new(1);
        }
    }
}

#[cube]
pub(super) fn accumulate_shape_in_group(
    shape_data: &Array<f32>,
    segment_data: &Array<f32>,
    shape_bounds: &Array<f32>,
    shape_xform: &Array<f32>,
    curve_data: &Array<f32>,
    path_bvh_bounds: &Array<f32>,
    path_bvh_nodes: &Array<u32>,
    path_bvh_indices: &Array<u32>,
    path_bvh_meta: &Array<u32>,
    shape_index: u32,
    local_px: f32,
    local_py: f32,
    fill_kind: u32,
    stroke_kind: u32,
    fill_rule: u32,
    use_prefiltering: u32,
    fill_min_dist: &mut f32,
    fill_winding: &mut f32,
    fill_crossings: &mut f32,
    stroke_min_dist: &mut f32,
    stroke_min_radius: &mut f32,
    stroke_hit: &mut f32,
) {
    let bounds_base = (shape_index * BOUNDS_STRIDE) as usize;
    let min_x = shape_bounds[bounds_base];
    let min_y = shape_bounds[bounds_base + 1];
    let max_x = shape_bounds[bounds_base + 2];
    let max_y = shape_bounds[bounds_base + 3];
    let mut in_bounds = min_x <= max_x && min_y <= max_y;
    if in_bounds {
        if local_px < min_x || local_px > max_x || local_py < min_y || local_py > max_y {
            in_bounds = false;
        }
    }
    if in_bounds {
        let shape_xform_base = (shape_index * XFORM_STRIDE) as usize;
        let shape_px = shape_xform[shape_xform_base] * local_px
            + shape_xform[shape_xform_base + 1] * local_py
            + shape_xform[shape_xform_base + 2];
        let shape_py = shape_xform[shape_xform_base + 3] * local_px
            + shape_xform[shape_xform_base + 4] * local_py
            + shape_xform[shape_xform_base + 5];

        if fill_kind != PAINT_NONE {
            accumulate_shape_fill(
                shape_data,
                segment_data,
                curve_data,
                path_bvh_bounds,
                path_bvh_nodes,
                path_bvh_indices,
                path_bvh_meta,
                shape_index,
                fill_rule,
                shape_px,
                shape_py,
                fill_min_dist,
                fill_winding,
                fill_crossings,
            );
        }

        if stroke_kind != PAINT_NONE {
            accumulate_shape_stroke(
                shape_data,
                segment_data,
                curve_data,
                path_bvh_bounds,
                path_bvh_nodes,
                path_bvh_indices,
                path_bvh_meta,
                shape_index,
                shape_px,
                shape_py,
                use_prefiltering,
                stroke_min_dist,
                stroke_min_radius,
                stroke_hit,
            );
        }
    }
}

/// Float-atomic variant of weight accumulation.
#[cube(launch_unchecked)]
pub(crate) fn rasterize_weights_f32(
    width: u32,
    height: u32,
    filter_type: u32,
    filter_radius: f32,
    filter_radius_i: u32,
    samples_x: u32,
    samples_y: u32,
    seed: u32,
    jitter: u32,
    weight_accum: &mut Array<Atomic<f32>>,
) {
    let idx = ABSOLUTE_POS;
    let samples_per_pixel = samples_x * samples_y;
    if samples_per_pixel == u32::new(0) {
        terminate!();
    }

    let total_samples = width * height * samples_x * samples_y;
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
    if jitter != 0 {
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

    let ri = filter_radius_i as i32;
    let zero = f32::new(0.0);
    let half = f32::new(0.5);

    for dy in -ri..=ri {
        for dx in -ri..=ri {
            let xx = x as i32 + dx;
            let yy = y as i32 + dy;
            if xx >= 0 && yy >= 0 && xx < width as i32 && yy < height as i32 {
                let xc = f32::cast_from(xx) + half;
                let yc = f32::cast_from(yy) + half;
                let w = filter_weight(filter_type, xc - px, yc - py, filter_radius);
                if w > zero {
                    let base = (yy as u32 * width + xx as u32) as usize;
                    weight_accum[base].fetch_add(w);
                }
            }
        }
    }
}

/// Float-atomic variant of the splat renderer.
#[cube(launch_unchecked)]
pub(crate) fn rasterize_splat_f32(
    shape_data: &Array<f32>,
    segment_data: &Array<f32>,
    shape_bounds: &Array<f32>,
    group_data: &Array<f32>,
    group_xform: &Array<f32>,
    group_inv_scale: &Array<f32>,
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
    tile_offsets: &Array<u32>,
    tile_entries: &Array<u32>,
    tile_order: &Array<u32>,
    tile_count_x: u32,
    tile_count_y: u32,
    tile_size: u32,
    width: u32,
    height: u32,
    _num_groups: u32,
    background_image: &Array<f32>,
    has_background_image: u32,
    background_r: f32,
    background_g: f32,
    background_b: f32,
    background_a: f32,
    filter_type: u32,
    filter_radius: f32,
    filter_radius_i: u32,
    use_prefiltering: u32,
    samples_x: u32,
    samples_y: u32,
    seed: u32,
    jitter: u32,
    weight_accum: &Array<Atomic<f32>>,
    color_accum: &mut Array<Atomic<f32>>,
) {
    let idx = ABSOLUTE_POS;
    let samples_per_pixel = samples_x * samples_y;
    if samples_per_pixel == u32::new(0)
        || tile_size == u32::new(0)
        || tile_count_x == u32::new(0)
        || tile_count_y == u32::new(0)
    {
        terminate!();
    }

    let tile_pixels = tile_size * tile_size;
    let tile_samples = tile_pixels * samples_per_pixel;
    let total_samples = tile_samples * tile_count_x * tile_count_y;
    if idx >= total_samples as usize {
        terminate!();
    }

    let idx_u32 = idx as u32;
    let tile_slot = idx_u32 / tile_samples;
    let local_idx = idx_u32 - tile_slot * tile_samples;
    let tile_id = tile_order[tile_slot as usize];
    let pixel_index = local_idx / samples_per_pixel;
    let sample_index = local_idx - pixel_index * samples_per_pixel;

    let lx = pixel_index % tile_size;
    let ly = pixel_index / tile_size;
    let tile_x = tile_id % tile_count_x;
    let tile_y = tile_id / tile_count_x;
    let x = tile_x * tile_size + lx;
    let y = tile_y * tile_size + ly;

    if x >= width || y >= height {
        terminate!();
    }

    let sx = sample_index % samples_x;
    let sy = sample_index / samples_x;

    let inv_sx = f32::new(1.0) / f32::cast_from(samples_x);
    let inv_sy = f32::new(1.0) / f32::cast_from(samples_y);

    let half = f32::new(0.5);
    let mut rx = half;
    let mut ry = half;
    if jitter != 0 {
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

    let mut bg_r = background_r;
    let mut bg_g = background_g;
    let mut bg_b = background_b;
    let mut bg_a = background_a;
    if has_background_image != u32::new(0) {
        let idx4 = ((y * width + x) as usize) * 4;
        bg_r = background_image[idx4];
        bg_g = background_image[idx4 + 1];
        bg_b = background_image[idx4 + 2];
        bg_a = background_image[idx4 + 3];
    }

    let color = eval_scene_tiled(
        shape_data,
        segment_data,
        shape_bounds,
        group_data,
        group_xform,
        group_inv_scale,
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
        tile_offsets,
        tile_entries,
        tile_id,
        px,
        py,
        bg_r,
        bg_g,
        bg_b,
        bg_a,
        use_prefiltering,
    );

    let ri = filter_radius_i as i32;
    let zero = f32::new(0.0);
    let half = f32::new(0.5);

    for dy in -ri..=ri {
        for dx in -ri..=ri {
            let xx = x as i32 + dx;
            let yy = y as i32 + dy;
            if xx >= 0 && yy >= 0 && xx < width as i32 && yy < height as i32 {
                let base = (yy as u32 * width + xx as u32) as usize;
                let weight_sum = weight_accum[base].load();
                if weight_sum > zero {
                    let xc = f32::cast_from(xx) + half;
                    let yc = f32::cast_from(yy) + half;
                    let w = filter_weight(filter_type, xc - px, yc - py, filter_radius);
                    if w > zero {
                        let w_norm = w / weight_sum;
                        let idx4 = base * 4;
                        color_accum[idx4].fetch_add(color[0] * w_norm);
                        color_accum[idx4 + 1].fetch_add(color[1] * w_norm);
                        color_accum[idx4 + 2].fetch_add(color[2] * w_norm);
                        color_accum[idx4 + 3].fetch_add(color[3] * w_norm);
                    }
                }
            }
        }
    }
}
