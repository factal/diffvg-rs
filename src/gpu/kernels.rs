//! GPU kernels for diffvg-rs rendering and distance evaluation.

use cubecl::prelude::*;

use super::constants::*;

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

// 64-bit integer helpers for PCG in the shader environment.
#[cube]
fn mul32_full(a: u32, b: u32) -> Line<u32> {
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

#[cube]
fn mul64_low(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> Line<u32> {
    let prod0 = mul32_full(a_lo, b_lo);
    let sum_low = a_lo * b_hi + a_hi * b_lo;
    let hi = prod0[1] + sum_low;
    let mut out = Line::empty(2usize);
    out[0] = prod0[0];
    out[1] = hi;
    out
}

#[cube]
fn add64(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> Line<u32> {
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

#[cube]
fn shr64(lo: u32, hi: u32, shift: u32) -> Line<u32> {
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

#[cube]
fn shr64_to_u32(lo: u32, hi: u32, shift: u32) -> u32 {
    let shifted = shr64(lo, hi, shift);
    shifted[0]
}

#[cube]
fn pcg32_next(state_lo: u32, state_hi: u32, inc_lo: u32, inc_hi: u32) -> Line<u32> {
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

#[cube]
fn pcg32_init(idx: u32, seed: u32) -> Line<u32> {
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

#[cube]
fn pcg32_f32(x: u32) -> f32 {
    let mantissa = x >> 9;
    f32::cast_from(mantissa) * f32::new(1.0 / 8_388_608.0)
}

#[cube]
fn blend_group(
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
fn accumulate_shape_fill(
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
fn accumulate_shape_stroke(
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
fn bounds_distance(min_x: f32, min_y: f32, max_x: f32, max_y: f32, px: f32, py: f32) -> f32 {
    let zero = f32::new(0.0);
    let dx = max_f32(max_f32(min_x - px, zero), px - max_x);
    let dy = max_f32(max_f32(min_y - py, zero), py - max_y);
    (dx * dx + dy * dy).sqrt()
}

#[cube]
fn bounds_contains(min_x: f32, min_y: f32, max_x: f32, max_y: f32, px: f32, py: f32) -> bool {
    px >= min_x && px <= max_x && py >= min_y && py <= max_y
}

#[cube]
fn ray_intersects_bounds(_min_x: f32, min_y: f32, max_x: f32, max_y: f32, px: f32, py: f32) -> bool {
    !(py < min_y || py > max_y || px > max_x)
}

#[cube]
fn accumulate_path_fill_full(
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
fn accumulate_path_stroke_full(
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
fn accumulate_path_fill_bvh(
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
fn accumulate_path_stroke_bvh(
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
fn eval_scene_tiled(
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
fn accumulate_group_shapes(
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
fn accumulate_shape_in_group(
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

#[cube]
fn paint_color(
    kind: u32,
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
) -> Line<f32> {
    let zero = f32::new(0.0);
    let mut out = Line::empty(4usize);
    out[0] = zero;
    out[1] = zero;
    out[2] = zero;
    out[3] = zero;

    if kind == PAINT_SOLID {
        out[0] = solid_r;
        out[1] = solid_g;
        out[2] = solid_b;
        out[3] = solid_a;
    } else if kind == PAINT_LINEAR || kind == PAINT_RADIAL {
        out = sample_gradient(gradient_data, stop_offsets, stop_colors, gradient_index, px, py);
    }

    out
}

#[cube]
fn sample_gradient(
    gradient_data: &Array<f32>,
    stop_offsets: &Array<f32>,
    stop_colors: &Array<f32>,
    gradient_index: u32,
    px: f32,
    py: f32,
) -> Line<f32> {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let mut out = Line::empty(4usize);
    out[0] = zero;
    out[1] = zero;
    out[2] = zero;
    out[3] = zero;

    let base = (gradient_index * GRADIENT_STRIDE) as usize;
    let grad_type = gradient_data[base] as u32;
    let p0 = gradient_data[base + 1];
    let p1 = gradient_data[base + 2];
    let p2 = gradient_data[base + 3];
    let p3 = gradient_data[base + 4];
    let stop_offset = gradient_data[base + 5] as u32;
    let stop_count = gradient_data[base + 6] as u32;

    let has_stops = stop_count > 0;

    let t = if grad_type == 0 {
        let vx = p2 - p0;
        let vy = p3 - p1;
        let denom = max_f32(vx * vx + vy * vy, f32::new(1.0e-3));
        ((px - p0) * vx + (py - p1) * vy) / denom
    } else {
        let dx = (px - p0) / max_f32(p2, f32::new(1.0e-3));
        let dy = (py - p1) / max_f32(p3, f32::new(1.0e-3));
        (dx * dx + dy * dy).sqrt()
    };

    if has_stops {
        let first_offset = stop_offsets[stop_offset as usize];
        let last_offset = stop_offsets[(stop_offset + stop_count - 1) as usize];

        let mut color_r = zero;
        let mut color_g = zero;
        let mut color_b = zero;
        let mut color_a = zero;
        let mut found = zero;

        if t <= first_offset {
            let cbase = (stop_offset * 4) as usize;
            color_r = stop_colors[cbase];
            color_g = stop_colors[cbase + 1];
            color_b = stop_colors[cbase + 2];
            color_a = stop_colors[cbase + 3];
            found = one;
        } else if t >= last_offset {
            let cbase = ((stop_offset + stop_count - 1) * 4) as usize;
            color_r = stop_colors[cbase];
            color_g = stop_colors[cbase + 1];
            color_b = stop_colors[cbase + 2];
            color_a = stop_colors[cbase + 3];
            found = one;
        } else {
            let mut i = u32::new(0);
            let stop_last = stop_count - 1;
            while i < stop_last {
                let curr = stop_offsets[(stop_offset + i) as usize];
                let next = stop_offsets[(stop_offset + i + 1) as usize];
                if t >= curr && t < next && found == zero {
                    let tt = (t - curr) / max_f32(next - curr, f32::new(1.0e-5));
                    let c0 = ((stop_offset + i) * 4) as usize;
                    let c1 = ((stop_offset + i + 1) * 4) as usize;
                    color_r = stop_colors[c0] * (one - tt) + stop_colors[c1] * tt;
                    color_g = stop_colors[c0 + 1] * (one - tt) + stop_colors[c1 + 1] * tt;
                    color_b = stop_colors[c0 + 2] * (one - tt) + stop_colors[c1 + 2] * tt;
                    color_a = stop_colors[c0 + 3] * (one - tt) + stop_colors[c1 + 3] * tt;
                    found = one;
                }
                i += u32::new(1);
            }
        }

        if found > zero {
            out[0] = color_r;
            out[1] = color_g;
            out[2] = color_b;
            out[3] = color_a;
        }
    }

    out
}

#[cube]
fn shape_fill_coverage(
    shape_data: &Array<f32>,
    segment_data: &Array<f32>,
    curve_data: &Array<f32>,
    shape_index: u32,
    fill_rule: u32,
    px: f32,
    py: f32,
    aa: f32,
) -> f32 {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
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

    let mut coverage = zero;

    if kind == SHAPE_KIND_CIRCLE {
        let dx = px - p0;
        let dy = py - p1;
        let radius = p2;
        let dist = (dx * dx + dy * dy).sqrt() - radius;
        coverage = sdf_coverage(dist, aa);
    } else if kind == SHAPE_KIND_ELLIPSE {
        let dx = (px - p0) / max_f32(p2.abs(), f32::new(1.0e-3));
        let dy = (py - p1) / max_f32(p3.abs(), f32::new(1.0e-3));
        let len = (dx * dx + dy * dy).sqrt();
        let scale = min_f32(p2.abs(), p3.abs());
        let dist = (len - one) * scale;
        coverage = sdf_coverage(dist, aa);
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
        coverage = sdf_coverage(dist, aa);
    } else if kind == SHAPE_KIND_PATH {
        if curve_count > 0 || seg_count > 0 {
            let min_x = p0;
            let min_y = p1;
            let max_x = p2;
            let max_y = p3;
            if px >= min_x - aa && px <= max_x + aa && py >= min_y - aa && py <= max_y + aa {
                let mut min_dist = f32::new(1.0e20);
                let mut winding = zero;
                let mut crossings = zero;

                if curve_count > 0 {
                    for s in 0..curve_count {
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
                            distance_to_cubic(
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
                                use_distance_approx,
                            )
                        };
                        if dist < min_dist {
                            min_dist = dist;
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
                                &mut winding,
                                &mut crossings,
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
                                &mut winding,
                                &mut crossings,
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
                                &mut winding,
                                &mut crossings,
                            );
                        }
                    }
                } else {
                    for s in 0..seg_count {
                        let seg_base = ((seg_offset + s) * SEGMENT_STRIDE) as usize;
                        let x0 = segment_data[seg_base];
                        let y0 = segment_data[seg_base + 1];
                        let x1 = segment_data[seg_base + 2];
                        let y1 = segment_data[seg_base + 3];

                        let dist = distance_to_segment(px, py, x0, y0, x1, y1);
                        if dist < min_dist {
                            min_dist = dist;
                        }

                        winding_and_crossings_line(
                            px,
                            py,
                            x0,
                            y0,
                            x1,
                            y1,
                            fill_rule,
                            &mut winding,
                            &mut crossings,
                        );
                    }
                }

                let inside = if fill_rule == 1 {
                    crossings > zero
                } else {
                    winding != zero
                };
                let dist = if inside { -min_dist } else { min_dist };
                coverage = sdf_coverage(dist, aa);
            }
        }
    }

    coverage
}

#[cube]
fn shape_stroke_coverage(
    shape_data: &Array<f32>,
    segment_data: &Array<f32>,
    shape_index: u32,
    px: f32,
    py: f32,
    aa: f32,
) -> f32 {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let base = (shape_index * SHAPE_STRIDE) as usize;
    let kind = shape_data[base] as u32;
    let seg_offset = shape_data[base + 1] as u32;
    let seg_count = shape_data[base + 2] as u32;
    let stroke_width = shape_data[base + 3];
    let has_thickness = shape_data[base + 8] > f32::new(0.5);
    let join_type = shape_data[base + 9] as u32;
    let cap_type = shape_data[base + 10] as u32;
    let miter_limit = shape_data[base + 11];

    let p0 = shape_data[base + 4];
    let p1 = shape_data[base + 5];
    let p2 = shape_data[base + 6];
    let p3 = shape_data[base + 7];

    let mut coverage = zero;

    if has_thickness || stroke_width > zero {
        if kind == SHAPE_KIND_CIRCLE {
            let dx = px - p0;
            let dy = py - p1;
            let radius = p2;
            let dist = (dx * dx + dy * dy).sqrt();
            let sdf = abs_f32(dist - radius) - stroke_width;
            coverage = sdf_coverage(sdf, aa);
        } else if kind == SHAPE_KIND_ELLIPSE {
            let dx = (px - p0) / max_f32(p2.abs(), f32::new(1.0e-3));
            let dy = (py - p1) / max_f32(p3.abs(), f32::new(1.0e-3));
            let len = (dx * dx + dy * dy).sqrt();
            let scale = min_f32(p2.abs(), p3.abs());
            let dist = (len - one) * scale;
            let sdf = abs_f32(dist) - stroke_width;
            coverage = sdf_coverage(sdf, aa);
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
            let sdf = abs_f32(dist) - stroke_width;
            coverage = sdf_coverage(sdf, aa);
        } else if kind == SHAPE_KIND_PATH {
            if seg_count > 0 {
                let mut min_sdf = f32::new(1.0e20);
                for s in 0..seg_count {
                    let seg_base = ((seg_offset + s) * SEGMENT_STRIDE) as usize;
                    let x0 = segment_data[seg_base];
                    let y0 = segment_data[seg_base + 1];
                    let x1 = segment_data[seg_base + 2];
                    let y1 = segment_data[seg_base + 3];
                    let r0 = segment_data[seg_base + 4];
                    let r1 = segment_data[seg_base + 5];
                    let prev_dx = segment_data[seg_base + 6];
                    let prev_dy = segment_data[seg_base + 7];
                    let next_dx = segment_data[seg_base + 8];
                    let next_dy = segment_data[seg_base + 9];
                    let start_cap = segment_data[seg_base + 10];
                    let end_cap = segment_data[seg_base + 11];

                    let sdf = segment_sdf_with_caps(
                        px,
                        py,
                        x0,
                        y0,
                        x1,
                        y1,
                        r0,
                        r1,
                        start_cap,
                        end_cap,
                        join_type,
                        cap_type,
                        has_thickness,
                        stroke_width,
                    );
                    if sdf < min_sdf {
                        min_sdf = sdf;
                    }

                    if join_type == JOIN_MITER {
                        let dir = normalize_vec(x1 - x0, y1 - y0);
                        if start_cap <= f32::new(0.5) {
                            let miter_sdf = miter_join_sdf(
                                px,
                                py,
                                x0,
                                y0,
                                prev_dx,
                                prev_dy,
                                dir[0],
                                dir[1],
                                r0,
                                miter_limit,
                            );
                            if miter_sdf < min_sdf {
                                min_sdf = miter_sdf;
                            }
                        }
                        if end_cap <= f32::new(0.5) {
                            let miter_sdf = miter_join_sdf(
                                px,
                                py,
                                x1,
                                y1,
                                dir[0],
                                dir[1],
                                next_dx,
                                next_dy,
                                r1,
                                miter_limit,
                            );
                            if miter_sdf < min_sdf {
                                min_sdf = miter_sdf;
                            }
                        }
                    }
                }
                coverage = sdf_coverage(min_sdf, aa);
            }
        }
    }

    coverage
}

#[cube]
fn segment_sdf_with_caps(
    px: f32,
    py: f32,
    ax: f32,
    ay: f32,
    bx: f32,
    by: f32,
    r0_in: f32,
    r1_in: f32,
    start_cap: f32,
    end_cap: f32,
    join_type: u32,
    cap_type: u32,
    has_thickness: bool,
    stroke_width: f32,
) -> f32 {
    let zero = f32::new(0.0);
    let half = f32::new(0.5);
    let cap_round = f32::new(2.0);
    let cap_square = f32::new(1.0);
    let cap_butt = f32::new(0.0);

    let r0 = if has_thickness { r0_in } else { stroke_width };
    let r1 = if has_thickness { r1_in } else { stroke_width };

    let mut cap_start = cap_type as f32;
    if start_cap <= half {
        cap_start = if join_type == JOIN_ROUND { cap_round } else { cap_butt };
    }
    let mut cap_end = cap_type as f32;
    if end_cap <= half {
        cap_end = if join_type == JOIN_ROUND { cap_round } else { cap_butt };
    }

    let ext_start = if cap_start == cap_square { r0 } else { zero };
    let ext_end = if cap_end == cap_square { r1 } else { zero };

    let mut sdf = segment_rect_sdf(px, py, ax, ay, bx, by, r0, r1, ext_start, ext_end);

    if cap_start == cap_round {
        let cap_sdf = distance_to_point(px, py, ax, ay) - r0;
        sdf = min_f32(sdf, cap_sdf);
    }
    if cap_end == cap_round {
        let cap_sdf = distance_to_point(px, py, bx, by) - r1;
        sdf = min_f32(sdf, cap_sdf);
    }

    sdf
}

#[cube]
fn segment_rect_sdf(
    px: f32,
    py: f32,
    ax: f32,
    ay: f32,
    bx: f32,
    by: f32,
    r0: f32,
    r1: f32,
    ext_start: f32,
    ext_end: f32,
) -> f32 {
    let zero = f32::new(0.0);
    let vx = bx - ax;
    let vy = by - ay;
    let len = (vx * vx + vy * vy).sqrt();
    let mut sdf = distance_to_point(px, py, ax, ay) - r0;
    if len > f32::new(1.0e-6) {
        let tx = vx / len;
        let ty = vy / len;
        let nx = -ty;
        let ny = tx;

        let dx = px - ax;
        let dy = py - ay;
        let s = dx * tx + dy * ty;
        let n = dx * nx + dy * ny;

        let s0 = -ext_start;
        let s1 = len + ext_end;

        let s_clamped = clamp_range(s, zero, len);
        let t = s_clamped / len;
        let radius = r0 + t * (r1 - r0);

        let mut outside_x = zero;
        if s < s0 {
            outside_x = s0 - s;
        } else if s > s1 {
            outside_x = s - s1;
        }
        let mut outside_y = abs_f32(n) - radius;
        if outside_y < zero {
            outside_y = zero;
        }

        let outside = (outside_x * outside_x + outside_y * outside_y).sqrt();
        if outside_x > zero || outside_y > zero {
            sdf = outside;
        } else {
            let inside_x = min_f32(s - s0, s1 - s);
            let inside_y = radius - abs_f32(n);
            sdf = -min_f32(inside_x, inside_y);
        }
    }

    sdf
}

#[cube]
fn miter_join_sdf(
    px: f32,
    py: f32,
    vx: f32,
    vy: f32,
    dir0x: f32,
    dir0y: f32,
    dir1x: f32,
    dir1y: f32,
    radius: f32,
    miter_limit: f32,
) -> f32 {
    let zero = f32::new(0.0);
    let big = f32::new(1.0e20);
    let eps = f32::new(1.0e-5);
    let mut out = big;
    if radius > zero {
        let d0 = normalize_vec(dir0x, dir0y);
        let d1 = normalize_vec(dir1x, dir1y);
        if !(d0[0] == zero && d0[1] == zero) && !(d1[0] == zero && d1[1] == zero) {
            let cross = cross2(d0[0], d0[1], d1[0], d1[1]);
            if abs_f32(cross) >= eps {
                let sign = if cross > zero { f32::new(1.0) } else { f32::new(-1.0) };
                let n0x = -d0[1] * sign;
                let n0y = d0[0] * sign;
                let n1x = -d1[1] * sign;
                let n1y = d1[0] * sign;

                let p0x = vx + n0x * radius;
                let p0y = vy + n0y * radius;
                let p1x = vx + n1x * radius;
                let p1y = vy + n1y * radius;

                let denom = cross;
                let t = cross2(p1x - p0x, p1y - p0y, d1[0], d1[1]) / denom;
                let mx = p0x + d0[0] * t;
                let my = p0y + d0[1] * t;

                let miter_len = distance_to_point(mx, my, vx, vy);
                if miter_len <= miter_limit * radius {
                    out = distance_to_triangle(px, py, p0x, p0y, p1x, p1y, mx, my);
                }
            }
        }
    }
    out
}

#[cube]
fn distance_to_triangle(
    px: f32,
    py: f32,
    ax: f32,
    ay: f32,
    bx: f32,
    by: f32,
    cx: f32,
    cy: f32,
) -> f32 {
    let zero = f32::new(0.0);
    let c1 = cross2(bx - ax, by - ay, px - ax, py - ay);
    let c2 = cross2(cx - bx, cy - by, px - bx, py - by);
    let c3 = cross2(ax - cx, ay - cy, px - cx, py - cy);
    let has_neg = c1 < zero || c2 < zero || c3 < zero;
    let has_pos = c1 > zero || c2 > zero || c3 > zero;
    let inside = !(has_neg && has_pos);

    let d0 = distance_to_segment(px, py, ax, ay, bx, by);
    let d1 = distance_to_segment(px, py, bx, by, cx, cy);
    let d2 = distance_to_segment(px, py, cx, cy, ax, ay);
    let min_dist = min_f32(d0, min_f32(d1, d2));
    if inside {
        -min_dist
    } else {
        min_dist
    }
}

#[cube]
fn distance_to_point(px: f32, py: f32, ax: f32, ay: f32) -> f32 {
    let dx = px - ax;
    let dy = py - ay;
    (dx * dx + dy * dy).sqrt()
}

#[cube]
fn normalize_vec(x: f32, y: f32) -> Line<f32> {
    let zero = f32::new(0.0);
    let len = (x * x + y * y).sqrt();
    let mut out = Line::empty(2usize);
    if len > f32::new(1.0e-6) {
        out[0] = x / len;
        out[1] = y / len;
    } else {
        out[0] = zero;
        out[1] = zero;
    }
    out
}

#[cube]
fn cross2(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    ax * by - ay * bx
}

#[cube]
fn clamp_range(v: f32, min_v: f32, max_v: f32) -> f32 {
    if v < min_v {
        min_v
    } else if v > max_v {
        max_v
    } else {
        v
    }
}

#[cube]
fn distance_to_segment_with_t(
    px: f32,
    py: f32,
    ax: f32,
    ay: f32,
    bx: f32,
    by: f32,
) -> Line<f32> {
    let zero = f32::new(0.0);
    let vx = bx - ax;
    let vy = by - ay;
    let wx = px - ax;
    let wy = py - ay;
    let c1 = vx * wx + vy * wy;
    let c2 = vx * vx + vy * vy;
    let mut t = zero;
    if c2 > zero && c1 > zero {
        t = c1 / c2;
        t = clamp01(t);
    }
    let proj_x = ax + t * vx;
    let proj_y = ay + t * vy;
    let dist_sq = (px - proj_x) * (px - proj_x) + (py - proj_y) * (py - proj_y);
    let mut out = Line::empty(2usize);
    out[0] = dist_sq.sqrt();
    out[1] = t;
    out
}

#[cube]
fn distance_to_segment(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let zero = f32::new(0.0);
    let vx = bx - ax;
    let vy = by - ay;
    let wx = px - ax;
    let wy = py - ay;

    let c1 = vx * wx + vy * wy;
    let c2 = vx * vx + vy * vy;

    let mut dist_sq = (px - ax) * (px - ax) + (py - ay) * (py - ay);
    if c1 > zero {
        if c2 <= c1 {
            dist_sq = (px - bx) * (px - bx) + (py - by) * (py - by);
        } else {
            let t = c1 / c2;
            let proj_x = ax + t * vx;
            let proj_y = ay + t * vy;
            dist_sq = (px - proj_x) * (px - proj_x) + (py - proj_y) * (py - proj_y);
        }
    }

    dist_sq.sqrt()
}

#[cube]
fn winding_and_crossings_line(
    px: f32,
    py: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    fill_rule: u32,
    winding: &mut f32,
    crossings: &mut f32,
) {
    let one = f32::new(1.0);
    let y0_le = y0 <= py;
    let y1_le = y1 <= py;
    if y0_le != y1_le {
        let t = (py - y0) / (y1 - y0);
        let x_int = x0 + t * (x1 - x0);
        if x_int > px {
            if fill_rule == 1 {
                *crossings = one - *crossings;
            } else {
                let delta = if y1 > y0 { one } else { -one };
                *winding += delta;
            }
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
#[cube(launch_unchecked)]
pub(crate) fn bin_tiles_count(
    group_bounds: &Array<f32>,
    group_shape_xform: &Array<f32>,
    num_groups: u32,
    tile_count_x: u32,
    tile_count_y: u32,
    tile_size: u32,
    width: u32,
    height: u32,
    tile_counts: &mut Array<Atomic<u32>>,
) {
    let idx = ABSOLUTE_POS;
    if idx >= num_groups as usize {
        terminate!();
    }

    let group_id = idx as u32;

    let bounds_base = (group_id * BOUNDS_STRIDE) as usize;
    let min_x = group_bounds[bounds_base];
    let min_y = group_bounds[bounds_base + 1];
    let max_x = group_bounds[bounds_base + 2];
    let max_y = group_bounds[bounds_base + 3];

    let mut valid = min_x <= max_x && min_y <= max_y;
    if tile_count_x == u32::new(0) || tile_count_y == u32::new(0) {
        valid = false;
    }

    if valid {
        let xform_base = (group_id * XFORM_STRIDE) as usize;
        let m00 = group_shape_xform[xform_base];
        let m01 = group_shape_xform[xform_base + 1];
        let m02 = group_shape_xform[xform_base + 2];
        let m10 = group_shape_xform[xform_base + 3];
        let m11 = group_shape_xform[xform_base + 4];
        let m12 = group_shape_xform[xform_base + 5];

        let x0 = m00 * min_x + m01 * min_y + m02;
        let y0 = m10 * min_x + m11 * min_y + m12;
        let x1 = m00 * max_x + m01 * min_y + m02;
        let y1 = m10 * max_x + m11 * min_y + m12;
        let x2 = m00 * max_x + m01 * max_y + m02;
        let y2 = m10 * max_x + m11 * max_y + m12;
        let x3 = m00 * min_x + m01 * max_y + m02;
        let y3 = m10 * min_x + m11 * max_y + m12;

        let min_cx = min_f32(min_f32(x0, x1), min_f32(x2, x3));
        let max_cx = max_f32(max_f32(x0, x1), max_f32(x2, x3));
        let min_cy = min_f32(min_f32(y0, y1), min_f32(y2, y3));
        let max_cy = max_f32(max_f32(y0, y1), max_f32(y2, y3));

        let zero = f32::new(0.0);
        let w = f32::cast_from(width);
        let h = f32::cast_from(height);
        let in_view = !(max_cx < zero || max_cy < zero || min_cx >= w || min_cy >= h);
        if in_view {
            let tile_size_f = f32::cast_from(tile_size);
            let mut min_tx = (min_cx / tile_size_f).floor() as i32;
            let mut max_tx = (max_cx / tile_size_f).floor() as i32;
            let mut min_ty = (min_cy / tile_size_f).floor() as i32;
            let mut max_ty = (max_cy / tile_size_f).floor() as i32;

            let max_tile_x = tile_count_x as i32 - 1;
            let max_tile_y = tile_count_y as i32 - 1;

            if min_tx < 0 {
                min_tx = 0;
            } else if min_tx > max_tile_x {
                min_tx = max_tile_x;
            }
            if max_tx < 0 {
                max_tx = 0;
            } else if max_tx > max_tile_x {
                max_tx = max_tile_x;
            }
            if min_ty < 0 {
                min_ty = 0;
            } else if min_ty > max_tile_y {
                min_ty = max_tile_y;
            }
            if max_ty < 0 {
                max_ty = 0;
            } else if max_ty > max_tile_y {
                max_ty = max_tile_y;
            }

            if min_tx <= max_tx && min_ty <= max_ty {
                let min_tx = min_tx as u32;
                let max_tx = max_tx as u32;
                let min_ty = min_ty as u32;
                let max_ty = max_ty as u32;
                for ty in min_ty..=max_ty {
                    let row = ty * tile_count_x;
                    for tx in min_tx..=max_tx {
                        let tile_id = (row + tx) as usize;
                        tile_counts[tile_id].fetch_add(u32::new(1));
                    }
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

#[cube(launch_unchecked)]
pub(crate) fn init_tile_offsets(
    tile_counts: &Array<Atomic<u32>>,
    num_tiles: u32,
    offsets: &mut Array<u32>,
) {
    let idx = ABSOLUTE_POS;
    if idx >= num_tiles as usize {
        terminate!();
    }

    if idx == 0 {
        offsets[0] = u32::new(0);
    }
    offsets[idx + 1] = tile_counts[idx].load();
}

#[cube(launch_unchecked)]
pub(crate) fn scan_tile_offsets(
    offsets_in: &Array<u32>,
    offsets_out: &mut Array<u32>,
    num_entries: u32,
    stride: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= num_entries as usize {
        terminate!();
    }

    let idx_u32 = idx as u32;
    if idx_u32 >= stride {
        offsets_out[idx] = offsets_in[idx] + offsets_in[(idx_u32 - stride) as usize];
    } else {
        offsets_out[idx] = offsets_in[idx];
    }
}

#[cube(launch_unchecked)]
pub(crate) fn init_tile_cursor(
    tile_offsets: &Array<u32>,
    num_tiles: u32,
    tile_cursor: &mut Array<Atomic<u32>>,
) {
    let idx = ABSOLUTE_POS;
    if idx >= num_tiles as usize {
        terminate!();
    }
    let value = tile_offsets[idx];
    tile_cursor[idx].fetch_add(value);
}

#[cube(launch_unchecked)]
pub(crate) fn bin_tiles_write(
    group_bounds: &Array<f32>,
    group_shape_xform: &Array<f32>,
    num_groups: u32,
    tile_count_x: u32,
    tile_count_y: u32,
    tile_size: u32,
    width: u32,
    height: u32,
    tile_cursor: &mut Array<Atomic<u32>>,
    tile_entries: &mut Array<u32>,
) {
    let idx = ABSOLUTE_POS;
    if idx >= num_groups as usize {
        terminate!();
    }

    let group_id = idx as u32;

    let bounds_base = (group_id * BOUNDS_STRIDE) as usize;
    let min_x = group_bounds[bounds_base];
    let min_y = group_bounds[bounds_base + 1];
    let max_x = group_bounds[bounds_base + 2];
    let max_y = group_bounds[bounds_base + 3];

    let mut valid = min_x <= max_x && min_y <= max_y;
    if tile_count_x == u32::new(0) || tile_count_y == u32::new(0) {
        valid = false;
    }

    if valid {
        let xform_base = (group_id * XFORM_STRIDE) as usize;
        let m00 = group_shape_xform[xform_base];
        let m01 = group_shape_xform[xform_base + 1];
        let m02 = group_shape_xform[xform_base + 2];
        let m10 = group_shape_xform[xform_base + 3];
        let m11 = group_shape_xform[xform_base + 4];
        let m12 = group_shape_xform[xform_base + 5];

        let x0 = m00 * min_x + m01 * min_y + m02;
        let y0 = m10 * min_x + m11 * min_y + m12;
        let x1 = m00 * max_x + m01 * min_y + m02;
        let y1 = m10 * max_x + m11 * min_y + m12;
        let x2 = m00 * max_x + m01 * max_y + m02;
        let y2 = m10 * max_x + m11 * max_y + m12;
        let x3 = m00 * min_x + m01 * max_y + m02;
        let y3 = m10 * min_x + m11 * max_y + m12;

        let min_cx = min_f32(min_f32(x0, x1), min_f32(x2, x3));
        let max_cx = max_f32(max_f32(x0, x1), max_f32(x2, x3));
        let min_cy = min_f32(min_f32(y0, y1), min_f32(y2, y3));
        let max_cy = max_f32(max_f32(y0, y1), max_f32(y2, y3));

        let zero = f32::new(0.0);
        let w = f32::cast_from(width);
        let h = f32::cast_from(height);
        let in_view = !(max_cx < zero || max_cy < zero || min_cx >= w || min_cy >= h);
        if in_view {
            let tile_size_f = f32::cast_from(tile_size);
            let mut min_tx = (min_cx / tile_size_f).floor() as i32;
            let mut max_tx = (max_cx / tile_size_f).floor() as i32;
            let mut min_ty = (min_cy / tile_size_f).floor() as i32;
            let mut max_ty = (max_cy / tile_size_f).floor() as i32;

            let max_tile_x = tile_count_x as i32 - 1;
            let max_tile_y = tile_count_y as i32 - 1;

            if min_tx < 0 {
                min_tx = 0;
            } else if min_tx > max_tile_x {
                min_tx = max_tile_x;
            }
            if max_tx < 0 {
                max_tx = 0;
            } else if max_tx > max_tile_x {
                max_tx = max_tile_x;
            }
            if min_ty < 0 {
                min_ty = 0;
            } else if min_ty > max_tile_y {
                min_ty = max_tile_y;
            }
            if max_ty < 0 {
                max_ty = 0;
            } else if max_ty > max_tile_y {
                max_ty = max_tile_y;
            }

            if min_tx <= max_tx && min_ty <= max_ty {
                let min_tx = min_tx as u32;
                let max_tx = max_tx as u32;
                let min_ty = min_ty as u32;
                let max_ty = max_ty as u32;
                for ty in min_ty..=max_ty {
                    let row = ty * tile_count_x;
                    for tx in min_tx..=max_tx {
                        let tile_id = (row + tx) as usize;
                        let entry_index = tile_cursor[tile_id].fetch_add(u32::new(1));
                        let base = (entry_index * TILE_ENTRY_STRIDE) as usize;
                        tile_entries[base] = group_id;
                    }
                }
            }
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn sort_tile_entries(
    tile_offsets: &Array<u32>,
    tile_entries: &mut Array<u32>,
    num_tiles: u32,
) {
    let tile_id = ABSOLUTE_POS;
    if tile_id >= num_tiles as usize {
        terminate!();
    }
    let start = tile_offsets[tile_id];
    let end = tile_offsets[tile_id + 1];
    let one = u32::new(1);
    if end <= start + one {
        terminate!();
    }
    let mut i = start + one;
    while i < end {
        let key = tile_entries[i as usize];
        let mut j = i;
        while j > start {
            let prev = tile_entries[(j - one) as usize];
            if prev <= key {
                break;
            }
            tile_entries[j as usize] = prev;
            j -= one;
        }
        tile_entries[j as usize] = key;
        i += one;
    }
}

#[cube]
fn winding_and_crossings_quadratic(
    px: f32,
    py: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    fill_rule: u32,
    winding: &mut f32,
    crossings: &mut f32,
) {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    // y(t) = (y0 - 2y1 + y2) t^2 + (-2y0 + 2y1) t + y0
    let ay = y0 - f32::new(2.0) * y1 + y2;
    let by = -f32::new(2.0) * y0 + f32::new(2.0) * y1;
    let cy = y0 - py;
    let roots = solve_quadratic(ay, by, cy);
    let count = roots[0] as u32;
    let mut i = u32::new(0);
    while i < count {
        let t = roots[(i + 1) as usize];
        if t >= zero && t <= one {
            let tt = one - t;
            let x = (tt * tt) * x0 + (f32::new(2.0) * tt * t) * x1 + (t * t) * x2;
            if x > px {
                if fill_rule == 1 {
                    *crossings = one - *crossings;
                } else {
                    let dy = f32::new(2.0) * ay * t + by;
                    let delta = if dy > zero { one } else { -one };
                    *winding += delta;
                }
            }
        }
        i += u32::new(1);
    }
}

#[cube]
fn winding_and_crossings_cubic(
    px: f32,
    py: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    x3: f32,
    y3: f32,
    fill_rule: u32,
    winding: &mut f32,
    crossings: &mut f32,
) {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let a = -y0 + f32::new(3.0) * y1 - f32::new(3.0) * y2 + y3;
    let b = f32::new(3.0) * y0 - f32::new(6.0) * y1 + f32::new(3.0) * y2;
    let c = -f32::new(3.0) * y0 + f32::new(3.0) * y1;
    let d = y0 - py;
    let roots = solve_cubic(a, b, c, d);
    let count = roots[0] as u32;
    let mut i = u32::new(0);
    while i < count {
        let t = roots[(i + 1) as usize];
        if t >= zero && t <= one {
            let tt = one - t;
            let x = (tt * tt * tt) * x0
                + (f32::new(3.0) * tt * tt * t) * x1
                + (f32::new(3.0) * tt * t * t) * x2
                + (t * t * t) * x3;
            if x > px {
                if fill_rule == 1 {
                    *crossings = one - *crossings;
                } else {
                    let dy = f32::new(3.0) * a * t * t + f32::new(2.0) * b * t + c;
                    let delta = if dy > zero { one } else { -one };
                    *winding += delta;
                }
            }
        }
        i += u32::new(1);
    }
}

#[cube]
fn closest_point_quadratic_with_t(
    px: f32,
    py: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    use_distance_approx: bool,
) -> Line<f32> {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let mut best_t = zero;
    let mut best_dist = distance_to_point(px, py, x0, y0);
    if use_distance_approx {
        let approx = quadratic_closest_pt_approx(px, py, x0, y0, x1, y1, x2, y2);
        best_dist = distance_to_point(px, py, approx[0], approx[1]);
        best_t = approx[2];
    } else {
        let dist_end = distance_to_point(px, py, x2, y2);
        if dist_end < best_dist {
            best_dist = dist_end;
            best_t = one;
        }

        let ax = x0 - f32::new(2.0) * x1 + x2;
        let ay = y0 - f32::new(2.0) * y1 + y2;
        let bx = -x0 + x1;
        let by = -y0 + y1;
        let cx = x0 - px;
        let cy = y0 - py;

        let a = ax * ax + ay * ay;
        let b = f32::new(3.0) * (ax * bx + ay * by);
        let c = f32::new(2.0) * (bx * bx + by * by) + (ax * cx + ay * cy);
        let d = bx * cx + by * cy;

        let roots = solve_cubic(a, b, c, d);
        let count = roots[0] as u32;
        let mut i = u32::new(0);
        while i < count {
            let t = roots[(i + 1) as usize];
            if t >= zero && t <= one {
                let tt = one - t;
                let x = (tt * tt) * x0 + (f32::new(2.0) * tt * t) * x1 + (t * t) * x2;
                let y = (tt * tt) * y0 + (f32::new(2.0) * tt * t) * y1 + (t * t) * y2;
                let dist = distance_to_point(px, py, x, y);
                if dist < best_dist {
                    best_dist = dist;
                    best_t = t;
                }
            }
            i += u32::new(1);
        }
    }

    let mut out = Line::empty(2usize);
    out[0] = best_dist;
    out[1] = best_t;
    out
}

#[cube]
fn distance_to_quadratic(
    px: f32,
    py: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    use_distance_approx: bool,
) -> f32 {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let mut min_dist = distance_to_point(px, py, x0, y0);
    if use_distance_approx {
        let approx = quadratic_closest_pt_approx(px, py, x0, y0, x1, y1, x2, y2);
        min_dist = distance_to_point(px, py, approx[0], approx[1]);
    } else {
        let dist_end = distance_to_point(px, py, x2, y2);
        if dist_end < min_dist {
            min_dist = dist_end;
        }

        let ax = x0 - f32::new(2.0) * x1 + x2;
        let ay = y0 - f32::new(2.0) * y1 + y2;
        let bx = -x0 + x1;
        let by = -y0 + y1;
        let cx = x0 - px;
        let cy = y0 - py;

        let a = ax * ax + ay * ay;
        let b = f32::new(3.0) * (ax * bx + ay * by);
        let c = f32::new(2.0) * (bx * bx + by * by) + (ax * cx + ay * cy);
        let d = bx * cx + by * cy;

        let roots = solve_cubic(a, b, c, d);
        let count = roots[0] as u32;
        let mut i = u32::new(0);
        while i < count {
            let t = roots[(i + 1) as usize];
            if t >= zero && t <= one {
                let tt = one - t;
                let x = (tt * tt) * x0 + (f32::new(2.0) * tt * t) * x1 + (t * t) * x2;
                let y = (tt * tt) * y0 + (f32::new(2.0) * tt * t) * y1 + (t * t) * y2;
                let dist = distance_to_point(px, py, x, y);
                if dist < min_dist {
                    min_dist = dist;
                }
            }
            i += u32::new(1);
        }
    }

    min_dist
}

#[cube]
fn distance_to_cubic(
    px: f32,
    py: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    x3: f32,
    y3: f32,
    use_distance_approx: bool,
) -> f32 {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let min_dist = if use_distance_approx {
        distance_to_cubic_approx(px, py, x0, y0, x1, y1, x2, y2, x3, y3)
    } else {
        let mut min_dist = distance_to_point(px, py, x0, y0);
        let dist_end = distance_to_point(px, py, x3, y3);
        if dist_end < min_dist {
            min_dist = dist_end;
        }

        let ax = -x0 + f32::new(3.0) * x1 - f32::new(3.0) * x2 + x3;
        let ay = -y0 + f32::new(3.0) * y1 - f32::new(3.0) * y2 + y3;
        let bx = f32::new(3.0) * x0 - f32::new(6.0) * x1 + f32::new(3.0) * x2;
        let by = f32::new(3.0) * y0 - f32::new(6.0) * y1 + f32::new(3.0) * y2;
        let cx = -f32::new(3.0) * x0 + f32::new(3.0) * x1;
        let cy = -f32::new(3.0) * y0 + f32::new(3.0) * y1;
        let dx = x0 - px;
        let dy = y0 - py;

        let a = f32::new(3.0) * (ax * ax + ay * ay);
        if a.abs() > f32::new(1.0e-8) {
            let b = f32::new(5.0) * (ax * bx + ay * by);
            let c = f32::new(4.0) * (ax * cx + ay * cy)
                + f32::new(2.0) * (bx * bx + by * by);
            let d = f32::new(3.0) * ((bx * cx + by * cy) + (ax * dx + ay * dy));
            let e = (cx * cx + cy * cy) + f32::new(2.0) * (dx * bx + dy * by);
            let f = dx * cx + dy * cy;

            let b = b / a;
            let c = c / a;
            let d = d / a;
            let e = e / a;
            let f = f / a;

            let p1a = f32::new(2.0 / 5.0) * c - f32::new(4.0 / 25.0) * b * b;
            let p1b = f32::new(3.0 / 5.0) * d - f32::new(3.0 / 25.0) * b * c;
            let p1c = f32::new(4.0 / 5.0) * e - f32::new(2.0 / 25.0) * b * d;
            let p1d = f - b * e / f32::new(25.0);

            let q_root = -b / f32::new(5.0);

            let p_roots = solve_cubic(p1a, p1b, p1c, p1d);
            let num_p = p_roots[0] as u32;

            let mut intervals = Line::empty(4usize);
            let mut num_intervals = u32::new(0);
            if q_root >= zero && q_root <= one {
                intervals[num_intervals as usize] = q_root;
                num_intervals += 1;
            }
            let mut i = u32::new(0);
            while i < num_p {
                intervals[num_intervals as usize] = p_roots[(i + 1) as usize];
                num_intervals += 1;
                i += u32::new(1);
            }

            // sort intervals
            let mut j = u32::new(1);
            while j < num_intervals {
                let mut k = j;
                while k > 0 {
                    let a = intervals[(k - 1) as usize];
                    let b = intervals[k as usize];
                    if a <= b {
                        break;
                    }
                    intervals[(k - 1) as usize] = b;
                    intervals[k as usize] = a;
                    k -= u32::new(1);
                }
                j += u32::new(1);
            }

            let mut lower = zero;
            let mut idx = u32::new(0);
            while idx <= num_intervals {
                if idx < num_intervals && intervals[idx as usize] < zero {
                    idx += u32::new(1);
                } else {
                    let upper = if idx < num_intervals {
                        min_f32(intervals[idx as usize], one)
                    } else {
                        one
                    };
                    let mut lb = lower;
                    let mut ub = upper;
                    let mut lb_eval = eval_quintic(lb, b, c, d, e, f);
                    let mut ub_eval = eval_quintic(ub, b, c, d, e, f);
                    if lb_eval * ub_eval > zero {
                        lower = upper;
                        idx += u32::new(1);
                    } else {
                        if lb_eval > ub_eval {
                            let tmp = lb;
                            lb = ub;
                            ub = tmp;
                            let tmp_eval = lb_eval;
                            lb_eval = ub_eval;
                            ub_eval = tmp_eval;
                        }
                        let mut t = (lb + ub) * f32::new(0.5);
                        let mut it = u32::new(0);
                        while it < u32::new(20) {
                            if t < lb || t > ub {
                                t = (lb + ub) * f32::new(0.5);
                            }
                            let value = eval_quintic(t, b, c, d, e, f);
                            if abs_f32(value) < f32::new(1.0e-5) || it == u32::new(19) {
                                break;
                            }
                            if value > zero {
                                ub = t;
                            } else {
                                lb = t;
                            }
                            let derivative = eval_quintic_deriv(t, b, c, d, e);
                            t = t - value / derivative;
                            it += u32::new(1);
                        }

                        if t >= zero && t <= one {
                            let tt = one - t;
                            let x = (tt * tt * tt) * x0
                                + (f32::new(3.0) * tt * tt * t) * x1
                                + (f32::new(3.0) * tt * t * t) * x2
                                + (t * t * t) * x3;
                            let y = (tt * tt * tt) * y0
                                + (f32::new(3.0) * tt * tt * t) * y1
                                + (f32::new(3.0) * tt * t * t) * y2
                                + (t * t * t) * y3;
                            let dist = distance_to_point(px, py, x, y);
                            if dist < min_dist {
                                min_dist = dist;
                            }
                        }

                        if upper >= one {
                            break;
                        }
                        lower = upper;
                        idx += u32::new(1);
                    }
                }
            }
        }
        min_dist
    };

    min_dist
}

#[cube]
fn distance_to_cubic_approx_with_t(
    px: f32,
    py: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    x3: f32,
    y3: f32,
) -> Line<f32> {
    let steps = u32::new(8);
    let inv = f32::new(1.0) / f32::cast_from(steps);
    let mut prev_x = x0;
    let mut prev_y = y0;
    let mut prev_t = f32::new(0.0);
    let mut best_dist = distance_to_point(px, py, x0, y0);
    let mut best_t = prev_t;
    let mut i = u32::new(1);
    while i <= steps {
        let t = f32::cast_from(i) * inv;
        let tt = f32::new(1.0) - t;
        let tt2 = tt * tt;
        let t2 = t * t;
        let a = tt2 * tt;
        let b = f32::new(3.0) * tt2 * t;
        let c = f32::new(3.0) * tt * t2;
        let d = t2 * t;
        let cx = a * x0 + b * x1 + c * x2 + d * x3;
        let cy = a * y0 + b * y1 + c * y2 + d * y3;
        let seg = distance_to_segment_with_t(px, py, prev_x, prev_y, cx, cy);
        let dist = seg[0];
        let seg_t = seg[1];
        let local_t = prev_t + seg_t * (t - prev_t);
        if dist < best_dist {
            best_dist = dist;
            best_t = local_t;
        }
        prev_x = cx;
        prev_y = cy;
        prev_t = t;
        i += u32::new(1);
    }
    let mut out = Line::empty(2usize);
    out[0] = best_dist;
    out[1] = best_t;
    out
}

#[cube]
fn closest_point_cubic_with_t(
    px: f32,
    py: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    x3: f32,
    y3: f32,
    use_distance_approx: bool,
) -> Line<f32> {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let mut best_t = zero;
    let mut best_dist = distance_to_point(px, py, x0, y0);
    if use_distance_approx {
        let approx = distance_to_cubic_approx_with_t(px, py, x0, y0, x1, y1, x2, y2, x3, y3);
        best_dist = approx[0];
        best_t = approx[1];
    } else {
        let dist_end = distance_to_point(px, py, x3, y3);
        if dist_end < best_dist {
            best_dist = dist_end;
            best_t = one;
        }

        let ax = -x0 + f32::new(3.0) * x1 - f32::new(3.0) * x2 + x3;
        let ay = -y0 + f32::new(3.0) * y1 - f32::new(3.0) * y2 + y3;
        let bx = f32::new(3.0) * x0 - f32::new(6.0) * x1 + f32::new(3.0) * x2;
        let by = f32::new(3.0) * y0 - f32::new(6.0) * y1 + f32::new(3.0) * y2;
        let cx = -f32::new(3.0) * x0 + f32::new(3.0) * x1;
        let cy = -f32::new(3.0) * y0 + f32::new(3.0) * y1;
        let dx = x0 - px;
        let dy = y0 - py;

        let a = f32::new(3.0) * (ax * ax + ay * ay);
        if a.abs() > f32::new(1.0e-8) {
            let b = f32::new(5.0) * (ax * bx + ay * by);
            let c = f32::new(4.0) * (ax * cx + ay * cy)
                + f32::new(2.0) * (bx * bx + by * by);
            let d = f32::new(3.0) * ((bx * cx + by * cy) + (ax * dx + ay * dy));
            let e = (cx * cx + cy * cy) + f32::new(2.0) * (dx * bx + dy * by);
            let f = dx * cx + dy * cy;

            let b = b / a;
            let c = c / a;
            let d = d / a;
            let e = e / a;
            let f = f / a;

            let p1a = f32::new(2.0 / 5.0) * c - f32::new(4.0 / 25.0) * b * b;
            let p1b = f32::new(3.0 / 5.0) * d - f32::new(3.0 / 25.0) * b * c;
            let p1c = f32::new(4.0 / 5.0) * e - f32::new(2.0 / 25.0) * b * d;
            let p1d = f - b * e / f32::new(25.0);

            let q_root = -b / f32::new(5.0);

            let p_roots = solve_cubic(p1a, p1b, p1c, p1d);
            let num_p = p_roots[0] as u32;

            let mut intervals = Line::empty(4usize);
            let mut num_intervals = u32::new(0);
            if q_root >= zero && q_root <= one {
                intervals[num_intervals as usize] = q_root;
                num_intervals += 1;
            }
            let mut i = u32::new(0);
            while i < num_p {
                intervals[num_intervals as usize] = p_roots[(i + 1) as usize];
                num_intervals += 1;
                i += u32::new(1);
            }

            let mut j = u32::new(1);
            while j < num_intervals {
                let mut k = j;
                while k > 0 {
                    let a = intervals[(k - 1) as usize];
                    let b = intervals[k as usize];
                    if a <= b {
                        break;
                    }
                    intervals[(k - 1) as usize] = b;
                    intervals[k as usize] = a;
                    k -= u32::new(1);
                }
                j += u32::new(1);
            }

            let mut lower = zero;
            let mut idx = u32::new(0);
            while idx <= num_intervals {
                if idx < num_intervals && intervals[idx as usize] < zero {
                    idx += u32::new(1);
                } else {
                    let upper = if idx < num_intervals {
                        min_f32(intervals[idx as usize], one)
                    } else {
                        one
                    };
                    let mut lb = lower;
                    let mut ub = upper;
                    let mut lb_eval = eval_quintic(lb, b, c, d, e, f);
                    let mut ub_eval = eval_quintic(ub, b, c, d, e, f);
                    if lb_eval * ub_eval > zero {
                        lower = upper;
                        idx += u32::new(1);
                    } else {
                        if lb_eval > ub_eval {
                            let tmp = lb;
                            lb = ub;
                            ub = tmp;
                            let tmp_eval = lb_eval;
                            lb_eval = ub_eval;
                            ub_eval = tmp_eval;
                        }
                        let mut t = (lb + ub) * f32::new(0.5);
                        let mut it = u32::new(0);
                        while it < u32::new(20) {
                            if t < lb || t > ub {
                                t = (lb + ub) * f32::new(0.5);
                            }
                            let value = eval_quintic(t, b, c, d, e, f);
                            if abs_f32(value) < f32::new(1.0e-5) || it == u32::new(19) {
                                break;
                            }
                            if value > zero {
                                ub = t;
                            } else {
                                lb = t;
                            }
                            let derivative = eval_quintic_deriv(t, b, c, d, e);
                            t = t - value / derivative;
                            it += u32::new(1);
                        }

                        if t >= zero && t <= one {
                            let tt = one - t;
                            let x = (tt * tt * tt) * x0
                                + (f32::new(3.0) * tt * tt * t) * x1
                                + (f32::new(3.0) * tt * t * t) * x2
                                + (t * t * t) * x3;
                            let y = (tt * tt * tt) * y0
                                + (f32::new(3.0) * tt * tt * t) * y1
                                + (f32::new(3.0) * tt * t * t) * y2
                                + (t * t * t) * y3;
                            let dist = distance_to_point(px, py, x, y);
                            if dist < best_dist {
                                best_dist = dist;
                                best_t = t;
                            }
                        }

                        if upper >= one {
                            break;
                        }
                        lower = upper;
                        idx += u32::new(1);
                    }
                }
            }
        }
    }

    let mut out = Line::empty(2usize);
    out[0] = best_dist;
    out[1] = best_t;
    out
}

#[cube]
fn distance_to_cubic_approx(
    px: f32,
    py: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    x3: f32,
    y3: f32,
) -> f32 {
    let steps = u32::new(8);
    let inv = f32::new(1.0) / f32::cast_from(steps);
    let mut prev_x = x0;
    let mut prev_y = y0;
    let mut min_dist = distance_to_point(px, py, x0, y0);
    let mut i = u32::new(1);
    while i <= steps {
        let t = f32::cast_from(i) * inv;
        let tt = f32::new(1.0) - t;
        let tt2 = tt * tt;
        let t2 = t * t;
        let a = tt2 * tt;
        let b = f32::new(3.0) * tt2 * t;
        let c = f32::new(3.0) * tt * t2;
        let d = t2 * t;
        let cx = a * x0 + b * x1 + c * x2 + d * x3;
        let cy = a * y0 + b * y1 + c * y2 + d * y3;
        let dist = distance_to_segment(px, py, prev_x, prev_y, cx, cy);
        if dist < min_dist {
            min_dist = dist;
        }
        prev_x = cx;
        prev_y = cy;
        i += u32::new(1);
    }
    min_dist
}

#[cube]
fn eval_quintic(t: f32, b: f32, c: f32, d: f32, e: f32, f: f32) -> f32 {
    ((((t + b) * t + c) * t + d) * t + e) * t + f
}

#[cube]
fn eval_quintic_deriv(t: f32, b: f32, c: f32, d: f32, e: f32) -> f32 {
    (((f32::new(5.0) * t + f32::new(4.0) * b) * t + f32::new(3.0) * c) * t
        + f32::new(2.0) * d)
        * t
        + e
}

#[cube]
fn quadratic_closest_pt_approx(
    px: f32,
    py: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
) -> Line<f32> {
    let zero = f32::new(0.0);
    let two = f32::new(2.0);
    let b0x = x0 - px;
    let b0y = y0 - py;
    let b1x = x1 - px;
    let b1y = y1 - py;
    let b2x = x2 - px;
    let b2y = y2 - py;

    let a = det2(b0x, b0y, b2x, b2y);
    let b = two * det2(b1x, b1y, b0x, b0y);
    let d = two * det2(b2x, b2y, b1x, b1y);
    let f = b * d - a * a;

    let d21x = b2x - b1x;
    let d21y = b2y - b1y;
    let d10x = b1x - b0x;
    let d10y = b1y - b0y;
    let d20x = b2x - b0x;
    let d20y = b2y - b0y;

    let mut gfx = two * (b * d21x + d * d10x + a * d20x);
    let mut gfy = two * (b * d21y + d * d10y + a * d20y);
    // rotate 90 degrees
    let tmp = gfx;
    gfx = gfy;
    gfy = -tmp;

    let gf_dot = gfx * gfx + gfy * gfy;
    let mut t = zero;
    if gf_dot > f32::new(1.0e-8) {
        let ppx = -f * gfx / gf_dot;
        let ppy = -f * gfy / gf_dot;
        let d0px = b0x - ppx;
        let d0py = b0y - ppy;
        let ap = det2(d0px, d0py, d20x, d20y);
        let bp = two * det2(d10x, d10y, d0px, d0py);
        let denom = two * a + b + d;
        if denom.abs() > f32::new(1.0e-8) {
            t = clamp01((ap + bp) / denom);
        } else {
            t = clamp01((ap + bp) * f32::new(0.5));
        }
    }

    let one = f32::new(1.0);
    let tt = one - t;
    let mut out = Line::empty(4usize);
    out[0] = (tt * tt) * x0 + (two * tt * t) * x1 + (t * t) * x2;
    out[1] = (tt * tt) * y0 + (two * tt * t) * y1 + (t * t) * y2;
    out[2] = t;
    out[3] = f32::new(0.0);
    out
}

#[cube]
fn det2(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    ax * by - ay * bx
}

#[cube]
fn solve_quadratic(a: f32, b: f32, c: f32) -> Line<f32> {
    let mut out = Line::empty(4usize);
    let zero = f32::new(0.0);
    let discrim = b * b - f32::new(4.0) * a * c;
    if discrim < zero {
        out[0] = zero;
        out[1] = zero;
        out[2] = zero;
        out[3] = zero;
    } else {
        let root = discrim.sqrt();
        let half = f32::new(0.5);
        let q = if b < zero { -half * (b - root) } else { -half * (b + root) };
        let t0 = q / a;
        let t1 = c / q;
        let mut lo = t0;
        let mut hi = t1;
        if lo > hi {
            let tmp = lo;
            lo = hi;
            hi = tmp;
        }
        out[0] = f32::new(2.0);
        out[1] = lo;
        out[2] = hi;
        out[3] = zero;
    }
    out
}

#[cube]
fn solve_cubic(a: f32, b: f32, c: f32, d: f32) -> Line<f32> {
    let mut out = Line::empty(4usize);
    let zero = f32::new(0.0);
    let eps = f32::new(1.0e-6);
    if abs_f32(a) < eps {
        let roots = solve_quadratic(b, c, d);
        out[0] = roots[0];
        out[1] = roots[1];
        out[2] = roots[2];
        out[3] = zero;
    } else {
        let bb = b / a;
        let cc = c / a;
        let dd = d / a;

        let third = f32::new(1.0 / 3.0);
        let q = (bb * bb - f32::new(3.0) * cc) / f32::new(9.0);
        let r = (f32::new(2.0) * bb * bb * bb - f32::new(9.0) * bb * cc + f32::new(27.0) * dd)
            / f32::new(54.0);
        let r2 = r * r;
        let q3 = q * q * q;
        if r2 < q3 {
            let theta = (r / q3.sqrt()).acos();
            let two = f32::new(2.0);
            let sqrt_q = q.sqrt();
            out[0] = f32::new(3.0);
            out[1] = -two * sqrt_q * (theta * third).cos() - bb * third;
            out[2] =
                -two * sqrt_q * ((theta + f32::new(2.0) * f32::new(3.14159265)) * third).cos()
                    - bb * third;
            out[3] =
                -two * sqrt_q * ((theta - f32::new(2.0) * f32::new(3.14159265)) * third).cos()
                    - bb * third;
        } else {
            let a_root = if r > zero {
                -cbrt(r + (r2 - q3).sqrt())
            } else {
                cbrt(-r + (r2 - q3).sqrt())
            };
            let b_root = if abs_f32(a_root) > eps { q / a_root } else { zero };
            out[0] = f32::new(1.0);
            out[1] = (a_root + b_root) - bb * third;
            out[2] = zero;
            out[3] = zero;
        }
    }

    out
}

#[cube]
fn cbrt(x: f32) -> f32 {
    let zero = f32::new(0.0);
    let one_third = f32::new(1.0 / 3.0);
    if x > zero {
        x.powf(one_third)
    } else if x < zero {
        -(-x).powf(one_third)
    } else {
        zero
    }
}

#[cube]
fn sdf_coverage(dist: f32, aa: f32) -> f32 {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let two = f32::new(2.0);
    let three = f32::new(3.0);
    let mut coverage = if dist < zero { one } else { zero };
    if aa > zero {
        let inv = one / (two * aa);
        let mut t = (dist + aa) * inv;
        t = clamp01(t);
        let smooth = t * t * (three - two * t);
        coverage = one - smooth;
    }
    coverage
}

#[cube]
fn smoothstep_unit(d: f32) -> f32 {
    let t = clamp01((d + f32::new(1.0)) * f32::new(0.5));
    t * t * (f32::new(3.0) - f32::new(2.0) * t)
}

#[cube]
fn filter_weight(filter_type: u32, dx: f32, dy: f32, radius: f32) -> f32 {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let two = f32::new(2.0);
    let four = f32::new(4.0);
    let three = f32::new(3.0);
    let pi = f32::new(3.14159265);
    let adx = abs_f32(dx);
    let ady = abs_f32(dy);

    let weight = if radius <= zero {
        one
    } else if adx > radius || ady > radius {
        zero
    } else if filter_type == FILTER_BOX {
        let denom = (two * radius) * (two * radius);
        one / denom
    } else if filter_type == FILTER_TENT {
        let fx = radius - adx;
        let fy = radius - ady;
        let norm = one / (radius * radius);
        fx * fy * norm * norm
    } else if filter_type == FILTER_RADIAL_PARABOLIC {
        let rx = one - (dx / radius) * (dx / radius);
        let ry = one - (dy / radius) * (dy / radius);
        (four / three) * rx * (four / three) * ry
    } else if filter_type == FILTER_HANN {
        let ndx = (dx / (two * radius)) + f32::new(0.5);
        let ndy = (dy / (two * radius)) + f32::new(0.5);
        let fx = f32::new(0.5) * (one - (two * pi * ndx).cos());
        let fy = f32::new(0.5) * (one - (two * pi * ndy).cos());
        let norm = one / (radius * radius);
        fx * fy * norm
    } else {
        one
    };

    weight
}

#[cube]
fn clamp01(v: f32) -> f32 {
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
fn min_f32(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

#[cube]
fn max_f32(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

#[cube]
fn abs_f32(a: f32) -> f32 {
    let zero = f32::new(0.0);
    if a < zero { -a } else { a }
}

#[cube]
fn clamp_f32(v: f32, min_v: f32, max_v: f32) -> f32 {
    if v < min_v {
        min_v
    } else if v > max_v {
        max_v
    } else {
        v
    }
}

#[cube]
fn vec2_dot(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    ax * bx + ay * by
}

#[cube]
fn vec2_length(ax: f32, ay: f32) -> f32 {
    let l_sq = vec2_dot(ax, ay, ax, ay);
    l_sq.sqrt()
}

// === Backward kernels ===

const MAX_FRAGMENTS: usize = 256;
const MAX_PREFILTER_FRAGMENTS: usize = 64;
const MAT3_STRIDE: usize = 4;
const MAT3_SIZE: usize = 12;
const AFFINE_SIZE: usize = 8;

#[cube]
fn vec2_normalize(ax: f32, ay: f32) -> Line<f32> {
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
fn d_length_vec2(ax: f32, ay: f32, d_l: f32) -> Line<f32> {
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
fn d_distance(ax: f32, ay: f32, bx: f32, by: f32, d_out: f32, d_a: &mut Line<f32>, d_b: &mut Line<f32>) {
    let dx = bx - ax;
    let dy = by - ay;
    let d_len = d_length_vec2(dx, dy, d_out);
    d_a[0] -= d_len[0];
    d_a[1] -= d_len[1];
    d_b[0] += d_len[0];
    d_b[1] += d_len[1];
}

#[cube]
fn d_normalize_vec2(ax: f32, ay: f32, d_nx: f32, d_ny: f32) -> Line<f32> {
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
fn d_smoothstep_unit(d: f32, d_ret: f32) -> f32 {
    let mut out = f32::new(0.0);
    if d >= f32::new(-1.0) && d <= f32::new(1.0) {
        let t = (d + f32::new(1.0)) * f32::new(0.5);
        let d_t = d_ret * (f32::new(6.0) * t - f32::new(6.0) * t * t);
        out = d_t * f32::new(0.5);
    }
    out
}

#[cube]
fn xform_pt_affine(m00: f32, m01: f32, m02: f32, m10: f32, m11: f32, m12: f32, px: f32, py: f32) -> Line<f32> {
    let mut out = Line::empty(2usize);
    out[0] = m00 * px + m01 * py + m02;
    out[1] = m10 * px + m11 * py + m12;
    out
}

#[cube]
fn d_xform_pt_affine(
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

#[cube]
fn mat3_from_affine(m00: f32, m01: f32, m02: f32, m10: f32, m11: f32, m12: f32) -> Line<f32> {
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

#[cube]
fn mat3_transpose(m: Line<f32>) -> Line<f32> {
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

#[cube]
fn mat3_mul(a: Line<f32>, b: Line<f32>) -> Line<f32> {
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

#[cube]
fn mat3_scale(m: Line<f32>, s: f32) -> Line<f32> {
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

#[cube]
fn atomic_add_mat3(out: &mut Array<Atomic<f32>>, base: usize, m: Line<f32>) {
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

#[cube]
fn affine_grad_to_mat3(affine: Line<f32>) -> Line<f32> {
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

#[cube]
fn atomic_add_vec2(out: &mut Array<Atomic<f32>>, base: usize, x: f32, y: f32) {
    out[base].fetch_add(x);
    out[base + 1].fetch_add(y);
}

#[cube]
fn atomic_add_vec4(out: &mut Array<Atomic<f32>>, base: usize, x: f32, y: f32, z: f32, w: f32) {
    out[base].fetch_add(x);
    out[base + 1].fetch_add(y);
    out[base + 2].fetch_add(z);
    out[base + 3].fetch_add(w);
}

#[cube]
fn add_translation(d_translation: &mut Array<Atomic<f32>>, pixel_index: u32, dx: f32, dy: f32) {
    let base = (pixel_index * u32::new(2)) as usize;
    d_translation[base].fetch_add(dx);
    d_translation[base + 1].fetch_add(dy);
}

#[cube]
fn accumulate_background_grad(
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

#[cube]
fn closest_point_circle(cx: f32, cy: f32, radius: f32, px: f32, py: f32) -> Line<f32> {
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

#[cube]
fn closest_point_rect(min_x: f32, min_y: f32, max_x: f32, max_y: f32, px: f32, py: f32) -> Line<f32> {
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

#[cube]
fn closest_point_ellipse(cx: f32, cy: f32, rx_in: f32, ry_in: f32, px: f32, py: f32) -> Line<f32> {
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

#[cube]
fn closest_point_path(
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

#[cube]
fn closest_point_shape(
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

#[cube]
fn path_point_index(idx: u32, count: u32, is_closed: u32) -> u32 {
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

#[cube]
fn path_point_id(path_num_controls: &Array<u32>, ctrl_offset: u32, base_point_id: u32) -> u32 {
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

#[cube]
fn load_path_point(
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

#[cube]
fn add_path_point_grad(
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

#[cube]
fn rect_dist_to_seg(px: f32, py: f32, p0x: f32, p0y: f32, p1x: f32, p1y: f32) -> f32 {
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

#[cube]
fn rect_update_seg(
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

#[cube]
fn d_closest_point_rect(
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

#[cube]
fn d_closest_point_ellipse(
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

#[cube]
fn d_closest_point_path(
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

#[cube]
fn d_closest_point(
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

#[cube]
fn d_compute_distance(
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

#[cube]
fn gather_d_color(
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

#[cube]
fn d_compute_filter_weight(
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

#[cube]
fn accumulate_filter_gradient(
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

#[cube]
fn d_sample_paint(
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


#[cube]
fn fill_inside_group(
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

#[cube]
fn compute_distance_group(
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

#[cube]
fn sample_color(
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

#[cube]
fn sample_color_prefiltered(
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

#[cube]
fn sample_distance_sdf(
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
