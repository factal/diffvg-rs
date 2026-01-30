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

    let mut out = Line::empty(3usize);
    out[0] = out_val;
    out[1] = new_state[0];
    out[2] = new_state[1];
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
    let mut out = Line::empty(3usize);
    out[0] = (tt * tt) * x0 + (two * tt * t) * x1 + (t * t) * x2;
    out[1] = (tt * tt) * y0 + (two * tt * t) * y1 + (t * t) * y2;
    out[2] = t;
    out
}

#[cube]
fn det2(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    ax * by - ay * bx
}

#[cube]
fn solve_quadratic(a: f32, b: f32, c: f32) -> Line<f32> {
    let mut out = Line::empty(3usize);
    let zero = f32::new(0.0);
    let discrim = b * b - f32::new(4.0) * a * c;
    if discrim < zero {
        out[0] = zero;
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
