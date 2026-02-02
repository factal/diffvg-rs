use cubecl::prelude::*;
use crate::gpu::constants::*;
use super::math::*;

#[cube]
pub(super) fn paint_color(
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
pub(super) fn sample_gradient(
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
pub(super) fn filter_weight(filter_type: u32, dx: f32, dy: f32, radius: f32) -> f32 {
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
