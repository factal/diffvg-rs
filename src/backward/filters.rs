use crate::grad::DFilter;
use crate::math::{Vec2, Vec4};
use crate::renderer::rng::Pcg32;
use crate::scene::{FilterType, Scene};

use super::math::dot4;

pub(super) fn build_weight_image(
    scene: &Scene,
    samples_x: u32,
    samples_y: u32,
    seed: u32,
    use_prefiltering: bool,
    use_jitter: bool,
) -> Vec<f32> {
    let width = scene.width as i32;
    let height = scene.height as i32;
    let mut weight_image = vec![0.0f32; (width * height) as usize];
    let total_samples = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(samples_x as usize)
        .saturating_mul(samples_y as usize);
    for idx in 0..total_samples {
        let sx = idx % samples_x as usize;
        let sy = (idx / samples_x as usize) % samples_y as usize;
        let x = (idx / (samples_x as usize * samples_y as usize)) % width as usize;
        let y = idx / (samples_x as usize * samples_y as usize * width as usize);

        let mut rx = 0.5f32;
        let mut ry = 0.5f32;
        if use_jitter {
            let mut rng = Pcg32::new(idx as u64, seed as u64);
            rx = rng.next_f32();
            ry = rng.next_f32();
        }
        if use_prefiltering {
            rx = 0.5;
            ry = 0.5;
        }
        let px = x as f32 + (sx as f32 + rx) / samples_x as f32;
        let py = y as f32 + (sy as f32 + ry) / samples_y as f32;

        let radius = scene.filter.radius;
        let ri = radius.ceil() as i32;
        for dy in -ri..=ri {
            for dx in -ri..=ri {
                let xx = x as i32 + dx;
                let yy = y as i32 + dy;
                if xx >= 0 && yy >= 0 && xx < width && yy < height {
                    let xc = xx as f32 + 0.5;
                    let yc = yy as f32 + 0.5;
                    let w = compute_filter_weight(
                        scene.filter.filter_type,
                        scene.filter.radius,
                        xc - px,
                        yc - py,
                    );
                    weight_image[(yy as usize) * (width as usize) + (xx as usize)] += w;
                }
            }
        }
    }
    weight_image
}

pub(super) fn gather_d_color(
    filter_type: FilterType,
    radius: f32,
    d_render_image: &[f32],
    weight_image: &[f32],
    width: i32,
    height: i32,
    pt: Vec2,
) -> Vec4 {
    let x = pt.x.floor() as i32;
    let y = pt.y.floor() as i32;
    let ri = radius.ceil() as i32;
    let mut out = Vec4::ZERO;
    for dy in -ri..=ri {
        for dx in -ri..=ri {
            let xx = x + dx;
            let yy = y + dy;
            if xx >= 0 && yy >= 0 && xx < width && yy < height {
                let xc = xx as f32 + 0.5;
                let yc = yy as f32 + 0.5;
                let w = compute_filter_weight(filter_type, radius, xc - pt.x, yc - pt.y);
                let weight_sum = weight_image[(yy as usize) * (width as usize) + (xx as usize)];
                if weight_sum > 0.0 {
                    let base = (yy as usize * width as usize + xx as usize) * 4;
                    out.x += (w / weight_sum) * d_render_image[base];
                    out.y += (w / weight_sum) * d_render_image[base + 1];
                    out.z += (w / weight_sum) * d_render_image[base + 2];
                    out.w += (w / weight_sum) * d_render_image[base + 3];
                }
            }
        }
    }
    out
}

pub(super) fn accumulate_filter_gradient(
    filter_type: FilterType,
    radius: f32,
    color: &Vec4,
    d_render_image: &[f32],
    weight_image: &[f32],
    width: i32,
    height: i32,
    pt: Vec2,
    d_filter: &mut DFilter,
) {
    let x = pt.x.floor() as i32;
    let y = pt.y.floor() as i32;
    let ri = radius.ceil() as i32;
    for dy in -ri..=ri {
        for dx in -ri..=ri {
            let xx = x + dx;
            let yy = y + dy;
            if xx >= 0 && yy >= 0 && xx < width && yy < height {
                let weight_sum =
                    weight_image[(yy as usize) * (width as usize) + (xx as usize)];
                if weight_sum <= 0.0 {
                    continue;
                }
                let xc = xx as f32 + 0.5;
                let yc = yy as f32 + 0.5;
                let w = compute_filter_weight(filter_type, radius, xc - pt.x, yc - pt.y);
                if w <= 0.0 {
                    continue;
                }
                let base = (yy as usize * width as usize + xx as usize) * 4;
                let d_pixel = Vec4::new(
                    d_render_image[base],
                    d_render_image[base + 1],
                    d_render_image[base + 2],
                    d_render_image[base + 3],
                );
                let dot = dot4(d_pixel, *color);
                let denom = weight_sum * weight_sum;
                let d_weight = if denom > 0.0 {
                    (dot * weight_sum - w * dot * (weight_sum - w)) / denom
                } else {
                    0.0
                };
                if d_weight != 0.0 {
                    d_compute_filter_weight(filter_type, radius, xc - pt.x, yc - pt.y, d_weight, d_filter);
                }
            }
        }
    }
}

fn compute_filter_weight(filter_type: FilterType, radius: f32, dx: f32, dy: f32) -> f32 {
    if radius <= 0.0 {
        return 1.0;
    }
    if dx.abs() > radius || dy.abs() > radius {
        return 0.0;
    }
    match filter_type {
        FilterType::Box => 1.0 / (2.0 * radius).powi(2),
        FilterType::Tent => {
            let fx = radius - dx.abs();
            let fy = radius - dy.abs();
            let norm = 1.0 / (radius * radius);
            fx * fy * norm * norm
        }
        FilterType::RadialParabolic => {
            let rx = 1.0 - (dx / radius) * (dx / radius);
            let ry = 1.0 - (dy / radius) * (dy / radius);
            (4.0 / 3.0) * rx * (4.0 / 3.0) * ry
        }
        FilterType::Hann => {
            let ndx = (dx / (2.0 * radius)) + 0.5;
            let ndy = (dy / (2.0 * radius)) + 0.5;
            let fx = 0.5 * (1.0 - (2.0 * core::f32::consts::PI * ndx).cos());
            let fy = 0.5 * (1.0 - (2.0 * core::f32::consts::PI * ndy).cos());
            let norm = 1.0 / (radius * radius);
            fx * fy * norm
        }
    }
}
fn d_compute_filter_weight(
    filter_type: FilterType,
    radius: f32,
    dx: f32,
    dy: f32,
    d_return: f32,
    d_filter: &mut DFilter,
) {
    match filter_type {
        FilterType::Box => {
            let denom = 2.0 * radius;
            if denom != 0.0 {
                d_filter.radius += d_return * (-2.0) * denom / denom.powi(3);
            }
        }
        FilterType::Tent => {
            let fx = radius - dx.abs();
            let fy = radius - dy.abs();
            let norm = 1.0 / (radius * radius);
            let d_fx = d_return * fy * norm;
            let d_fy = d_return * fx * norm;
            let d_norm = d_return * fx * fy;
            if radius != 0.0 {
                d_filter.radius += d_fx + d_fy + (-4.0) * d_norm / radius.powi(5);
            }
        }
        FilterType::RadialParabolic => {
            let r3 = radius * radius * radius;
            if r3 != 0.0 {
                let d_radius = -(2.0 * dx * dx + 2.0 * dy * dy) / r3;
                d_filter.radius += d_radius;
            }
        }
        FilterType::Hann => {
            let ndx = (dx / (2.0 * radius)) + 0.5;
            let ndy = (dy / (2.0 * radius)) + 0.5;
            let fx = 0.5 * (1.0 - (2.0 * core::f32::consts::PI * ndx).cos());
            let fy = 0.5 * (1.0 - (2.0 * core::f32::consts::PI * ndy).cos());
            let norm = 1.0 / (radius * radius);
            let d_fx = d_return * fy * norm;
            let d_fy = d_return * fx * norm;
            let d_norm = d_return * fx * fy;
            let d_ndx = d_fx * 0.5 * (2.0 * core::f32::consts::PI * ndx).sin() * (2.0 * core::f32::consts::PI);
            let d_ndy = d_fy * 0.5 * (2.0 * core::f32::consts::PI * ndy).sin() * (2.0 * core::f32::consts::PI);
            d_filter.radius += d_ndx * (-2.0 * dx / (2.0 * radius).powi(2))
                + d_ndy * (-2.0 * dy / (2.0 * radius).powi(2))
                + (-2.0) * d_norm / radius.powi(3);
        }
    }
}
