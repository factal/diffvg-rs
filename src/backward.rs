//! CPU backward rendering path for diffvg-rs.

use crate::color::Color;
use crate::distance::{compute_distance_bvh, within_distance_bvh, ClosestPathPoint, DistanceOptions, SceneBvh};
use crate::grad::{DFilter, DPaint, DShape, DShapeGeometry, DShapeGroup, SceneGrad};
use crate::math::{Mat3, Vec2, Vec4};
use crate::path_utils::{path_point, path_point_radius};
use crate::geometry::Path;
use crate::renderer::rng::Pcg32;
use crate::{RenderError, RenderOptions};
use crate::scene::{FillRule, FilterType, Paint, Shape, ShapeGeometry, ShapeGroup};

#[derive(Debug, Copy, Clone)]
pub struct BackwardOptions {
    pub compute_translation: bool,
}

impl Default for BackwardOptions {
    fn default() -> Self {
        Self {
            compute_translation: false,
        }
    }
}

pub fn render_backward(
    scene: &crate::scene::Scene,
    options: RenderOptions,
    backward_options: BackwardOptions,
    d_render_image: Option<&[f32]>,
    d_sdf_image: Option<&[f32]>,
) -> Result<SceneGrad, RenderError> {
    let width = scene.width as usize;
    let height = scene.height as usize;
    let pixel_count = width.saturating_mul(height);
    if let Some(d_render) = d_render_image {
        if d_render.len() != pixel_count.saturating_mul(4) {
            return Err(RenderError::InvalidScene("d_render_image size mismatch"));
        }
    }
    if let Some(d_sdf) = d_sdf_image {
        if d_sdf.len() != pixel_count {
            return Err(RenderError::InvalidScene("d_sdf_image size mismatch"));
        }
    }

    let include_background_image = d_render_image.is_some() && scene.background_image.is_some();
    let mut grads = SceneGrad::zeros_from_scene(
        scene,
        include_background_image,
        backward_options.compute_translation,
    );

    if scene.width == 0 || scene.height == 0 {
        return Ok(grads);
    }

    let samples_x = options.samples_x.max(1);
    let samples_y = options.samples_y.max(1);
    let num_samples = (samples_x as usize) * (samples_y as usize);
    let total_samples = pixel_count
        .saturating_mul(samples_x as usize)
        .saturating_mul(samples_y as usize);

    let bvh = SceneBvh::new(scene);
    let dist_options = DistanceOptions {
        path_tolerance: options.path_tolerance,
    };

    let use_prefiltering = options.use_prefiltering;
    let use_jitter = options.jitter && !use_prefiltering;

    let weight_image = if d_render_image.is_some() {
        Some(build_weight_image(
            scene,
            samples_x,
            samples_y,
            options.seed,
            use_prefiltering,
            use_jitter,
        ))
    } else {
        None
    };

    for idx in 0..total_samples {
        let sx = idx % samples_x as usize;
        let sy = (idx / samples_x as usize) % samples_y as usize;
        let x = (idx / (samples_x as usize * samples_y as usize)) % width;
        let y = idx / (samples_x as usize * samples_y as usize * width);

        let mut rx = 0.5f32;
        let mut ry = 0.5f32;
        if use_jitter {
            let mut rng = Pcg32::new(idx as u64, options.seed as u64);
            rx = rng.next_f32();
            ry = rng.next_f32();
        }
        let px = x as f32 + (sx as f32 + rx) / samples_x as f32;
        let py = y as f32 + (sy as f32 + ry) / samples_y as f32;
        let pt = Vec2::new(px, py);
        let npt = Vec2::new(px / scene.width as f32, py / scene.height as f32);

        let pixel_index = y * width + x;
        let background = sample_background(scene, pixel_index);

        if let Some(d_render) = d_render_image {
            let mut d_color = Vec4::ZERO;
            if let Some(weight_image) = weight_image.as_ref() {
                d_color = gather_d_color(
                    scene.filter.filter_type,
                    scene.filter.radius,
                    d_render,
                    weight_image,
                    scene.width as i32,
                    scene.height as i32,
                    pt,
                );
            }

            let color = if use_prefiltering {
                sample_color_prefiltered(
                    scene,
                    &bvh,
                    npt,
                    Some(background),
                    Some(d_color),
                    &mut grads,
                    backward_options.compute_translation.then_some(pixel_index),
                    pixel_index,
                    dist_options,
                )
            } else {
                sample_color(
                    scene,
                    &bvh,
                    npt,
                    Some(background),
                    Some(d_color),
                    None,
                    &mut grads,
                    backward_options.compute_translation.then_some(pixel_index),
                    pixel_index,
                )
            };

            if let Some(weight_image) = weight_image.as_ref() {
                accumulate_filter_gradient(
                    scene.filter.filter_type,
                    scene.filter.radius,
                    &color,
                    d_render,
                    weight_image,
                    scene.width as i32,
                    scene.height as i32,
                    pt,
                    &mut grads.filter,
                );
            }
        }

        if let Some(d_sdf) = d_sdf_image {
            let d_dist = d_sdf[pixel_index];
            let weight = if num_samples > 0 {
                1.0 / num_samples as f32
            } else {
                1.0
            };
            sample_distance(
                scene,
                &bvh,
                pt,
                weight,
                d_dist,
                &mut grads,
                backward_options.compute_translation.then_some(pixel_index),
                dist_options,
            );
        }
    }

    if !use_prefiltering && d_render_image.is_some() {
        if let Some(weight_image) = weight_image.as_ref() {
            boundary_sampling(
                scene,
                &bvh,
                samples_x,
                samples_y,
                options.seed,
                d_render_image.unwrap(),
                weight_image,
                &mut grads,
                backward_options.compute_translation,
            );
        }
    }

    finalize_background_gradients(scene, &mut grads);

    Ok(grads)
}

#[derive(Clone, Copy)]
struct Rgb {
    r: f32,
    g: f32,
    b: f32,
}

impl Rgb {
    fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    fn dot(self, other: Self) -> f32 {
        self.r * other.r + self.g * other.g + self.b * other.b
    }

    fn scale(self, s: f32) -> Self {
        Self::new(self.r * s, self.g * s, self.b * s)
    }

    fn add(self, other: Self) -> Self {
        Self::new(self.r + other.r, self.g + other.g, self.b + other.b)
    }

    fn sub(self, other: Self) -> Self {
        Self::new(self.r - other.r, self.g - other.g, self.b - other.b)
    }
}

#[derive(Clone, Copy)]
struct EdgeQuery {
    shape_group_id: usize,
    shape_id: usize,
    hit: bool,
}

#[derive(Clone, Copy)]
struct PathInfo {
    base_point_id: usize,
    point_id: usize,
    t: f32,
}

#[derive(Clone, Copy)]
struct PathBoundaryData {
    base_point_id: usize,
    point_id: usize,
    t: f32,
}

#[derive(Clone, Copy)]
struct BoundaryData {
    path: PathBoundaryData,
    is_stroke: bool,
}

#[derive(Clone, Copy)]
struct BoundarySample {
    pt: Vec2,
    local_pt: Vec2,
    normal: Vec2,
    shape_group_id: usize,
    shape_id: usize,
    t: f32,
    data: BoundaryData,
    pdf: f32,
}

fn sample_background(scene: &crate::scene::Scene, pixel_index: usize) -> Vec4 {
    if let Some(background_image) = scene.background_image.as_ref() {
        let base = pixel_index * 4;
        let r = background_image[base];
        let g = background_image[base + 1];
        let b = background_image[base + 2];
        let a = background_image[base + 3];
        Vec4::new(r * a, g * a, b * a, a)
    } else {
        Vec4::new(
            scene.background.r * scene.background.a,
            scene.background.g * scene.background.a,
            scene.background.b * scene.background.a,
            scene.background.a,
        )
    }
}

fn finalize_background_gradients(scene: &crate::scene::Scene, grads: &mut SceneGrad) {
    if let Some(image) = scene.background_image.as_ref() {
        if let Some(d_bg) = grads.background_image.as_mut() {
            for (i, chunk) in d_bg.chunks_mut(4).enumerate() {
                let base = i * 4;
                let r = image[base];
                let g = image[base + 1];
                let b = image[base + 2];
                let a = image[base + 3];
                let dr = chunk[0] * a;
                let dg = chunk[1] * a;
                let db = chunk[2] * a;
                let da = chunk[0] * r + chunk[1] * g + chunk[2] * b + chunk[3];
                chunk[0] = dr;
                chunk[1] = dg;
                chunk[2] = db;
                chunk[3] = da;
            }
        }
    } else {
        let r = scene.background.r;
        let g = scene.background.g;
        let b = scene.background.b;
        let a = scene.background.a;
        let d_pre = grads.background;
        grads.background = Color {
            r: d_pre.r * a,
            g: d_pre.g * a,
            b: d_pre.b * a,
            a: d_pre.r * r + d_pre.g * g + d_pre.b * b + d_pre.a,
        };
    }
}

fn build_weight_image(
    scene: &crate::scene::Scene,
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

fn gather_d_color(
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

fn accumulate_filter_gradient(
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

fn sample_distance(
    scene: &crate::scene::Scene,
    bvh: &SceneBvh,
    pt: Vec2,
    weight: f32,
    d_dist: f32,
    grads: &mut SceneGrad,
    translation_index: Option<usize>,
    options: DistanceOptions,
) {
    let mut best_dist = f32::INFINITY;
    let mut best_group = None;
    let mut best_shape = None;
    let mut best_point = Vec2::ZERO;
    let mut best_path: Option<PathInfo> = None;

    for group_id in (0..scene.groups.len()).rev() {
        if let Some(hit) = compute_distance_bvh(scene, bvh, group_id, pt, best_dist, options) {
            if hit.distance < best_dist {
                best_dist = hit.distance;
                best_group = Some(group_id);
                best_shape = Some(hit.shape_index);
                best_point = hit.point;
                best_path = path_info_from_closest(scene, hit.shape_index, hit.path);
            }
        }
    }

    let Some(group_id) = best_group else {
        return;
    };
    let shape_id = best_shape.unwrap_or(0);
    let mut signed_dist = best_dist * weight;
    let inside = scene.groups[group_id].fill.is_some() && is_inside_bvh(scene, bvh, group_id, pt);
    if inside {
        signed_dist = -signed_dist;
    }

    let d_abs_dist = if inside { -d_dist } else { d_dist };
    let d_pt = {
        let d_shape = &mut grads.shapes[shape_id];
        let d_group = &mut grads.shape_groups[group_id];
        d_compute_distance(
            &scene.groups[group_id],
            &scene.shapes[shape_id],
            pt,
            best_point,
            best_path,
            d_abs_dist,
            d_shape,
            d_group,
        )
    };
    accumulate_translation(grads, translation_index, d_pt * -1.0);
}

fn is_inside_bvh(scene: &crate::scene::Scene, bvh: &SceneBvh, group_id: usize, pt: Vec2) -> bool {
    crate::distance::is_inside_bvh(scene, bvh, group_id, pt)
}

fn sample_color(
    scene: &crate::scene::Scene,
    bvh: &SceneBvh,
    screen_pt: Vec2,
    background: Option<Vec4>,
    d_color: Option<Vec4>,
    mut edge_query: Option<&mut EdgeQuery>,
    grads: &mut SceneGrad,
    translation_index: Option<usize>,
    pixel_index: usize,
) -> Vec4 {
    if let Some(eq) = edge_query.as_deref_mut() {
        eq.hit = false;
    }

    let pt = Vec2::new(screen_pt.x * scene.width as f32, screen_pt.y * scene.height as f32);
    let mut fragments: Vec<Fragment> = Vec::new();
    for group_id in 0..scene.groups.len() {
        let group = &scene.groups[group_id];
        if group.stroke.is_some() {
            let hit = within_distance_edge(scene, bvh, group_id, pt, edge_query.as_deref_mut());
            if hit {
                let color = paint_color(group.stroke.as_ref().unwrap(), pt);
                fragments.push(Fragment {
                    color: Rgb::new(color.x, color.y, color.z),
                    alpha: color.w,
                    group_id,
                    is_stroke: true,
                });
            }
        }
        if group.fill.is_some() {
            let hit = is_inside_edge(scene, bvh, group_id, pt, edge_query.as_deref_mut());
            if hit {
                let color = paint_color(group.fill.as_ref().unwrap(), pt);
                fragments.push(Fragment {
                    color: Rgb::new(color.x, color.y, color.z),
                    alpha: color.w,
                    group_id,
                    is_stroke: false,
                });
            }
        }
    }

    if fragments.is_empty() {
        if let Some(bg) = background {
            if let Some(d) = d_color {
                accumulate_background_grad(scene, grads, pixel_index, d);
            }
            return bg;
        }
        return Vec4::ZERO;
    }

    fragments.sort_by_key(|f| f.group_id);

    let mut accum_color: Vec<Rgb> = Vec::with_capacity(fragments.len());
    let mut accum_alpha: Vec<f32> = Vec::with_capacity(fragments.len());
    let mut prev_color = Rgb::new(0.0, 0.0, 0.0);
    let mut prev_alpha = 0.0f32;
    if let Some(bg) = background {
        prev_color = Rgb::new(bg.x, bg.y, bg.z);
        prev_alpha = bg.w;
    }

    for fragment in &fragments {
        let new_color = fragment.color;
        let new_alpha = fragment.alpha;
        if let Some(eq) = edge_query.as_deref_mut() {
            if new_alpha >= 1.0 && eq.hit {
                eq.hit = false;
            }
            if eq.shape_group_id == fragment.group_id {
                eq.hit = true;
            }
        }
        let blended = prev_color.scale(1.0 - new_alpha).add(new_color.scale(new_alpha));
        let blended_alpha = prev_alpha * (1.0 - new_alpha) + new_alpha;
        accum_color.push(blended);
        accum_alpha.push(blended_alpha);
        prev_color = blended;
        prev_alpha = blended_alpha;
    }

    let mut final_color = accum_color[accum_color.len() - 1];
    let final_alpha = accum_alpha[accum_alpha.len() - 1];
    if final_alpha > 1.0e-6 {
        final_color = final_color.scale(1.0 / final_alpha);
    }

    if let Some(d) = d_color {
        let mut d_curr_color = Rgb::new(d.x, d.y, d.z);
        let mut d_curr_alpha = d.w;
        if final_alpha > 1.0e-6 {
            d_curr_color = d_curr_color.scale(1.0 / final_alpha);
            d_curr_alpha -= d_curr_color.dot(final_color) / final_alpha;
        }
        for i in (0..fragments.len()).rev() {
            let prev_alpha = if i > 0 {
                accum_alpha[i - 1]
            } else {
                background.map(|b| b.w).unwrap_or(0.0)
            };
            let prev_color = if i > 0 {
                accum_color[i - 1]
            } else {
                background
                    .map(|b| Rgb::new(b.x, b.y, b.z))
                    .unwrap_or(Rgb::new(0.0, 0.0, 0.0))
            };
            let fragment = fragments[i];
            let d_prev_alpha = d_curr_alpha * (1.0 - fragment.alpha);
            let mut d_alpha_i = d_curr_alpha * (1.0 - prev_alpha);
            d_alpha_i += d_curr_color.dot(fragment.color.sub(prev_color));
            let d_prev_color = d_curr_color.scale(1.0 - fragment.alpha);
            let d_color_i = d_curr_color.scale(fragment.alpha);

            let d_vec4 = Vec4::new(d_color_i.r, d_color_i.g, d_color_i.b, d_alpha_i);
            let mut translation_delta = Vec2::ZERO;
            let group = &scene.groups[fragment.group_id];
            let d_group = &mut grads.shape_groups[fragment.group_id];
            if fragment.is_stroke {
                if let (Some(paint), Some(d_paint)) = (group.stroke.as_ref(), d_group.stroke.as_mut()) {
                    d_sample_paint(paint, d_paint, pt, d_vec4, &mut translation_delta);
                }
            } else if let (Some(paint), Some(d_paint)) = (group.fill.as_ref(), d_group.fill.as_mut()) {
                d_sample_paint(paint, d_paint, pt, d_vec4, &mut translation_delta);
            }
            accumulate_translation(grads, translation_index, translation_delta);

            d_curr_color = d_prev_color;
            d_curr_alpha = d_prev_alpha;
        }

        if background.is_some() {
            let d_bg = Color {
                r: d_curr_color.r,
                g: d_curr_color.g,
                b: d_curr_color.b,
                a: d_curr_alpha,
            };
            accumulate_background_grad(scene, grads, pixel_index, d_bg.to_vec4());
        }
    }

    Vec4::new(final_color.r, final_color.g, final_color.b, final_alpha)
}

fn sample_color_prefiltered(
    scene: &crate::scene::Scene,
    bvh: &SceneBvh,
    screen_pt: Vec2,
    background: Option<Vec4>,
    d_color: Option<Vec4>,
    grads: &mut SceneGrad,
    translation_index: Option<usize>,
    pixel_index: usize,
    dist_options: DistanceOptions,
) -> Vec4 {
    let pt = Vec2::new(screen_pt.x * scene.width as f32, screen_pt.y * scene.height as f32);
    let mut fragments: Vec<PrefilterFragment> = Vec::new();
    for group_id in 0..scene.groups.len() {
        let group = &scene.groups[group_id];
        if group.stroke.is_some() {
            if let Some(hit) = compute_distance_bvh(scene, bvh, group_id, pt, f32::INFINITY, dist_options) {
                let shape = &scene.shapes[hit.shape_index];
                let d = hit.distance;
                let w = smoothstep(d.abs() + shape.stroke_width) - smoothstep(d.abs() - shape.stroke_width);
                if w > 0.0 {
                    let mut color = paint_color(group.stroke.as_ref().unwrap(), pt);
                    color.w *= w;
                    fragments.push(PrefilterFragment {
                        color: Rgb::new(color.x, color.y, color.z),
                        alpha: color.w,
                        group_id,
                        shape_id: hit.shape_index,
                        distance: d,
                        closest_pt: hit.point,
                        is_stroke: true,
                        path_info: path_info_from_closest(scene, hit.shape_index, hit.path),
                        within_distance: true,
                    });
                }
            }
        }
        if group.fill.is_some() {
            let hit = compute_distance_bvh(scene, bvh, group_id, pt, 1.0, dist_options);
            let inside = is_inside_bvh(scene, bvh, group_id, pt);
            if hit.is_some() || inside {
                let (shape_id, closest_pt, path_info, mut d) = if let Some(hit) = hit {
                    (hit.shape_index, hit.point, path_info_from_closest(scene, hit.shape_index, hit.path), hit.distance)
                } else {
                    (0usize, Vec2::ZERO, None, 0.0)
                };
                if !inside {
                    d = -d;
                }
                let w = smoothstep(d);
                if w > 0.0 {
                    let mut color = paint_color(group.fill.as_ref().unwrap(), pt);
                    color.w *= w;
                    fragments.push(PrefilterFragment {
                        color: Rgb::new(color.x, color.y, color.z),
                        alpha: color.w,
                        group_id,
                        shape_id,
                        distance: d,
                        closest_pt,
                        is_stroke: false,
                        path_info,
                        within_distance: hit.is_some(),
                    });
                }
            }
        }
    }

    if fragments.is_empty() {
        if let Some(bg) = background {
            if let Some(d) = d_color {
                accumulate_background_grad(scene, grads, pixel_index, d);
            }
            return bg;
        }
        return Vec4::ZERO;
    }

    fragments.sort_by_key(|f| f.group_id);

    let mut accum_color: Vec<Rgb> = Vec::with_capacity(fragments.len());
    let mut accum_alpha: Vec<f32> = Vec::with_capacity(fragments.len());
    let mut prev_color = Rgb::new(0.0, 0.0, 0.0);
    let mut prev_alpha = 0.0f32;
    if let Some(bg) = background {
        prev_color = Rgb::new(bg.x, bg.y, bg.z);
        prev_alpha = bg.w;
    }

    for fragment in &fragments {
        let new_color = fragment.color;
        let new_alpha = fragment.alpha;
        let blended = prev_color.scale(1.0 - new_alpha).add(new_color.scale(new_alpha));
        let blended_alpha = prev_alpha * (1.0 - new_alpha) + new_alpha;
        accum_color.push(blended);
        accum_alpha.push(blended_alpha);
        prev_color = blended;
        prev_alpha = blended_alpha;
    }

    let mut final_color = accum_color[accum_color.len() - 1];
    let final_alpha = accum_alpha[accum_alpha.len() - 1];
    if final_alpha > 1.0e-6 {
        final_color = final_color.scale(1.0 / final_alpha);
    }

    if let Some(d) = d_color {
        let mut d_curr_color = Rgb::new(d.x, d.y, d.z);
        let mut d_curr_alpha = d.w;
        if final_alpha > 1.0e-6 {
            d_curr_color = d_curr_color.scale(1.0 / final_alpha);
            d_curr_alpha -= d_curr_color.dot(final_color) / final_alpha;
        }
        for i in (0..fragments.len()).rev() {
            let prev_alpha = if i > 0 {
                accum_alpha[i - 1]
            } else {
                background.map(|b| b.w).unwrap_or(0.0)
            };
            let prev_color = if i > 0 {
                accum_color[i - 1]
            } else {
                background
                    .map(|b| Rgb::new(b.x, b.y, b.z))
                    .unwrap_or(Rgb::new(0.0, 0.0, 0.0))
            };
            let fragment = fragments[i];
            let d_prev_alpha = d_curr_alpha * (1.0 - fragment.alpha);
            let mut d_alpha_i = d_curr_alpha * (1.0 - prev_alpha);
            d_alpha_i += d_curr_color.dot(fragment.color.sub(prev_color));
            let d_prev_color = d_curr_color.scale(1.0 - fragment.alpha);
            let d_color_i = d_curr_color.scale(fragment.alpha);

            if fragment.is_stroke {
                let shape = &scene.shapes[fragment.shape_id];
                let d = fragment.distance;
                let abs_plus = d.abs() + shape.stroke_width;
                let abs_minus = d.abs() - shape.stroke_width;
                let w = smoothstep(abs_plus) - smoothstep(abs_minus);
                if w != 0.0 {
                    let d_w = if w > 0.0 { (fragment.alpha / w) * d_alpha_i } else { 0.0 };
                    let mut d_alpha_i = d_alpha_i * w;

                    let mut translation_delta = Vec2::ZERO;
                    let group = &scene.groups[fragment.group_id];
                    let d_group = &mut grads.shape_groups[fragment.group_id];
                    if let (Some(paint), Some(d_paint)) = (group.stroke.as_ref(), d_group.stroke.as_mut()) {
                        d_sample_paint(
                            paint,
                            d_paint,
                            pt,
                            Vec4::new(d_color_i.r, d_color_i.g, d_color_i.b, d_alpha_i),
                            &mut translation_delta,
                        );
                    }
                    accumulate_translation(grads, translation_index, translation_delta);

                    let d_abs_plus = d_smoothstep(abs_plus, d_w);
                    let d_abs_minus = -d_smoothstep(abs_minus, d_w);
                    let mut d_d = d_abs_plus + d_abs_minus;
                    if d < 0.0 {
                        d_d = -d_d;
                    }
                    let d_stroke_width = d_abs_plus - d_abs_minus;

                    let d_shape = &mut grads.shapes[fragment.shape_id];
                    d_shape.stroke_width += d_stroke_width;

                    if d_d.abs() > 1.0e-10 {
                        let d_pt = {
                            let d_group = &mut grads.shape_groups[fragment.group_id];
                            d_compute_distance(
                                &scene.groups[fragment.group_id],
                                shape,
                                pt,
                                fragment.closest_pt,
                                fragment.path_info,
                                d_d,
                                d_shape,
                                d_group,
                            )
                        };
                        accumulate_translation(grads, translation_index, d_pt * -1.0);
                    }
                }
            } else {
                let d = fragment.distance;
                let w = smoothstep(d);
                if w != 0.0 {
                    let d_w = if w > 0.0 { (fragment.alpha / w) * d_alpha_i } else { 0.0 };
                    let mut d_alpha_i = d_alpha_i * w;

                    let mut translation_delta = Vec2::ZERO;
                    let group = &scene.groups[fragment.group_id];
                    let d_group = &mut grads.shape_groups[fragment.group_id];
                    if let (Some(paint), Some(d_paint)) = (group.fill.as_ref(), d_group.fill.as_mut()) {
                        d_sample_paint(
                            paint,
                            d_paint,
                            pt,
                            Vec4::new(d_color_i.r, d_color_i.g, d_color_i.b, d_alpha_i),
                            &mut translation_delta,
                        );
                    }
                    accumulate_translation(grads, translation_index, translation_delta);

                    let mut d_d = d_smoothstep(d, d_w);
                    if d < 0.0 {
                        d_d = -d_d;
                    }

                    if d_d.abs() > 1.0e-10 && fragment.within_distance {
                        let d_pt = {
                            let d_shape = &mut grads.shapes[fragment.shape_id];
                            let d_group = &mut grads.shape_groups[fragment.group_id];
                            d_compute_distance(
                                &scene.groups[fragment.group_id],
                                &scene.shapes[fragment.shape_id],
                                pt,
                                fragment.closest_pt,
                                fragment.path_info,
                                d_d,
                                d_shape,
                                d_group,
                            )
                        };
                        accumulate_translation(grads, translation_index, d_pt * -1.0);
                    }
                }
            }

            d_curr_color = d_prev_color;
            d_curr_alpha = d_prev_alpha;
        }

        if background.is_some() {
            let d_bg = Color {
                r: d_curr_color.r,
                g: d_curr_color.g,
                b: d_curr_color.b,
                a: d_curr_alpha,
            };
            accumulate_background_grad(scene, grads, pixel_index, d_bg.to_vec4());
        }
    }

    Vec4::new(final_color.r, final_color.g, final_color.b, final_alpha)
}

#[derive(Clone, Copy)]
struct Fragment {
    color: Rgb,
    alpha: f32,
    group_id: usize,
    is_stroke: bool,
}

#[derive(Clone, Copy)]
struct PrefilterFragment {
    color: Rgb,
    alpha: f32,
    group_id: usize,
    shape_id: usize,
    distance: f32,
    closest_pt: Vec2,
    path_info: Option<PathInfo>,
    within_distance: bool,
    is_stroke: bool,
}

fn path_info_from_closest(
    scene: &crate::scene::Scene,
    shape_index: usize,
    info: Option<ClosestPathPoint>,
) -> Option<PathInfo> {
    let ShapeGeometry::Path { path } = &scene.shapes[shape_index].geometry else {
        return None;
    };
    let Some(info) = info else {
        return None;
    };
    let base_point_id = info.segment_index;
    let point_id = path_point_id(path, base_point_id);
    Some(PathInfo {
        base_point_id,
        point_id,
        t: info.t,
    })
}

fn path_point_id(path: &Path, base_point_id: usize) -> usize {
    let mut point_id = 0usize;
    let count = base_point_id.min(path.num_control_points.len());
    for &controls in path.num_control_points.iter().take(count) {
        point_id += match controls {
            0 => 1,
            1 => 2,
            2 => 3,
            _ => 0,
        };
    }
    point_id
}

fn paint_color(paint: &Paint, pt: Vec2) -> Vec4 {
    match paint {
        Paint::Solid(color) => Vec4::new(color.r, color.g, color.b, color.a),
        Paint::LinearGradient(gradient) => {
            let beg = gradient.start;
            let end = gradient.end;
            let denom = (end - beg).dot(end - beg).max(1.0e-3);
            let t = (pt - beg).dot(end - beg) / denom;
            if gradient.stops.is_empty() {
                return Vec4::ZERO;
            }
            if t < gradient.stops[0].offset {
                let c = gradient.stops[0].color;
                return Vec4::new(c.r, c.g, c.b, c.a);
            }
            for i in 0..gradient.stops.len() - 1 {
                let curr = gradient.stops[i];
                let next = gradient.stops[i + 1];
                if t >= curr.offset && t < next.offset {
                    let tt = (t - curr.offset) / (next.offset - curr.offset);
                    let c0 = curr.color;
                    let c1 = next.color;
                    return Vec4::new(
                        c0.r * (1.0 - tt) + c1.r * tt,
                        c0.g * (1.0 - tt) + c1.g * tt,
                        c0.b * (1.0 - tt) + c1.b * tt,
                        c0.a * (1.0 - tt) + c1.a * tt,
                    );
                }
            }
            let last = gradient.stops[gradient.stops.len() - 1].color;
            Vec4::new(last.r, last.g, last.b, last.a)
        }
        Paint::RadialGradient(gradient) => {
            let offset = pt - gradient.center;
            let normalized = Vec2::new(
                offset.x / gradient.radius.x,
                offset.y / gradient.radius.y,
            );
            let t = normalized.length();
            if gradient.stops.is_empty() {
                return Vec4::ZERO;
            }
            if t < gradient.stops[0].offset {
                let c = gradient.stops[0].color;
                return Vec4::new(c.r, c.g, c.b, c.a);
            }
            for i in 0..gradient.stops.len() - 1 {
                let curr = gradient.stops[i];
                let next = gradient.stops[i + 1];
                if t >= curr.offset && t < next.offset {
                    let tt = (t - curr.offset) / (next.offset - curr.offset);
                    let c0 = curr.color;
                    let c1 = next.color;
                    return Vec4::new(
                        c0.r * (1.0 - tt) + c1.r * tt,
                        c0.g * (1.0 - tt) + c1.g * tt,
                        c0.b * (1.0 - tt) + c1.b * tt,
                        c0.a * (1.0 - tt) + c1.a * tt,
                    );
                }
            }
            let last = gradient.stops[gradient.stops.len() - 1].color;
            Vec4::new(last.r, last.g, last.b, last.a)
        }
    }
}

fn d_sample_paint(
    paint: &Paint,
    d_paint: &mut DPaint,
    pt: Vec2,
    d_color: Vec4,
    d_translation: &mut Vec2,
) {
    match (paint, d_paint) {
        (Paint::Solid(_), DPaint::Solid(color)) => {
            color.r += d_color.x;
            color.g += d_color.y;
            color.b += d_color.z;
            color.a += d_color.w;
        }
        (Paint::LinearGradient(gradient), DPaint::LinearGradient(d_grad)) => {
            let beg = gradient.start;
            let end = gradient.end;
            let denom = (end - beg).dot(end - beg).max(1.0e-3);
            let t = (pt - beg).dot(end - beg) / denom;
            if gradient.stops.is_empty() {
                return;
            }
            if t < gradient.stops[0].offset {
                if let Some(stop) = d_grad.stops.get_mut(0) {
                    stop.color.r += d_color.x;
                    stop.color.g += d_color.y;
                    stop.color.b += d_color.z;
                    stop.color.a += d_color.w;
                }
                return;
            }
            for i in 0..gradient.stops.len() - 1 {
                let curr = gradient.stops[i];
                let next = gradient.stops[i + 1];
                if t >= curr.offset && t < next.offset {
                    let tt = (t - curr.offset) / (next.offset - curr.offset);
                    let color_curr = curr.color.to_vec4();
                    let color_next = next.color.to_vec4();
                    let d_color_curr = Vec4::new(
                        d_color.x * (1.0 - tt),
                        d_color.y * (1.0 - tt),
                        d_color.z * (1.0 - tt),
                        d_color.w * (1.0 - tt),
                    );
                    let d_color_next = Vec4::new(
                        d_color.x * tt,
                        d_color.y * tt,
                        d_color.z * tt,
                        d_color.w * tt,
                    );
                    if let Some(stop) = d_grad.stops.get_mut(i) {
                        stop.color.r += d_color_curr.x;
                        stop.color.g += d_color_curr.y;
                        stop.color.b += d_color_curr.z;
                        stop.color.a += d_color_curr.w;
                    }
                    if let Some(stop) = d_grad.stops.get_mut(i + 1) {
                        stop.color.r += d_color_next.x;
                        stop.color.g += d_color_next.y;
                        stop.color.b += d_color_next.z;
                        stop.color.a += d_color_next.w;
                    }
                    let diff = Vec4::new(
                        color_next.x - color_curr.x,
                        color_next.y - color_curr.y,
                        color_next.z - color_curr.z,
                        color_next.w - color_curr.w,
                    );
                    let d_tt = dot4(d_color, diff);
                    let denom_offset = next.offset - curr.offset;
                    if denom_offset.abs() > 0.0 {
                        let d_offset_next = -d_tt * tt / denom_offset;
                        let d_offset_curr = d_tt * (tt - 1.0) / denom_offset;
                        if let Some(stop) = d_grad.stops.get_mut(i) {
                            stop.offset += d_offset_curr;
                        }
                        if let Some(stop) = d_grad.stops.get_mut(i + 1) {
                            stop.offset += d_offset_next;
                        }
                    }
                    let d_t = d_tt / denom_offset;
                    let d_beg = Vec2::new(
                        d_t * (-(pt.x - beg.x) - (end.x - beg.x)) / denom,
                        d_t * (-(pt.y - beg.y) - (end.y - beg.y)) / denom,
                    );
                    let d_end = Vec2::new(
                        d_t * (pt.x - beg.x) / denom,
                        d_t * (pt.y - beg.y) / denom,
                    );
                    let d_l = -d_t * t / denom;
                    if (end - beg).dot(end - beg) > 1.0e-3 {
                        let adjust = Vec2::new(beg.x - end.x, beg.y - end.y);
                        d_grad.start += Vec2::new(d_beg.x + 2.0 * d_l * adjust.x, d_beg.y + 2.0 * d_l * adjust.y);
                        d_grad.end += Vec2::new(d_end.x + 2.0 * d_l * (end.x - beg.x), d_end.y + 2.0 * d_l * (end.y - beg.y));
                    } else {
                        d_grad.start += d_beg;
                        d_grad.end += d_end;
                    }
                    *d_translation += d_beg + d_end;
                    return;
                }
            }
            if let Some(stop) = d_grad.stops.last_mut() {
                stop.color.r += d_color.x;
                stop.color.g += d_color.y;
                stop.color.b += d_color.z;
                stop.color.a += d_color.w;
            }
        }
        (Paint::RadialGradient(gradient), DPaint::RadialGradient(d_grad)) => {
            let offset = pt - gradient.center;
            let normalized = Vec2::new(
                offset.x / gradient.radius.x,
                offset.y / gradient.radius.y,
            );
            let t = normalized.length();
            if gradient.stops.is_empty() {
                return;
            }
            if t < gradient.stops[0].offset {
                if let Some(stop) = d_grad.stops.get_mut(0) {
                    stop.color.r += d_color.x;
                    stop.color.g += d_color.y;
                    stop.color.b += d_color.z;
                    stop.color.a += d_color.w;
                }
                return;
            }
            for i in 0..gradient.stops.len() - 1 {
                let curr = gradient.stops[i];
                let next = gradient.stops[i + 1];
                if t >= curr.offset && t < next.offset {
                    let tt = (t - curr.offset) / (next.offset - curr.offset);
                    let color_curr = curr.color.to_vec4();
                    let color_next = next.color.to_vec4();
                    let d_color_curr = Vec4::new(
                        d_color.x * (1.0 - tt),
                        d_color.y * (1.0 - tt),
                        d_color.z * (1.0 - tt),
                        d_color.w * (1.0 - tt),
                    );
                    let d_color_next = Vec4::new(
                        d_color.x * tt,
                        d_color.y * tt,
                        d_color.z * tt,
                        d_color.w * tt,
                    );
                    if let Some(stop) = d_grad.stops.get_mut(i) {
                        stop.color.r += d_color_curr.x;
                        stop.color.g += d_color_curr.y;
                        stop.color.b += d_color_curr.z;
                        stop.color.a += d_color_curr.w;
                    }
                    if let Some(stop) = d_grad.stops.get_mut(i + 1) {
                        stop.color.r += d_color_next.x;
                        stop.color.g += d_color_next.y;
                        stop.color.b += d_color_next.z;
                        stop.color.a += d_color_next.w;
                    }
                    let diff = Vec4::new(
                        color_next.x - color_curr.x,
                        color_next.y - color_curr.y,
                        color_next.z - color_curr.z,
                        color_next.w - color_curr.w,
                    );
                    let d_tt = dot4(d_color, diff);
                    let denom_offset = next.offset - curr.offset;
                    if denom_offset.abs() > 0.0 {
                        let d_offset_next = -d_tt * tt / denom_offset;
                        let d_offset_curr = d_tt * (tt - 1.0) / denom_offset;
                        if let Some(stop) = d_grad.stops.get_mut(i) {
                            stop.offset += d_offset_curr;
                        }
                        if let Some(stop) = d_grad.stops.get_mut(i + 1) {
                            stop.offset += d_offset_next;
                        }
                    }
                    let d_t = d_tt / denom_offset;
                    let d_normalized = d_length(normalized, d_t);
                    let d_offset = Vec2::new(
                        d_normalized.x / gradient.radius.x,
                        d_normalized.y / gradient.radius.y,
                    );
                    let d_radius = Vec2::new(
                        -d_normalized.x * offset.x / (gradient.radius.x * gradient.radius.x),
                        -d_normalized.y * offset.y / (gradient.radius.y * gradient.radius.y),
                    );
                    let d_center = Vec2::new(-d_offset.x, -d_offset.y);
                    d_grad.center += d_center;
                    d_grad.radius += d_radius;
                    *d_translation += d_center;
                    return;
                }
            }
            if let Some(stop) = d_grad.stops.last_mut() {
                stop.color.r += d_color.x;
                stop.color.g += d_color.y;
                stop.color.b += d_color.z;
                stop.color.a += d_color.w;
            }
        }
        _ => {}
    }
}

fn within_distance_edge(
    scene: &crate::scene::Scene,
    bvh: &SceneBvh,
    group_id: usize,
    pt: Vec2,
    edge_query: Option<&mut EdgeQuery>,
) -> bool {
    let hit = within_distance_bvh(scene, bvh, group_id, pt);
    let Some(eq) = edge_query else {
        return hit;
    };
    if eq.shape_group_id != group_id {
        return hit;
    }
    let Some(group) = scene.groups.get(group_id) else {
        return hit;
    };
    let Some(shape) = scene.shapes.get(eq.shape_id) else {
        return hit;
    };
    if shape.stroke_width <= 0.0 {
        return hit;
    }
    let local_pt = group.canvas_to_shape.transform_point(pt);
    if crate::distance::closest::within_distance_shape(shape, local_pt, shape.stroke_width) {
        eq.hit = true;
    }
    hit
}

fn is_inside_edge(
    scene: &crate::scene::Scene,
    bvh: &SceneBvh,
    group_id: usize,
    pt: Vec2,
    edge_query: Option<&mut EdgeQuery>,
) -> bool {
    let inside = is_inside_bvh(scene, bvh, group_id, pt);
    let Some(eq) = edge_query else {
        return inside;
    };
    if eq.shape_group_id != group_id {
        return inside;
    }
    let Some(group) = scene.groups.get(group_id) else {
        return inside;
    };
    let Some(shape) = scene.shapes.get(eq.shape_id) else {
        return inside;
    };
    let local_pt = group.canvas_to_shape.transform_point(pt);
    let shape_pt = transform_point_inverse(shape.transform, local_pt);
    let path_bvh = bvh
        .groups
        .get(group_id)
        .and_then(|gbvh| gbvh.shapes.iter().find(|s| s.shape_index == eq.shape_id))
        .and_then(|shape_bvh| shape_bvh.path_bvh.as_ref());
    let winding = crate::distance::winding::winding_number_shape(shape, path_bvh, shape_pt);
    let hit = match group.fill_rule {
        FillRule::EvenOdd => winding.abs() % 2 == 1,
        FillRule::NonZero => winding != 0,
    };
    if hit {
        eq.hit = true;
    }
    inside
}

fn accumulate_background_grad(
    scene: &crate::scene::Scene,
    grads: &mut SceneGrad,
    pixel_index: usize,
    d_color: Vec4,
) {
    if scene.background_image.is_some() {
        if let Some(d_image) = grads.background_image.as_mut() {
            let base = pixel_index * 4;
            if base + 3 < d_image.len() {
                d_image[base] += d_color.x;
                d_image[base + 1] += d_color.y;
                d_image[base + 2] += d_color.z;
                d_image[base + 3] += d_color.w;
            }
        }
        return;
    }

    let d_bg = Color {
        r: d_color.x,
        g: d_color.y,
        b: d_color.z,
        a: d_color.w,
    };
    grads.background = add_color(grads.background, d_bg);
}

fn accumulate_translation(grads: &mut SceneGrad, translation_index: Option<usize>, delta: Vec2) {
    if let Some(index) = translation_index {
        if let Some(trans) = grads.translation.as_mut() {
            let idx = index * 2;
            trans[idx] += delta.x;
            trans[idx + 1] += delta.y;
        }
    }
}

fn add_color(a: Color, b: Color) -> Color {
    Color {
        r: a.r + b.r,
        g: a.g + b.g,
        b: a.b + b.b,
        a: a.a + b.a,
    }
}

fn dot4(a: Vec4, b: Vec4) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
}

fn smoothstep(d: f32) -> f32 {
    let t = ((d + 1.0) * 0.5).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn d_smoothstep(d: f32, d_ret: f32) -> f32 {
    if d < -1.0 || d > 1.0 {
        return 0.0;
    }
    let t = (d + 1.0) * 0.5;
    let d_t = d_ret * (6.0 * t - 6.0 * t * t);
    d_t * 0.5
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

fn d_compute_distance(
    group: &ShapeGroup,
    shape: &Shape,
    pt: Vec2,
    closest_pt: Vec2,
    path_info: Option<PathInfo>,
    d_dist: f32,
    d_shape: &mut DShape,
    d_group: &mut DShapeGroup,
) -> Vec2 {
    if (pt - closest_pt).length_squared() < 1.0e-10 {
        return Vec2::ZERO;
    }

    let local_pt_group = group.canvas_to_shape.transform_point(pt);
    let local_closest_group = group.canvas_to_shape.transform_point(closest_pt);
    let shape_inv = shape.transform.inverse().unwrap_or(Mat3::identity());
    let local_pt_shape = shape_inv.transform_point(local_pt_group);
    let local_closest_shape = shape_inv.transform_point(local_closest_group);

    let mut d_pt = Vec2::ZERO;
    let mut d_closest = Vec2::ZERO;
    d_distance(closest_pt, pt, d_dist, &mut d_closest, &mut d_pt);

    let mut d_shape_to_canvas = zero_mat3();
    let mut d_local_closest_group = Vec2::ZERO;
    d_xform_pt(
        group.shape_to_canvas,
        local_closest_group,
        d_closest,
        &mut d_shape_to_canvas,
        &mut d_local_closest_group,
    );

    let mut d_shape_transform = zero_mat3();
    let mut d_local_closest_shape = Vec2::ZERO;
    d_xform_pt(
        shape.transform,
        local_closest_shape,
        d_local_closest_group,
        &mut d_shape_transform,
        &mut d_local_closest_shape,
    );

    let mut d_local_pt_shape = Vec2::ZERO;
    d_closest_point(
        shape,
        local_pt_shape,
        d_local_closest_shape,
        path_info,
        d_shape,
        &mut d_local_pt_shape,
    );

    let mut d_shape_inv = zero_mat3();
    let mut d_local_pt_group = Vec2::ZERO;
    d_xform_pt(
        shape_inv,
        local_pt_group,
        d_local_pt_shape,
        &mut d_shape_inv,
        &mut d_local_pt_group,
    );

    let mut d_canvas_to_shape = zero_mat3();
    d_xform_pt(
        group.canvas_to_shape,
        pt,
        d_local_pt_group,
        &mut d_canvas_to_shape,
        &mut d_pt,
    );

    let tc2s = mat3_transpose(group.canvas_to_shape);
    let d_shape_to_canvas_corr =
        mat3_mul(mat3_mul(mat3_scale(tc2s, -1.0), d_canvas_to_shape), tc2s);
    d_group.shape_to_canvas = mat3_add(d_group.shape_to_canvas, d_shape_to_canvas);
    d_group.shape_to_canvas = mat3_add(d_group.shape_to_canvas, d_shape_to_canvas_corr);

    let ts_inv = mat3_transpose(shape_inv);
    let d_shape_transform_corr =
        mat3_mul(mat3_mul(mat3_scale(ts_inv, -1.0), d_shape_inv), ts_inv);
    d_shape.transform = mat3_add(d_shape.transform, d_shape_transform_corr);
    d_shape.transform = mat3_add(d_shape.transform, d_shape_transform);

    d_pt
}

fn d_closest_point(
    shape: &Shape,
    pt: Vec2,
    d_closest_pt: Vec2,
    path_info: Option<PathInfo>,
    d_shape: &mut DShape,
    d_pt: &mut Vec2,
) {
    match &shape.geometry {
        ShapeGeometry::Circle { center, radius } => {
            let v = pt - *center;
            let n = normalize(v);
            let mut d_center = d_closest_pt;
            let d_radius = d_closest_pt.dot(n);
            let d_n = d_closest_pt * *radius;
            let d_v = d_normalize(v, d_n);
            d_center -= d_v;
            *d_pt += d_v;
            if let DShapeGeometry::Circle { center: d_c, radius: d_r } = &mut d_shape.geometry {
                *d_c += d_center;
                *d_r += d_radius;
            }
        }
        ShapeGeometry::Ellipse { center, radius } => {
            let mut d_center = Vec2::ZERO;
            let mut d_radius = Vec2::ZERO;
            d_closest_point_ellipse(*center, *radius, pt, d_closest_pt, &mut d_center, &mut d_radius, d_pt);
            if let DShapeGeometry::Ellipse { center: d_c, radius: d_r } = &mut d_shape.geometry {
                *d_c += d_center;
                *d_r += d_radius;
            }
        }
        ShapeGeometry::Rect { min, max } => {
            let mut d_min = Vec2::ZERO;
            let mut d_max = Vec2::ZERO;
            d_closest_point_rect(*min, *max, pt, d_closest_pt, &mut d_min, &mut d_max, d_pt);
            if let DShapeGeometry::Rect { min: d_pmin, max: d_pmax } = &mut d_shape.geometry {
                *d_pmin += d_min;
                *d_pmax += d_max;
            }
        }
        ShapeGeometry::Path { path } => {
            let Some(info) = path_info else {
                return;
            };
            d_closest_point_path(path, pt, d_closest_pt, info, d_shape, d_pt);
        }
    }
}

fn d_closest_point_path(
    path: &Path,
    pt: Vec2,
    d_closest_pt: Vec2,
    path_info: PathInfo,
    d_shape: &mut DShape,
    d_pt: &mut Vec2,
) {
    let base_point_id = path_info.base_point_id;
    let point_id = path_info.point_id;
    let min_t_root = path_info.t;
    let num_points = path.points.len();
    if num_points == 0 {
        return;
    }

    let get_point = |idx: usize| -> Vec2 {
        if path.is_closed {
            path.points[idx % num_points]
        } else {
            path.points[idx.min(num_points - 1)]
        }
    };

    let mut add_point = |idx: usize, delta: Vec2, d_shape: &mut DShape| {
        if let DShapeGeometry::Path { points, .. } = &mut d_shape.geometry {
            let index = if path.is_closed { idx % num_points } else { idx };
            if let Some(p) = points.get_mut(index) {
                *p += delta;
            }
        }
    };

    match path.num_control_points.get(base_point_id).copied().unwrap_or(0) {
        0 => {
            let i0 = point_id;
            let i1 = (point_id + 1) % num_points;
            let p0 = get_point(i0);
            let p1 = get_point(i1);
            let t = (pt - p0).dot(p1 - p0) / (p1 - p0).dot(p1 - p0);
            if t < 0.0 {
                add_point(i0, d_closest_pt, d_shape);
            } else if t > 1.0 {
                add_point(i1, d_closest_pt, d_shape);
            } else {
                add_point(i0, d_closest_pt * (1.0 - t), d_shape);
                add_point(i1, d_closest_pt * t, d_shape);
            }
        }
        1 => {
            let i0 = point_id;
            let i1 = point_id + 1;
            let i2 = (point_id + 2) % num_points;
            let p0 = get_point(i0);
            let p1 = get_point(i1);
            let p2 = get_point(i2);
            let t = min_t_root;
            let mut d_p0 = Vec2::ZERO;
            let mut d_p1 = Vec2::ZERO;
            let mut d_p2 = Vec2::ZERO;
            if t == 0.0 {
                d_p0 += d_closest_pt;
            } else if t == 1.0 {
                d_p2 += d_closest_pt;
            } else {
                let a = p0 - p1 * 2.0 + p2;
                let b = p1 - p0;
                let A = a.dot(a);
                let B = 3.0 * a.dot(b);
                let C = 2.0 * b.dot(b) + a.dot(p0 - pt);
                let tt = 1.0 - t;
                let d_p = d_closest_pt;
                let d_tt = 2.0 * tt * d_p.dot(p0) + 2.0 * t * d_p.dot(p1);
                let d_t = -d_tt + 2.0 * tt * d_p.dot(p1) + 2.0 * t * d_p.dot(p2);
                d_p0 += d_p * (tt * tt);
                d_p1 += d_p * (2.0 * tt * t);
                d_p2 += d_p * (t * t);
                let poly_deriv_t = 3.0 * A * t * t + 2.0 * B * t + C;
                if poly_deriv_t.abs() > 1.0e-6 {
                    let d_A = -(d_t / poly_deriv_t) * t * t * t;
                    let d_B = -(d_t / poly_deriv_t) * t * t;
                    let d_C = -(d_t / poly_deriv_t) * t;
                    let d_D = -(d_t / poly_deriv_t);

                    d_p0 += a * (2.0 * d_A)
                        + (b - a) * (3.0 * d_B)
                        + (b * -4.0) * d_C
                        + (p0 - pt + a) * d_C
                        + (b - (p0 - pt)) * (2.0 * d_D);
                    d_p1 += a * (-4.0 * d_A)
                        + (a + b * -2.0) * (3.0 * d_B)
                        + (b * 2.0) * d_C
                        + (p0 - pt) * (-2.0 * d_C)
                        + (p0 - pt) * d_D;
                    d_p2 += a * (2.0 * d_A) + b * (3.0 * d_B) + (p0 - pt) * d_C;
                    *d_pt += a * (-d_C) + b * d_D;
                }
            }
            add_point(i0, d_p0, d_shape);
            add_point(i1, d_p1, d_shape);
            add_point(i2, d_p2, d_shape);
        }
        2 => {
            let i0 = point_id;
            let i1 = point_id + 1;
            let i2 = point_id + 2;
            let i3 = (point_id + 3) % num_points;
            let p0 = get_point(i0);
            let p1 = get_point(i1);
            let p2 = get_point(i2);
            let p3 = get_point(i3);
            let t = min_t_root;
            let mut d_p0 = Vec2::ZERO;
            let mut d_p1 = Vec2::ZERO;
            let mut d_p2 = Vec2::ZERO;
            let mut d_p3 = Vec2::ZERO;
            if t == 0.0 {
                d_p0 += d_closest_pt;
            } else if t == 1.0 {
                d_p3 += d_closest_pt;
            } else {
                let a = p0 * -1.0 + p1 * 3.0 + p2 * -3.0 + p3;
                let b = p0 * 3.0 + p1 * -6.0 + p2 * 3.0;
                let c = p0 * -3.0 + p1 * 3.0;
                let A: f32 = 3.0 * a.dot(a);
                if A.abs() < 1.0e-10 {
                    return;
                }
                let B = 5.0 * a.dot(b);
                let C = 4.0 * a.dot(c) + 2.0 * b.dot(b);
                let D = 3.0 * (b.dot(c) + a.dot(p0 - pt));
                let E = c.dot(c) + 2.0 * (p0 - pt).dot(b);
                let F = (p0 - pt).dot(c);
                let B = B / A;
                let C = C / A;
                let D = D / A;
                let E = E / A;
                let F = F / A;

                let tt = 1.0 - t;
                let d_p = d_closest_pt;
                let d_tt = 3.0 * tt * tt * d_p.dot(p0)
                    + 6.0 * tt * t * d_p.dot(p1)
                    + 3.0 * t * t * d_p.dot(p2);
                let d_t = -d_tt
                    + 3.0 * tt * tt * d_p.dot(p1)
                    + 6.0 * tt * t * d_p.dot(p2)
                    + 3.0 * t * t * d_p.dot(p3);
                d_p0 += d_p * (tt * tt * tt);
                d_p1 += d_p * (3.0 * tt * tt * t);
                d_p2 += d_p * (3.0 * tt * t * t);
                d_p3 += d_p * (t * t * t);

                let poly_deriv_t = 5.0 * t * t * t * t
                    + 4.0 * B * t * t * t
                    + 3.0 * C * t * t
                    + 2.0 * D * t
                    + E;
                if poly_deriv_t.abs() > 1.0e-10 {
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

                    d_p0 += a * (3.0 * -1.0 * 2.0 * d_A);
                    d_p1 += a * (3.0 * 3.0 * 2.0 * d_A);
                    d_p2 += a * (3.0 * -3.0 * 2.0 * d_A);
                    d_p3 += a * (3.0 * 1.0 * 2.0 * d_A);
                    d_p0 += (b * -1.0 + a * 3.0) * (5.0 * d_B);
                    d_p1 += (b * 3.0 + a * -6.0) * (5.0 * d_B);
                    d_p2 += (b * -3.0 + a * 3.0) * (5.0 * d_B);
                    d_p3 += b * (5.0 * d_B);
                    d_p0 += (c * -1.0 + a * -3.0) * (4.0 * d_C) + b * (3.0 * 2.0 * d_C);
                    d_p1 += (c * 3.0 + a * 3.0) * (4.0 * d_C) + b * (-6.0 * 2.0 * d_C);
                    d_p2 += c * (-3.0 * d_C * 4.0) + b * (3.0 * 2.0 * d_C);
                    d_p3 += c * (4.0 * d_C);
                    d_p0 += (c * 3.0 + b * -3.0) * (3.0 * d_D) + (a * -1.0 + p0 - pt) * (3.0 * d_D);
                    d_p1 += (c * -6.0 + b * 3.0) * (3.0 * d_D) + (p0 - pt) * (3.0 * d_D);
                    d_p2 += c * (3.0 * 3.0 * d_D) + (p0 - pt) * (-3.0 * d_D);
                    *d_pt += a * (-1.0 * 3.0 * d_D);
                    d_p0 += c * (-3.0 * 2.0 * d_E) + (b + (p0 - pt) * 3.0) * (2.0 * d_E);
                    d_p1 += c * (3.0 * 2.0 * d_E) + (p0 - pt) * (-6.0 * 2.0 * d_E);
                    d_p2 += (p0 - pt) * (3.0 * 2.0 * d_E);
                    *d_pt += b * (-1.0 * 2.0 * d_E);
                    d_p0 += (c * -3.0 + (p0 - pt) * -3.0) * d_F + (c * 1.0) * d_F;
                    d_p1 += (p0 - pt) * (3.0 * d_F);
                    *d_pt += c * (-1.0 * d_F);
                }
            }
            add_point(i0, d_p0, d_shape);
            add_point(i1, d_p1, d_shape);
            add_point(i2, d_p2, d_shape);
            add_point(i3, d_p3, d_shape);
        }
        _ => {}
    }
}

fn d_closest_point_rect(
    min: Vec2,
    max: Vec2,
    pt: Vec2,
    d_closest_pt: Vec2,
    d_min: &mut Vec2,
    d_max: &mut Vec2,
    d_pt: &mut Vec2,
) {
    let dist_to_seg = |p0: Vec2, p1: Vec2| -> f32 {
        let t = (pt - p0).dot(p1 - p0) / (p1 - p0).dot(p1 - p0);
        if t < 0.0 {
            (p0 - pt).length()
        } else if t > 1.0 {
            (p1 - pt).length()
        } else {
            (p0 + (p1 - p0) * t - pt).length()
        }
    };
    let left_top = min;
    let right_top = Vec2::new(max.x, min.y);
    let left_bottom = Vec2::new(min.x, max.y);
    let right_bottom = max;
    let mut min_id = 0;
    let mut min_dist = dist_to_seg(left_top, left_bottom);
    let top_dist = dist_to_seg(left_top, right_top);
    let right_dist = dist_to_seg(right_top, right_bottom);
    let bottom_dist = dist_to_seg(left_bottom, right_bottom);
    if top_dist < min_dist {
        min_dist = top_dist;
        min_id = 1;
    }
    if right_dist < min_dist {
        min_dist = right_dist;
        min_id = 2;
    }
    if bottom_dist < min_dist {
        min_dist = bottom_dist;
        min_id = 3;
    }

    let mut update = |p0: Vec2, p1: Vec2, d_closest_pt: Vec2, d_p0: &mut Vec2, d_p1: &mut Vec2| {
        let t = (pt - p0).dot(p1 - p0) / (p1 - p0).dot(p1 - p0);
        if t < 0.0 {
            *d_p0 += d_closest_pt;
        } else if t > 1.0 {
            *d_p1 += d_closest_pt;
        } else {
            let d_p = d_closest_pt;
            *d_p0 += d_p * (1.0 - t);
            *d_p1 += d_p * t;
            let d_t = d_p.dot(p1 - p0);
            let denom = (p1 - p0).dot(p1 - p0);
            let d_num = d_t / denom;
            let d_den = d_t * (-t) / denom;
            *d_pt += (p1 - p0) * d_num;
            *d_p1 += (pt - p0) * d_num;
            *d_p0 += (p0 - p1 + p0 - pt) * d_num;
            *d_p1 += (p1 - p0) * (2.0 * d_den);
            *d_p0 += (p0 - p1) * (2.0 * d_den);
        }
    };

    let mut d_left_top = Vec2::ZERO;
    let mut d_right_top = Vec2::ZERO;
    let mut d_left_bottom = Vec2::ZERO;
    let mut d_right_bottom = Vec2::ZERO;
    match min_id {
        0 => update(left_top, left_bottom, d_closest_pt, &mut d_left_top, &mut d_left_bottom),
        1 => update(left_top, right_top, d_closest_pt, &mut d_left_top, &mut d_right_top),
        2 => update(right_top, right_bottom, d_closest_pt, &mut d_right_top, &mut d_right_bottom),
        _ => update(left_bottom, right_bottom, d_closest_pt, &mut d_left_bottom, &mut d_right_bottom),
    }
    *d_min += d_left_top;
    d_max.x += d_right_top.x;
    d_min.y += d_right_top.y;
    d_min.x += d_left_bottom.x;
    d_max.y += d_left_bottom.y;
    *d_max += d_right_bottom;
}

fn d_closest_point_ellipse(
    center: Vec2,
    radius: Vec2,
    pt: Vec2,
    d_closest_pt: Vec2,
    d_center: &mut Vec2,
    d_radius: &mut Vec2,
    d_pt: &mut Vec2,
) {
    let rx = radius.x.abs();
    let ry = radius.y.abs();
    let eps = 1.0e-6;
    let local = pt - center;
    if rx < eps && ry < eps {
        *d_center += d_closest_pt;
        return;
    }
    if rx < eps {
        let hit = if local.y >= -ry && local.y <= ry { 1.0 } else { 0.0 };
        *d_center += Vec2::new(d_closest_pt.x, d_closest_pt.y);
        d_pt.y += d_closest_pt.y * hit;
        if local.y > ry {
            d_radius.y += d_closest_pt.y;
        } else if local.y < -ry {
            d_radius.y -= d_closest_pt.y;
        }
        return;
    }
    if ry < eps {
        let hit = if local.x >= -rx && local.x <= rx { 1.0 } else { 0.0 };
        *d_center += Vec2::new(d_closest_pt.x, d_closest_pt.y);
        d_pt.x += d_closest_pt.x * hit;
        if local.x > rx {
            d_radius.x += d_closest_pt.x;
        } else if local.x < -rx {
            d_radius.x -= d_closest_pt.x;
        }
        return;
    }

    let sign_x = if local.x < 0.0 { -1.0 } else { 1.0 };
    let sign_y = if local.y < 0.0 { -1.0 } else { 1.0 };
    let x = local.x.abs();
    let y = local.y.abs();
    let mut t = (y * rx).atan2(x * ry);
    let mut g_t = 0.0;
    for _ in 0..20 {
        let (s, c) = t.sin_cos();
        let g = rx * x * s - ry * y * c + (ry * ry - rx * rx) * s * c;
        g_t = rx * x * c + ry * y * s + (ry * ry - rx * rx) * (c * c - s * s);
        if g_t.abs() < 1.0e-12 {
            break;
        }
        let next = (t - g / g_t).clamp(0.0, core::f32::consts::FRAC_PI_2);
        if (next - t).abs() < 1.0e-6 {
            t = next;
            break;
        }
        t = next;
    }
    let (s, c) = t.sin_cos();
    let mut d_t = 0.0;
    d_radius.x += d_closest_pt.x * sign_x * c;
    d_t += d_closest_pt.x * sign_x * (-rx * s);
    d_radius.y += d_closest_pt.y * sign_y * s;
    d_t += d_closest_pt.y * sign_y * (ry * c);

    let g_a = x * s - 2.0 * rx * s * c;
    let g_b = -y * c + 2.0 * ry * s * c;
    let g_x = rx * s;
    let g_y = -ry * c;
    if g_t.abs() > 1.0e-12 {
        let inv = -d_t / g_t;
        d_radius.x += inv * g_a;
        d_radius.y += inv * g_b;
        let d_x = inv * g_x;
        let d_y = inv * g_y;
        d_pt.x += d_x * sign_x;
        d_pt.y += d_y * sign_y;
        d_center.x -= d_x * sign_x;
        d_center.y -= d_y * sign_y;
    }
}

fn d_distance(a: Vec2, b: Vec2, d_out: f32, d_a: &mut Vec2, d_b: &mut Vec2) {
    let d = d_length(b - a, d_out);
    *d_a -= d;
    *d_b += d;
}

fn d_length(v: Vec2, d_l: f32) -> Vec2 {
    let l_sq = v.dot(v);
    let l = l_sq.sqrt();
    if l == 0.0 {
        return Vec2::ZERO;
    }
    let d_l_sq = 0.5 * d_l / l;
    v * (2.0 * d_l_sq)
}

fn normalize(v: Vec2) -> Vec2 {
    let len = v.length();
    if len == 0.0 {
        Vec2::ZERO
    } else {
        v / len
    }
}

fn d_normalize(v: Vec2, d_n: Vec2) -> Vec2 {
    let l = v.length();
    if l == 0.0 {
        return Vec2::ZERO;
    }
    let n = v / l;
    let mut d_v = d_n / l;
    let d_l = -d_n.dot(n) / l;
    d_v += d_length(v, d_l);
    d_v
}

fn zero_mat3() -> Mat3 {
    Mat3 { m: [[0.0; 3]; 3] }
}

fn mat3_add(a: Mat3, b: Mat3) -> Mat3 {
    let mut out = a;
    for r in 0..3 {
        for c in 0..3 {
            out.m[r][c] += b.m[r][c];
        }
    }
    out
}

fn mat3_scale(m: Mat3, s: f32) -> Mat3 {
    let mut out = m;
    for r in 0..3 {
        for c in 0..3 {
            out.m[r][c] *= s;
        }
    }
    out
}

fn mat3_mul(a: Mat3, b: Mat3) -> Mat3 {
    a.mul(b)
}

fn mat3_transpose(m: Mat3) -> Mat3 {
    let mut out = Mat3 { m: [[0.0; 3]; 3] };
    for r in 0..3 {
        for c in 0..3 {
            out.m[r][c] = m.m[c][r];
        }
    }
    out
}

fn d_xform_pt(m: Mat3, pt: Vec2, d_out: Vec2, d_m: &mut Mat3, d_pt: &mut Vec2) {
    let t0 = m.m[0][0] * pt.x + m.m[0][1] * pt.y + m.m[0][2];
    let t1 = m.m[1][0] * pt.x + m.m[1][1] * pt.y + m.m[1][2];
    let t2 = m.m[2][0] * pt.x + m.m[2][1] * pt.y + m.m[2][2];
    let out = Vec2::new(t0 / t2, t1 / t2);
    let d_t0 = d_out.x / t2;
    let d_t1 = d_out.y / t2;
    let d_t2 = -(d_out.x * out.x + d_out.y * out.y) / t2;
    d_m.m[0][0] += d_t0 * pt.x;
    d_m.m[0][1] += d_t0 * pt.y;
    d_m.m[0][2] += d_t0;
    d_m.m[1][0] += d_t1 * pt.x;
    d_m.m[1][1] += d_t1 * pt.y;
    d_m.m[1][2] += d_t1;
    d_m.m[2][0] += d_t2 * pt.x;
    d_m.m[2][1] += d_t2 * pt.y;
    d_m.m[2][2] += d_t2;
    d_pt.x += d_t0 * m.m[0][0] + d_t1 * m.m[1][0] + d_t2 * m.m[2][0];
    d_pt.y += d_t0 * m.m[0][1] + d_t1 * m.m[1][1] + d_t2 * m.m[2][1];
}

fn transform_point_inverse(m: Mat3, pt: Vec2) -> Vec2 {
    m.inverse()
        .unwrap_or(Mat3::identity())
        .transform_point(pt)
}

#[derive(Clone)]
struct ShapeCdf {
    shape_ids: Vec<usize>,
    group_ids: Vec<usize>,
    cdf: Vec<f32>,
    pmf: Vec<f32>,
}

#[derive(Clone)]
struct PathCdf {
    cdf: Vec<f32>,
    pmf: Vec<f32>,
    point_id_map: Vec<usize>,
    length: f32,
}

fn boundary_sampling(
    scene: &crate::scene::Scene,
    bvh: &SceneBvh,
    samples_x: u32,
    samples_y: u32,
    seed: u32,
    d_render_image: &[f32],
    weight_image: &[f32],
    grads: &mut SceneGrad,
    compute_translation: bool,
) {
    if scene.groups.is_empty() {
        return;
    }

    let shape_lengths = compute_shape_lengths(scene);
    let Some(shape_cdf) = build_shape_cdf(scene, &shape_lengths) else {
        return;
    };
    let path_cdfs = build_path_cdfs(scene, &shape_lengths);

    let width = scene.width as usize;
    let height = scene.height as usize;
    let total_samples = width
        .saturating_mul(height)
        .saturating_mul(samples_x as usize)
        .saturating_mul(samples_y as usize);

    for idx in 0..total_samples {
        let mut rng = Pcg32::new(idx as u64, seed as u64);
        let u = rng.next_f32();
        let t = rng.next_f32();
        let (sample_id, _) = sample_cdf(&shape_cdf.cdf, u);
        let Some(shape_id) = shape_cdf.shape_ids.get(sample_id).copied() else {
            continue;
        };
        let Some(group_id) = shape_cdf.group_ids.get(sample_id).copied() else {
            continue;
        };
        let shape_pmf = shape_cdf.pmf.get(sample_id).copied().unwrap_or(0.0);
        if shape_pmf <= 0.0 {
            continue;
        }
        let shape_length = shape_lengths.get(shape_id).copied().unwrap_or(0.0);
        let path_cdf = path_cdfs.get(shape_id).and_then(|v| v.as_ref());
        let Some((local_pt, normal, boundary_pdf, data)) =
            sample_boundary_point(scene, shape_id, group_id, t, shape_length, path_cdf)
        else {
            continue;
        };
        if boundary_pdf <= 0.0 {
            continue;
        }

        let group = &scene.groups[group_id];
        let shape = &scene.shapes[shape_id];
        let shape_to_canvas = group.shape_to_canvas.mul(shape.transform);
        let mut boundary_pt = shape_to_canvas.transform_point(local_pt);
        let shape_inv = shape.transform.inverse().unwrap_or(Mat3::identity());
        let composite_inv = shape_inv.mul(group.canvas_to_shape);
        let normal_canvas = xform_normal(composite_inv, normal);
        boundary_pt.x /= scene.width as f32;
        boundary_pt.y /= scene.height as f32;

        let sample = BoundarySample {
            pt: boundary_pt,
            local_pt,
            normal: normal_canvas,
            shape_group_id: group_id,
            shape_id,
            t,
            data,
            pdf: shape_pmf * boundary_pdf,
        };
        render_edge_sample(
            scene,
            bvh,
            d_render_image,
            weight_image,
            grads,
            compute_translation,
            sample,
        );
    }
}

fn compute_shape_lengths(scene: &crate::scene::Scene) -> Vec<f32> {
    let mut lengths = vec![0.0f32; scene.shapes.len()];
    for (shape_id, shape) in scene.shapes.iter().enumerate() {
        let length = match &shape.geometry {
            ShapeGeometry::Circle { radius, .. } => {
                2.0 * core::f32::consts::PI * radius.abs()
            }
            ShapeGeometry::Ellipse { radius, .. } => {
                let a = radius.x.abs();
                let b = radius.y.abs();
                if a == 0.0 || b == 0.0 {
                    0.0
                } else {
                    core::f32::consts::PI
                        * (3.0 * (a + b) - ((3.0 * a + b) * (a + 3.0 * b)).sqrt())
                }
            }
            ShapeGeometry::Rect { min, max } => {
                let w = (max.x - min.x).abs();
                let h = (max.y - min.y).abs();
                2.0 * (w + h)
            }
            ShapeGeometry::Path { path } => path_length(path),
        };
        lengths[shape_id] = length;
    }
    lengths
}

fn path_length(path: &Path) -> f32 {
    let total_points = path.points.len();
    if total_points == 0 {
        return 0.0;
    }
    let mut length = 0.0f32;
    let mut point_id = 0usize;
    for &controls in &path.num_control_points {
        match controls {
            0 => {
                let i0 = point_id;
                let i1 = point_id + 1;
                let Some(p0) = path_point(path, i0, total_points) else {
                    break;
                };
                let Some(p1) = path_point(path, i1, total_points) else {
                    break;
                };
                length += (p1 - p0).length();
                point_id += 1;
            }
            1 => {
                let i0 = point_id;
                let i1 = point_id + 1;
                let i2 = point_id + 2;
                let Some(p0) = path_point(path, i0, total_points) else {
                    break;
                };
                let Some(p1) = path_point(path, i1, total_points) else {
                    break;
                };
                let Some(p2) = path_point(path, i2, total_points) else {
                    break;
                };
                let eval = |t: f32| {
                    let tt = 1.0 - t;
                    p0 * (tt * tt) + p1 * (2.0 * tt * t) + p2 * (t * t)
                };
                let v0 = p0;
                let v1 = eval(0.5);
                let v2 = p2;
                length += (v1 - v0).length() + (v2 - v1).length();
                point_id += 2;
            }
            2 => {
                let i0 = point_id;
                let i1 = point_id + 1;
                let i2 = point_id + 2;
                let i3 = point_id + 3;
                let Some(p0) = path_point(path, i0, total_points) else {
                    break;
                };
                let Some(p1) = path_point(path, i1, total_points) else {
                    break;
                };
                let Some(p2) = path_point(path, i2, total_points) else {
                    break;
                };
                let Some(p3) = path_point(path, i3, total_points) else {
                    break;
                };
                let eval = |t: f32| {
                    let tt = 1.0 - t;
                    p0 * (tt * tt * tt)
                        + p1 * (3.0 * tt * tt * t)
                        + p2 * (3.0 * tt * t * t)
                        + p3 * (t * t * t)
                };
                let v0 = p0;
                let v1 = eval(1.0 / 3.0);
                let v2 = eval(2.0 / 3.0);
                let v3 = p3;
                length += (v1 - v0).length() + (v2 - v1).length() + (v3 - v2).length();
                point_id += 3;
            }
            _ => break,
        }
    }
    length
}

fn build_shape_cdf(
    scene: &crate::scene::Scene,
    shape_lengths: &[f32],
) -> Option<ShapeCdf> {
    let mut shape_ids = Vec::new();
    let mut group_ids = Vec::new();
    let mut cdf = Vec::new();
    let mut pmf = Vec::new();
    let mut accum = 0.0f32;
    for (group_id, group) in scene.groups.iter().enumerate() {
        for &shape_id in &group.shape_indices {
            let length = shape_lengths.get(shape_id).copied().unwrap_or(0.0);
            shape_ids.push(shape_id);
            group_ids.push(group_id);
            accum += length;
            cdf.push(accum);
            pmf.push(length);
        }
    }
    if accum <= 0.0 || !accum.is_finite() || cdf.is_empty() {
        return None;
    }
    for value in cdf.iter_mut() {
        *value /= accum;
    }
    for value in pmf.iter_mut() {
        *value /= accum;
    }
    Some(ShapeCdf {
        shape_ids,
        group_ids,
        cdf,
        pmf,
    })
}

fn build_path_cdfs(
    scene: &crate::scene::Scene,
    shape_lengths: &[f32],
) -> Vec<Option<PathCdf>> {
    let mut out = vec![None; scene.shapes.len()];
    for (shape_id, shape) in scene.shapes.iter().enumerate() {
        let ShapeGeometry::Path { path } = &shape.geometry else {
            continue;
        };
        let length = shape_lengths.get(shape_id).copied().unwrap_or(0.0);
        if length <= 0.0 || !length.is_finite() {
            continue;
        }
        let num_base = path.num_control_points.len();
        if num_base == 0 {
            continue;
        }
        let inv_length = 1.0 / length;
        let total_points = path.points.len();
        let mut cdf = Vec::with_capacity(num_base);
        let mut pmf = Vec::with_capacity(num_base);
        let mut point_id_map = Vec::with_capacity(num_base);
        let mut point_id = 0usize;
        for &controls in &path.num_control_points {
            point_id_map.push(point_id);
            let seg_len = match controls {
                0 => {
                    let i0 = point_id;
                    let i1 = point_id + 1;
                    point_id += 1;
                    match (path_point(path, i0, total_points), path_point(path, i1, total_points)) {
                        (Some(p0), Some(p1)) => (p1 - p0).length(),
                        _ => 0.0,
                    }
                }
                1 => {
                    let i0 = point_id;
                    let i1 = point_id + 1;
                    let i2 = point_id + 2;
                    point_id += 2;
                    match (
                        path_point(path, i0, total_points),
                        path_point(path, i1, total_points),
                        path_point(path, i2, total_points),
                    ) {
                        (Some(p0), Some(p1), Some(p2)) => {
                            let eval = |t: f32| {
                                let tt = 1.0 - t;
                                p0 * (tt * tt) + p1 * (2.0 * tt * t) + p2 * (t * t)
                            };
                            let v0 = p0;
                            let v1 = eval(0.5);
                            let v2 = p2;
                            (v1 - v0).length() + (v2 - v1).length()
                        }
                        _ => 0.0,
                    }
                }
                2 => {
                    let i0 = point_id;
                    let i1 = point_id + 1;
                    let i2 = point_id + 2;
                    let i3 = point_id + 3;
                    point_id += 3;
                    match (
                        path_point(path, i0, total_points),
                        path_point(path, i1, total_points),
                        path_point(path, i2, total_points),
                        path_point(path, i3, total_points),
                    ) {
                        (Some(p0), Some(p1), Some(p2), Some(p3)) => {
                            let eval = |t: f32| {
                                let tt = 1.0 - t;
                                p0 * (tt * tt * tt)
                                    + p1 * (3.0 * tt * tt * t)
                                    + p2 * (3.0 * tt * t * t)
                                    + p3 * (t * t * t)
                            };
                            let v0 = p0;
                            let v1 = eval(1.0 / 3.0);
                            let v2 = eval(2.0 / 3.0);
                            let v3 = p3;
                            (v1 - v0).length() + (v2 - v1).length() + (v3 - v2).length()
                        }
                        _ => 0.0,
                    }
                }
                _ => 0.0,
            };
            let seg_norm = seg_len * inv_length;
            pmf.push(seg_norm);
            let accum = seg_norm + cdf.last().copied().unwrap_or(0.0);
            cdf.push(accum);
        }
        out[shape_id] = Some(PathCdf {
            cdf,
            pmf,
            point_id_map,
            length,
        });
    }
    out
}

fn sample_cdf(cdf: &[f32], u: f32) -> (usize, f32) {
    if cdf.is_empty() {
        return (0, 0.0);
    }
    let mut idx = 0usize;
    while idx < cdf.len() && u > cdf[idx] {
        idx += 1;
    }
    if idx >= cdf.len() {
        idx = cdf.len() - 1;
    }
    let prev = if idx == 0 { 0.0 } else { cdf[idx - 1] };
    let denom = (cdf[idx] - prev).max(1.0e-6);
    let t = ((u - prev) / denom).clamp(0.0, 1.0);
    (idx, t)
}

fn sample_boundary_point(
    scene: &crate::scene::Scene,
    shape_id: usize,
    group_id: usize,
    t: f32,
    shape_length: f32,
    path_cdf: Option<&PathCdf>,
) -> Option<(Vec2, Vec2, f32, BoundaryData)> {
    let group = scene.groups.get(group_id)?;
    let shape = scene.shapes.get(shape_id)?;

    if group.fill.is_none() && group.stroke.is_none() {
        return None;
    }

    let mut pdf = 1.0f32;
    let mut local_t = t;
    let mut stroke_perturb = false;
    if group.fill.is_some() && group.stroke.is_some() {
        if local_t < 0.5 {
            stroke_perturb = false;
            local_t *= 2.0;
            pdf = 0.5;
        } else {
            stroke_perturb = true;
            local_t = 2.0 * (local_t - 0.5);
            pdf = 0.5;
        }
    } else if group.stroke.is_some() {
        stroke_perturb = true;
    }
    let mut stroke_direction = 0.0f32;
    if stroke_perturb {
        if local_t < 0.5 {
            stroke_direction = -1.0;
            local_t *= 2.0;
            pdf *= 0.5;
        } else {
            stroke_direction = 1.0;
            local_t = 2.0 * (local_t - 0.5);
            pdf *= 0.5;
        }
    }

    let mut data = BoundaryData {
        path: PathBoundaryData {
            base_point_id: 0,
            point_id: 0,
            t: 0.0,
        },
        is_stroke: stroke_perturb,
    };

    let mut normal = Vec2::ZERO;
    let pt = match &shape.geometry {
        ShapeGeometry::Circle { center, radius } => {
            let r = radius.abs();
            if r <= 0.0 {
                return None;
            }
            let angle = 2.0 * core::f32::consts::PI * local_t;
            let offset = Vec2::new(r * angle.cos(), r * angle.sin());
            normal = normalize(offset);
            pdf /= 2.0 * core::f32::consts::PI * r;
            let mut out = *center + offset;
            if stroke_direction != 0.0 {
                out += normal * (stroke_direction * shape.stroke_width);
                if stroke_direction < 0.0 {
                    normal = normal * -1.0;
                }
            }
            out
        }
        ShapeGeometry::Ellipse { center, radius } => {
            let rx = radius.x.abs();
            let ry = radius.y.abs();
            if rx <= 0.0 || ry <= 0.0 {
                return None;
            }
            let angle = 2.0 * core::f32::consts::PI * local_t;
            let (s, c) = angle.sin_cos();
            let offset = Vec2::new(rx * c, ry * s);
            let dxdt = -rx * s * 2.0 * core::f32::consts::PI;
            let dydt = ry * c * 2.0 * core::f32::consts::PI;
            normal = normalize(Vec2::new(dydt, -dxdt));
            pdf /= (dxdt * dxdt + dydt * dydt).sqrt();
            let mut out = *center + offset;
            if stroke_direction != 0.0 {
                out += normal * (stroke_direction * shape.stroke_width);
                if stroke_direction < 0.0 {
                    normal = normal * -1.0;
                }
            }
            out
        }
        ShapeGeometry::Rect { min, max } => {
            let w = max.x - min.x;
            let h = max.y - min.y;
            if w == 0.0 && h == 0.0 {
                return None;
            }
            pdf /= 2.0 * (w + h);
            let mut out = Vec2::ZERO;
            if local_t <= w / (w + h) {
                local_t *= (w + h) / w;
                if local_t < 0.5 {
                    normal = Vec2::new(0.0, -1.0);
                    out = *min + Vec2::new(2.0 * local_t * (max.x - min.x), 0.0);
                } else {
                    normal = Vec2::new(0.0, 1.0);
                    out = Vec2::new(min.x, max.y)
                        + Vec2::new(2.0 * (local_t - 0.5) * (max.x - min.x), 0.0);
                }
            } else {
                local_t = (local_t - w / (w + h)) * (w + h) / h;
                if local_t < 0.5 {
                    normal = Vec2::new(-1.0, 0.0);
                    out = *min + Vec2::new(0.0, 2.0 * local_t * (max.y - min.y));
                } else {
                    normal = Vec2::new(1.0, 0.0);
                    out = Vec2::new(max.x, min.y)
                        + Vec2::new(0.0, 2.0 * (local_t - 0.5) * (max.y - min.y));
                }
            }
            if stroke_direction != 0.0 {
                out += normal * (stroke_direction * shape.stroke_width);
                if stroke_direction < 0.0 {
                    normal = normal * -1.0;
                }
            }
            out
        }
        ShapeGeometry::Path { path } => {
            let path_cdf = path_cdf?;
            if path_cdf.length <= 0.0 {
                return None;
            }
            sample_boundary_path(
                path,
                path_cdf,
                shape_length,
                local_t,
                &mut normal,
                &mut pdf,
                &mut data,
                stroke_direction,
                shape.stroke_width,
            )
        }
    };

    if !pdf.is_finite() || pdf <= 0.0 {
        return None;
    }
    Some((pt, normal, pdf, data))
}

fn sample_boundary_path(
    path: &Path,
    path_cdf: &PathCdf,
    path_length: f32,
    mut t: f32,
    normal: &mut Vec2,
    pdf: &mut f32,
    data: &mut BoundaryData,
    stroke_direction: f32,
    stroke_radius: f32,
) -> Vec2 {
    let num_points = path.points.len();
    let num_base = path.num_control_points.len();
    if num_points == 0 || num_base == 0 {
        *pdf = 0.0;
        return Vec2::ZERO;
    }
    if stroke_direction != 0.0 && !path.is_closed {
        let mut cap_length = 0.0f32;
        if let Some(thickness) = &path.thickness {
            if !thickness.is_empty() {
                let r0 = thickness[0];
                let r1 = thickness[thickness.len() - 1];
                cap_length = core::f32::consts::PI * (r0 + r1);
            }
        } else {
            cap_length = 2.0 * core::f32::consts::PI * stroke_radius;
        }
        let denom = cap_length + path_length;
        if denom > 0.0 {
            let cap_prob = cap_length / denom;
            if t < cap_prob {
                t /= cap_prob;
                *pdf *= cap_prob;
                let mut r0 = stroke_radius;
                let mut r1 = stroke_radius;
                if let Some(thickness) = &path.thickness {
                    if !thickness.is_empty() {
                        r0 = thickness[0];
                        r1 = thickness[thickness.len() - 1];
                    }
                }
                let angle = 2.0 * core::f32::consts::PI * t;
                if stroke_direction < 0.0 {
                    let p0 = path.points[0];
                    let offset = Vec2::new(r0 * angle.cos(), r0 * angle.sin());
                    *normal = normalize(offset);
                    *pdf /= 2.0 * core::f32::consts::PI * r0;
                    data.path.base_point_id = 0;
                    data.path.point_id = 0;
                    data.path.t = 0.0;
                    return p0 + offset;
                } else {
                    let p0 = path.points[num_points - 1];
                    let offset = Vec2::new(r1 * angle.cos(), r1 * angle.sin());
                    *normal = normalize(offset);
                    *pdf /= 2.0 * core::f32::consts::PI * r1;
                    data.path.base_point_id = num_base - 1;
                    let controls = path.num_control_points[data.path.base_point_id] as usize;
                    data.path.point_id = num_points.saturating_sub(2 + controls);
                    data.path.t = 1.0;
                    return p0 + offset;
                }
            } else {
                t = (t - cap_prob) / (1.0 - cap_prob);
                *pdf *= 1.0 - cap_prob;
            }
        }
    }

    let (sample_id, local_t) = sample_cdf(&path_cdf.cdf, t);
    if sample_id >= path_cdf.point_id_map.len() {
        *pdf = 0.0;
        return Vec2::ZERO;
    }
    let point_id = path_cdf.point_id_map[sample_id];
    data.path.base_point_id = sample_id;
    data.path.point_id = point_id;
    data.path.t = local_t;
    if local_t < -1.0e-3 || local_t > 1.0 + 1.0e-3 {
        *pdf = 0.0;
        return Vec2::ZERO;
    }

    let next_index = |idx: usize| -> usize {
        if path.is_closed {
            idx % num_points
        } else {
            idx.min(num_points.saturating_sub(1))
        }
    };

    match path.num_control_points.get(sample_id).copied().unwrap_or(0) {
        0 => {
            let i0 = point_id;
            let i1 = next_index(point_id + 1);
            let Some((p0, r0)) = path_point_radius(path, i0, num_points, stroke_radius, 1.0) else {
                *pdf = 0.0;
                return Vec2::ZERO;
            };
            let Some((p1, r1)) = path_point_radius(path, i1, num_points, stroke_radius, 1.0) else {
                *pdf = 0.0;
                return Vec2::ZERO;
            };
            let tangent = p1 - p0;
            let tan_len = tangent.length();
            if tan_len == 0.0 {
                *pdf = 0.0;
                return Vec2::ZERO;
            }
            *normal = Vec2::new(-tangent.y, tangent.x) / tan_len;
            *pdf *= path_cdf.pmf.get(sample_id).copied().unwrap_or(0.0) / tan_len;
            let mut out = p0 + tangent * local_t;
            if stroke_direction != 0.0 {
                let r = r0 + local_t * (r1 - r0);
                out += *normal * (stroke_direction * r);
                if stroke_direction < 0.0 {
                    *normal = *normal * -1.0;
                }
            }
            out
        }
        1 => {
            let i0 = point_id;
            let i1 = point_id + 1;
            let i2 = next_index(point_id + 2);
            let Some((p0, r0)) = path_point_radius(path, i0, num_points, stroke_radius, 1.0) else {
                *pdf = 0.0;
                return Vec2::ZERO;
            };
            let Some((p1, r1)) = path_point_radius(path, i1, num_points, stroke_radius, 1.0) else {
                *pdf = 0.0;
                return Vec2::ZERO;
            };
            let Some((p2, r2)) = path_point_radius(path, i2, num_points, stroke_radius, 1.0) else {
                *pdf = 0.0;
                return Vec2::ZERO;
            };
            let tt = 1.0 - local_t;
            let eval = p0 * (tt * tt) + p1 * (2.0 * tt * local_t) + p2 * (local_t * local_t);
            let tangent = (p1 - p0) * (2.0 * tt) + (p2 - p1) * (2.0 * local_t);
            let tan_len = tangent.length();
            if tan_len == 0.0 {
                *pdf = 0.0;
                return Vec2::ZERO;
            }
            *normal = Vec2::new(-tangent.y, tangent.x) / tan_len;
            *pdf *= path_cdf.pmf.get(sample_id).copied().unwrap_or(0.0) / tan_len;
            let mut out = eval;
            if stroke_direction != 0.0 {
                let r = r0 * (tt * tt) + r1 * (2.0 * tt * local_t) + r2 * (local_t * local_t);
                out += *normal * (stroke_direction * r);
                if stroke_direction < 0.0 {
                    *normal = *normal * -1.0;
                }
            }
            out
        }
        2 => {
            let i0 = point_id;
            let i1 = point_id + 1;
            let i2 = point_id + 2;
            let i3 = next_index(point_id + 3);
            let Some((p0, r0)) = path_point_radius(path, i0, num_points, stroke_radius, 1.0) else {
                *pdf = 0.0;
                return Vec2::ZERO;
            };
            let Some((p1, r1)) = path_point_radius(path, i1, num_points, stroke_radius, 1.0) else {
                *pdf = 0.0;
                return Vec2::ZERO;
            };
            let Some((p2, r2)) = path_point_radius(path, i2, num_points, stroke_radius, 1.0) else {
                *pdf = 0.0;
                return Vec2::ZERO;
            };
            let Some((p3, r3)) = path_point_radius(path, i3, num_points, stroke_radius, 1.0) else {
                *pdf = 0.0;
                return Vec2::ZERO;
            };
            let tt = 1.0 - local_t;
            let eval = p0 * (tt * tt * tt)
                + p1 * (3.0 * tt * tt * local_t)
                + p2 * (3.0 * tt * local_t * local_t)
                + p3 * (local_t * local_t * local_t);
            let tangent = (p1 - p0) * (3.0 * tt * tt)
                + (p2 - p1) * (6.0 * tt * local_t)
                + (p3 - p2) * (3.0 * local_t * local_t);
            let tan_len = tangent.length();
            if tan_len == 0.0 {
                *pdf = 0.0;
                return Vec2::ZERO;
            }
            *normal = Vec2::new(-tangent.y, tangent.x) / tan_len;
            *pdf *= path_cdf.pmf.get(sample_id).copied().unwrap_or(0.0) / tan_len;
            let mut out = eval;
            if stroke_direction != 0.0 {
                let r = r0 * (tt * tt * tt)
                    + r1 * (3.0 * tt * tt * local_t)
                    + r2 * (3.0 * tt * local_t * local_t)
                    + r3 * (local_t * local_t * local_t);
                out += *normal * (stroke_direction * r);
                if stroke_direction < 0.0 {
                    *normal = *normal * -1.0;
                }
            }
            out
        }
        _ => {
            *pdf = 0.0;
            Vec2::ZERO
        }
    }
}

fn render_edge_sample(
    scene: &crate::scene::Scene,
    bvh: &SceneBvh,
    d_render_image: &[f32],
    weight_image: &[f32],
    grads: &mut SceneGrad,
    compute_translation: bool,
    sample: BoundarySample,
) {
    let width = scene.width as i32;
    let height = scene.height as i32;
    if width <= 0 || height <= 0 {
        return;
    }
    if sample.pdf <= 0.0 {
        return;
    }
    let bx = (sample.pt.x * width as f32) as i32;
    let by = (sample.pt.y * height as f32) as i32;
    if bx < 0 || bx >= width || by < 0 || by >= height {
        return;
    }
    let pixel_index = by as usize * width as usize + bx as usize;
    let background = Some(sample_background(scene, pixel_index));

    let mut inside_query = EdgeQuery {
        shape_group_id: sample.shape_group_id,
        shape_id: sample.shape_id,
        hit: false,
    };
    let mut outside_query = EdgeQuery {
        shape_group_id: sample.shape_group_id,
        shape_id: sample.shape_id,
        hit: false,
    };

    let mut normal = sample.normal;
    let mut color_inside = sample_color(
        scene,
        bvh,
        sample.pt - normal * 1.0e-4,
        background,
        None,
        Some(&mut inside_query),
        grads,
        None,
        pixel_index,
    );
    let mut color_outside = sample_color(
        scene,
        bvh,
        sample.pt + normal * 1.0e-4,
        background,
        None,
        Some(&mut outside_query),
        grads,
        None,
        pixel_index,
    );

    if !inside_query.hit && !outside_query.hit {
        return;
    }
    if !inside_query.hit {
        normal = normal * -1.0;
        core::mem::swap(&mut color_inside, &mut color_outside);
    }

    let sboundary_pt = Vec2::new(sample.pt.x * width as f32, sample.pt.y * height as f32);
    let mut d_color = gather_d_color(
        scene.filter.filter_type,
        scene.filter.radius,
        d_render_image,
        weight_image,
        width,
        height,
        sboundary_pt,
    );
    let norm = (scene.width as f32) * (scene.height as f32);
    if norm > 0.0 {
        d_color.x /= norm;
        d_color.y /= norm;
        d_color.z /= norm;
        d_color.w /= norm;
    }

    let diff = Vec4::new(
        color_inside.x - color_outside.x,
        color_inside.y - color_outside.y,
        color_inside.z - color_outside.z,
        color_inside.w - color_outside.w,
    );
    let contrib = dot4(diff, d_color) / sample.pdf;
    if !contrib.is_finite() {
        return;
    }

    let shape = &scene.shapes[sample.shape_id];
    let group = &scene.groups[sample.shape_group_id];
    let d_shape = &mut grads.shapes[sample.shape_id];
    let d_group = &mut grads.shape_groups[sample.shape_group_id];
    accumulate_boundary_gradient(
        shape,
        contrib,
        sample.t,
        normal,
        sample.data,
        d_shape,
        group.shape_to_canvas,
        sample.local_pt,
        d_group,
    );

    if compute_translation {
        if let Some(trans) = grads.translation.as_mut() {
            let idx = pixel_index * 2;
            if idx + 1 < trans.len() {
                trans[idx] += normal.x * contrib;
                trans[idx + 1] += normal.y * contrib;
            }
        }
    }
}

fn accumulate_boundary_gradient(
    shape: &Shape,
    contrib: f32,
    t: f32,
    normal: Vec2,
    boundary_data: BoundaryData,
    d_shape: &mut DShape,
    group_shape_to_canvas: Mat3,
    local_boundary_pt: Vec2,
    d_group: &mut DShapeGroup,
) {
    if !contrib.is_finite() {
        return;
    }

    if boundary_data.is_stroke {
        let has_thickness = matches!(
            shape.geometry,
            ShapeGeometry::Path { ref path } if path.thickness.is_some()
        );
        if has_thickness {
            if let ShapeGeometry::Path { path } = &shape.geometry {
                let base_point_id = boundary_data.path.base_point_id;
                let point_id = boundary_data.path.point_id;
                let t = boundary_data.path.t;
                if let DShapeGeometry::Path { thickness: Some(d_thickness), .. } =
                    &mut d_shape.geometry
                {
                    match path.num_control_points.get(base_point_id).copied().unwrap_or(0) {
                        0 => {
                            let i0 = point_id;
                            let i1 = if path.is_closed {
                                (point_id + 1) % path.points.len()
                            } else {
                                point_id + 1
                            };
                            if let Some(v0) = d_thickness.get_mut(i0) {
                                *v0 += (1.0 - t) * contrib;
                            }
                            if let Some(v1) = d_thickness.get_mut(i1) {
                                *v1 += t * contrib;
                            }
                        }
                        1 => {
                            let i0 = point_id;
                            let i1 = point_id + 1;
                            let i2 = if path.is_closed {
                                (point_id + 2) % path.points.len()
                            } else {
                                point_id + 2
                            };
                            let tt = 1.0 - t;
                            if let Some(v0) = d_thickness.get_mut(i0) {
                                *v0 += tt * tt * contrib;
                            }
                            if let Some(v1) = d_thickness.get_mut(i1) {
                                *v1 += 2.0 * tt * t * contrib;
                            }
                            if let Some(v2) = d_thickness.get_mut(i2) {
                                *v2 += t * t * contrib;
                            }
                        }
                        2 => {
                            let i0 = point_id;
                            let i1 = point_id + 1;
                            let i2 = point_id + 2;
                            let i3 = if path.is_closed {
                                (point_id + 3) % path.points.len()
                            } else {
                                point_id + 3
                            };
                            let tt = 1.0 - t;
                            if let Some(v0) = d_thickness.get_mut(i0) {
                                *v0 += tt * tt * tt * contrib;
                            }
                            if let Some(v1) = d_thickness.get_mut(i1) {
                                *v1 += 3.0 * tt * tt * t * contrib;
                            }
                            if let Some(v2) = d_thickness.get_mut(i2) {
                                *v2 += 3.0 * tt * t * t * contrib;
                            }
                            if let Some(v3) = d_thickness.get_mut(i3) {
                                *v3 += t * t * t * contrib;
                            }
                        }
                        _ => {}
                    }
                }
            }
        } else {
            d_shape.stroke_width += contrib;
        }
    }

    match &shape.geometry {
        ShapeGeometry::Circle { .. } => {
            if let DShapeGeometry::Circle { center, radius } = &mut d_shape.geometry {
                *center += normal * contrib;
                *radius += contrib;
            }
        }
        ShapeGeometry::Ellipse { .. } => {
            if let DShapeGeometry::Ellipse { center, radius } = &mut d_shape.geometry {
                *center += normal * contrib;
                let angle = 2.0 * core::f32::consts::PI * t;
                radius.x += angle.cos() * normal.x * contrib;
                radius.y += angle.sin() * normal.y * contrib;
            }
        }
        ShapeGeometry::Path { path } => {
            let base_point_id = boundary_data.path.base_point_id;
            let point_id = boundary_data.path.point_id;
            let t = boundary_data.path.t;
            let num_points = path.points.len();
            if let DShapeGeometry::Path { points, .. } = &mut d_shape.geometry {
                let mut add_point = |idx: usize, weight: f32, points: &mut [Vec2]| {
                    let index = if path.is_closed {
                        idx % num_points
                    } else {
                        idx
                    };
                    if let Some(p) = points.get_mut(index) {
                        *p += normal * (weight * contrib);
                    }
                };
                match path.num_control_points.get(base_point_id).copied().unwrap_or(0) {
                    0 => {
                        add_point(point_id, 1.0 - t, points);
                        add_point(point_id + 1, t, points);
                    }
                    1 => {
                        let tt = 1.0 - t;
                        add_point(point_id, tt * tt, points);
                        add_point(point_id + 1, 2.0 * tt * t, points);
                        add_point(point_id + 2, t * t, points);
                    }
                    2 => {
                        let tt = 1.0 - t;
                        add_point(point_id, tt * tt * tt, points);
                        add_point(point_id + 1, 3.0 * tt * tt * t, points);
                        add_point(point_id + 2, 3.0 * tt * t * t, points);
                        add_point(point_id + 3, t * t * t, points);
                    }
                    _ => {}
                }
            }
        }
        ShapeGeometry::Rect { .. } => {
            if let DShapeGeometry::Rect { min, max } = &mut d_shape.geometry {
                if normal == Vec2::new(-1.0, 0.0) {
                    min.x += -contrib;
                } else if normal == Vec2::new(1.0, 0.0) {
                    max.x += contrib;
                } else if normal == Vec2::new(0.0, -1.0) {
                    min.y += -contrib;
                } else if normal == Vec2::new(0.0, 1.0) {
                    max.y += contrib;
                }
            }
        }
    }

    let shape_to_canvas = group_shape_to_canvas.mul(shape.transform);
    let mut d_shape_to_canvas = zero_mat3();
    let mut d_local_boundary_pt = Vec2::ZERO;
    d_xform_pt(
        shape_to_canvas,
        local_boundary_pt,
        normal * contrib,
        &mut d_shape_to_canvas,
        &mut d_local_boundary_pt,
    );

    let d_group_mat = mat3_mul(d_shape_to_canvas, mat3_transpose(shape.transform));
    d_group.shape_to_canvas = mat3_add(d_group.shape_to_canvas, d_group_mat);

    let d_shape_mat = mat3_mul(mat3_transpose(group_shape_to_canvas), d_shape_to_canvas);
    d_shape.transform = mat3_add(d_shape.transform, d_shape_mat);
}

fn xform_normal(m_inv: Mat3, n: Vec2) -> Vec2 {
    let x = m_inv.m[0][0] * n.x + m_inv.m[1][0] * n.y;
    let y = m_inv.m[0][1] * n.x + m_inv.m[1][1] * n.y;
    normalize(Vec2::new(x, y))
}
