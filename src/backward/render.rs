use crate::distance::{DistanceOptions, SceneBvh};
use crate::grad::SceneGrad;
use crate::math::{Vec2, Vec4};
use crate::renderer::rng::Pcg32;
use crate::scene::Scene;
use crate::{RenderError, RenderOptions};

use super::background::{finalize_background_gradients, sample_background};
use super::boundary::boundary_sampling;
use super::filters::{accumulate_filter_gradient, build_weight_image, gather_d_color};
use super::sampling::{sample_color, sample_color_prefiltered, sample_distance};

/// Options controlling which auxiliary gradients are computed in the backward pass.
///
/// Translation gradients allocate a per-pixel buffer in `SceneGrad` (width * height * 2)
/// and accumulate contributions from render/SDF sampling. Disable them to save memory
/// and skip related work when you do not need translation adjoints.
#[derive(Debug, Copy, Clone)]
pub struct BackwardOptions {
    /// Accumulate per-pixel translation gradients into `SceneGrad.translation`.
    pub compute_translation: bool,
}

impl Default for BackwardOptions {
    /// Default backward options with translation gradients disabled.
    fn default() -> Self {
        Self {
            compute_translation: false,
        }
    }
}

/// Run the backward pass for render and/or SDF targets, producing scene gradients.
pub fn render_backward(
    scene: &Scene,
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
    if let Some(background) = scene.background_image.as_ref() {
        let expected_len = (scene.width as usize)
            .checked_mul(scene.height as usize)
            .and_then(|v| v.checked_mul(4))
            .ok_or(RenderError::InvalidScene("image size overflow"))?;
        if background.len() != expected_len {
            return Err(RenderError::InvalidScene("background image size mismatch"));
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

/// Backward pass for SDF values evaluated at arbitrary positions.
pub fn render_backward_positions(
    scene: &Scene,
    options: RenderOptions,
    backward_options: BackwardOptions,
    eval_positions: &[Vec2],
    d_sdf_image: Option<&[f32]>,
) -> Result<SceneGrad, RenderError> {
    if let Some(d_sdf) = d_sdf_image {
        if d_sdf.len() != eval_positions.len() {
            return Err(RenderError::InvalidScene(
                "d_sdf_image size mismatch for eval_positions",
            ));
        }
    }

    let mut grads = SceneGrad::zeros_from_scene(scene, false, backward_options.compute_translation);
    if eval_positions.is_empty() {
        return Ok(grads);
    }

    let Some(d_sdf) = d_sdf_image else {
        return Ok(grads);
    };

    let bvh = SceneBvh::new(scene);
    let dist_options = DistanceOptions {
        path_tolerance: options.path_tolerance,
    };

    for (idx, &pt) in eval_positions.iter().enumerate() {
        let d_dist = d_sdf[idx];
        if d_dist == 0.0 {
            continue;
        }
        let translation_index = if backward_options.compute_translation {
            translation_index_for_point(scene, pt)
        } else {
            None
        };
        sample_distance(
            scene,
            &bvh,
            pt,
            1.0,
            d_dist,
            &mut grads,
            translation_index,
            dist_options,
        );
    }

    Ok(grads)
}

/// Map a point to a pixel index for translation gradients if the point is in-bounds.
fn translation_index_for_point(scene: &Scene, pt: Vec2) -> Option<usize> {
    let x = pt.x as i32;
    let y = pt.y as i32;
    if x < 0 || y < 0 {
        return None;
    }
    let width = scene.width as i32;
    let height = scene.height as i32;
    if x >= width || y >= height {
        return None;
    }
    let idx = (y as usize) * (scene.width as usize) + (x as usize);
    Some(idx)
}
