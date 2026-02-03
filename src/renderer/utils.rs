//! Misc renderer utilities shared by GPU and CPU paths.

use crate::distance::{compute_distance_bvh, is_inside_bvh, DistanceOptions, SceneBvh};
use crate::math::Vec2;
use crate::scene::Scene;
use crate::renderer::types::RenderError;

/// Ensure a float buffer is non-empty by inserting a single filler value.
pub(crate) fn ensure_nonempty(mut data: Vec<f32>, filler: f32) -> Vec<f32> {
    if data.is_empty() {
        data.push(filler);
    }
    data
}

/// Ensure a u32 buffer is non-empty by inserting a single filler value.
pub(crate) fn ensure_nonempty_u32(mut data: Vec<u32>, filler: u32) -> Vec<u32> {
    if data.is_empty() {
        data.push(filler);
    }
    data
}

/// Validate optional backward gradient image lengths against the pixel count.
pub(crate) fn validate_backward_image_lengths(
    pixel_count: usize,
    d_render_image: Option<&[f32]>,
    d_sdf_image: Option<&[f32]>,
) -> Result<(), RenderError> {
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
    Ok(())
}

/// Compute a signed distance at `pt` using the BVH-accelerated scene query.
pub(crate) fn sample_distance(scene: &Scene, bvh: &SceneBvh, pt: Vec2, options: DistanceOptions) -> f32 {
    let mut best_dist = f32::INFINITY;
    let mut best_group = None;
    for group_id in (0..scene.groups.len()).rev() {
        if let Some(hit) = compute_distance_bvh(scene, bvh, group_id, pt, best_dist, options) {
            if hit.distance < best_dist {
                best_dist = hit.distance;
                best_group = Some(group_id);
            }
        }
    }
    let Some(group_id) = best_group else {
        return 0.0;
    };
    let mut dist = best_dist;
    if scene.groups[group_id].fill.is_some() && is_inside_bvh(scene, bvh, group_id, pt) {
        dist = -dist;
    }
    dist
}
