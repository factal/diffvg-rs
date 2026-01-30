//! Misc renderer utilities shared by GPU and CPU paths.

use crate::distance::{compute_distance_bvh, is_inside_bvh, DistanceOptions, SceneBvh};
use crate::math::Vec2;
use crate::scene::Scene;

pub(crate) fn ensure_nonempty(mut data: Vec<f32>, filler: f32) -> Vec<f32> {
    if data.is_empty() {
        data.push(filler);
    }
    data
}

pub(crate) fn ensure_nonempty_u32(mut data: Vec<u32>, filler: u32) -> Vec<u32> {
    if data.is_empty() {
        data.push(filler);
    }
    data
}

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
