//! Public API for CPU distance and winding queries.

use crate::math::Vec2;
use crate::scene::{FillRule, Scene};

use super::bvh::{SceneBvh, BVH_NONE};
use super::closest::{
    closest_point_path_bvh, closest_point_shape, within_distance_path_bvh, within_distance_shape,
};
use super::types::{ClosestPoint, DistanceOptions};
use super::utils::transform_point_inverse;
use super::winding::winding_number_shape;

/// Check if any stroked shape in the group is within its stroke radius of `pt`.
pub fn within_distance(scene: &Scene, group_index: usize, pt: Vec2) -> bool {
    let Some(group) = scene.groups.get(group_index) else {
        return false;
    };
    let local_pt = group.canvas_to_shape.transform_point(pt);
    for &shape_index in &group.shape_indices {
        let Some(shape) = scene.shapes.get(shape_index) else {
            continue;
        };
        if shape.stroke_width <= 0.0 {
            continue;
        }
        let stroke_width = shape.stroke_width;
        if within_distance_shape(shape, local_pt, stroke_width) {
            return true;
        }
    }
    false
}

/// Compute the closest point to `pt` within a given group.
pub fn compute_distance(
    scene: &Scene,
    group_index: usize,
    pt: Vec2,
    max_radius: f32,
    options: DistanceOptions,
) -> Option<ClosestPoint> {
    let Some(group) = scene.groups.get(group_index) else {
        return None;
    };
    let local_pt = group.canvas_to_shape.transform_point(pt);
    let mut best: Option<ClosestPoint> = None;

    for &shape_index in &group.shape_indices {
        let Some(shape) = scene.shapes.get(shape_index) else {
            continue;
        };
        let local = closest_point_shape(shape, local_pt, options.path_tolerance);
        let Some((local_closest, path_info)) = local else {
            continue;
        };
        let closest = group.shape_to_canvas.transform_point(local_closest);
        let distance = (closest - pt).length();
        if distance <= max_radius {
            let update = match best {
                None => true,
                Some(prev) => distance < prev.distance,
            };
            if update {
                best = Some(ClosestPoint {
                    shape_index,
                    point: closest,
                    distance,
                    path: path_info,
                });
            }
        }
    }

    best
}

/// BVH-accelerated variant of `within_distance`.
pub fn within_distance_bvh(scene: &Scene, bvh: &SceneBvh, group_index: usize, pt: Vec2) -> bool {
    let Some(group) = scene.groups.get(group_index) else {
        return false;
    };
    let Some(group_bvh) = bvh.groups.get(group_index) else {
        return false;
    };
    if group_bvh.nodes.is_empty() {
        return within_distance(scene, group_index, pt);
    }

    let local_pt = group.canvas_to_shape.transform_point(pt);
    let mut stack = Vec::new();
    stack.push(0usize);

    while let Some(node_index) = stack.pop() {
        let node = &group_bvh.nodes[node_index];
        if !node.bounds.contains(local_pt) {
            continue;
        }
        if node.count > 0 {
            let start = node.start as usize;
            let end = start + node.count as usize;
            for i in start..end {
                let shape_slot = group_bvh.indices[i];
                let shape_meta = &group_bvh.shapes[shape_slot];
                let shape_index = shape_meta.shape_index;
                let Some(shape) = scene.shapes.get(shape_index) else {
                    continue;
                };
                if shape.stroke_width <= 0.0 {
                    continue;
                }
                let stroke_width = shape.stroke_width;
                if let Some(path_bvh) = &shape_meta.path_bvh {
                    let local_pt = transform_point_inverse(shape.transform, local_pt);
                    if within_distance_path_bvh(path_bvh, local_pt) {
                        return true;
                    }
                } else if within_distance_shape(shape, local_pt, stroke_width) {
                    return true;
                }
            }
        } else {
            let left = node.left as usize;
            let right = node.right as usize;
            if left != BVH_NONE as usize {
                stack.push(left);
            }
            if right != BVH_NONE as usize {
                stack.push(right);
            }
        }
    }

    false
}

/// BVH-accelerated variant of `compute_distance`.
pub fn compute_distance_bvh(
    scene: &Scene,
    bvh: &SceneBvh,
    group_index: usize,
    pt: Vec2,
    max_radius: f32,
    options: DistanceOptions,
) -> Option<ClosestPoint> {
    let Some(group) = scene.groups.get(group_index) else {
        return None;
    };
    let Some(group_bvh) = bvh.groups.get(group_index) else {
        return None;
    };
    if group_bvh.nodes.is_empty() {
        return compute_distance(scene, group_index, pt, max_radius, options);
    }

    let local_pt = group.canvas_to_shape.transform_point(pt);
    let mut best: Option<ClosestPoint> = None;
    let mut best_dist = max_radius;

    let mut stack = Vec::new();
    stack.push(0usize);

    while let Some(node_index) = stack.pop() {
        let node = &group_bvh.nodes[node_index];
        let node_dist = node.bounds.distance(local_pt);
        if node_dist > best_dist {
            continue;
        }

        if node.count > 0 {
            let start = node.start as usize;
            let end = start + node.count as usize;
            for i in start..end {
                let shape_slot = group_bvh.indices[i];
                let shape_meta = &group_bvh.shapes[shape_slot];
                let shape_index = shape_meta.shape_index;
                let Some(shape) = scene.shapes.get(shape_index) else {
                    continue;
                };
                let local = if let Some(path_bvh) = &shape_meta.path_bvh {
                    let shape_pt = transform_point_inverse(shape.transform, local_pt);
                    closest_point_path_bvh(path_bvh, shape_pt).map(|(shape_closest, path_info)| {
                        let group_closest = shape.transform.transform_point(shape_closest);
                        (group_closest, path_info)
                    })
                } else {
                    closest_point_shape(shape, local_pt, options.path_tolerance)
                };
                let Some((local_closest, path_info)) = local else {
                    continue;
                };
                let closest = group.shape_to_canvas.transform_point(local_closest);
                let distance = (closest - pt).length();
                if distance <= best_dist {
                    best_dist = distance;
                    best = Some(ClosestPoint {
                        shape_index,
                        point: closest,
                        distance,
                        path: path_info,
                    });
                }
            }
        } else {
            let left = node.left as usize;
            let right = node.right as usize;
            let mut left_dist = f32::INFINITY;
            let mut right_dist = f32::INFINITY;
            if left != BVH_NONE as usize {
                left_dist = group_bvh.nodes[left].bounds.distance(local_pt);
            }
            if right != BVH_NONE as usize {
                right_dist = group_bvh.nodes[right].bounds.distance(local_pt);
            }

            // Traverse the closer child first to improve pruning.
            if left_dist < right_dist {
                if right_dist <= best_dist {
                    stack.push(right);
                }
                if left_dist <= best_dist {
                    stack.push(left);
                }
            } else {
                if left_dist <= best_dist {
                    stack.push(left);
                }
                if right_dist <= best_dist {
                    stack.push(right);
                }
            }
        }
    }

    best
}

/// Determine whether a point is inside the group's filled region.
pub(crate) fn is_inside(scene: &Scene, group_index: usize, pt: Vec2) -> bool {
    let Some(group) = scene.groups.get(group_index) else {
        return false;
    };
    let local_pt = group.canvas_to_shape.transform_point(pt);
    let mut winding = 0i32;
    for &shape_index in &group.shape_indices {
        let Some(shape) = scene.shapes.get(shape_index) else {
            continue;
        };
        let shape_pt = transform_point_inverse(shape.transform, local_pt);
        winding += winding_number_shape(shape, None, shape_pt);
    }
    match group.fill_rule {
        FillRule::EvenOdd => winding.abs() % 2 == 1,
        FillRule::NonZero => winding != 0,
    }
}

/// BVH-accelerated inside test used for fill evaluation.
pub(crate) fn is_inside_bvh(scene: &Scene, bvh: &SceneBvh, group_index: usize, pt: Vec2) -> bool {
    let Some(group) = scene.groups.get(group_index) else {
        return false;
    };
    let Some(group_bvh) = bvh.groups.get(group_index) else {
        return false;
    };
    if group_bvh.nodes.is_empty() {
        return is_inside(scene, group_index, pt);
    }
    let local_pt = group.canvas_to_shape.transform_point(pt);
    if !group_bvh.nodes[0].bounds.contains(local_pt) {
        return false;
    }
    let mut winding = 0i32;
    let mut stack = Vec::new();
    stack.push(0usize);
    while let Some(node_index) = stack.pop() {
        let node = &group_bvh.nodes[node_index];
        if !node.bounds.contains(local_pt) {
            continue;
        }
        if node.count > 0 {
            let start = node.start as usize;
            let end = start + node.count as usize;
            for i in start..end {
                let shape_slot = group_bvh.indices[i];
                let shape_meta = &group_bvh.shapes[shape_slot];
                let shape_index = shape_meta.shape_index;
                let Some(shape) = scene.shapes.get(shape_index) else {
                    continue;
                };
                let shape_pt = transform_point_inverse(shape.transform, local_pt);
                winding += winding_number_shape(shape, shape_meta.path_bvh.as_ref(), shape_pt);
            }
        } else {
            let left = node.left as usize;
            let right = node.right as usize;
            if left != BVH_NONE as usize {
                stack.push(left);
            }
            if right != BVH_NONE as usize {
                stack.push(right);
            }
        }
    }
    match group.fill_rule {
        FillRule::EvenOdd => winding.abs() % 2 == 1,
        FillRule::NonZero => winding != 0,
    }
}
