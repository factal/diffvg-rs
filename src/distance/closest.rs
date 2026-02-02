//! Closest-point and within-distance evaluation helpers.

use crate::math::Vec2;
use crate::scene::{Shape, ShapeGeometry};
use crate::path_utils::{path_point, path_point_radius};

use super::bvh::{PathBvh, BVH_NONE};
use super::curve::{closest_point_cubic, closest_point_quadratic, distance_to_segment};
use super::shape::{
    closest_point_circle, closest_point_ellipse, closest_point_rect, ellipse_signed_distance,
    rect_signed_distance,
};
use super::utils::transform_point_inverse;

/// Returns true if `pt` lies within the per-segment stroke radius of any curve in `path_bvh`.
///
/// `pt` must be in the path's local space (after applying the per-shape inverse transform).
/// The query uses the path BVH for pruning and respects `use_distance_approx` when evaluating
/// quadratic/cubic distances.
pub(crate) fn within_distance_path_bvh(path_bvh: &PathBvh, pt: Vec2) -> bool {
    if path_bvh.nodes.is_empty() {
        return false;
    }

    let mut stack = Vec::new();
    stack.push(0usize);

    while let Some(node_index) = stack.pop() {
        let node = &path_bvh.nodes[node_index];
        if !node.bounds.contains(pt) {
            continue;
        }
        if node.count > 0 {
            let start = node.start as usize;
            let end = start + node.count as usize;
            for i in start..end {
                let seg = &path_bvh.segments[path_bvh.indices[i]];
                let (dist, t) = match seg.kind {
                    0 => {
                        let (dist, _cp, t) = distance_to_segment(pt, seg.p0, seg.p1);
                        (dist, t)
                    }
                    1 => {
                        let (_cp, t, dist) = closest_point_quadratic(
                            pt,
                            seg.p0,
                            seg.p1,
                            seg.p2,
                            path_bvh.use_distance_approx,
                        );
                        (dist, t)
                    }
                    _ => {
                        let (_cp, t, dist) = closest_point_cubic(
                            pt,
                            seg.p0,
                            seg.p1,
                            seg.p2,
                            seg.p3,
                            path_bvh.use_distance_approx,
                        );
                        (dist, t)
                    }
                };

                let radius = match seg.kind {
                    0 => seg.r0 + t * (seg.r1 - seg.r0),
                    1 => {
                        let tt = 1.0 - t;
                        tt * tt * seg.r0 + 2.0 * tt * t * seg.r1 + t * t * seg.r2
                    }
                    _ => {
                        let tt = 1.0 - t;
                        tt * tt * tt * seg.r0
                            + 3.0 * tt * tt * t * seg.r1
                            + 3.0 * tt * t * t * seg.r2
                            + t * t * t * seg.r3
                    }
                };

                if dist < radius {
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

/// Finds the closest point on any segment in `path_bvh` to `pt`.
///
/// Returns the closest point in path-local space along with optional path metadata
/// (segment index + parametric `t`). Returns `None` when the BVH is empty.
pub(crate) fn closest_point_path_bvh(
    path_bvh: &PathBvh,
    pt: Vec2,
) -> Option<(Vec2, Option<super::types::ClosestPathPoint>)> {
    if path_bvh.nodes.is_empty() {
        return None;
    }

    let mut best_dist = f32::INFINITY;
    let mut best_pt = Vec2::ZERO;
    let mut best_seg = 0usize;
    let mut best_t = 0.0;

    let mut stack = Vec::new();
    stack.push(0usize);

    while let Some(node_index) = stack.pop() {
        let node = &path_bvh.nodes[node_index];
        if node.bounds.distance(pt) > best_dist {
            continue;
        }
        if node.count > 0 {
            let start = node.start as usize;
            let end = start + node.count as usize;
            for i in start..end {
                let seg = &path_bvh.segments[path_bvh.indices[i]];
                let (cp, t, dist) = match seg.kind {
                    0 => {
                        let (dist, cp, t) = distance_to_segment(pt, seg.p0, seg.p1);
                        (cp, t, dist)
                    }
                    1 => closest_point_quadratic(
                        pt,
                        seg.p0,
                        seg.p1,
                        seg.p2,
                        path_bvh.use_distance_approx,
                    ),
                    _ => closest_point_cubic(
                        pt,
                        seg.p0,
                        seg.p1,
                        seg.p2,
                        seg.p3,
                        path_bvh.use_distance_approx,
                    ),
                };
                if dist < best_dist {
                    best_dist = dist;
                    best_pt = cp;
                    best_seg = seg.segment_index;
                    best_t = t;
                }
            }
        } else {
            let left = node.left as usize;
            let right = node.right as usize;
            let mut left_dist = f32::INFINITY;
            let mut right_dist = f32::INFINITY;
            if left != BVH_NONE as usize {
                left_dist = path_bvh.nodes[left].bounds.distance(pt);
            }
            if right != BVH_NONE as usize {
                right_dist = path_bvh.nodes[right].bounds.distance(pt);
            }

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

    if best_dist.is_finite() {
        Some((
            best_pt,
            Some(super::types::ClosestPathPoint {
                segment_index: best_seg,
                t: best_t,
            }),
        ))
    } else {
        None
    }
}

/// Returns true if `pt` is within distance `r` of the shape boundary.
///
/// `pt` is expected in the group-local space (after applying `ShapeGroup.canvas_to_shape`);
/// the per-shape inverse transform is applied internally. For paths, `r` is treated as
/// the stroke radius and per-point thickness values are honored. Returns `false` when `r <= 0`.
pub(crate) fn within_distance_shape(shape: &Shape, pt: Vec2, r: f32) -> bool {
    if r <= 0.0 {
        return false;
    }
    let local_pt = transform_point_inverse(shape.transform, pt);
    match &shape.geometry {
        ShapeGeometry::Circle { center, radius } => {
            let dist = (local_pt - *center).length() - radius.abs();
            dist.abs() < r
        }
        ShapeGeometry::Ellipse { center, radius } => {
            let dist = ellipse_signed_distance(
                local_pt,
                *center,
                Vec2::new(radius.x.abs(), radius.y.abs()),
            );
            dist.abs() < r
        }
        ShapeGeometry::Rect { min, max } => {
            let dist = rect_signed_distance(local_pt, *min, *max);
            dist.abs() < r
        }
        ShapeGeometry::Path { path } => {
            let stroke_width = r;
            let total_points = path.points.len();
            if total_points == 0 {
                return false;
            }
            let mut point_id = 0usize;
            for &num_controls in &path.num_control_points {
                match num_controls {
                    0 => {
                        let i0 = point_id;
                        let i1 = point_id + 1;
                        let (p0, r0) = match path_point_radius(
                            path,
                            i0,
                            total_points,
                            stroke_width,
                            1.0,
                        ) {
                            Some(value) => value,
                            None => break,
                        };
                        let (p1, r1) = match path_point_radius(
                            path,
                            i1,
                            total_points,
                            stroke_width,
                            1.0,
                        ) {
                            Some(value) => value,
                            None => break,
                        };
                        let (dist, _cp, t) = distance_to_segment(local_pt, p0, p1);
                        let radius = r0 + t * (r1 - r0);
                        if dist < radius {
                            return true;
                        }
                        point_id += 1;
                    }
                    1 => {
                        let i0 = point_id;
                        let i1 = point_id + 1;
                        let i2 = point_id + 2;
                        let (p0, r0) = match path_point_radius(
                            path,
                            i0,
                            total_points,
                            stroke_width,
                            1.0,
                        ) {
                            Some(value) => value,
                            None => break,
                        };
                        let (p1, r1) = match path_point_radius(
                            path,
                            i1,
                            total_points,
                            stroke_width,
                            1.0,
                        ) {
                            Some(value) => value,
                            None => break,
                        };
                        let (p2, r2) = match path_point_radius(
                            path,
                            i2,
                            total_points,
                            stroke_width,
                            1.0,
                        ) {
                            Some(value) => value,
                            None => break,
                        };
                        let (_cp, t, dist) =
                            closest_point_quadratic(local_pt, p0, p1, p2, path.use_distance_approx);
                        let tt = 1.0 - t;
                        let radius = tt * tt * r0 + 2.0 * tt * t * r1 + t * t * r2;
                        if dist < radius {
                            return true;
                        }
                        point_id += 2;
                    }
                    2 => {
                        let i0 = point_id;
                        let i1 = point_id + 1;
                        let i2 = point_id + 2;
                        let i3 = point_id + 3;
                        let (p0, r0) = match path_point_radius(
                            path,
                            i0,
                            total_points,
                            stroke_width,
                            1.0,
                        ) {
                            Some(value) => value,
                            None => break,
                        };
                        let (p1, r1) = match path_point_radius(
                            path,
                            i1,
                            total_points,
                            stroke_width,
                            1.0,
                        ) {
                            Some(value) => value,
                            None => break,
                        };
                        let (p2, r2) = match path_point_radius(
                            path,
                            i2,
                            total_points,
                            stroke_width,
                            1.0,
                        ) {
                            Some(value) => value,
                            None => break,
                        };
                        let (p3, r3) = match path_point_radius(
                            path,
                            i3,
                            total_points,
                            stroke_width,
                            1.0,
                        ) {
                            Some(value) => value,
                            None => break,
                        };
                        let (_cp, t, dist) = closest_point_cubic(
                            local_pt,
                            p0,
                            p1,
                            p2,
                            p3,
                            path.use_distance_approx,
                        );
                        let tt = 1.0 - t;
                        let radius = tt * tt * tt * r0
                            + 3.0 * tt * tt * t * r1
                            + 3.0 * tt * t * t * r2
                            + t * t * t * r3;
                        if dist < radius {
                            return true;
                        }
                        point_id += 3;
                    }
                    _ => break,
                }
            }
            false
        }
    }
}

/// Computes the closest point on `shape` to `pt` in group-local space.
///
/// The returned point is in group-local space, and path metadata is provided only for paths.
/// Returns `None` when the path has no points. `tolerance` is reserved for future path
/// approximation controls and is currently unused.
pub(crate) fn closest_point_shape(
    shape: &Shape,
    pt: Vec2,
    tolerance: f32,
) -> Option<(Vec2, Option<super::types::ClosestPathPoint>)> {
    let local_pt = transform_point_inverse(shape.transform, pt);
    match &shape.geometry {
        ShapeGeometry::Circle { center, radius } => {
            let local_closest = closest_point_circle(*center, radius.abs(), local_pt);
            let closest = shape.transform.transform_point(local_closest);
            Some((closest, None))
        }
        ShapeGeometry::Ellipse { center, radius } => {
            let local_closest =
                closest_point_ellipse(*center, Vec2::new(radius.x.abs(), radius.y.abs()), local_pt);
            let closest = shape.transform.transform_point(local_closest);
            Some((closest, None))
        }
        ShapeGeometry::Rect { min, max } => {
            let local_closest = closest_point_rect(*min, *max, local_pt);
            let closest = shape.transform.transform_point(local_closest);
            Some((closest, None))
        }
        ShapeGeometry::Path { path } => {
            let total_points = path.points.len();
            if total_points == 0 {
                return None;
            }
            let mut best_dist = f32::INFINITY;
            let mut best_pt = Vec2::ZERO;
            let mut best_seg = 0usize;
            let mut best_t = 0.0;
            let mut point_id = 0usize;
            for (seg_index, &num_controls) in path.num_control_points.iter().enumerate() {
                match num_controls {
                    0 => {
                        let i0 = point_id;
                        let i1 = point_id + 1;
                        let p0 = match path_point(path, i0, total_points) {
                            Some(value) => value,
                            None => break,
                        };
                        let p1 = match path_point(path, i1, total_points) {
                            Some(value) => value,
                            None => break,
                        };
                        let (dist, cp, t) = distance_to_segment(local_pt, p0, p1);
                        if dist < best_dist {
                            best_dist = dist;
                            best_pt = cp;
                            best_seg = seg_index;
                            best_t = t;
                        }
                        point_id += 1;
                    }
                    1 => {
                        let i0 = point_id;
                        let i1 = point_id + 1;
                        let i2 = point_id + 2;
                        let p0 = match path_point(path, i0, total_points) {
                            Some(value) => value,
                            None => break,
                        };
                        let p1 = match path_point(path, i1, total_points) {
                            Some(value) => value,
                            None => break,
                        };
                        let p2 = match path_point(path, i2, total_points) {
                            Some(value) => value,
                            None => break,
                        };
                        let (cp, t, dist) =
                            closest_point_quadratic(local_pt, p0, p1, p2, path.use_distance_approx);
                        if dist < best_dist {
                            best_dist = dist;
                            best_pt = cp;
                            best_seg = seg_index;
                            best_t = t;
                        }
                        point_id += 2;
                    }
                    2 => {
                        let i0 = point_id;
                        let i1 = point_id + 1;
                        let i2 = point_id + 2;
                        let i3 = point_id + 3;
                        let p0 = match path_point(path, i0, total_points) {
                            Some(value) => value,
                            None => break,
                        };
                        let p1 = match path_point(path, i1, total_points) {
                            Some(value) => value,
                            None => break,
                        };
                        let p2 = match path_point(path, i2, total_points) {
                            Some(value) => value,
                            None => break,
                        };
                        let p3 = match path_point(path, i3, total_points) {
                            Some(value) => value,
                            None => break,
                        };
                        let (cp, t, dist) = closest_point_cubic(
                            local_pt,
                            p0,
                            p1,
                            p2,
                            p3,
                            path.use_distance_approx,
                        );
                        if dist < best_dist {
                            best_dist = dist;
                            best_pt = cp;
                            best_seg = seg_index;
                            best_t = t;
                        }
                        point_id += 3;
                    }
                    _ => break,
                }
            }
            if best_dist.is_finite() {
                let closest = shape.transform.transform_point(best_pt);
                Some((
                    closest,
                    Some(super::types::ClosestPathPoint {
                        segment_index: best_seg,
                        t: best_t,
                    }),
                ))
            } else {
                None
            }
        }
    }
}
