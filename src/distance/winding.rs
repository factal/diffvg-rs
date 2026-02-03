//! Winding number helpers for fill queries.

use crate::geometry::Path;
use crate::math::Vec2;
use crate::scene::{Shape, ShapeGeometry};

use super::bvh::{Bounds, PathBvh, BVH_NONE};
use super::utils::{path_segment_points, solve_cubic, solve_quadratic};

/// Returns the winding number contribution for a single shape at `pt`.
///
/// If `path_bvh` is provided, it is used for path shapes to accelerate queries.
pub(crate) fn winding_number_shape(shape: &Shape, path_bvh: Option<&PathBvh>, pt: Vec2) -> i32 {
    match &shape.geometry {
        ShapeGeometry::Circle { center, radius } => {
            if (pt - *center).length_squared() < radius.abs() * radius.abs() {
                1
            } else {
                0
            }
        }
        ShapeGeometry::Ellipse { center, radius } => {
            let rx = radius.x.abs().max(1.0e-6);
            let ry = radius.y.abs().max(1.0e-6);
            let dx = (pt.x - center.x) / rx;
            let dy = (pt.y - center.y) / ry;
            if dx * dx + dy * dy < 1.0 {
                1
            } else {
                0
            }
        }
        ShapeGeometry::Rect { min, max } => {
            if pt.x > min.x && pt.x < max.x && pt.y > min.y && pt.y < max.y {
                1
            } else {
                0
            }
        }
        ShapeGeometry::Path { path } => {
            if let Some(path_bvh) = path_bvh {
                winding_number_path_bvh(path_bvh, pt)
            } else {
                winding_number_path(path, pt)
            }
        }
    }
}

/// Computes the winding number of a path at `pt` by walking segments.
fn winding_number_path(path: &Path, pt: Vec2) -> i32 {
    if path.points.is_empty() {
        return 0;
    }
    let total_points = path.points.len();
    let mut winding = 0i32;
    let mut point_id = 0usize;
    for &num_controls in &path.num_control_points {
        let Some((p0, p1, p2, p3, next_point_id)) =
            path_segment_points(path, point_id, total_points, num_controls)
        else {
            break;
        };
        winding += winding_number_segment(num_controls, p0, p1, p2, p3, pt);
        point_id = next_point_id;
    }
    winding
}

/// Computes the winding number using a path BVH for pruning.
fn winding_number_path_bvh(path_bvh: &PathBvh, pt: Vec2) -> i32 {
    if path_bvh.nodes.is_empty() {
        return 0;
    }
    let mut winding = 0i32;
    let mut stack = Vec::new();
    stack.push(0usize);
    while let Some(node_index) = stack.pop() {
        let node = &path_bvh.nodes[node_index];
        if !ray_intersects_bounds(&node.bounds, pt) {
            continue;
        }
        if node.count > 0 {
            let start = node.start as usize;
            let end = start + node.count as usize;
            for i in start..end {
                let seg = &path_bvh.segments[path_bvh.indices[i]];
                winding += winding_number_segment(seg.kind, seg.p0, seg.p1, seg.p2, seg.p3, pt);
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
    winding
}

/// Tests whether a +X ray from `pt` can intersect `bounds`.
fn ray_intersects_bounds(bounds: &Bounds, pt: Vec2) -> bool {
    if pt.y < bounds.min.y || pt.y > bounds.max.y {
        return false;
    }
    if pt.x > bounds.max.x {
        return false;
    }
    true
}

/// Returns the winding contribution of a segment for a +X ray at `pt`.
///
/// `kind` selects line (0), quadratic (1), or cubic (2).
fn winding_number_segment(kind: u8, p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, pt: Vec2) -> i32 {
    match kind {
        0 => {
            let dy = p1.y - p0.y;
            if dy.abs() <= 1.0e-8 {
                return 0;
            }
            let t = (pt.y - p0.y) / dy;
            if t >= 0.0 && t <= 1.0 {
                let tp = p0.x - pt.x + t * (p1.x - p0.x);
                if tp >= 0.0 {
                    return if dy > 0.0 { 1 } else { -1 };
                }
            }
            0
        }
        1 => {
            // Quadratic: solve y(t) = pt.y to count crossings.
            let a = (p0.y - 2.0 * p1.y + p2.y) as f64;
            let b = (-2.0 * p0.y + 2.0 * p1.y) as f64;
            let c = (p0.y - pt.y) as f64;
            let mut roots = [0.0f64; 2];
            let mut num_roots = 0usize;
            if a.abs() < 1.0e-12 {
                if b.abs() > 1.0e-12 {
                    roots[0] = -c / b;
                    num_roots = 1;
                }
            } else if solve_quadratic(a, b, c, &mut roots) {
                num_roots = 2;
            }
            let mut winding = 0i32;
            let ax = (p0.x - 2.0 * p1.x + p2.x) as f64;
            let bx = (-2.0 * p0.x + 2.0 * p1.x) as f64;
            let cx = (p0.x - pt.x) as f64;
            for i in 0..num_roots {
                let t = roots[i];
                if t >= 0.0 && t <= 1.0 {
                    let tp = ax * t * t + bx * t + cx;
                    if tp > 0.0 {
                        let deriv = 2.0 * a * t + b;
                        winding += if deriv > 0.0 { 1 } else { -1 };
                    }
                }
            }
            winding
        }
        2 => {
            // Cubic: solve y(t) = pt.y to count crossings.
            let a = (-p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y) as f64;
            let b = (3.0 * p0.y - 6.0 * p1.y + 3.0 * p2.y) as f64;
            let c = (-3.0 * p0.y + 3.0 * p1.y) as f64;
            let d = (p0.y - pt.y) as f64;
            let mut roots = [0.0f64; 3];
            let num_roots = solve_cubic(a, b, c, d, &mut roots);
            let ax = (-p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x) as f64;
            let bx = (3.0 * p0.x - 6.0 * p1.x + 3.0 * p2.x) as f64;
            let cx = (-3.0 * p0.x + 3.0 * p1.x) as f64;
            let dx = (p0.x - pt.x) as f64;
            let mut winding = 0i32;
            for i in 0..num_roots {
                let t = roots[i];
                if t >= 0.0 && t <= 1.0 {
                    let tp = ax * t * t * t + bx * t * t + cx * t + dx;
                    if tp > 0.0 {
                        let deriv = 3.0 * a * t * t + 2.0 * b * t + c;
                        winding += if deriv > 0.0 { 1 } else { -1 };
                    }
                }
            }
            winding
        }
        _ => 0,
    }
}

