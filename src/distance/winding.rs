//! Winding number helpers for fill queries.

use crate::geometry::Path;
use crate::math::Vec2;
use crate::scene::{Shape, ShapeGeometry};
use crate::path_utils::path_point;

use super::bvh::{Bounds, PathBvh, BVH_NONE};

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
                winding += winding_number_segment(0, p0, p1, p1, p1, pt);
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
                winding += winding_number_segment(1, p0, p1, p2, p2, pt);
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
                winding += winding_number_segment(2, p0, p1, p2, p3, pt);
                point_id += 3;
            }
            _ => break,
        }
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

/// Solves a quadratic and stores real roots in ascending order.
fn solve_quadratic(a: f64, b: f64, c: f64, t: &mut [f64; 2]) -> bool {
    let eps = 1.0e-12;
    if a.abs() < eps {
        if b.abs() < eps {
            return false;
        }
        let root = -c / b;
        t[0] = root;
        t[1] = root;
        return true;
    }
    let discrim = b * b - 4.0 * a * c;
    if discrim < 0.0 {
        return false;
    }
    let root = discrim.sqrt();
    if root.abs() < eps {
        let r = -0.5 * b / a;
        t[0] = r;
        t[1] = r;
        return true;
    }
    let q = if b < 0.0 { -0.5 * (b - root) } else { -0.5 * (b + root) };
    if q.abs() < eps {
        let inv = 0.5 / a;
        t[0] = (-b - root) * inv;
        t[1] = (-b + root) * inv;
    } else {
        t[0] = q / a;
        t[1] = c / q;
    }
    if t[0] > t[1] {
        t.swap(0, 1);
    }
    true
}

/// Solves a cubic and stores real roots; returns the count written.
fn solve_cubic(a: f64, b: f64, c: f64, d: f64, t: &mut [f64; 3]) -> usize {
    if a.abs() < 1.0e-6 {
        let mut roots = [0.0f64; 2];
        if solve_quadratic(b, c, d, &mut roots) {
            t[0] = roots[0];
            t[1] = roots[1];
            return 2;
        }
        return 0;
    }
    let b = b / a;
    let c = c / a;
    let d = d / a;
    let q = (b * b - 3.0 * c) / 9.0;
    let r = (2.0 * b * b * b - 9.0 * b * c + 27.0 * d) / 54.0;
    if r * r < q * q * q {
        let theta = (r / (q * q * q).sqrt()).acos();
        t[0] = -2.0 * q.sqrt() * (theta / 3.0).cos() - b / 3.0;
        t[1] = -2.0 * q.sqrt() * ((theta + 2.0 * std::f64::consts::PI) / 3.0).cos() - b / 3.0;
        t[2] = -2.0 * q.sqrt() * ((theta - 2.0 * std::f64::consts::PI) / 3.0).cos() - b / 3.0;
        return 3;
    }
    let a_root = if r > 0.0 {
        -cbrt(r + (r * r - q * q * q).sqrt())
    } else {
        cbrt(-r + (r * r - q * q * q).sqrt())
    };
    let b_root = if a_root.abs() > 1.0e-6 { q / a_root } else { 0.0 };
    t[0] = (a_root + b_root) - b / 3.0;
    1
}

/// Returns the real cubic root of `x`.
fn cbrt(x: f64) -> f64 {
    if x > 0.0 {
        x.powf(1.0 / 3.0)
    } else if x < 0.0 {
        -(-x).powf(1.0 / 3.0)
    } else {
        0.0
    }
}
