//! Small math helpers used across distance modules.

use crate::geometry::Path;
use crate::math::{Mat3, Vec2};
use crate::path_utils::path_point;


/// Transforms a point by the inverse of the given transformation matrix.
pub(crate) fn transform_point_inverse(transform: Mat3, pt: Vec2) -> Vec2 {
    if transform.is_identity() {
        return pt;
    }
    match transform.inverse() {
        Some(inv) => inv.transform_point(pt),
        None => pt,
    }
}

/// Fetch the control points for a path segment and the next point index.
pub(crate) fn path_segment_points(
    path: &Path,
    point_id: usize,
    total_points: usize,
    num_controls: u8,
) -> Option<(Vec2, Vec2, Vec2, Vec2, usize)> {
    let p0 = path_point(path, point_id, total_points)?;
    let p1 = path_point(path, point_id + 1, total_points)?;
    match num_controls {
        0 => Some((p0, p1, p1, p1, point_id + 1)),
        1 => {
            let p2 = path_point(path, point_id + 2, total_points)?;
            Some((p0, p1, p2, p2, point_id + 2))
        }
        2 => {
            let p2 = path_point(path, point_id + 2, total_points)?;
            let p3 = path_point(path, point_id + 3, total_points)?;
            Some((p0, p1, p2, p3, point_id + 3))
        }
        _ => None,
    }
}

/// Solves a quadratic and stores real roots in ascending order.
pub(crate) fn solve_quadratic(a: f64, b: f64, c: f64, t: &mut [f64; 2]) -> bool {
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
pub(crate) fn solve_cubic(a: f64, b: f64, c: f64, d: f64, t: &mut [f64; 3]) -> usize {
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
