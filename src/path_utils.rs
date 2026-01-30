//! Shared path and shape helpers used by CPU distance and renderer prep.

use crate::geometry::{Path, StrokeSegment};
use crate::math::{Mat3, Vec2};

pub(crate) fn path_point(path: &Path, index: usize, total_points: usize) -> Option<Vec2> {
    if total_points == 0 {
        return None;
    }
    if path.is_closed {
        let idx = index % total_points;
        return Some(path.points[idx]);
    }
    path.points.get(index).copied()
}

pub(crate) fn path_point_radius(
    path: &Path,
    index: usize,
    total_points: usize,
    default_radius: f32,
    thickness_scale: f32,
) -> Option<(Vec2, f32)> {
    let p = path_point(path, index, total_points)?;
    let radius = path
        .thickness
        .as_ref()
        .and_then(|values| {
            if path.is_closed {
                let idx = index % total_points;
                values.get(idx).copied()
            } else {
                values.get(index).copied()
            }
        })
        .map(|value| value * thickness_scale)
        .unwrap_or(default_radius);
    Some((p, radius))
}

pub(crate) fn rect_corners(min: Vec2, max: Vec2) -> [Vec2; 4] {
    [
        Vec2::new(min.x, min.y),
        Vec2::new(max.x, min.y),
        Vec2::new(max.x, max.y),
        Vec2::new(min.x, max.y),
    ]
}

pub(crate) fn ellipse_to_segments(
    center: Vec2,
    radius: Vec2,
    transform: Mat3,
    tolerance: f32,
) -> Vec<StrokeSegment> {
    let rx = radius.x.abs();
    let ry = radius.y.abs();
    if rx == 0.0 || ry == 0.0 {
        return Vec::new();
    }

    let circumference = 2.0 * core::f32::consts::PI * 0.5 * (rx + ry);
    let mut steps = (circumference / tolerance.max(0.01)).ceil() as usize;
    steps = steps.clamp(12, 256);

    let mut points = Vec::with_capacity(steps);
    for i in 0..steps {
        let angle = (i as f32) * (2.0 * core::f32::consts::PI / steps as f32);
        let (sin, cos) = angle.sin_cos();
        let point = Vec2::new(center.x + cos * rx, center.y + sin * ry);
        points.push(transform.transform_point(point));
    }

    let mut segs = Vec::with_capacity(steps);
    for i in 0..steps {
        let a = points[i];
        let b = points[(i + 1) % steps];
        segs.push(StrokeSegment::new(a, b, 0.0, 0.0));
    }
    segs
}

pub(crate) fn transform_path(path: &Path, transform: Mat3) -> Path {
    if transform.is_identity() {
        return path.clone();
    }
    let points = path
        .points
        .iter()
        .map(|p| transform.transform_point(*p))
        .collect::<Vec<_>>();
    Path {
        num_control_points: path.num_control_points.clone(),
        points,
        thickness: path.thickness.clone(),
        is_closed: path.is_closed,
        use_distance_approx: path.use_distance_approx,
    }
}

pub(crate) fn bounds_from_points(points: &[Vec2]) -> (Vec2, Vec2) {
    let mut min = Vec2::new(f32::INFINITY, f32::INFINITY);
    let mut max = Vec2::new(f32::NEG_INFINITY, f32::NEG_INFINITY);
    for p in points {
        min = min.min(*p);
        max = max.max(*p);
    }
    (min, max)
}
