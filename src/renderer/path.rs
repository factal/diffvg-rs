//! Path-to-curve preprocessing helpers.

use crate::geometry::{Path, StrokeSegment};
use crate::math::Vec2;
use crate::path_utils::path_point_radius;

use super::constants::{CURVE_STRIDE, SEGMENT_STRIDE};

/// Stroke curve segment with optional varying radius.
#[derive(Debug, Copy, Clone)]
pub(crate) struct CurveSegment {
    /// Segment kind: 0 = line, 1 = quad, 2 = cubic.
    pub(crate) kind: u32,
    /// Start point.
    pub(crate) p0: Vec2,
    /// First control point (or end point for lines).
    pub(crate) p1: Vec2,
    /// Second control point (or end point for lines/quads).
    pub(crate) p2: Vec2,
    /// End point.
    pub(crate) p3: Vec2,
    /// Radius at `p0`.
    pub(crate) r0: f32,
    /// Radius at `p1`.
    pub(crate) r1: f32,
    /// Radius at `p2`.
    pub(crate) r2: f32,
    /// Radius at `p3`.
    pub(crate) r3: f32,
}

impl CurveSegment {
    /// Build a line segment with per-end radii.
    pub(crate) fn line(p0: Vec2, p1: Vec2, r0: f32, r1: f32) -> Self {
        Self {
            kind: 0,
            p0,
            p1,
            p2: p1,
            p3: p1,
            r0,
            r1,
            r2: r1,
            r3: r1,
        }
    }

    /// Build a quadratic segment with per-point radii.
    pub(crate) fn quad(p0: Vec2, p1: Vec2, p2: Vec2, r0: f32, r1: f32, r2: f32) -> Self {
        Self {
            kind: 1,
            p0,
            p1,
            p2,
            p3: p2,
            r0,
            r1,
            r2,
            r3: r2,
        }
    }

    /// Build a cubic segment with per-point radii.
    pub(crate) fn cubic(
        p0: Vec2,
        p1: Vec2,
        p2: Vec2,
        p3: Vec2,
        r0: f32,
        r1: f32,
        r2: f32,
        r3: f32,
    ) -> Self {
        Self {
            kind: 2,
            p0,
            p1,
            p2,
            p3,
            r0,
            r1,
            r2,
            r3,
        }
    }
}

/// Append curve segments and return `(offset, count)` for packing.
pub(crate) fn push_curve_segments(
    curves: &mut Vec<CurveSegment>,
    new_segments: &[CurveSegment],
) -> (u32, u32) {
    let offset = curves.len() as u32;
    curves.extend_from_slice(new_segments);
    let count = new_segments.len() as u32;
    (offset, count)
}

/// Expand a path into per-segment curve data with interpolated stroke radii.
pub(crate) fn path_to_curve_segments(path: &Path, stroke_width: f32) -> Vec<CurveSegment> {
    let mut out = Vec::new();
    if path.is_empty() {
        return out;
    }

    let mut point_id = 0usize;
    let total_points = path.points.len();

    for &num_controls in &path.num_control_points {
        match num_controls {
            0 => {
                let i0 = point_id;
                let i1 = point_id + 1;
                let (p0, r0) = match path_point_radius(path, i0, total_points, stroke_width, 1.0) {
                    Some(values) => values,
                    None => break,
                };
                let (p1, r1) = match path_point_radius(path, i1, total_points, stroke_width, 1.0) {
                    Some(values) => values,
                    None => break,
                };
                out.push(CurveSegment::line(p0, p1, r0, r1));
                point_id += 1;
            }
            1 => {
                let i0 = point_id;
                let i1 = point_id + 1;
                let i2 = point_id + 2;
                let (p0, r0) = match path_point_radius(path, i0, total_points, stroke_width, 1.0) {
                    Some(values) => values,
                    None => break,
                };
                let (p1, r1) = match path_point_radius(path, i1, total_points, stroke_width, 1.0) {
                    Some(values) => values,
                    None => break,
                };
                let (p2, r2) = match path_point_radius(path, i2, total_points, stroke_width, 1.0) {
                    Some(values) => values,
                    None => break,
                };
                out.push(CurveSegment::quad(p0, p1, p2, r0, r1, r2));
                point_id += 2;
            }
            2 => {
                let i0 = point_id;
                let i1 = point_id + 1;
                let i2 = point_id + 2;
                let i3 = point_id + 3;
                let (p0, r0) = match path_point_radius(path, i0, total_points, stroke_width, 1.0) {
                    Some(values) => values,
                    None => break,
                };
                let (p1, r1) = match path_point_radius(path, i1, total_points, stroke_width, 1.0) {
                    Some(values) => values,
                    None => break,
                };
                let (p2, r2) = match path_point_radius(path, i2, total_points, stroke_width, 1.0) {
                    Some(values) => values,
                    None => break,
                };
                let (p3, r3) = match path_point_radius(path, i3, total_points, stroke_width, 1.0) {
                    Some(values) => values,
                    None => break,
                };
                out.push(CurveSegment::cubic(p0, p1, p2, p3, r0, r1, r2, r3));
                point_id += 3;
            }
            _ => break,
        }
    }

    out
}

/// Pack stroke segments into the GPU float layout.
pub(crate) fn segments_to_f32(segments: &[StrokeSegment]) -> Vec<f32> {
    let mut data = Vec::with_capacity(segments.len() * SEGMENT_STRIDE);
    for seg in segments {
        data.push(seg.start.x);
        data.push(seg.start.y);
        data.push(seg.end.x);
        data.push(seg.end.y);
        data.push(seg.r0);
        data.push(seg.r1);
        data.push(seg.prev_dir.x);
        data.push(seg.prev_dir.y);
        data.push(seg.next_dir.x);
        data.push(seg.next_dir.y);
        data.push(if seg.start_cap { 1.0 } else { 0.0 });
        data.push(if seg.end_cap { 1.0 } else { 0.0 });
    }
    data
}

/// Pack curve segments into the GPU float layout.
pub(crate) fn curve_segments_to_f32(curves: &[CurveSegment]) -> Vec<f32> {
    let mut data = Vec::with_capacity(curves.len() * CURVE_STRIDE);
    for seg in curves {
        data.push(seg.kind as f32);
        data.push(seg.p0.x);
        data.push(seg.p0.y);
        data.push(seg.p1.x);
        data.push(seg.p1.y);
        data.push(seg.p2.x);
        data.push(seg.p2.y);
        data.push(seg.p3.x);
        data.push(seg.p3.y);
        data.push(seg.r0);
        data.push(seg.r1);
        data.push(seg.r2);
        data.push(seg.r3);
    }
    data
}
