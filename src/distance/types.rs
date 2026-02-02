//! Public types for CPU distance queries.

use crate::math::Vec2;

/// Closest location along a path segment.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ClosestPathPoint {
    /// Segment index within the path that contains the closest point.
    pub segment_index: usize,
    /// Parametric position along the segment (0..1).
    pub t: f32,
}

/// Closest point on any shape in a group.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ClosestPoint {
    /// Index into `Scene.shapes` for the closest shape.
    pub shape_index: usize,
    /// Closest point in canvas space.
    pub point: Vec2,
    /// Euclidean distance from query point to `point`.
    pub distance: f32,
    /// Optional path metadata when the closest shape is a path.
    pub path: Option<ClosestPathPoint>,
}

/// Tuning options for distance evaluation.
#[derive(Debug, Copy, Clone)]
pub struct DistanceOptions {
    /// Curve subdivision tolerance used for path distance evaluation.
    pub path_tolerance: f32,
}

impl Default for DistanceOptions {
    fn default() -> Self {
        // Matches diffvg's default subdivision tolerance for path distance queries.
        Self { path_tolerance: 0.5 }
    }
}
