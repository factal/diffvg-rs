//! Public types for CPU distance queries.

use crate::math::Vec2;

/// Closest location along a path segment.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ClosestPathPoint {
    pub segment_index: usize,
    pub t: f32,
}

/// Closest point on any shape in a group.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ClosestPoint {
    pub shape_index: usize,
    pub point: Vec2,
    pub distance: f32,
    pub path: Option<ClosestPathPoint>,
}

/// Tuning options for distance evaluation.
#[derive(Debug, Copy, Clone)]
pub struct DistanceOptions {
    pub path_tolerance: f32,
}

impl Default for DistanceOptions {
    fn default() -> Self {
        Self { path_tolerance: 0.5 }
    }
}
