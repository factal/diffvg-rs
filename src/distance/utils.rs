//! Small math helpers used across distance modules.

use crate::math::{Mat3, Vec2};


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
