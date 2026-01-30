//! CPU-side distance queries and BVH acceleration.

mod api;
mod bvh;
mod closest;
mod curve;
mod shape;
mod types;
mod utils;
mod winding;

#[cfg(test)]
mod tests;

pub use api::{compute_distance, compute_distance_bvh, within_distance, within_distance_bvh};
pub use bvh::SceneBvh;
pub use types::{ClosestPathPoint, ClosestPoint, DistanceOptions};

pub(crate) use api::is_inside_bvh;
