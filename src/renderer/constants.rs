//! Renderer-side constants shared across preparation and GPU packing.

use crate::gpu::constants as gpu_constants;

/// Float stride for packed shape data.
pub(crate) const SHAPE_STRIDE: usize = gpu_constants::SHAPE_STRIDE as usize;
/// Float stride for packed group data.
pub(crate) const GROUP_STRIDE: usize = gpu_constants::GROUP_STRIDE as usize;
/// Float stride for packed curve data.
pub(crate) const CURVE_STRIDE: usize = gpu_constants::CURVE_STRIDE as usize;
/// Float stride for packed gradient data.
pub(crate) const GRADIENT_STRIDE: usize = gpu_constants::GRADIENT_STRIDE as usize;
/// Float stride for packed stroke segment data.
pub(crate) const SEGMENT_STRIDE: usize = gpu_constants::SEGMENT_STRIDE as usize;

/// Largest integer that can be represented exactly in f32.
pub(crate) const MAX_F32_INT: usize = 16_777_216;
/// Default CPU curve flattening tolerance.
pub(crate) const DEFAULT_PATH_TOLERANCE: f32 = 0.5;
/// Tile size in pixels for GPU binning.
pub(crate) const TILE_SIZE: u32 = 16;
/// Maximum primitives per BVH leaf.
pub(crate) const BVH_LEAF_SIZE: usize = 8;
/// Sentinel value for invalid BVH nodes.
pub(crate) const BVH_NONE: u32 = u32::MAX;
