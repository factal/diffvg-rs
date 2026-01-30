//! Renderer-side constants shared across preparation and GPU packing.

use crate::gpu::constants as gpu_constants;

pub(crate) const SHAPE_STRIDE: usize = gpu_constants::SHAPE_STRIDE as usize;
pub(crate) const GROUP_STRIDE: usize = gpu_constants::GROUP_STRIDE as usize;
pub(crate) const CURVE_STRIDE: usize = gpu_constants::CURVE_STRIDE as usize;
pub(crate) const GRADIENT_STRIDE: usize = gpu_constants::GRADIENT_STRIDE as usize;
pub(crate) const SEGMENT_STRIDE: usize = gpu_constants::SEGMENT_STRIDE as usize;

pub(crate) const MAX_F32_INT: usize = 16_777_216;
pub(crate) const DEFAULT_PATH_TOLERANCE: f32 = 0.5;
pub(crate) const TILE_SIZE: u32 = 16;
pub(crate) const BVH_LEAF_SIZE: usize = 8;
pub(crate) const BVH_NONE: u32 = u32::MAX;
