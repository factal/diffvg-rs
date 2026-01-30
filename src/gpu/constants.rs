//! GPU data layout and kernel constants.

pub(crate) const SHAPE_STRIDE: u32 = 15;
pub(crate) const GROUP_STRIDE: u32 = 16;
pub(crate) const SEGMENT_STRIDE: u32 = 12;
pub(crate) const CURVE_STRIDE: u32 = 13;
pub(crate) const GRADIENT_STRIDE: u32 = 8;
pub(crate) const XFORM_STRIDE: u32 = 6;
pub(crate) const BOUNDS_STRIDE: u32 = 4;
pub(crate) const TILE_ENTRY_STRIDE: u32 = 1;
pub(crate) const BVH_NODE_STRIDE: u32 = 4;
pub(crate) const BVH_META_STRIDE: u32 = 4;
pub(crate) const BVH_NONE: u32 = 0xffff_ffffu32;

pub(crate) const SHAPE_KIND_CIRCLE: u32 = 0;
pub(crate) const SHAPE_KIND_RECT: u32 = 1;
pub(crate) const SHAPE_KIND_PATH: u32 = 2;
pub(crate) const SHAPE_KIND_ELLIPSE: u32 = 3;

pub(crate) const PAINT_NONE: u32 = 0;
pub(crate) const PAINT_SOLID: u32 = 1;
pub(crate) const PAINT_LINEAR: u32 = 2;
pub(crate) const PAINT_RADIAL: u32 = 3;

pub(crate) const JOIN_MITER: u32 = 0;
pub(crate) const JOIN_ROUND: u32 = 2;

pub(crate) const FILTER_BOX: u32 = 0;
pub(crate) const FILTER_TENT: u32 = 1;
pub(crate) const FILTER_RADIAL_PARABOLIC: u32 = 2;
pub(crate) const FILTER_HANN: u32 = 3;

pub(crate) const ACCUM_SCALE: f32 = 1048576.0;

pub(crate) const PCG_MULT_LO: u32 = 0x4c957f2d;
pub(crate) const PCG_MULT_HI: u32 = 0x5851f42d;
pub(crate) const PCG_INIT_LO: u32 = 0x748fea9b;
pub(crate) const PCG_INIT_HI: u32 = 0x853c49e6;
