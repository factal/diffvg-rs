//! GPU data layout and kernel constants.

/// Float stride for packed shape data (`shape_data`).
pub(crate) const SHAPE_STRIDE: u32 = 15;
/// Float stride for packed group data (`group_data`).
pub(crate) const GROUP_STRIDE: u32 = 16;
/// Float stride for packed stroke segment data (`segment_data`).
pub(crate) const SEGMENT_STRIDE: u32 = 12;
/// Float stride for packed curve data (`curve_data`).
pub(crate) const CURVE_STRIDE: u32 = 13;
/// Float stride for packed gradient data (`gradient_data`).
pub(crate) const GRADIENT_STRIDE: u32 = 8;
/// Float stride for packed affine transform data (`*_xform`).
pub(crate) const XFORM_STRIDE: u32 = 6;
/// Float stride for packed AABB bounds data.
pub(crate) const BOUNDS_STRIDE: u32 = 4;
/// Element stride for tile entry lists (`tile_entries`).
pub(crate) const TILE_ENTRY_STRIDE: u32 = 1;
/// Element stride for packed BVH nodes.
pub(crate) const BVH_NODE_STRIDE: u32 = 4;
/// Element stride for packed BVH metadata.
pub(crate) const BVH_META_STRIDE: u32 = 4;
/// Sentinel for missing BVH nodes or indices.
pub(crate) const BVH_NONE: u32 = 0xffff_ffffu32;

/// Shape kind id for circles.
pub(crate) const SHAPE_KIND_CIRCLE: u32 = 0;
/// Shape kind id for rectangles.
pub(crate) const SHAPE_KIND_RECT: u32 = 1;
/// Shape kind id for paths.
pub(crate) const SHAPE_KIND_PATH: u32 = 2;
/// Shape kind id for ellipses.
pub(crate) const SHAPE_KIND_ELLIPSE: u32 = 3;

/// Paint kind id for "no paint".
pub(crate) const PAINT_NONE: u32 = 0;
/// Paint kind id for solid colors.
pub(crate) const PAINT_SOLID: u32 = 1;
/// Paint kind id for linear gradients.
pub(crate) const PAINT_LINEAR: u32 = 2;
/// Paint kind id for radial gradients.
pub(crate) const PAINT_RADIAL: u32 = 3;

/// Stroke join id for miter joins.
pub(crate) const JOIN_MITER: u32 = 0;
/// Stroke join id for round joins.
pub(crate) const JOIN_ROUND: u32 = 2;

/// Filter kind id for box filters.
pub(crate) const FILTER_BOX: u32 = 0;
/// Filter kind id for tent filters.
pub(crate) const FILTER_TENT: u32 = 1;
/// Filter kind id for radial parabolic filters.
pub(crate) const FILTER_RADIAL_PARABOLIC: u32 = 2;
/// Filter kind id for Hann filters.
pub(crate) const FILTER_HANN: u32 = 3;

/// Fixed-point scale for accumulation when float atomics are unavailable.
pub(crate) const ACCUM_SCALE: f32 = 1048576.0;

/// PCG32 multiplier low 32 bits.
pub(crate) const PCG_MULT_LO: u32 = 0x4c957f2d;
/// PCG32 multiplier high 32 bits.
pub(crate) const PCG_MULT_HI: u32 = 0x5851f42d;
/// PCG32 initial state low 32 bits.
pub(crate) const PCG_INIT_LO: u32 = 0x748fea9b;
/// PCG32 initial state high 32 bits.
pub(crate) const PCG_INIT_HI: u32 = 0x853c49e6;
