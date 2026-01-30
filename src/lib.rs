//! diffvg-rs core types, scene model, and render entry points.
//!
//! The crate exposes a Rust-friendly scene graph, CPU distance queries, and
//! GPU-backed rendering helpers built on CubeCL/WGPU.

mod color;
mod backward;
mod distance;
mod geometry;
mod grad;
mod gpu;
mod math;
mod path_utils;
mod renderer;
mod scene;

pub use color::Color;
pub use backward::{render_backward, BackwardOptions};
pub use distance::{
    compute_distance, compute_distance_bvh, within_distance, within_distance_bvh, ClosestPathPoint,
    ClosestPoint, DistanceOptions, SceneBvh,
};
pub use geometry::{LineSegment, Path, PathSegment, StrokeSegment};
pub use grad::{
    DFilter, DGradientStop, DLinearGradient, DPaint, DRadialGradient, DShape, DShapeGeometry,
    DShapeGroup, SceneGrad,
};
pub use math::{Mat3, Vec2, Vec4};
pub use renderer::{Image, RenderError, RenderOptions, Renderer, SdfImage};
pub use scene::{
    FillRule, Filter, FilterType, GradientStop, LinearGradient, Paint, RadialGradient, Scene,
    Shape, ShapeGeometry, ShapeGroup, StrokeCap, StrokeJoin,
};
