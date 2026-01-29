mod color;
mod distance;
mod geometry;
mod gpu;
mod math;
mod renderer;
mod scene;

pub use color::Color;
pub use distance::{
    compute_distance, compute_distance_bvh, within_distance, within_distance_bvh, ClosestPathPoint,
    ClosestPoint, DistanceOptions, SceneBvh,
};
pub use geometry::{LineSegment, Path, PathSegment, StrokeSegment};
pub use math::{Mat3, Vec2, Vec4};
pub use renderer::{Image, RenderError, RenderOptions, Renderer, SdfImage};
pub use scene::{
    FillRule, Filter, FilterType, GradientStop, LinearGradient, Paint, RadialGradient, Scene,
    Shape, ShapeGeometry, ShapeGroup, StrokeCap, StrokeJoin,
};
