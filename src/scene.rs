//! Scene graph types for diffvg-rs.

use crate::{color::Color, geometry::Path, math::Mat3, math::Vec2};

/// Reconstruction filter types used for prefiltering.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FilterType {
    Box,
    Tent,
    RadialParabolic,
    Hann,
}

impl FilterType {
    pub(crate) fn as_u32(self) -> u32 {
        match self {
            FilterType::Box => 0,
            FilterType::Tent => 1,
            FilterType::RadialParabolic => 2,
            FilterType::Hann => 3,
        }
    }
}

/// Reconstruction filter configuration.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Filter {
    pub filter_type: FilterType,
    pub radius: f32,
}

impl Filter {
    pub fn new(filter_type: FilterType, radius: f32) -> Self {
        Self { filter_type, radius }
    }
}

/// Fill rule for shape groups.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FillRule {
    NonZero,
    EvenOdd,
}

impl FillRule {
    pub(crate) fn as_u32(self) -> u32 {
        match self {
            FillRule::NonZero => 0,
            FillRule::EvenOdd => 1,
        }
    }
}

/// Stroke join style used for path stroking.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum StrokeJoin {
    Miter,
    Bevel,
    Round,
}

impl StrokeJoin {
    pub(crate) fn as_u32(self) -> u32 {
        match self {
            StrokeJoin::Miter => 0,
            StrokeJoin::Bevel => 1,
            StrokeJoin::Round => 2,
        }
    }
}

/// Stroke end-cap style.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum StrokeCap {
    Butt,
    Square,
    Round,
}

impl StrokeCap {
    pub(crate) fn as_u32(self) -> u32 {
        match self {
            StrokeCap::Butt => 0,
            StrokeCap::Square => 1,
            StrokeCap::Round => 2,
        }
    }
}

/// A gradient stop with a normalized offset in [0, 1].
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct GradientStop {
    pub offset: f32,
    pub color: Color,
}

/// Linear gradient definition in canvas space.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearGradient {
    pub start: Vec2,
    pub end: Vec2,
    pub stops: Vec<GradientStop>,
}

/// Radial gradient definition in canvas space.
#[derive(Debug, Clone, PartialEq)]
pub struct RadialGradient {
    pub center: Vec2,
    pub radius: Vec2,
    pub stops: Vec<GradientStop>,
}

/// Paint definition for fills and strokes.
#[derive(Debug, Clone, PartialEq)]
pub enum Paint {
    Solid(Color),
    LinearGradient(LinearGradient),
    RadialGradient(RadialGradient),
}

/// Shape geometry variants supported by diffvg-rs.
#[derive(Debug, Clone, PartialEq)]
pub enum ShapeGeometry {
    Circle { center: Vec2, radius: f32 },
    Ellipse { center: Vec2, radius: Vec2 },
    Rect { min: Vec2, max: Vec2 },
    Path { path: Path },
}

/// Renderable shape with local transform and stroke settings.
#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    pub geometry: ShapeGeometry,
    pub transform: Mat3,
    pub stroke_width: f32,
    pub stroke_join: StrokeJoin,
    pub stroke_cap: StrokeCap,
    pub stroke_miter_limit: f32,
}

impl Shape {
    /// Construct a shape with default transform and stroke settings.
    pub fn new(geometry: ShapeGeometry) -> Self {
        Self {
            geometry,
            transform: Mat3::identity(),
            stroke_width: 0.0,
            stroke_join: StrokeJoin::Round,
            stroke_cap: StrokeCap::Round,
            stroke_miter_limit: 4.0,
        }
    }
}

/// A group of shapes sharing fill/stroke paints and a group transform.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeGroup {
    pub shape_indices: Vec<usize>,
    pub fill: Option<Paint>,
    pub stroke: Option<Paint>,
    pub opacity: f32,
    pub fill_rule: FillRule,
    pub shape_to_canvas: Mat3,
    pub canvas_to_shape: Mat3,
}

impl ShapeGroup {
    /// Construct a shape group with default opacity and transforms.
    pub fn new(shape_indices: Vec<usize>, fill: Option<Paint>, stroke: Option<Paint>) -> Self {
        Self {
            shape_indices,
            fill,
            stroke,
            opacity: 1.0,
            fill_rule: FillRule::NonZero,
            shape_to_canvas: Mat3::identity(),
            canvas_to_shape: Mat3::identity(),
        }
    }

    /// Set shape-to-canvas transform and update its inverse.
    pub fn set_shape_to_canvas(&mut self, shape_to_canvas: Mat3) {
        self.shape_to_canvas = shape_to_canvas;
        self.canvas_to_shape = shape_to_canvas.inverse().unwrap_or(Mat3::identity());
    }

    /// Set canvas-to-shape transform and update its inverse.
    pub fn set_canvas_to_shape(&mut self, canvas_to_shape: Mat3) {
        self.canvas_to_shape = canvas_to_shape;
        self.shape_to_canvas = canvas_to_shape.inverse().unwrap_or(Mat3::identity());
    }
}

/// Root scene container for rendering.
#[derive(Debug, Clone)]
pub struct Scene {
    pub width: u32,
    pub height: u32,
    pub shapes: Vec<Shape>,
    pub groups: Vec<ShapeGroup>,
    pub background: Color,
    pub background_image: Option<Vec<f32>>,
    pub filter: Filter,
}

impl Scene {
    /// Construct an empty scene with a default box filter.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            shapes: Vec::new(),
            groups: Vec::new(),
            background: Color::TRANSPARENT,
            background_image: None,
            filter: Filter::new(FilterType::Box, 0.5),
        }
    }
}
