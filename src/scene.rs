//! Scene graph types for diffvg-rs.

use crate::{color::Color, geometry::Path, math::Mat3, math::Vec2};

/// Reconstruction filter types used for prefiltering.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FilterType {
    /// Box filter with uniform weight inside the radius.
    Box,
    /// Tent filter with linear falloff from the center.
    Tent,
    /// Radial parabolic filter (smooth quadratic falloff).
    RadialParabolic,
    /// Hann window filter.
    Hann,
}

impl FilterType {
    /// Convert the filter type to the GPU enum value.
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
    /// Filter kernel type.
    pub filter_type: FilterType,
    /// Filter radius in pixels.
    pub radius: f32,
}

impl Filter {
    /// Create a filter with the given type and radius.
    pub fn new(filter_type: FilterType, radius: f32) -> Self {
        Self { filter_type, radius }
    }
}

/// Fill rule for shape groups.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FillRule {
    /// Non-zero winding rule.
    NonZero,
    /// Even-odd rule.
    EvenOdd,
}

impl FillRule {
    /// Convert the fill rule to the GPU enum value.
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
    /// Miter join.
    Miter,
    /// Bevel join.
    Bevel,
    /// Round join.
    Round,
}

impl StrokeJoin {
    /// Convert the join style to the GPU enum value.
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
    /// Butt cap.
    Butt,
    /// Square cap.
    Square,
    /// Round cap.
    Round,
}

impl StrokeCap {
    /// Convert the cap style to the GPU enum value.
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
    /// Normalized offset in [0, 1].
    pub offset: f32,
    /// Stop color in linear space.
    pub color: Color,
}

/// Linear gradient definition in canvas space.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearGradient {
    /// Gradient start position in canvas space.
    pub start: Vec2,
    /// Gradient end position in canvas space.
    pub end: Vec2,
    /// Ordered list of gradient stops.
    pub stops: Vec<GradientStop>,
}

/// Radial gradient definition in canvas space.
#[derive(Debug, Clone, PartialEq)]
pub struct RadialGradient {
    /// Gradient center in canvas space.
    pub center: Vec2,
    /// Gradient radius in canvas space (x/y for elliptical gradients).
    pub radius: Vec2,
    /// Ordered list of gradient stops.
    pub stops: Vec<GradientStop>,
}

/// Paint definition for fills and strokes.
#[derive(Debug, Clone, PartialEq)]
pub enum Paint {
    /// Solid color paint.
    Solid(Color),
    /// Linear gradient paint.
    LinearGradient(LinearGradient),
    /// Radial gradient paint.
    RadialGradient(RadialGradient),
}

/// Shape geometry variants supported by diffvg-rs.
#[derive(Debug, Clone, PartialEq)]
pub enum ShapeGeometry {
    /// Circle defined by center and radius.
    Circle {
        /// Center in local space.
        center: Vec2,
        /// Radius in local space.
        radius: f32,
    },
    /// Ellipse defined by center and radii.
    Ellipse {
        /// Center in local space.
        center: Vec2,
        /// Radii in local space (x/y).
        radius: Vec2,
    },
    /// Axis-aligned rectangle defined by minimum and maximum corners.
    Rect {
        /// Minimum corner in local space.
        min: Vec2,
        /// Maximum corner in local space.
        max: Vec2,
    },
    /// Path geometry using diffvg control point encoding.
    Path {
        /// Path data in local space.
        path: Path,
    },
}

/// Renderable shape with local transform and stroke settings.
#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    /// Shape geometry stored in local space.
    pub geometry: ShapeGeometry,
    /// Local transform from shape space to group space.
    pub transform: Mat3,
    /// Constant stroke radius used when no per-point thickness is provided.
    pub stroke_width: f32,
    /// Join style for stroked paths.
    pub stroke_join: StrokeJoin,
    /// Cap style for open paths.
    pub stroke_cap: StrokeCap,
    /// Miter limit for miter joins.
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
    /// Indices into `Scene::shapes` defining draw order for this group.
    pub shape_indices: Vec<usize>,
    /// Fill paint for the group.
    pub fill: Option<Paint>,
    /// Stroke paint for the group.
    pub stroke: Option<Paint>,
    /// Opacity multiplier applied to the group.
    pub opacity: f32,
    /// Fill rule for path winding evaluation.
    pub fill_rule: FillRule,
    /// Transform from group space to canvas space.
    pub shape_to_canvas: Mat3,
    /// Inverse transform from canvas space to group space.
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
    /// Scene width in pixels.
    pub width: u32,
    /// Scene height in pixels.
    pub height: u32,
    /// All shapes referenced by groups.
    pub shapes: Vec<Shape>,
    /// Ordered shape groups to render.
    pub groups: Vec<ShapeGroup>,
    /// Background color.
    pub background: Color,
    /// Optional RGBA background image (width * height * 4).
    pub background_image: Option<Vec<f32>>,
    /// Reconstruction filter configuration.
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
