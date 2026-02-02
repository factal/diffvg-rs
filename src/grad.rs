//! Gradient data structures for differentiable rendering.

use crate::color::Color;
use crate::math::{Mat3, Vec2};
use crate::scene::{Paint, Scene, Shape, ShapeGeometry};

/// Gradient stop gradient (offset + RGBA).
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct DGradientStop {
    /// Gradient with respect to the normalized stop offset.
    pub offset: f32,
    /// Gradient with respect to the stop color in linear RGBA.
    pub color: Color,
}

/// Linear gradient adjoints in canvas space.
#[derive(Debug, Clone, PartialEq)]
pub struct DLinearGradient {
    /// Gradient with respect to the start point (canvas space).
    pub start: Vec2,
    /// Gradient with respect to the end point (canvas space).
    pub end: Vec2,
    /// Gradients for each stop, matching the original stop order.
    pub stops: Vec<DGradientStop>,
}

/// Radial gradient adjoints in canvas space.
#[derive(Debug, Clone, PartialEq)]
pub struct DRadialGradient {
    /// Gradient with respect to the radial center (canvas space).
    pub center: Vec2,
    /// Gradient with respect to the radial radii (canvas space).
    pub radius: Vec2,
    /// Gradients for each stop, matching the original stop order.
    pub stops: Vec<DGradientStop>,
}

/// Paint adjoints for fills and strokes.
#[derive(Debug, Clone, PartialEq)]
pub enum DPaint {
    /// Solid color gradient in linear RGBA.
    Solid(Color),
    /// Linear gradient adjoints (canvas space).
    LinearGradient(DLinearGradient),
    /// Radial gradient adjoints (canvas space).
    RadialGradient(DRadialGradient),
}

/// Geometry adjoints per shape.
#[derive(Debug, Clone, PartialEq)]
pub enum DShapeGeometry {
    /// Circle geometry gradients.
    Circle {
        /// Gradient with respect to the circle center (shape local).
        center: Vec2,
        /// Gradient with respect to the circle radius.
        radius: f32,
    },
    /// Ellipse geometry gradients.
    Ellipse {
        /// Gradient with respect to the ellipse center (shape local).
        center: Vec2,
        /// Gradient with respect to the ellipse radii.
        radius: Vec2,
    },
    /// Rectangle geometry gradients.
    Rect {
        /// Gradient with respect to the rectangle min corner (shape local).
        min: Vec2,
        /// Gradient with respect to the rectangle max corner (shape local).
        max: Vec2,
    },
    /// Path geometry gradients.
    Path {
        /// Gradients for each path control point in order.
        points: Vec<Vec2>,
        /// Optional gradients for per-point stroke thickness.
        thickness: Option<Vec<f32>>,
    },
}

/// Shape adjoints, including per-shape transform gradients.
#[derive(Debug, Clone, PartialEq)]
pub struct DShape {
    /// Geometry parameter gradients for this shape.
    pub geometry: DShapeGeometry,
    /// Gradient with respect to the per-shape transform.
    pub transform: Mat3,
    /// Gradient with respect to the base stroke width.
    pub stroke_width: f32,
}

/// Shape group adjoints.
#[derive(Debug, Clone, PartialEq)]
pub struct DShapeGroup {
    /// Gradient with respect to the group shape-to-canvas transform.
    pub shape_to_canvas: Mat3,
    /// Optional fill paint gradients.
    pub fill: Option<DPaint>,
    /// Optional stroke paint gradients.
    pub stroke: Option<DPaint>,
}

/// Filter adjoints.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct DFilter {
    /// Gradient with respect to the filter radius.
    pub radius: f32,
}

/// Backward outputs mirroring the forward scene data.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneGrad {
    /// Per-shape gradients in scene order.
    pub shapes: Vec<DShape>,
    /// Per-shape-group gradients in scene order.
    pub shape_groups: Vec<DShapeGroup>,
    /// Reconstruction filter gradients.
    pub filter: DFilter,
    /// Gradient with respect to the constant background color.
    pub background: Color,
    /// Optional gradients for the background image (RGBA per pixel).
    pub background_image: Option<Vec<f32>>,
    /// Optional gradients for per-pixel translation (x, y per pixel).
    pub translation: Option<Vec<f32>>,
}

impl SceneGrad {
    /// Allocate zeroed gradients sized to match the provided scene.
    ///
    /// When `include_background_image` is true and the scene has a background image,
    /// the gradient buffer is sized to `width * height * 4` (RGBA). When
    /// `include_translation` is true, the translation buffer is sized to
    /// `width * height * 2` (x, y per pixel).
    pub fn zeros_from_scene(
        scene: &Scene,
        include_background_image: bool,
        include_translation: bool,
    ) -> Self {
        let shapes = scene
            .shapes
            .iter()
            .map(|shape| DShape {
                geometry: zero_shape_geometry(shape),
                transform: zero_mat3(),
                stroke_width: 0.0,
            })
            .collect();

        let shape_groups = scene
            .groups
            .iter()
            .map(|group| DShapeGroup {
                shape_to_canvas: zero_mat3(),
                fill: group.fill.as_ref().map(zero_paint),
                stroke: group.stroke.as_ref().map(zero_paint),
            })
            .collect();

        let background_image = if include_background_image {
            scene
                .background_image
                .as_ref()
                .map(|image| vec![0.0f32; image.len()])
        } else {
            None
        };

        let translation = if include_translation {
            let len = (scene.width as usize)
                .saturating_mul(scene.height as usize)
                .saturating_mul(2);
            Some(vec![0.0f32; len])
        } else {
            None
        };

        SceneGrad {
            shapes,
            shape_groups,
            filter: DFilter { radius: 0.0 },
            background: Color::TRANSPARENT,
            background_image,
            translation,
        }
    }

    /// Accumulate another gradient into this one (element-wise add).
    ///
    /// Variable-length buffers (path points, thickness, optional slices) are
    /// added up to the shorter length, with missing optional data cloned from
    /// `other` when needed.
    pub fn accumulate_from(&mut self, other: &SceneGrad) {
        for (dst_shape, src_shape) in self.shapes.iter_mut().zip(other.shapes.iter()) {
            dst_shape.stroke_width += src_shape.stroke_width;
            add_mat3(&mut dst_shape.transform, src_shape.transform);
            add_shape_geometry(&mut dst_shape.geometry, &src_shape.geometry);
        }

        for (dst_group, src_group) in self.shape_groups.iter_mut().zip(other.shape_groups.iter()) {
            add_mat3(&mut dst_group.shape_to_canvas, src_group.shape_to_canvas);
            add_paint_option(&mut dst_group.fill, &src_group.fill);
            add_paint_option(&mut dst_group.stroke, &src_group.stroke);
        }

        self.filter.radius += other.filter.radius;
        add_color_in_place(&mut self.background, other.background);

        add_slice_option(&mut self.background_image, &other.background_image);
        add_slice_option(&mut self.translation, &other.translation);
    }
}

fn zero_mat3() -> Mat3 {
    Mat3 { m: [[0.0; 3]; 3] }
}

fn zero_color() -> Color {
    Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    }
}

fn zero_shape_geometry(shape: &Shape) -> DShapeGeometry {
    match &shape.geometry {
        ShapeGeometry::Circle { .. } => DShapeGeometry::Circle {
            center: Vec2::ZERO,
            radius: 0.0,
        },
        ShapeGeometry::Ellipse { .. } => DShapeGeometry::Ellipse {
            center: Vec2::ZERO,
            radius: Vec2::ZERO,
        },
        ShapeGeometry::Rect { .. } => DShapeGeometry::Rect {
            min: Vec2::ZERO,
            max: Vec2::ZERO,
        },
        ShapeGeometry::Path { path } => DShapeGeometry::Path {
            points: vec![Vec2::ZERO; path.points.len()],
            thickness: path.thickness.as_ref().map(|vals| vec![0.0; vals.len()]),
        },
    }
}

fn zero_paint(paint: &Paint) -> DPaint {
    match paint {
        Paint::Solid(_) => DPaint::Solid(zero_color()),
        Paint::LinearGradient(gradient) => DPaint::LinearGradient(DLinearGradient {
            start: Vec2::ZERO,
            end: Vec2::ZERO,
            stops: gradient
                .stops
                .iter()
                .map(|_| DGradientStop {
                    offset: 0.0,
                    color: zero_color(),
                })
                .collect(),
        }),
        Paint::RadialGradient(gradient) => DPaint::RadialGradient(DRadialGradient {
            center: Vec2::ZERO,
            radius: Vec2::ZERO,
            stops: gradient
                .stops
                .iter()
                .map(|_| DGradientStop {
                    offset: 0.0,
                    color: zero_color(),
                })
                .collect(),
        }),
    }
}

fn add_mat3(dst: &mut Mat3, src: Mat3) {
    for r in 0..3 {
        for c in 0..3 {
            dst.m[r][c] += src.m[r][c];
        }
    }
}

fn add_vec2(dst: &mut Vec2, src: Vec2) {
    dst.x += src.x;
    dst.y += src.y;
}

fn add_color_in_place(dst: &mut Color, src: Color) {
    dst.r += src.r;
    dst.g += src.g;
    dst.b += src.b;
    dst.a += src.a;
}

fn add_shape_geometry(dst: &mut DShapeGeometry, src: &DShapeGeometry) {
    match (dst, src) {
        (DShapeGeometry::Circle { center, radius }, DShapeGeometry::Circle { center: src_center, radius: src_radius }) => {
            add_vec2(center, *src_center);
            *radius += *src_radius;
        }
        (DShapeGeometry::Ellipse { center, radius }, DShapeGeometry::Ellipse { center: src_center, radius: src_radius }) => {
            add_vec2(center, *src_center);
            add_vec2(radius, *src_radius);
        }
        (DShapeGeometry::Rect { min, max }, DShapeGeometry::Rect { min: src_min, max: src_max }) => {
            add_vec2(min, *src_min);
            add_vec2(max, *src_max);
        }
        (DShapeGeometry::Path { points, thickness }, DShapeGeometry::Path { points: src_points, thickness: src_thickness }) => {
            let count = points.len().min(src_points.len());
            for i in 0..count {
                points[i].x += src_points[i].x;
                points[i].y += src_points[i].y;
            }
            if let Some(dst_thickness) = thickness.as_mut() {
                if let Some(src_vals) = src_thickness.as_ref() {
                    let len = dst_thickness.len().min(src_vals.len());
                    for i in 0..len {
                        dst_thickness[i] += src_vals[i];
                    }
                }
            } else if let Some(src_vals) = src_thickness.as_ref() {
                *thickness = Some(src_vals.clone());
            }
        }
        _ => {}
    }
}

fn add_paint_option(dst: &mut Option<DPaint>, src: &Option<DPaint>) {
    if let Some(dst_paint) = dst.as_mut() {
        if let Some(src_paint) = src.as_ref() {
            add_paint(dst_paint, src_paint);
        }
    } else if let Some(src_paint) = src.as_ref() {
        *dst = Some(src_paint.clone());
    }
}

fn add_paint(dst: &mut DPaint, src: &DPaint) {
    match (dst, src) {
        (DPaint::Solid(dst_color), DPaint::Solid(src_color)) => {
            add_color_in_place(dst_color, *src_color);
        }
        (DPaint::LinearGradient(dst_grad), DPaint::LinearGradient(src_grad)) => {
            add_vec2(&mut dst_grad.start, src_grad.start);
            add_vec2(&mut dst_grad.end, src_grad.end);
            let count = dst_grad.stops.len().min(src_grad.stops.len());
            for i in 0..count {
                dst_grad.stops[i].offset += src_grad.stops[i].offset;
                add_color_in_place(&mut dst_grad.stops[i].color, src_grad.stops[i].color);
            }
        }
        (DPaint::RadialGradient(dst_grad), DPaint::RadialGradient(src_grad)) => {
            add_vec2(&mut dst_grad.center, src_grad.center);
            add_vec2(&mut dst_grad.radius, src_grad.radius);
            let count = dst_grad.stops.len().min(src_grad.stops.len());
            for i in 0..count {
                dst_grad.stops[i].offset += src_grad.stops[i].offset;
                add_color_in_place(&mut dst_grad.stops[i].color, src_grad.stops[i].color);
            }
        }
        _ => {}
    }
}

fn add_slice_option(dst: &mut Option<Vec<f32>>, src: &Option<Vec<f32>>) {
    if let Some(dst_vals) = dst.as_mut() {
        if let Some(src_vals) = src.as_ref() {
            let len = dst_vals.len().min(src_vals.len());
            for i in 0..len {
                dst_vals[i] += src_vals[i];
            }
        }
    } else if let Some(src_vals) = src.as_ref() {
        *dst = Some(src_vals.clone());
    }
}

/// diffvg-compatible path gradients with metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffvgPathGrad {
    /// Per-segment control point counts copied from the source path.
    pub num_control_points: Vec<u8>,
    /// Gradients for each path control point in order.
    pub points: Vec<Vec2>,
    /// Optional gradients for per-point stroke thickness.
    pub thickness: Option<Vec<f32>>,
    /// Whether the path is closed.
    pub is_closed: bool,
    /// Whether distance approximation was enabled for the path.
    pub use_distance_approx: bool,
}

/// diffvg-compatible shape geometry gradients.
#[derive(Debug, Clone, PartialEq)]
pub enum DiffvgShapeGeometry {
    /// Circle geometry gradients.
    Circle {
        /// Gradient with respect to the circle center (shape local).
        center: Vec2,
        /// Gradient with respect to the circle radius.
        radius: f32,
    },
    /// Ellipse geometry gradients.
    Ellipse {
        /// Gradient with respect to the ellipse center (shape local).
        center: Vec2,
        /// Gradient with respect to the ellipse radii.
        radius: Vec2,
    },
    /// Rectangle geometry gradients.
    Rect {
        /// Gradient with respect to the rectangle min corner (shape local).
        min: Vec2,
        /// Gradient with respect to the rectangle max corner (shape local).
        max: Vec2,
    },
    /// Path geometry gradients with diffvg metadata.
    Path {
        /// Path gradients and metadata in diffvg layout.
        path: DiffvgPathGrad,
    },
}

/// diffvg-compatible shape gradients (includes per-shape transform for diffvg-rs).
#[derive(Debug, Clone, PartialEq)]
pub struct DiffvgShapeGrad {
    /// Geometry gradients in diffvg layout.
    pub geometry: DiffvgShapeGeometry,
    /// Gradient with respect to the per-shape transform (diffvg-rs extension).
    pub transform: Mat3,
    /// Gradient with respect to the base stroke width.
    pub stroke_width: f32,
}

/// diffvg-compatible shape group gradients.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffvgShapeGroupGrad {
    /// Gradient with respect to the group shape-to-canvas transform.
    pub shape_to_canvas: Mat3,
    /// Optional fill paint gradients.
    pub fill: Option<DPaint>,
    /// Optional stroke paint gradients.
    pub stroke: Option<DPaint>,
}

/// diffvg-compatible scene gradient layout (for bindings/ABI parity).
#[derive(Debug, Clone, PartialEq)]
pub struct DiffvgSceneGrad {
    /// Per-shape gradients in diffvg layout.
    pub shapes: Vec<DiffvgShapeGrad>,
    /// Per-shape-group gradients in diffvg layout.
    pub shape_groups: Vec<DiffvgShapeGroupGrad>,
    /// Reconstruction filter gradients.
    pub filter: DFilter,
    /// Gradient with respect to the constant background color.
    pub background: Color,
    /// Optional gradients for the background image (RGBA per pixel).
    pub background_image: Option<Vec<f32>>,
    /// Optional gradients for per-pixel translation (x, y per pixel).
    pub translation: Option<Vec<f32>>,
}

impl SceneGrad {
    /// Convert gradients into a diffvg-compatible layout (with path metadata).
    ///
    /// Path metadata (control point counts, closure, and distance approximation)
    /// are copied from the provided scene, and per-point buffers are resized to
    /// match the source path lengths.
    pub fn to_diffvg_layout(&self, scene: &Scene) -> DiffvgSceneGrad {
        let mut shapes = Vec::with_capacity(self.shapes.len());
        for (idx, d_shape) in self.shapes.iter().enumerate() {
            let scene_shape = scene.shapes.get(idx);
            let geometry = match (&d_shape.geometry, scene_shape.map(|s| &s.geometry)) {
                (DShapeGeometry::Circle { center, radius }, _) => {
                    DiffvgShapeGeometry::Circle { center: *center, radius: *radius }
                }
                (DShapeGeometry::Ellipse { center, radius }, _) => {
                    DiffvgShapeGeometry::Ellipse { center: *center, radius: *radius }
                }
                (DShapeGeometry::Rect { min, max }, _) => {
                    DiffvgShapeGeometry::Rect { min: *min, max: *max }
                }
                (DShapeGeometry::Path { points, thickness }, Some(ShapeGeometry::Path { path })) => {
                    let mut out_points = vec![Vec2::ZERO; path.points.len()];
                    for (dst, src) in out_points.iter_mut().zip(points.iter()) {
                        *dst = *src;
                    }
                    let out_thickness = match (thickness, path.thickness.as_ref()) {
                        (Some(values), Some(src)) => {
                            let mut out = vec![0.0f32; src.len()];
                            for (dst, val) in out.iter_mut().zip(values.iter()) {
                                *dst = *val;
                            }
                            Some(out)
                        }
                        (Some(values), None) => Some(values.clone()),
                        (None, Some(src)) => Some(vec![0.0f32; src.len()]),
                        (None, None) => None,
                    };
                    DiffvgShapeGeometry::Path {
                        path: DiffvgPathGrad {
                            num_control_points: path.num_control_points.clone(),
                            points: out_points,
                            thickness: out_thickness,
                            is_closed: path.is_closed,
                            use_distance_approx: path.use_distance_approx,
                        },
                    }
                }
                (DShapeGeometry::Path { points, thickness }, _) => {
                    DiffvgShapeGeometry::Path {
                        path: DiffvgPathGrad {
                            num_control_points: Vec::new(),
                            points: points.clone(),
                            thickness: thickness.clone(),
                            is_closed: false,
                            use_distance_approx: false,
                        },
                    }
                }
            };

            shapes.push(DiffvgShapeGrad {
                geometry,
                transform: d_shape.transform,
                stroke_width: d_shape.stroke_width,
            });
        }

        let shape_groups = self
            .shape_groups
            .iter()
            .map(|group| DiffvgShapeGroupGrad {
                shape_to_canvas: group.shape_to_canvas,
                fill: group.fill.clone(),
                stroke: group.stroke.clone(),
            })
            .collect();

        DiffvgSceneGrad {
            shapes,
            shape_groups,
            filter: self.filter,
            background: self.background,
            background_image: self.background_image.clone(),
            translation: self.translation.clone(),
        }
    }
}
