//! Gradient data structures for differentiable rendering.

use crate::color::Color;
use crate::math::{Mat3, Vec2};
use crate::scene::{Paint, Scene, Shape, ShapeGeometry};

/// Gradient stop gradient (offset + RGBA).
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct DGradientStop {
    pub offset: f32,
    pub color: Color,
}

/// Linear gradient adjoints in canvas space.
#[derive(Debug, Clone, PartialEq)]
pub struct DLinearGradient {
    pub start: Vec2,
    pub end: Vec2,
    pub stops: Vec<DGradientStop>,
}

/// Radial gradient adjoints in canvas space.
#[derive(Debug, Clone, PartialEq)]
pub struct DRadialGradient {
    pub center: Vec2,
    pub radius: Vec2,
    pub stops: Vec<DGradientStop>,
}

/// Paint adjoints for fills and strokes.
#[derive(Debug, Clone, PartialEq)]
pub enum DPaint {
    Solid(Color),
    LinearGradient(DLinearGradient),
    RadialGradient(DRadialGradient),
}

/// Geometry adjoints per shape.
#[derive(Debug, Clone, PartialEq)]
pub enum DShapeGeometry {
    Circle { center: Vec2, radius: f32 },
    Ellipse { center: Vec2, radius: Vec2 },
    Rect { min: Vec2, max: Vec2 },
    Path { points: Vec<Vec2>, thickness: Option<Vec<f32>> },
}

/// Shape adjoints, including per-shape transform gradients.
#[derive(Debug, Clone, PartialEq)]
pub struct DShape {
    pub geometry: DShapeGeometry,
    pub transform: Mat3,
    pub stroke_width: f32,
}

/// Shape group adjoints.
#[derive(Debug, Clone, PartialEq)]
pub struct DShapeGroup {
    pub shape_to_canvas: Mat3,
    pub fill: Option<DPaint>,
    pub stroke: Option<DPaint>,
}

/// Filter adjoints.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct DFilter {
    pub radius: f32,
}

/// Backward outputs mirroring the forward scene data.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneGrad {
    pub shapes: Vec<DShape>,
    pub shape_groups: Vec<DShapeGroup>,
    pub filter: DFilter,
    pub background: Color,
    pub background_image: Option<Vec<f32>>,
    pub translation: Option<Vec<f32>>,
}

impl SceneGrad {
    /// Allocate zeroed gradients matching the provided scene.
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
