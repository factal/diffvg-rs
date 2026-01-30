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

/// diffvg-compatible path gradients with metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffvgPathGrad {
    pub num_control_points: Vec<u8>,
    pub points: Vec<Vec2>,
    pub thickness: Option<Vec<f32>>,
    pub is_closed: bool,
    pub use_distance_approx: bool,
}

/// diffvg-compatible shape geometry gradients.
#[derive(Debug, Clone, PartialEq)]
pub enum DiffvgShapeGeometry {
    Circle { center: Vec2, radius: f32 },
    Ellipse { center: Vec2, radius: Vec2 },
    Rect { min: Vec2, max: Vec2 },
    Path { path: DiffvgPathGrad },
}

/// diffvg-compatible shape gradients (includes per-shape transform for diffvg-rs).
#[derive(Debug, Clone, PartialEq)]
pub struct DiffvgShapeGrad {
    pub geometry: DiffvgShapeGeometry,
    pub transform: Mat3,
    pub stroke_width: f32,
}

/// diffvg-compatible shape group gradients.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffvgShapeGroupGrad {
    pub shape_to_canvas: Mat3,
    pub fill: Option<DPaint>,
    pub stroke: Option<DPaint>,
}

/// diffvg-compatible scene gradient layout (for bindings/ABI parity).
#[derive(Debug, Clone, PartialEq)]
pub struct DiffvgSceneGrad {
    pub shapes: Vec<DiffvgShapeGrad>,
    pub shape_groups: Vec<DiffvgShapeGroupGrad>,
    pub filter: DFilter,
    pub background: Color,
    pub background_image: Option<Vec<f32>>,
    pub translation: Option<Vec<f32>>,
}

impl SceneGrad {
    /// Convert gradients into a diffvg-compatible layout (with path metadata).
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
