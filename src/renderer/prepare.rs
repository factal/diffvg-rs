//! Scene preprocessing and GPU buffer packing.

use crate::color::Color;
use crate::geometry::StrokeSegment;
use crate::math::{Mat3, Vec2};
use crate::scene::{Paint, Scene, Shape, ShapeGeometry, StrokeJoin};
use crate::path_utils::{bounds_from_points, rect_corners};

use super::bvh::{append_group_bvh, append_path_bvh, Bounds};
use super::constants::{
    CURVE_STRIDE, GRADIENT_STRIDE, GROUP_STRIDE, MAX_F32_INT, SEGMENT_STRIDE, SHAPE_STRIDE,
};
use super::path::{
    curve_segments_to_f32, path_to_curve_segments, push_curve_segments, segments_to_f32,
    CurveSegment,
};
use super::types::{RenderError, RenderOptions};

/// GPU-ready buffers and metadata produced by scene preprocessing.
pub(crate) struct PreparedScene {
    /// Packed per-shape metadata (kind, offsets, stroke params, curve info).
    pub(crate) shape_data: Vec<f32>,
    /// Packed flattened stroke segments for path rasterization.
    pub(crate) segment_data: Vec<f32>,
    /// Packed curve segments for distance evaluation.
    pub(crate) curve_data: Vec<f32>,
    /// Per-shape bounds in group-local space, `[min.x, min.y, max.x, max.y]`.
    pub(crate) shape_bounds: Vec<f32>,
    /// Per-group bounds in group-local space, `[min.x, min.y, max.x, max.y]`.
    pub(crate) group_bounds: Vec<f32>,
    /// Packed group metadata (shape range, paint, fill rule, colors).
    pub(crate) group_data: Vec<f32>,
    /// Packed `canvas_to_shape` transforms (2x3) per group.
    pub(crate) group_xform: Vec<f32>,
    /// Packed `shape_to_canvas` transforms (2x3) per group.
    pub(crate) group_shape_xform: Vec<f32>,
    /// Inverse average scale per group for shape-space AA.
    pub(crate) group_inv_scale: Vec<f32>,
    /// Flattened shape indices per group stored as `f32`.
    pub(crate) group_shapes: Vec<f32>,
    /// Packed inverse per-shape transforms (2x3).
    pub(crate) shape_xform: Vec<f32>,
    /// BVH node bounds for group-local shapes.
    pub(crate) group_bvh_bounds: Vec<f32>,
    /// BVH node data for group-local shapes.
    pub(crate) group_bvh_nodes: Vec<u32>,
    /// BVH index array mapping nodes to shape indices.
    pub(crate) group_bvh_indices: Vec<u32>,
    /// BVH metadata per group (node/index offsets and counts).
    pub(crate) group_bvh_meta: Vec<u32>,
    /// BVH node bounds for path curve segments.
    pub(crate) path_bvh_bounds: Vec<f32>,
    /// BVH node data for path curve segments.
    pub(crate) path_bvh_nodes: Vec<u32>,
    /// BVH index array mapping nodes to curve segment indices.
    pub(crate) path_bvh_indices: Vec<u32>,
    /// BVH metadata per path (node/index offsets and counts).
    pub(crate) path_bvh_meta: Vec<u32>,
    /// Packed gradient headers.
    pub(crate) gradient_data: Vec<f32>,
    /// Gradient stop offsets.
    pub(crate) stop_offsets: Vec<f32>,
    /// Gradient stop colors in RGBA.
    pub(crate) stop_colors: Vec<f32>,
    /// Number of groups in the scene.
    pub(crate) num_groups: u32,
}

#[derive(Debug)]
/// Per-shape packed data produced during preprocessing.
pub(crate) struct PreparedShape {
    /// Shape kind identifier expected by GPU kernels.
    pub(crate) kind: u32,
    /// Offset into the flattened stroke segment buffer.
    pub(crate) seg_offset: u32,
    /// Number of flattened stroke segments.
    pub(crate) seg_count: u32,
    /// Offset into the curve segment buffer.
    pub(crate) curve_offset: u32,
    /// Number of curve segments.
    pub(crate) curve_count: u32,
    /// Stroke width in shape space.
    pub(crate) stroke_width: f32,
    /// Packed shape params plus stroke metadata.
    pub(crate) params: [f32; 8],
    /// Whether distance approximation should be used.
    pub(crate) use_distance_approx: bool,
    /// Axis-aligned bounds in shape space, if any.
    pub(crate) bounds: Option<(Vec2, Vec2)>,
    /// Maximum radius used for stroke padding.
    pub(crate) max_stroke_radius: f32,
}

#[derive(Debug, Copy, Clone)]
/// Packed paint metadata for a group fill or stroke.
struct PaintPack {
    /// Paint kind: 0 = none, 1 = solid, 2 = linear, 3 = radial.
    kind: u32,
    /// Index into the gradient buffer when applicable.
    gradient_index: u32,
    /// Solid color value or fallback.
    color: Color,
}

// Encode stroke-specific metadata into the packed shape params.
fn apply_stroke_meta(params: &mut [f32; 8], shape: &Shape, has_thickness: bool) {
    params[4] = if has_thickness { 1.0 } else { 0.0 };
    params[5] = shape.stroke_join.as_u32() as f32;
    params[6] = shape.stroke_cap.as_u32() as f32;
    params[7] = shape.stroke_miter_limit;
}

/// Preprocess a scene into GPU-ready buffers and BVH metadata.
pub(crate) fn prepare_scene(scene: &Scene, options: &RenderOptions) -> Result<PreparedScene, RenderError> {
    let mut segments = Vec::new();
    let mut curves = Vec::new();
    let mut shape_data = Vec::new();
    let mut shape_bounds = Vec::new();
    let mut group_bounds = Vec::new();
    let mut group_data = Vec::with_capacity(scene.groups.len() * GROUP_STRIDE);
    let mut group_xform = Vec::with_capacity(scene.groups.len() * 6);
    let mut group_shape_xform = Vec::with_capacity(scene.groups.len() * 6);
    let mut group_inv_scale = Vec::with_capacity(scene.groups.len());
    let mut group_shapes = Vec::new();
    let mut shape_xform = Vec::new();
    let mut group_bvh_bounds = Vec::new();
    let mut group_bvh_nodes = Vec::new();
    let mut group_bvh_indices = Vec::new();
    let mut group_bvh_meta = Vec::new();
    let mut path_bvh_bounds = Vec::new();
    let mut path_bvh_nodes = Vec::new();
    let mut path_bvh_indices = Vec::new();
    let mut path_bvh_meta = Vec::new();
    let mut gradient_data = Vec::new();
    let mut stop_offsets = Vec::new();
    let mut stop_colors = Vec::new();
    let mut shape_bounds_list: Vec<Option<Bounds>> = Vec::new();

    if scene.shapes.len() > MAX_F32_INT {
        return Err(RenderError::InvalidScene("too many shapes for f32 indexing"));
    }

    for group in scene.groups.iter() {
        let shape_offset = group_shapes.len() as u32;
        let mut shape_count = 0u32;
        let group_scale = group.shape_to_canvas.average_scale().max(1.0e-6);
        let inv_scale = 1.0 / group_scale;
        let base_tol = options.path_tolerance.max(0.01);
        // Prefiltering uses AA in shape space, so scale tolerance by transforms.
        let aa_local = if options.use_prefiltering { inv_scale } else { 0.0 };
        let mut group_shape_indices = Vec::new();
        let mut group_bounds_acc = Bounds::empty();
        let mut group_bounds_valid = false;

        let xform = group.canvas_to_shape;
        group_xform.extend_from_slice(&[
            xform.m[0][0],
            xform.m[0][1],
            xform.m[0][2],
            xform.m[1][0],
            xform.m[1][1],
            xform.m[1][2],
        ]);
        let group_shape = group.shape_to_canvas;
        group_shape_xform.extend_from_slice(&[
            group_shape.m[0][0],
            group_shape.m[0][1],
            group_shape.m[0][2],
            group_shape.m[1][0],
            group_shape.m[1][1],
            group_shape.m[1][2],
        ]);
        group_inv_scale.push(inv_scale);

        for &shape_idx in &group.shape_indices {
            if shape_idx >= scene.shapes.len() {
                return Err(RenderError::InvalidScene("group references invalid shape index"));
            }

            let shape = &scene.shapes[shape_idx];
            let total_scale = (group_scale * shape.transform.average_scale()).max(1.0e-6);
            let shape_tolerance = (base_tol / total_scale).max(1.0e-3);
            // Flatten curves in shape space and pack GPU-ready geometry buffers.
            let prepared = prepare_shape(shape, &mut segments, &mut curves, shape_tolerance);
            let shape_inv = shape.transform.inverse().unwrap_or(Mat3::identity());
            shape_xform.extend_from_slice(&[
                shape_inv.m[0][0],
                shape_inv.m[0][1],
                shape_inv.m[0][2],
                shape_inv.m[1][0],
                shape_inv.m[1][1],
                shape_inv.m[1][2],
            ]);
            let shape_index = (shape_data.len() / SHAPE_STRIDE) as u32;
            group_shapes.push(shape_index as f32);
            group_shape_indices.push(shape_index);

            shape_data.push(prepared.kind as f32);
            shape_data.push(prepared.seg_offset as f32);
            shape_data.push(prepared.seg_count as f32);
            shape_data.push(prepared.stroke_width);
            shape_data.extend_from_slice(&prepared.params);
            shape_data.push(prepared.curve_offset as f32);
            shape_data.push(prepared.curve_count as f32);
            shape_data.push(if prepared.use_distance_approx { 1.0 } else { 0.0 });

            let mut stroke_pad = prepared.max_stroke_radius.abs() * shape.transform.max_scale().max(0.0);
            if shape.stroke_join == StrokeJoin::Miter {
                stroke_pad *= shape.stroke_miter_limit.max(1.0);
            }
            let pad = aa_local + stroke_pad;
            let bounds_group = prepared
                .bounds
                .map(|bounds| transform_bounds(bounds, shape.transform));
            let bounds_local = inflate_bounds(bounds_group, pad);
            if let Some(bounds) = bounds_local {
                shape_bounds.extend_from_slice(&[bounds.0.x, bounds.0.y, bounds.1.x, bounds.1.y]);
                shape_bounds_list.push(Some(Bounds {
                    min: bounds.0,
                    max: bounds.1,
                }));
                if !group_bounds_valid {
                    group_bounds_acc = Bounds {
                        min: bounds.0,
                        max: bounds.1,
                    };
                    group_bounds_valid = true;
                } else {
                    group_bounds_acc.include(Bounds {
                        min: bounds.0,
                        max: bounds.1,
                    });
                }
            } else {
                shape_bounds.extend_from_slice(&[1.0, 1.0, 0.0, 0.0]);
                shape_bounds_list.push(None);
            }

            let path_meta = if let ShapeGeometry::Path { path } = &shape.geometry {
                let curve_segments = path_to_curve_segments(path, shape.stroke_width);
                append_path_bvh(
                    &curve_segments,
                    prepared.curve_offset,
                    &mut path_bvh_bounds,
                    &mut path_bvh_nodes,
                    &mut path_bvh_indices,
                )
            } else {
                [0u32; 4]
            };
            path_bvh_meta.extend_from_slice(&path_meta);

            shape_count += 1;
        }

        if group_shapes.len() > MAX_F32_INT {
            return Err(RenderError::InvalidScene(
                "too many group shape indices for f32 indexing",
            ));
        }

        let fill_pack = pack_paint(
            group.fill.as_ref(),
            &mut gradient_data,
            &mut stop_offsets,
            &mut stop_colors,
        );
        let stroke_pack = pack_paint(
            group.stroke.as_ref(),
            &mut gradient_data,
            &mut stop_offsets,
            &mut stop_colors,
        );

        let group_bounds_entry = if group_bounds_valid {
            group_bounds_acc
        } else {
            Bounds::empty()
        };
        if group_bounds_valid {
            group_bounds.extend_from_slice(&[
                group_bounds_entry.min.x,
                group_bounds_entry.min.y,
                group_bounds_entry.max.x,
                group_bounds_entry.max.y,
            ]);
        } else {
            group_bounds.extend_from_slice(&[1.0, 1.0, 0.0, 0.0]);
        }
        // Build a per-group BVH for tile binning and inside tests.
        let group_bvh = append_group_bvh(
            &group_shape_indices,
            &shape_bounds_list,
            &mut group_bvh_bounds,
            &mut group_bvh_nodes,
            &mut group_bvh_indices,
        );
        group_bvh_meta.extend_from_slice(&group_bvh);

        group_data.push(shape_offset as f32);
        group_data.push(shape_count as f32);
        group_data.push(fill_pack.kind as f32);
        group_data.push(fill_pack.gradient_index as f32);
        group_data.push(stroke_pack.kind as f32);
        group_data.push(stroke_pack.gradient_index as f32);
        group_data.push(1.0);
        group_data.push(group.fill_rule.as_u32() as f32);
        group_data.push(fill_pack.color.r);
        group_data.push(fill_pack.color.g);
        group_data.push(fill_pack.color.b);
        group_data.push(fill_pack.color.a);
        group_data.push(stroke_pack.color.r);
        group_data.push(stroke_pack.color.g);
        group_data.push(stroke_pack.color.b);
        group_data.push(stroke_pack.color.a);
    }

    let segment_data = segments_to_f32(&segments);
    let curve_data = curve_segments_to_f32(&curves);

    if segment_data.len() / SEGMENT_STRIDE > MAX_F32_INT {
        return Err(RenderError::InvalidScene(
            "too many segments for f32 indexing",
        ));
    }
    if curve_data.len() / CURVE_STRIDE > MAX_F32_INT {
        return Err(RenderError::InvalidScene(
            "too many curve segments for f32 indexing",
        ));
    }

    Ok(PreparedScene {
        shape_data,
        segment_data,
        curve_data,
        shape_bounds,
        group_bounds,
        group_data,
        group_xform,
        group_shape_xform,
        group_inv_scale,
        group_shapes,
        shape_xform,
        group_bvh_bounds,
        group_bvh_nodes,
        group_bvh_indices,
        group_bvh_meta,
        path_bvh_bounds,
        path_bvh_nodes,
        path_bvh_indices,
        path_bvh_meta,
        gradient_data,
        stop_offsets,
        stop_colors,
        num_groups: scene.groups.len() as u32,
    })
}

fn pack_paint(
    paint: Option<&Paint>,
    gradient_data: &mut Vec<f32>,
    stop_offsets: &mut Vec<f32>,
    stop_colors: &mut Vec<f32>,
) -> PaintPack {
    match paint {
        None => PaintPack {
            kind: 0,
            gradient_index: 0,
            color: Color::TRANSPARENT,
        },
        Some(Paint::Solid(color)) => PaintPack {
            kind: 1,
            gradient_index: 0,
            color: *color,
        },
        Some(Paint::LinearGradient(gradient)) => {
            let index = push_linear_gradient(gradient_data, stop_offsets, stop_colors, gradient);
            PaintPack {
                kind: 2,
                gradient_index: index,
                color: Color::TRANSPARENT,
            }
        }
        Some(Paint::RadialGradient(gradient)) => {
            let index = push_radial_gradient(gradient_data, stop_offsets, stop_colors, gradient);
            PaintPack {
                kind: 3,
                gradient_index: index,
                color: Color::TRANSPARENT,
            }
        }
    }
}

fn push_linear_gradient(
    gradient_data: &mut Vec<f32>,
    stop_offsets: &mut Vec<f32>,
    stop_colors: &mut Vec<f32>,
    gradient: &crate::scene::LinearGradient,
) -> u32 {
    let start = gradient.start;
    let end = gradient.end;

    let stop_offset = stop_offsets.len() as u32;
    let stop_count = gradient.stops.len() as u32;

    for stop in &gradient.stops {
        stop_offsets.push(stop.offset);
        stop_colors.push(stop.color.r);
        stop_colors.push(stop.color.g);
        stop_colors.push(stop.color.b);
        stop_colors.push(stop.color.a);
    }

    let index = (gradient_data.len() / GRADIENT_STRIDE) as u32;
    gradient_data.push(0.0);
    gradient_data.push(start.x);
    gradient_data.push(start.y);
    gradient_data.push(end.x);
    gradient_data.push(end.y);
    gradient_data.push(stop_offset as f32);
    gradient_data.push(stop_count as f32);
    gradient_data.push(0.0);
    index
}

fn push_radial_gradient(
    gradient_data: &mut Vec<f32>,
    stop_offsets: &mut Vec<f32>,
    stop_colors: &mut Vec<f32>,
    gradient: &crate::scene::RadialGradient,
) -> u32 {
    let center = gradient.center;
    let radius = gradient.radius;

    let stop_offset = stop_offsets.len() as u32;
    let stop_count = gradient.stops.len() as u32;

    for stop in &gradient.stops {
        stop_offsets.push(stop.offset);
        stop_colors.push(stop.color.r);
        stop_colors.push(stop.color.g);
        stop_colors.push(stop.color.b);
        stop_colors.push(stop.color.a);
    }

    let index = (gradient_data.len() / GRADIENT_STRIDE) as u32;
    gradient_data.push(1.0);
    gradient_data.push(center.x);
    gradient_data.push(center.y);
    gradient_data.push(radius.x);
    gradient_data.push(radius.y);
    gradient_data.push(stop_offset as f32);
    gradient_data.push(stop_count as f32);
    gradient_data.push(0.0);
    index
}

fn prepare_shape(
    shape: &Shape,
    segments: &mut Vec<StrokeSegment>,
    curves: &mut Vec<CurveSegment>,
    tolerance: f32,
) -> PreparedShape {
    let stroke_width = shape.stroke_width;
    let mut max_stroke_radius = stroke_width.abs();

    match &shape.geometry {
        ShapeGeometry::Circle { center, radius } => {
            let r = radius.abs();
            let mut params = [center.x, center.y, r, 0.0, 0.0, 0.0, 0.0, 0.0];
            apply_stroke_meta(&mut params, shape, false);
            let bounds = Some((
                Vec2::new(center.x - r, center.y - r),
                Vec2::new(center.x + r, center.y + r),
            ));
            PreparedShape {
                kind: 0,
                seg_offset: 0,
                seg_count: 0,
                curve_offset: 0,
                curve_count: 0,
                stroke_width,
                params,
                use_distance_approx: false,
                bounds,
                max_stroke_radius,
            }
        }
        ShapeGeometry::Ellipse { center, radius } => {
            let rx = radius.x.abs();
            let ry = radius.y.abs();
            let mut params = [center.x, center.y, rx, ry, 0.0, 0.0, 0.0, 0.0];
            apply_stroke_meta(&mut params, shape, false);
            let bounds = Some((
                Vec2::new(center.x - rx, center.y - ry),
                Vec2::new(center.x + rx, center.y + ry),
            ));
            PreparedShape {
                kind: 3,
                seg_offset: 0,
                seg_count: 0,
                curve_offset: 0,
                curve_count: 0,
                stroke_width,
                params,
                use_distance_approx: false,
                bounds,
                max_stroke_radius,
            }
        }
        ShapeGeometry::Rect { min, max } => {
            let mut params = [min.x, min.y, max.x, max.y, 0.0, 0.0, 0.0, 0.0];
            apply_stroke_meta(&mut params, shape, false);
            PreparedShape {
                kind: 1,
                seg_offset: 0,
                seg_count: 0,
                curve_offset: 0,
                curve_count: 0,
                stroke_width,
                params,
                use_distance_approx: false,
                bounds: Some((*min, *max)),
                max_stroke_radius,
            }
        }
        ShapeGeometry::Path { path } => {
            let has_thickness = path.thickness.is_some();
            let path_tolerance = tolerance;
            let segs = path.flatten_with_thickness(path_tolerance, stroke_width, 1.0);
            for seg in &segs {
                max_stroke_radius = max_stroke_radius.max(seg.r0.abs()).max(seg.r1.abs());
            }
            let (seg_offset, seg_count, bounds) = push_segments(segments, &segs);
            let path_bounds = if path.points.is_empty() {
                bounds
            } else {
                Some(bounds_from_points(&path.points))
            };
            let (curve_offset, curve_count) = {
                let curve_segments = path_to_curve_segments(path, stroke_width);
                push_curve_segments(curves, &curve_segments)
            };
            let mut params = bounds_to_params(path_bounds);
            apply_stroke_meta(&mut params, shape, has_thickness);
            PreparedShape {
                kind: 2,
                seg_offset,
                seg_count,
                curve_offset,
                curve_count,
                stroke_width,
                params,
                use_distance_approx: path.use_distance_approx,
                bounds: path_bounds,
                max_stroke_radius,
            }
        }
    }
}

fn push_segments(
    segments: &mut Vec<StrokeSegment>,
    new_segments: &[StrokeSegment],
) -> (u32, u32, Option<(Vec2, Vec2)>) {
    let seg_offset = segments.len() as u32;
    let mut bounds: Option<(Vec2, Vec2)> = None;
    for seg in new_segments {
        bounds = Some(match bounds {
            None => (seg.start.min(seg.end), seg.start.max(seg.end)),
            Some((min, max)) => (min.min(seg.start).min(seg.end), max.max(seg.start).max(seg.end)),
        });
    }
    segments.extend_from_slice(new_segments);
    let seg_count = new_segments.len() as u32;
    (seg_offset, seg_count, bounds)
}

fn bounds_to_params(bounds: Option<(Vec2, Vec2)>) -> [f32; 8] {
    if let Some((min, max)) = bounds {
        [min.x, min.y, max.x, max.y, 0.0, 0.0, 0.0, 0.0]
    } else {
        [0.0; 8]
    }
}

fn transform_bounds(bounds: (Vec2, Vec2), transform: Mat3) -> (Vec2, Vec2) {
    let corners = rect_corners(bounds.0, bounds.1)
        .into_iter()
        .map(|p| transform.transform_point(p))
        .collect::<Vec<_>>();
    bounds_from_points(&corners)
}

fn inflate_bounds(bounds: Option<(Vec2, Vec2)>, pad: f32) -> Option<(Vec2, Vec2)> {
    bounds.map(|(min, max)| {
        let pad = pad.max(0.0);
        (
            Vec2::new(min.x - pad, min.y - pad),
            Vec2::new(max.x + pad, max.y + pad),
        )
    })
}
