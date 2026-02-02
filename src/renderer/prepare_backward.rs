//! Backward-specific scene preprocessing for GPU gradients.

use crate::scene::{Scene, ShapeGeometry};

use super::prepare::{prepare_scene, PreparedScene};
use super::types::{RenderError, RenderOptions};

/// GPU-ready buffers for backward rendering, including path metadata.
///
/// Per-shape arrays are indexed by the `Scene::shapes` order; entries for
/// non-path shapes are zeroed.
pub(crate) struct PreparedBackwardScene {
    /// Forward-prepared buffers shared with the renderer.
    pub(crate) prepared: PreparedScene,
    /// Packed per-shape transforms (2x3) from shape space to group space.
    ///
    /// The layout is row-major: `[m00, m01, m02, m10, m11, m12]`.
    pub(crate) shape_transform: Vec<f32>,
    /// Flattened path control points (`x, y` pairs) in shape-local space.
    pub(crate) path_points: Vec<f32>,
    /// Per-segment control counts (0=line, 1=quad, 2=cubic) in diffvg order.
    pub(crate) path_num_control_points: Vec<u32>,
    /// Optional per-point thickness (radii) aligned to `path_points`.
    pub(crate) path_thickness: Vec<f32>,
    /// Per-shape offsets into `path_points`, in point units.
    pub(crate) shape_path_offsets: Vec<u32>,
    /// Per-shape point counts for paths.
    pub(crate) shape_path_point_counts: Vec<u32>,
    /// Per-shape offsets into `path_num_control_points`, in segment units.
    pub(crate) shape_path_ctrl_offsets: Vec<u32>,
    /// Per-shape control counts for paths.
    pub(crate) shape_path_ctrl_counts: Vec<u32>,
    /// Per-shape offsets into `path_thickness`, in point units.
    pub(crate) shape_path_thickness_offsets: Vec<u32>,
    /// Per-shape thickness counts for paths.
    pub(crate) shape_path_thickness_counts: Vec<u32>,
    /// Per-shape closed flags (1 = closed, 0 = open).
    pub(crate) shape_path_is_closed: Vec<u32>,
}

/// Preprocess a scene for backward rendering and gradient evaluation.
///
/// This reuses forward preprocessing and additionally flattens path data
/// needed by the backward GPU kernels.
pub(crate) fn prepare_scene_backward(
    scene: &Scene,
    options: &RenderOptions,
) -> Result<PreparedBackwardScene, RenderError> {
    let prepared = prepare_scene(scene, options)?;

    let mut shape_transform = Vec::with_capacity(scene.shapes.len() * 6);
    let mut path_points: Vec<f32> = Vec::new();
    let mut path_num_control_points: Vec<u32> = Vec::new();
    let mut path_thickness: Vec<f32> = Vec::new();

    let mut shape_path_offsets = vec![0u32; scene.shapes.len()];
    let mut shape_path_point_counts = vec![0u32; scene.shapes.len()];
    let mut shape_path_ctrl_offsets = vec![0u32; scene.shapes.len()];
    let mut shape_path_ctrl_counts = vec![0u32; scene.shapes.len()];
    let mut shape_path_thickness_offsets = vec![0u32; scene.shapes.len()];
    let mut shape_path_thickness_counts = vec![0u32; scene.shapes.len()];
    let mut shape_path_is_closed = vec![0u32; scene.shapes.len()];

    for (shape_idx, shape) in scene.shapes.iter().enumerate() {
        let t = shape.transform;
        shape_transform.extend_from_slice(&[
            t.m[0][0],
            t.m[0][1],
            t.m[0][2],
            t.m[1][0],
            t.m[1][1],
            t.m[1][2],
        ]);

        if let ShapeGeometry::Path { path } = &shape.geometry {
            shape_path_offsets[shape_idx] = (path_points.len() / 2) as u32;
            shape_path_point_counts[shape_idx] = path.points.len() as u32;
            shape_path_ctrl_offsets[shape_idx] = path_num_control_points.len() as u32;
            shape_path_ctrl_counts[shape_idx] = path.num_control_points.len() as u32;
            shape_path_thickness_offsets[shape_idx] = path_thickness.len() as u32;
            shape_path_thickness_counts[shape_idx] = path
                .thickness
                .as_ref()
                .map(|vals| vals.len() as u32)
                .unwrap_or(0);
            shape_path_is_closed[shape_idx] = if path.is_closed { 1 } else { 0 };

            for pt in &path.points {
                path_points.push(pt.x);
                path_points.push(pt.y);
            }
            for &count in &path.num_control_points {
                path_num_control_points.push(count as u32);
            }
            if let Some(thickness) = path.thickness.as_ref() {
                path_thickness.extend_from_slice(thickness);
            }
        }
    }

    Ok(PreparedBackwardScene {
        prepared,
        shape_transform,
        path_points,
        path_num_control_points,
        path_thickness,
        shape_path_offsets,
        shape_path_point_counts,
        shape_path_ctrl_offsets,
        shape_path_ctrl_counts,
        shape_path_thickness_offsets,
        shape_path_thickness_counts,
        shape_path_is_closed,
    })
}
