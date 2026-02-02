use crate::distance::SceneBvh;
use crate::geometry::Path;
use crate::grad::{DShape, DShapeGeometry, DShapeGroup, SceneGrad};
use crate::math::{Mat3, Vec2, Vec4};
use crate::path_utils::{path_point, path_point_radius};
use crate::renderer::rng::Pcg32;
use crate::scene::{Scene, Shape, ShapeGeometry};

use super::background::sample_background;
use super::filters::gather_d_color;
use super::math::{
    d_xform_pt, dot4, mat3_add, mat3_mul, mat3_transpose, normalize, xform_normal, zero_mat3,
};
use super::sampling::sample_color;
use super::types::{
    BoundaryData, BoundarySample, EdgeQuery, PathBoundaryData, PathCdf, ShapeCdf,
};

pub(crate) struct BoundarySamplingData {
    pub(crate) shape_lengths: Vec<f32>,
    pub(crate) shape_cdf: Vec<f32>,
    pub(crate) shape_pmf: Vec<f32>,
    pub(crate) shape_ids: Vec<u32>,
    pub(crate) group_ids: Vec<u32>,
    pub(crate) path_cdf: Vec<f32>,
    pub(crate) path_pmf: Vec<f32>,
    pub(crate) path_point_ids: Vec<u32>,
    pub(crate) path_cdf_offsets: Vec<u32>,
    pub(crate) path_cdf_counts: Vec<u32>,
    pub(crate) path_point_offsets: Vec<u32>,
}

pub(crate) fn build_boundary_sampling_data(scene: &Scene) -> Option<BoundarySamplingData> {
    if scene.groups.is_empty() {
        return None;
    }

    let shape_lengths = compute_shape_lengths(scene);
    let shape_cdf = build_shape_cdf(scene, &shape_lengths)?;
    let path_cdfs = build_path_cdfs(scene, &shape_lengths);

    if shape_cdf.cdf.len() > u32::MAX as usize {
        return None;
    }

    let mut shape_ids = Vec::with_capacity(shape_cdf.shape_ids.len());
    for &id in &shape_cdf.shape_ids {
        shape_ids.push(u32::try_from(id).ok()?);
    }
    let mut group_ids = Vec::with_capacity(shape_cdf.group_ids.len());
    for &id in &shape_cdf.group_ids {
        group_ids.push(u32::try_from(id).ok()?);
    }

    let mut path_cdf = Vec::new();
    let mut path_pmf = Vec::new();
    let mut path_point_ids = Vec::new();
    let mut path_cdf_offsets = vec![0u32; scene.shapes.len()];
    let mut path_cdf_counts = vec![0u32; scene.shapes.len()];
    let mut path_point_offsets = vec![0u32; scene.shapes.len()];

    for (shape_id, cdf) in path_cdfs.iter().enumerate() {
        let Some(path_cdf_data) = cdf.as_ref() else {
            continue;
        };

        let cdf_offset = path_cdf.len();
        let point_offset = path_point_ids.len();
        if cdf_offset > u32::MAX as usize || point_offset > u32::MAX as usize {
            return None;
        }

        path_cdf_offsets[shape_id] = cdf_offset as u32;
        path_cdf_counts[shape_id] = u32::try_from(path_cdf_data.cdf.len()).ok()?;
        path_point_offsets[shape_id] = point_offset as u32;

        path_cdf.extend_from_slice(&path_cdf_data.cdf);
        path_pmf.extend_from_slice(&path_cdf_data.pmf);
        for &pid in &path_cdf_data.point_id_map {
            path_point_ids.push(u32::try_from(pid).ok()?);
        }
    }

    Some(BoundarySamplingData {
        shape_lengths,
        shape_cdf: shape_cdf.cdf,
        shape_pmf: shape_cdf.pmf,
        shape_ids,
        group_ids,
        path_cdf,
        path_pmf,
        path_point_ids,
        path_cdf_offsets,
        path_cdf_counts,
        path_point_offsets,
    })
}

/// Performs stochastic boundary sampling and accumulates edge gradients.
pub(crate) fn boundary_sampling(
    scene: &Scene,
    bvh: &SceneBvh,
    samples_x: u32,
    samples_y: u32,
    seed: u32,
    d_render_image: &[f32],
    weight_image: &[f32],
    grads: &mut SceneGrad,
    compute_translation: bool,
) {
    if scene.groups.is_empty() {
        return;
    }

    let shape_lengths = compute_shape_lengths(scene);
    let Some(shape_cdf) = build_shape_cdf(scene, &shape_lengths) else {
        return;
    };
    let path_cdfs = build_path_cdfs(scene, &shape_lengths);

    let width = scene.width as usize;
    let height = scene.height as usize;
    let total_samples = width
        .saturating_mul(height)
        .saturating_mul(samples_x as usize)
        .saturating_mul(samples_y as usize);

    for idx in 0..total_samples {
        let mut rng = Pcg32::new(idx as u64, seed as u64);
        let u = rng.next_f32();
        let t = rng.next_f32();
        let (sample_id, _) = sample_cdf(&shape_cdf.cdf, u);
        let Some(shape_id) = shape_cdf.shape_ids.get(sample_id).copied() else {
            continue;
        };
        let Some(group_id) = shape_cdf.group_ids.get(sample_id).copied() else {
            continue;
        };
        let shape_pmf = shape_cdf.pmf.get(sample_id).copied().unwrap_or(0.0);
        if shape_pmf <= 0.0 {
            continue;
        }
        let shape_length = shape_lengths.get(shape_id).copied().unwrap_or(0.0);
        let path_cdf = path_cdfs.get(shape_id).and_then(|v| v.as_ref());
        let Some((local_pt, normal, boundary_pdf, data)) =
            sample_boundary_point(scene, shape_id, group_id, t, shape_length, path_cdf)
        else {
            continue;
        };
        if boundary_pdf <= 0.0 {
            continue;
        }

        let group = &scene.groups[group_id];
        let shape = &scene.shapes[shape_id];
        let shape_to_canvas = group.shape_to_canvas.mul(shape.transform);
        let mut boundary_pt = shape_to_canvas.transform_point(local_pt);
        let shape_inv = shape.transform.inverse().unwrap_or(Mat3::identity());
        let composite_inv = shape_inv.mul(group.canvas_to_shape);
        let normal_canvas = xform_normal(composite_inv, normal);
        boundary_pt.x /= scene.width as f32;
        boundary_pt.y /= scene.height as f32;

        let sample = BoundarySample {
            pt: boundary_pt,
            local_pt,
            normal: normal_canvas,
            shape_group_id: group_id,
            shape_id,
            t,
            data,
            pdf: shape_pmf * boundary_pdf,
        };
        render_edge_sample(
            scene,
            bvh,
            d_render_image,
            weight_image,
            grads,
            compute_translation,
            sample,
        );
    }
}

/// Approximates per-shape boundary lengths for sampling.
fn compute_shape_lengths(scene: &crate::scene::Scene) -> Vec<f32> {
    let mut lengths = vec![0.0f32; scene.shapes.len()];
    for (shape_id, shape) in scene.shapes.iter().enumerate() {
        let length = match &shape.geometry {
            ShapeGeometry::Circle { radius, .. } => {
                2.0 * core::f32::consts::PI * radius.abs()
            }
            ShapeGeometry::Ellipse { radius, .. } => {
                let a = radius.x.abs();
                let b = radius.y.abs();
                if a == 0.0 || b == 0.0 {
                    0.0
                } else {
                    core::f32::consts::PI
                        * (3.0 * (a + b) - ((3.0 * a + b) * (a + 3.0 * b)).sqrt())
                }
            }
            ShapeGeometry::Rect { min, max } => {
                let w = (max.x - min.x).abs();
                let h = (max.y - min.y).abs();
                2.0 * (w + h)
            }
            ShapeGeometry::Path { path } => path_length(path),
        };
        lengths[shape_id] = length;
    }
    lengths
}

/// Estimates a path length by linearizing each segment.
fn path_length(path: &Path) -> f32 {
    let total_points = path.points.len();
    if total_points == 0 {
        return 0.0;
    }
    let mut length = 0.0f32;
    let mut point_id = 0usize;
    for &controls in &path.num_control_points {
        match controls {
            0 => {
                let i0 = point_id;
                let i1 = point_id + 1;
                let Some(p0) = path_point(path, i0, total_points) else {
                    break;
                };
                let Some(p1) = path_point(path, i1, total_points) else {
                    break;
                };
                length += (p1 - p0).length();
                point_id += 1;
            }
            1 => {
                let i0 = point_id;
                let i1 = point_id + 1;
                let i2 = point_id + 2;
                let Some(p0) = path_point(path, i0, total_points) else {
                    break;
                };
                let Some(p1) = path_point(path, i1, total_points) else {
                    break;
                };
                let Some(p2) = path_point(path, i2, total_points) else {
                    break;
                };
                let eval = |t: f32| {
                    let tt = 1.0 - t;
                    p0 * (tt * tt) + p1 * (2.0 * tt * t) + p2 * (t * t)
                };
                let v0 = p0;
                let v1 = eval(0.5);
                let v2 = p2;
                length += (v1 - v0).length() + (v2 - v1).length();
                point_id += 2;
            }
            2 => {
                let i0 = point_id;
                let i1 = point_id + 1;
                let i2 = point_id + 2;
                let i3 = point_id + 3;
                let Some(p0) = path_point(path, i0, total_points) else {
                    break;
                };
                let Some(p1) = path_point(path, i1, total_points) else {
                    break;
                };
                let Some(p2) = path_point(path, i2, total_points) else {
                    break;
                };
                let Some(p3) = path_point(path, i3, total_points) else {
                    break;
                };
                let eval = |t: f32| {
                    let tt = 1.0 - t;
                    p0 * (tt * tt * tt)
                        + p1 * (3.0 * tt * tt * t)
                        + p2 * (3.0 * tt * t * t)
                        + p3 * (t * t * t)
                };
                let v0 = p0;
                let v1 = eval(1.0 / 3.0);
                let v2 = eval(2.0 / 3.0);
                let v3 = p3;
                length += (v1 - v0).length() + (v2 - v1).length() + (v3 - v2).length();
                point_id += 3;
            }
            _ => break,
        }
    }
    length
}

/// Builds a CDF/PMF over shapes weighted by boundary length.
fn build_shape_cdf(
    scene: &crate::scene::Scene,
    shape_lengths: &[f32],
) -> Option<ShapeCdf> {
    let mut shape_ids = Vec::new();
    let mut group_ids = Vec::new();
    let mut cdf = Vec::new();
    let mut pmf = Vec::new();
    let mut accum = 0.0f32;
    for (group_id, group) in scene.groups.iter().enumerate() {
        for &shape_id in &group.shape_indices {
            let length = shape_lengths.get(shape_id).copied().unwrap_or(0.0);
            shape_ids.push(shape_id);
            group_ids.push(group_id);
            accum += length;
            cdf.push(accum);
            pmf.push(length);
        }
    }
    if accum <= 0.0 || !accum.is_finite() || cdf.is_empty() {
        return None;
    }
    for value in cdf.iter_mut() {
        *value /= accum;
    }
    for value in pmf.iter_mut() {
        *value /= accum;
    }
    Some(ShapeCdf {
        shape_ids,
        group_ids,
        cdf,
        pmf,
    })
}

/// Builds per-path segment CDFs for boundary sampling.
fn build_path_cdfs(
    scene: &crate::scene::Scene,
    shape_lengths: &[f32],
) -> Vec<Option<PathCdf>> {
    let mut out = vec![None; scene.shapes.len()];
    for (shape_id, shape) in scene.shapes.iter().enumerate() {
        let ShapeGeometry::Path { path } = &shape.geometry else {
            continue;
        };
        let length = shape_lengths.get(shape_id).copied().unwrap_or(0.0);
        if length <= 0.0 || !length.is_finite() {
            continue;
        }
        let num_base = path.num_control_points.len();
        if num_base == 0 {
            continue;
        }
        let inv_length = 1.0 / length;
        let total_points = path.points.len();
        let mut cdf = Vec::with_capacity(num_base);
        let mut pmf = Vec::with_capacity(num_base);
        let mut point_id_map = Vec::with_capacity(num_base);
        let mut point_id = 0usize;
        for &controls in &path.num_control_points {
            point_id_map.push(point_id);
            let seg_len = match controls {
                0 => {
                    let i0 = point_id;
                    let i1 = point_id + 1;
                    point_id += 1;
                    match (path_point(path, i0, total_points), path_point(path, i1, total_points)) {
                        (Some(p0), Some(p1)) => (p1 - p0).length(),
                        _ => 0.0,
                    }
                }
                1 => {
                    let i0 = point_id;
                    let i1 = point_id + 1;
                    let i2 = point_id + 2;
                    point_id += 2;
                    match (
                        path_point(path, i0, total_points),
                        path_point(path, i1, total_points),
                        path_point(path, i2, total_points),
                    ) {
                        (Some(p0), Some(p1), Some(p2)) => {
                            let eval = |t: f32| {
                                let tt = 1.0 - t;
                                p0 * (tt * tt) + p1 * (2.0 * tt * t) + p2 * (t * t)
                            };
                            let v0 = p0;
                            let v1 = eval(0.5);
                            let v2 = p2;
                            (v1 - v0).length() + (v2 - v1).length()
                        }
                        _ => 0.0,
                    }
                }
                2 => {
                    let i0 = point_id;
                    let i1 = point_id + 1;
                    let i2 = point_id + 2;
                    let i3 = point_id + 3;
                    point_id += 3;
                    match (
                        path_point(path, i0, total_points),
                        path_point(path, i1, total_points),
                        path_point(path, i2, total_points),
                        path_point(path, i3, total_points),
                    ) {
                        (Some(p0), Some(p1), Some(p2), Some(p3)) => {
                            let eval = |t: f32| {
                                let tt = 1.0 - t;
                                p0 * (tt * tt * tt)
                                    + p1 * (3.0 * tt * tt * t)
                                    + p2 * (3.0 * tt * t * t)
                                    + p3 * (t * t * t)
                            };
                            let v0 = p0;
                            let v1 = eval(1.0 / 3.0);
                            let v2 = eval(2.0 / 3.0);
                            let v3 = p3;
                            (v1 - v0).length() + (v2 - v1).length() + (v3 - v2).length()
                        }
                        _ => 0.0,
                    }
                }
                _ => 0.0,
            };
            let seg_norm = seg_len * inv_length;
            pmf.push(seg_norm);
            let accum = seg_norm + cdf.last().copied().unwrap_or(0.0);
            cdf.push(accum);
        }
        out[shape_id] = Some(PathCdf {
            cdf,
            pmf,
            point_id_map,
            length,
        });
    }
    out
}

/// Samples a piecewise-constant CDF and returns (index, local_t).
fn sample_cdf(cdf: &[f32], u: f32) -> (usize, f32) {
    if cdf.is_empty() {
        return (0, 0.0);
    }
    let mut idx = 0usize;
    while idx < cdf.len() && u > cdf[idx] {
        idx += 1;
    }
    if idx >= cdf.len() {
        idx = cdf.len() - 1;
    }
    let prev = if idx == 0 { 0.0 } else { cdf[idx - 1] };
    let denom = (cdf[idx] - prev).max(1.0e-6);
    let t = ((u - prev) / denom).clamp(0.0, 1.0);
    (idx, t)
}

/// Samples a boundary point on a shape and returns point, normal, pdf, and metadata.
fn sample_boundary_point(
    scene: &crate::scene::Scene,
    shape_id: usize,
    group_id: usize,
    t: f32,
    shape_length: f32,
    path_cdf: Option<&PathCdf>,
) -> Option<(Vec2, Vec2, f32, BoundaryData)> {
    let group = scene.groups.get(group_id)?;
    let shape = scene.shapes.get(shape_id)?;

    if group.fill.is_none() && group.stroke.is_none() {
        return None;
    }

    let mut pdf = 1.0f32;
    let mut local_t = t;
    let mut stroke_perturb = false;
    if group.fill.is_some() && group.stroke.is_some() {
        if local_t < 0.5 {
            stroke_perturb = false;
            local_t *= 2.0;
            pdf = 0.5;
        } else {
            stroke_perturb = true;
            local_t = 2.0 * (local_t - 0.5);
            pdf = 0.5;
        }
    } else if group.stroke.is_some() {
        stroke_perturb = true;
    }
    let mut stroke_direction = 0.0f32;
    if stroke_perturb {
        if local_t < 0.5 {
            stroke_direction = -1.0;
            local_t *= 2.0;
            pdf *= 0.5;
        } else {
            stroke_direction = 1.0;
            local_t = 2.0 * (local_t - 0.5);
            pdf *= 0.5;
        }
    }

    let mut data = BoundaryData {
        path: PathBoundaryData {
            base_point_id: 0,
            point_id: 0,
            t: 0.0,
        },
        is_stroke: stroke_perturb,
    };

    let mut normal = Vec2::ZERO;
    let pt = match &shape.geometry {
        ShapeGeometry::Circle { center, radius } => {
            let r = radius.abs();
            if r <= 0.0 {
                return None;
            }
            let angle = 2.0 * core::f32::consts::PI * local_t;
            let offset = Vec2::new(r * angle.cos(), r * angle.sin());
            normal = normalize(offset);
            pdf /= 2.0 * core::f32::consts::PI * r;
            let mut out = *center + offset;
            if stroke_direction != 0.0 {
                out += normal * (stroke_direction * shape.stroke_width);
                if stroke_direction < 0.0 {
                    normal = normal * -1.0;
                }
            }
            out
        }
        ShapeGeometry::Ellipse { center, radius } => {
            let rx = radius.x.abs();
            let ry = radius.y.abs();
            if rx <= 0.0 || ry <= 0.0 {
                return None;
            }
            let angle = 2.0 * core::f32::consts::PI * local_t;
            let (s, c) = angle.sin_cos();
            let offset = Vec2::new(rx * c, ry * s);
            let dxdt = -rx * s * 2.0 * core::f32::consts::PI;
            let dydt = ry * c * 2.0 * core::f32::consts::PI;
            normal = normalize(Vec2::new(dydt, -dxdt));
            pdf /= (dxdt * dxdt + dydt * dydt).sqrt();
            let mut out = *center + offset;
            if stroke_direction != 0.0 {
                out += normal * (stroke_direction * shape.stroke_width);
                if stroke_direction < 0.0 {
                    normal = normal * -1.0;
                }
            }
            out
        }
        ShapeGeometry::Rect { min, max } => {
            let w = max.x - min.x;
            let h = max.y - min.y;
            if w == 0.0 && h == 0.0 {
                return None;
            }
            pdf /= 2.0 * (w + h);
            let mut out = Vec2::ZERO;
            if local_t <= w / (w + h) {
                local_t *= (w + h) / w;
                if local_t < 0.5 {
                    normal = Vec2::new(0.0, -1.0);
                    out = *min + Vec2::new(2.0 * local_t * (max.x - min.x), 0.0);
                } else {
                    normal = Vec2::new(0.0, 1.0);
                    out = Vec2::new(min.x, max.y)
                        + Vec2::new(2.0 * (local_t - 0.5) * (max.x - min.x), 0.0);
                }
            } else {
                local_t = (local_t - w / (w + h)) * (w + h) / h;
                if local_t < 0.5 {
                    normal = Vec2::new(-1.0, 0.0);
                    out = *min + Vec2::new(0.0, 2.0 * local_t * (max.y - min.y));
                } else {
                    normal = Vec2::new(1.0, 0.0);
                    out = Vec2::new(max.x, min.y)
                        + Vec2::new(0.0, 2.0 * (local_t - 0.5) * (max.y - min.y));
                }
            }
            if stroke_direction != 0.0 {
                out += normal * (stroke_direction * shape.stroke_width);
                if stroke_direction < 0.0 {
                    normal = normal * -1.0;
                }
            }
            out
        }
        ShapeGeometry::Path { path } => {
            let path_cdf = path_cdf?;
            if path_cdf.length <= 0.0 {
                return None;
            }
            sample_boundary_path(
                path,
                path_cdf,
                shape_length,
                local_t,
                &mut normal,
                &mut pdf,
                &mut data,
                stroke_direction,
                shape.stroke_width,
            )
        }
    };

    if !pdf.is_finite() || pdf <= 0.0 {
        return None;
    }
    Some((pt, normal, pdf, data))
}

/// Samples a boundary point along a path segment (including stroke caps).
fn sample_boundary_path(
    path: &Path,
    path_cdf: &PathCdf,
    path_length: f32,
    mut t: f32,
    normal: &mut Vec2,
    pdf: &mut f32,
    data: &mut BoundaryData,
    stroke_direction: f32,
    stroke_radius: f32,
) -> Vec2 {
    let num_points = path.points.len();
    let num_base = path.num_control_points.len();
    if num_points == 0 || num_base == 0 {
        *pdf = 0.0;
        return Vec2::ZERO;
    }
    if stroke_direction != 0.0 && !path.is_closed {
        let mut cap_length = 0.0f32;
        if let Some(thickness) = &path.thickness {
            if !thickness.is_empty() {
                let r0 = thickness[0];
                let r1 = thickness[thickness.len() - 1];
                cap_length = core::f32::consts::PI * (r0 + r1);
            }
        } else {
            cap_length = 2.0 * core::f32::consts::PI * stroke_radius;
        }
        let denom = cap_length + path_length;
        if denom > 0.0 {
            let cap_prob = cap_length / denom;
            if t < cap_prob {
                t /= cap_prob;
                *pdf *= cap_prob;
                let mut r0 = stroke_radius;
                let mut r1 = stroke_radius;
                if let Some(thickness) = &path.thickness {
                    if !thickness.is_empty() {
                        r0 = thickness[0];
                        r1 = thickness[thickness.len() - 1];
                    }
                }
                let angle = 2.0 * core::f32::consts::PI * t;
                if stroke_direction < 0.0 {
                    let p0 = path.points[0];
                    let offset = Vec2::new(r0 * angle.cos(), r0 * angle.sin());
                    *normal = normalize(offset);
                    *pdf /= 2.0 * core::f32::consts::PI * r0;
                    data.path.base_point_id = 0;
                    data.path.point_id = 0;
                    data.path.t = 0.0;
                    return p0 + offset;
                } else {
                    let p0 = path.points[num_points - 1];
                    let offset = Vec2::new(r1 * angle.cos(), r1 * angle.sin());
                    *normal = normalize(offset);
                    *pdf /= 2.0 * core::f32::consts::PI * r1;
                    data.path.base_point_id = num_base - 1;
                    let controls = path.num_control_points[data.path.base_point_id] as usize;
                    data.path.point_id = num_points.saturating_sub(2 + controls);
                    data.path.t = 1.0;
                    return p0 + offset;
                }
            } else {
                t = (t - cap_prob) / (1.0 - cap_prob);
                *pdf *= 1.0 - cap_prob;
            }
        }
    }

    let (sample_id, local_t) = sample_cdf(&path_cdf.cdf, t);
    if sample_id >= path_cdf.point_id_map.len() {
        *pdf = 0.0;
        return Vec2::ZERO;
    }
    let point_id = path_cdf.point_id_map[sample_id];
    data.path.base_point_id = sample_id;
    data.path.point_id = point_id;
    data.path.t = local_t;
    if local_t < -1.0e-3 || local_t > 1.0 + 1.0e-3 {
        *pdf = 0.0;
        return Vec2::ZERO;
    }

    let next_index = |idx: usize| -> usize {
        if path.is_closed {
            idx % num_points
        } else {
            idx.min(num_points.saturating_sub(1))
        }
    };

    match path.num_control_points.get(sample_id).copied().unwrap_or(0) {
        0 => {
            let i0 = point_id;
            let i1 = next_index(point_id + 1);
            let Some((p0, r0)) = path_point_radius(path, i0, num_points, stroke_radius, 1.0) else {
                *pdf = 0.0;
                return Vec2::ZERO;
            };
            let Some((p1, r1)) = path_point_radius(path, i1, num_points, stroke_radius, 1.0) else {
                *pdf = 0.0;
                return Vec2::ZERO;
            };
            let tangent = p1 - p0;
            let tan_len = tangent.length();
            if tan_len == 0.0 {
                *pdf = 0.0;
                return Vec2::ZERO;
            }
            *normal = Vec2::new(-tangent.y, tangent.x) / tan_len;
            *pdf *= path_cdf.pmf.get(sample_id).copied().unwrap_or(0.0) / tan_len;
            let mut out = p0 + tangent * local_t;
            if stroke_direction != 0.0 {
                let r = r0 + local_t * (r1 - r0);
                out += *normal * (stroke_direction * r);
                if stroke_direction < 0.0 {
                    *normal = *normal * -1.0;
                }
            }
            out
        }
        1 => {
            let i0 = point_id;
            let i1 = point_id + 1;
            let i2 = next_index(point_id + 2);
            let Some((p0, r0)) = path_point_radius(path, i0, num_points, stroke_radius, 1.0) else {
                *pdf = 0.0;
                return Vec2::ZERO;
            };
            let Some((p1, r1)) = path_point_radius(path, i1, num_points, stroke_radius, 1.0) else {
                *pdf = 0.0;
                return Vec2::ZERO;
            };
            let Some((p2, r2)) = path_point_radius(path, i2, num_points, stroke_radius, 1.0) else {
                *pdf = 0.0;
                return Vec2::ZERO;
            };
            let tt = 1.0 - local_t;
            let eval = p0 * (tt * tt) + p1 * (2.0 * tt * local_t) + p2 * (local_t * local_t);
            let tangent = (p1 - p0) * (2.0 * tt) + (p2 - p1) * (2.0 * local_t);
            let tan_len = tangent.length();
            if tan_len == 0.0 {
                *pdf = 0.0;
                return Vec2::ZERO;
            }
            *normal = Vec2::new(-tangent.y, tangent.x) / tan_len;
            *pdf *= path_cdf.pmf.get(sample_id).copied().unwrap_or(0.0) / tan_len;
            let mut out = eval;
            if stroke_direction != 0.0 {
                let r = r0 * (tt * tt) + r1 * (2.0 * tt * local_t) + r2 * (local_t * local_t);
                out += *normal * (stroke_direction * r);
                if stroke_direction < 0.0 {
                    *normal = *normal * -1.0;
                }
            }
            out
        }
        2 => {
            let i0 = point_id;
            let i1 = point_id + 1;
            let i2 = point_id + 2;
            let i3 = next_index(point_id + 3);
            let Some((p0, r0)) = path_point_radius(path, i0, num_points, stroke_radius, 1.0) else {
                *pdf = 0.0;
                return Vec2::ZERO;
            };
            let Some((p1, r1)) = path_point_radius(path, i1, num_points, stroke_radius, 1.0) else {
                *pdf = 0.0;
                return Vec2::ZERO;
            };
            let Some((p2, r2)) = path_point_radius(path, i2, num_points, stroke_radius, 1.0) else {
                *pdf = 0.0;
                return Vec2::ZERO;
            };
            let Some((p3, r3)) = path_point_radius(path, i3, num_points, stroke_radius, 1.0) else {
                *pdf = 0.0;
                return Vec2::ZERO;
            };
            let tt = 1.0 - local_t;
            let eval = p0 * (tt * tt * tt)
                + p1 * (3.0 * tt * tt * local_t)
                + p2 * (3.0 * tt * local_t * local_t)
                + p3 * (local_t * local_t * local_t);
            let tangent = (p1 - p0) * (3.0 * tt * tt)
                + (p2 - p1) * (6.0 * tt * local_t)
                + (p3 - p2) * (3.0 * local_t * local_t);
            let tan_len = tangent.length();
            if tan_len == 0.0 {
                *pdf = 0.0;
                return Vec2::ZERO;
            }
            *normal = Vec2::new(-tangent.y, tangent.x) / tan_len;
            *pdf *= path_cdf.pmf.get(sample_id).copied().unwrap_or(0.0) / tan_len;
            let mut out = eval;
            if stroke_direction != 0.0 {
                let r = r0 * (tt * tt * tt)
                    + r1 * (3.0 * tt * tt * local_t)
                    + r2 * (3.0 * tt * local_t * local_t)
                    + r3 * (local_t * local_t * local_t);
                out += *normal * (stroke_direction * r);
                if stroke_direction < 0.0 {
                    *normal = *normal * -1.0;
                }
            }
            out
        }
        _ => {
            *pdf = 0.0;
            Vec2::ZERO
        }
    }
}

/// Renders a boundary sample and accumulates its gradient contribution.
fn render_edge_sample(
    scene: &crate::scene::Scene,
    bvh: &SceneBvh,
    d_render_image: &[f32],
    weight_image: &[f32],
    grads: &mut SceneGrad,
    compute_translation: bool,
    sample: BoundarySample,
) {
    let width = scene.width as i32;
    let height = scene.height as i32;
    if width <= 0 || height <= 0 {
        return;
    }
    if sample.pdf <= 0.0 {
        return;
    }
    let bx = (sample.pt.x * width as f32) as i32;
    let by = (sample.pt.y * height as f32) as i32;
    if bx < 0 || bx >= width || by < 0 || by >= height {
        return;
    }
    let pixel_index = by as usize * width as usize + bx as usize;
    let background = Some(sample_background(scene, pixel_index));

    let mut inside_query = EdgeQuery {
        shape_group_id: sample.shape_group_id,
        shape_id: sample.shape_id,
        hit: false,
    };
    let mut outside_query = EdgeQuery {
        shape_group_id: sample.shape_group_id,
        shape_id: sample.shape_id,
        hit: false,
    };

    let mut normal = sample.normal;
    let mut color_inside = sample_color(
        scene,
        bvh,
        sample.pt - normal * 1.0e-4,
        background,
        None,
        Some(&mut inside_query),
        grads,
        None,
        pixel_index,
    );
    let mut color_outside = sample_color(
        scene,
        bvh,
        sample.pt + normal * 1.0e-4,
        background,
        None,
        Some(&mut outside_query),
        grads,
        None,
        pixel_index,
    );

    if !inside_query.hit && !outside_query.hit {
        return;
    }
    if !inside_query.hit {
        normal = normal * -1.0;
        core::mem::swap(&mut color_inside, &mut color_outside);
    }

    let sboundary_pt = Vec2::new(sample.pt.x * width as f32, sample.pt.y * height as f32);
    let mut d_color = gather_d_color(
        scene.filter.filter_type,
        scene.filter.radius,
        d_render_image,
        weight_image,
        width,
        height,
        sboundary_pt,
    );
    let norm = (scene.width as f32) * (scene.height as f32);
    if norm > 0.0 {
        d_color.x /= norm;
        d_color.y /= norm;
        d_color.z /= norm;
        d_color.w /= norm;
    }

    let diff = Vec4::new(
        color_inside.x - color_outside.x,
        color_inside.y - color_outside.y,
        color_inside.z - color_outside.z,
        color_inside.w - color_outside.w,
    );
    let contrib = dot4(diff, d_color) / sample.pdf;
    if !contrib.is_finite() {
        return;
    }

    let shape = &scene.shapes[sample.shape_id];
    let group = &scene.groups[sample.shape_group_id];
    let d_shape = &mut grads.shapes[sample.shape_id];
    let d_group = &mut grads.shape_groups[sample.shape_group_id];
    accumulate_boundary_gradient(
        shape,
        contrib,
        sample.t,
        normal,
        sample.data,
        d_shape,
        group.shape_to_canvas,
        sample.local_pt,
        d_group,
    );

    if compute_translation {
        if let Some(trans) = grads.translation.as_mut() {
            let idx = pixel_index * 2;
            if idx + 1 < trans.len() {
                trans[idx] += normal.x * contrib;
                trans[idx + 1] += normal.y * contrib;
            }
        }
    }
}

/// Accumulates shape and transform gradients for a boundary contribution.
fn accumulate_boundary_gradient(
    shape: &Shape,
    contrib: f32,
    t: f32,
    normal: Vec2,
    boundary_data: BoundaryData,
    d_shape: &mut DShape,
    group_shape_to_canvas: Mat3,
    local_boundary_pt: Vec2,
    d_group: &mut DShapeGroup,
) {
    if !contrib.is_finite() {
        return;
    }

    if boundary_data.is_stroke {
        let has_thickness = matches!(
            shape.geometry,
            ShapeGeometry::Path { ref path } if path.thickness.is_some()
        );
        if has_thickness {
            if let ShapeGeometry::Path { path } = &shape.geometry {
                let base_point_id = boundary_data.path.base_point_id;
                let point_id = boundary_data.path.point_id;
                let t = boundary_data.path.t;
                if let DShapeGeometry::Path { thickness: Some(d_thickness), .. } =
                    &mut d_shape.geometry
                {
                    match path.num_control_points.get(base_point_id).copied().unwrap_or(0) {
                        0 => {
                            let i0 = point_id;
                            let i1 = if path.is_closed {
                                (point_id + 1) % path.points.len()
                            } else {
                                point_id + 1
                            };
                            if let Some(v0) = d_thickness.get_mut(i0) {
                                *v0 += (1.0 - t) * contrib;
                            }
                            if let Some(v1) = d_thickness.get_mut(i1) {
                                *v1 += t * contrib;
                            }
                        }
                        1 => {
                            let i0 = point_id;
                            let i1 = point_id + 1;
                            let i2 = if path.is_closed {
                                (point_id + 2) % path.points.len()
                            } else {
                                point_id + 2
                            };
                            let tt = 1.0 - t;
                            if let Some(v0) = d_thickness.get_mut(i0) {
                                *v0 += tt * tt * contrib;
                            }
                            if let Some(v1) = d_thickness.get_mut(i1) {
                                *v1 += 2.0 * tt * t * contrib;
                            }
                            if let Some(v2) = d_thickness.get_mut(i2) {
                                *v2 += t * t * contrib;
                            }
                        }
                        2 => {
                            let i0 = point_id;
                            let i1 = point_id + 1;
                            let i2 = point_id + 2;
                            let i3 = if path.is_closed {
                                (point_id + 3) % path.points.len()
                            } else {
                                point_id + 3
                            };
                            let tt = 1.0 - t;
                            if let Some(v0) = d_thickness.get_mut(i0) {
                                *v0 += tt * tt * tt * contrib;
                            }
                            if let Some(v1) = d_thickness.get_mut(i1) {
                                *v1 += 3.0 * tt * tt * t * contrib;
                            }
                            if let Some(v2) = d_thickness.get_mut(i2) {
                                *v2 += 3.0 * tt * t * t * contrib;
                            }
                            if let Some(v3) = d_thickness.get_mut(i3) {
                                *v3 += t * t * t * contrib;
                            }
                        }
                        _ => {}
                    }
                }
            }
        } else {
            d_shape.stroke_width += contrib;
        }
    }

    match &shape.geometry {
        ShapeGeometry::Circle { .. } => {
            if let DShapeGeometry::Circle { center, radius } = &mut d_shape.geometry {
                *center += normal * contrib;
                *radius += contrib;
            }
        }
        ShapeGeometry::Ellipse { .. } => {
            if let DShapeGeometry::Ellipse { center, radius } = &mut d_shape.geometry {
                *center += normal * contrib;
                let angle = 2.0 * core::f32::consts::PI * t;
                radius.x += angle.cos() * normal.x * contrib;
                radius.y += angle.sin() * normal.y * contrib;
            }
        }
        ShapeGeometry::Path { path } => {
            let base_point_id = boundary_data.path.base_point_id;
            let point_id = boundary_data.path.point_id;
            let t = boundary_data.path.t;
            let num_points = path.points.len();
            if let DShapeGeometry::Path { points, .. } = &mut d_shape.geometry {
                let mut add_point = |idx: usize, weight: f32, points: &mut [Vec2]| {
                    let index = if path.is_closed {
                        idx % num_points
                    } else {
                        idx
                    };
                    if let Some(p) = points.get_mut(index) {
                        *p += normal * (weight * contrib);
                    }
                };
                match path.num_control_points.get(base_point_id).copied().unwrap_or(0) {
                    0 => {
                        add_point(point_id, 1.0 - t, points);
                        add_point(point_id + 1, t, points);
                    }
                    1 => {
                        let tt = 1.0 - t;
                        add_point(point_id, tt * tt, points);
                        add_point(point_id + 1, 2.0 * tt * t, points);
                        add_point(point_id + 2, t * t, points);
                    }
                    2 => {
                        let tt = 1.0 - t;
                        add_point(point_id, tt * tt * tt, points);
                        add_point(point_id + 1, 3.0 * tt * tt * t, points);
                        add_point(point_id + 2, 3.0 * tt * t * t, points);
                        add_point(point_id + 3, t * t * t, points);
                    }
                    _ => {}
                }
            }
        }
        ShapeGeometry::Rect { .. } => {
            if let DShapeGeometry::Rect { min, max } = &mut d_shape.geometry {
                if normal == Vec2::new(-1.0, 0.0) {
                    min.x += -contrib;
                } else if normal == Vec2::new(1.0, 0.0) {
                    max.x += contrib;
                } else if normal == Vec2::new(0.0, -1.0) {
                    min.y += -contrib;
                } else if normal == Vec2::new(0.0, 1.0) {
                    max.y += contrib;
                }
            }
        }
    }

    let shape_to_canvas = group_shape_to_canvas.mul(shape.transform);
    let mut d_shape_to_canvas = zero_mat3();
    let mut d_local_boundary_pt = Vec2::ZERO;
    d_xform_pt(
        shape_to_canvas,
        local_boundary_pt,
        normal * contrib,
        &mut d_shape_to_canvas,
        &mut d_local_boundary_pt,
    );

    let d_group_mat = mat3_mul(d_shape_to_canvas, mat3_transpose(shape.transform));
    d_group.shape_to_canvas = mat3_add(d_group.shape_to_canvas, d_group_mat);

    let d_shape_mat = mat3_mul(mat3_transpose(group_shape_to_canvas), d_shape_to_canvas);
    d_shape.transform = mat3_add(d_shape.transform, d_shape_mat);
}
