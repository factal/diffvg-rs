use crate::distance::ClosestPathPoint;
use crate::grad::{DShape, DShapeGeometry, DShapeGroup};
use crate::geometry::Path;
use crate::math::{Mat3, Vec2};
use crate::scene::{Shape, ShapeGeometry, ShapeGroup};

use super::math::{
    d_distance, d_normalize, d_xform_pt, mat3_add, mat3_mul, mat3_scale,
    mat3_transpose, normalize, zero_mat3,
};
use super::types::PathInfo;

pub(super) fn path_info_from_closest(
    scene: &crate::scene::Scene,
    shape_index: usize,
    info: Option<ClosestPathPoint>,
) -> Option<PathInfo> {
    let ShapeGeometry::Path { path } = &scene.shapes[shape_index].geometry else {
        return None;
    };
    let Some(info) = info else {
        return None;
    };
    let base_point_id = info.segment_index;
    let point_id = path_point_id(path, base_point_id);
    Some(PathInfo {
        base_point_id,
        point_id,
        t: info.t,
    })
}

pub(super) fn path_point_id(path: &Path, base_point_id: usize) -> usize {
    let mut point_id = 0usize;
    let count = base_point_id.min(path.num_control_points.len());
    for &controls in path.num_control_points.iter().take(count) {
        point_id += match controls {
            0 => 1,
            1 => 2,
            2 => 3,
            _ => 0,
        };
    }
    point_id
}

pub(super) fn d_compute_distance(
    group: &ShapeGroup,
    shape: &Shape,
    pt: Vec2,
    closest_pt: Vec2,
    path_info: Option<PathInfo>,
    d_dist: f32,
    d_shape: &mut DShape,
    d_group: &mut DShapeGroup,
) -> Vec2 {
    if (pt - closest_pt).length_squared() < 1.0e-10 {
        return Vec2::ZERO;
    }

    let local_pt_group = group.canvas_to_shape.transform_point(pt);
    let local_closest_group = group.canvas_to_shape.transform_point(closest_pt);
    let shape_inv = shape.transform.inverse().unwrap_or(Mat3::identity());
    let local_pt_shape = shape_inv.transform_point(local_pt_group);
    let local_closest_shape = shape_inv.transform_point(local_closest_group);

    let mut d_pt = Vec2::ZERO;
    let mut d_closest = Vec2::ZERO;
    d_distance(closest_pt, pt, d_dist, &mut d_closest, &mut d_pt);

    let mut d_shape_to_canvas = zero_mat3();
    let mut d_local_closest_group = Vec2::ZERO;
    d_xform_pt(
        group.shape_to_canvas,
        local_closest_group,
        d_closest,
        &mut d_shape_to_canvas,
        &mut d_local_closest_group,
    );

    let mut d_shape_transform = zero_mat3();
    let mut d_local_closest_shape = Vec2::ZERO;
    d_xform_pt(
        shape.transform,
        local_closest_shape,
        d_local_closest_group,
        &mut d_shape_transform,
        &mut d_local_closest_shape,
    );

    let mut d_local_pt_shape = Vec2::ZERO;
    d_closest_point(
        shape,
        local_pt_shape,
        d_local_closest_shape,
        path_info,
        d_shape,
        &mut d_local_pt_shape,
    );

    let mut d_shape_inv = zero_mat3();
    let mut d_local_pt_group = Vec2::ZERO;
    d_xform_pt(
        shape_inv,
        local_pt_group,
        d_local_pt_shape,
        &mut d_shape_inv,
        &mut d_local_pt_group,
    );

    let mut d_canvas_to_shape = zero_mat3();
    d_xform_pt(
        group.canvas_to_shape,
        pt,
        d_local_pt_group,
        &mut d_canvas_to_shape,
        &mut d_pt,
    );

    let tc2s = mat3_transpose(group.canvas_to_shape);
    let d_shape_to_canvas_corr =
        mat3_mul(mat3_mul(mat3_scale(tc2s, -1.0), d_canvas_to_shape), tc2s);
    d_group.shape_to_canvas = mat3_add(d_group.shape_to_canvas, d_shape_to_canvas);
    d_group.shape_to_canvas = mat3_add(d_group.shape_to_canvas, d_shape_to_canvas_corr);

    let ts_inv = mat3_transpose(shape_inv);
    let d_shape_transform_corr =
        mat3_mul(mat3_mul(mat3_scale(ts_inv, -1.0), d_shape_inv), ts_inv);
    d_shape.transform = mat3_add(d_shape.transform, d_shape_transform_corr);
    d_shape.transform = mat3_add(d_shape.transform, d_shape_transform);

    d_pt
}

fn d_closest_point(
    shape: &Shape,
    pt: Vec2,
    d_closest_pt: Vec2,
    path_info: Option<PathInfo>,
    d_shape: &mut DShape,
    d_pt: &mut Vec2,
) {
    match &shape.geometry {
        ShapeGeometry::Circle { center, radius } => {
            let v = pt - *center;
            let n = normalize(v);
            let mut d_center = d_closest_pt;
            let d_radius = d_closest_pt.dot(n);
            let d_n = d_closest_pt * *radius;
            let d_v = d_normalize(v, d_n);
            d_center -= d_v;
            *d_pt += d_v;
            if let DShapeGeometry::Circle { center: d_c, radius: d_r } = &mut d_shape.geometry {
                *d_c += d_center;
                *d_r += d_radius;
            }
        }
        ShapeGeometry::Ellipse { center, radius } => {
            let mut d_center = Vec2::ZERO;
            let mut d_radius = Vec2::ZERO;
            d_closest_point_ellipse(*center, *radius, pt, d_closest_pt, &mut d_center, &mut d_radius, d_pt);
            if let DShapeGeometry::Ellipse { center: d_c, radius: d_r } = &mut d_shape.geometry {
                *d_c += d_center;
                *d_r += d_radius;
            }
        }
        ShapeGeometry::Rect { min, max } => {
            let mut d_min = Vec2::ZERO;
            let mut d_max = Vec2::ZERO;
            d_closest_point_rect(*min, *max, pt, d_closest_pt, &mut d_min, &mut d_max, d_pt);
            if let DShapeGeometry::Rect { min: d_pmin, max: d_pmax } = &mut d_shape.geometry {
                *d_pmin += d_min;
                *d_pmax += d_max;
            }
        }
        ShapeGeometry::Path { path } => {
            let Some(info) = path_info else {
                return;
            };
            d_closest_point_path(path, pt, d_closest_pt, info, d_shape, d_pt);
        }
    }
}

fn d_closest_point_path(
    path: &Path,
    pt: Vec2,
    d_closest_pt: Vec2,
    path_info: PathInfo,
    d_shape: &mut DShape,
    d_pt: &mut Vec2,
) {
    let base_point_id = path_info.base_point_id;
    let point_id = path_info.point_id;
    let min_t_root = path_info.t;
    let num_points = path.points.len();
    if num_points == 0 {
        return;
    }

    let get_point = |idx: usize| -> Vec2 {
        if path.is_closed {
            path.points[idx % num_points]
        } else {
            path.points[idx.min(num_points - 1)]
        }
    };

    let mut add_point = |idx: usize, delta: Vec2, d_shape: &mut DShape| {
        if let DShapeGeometry::Path { points, .. } = &mut d_shape.geometry {
            let index = if path.is_closed { idx % num_points } else { idx };
            if let Some(p) = points.get_mut(index) {
                *p += delta;
            }
        }
    };

    match path.num_control_points.get(base_point_id).copied().unwrap_or(0) {
        0 => {
            let i0 = point_id;
            let i1 = (point_id + 1) % num_points;
            let p0 = get_point(i0);
            let p1 = get_point(i1);
            let t = (pt - p0).dot(p1 - p0) / (p1 - p0).dot(p1 - p0);
            if t < 0.0 {
                add_point(i0, d_closest_pt, d_shape);
            } else if t > 1.0 {
                add_point(i1, d_closest_pt, d_shape);
            } else {
                add_point(i0, d_closest_pt * (1.0 - t), d_shape);
                add_point(i1, d_closest_pt * t, d_shape);
            }
        }
        1 => {
            let i0 = point_id;
            let i1 = point_id + 1;
            let i2 = (point_id + 2) % num_points;
            let p0 = get_point(i0);
            let p1 = get_point(i1);
            let p2 = get_point(i2);
            let t = min_t_root;
            let mut d_p0 = Vec2::ZERO;
            let mut d_p1 = Vec2::ZERO;
            let mut d_p2 = Vec2::ZERO;
            if t == 0.0 {
                d_p0 += d_closest_pt;
            } else if t == 1.0 {
                d_p2 += d_closest_pt;
            } else {
                let a = p0 - p1 * 2.0 + p2;
                let b = p1 - p0;
                let A = a.dot(a);
                let B = 3.0 * a.dot(b);
                let C = 2.0 * b.dot(b) + a.dot(p0 - pt);
                let tt = 1.0 - t;
                let d_p = d_closest_pt;
                let d_tt = 2.0 * tt * d_p.dot(p0) + 2.0 * t * d_p.dot(p1);
                let d_t = -d_tt + 2.0 * tt * d_p.dot(p1) + 2.0 * t * d_p.dot(p2);
                d_p0 += d_p * (tt * tt);
                d_p1 += d_p * (2.0 * tt * t);
                d_p2 += d_p * (t * t);
                let poly_deriv_t = 3.0 * A * t * t + 2.0 * B * t + C;
                if poly_deriv_t.abs() > 1.0e-6 {
                    let d_A = -(d_t / poly_deriv_t) * t * t * t;
                    let d_B = -(d_t / poly_deriv_t) * t * t;
                    let d_C = -(d_t / poly_deriv_t) * t;
                    let d_D = -(d_t / poly_deriv_t);

                    d_p0 += a * (2.0 * d_A)
                        + (b - a) * (3.0 * d_B)
                        + (b * -4.0) * d_C
                        + (p0 - pt + a) * d_C
                        + (b - (p0 - pt)) * (2.0 * d_D);
                    d_p1 += a * (-4.0 * d_A)
                        + (a + b * -2.0) * (3.0 * d_B)
                        + (b * 2.0) * d_C
                        + (p0 - pt) * (-2.0 * d_C)
                        + (p0 - pt) * d_D;
                    d_p2 += a * (2.0 * d_A) + b * (3.0 * d_B) + (p0 - pt) * d_C;
                    *d_pt += a * (-d_C) + b * d_D;
                }
            }
            add_point(i0, d_p0, d_shape);
            add_point(i1, d_p1, d_shape);
            add_point(i2, d_p2, d_shape);
        }
        2 => {
            let i0 = point_id;
            let i1 = point_id + 1;
            let i2 = point_id + 2;
            let i3 = (point_id + 3) % num_points;
            let p0 = get_point(i0);
            let p1 = get_point(i1);
            let p2 = get_point(i2);
            let p3 = get_point(i3);
            let t = min_t_root;
            let mut d_p0 = Vec2::ZERO;
            let mut d_p1 = Vec2::ZERO;
            let mut d_p2 = Vec2::ZERO;
            let mut d_p3 = Vec2::ZERO;
            if t == 0.0 {
                d_p0 += d_closest_pt;
            } else if t == 1.0 {
                d_p3 += d_closest_pt;
            } else {
                let a = p0 * -1.0 + p1 * 3.0 + p2 * -3.0 + p3;
                let b = p0 * 3.0 + p1 * -6.0 + p2 * 3.0;
                let c = p0 * -3.0 + p1 * 3.0;
                let A: f32 = 3.0 * a.dot(a);
                if A.abs() < 1.0e-10 {
                    return;
                }
                let B = 5.0 * a.dot(b);
                let C = 4.0 * a.dot(c) + 2.0 * b.dot(b);
                let D = 3.0 * (b.dot(c) + a.dot(p0 - pt));
                let E = c.dot(c) + 2.0 * (p0 - pt).dot(b);
                let F = (p0 - pt).dot(c);
                let B = B / A;
                let C = C / A;
                let D = D / A;
                let E = E / A;
                let F = F / A;

                let tt = 1.0 - t;
                let d_p = d_closest_pt;
                let d_tt = 3.0 * tt * tt * d_p.dot(p0)
                    + 6.0 * tt * t * d_p.dot(p1)
                    + 3.0 * t * t * d_p.dot(p2);
                let d_t = -d_tt
                    + 3.0 * tt * tt * d_p.dot(p1)
                    + 6.0 * tt * t * d_p.dot(p2)
                    + 3.0 * t * t * d_p.dot(p3);
                d_p0 += d_p * (tt * tt * tt);
                d_p1 += d_p * (3.0 * tt * tt * t);
                d_p2 += d_p * (3.0 * tt * t * t);
                d_p3 += d_p * (t * t * t);

                let poly_deriv_t = 5.0 * t * t * t * t
                    + 4.0 * B * t * t * t
                    + 3.0 * C * t * t
                    + 2.0 * D * t
                    + E;
                if poly_deriv_t.abs() > 1.0e-10 {
                    let mut d_B = -(d_t / poly_deriv_t) * t * t * t * t;
                    let mut d_C = -(d_t / poly_deriv_t) * t * t * t;
                    let mut d_D = -(d_t / poly_deriv_t) * t * t;
                    let mut d_E = -(d_t / poly_deriv_t) * t;
                    let mut d_F = -(d_t / poly_deriv_t);
                    let mut d_A = -d_B * B / A - d_C * C / A - d_D * D / A - d_E * E / A - d_F * F / A;
                    d_B /= A;
                    d_C /= A;
                    d_D /= A;
                    d_E /= A;
                    d_F /= A;

                    d_p0 += a * (3.0 * -1.0 * 2.0 * d_A);
                    d_p1 += a * (3.0 * 3.0 * 2.0 * d_A);
                    d_p2 += a * (3.0 * -3.0 * 2.0 * d_A);
                    d_p3 += a * (3.0 * 1.0 * 2.0 * d_A);
                    d_p0 += (b * -1.0 + a * 3.0) * (5.0 * d_B);
                    d_p1 += (b * 3.0 + a * -6.0) * (5.0 * d_B);
                    d_p2 += (b * -3.0 + a * 3.0) * (5.0 * d_B);
                    d_p3 += b * (5.0 * d_B);
                    d_p0 += (c * -1.0 + a * -3.0) * (4.0 * d_C) + b * (3.0 * 2.0 * d_C);
                    d_p1 += (c * 3.0 + a * 3.0) * (4.0 * d_C) + b * (-6.0 * 2.0 * d_C);
                    d_p2 += c * (-3.0 * d_C * 4.0) + b * (3.0 * 2.0 * d_C);
                    d_p3 += c * (4.0 * d_C);
                    d_p0 += (c * 3.0 + b * -3.0) * (3.0 * d_D) + (a * -1.0 + p0 - pt) * (3.0 * d_D);
                    d_p1 += (c * -6.0 + b * 3.0) * (3.0 * d_D) + (p0 - pt) * (3.0 * d_D);
                    d_p2 += c * (3.0 * 3.0 * d_D) + (p0 - pt) * (-3.0 * d_D);
                    *d_pt += a * (-1.0 * 3.0 * d_D);
                    d_p0 += c * (-3.0 * 2.0 * d_E) + (b + (p0 - pt) * 3.0) * (2.0 * d_E);
                    d_p1 += c * (3.0 * 2.0 * d_E) + (p0 - pt) * (-6.0 * 2.0 * d_E);
                    d_p2 += (p0 - pt) * (3.0 * 2.0 * d_E);
                    *d_pt += b * (-1.0 * 2.0 * d_E);
                    d_p0 += (c * -3.0 + (p0 - pt) * -3.0) * d_F + (c * 1.0) * d_F;
                    d_p1 += (p0 - pt) * (3.0 * d_F);
                    *d_pt += c * (-1.0 * d_F);
                }
            }
            add_point(i0, d_p0, d_shape);
            add_point(i1, d_p1, d_shape);
            add_point(i2, d_p2, d_shape);
            add_point(i3, d_p3, d_shape);
        }
        _ => {}
    }
}

fn d_closest_point_rect(
    min: Vec2,
    max: Vec2,
    pt: Vec2,
    d_closest_pt: Vec2,
    d_min: &mut Vec2,
    d_max: &mut Vec2,
    d_pt: &mut Vec2,
) {
    let dist_to_seg = |p0: Vec2, p1: Vec2| -> f32 {
        let t = (pt - p0).dot(p1 - p0) / (p1 - p0).dot(p1 - p0);
        if t < 0.0 {
            (p0 - pt).length()
        } else if t > 1.0 {
            (p1 - pt).length()
        } else {
            (p0 + (p1 - p0) * t - pt).length()
        }
    };
    let left_top = min;
    let right_top = Vec2::new(max.x, min.y);
    let left_bottom = Vec2::new(min.x, max.y);
    let right_bottom = max;
    let mut min_id = 0;
    let mut min_dist = dist_to_seg(left_top, left_bottom);
    let top_dist = dist_to_seg(left_top, right_top);
    let right_dist = dist_to_seg(right_top, right_bottom);
    let bottom_dist = dist_to_seg(left_bottom, right_bottom);
    if top_dist < min_dist {
        min_dist = top_dist;
        min_id = 1;
    }
    if right_dist < min_dist {
        min_dist = right_dist;
        min_id = 2;
    }
    if bottom_dist < min_dist {
        min_dist = bottom_dist;
        min_id = 3;
    }

    let mut update = |p0: Vec2, p1: Vec2, d_closest_pt: Vec2, d_p0: &mut Vec2, d_p1: &mut Vec2| {
        let t = (pt - p0).dot(p1 - p0) / (p1 - p0).dot(p1 - p0);
        if t < 0.0 {
            *d_p0 += d_closest_pt;
        } else if t > 1.0 {
            *d_p1 += d_closest_pt;
        } else {
            let d_p = d_closest_pt;
            *d_p0 += d_p * (1.0 - t);
            *d_p1 += d_p * t;
            let d_t = d_p.dot(p1 - p0);
            let denom = (p1 - p0).dot(p1 - p0);
            let d_num = d_t / denom;
            let d_den = d_t * (-t) / denom;
            *d_pt += (p1 - p0) * d_num;
            *d_p1 += (pt - p0) * d_num;
            *d_p0 += (p0 - p1 + p0 - pt) * d_num;
            *d_p1 += (p1 - p0) * (2.0 * d_den);
            *d_p0 += (p0 - p1) * (2.0 * d_den);
        }
    };

    let mut d_left_top = Vec2::ZERO;
    let mut d_right_top = Vec2::ZERO;
    let mut d_left_bottom = Vec2::ZERO;
    let mut d_right_bottom = Vec2::ZERO;
    match min_id {
        0 => update(left_top, left_bottom, d_closest_pt, &mut d_left_top, &mut d_left_bottom),
        1 => update(left_top, right_top, d_closest_pt, &mut d_left_top, &mut d_right_top),
        2 => update(right_top, right_bottom, d_closest_pt, &mut d_right_top, &mut d_right_bottom),
        _ => update(left_bottom, right_bottom, d_closest_pt, &mut d_left_bottom, &mut d_right_bottom),
    }
    *d_min += d_left_top;
    d_max.x += d_right_top.x;
    d_min.y += d_right_top.y;
    d_min.x += d_left_bottom.x;
    d_max.y += d_left_bottom.y;
    *d_max += d_right_bottom;
}

fn d_closest_point_ellipse(
    center: Vec2,
    radius: Vec2,
    pt: Vec2,
    d_closest_pt: Vec2,
    d_center: &mut Vec2,
    d_radius: &mut Vec2,
    d_pt: &mut Vec2,
) {
    let rx = radius.x.abs();
    let ry = radius.y.abs();
    let eps = 1.0e-6;
    let local = pt - center;
    if rx < eps && ry < eps {
        *d_center += d_closest_pt;
        return;
    }
    if rx < eps {
        let hit = if local.y >= -ry && local.y <= ry { 1.0 } else { 0.0 };
        *d_center += Vec2::new(d_closest_pt.x, d_closest_pt.y);
        d_pt.y += d_closest_pt.y * hit;
        if local.y > ry {
            d_radius.y += d_closest_pt.y;
        } else if local.y < -ry {
            d_radius.y -= d_closest_pt.y;
        }
        return;
    }
    if ry < eps {
        let hit = if local.x >= -rx && local.x <= rx { 1.0 } else { 0.0 };
        *d_center += Vec2::new(d_closest_pt.x, d_closest_pt.y);
        d_pt.x += d_closest_pt.x * hit;
        if local.x > rx {
            d_radius.x += d_closest_pt.x;
        } else if local.x < -rx {
            d_radius.x -= d_closest_pt.x;
        }
        return;
    }

    let sign_x = if local.x < 0.0 { -1.0 } else { 1.0 };
    let sign_y = if local.y < 0.0 { -1.0 } else { 1.0 };
    let x = local.x.abs();
    let y = local.y.abs();
    let mut t = (y * rx).atan2(x * ry);
    let mut g_t = 0.0;
    for _ in 0..20 {
        let (s, c) = t.sin_cos();
        let g = rx * x * s - ry * y * c + (ry * ry - rx * rx) * s * c;
        g_t = rx * x * c + ry * y * s + (ry * ry - rx * rx) * (c * c - s * s);
        if g_t.abs() < 1.0e-12 {
            break;
        }
        let next = (t - g / g_t).clamp(0.0, core::f32::consts::FRAC_PI_2);
        if (next - t).abs() < 1.0e-6 {
            t = next;
            break;
        }
        t = next;
    }
    let (s, c) = t.sin_cos();
    let mut d_t = 0.0;
    d_radius.x += d_closest_pt.x * sign_x * c;
    d_t += d_closest_pt.x * sign_x * (-rx * s);
    d_radius.y += d_closest_pt.y * sign_y * s;
    d_t += d_closest_pt.y * sign_y * (ry * c);

    let g_a = x * s - 2.0 * rx * s * c;
    let g_b = -y * c + 2.0 * ry * s * c;
    let g_x = rx * s;
    let g_y = -ry * c;
    if g_t.abs() > 1.0e-12 {
        let inv = -d_t / g_t;
        d_radius.x += inv * g_a;
        d_radius.y += inv * g_b;
        let d_x = inv * g_x;
        let d_y = inv * g_y;
        d_pt.x += d_x * sign_x;
        d_pt.y += d_y * sign_y;
        d_center.x -= d_x * sign_x;
        d_center.y -= d_y * sign_y;
    }
}
