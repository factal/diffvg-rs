//! Shape-level helpers for bounds and signed distances.

use crate::math::{Mat3, Vec2};
use crate::scene::{Shape, ShapeGeometry, StrokeJoin};
use crate::path_utils::{bounds_from_points, rect_corners, transform_path};

use super::bvh::Bounds;

pub(crate) fn shape_bounds(shape: &Shape) -> Option<Bounds> {
    let geom_bounds = shape_geom_bounds(shape)?;
    let pad = max_stroke_radius(shape);
    Some(inflate_bounds(geom_bounds, pad))
}

pub(crate) fn shape_geom_bounds(shape: &Shape) -> Option<Bounds> {
    match &shape.geometry {
        ShapeGeometry::Circle { center, radius } => {
            let transform = shape.transform;
            let center_t = transform.transform_point(*center);
            let r = radius.abs();
            let ext = ellipse_extents(transform, r, r);
            Some(Bounds {
                min: Vec2::new(center_t.x - ext.x, center_t.y - ext.y),
                max: Vec2::new(center_t.x + ext.x, center_t.y + ext.y),
            })
        }
        ShapeGeometry::Ellipse { center, radius } => {
            let transform = shape.transform;
            let center_t = transform.transform_point(*center);
            let rx = radius.x.abs();
            let ry = radius.y.abs();
            let ext = ellipse_extents(transform, rx, ry);
            Some(Bounds {
                min: Vec2::new(center_t.x - ext.x, center_t.y - ext.y),
                max: Vec2::new(center_t.x + ext.x, center_t.y + ext.y),
            })
        }
        ShapeGeometry::Rect { min, max } => {
            let transform = shape.transform;
            let corners = rect_corners(*min, *max)
                .into_iter()
                .map(|p| transform.transform_point(p))
                .collect::<Vec<_>>();
            let (min, max) = bounds_from_points(&corners);
            Some(Bounds { min, max })
        }
        ShapeGeometry::Path { path } => {
            let transform = shape.transform;
            let transformed = transform_path(path, transform);
            if transformed.points.is_empty() {
                return None;
            }
            let (min, max) = bounds_from_points(&transformed.points);
            Some(Bounds { min, max })
        }
    }
}

pub(crate) fn max_stroke_radius(shape: &Shape) -> f32 {
    let stroke_scale = shape.transform.max_scale().max(0.0);
    let mut radius = shape.stroke_width.abs() * stroke_scale;
    if let ShapeGeometry::Path { path } = &shape.geometry {
        if let Some(thickness) = &path.thickness {
            let mut max_thickness = 0.0f32;
            for &value in thickness {
                max_thickness = max_thickness.max(value.abs());
            }
            radius = radius.max(max_thickness * stroke_scale);
        }
    }
    if radius > 0.0 && shape.stroke_join == StrokeJoin::Miter {
        radius *= shape.stroke_miter_limit.max(1.0);
    }
    radius
}

fn ellipse_extents(transform: Mat3, rx: f32, ry: f32) -> Vec2 {
    let m00 = transform.m[0][0];
    let m01 = transform.m[0][1];
    let m10 = transform.m[1][0];
    let m11 = transform.m[1][1];
    let ex = ((m00 * rx) * (m00 * rx) + (m01 * ry) * (m01 * ry)).sqrt();
    let ey = ((m10 * rx) * (m10 * rx) + (m11 * ry) * (m11 * ry)).sqrt();
    Vec2::new(ex, ey)
}

fn inflate_bounds(bounds: Bounds, pad: f32) -> Bounds {
    let pad = pad.max(0.0);
    Bounds {
        min: Vec2::new(bounds.min.x - pad, bounds.min.y - pad),
        max: Vec2::new(bounds.max.x + pad, bounds.max.y + pad),
    }
}

pub(crate) fn segment_bounds(points: &[Vec2], pad: f32) -> Bounds {
    let (min, max) = bounds_from_points(points);
    inflate_bounds(Bounds { min, max }, pad)
}

pub(crate) fn closest_point_circle(center: Vec2, radius: f32, pt: Vec2) -> Vec2 {
    let d = pt - center;
    let len = d.length();
    if len <= 1.0e-8 {
        return Vec2::new(center.x + radius, center.y);
    }
    center + d * (radius / len)
}

pub(crate) fn closest_point_rect(min: Vec2, max: Vec2, pt: Vec2) -> Vec2 {
    let inside = pt.x >= min.x && pt.x <= max.x && pt.y >= min.y && pt.y <= max.y;
    if !inside {
        return Vec2::new(pt.x.clamp(min.x, max.x), pt.y.clamp(min.y, max.y));
    }
    let dl = pt.x - min.x;
    let dr = max.x - pt.x;
    let db = pt.y - min.y;
    let dt = max.y - pt.y;
    if dl <= dr && dl <= db && dl <= dt {
        Vec2::new(min.x, pt.y)
    } else if dr <= db && dr <= dt {
        Vec2::new(max.x, pt.y)
    } else if db <= dt {
        Vec2::new(pt.x, min.y)
    } else {
        Vec2::new(pt.x, max.y)
    }
}

pub(crate) fn rect_signed_distance(pt: Vec2, min: Vec2, max: Vec2) -> f32 {
    let dx = (min.x - pt.x).max(0.0).max(pt.x - max.x);
    let dy = (min.y - pt.y).max(0.0).max(pt.y - max.y);
    let outside = (dx * dx + dy * dy).sqrt();
    let inside = (pt.x - min.x)
        .min(max.x - pt.x)
        .min(pt.y - min.y)
        .min(max.y - pt.y);
    if outside > 0.0 {
        outside
    } else {
        -inside
    }
}

pub(crate) fn ellipse_signed_distance(pt: Vec2, center: Vec2, radius: Vec2) -> f32 {
    let rx = radius.x.abs().max(1.0e-6);
    let ry = radius.y.abs().max(1.0e-6);
    let dx = (pt.x - center.x) / rx;
    let dy = (pt.y - center.y) / ry;
    let len = (dx * dx + dy * dy).sqrt();
    let scale = rx.min(ry);
    (len - 1.0) * scale
}
