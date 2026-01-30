use crate::grad::DPaint;
use crate::math::{Vec2, Vec4};
use crate::scene::Paint;

use super::math::{d_length, dot4};

pub(super) fn paint_color(paint: &Paint, pt: Vec2) -> Vec4 {
    match paint {
        Paint::Solid(color) => Vec4::new(color.r, color.g, color.b, color.a),
        Paint::LinearGradient(gradient) => {
            let beg = gradient.start;
            let end = gradient.end;
            let denom = (end - beg).dot(end - beg).max(1.0e-3);
            let t = (pt - beg).dot(end - beg) / denom;
            if gradient.stops.is_empty() {
                return Vec4::ZERO;
            }
            if t < gradient.stops[0].offset {
                let c = gradient.stops[0].color;
                return Vec4::new(c.r, c.g, c.b, c.a);
            }
            for i in 0..gradient.stops.len() - 1 {
                let curr = gradient.stops[i];
                let next = gradient.stops[i + 1];
                if t >= curr.offset && t < next.offset {
                    let tt = (t - curr.offset) / (next.offset - curr.offset);
                    let c0 = curr.color;
                    let c1 = next.color;
                    return Vec4::new(
                        c0.r * (1.0 - tt) + c1.r * tt,
                        c0.g * (1.0 - tt) + c1.g * tt,
                        c0.b * (1.0 - tt) + c1.b * tt,
                        c0.a * (1.0 - tt) + c1.a * tt,
                    );
                }
            }
            let last = gradient.stops[gradient.stops.len() - 1].color;
            Vec4::new(last.r, last.g, last.b, last.a)
        }
        Paint::RadialGradient(gradient) => {
            let offset = pt - gradient.center;
            let normalized = Vec2::new(
                offset.x / gradient.radius.x,
                offset.y / gradient.radius.y,
            );
            let t = normalized.length();
            if gradient.stops.is_empty() {
                return Vec4::ZERO;
            }
            if t < gradient.stops[0].offset {
                let c = gradient.stops[0].color;
                return Vec4::new(c.r, c.g, c.b, c.a);
            }
            for i in 0..gradient.stops.len() - 1 {
                let curr = gradient.stops[i];
                let next = gradient.stops[i + 1];
                if t >= curr.offset && t < next.offset {
                    let tt = (t - curr.offset) / (next.offset - curr.offset);
                    let c0 = curr.color;
                    let c1 = next.color;
                    return Vec4::new(
                        c0.r * (1.0 - tt) + c1.r * tt,
                        c0.g * (1.0 - tt) + c1.g * tt,
                        c0.b * (1.0 - tt) + c1.b * tt,
                        c0.a * (1.0 - tt) + c1.a * tt,
                    );
                }
            }
            let last = gradient.stops[gradient.stops.len() - 1].color;
            Vec4::new(last.r, last.g, last.b, last.a)
        }
    }
}

pub(super) fn d_sample_paint(
    paint: &Paint,
    d_paint: &mut DPaint,
    pt: Vec2,
    d_color: Vec4,
    d_translation: &mut Vec2,
) {
    match (paint, d_paint) {
        (Paint::Solid(_), DPaint::Solid(color)) => {
            color.r += d_color.x;
            color.g += d_color.y;
            color.b += d_color.z;
            color.a += d_color.w;
        }
        (Paint::LinearGradient(gradient), DPaint::LinearGradient(d_grad)) => {
            let beg = gradient.start;
            let end = gradient.end;
            let denom = (end - beg).dot(end - beg).max(1.0e-3);
            let t = (pt - beg).dot(end - beg) / denom;
            if gradient.stops.is_empty() {
                return;
            }
            if t < gradient.stops[0].offset {
                if let Some(stop) = d_grad.stops.get_mut(0) {
                    stop.color.r += d_color.x;
                    stop.color.g += d_color.y;
                    stop.color.b += d_color.z;
                    stop.color.a += d_color.w;
                }
                return;
            }
            for i in 0..gradient.stops.len() - 1 {
                let curr = gradient.stops[i];
                let next = gradient.stops[i + 1];
                if t >= curr.offset && t < next.offset {
                    let tt = (t - curr.offset) / (next.offset - curr.offset);
                    let color_curr = curr.color.to_vec4();
                    let color_next = next.color.to_vec4();
                    let d_color_curr = Vec4::new(
                        d_color.x * (1.0 - tt),
                        d_color.y * (1.0 - tt),
                        d_color.z * (1.0 - tt),
                        d_color.w * (1.0 - tt),
                    );
                    let d_color_next = Vec4::new(
                        d_color.x * tt,
                        d_color.y * tt,
                        d_color.z * tt,
                        d_color.w * tt,
                    );
                    if let Some(stop) = d_grad.stops.get_mut(i) {
                        stop.color.r += d_color_curr.x;
                        stop.color.g += d_color_curr.y;
                        stop.color.b += d_color_curr.z;
                        stop.color.a += d_color_curr.w;
                    }
                    if let Some(stop) = d_grad.stops.get_mut(i + 1) {
                        stop.color.r += d_color_next.x;
                        stop.color.g += d_color_next.y;
                        stop.color.b += d_color_next.z;
                        stop.color.a += d_color_next.w;
                    }
                    let diff = Vec4::new(
                        color_next.x - color_curr.x,
                        color_next.y - color_curr.y,
                        color_next.z - color_curr.z,
                        color_next.w - color_curr.w,
                    );
                    let d_tt = dot4(d_color, diff);
                    let denom_offset = next.offset - curr.offset;
                    if denom_offset.abs() > 0.0 {
                        let d_offset_next = -d_tt * tt / denom_offset;
                        let d_offset_curr = d_tt * (tt - 1.0) / denom_offset;
                        if let Some(stop) = d_grad.stops.get_mut(i) {
                            stop.offset += d_offset_curr;
                        }
                        if let Some(stop) = d_grad.stops.get_mut(i + 1) {
                            stop.offset += d_offset_next;
                        }
                    }
                    let d_t = d_tt / denom_offset;
                    let d_beg = Vec2::new(
                        d_t * (-(pt.x - beg.x) - (end.x - beg.x)) / denom,
                        d_t * (-(pt.y - beg.y) - (end.y - beg.y)) / denom,
                    );
                    let d_end = Vec2::new(
                        d_t * (pt.x - beg.x) / denom,
                        d_t * (pt.y - beg.y) / denom,
                    );
                    let d_l = -d_t * t / denom;
                    if (end - beg).dot(end - beg) > 1.0e-3 {
                        let adjust = Vec2::new(beg.x - end.x, beg.y - end.y);
                        d_grad.start += Vec2::new(d_beg.x + 2.0 * d_l * adjust.x, d_beg.y + 2.0 * d_l * adjust.y);
                        d_grad.end += Vec2::new(d_end.x + 2.0 * d_l * (end.x - beg.x), d_end.y + 2.0 * d_l * (end.y - beg.y));
                    } else {
                        d_grad.start += d_beg;
                        d_grad.end += d_end;
                    }
                    *d_translation += d_beg + d_end;
                    return;
                }
            }
            if let Some(stop) = d_grad.stops.last_mut() {
                stop.color.r += d_color.x;
                stop.color.g += d_color.y;
                stop.color.b += d_color.z;
                stop.color.a += d_color.w;
            }
        }
        (Paint::RadialGradient(gradient), DPaint::RadialGradient(d_grad)) => {
            let offset = pt - gradient.center;
            let normalized = Vec2::new(
                offset.x / gradient.radius.x,
                offset.y / gradient.radius.y,
            );
            let t = normalized.length();
            if gradient.stops.is_empty() {
                return;
            }
            if t < gradient.stops[0].offset {
                if let Some(stop) = d_grad.stops.get_mut(0) {
                    stop.color.r += d_color.x;
                    stop.color.g += d_color.y;
                    stop.color.b += d_color.z;
                    stop.color.a += d_color.w;
                }
                return;
            }
            for i in 0..gradient.stops.len() - 1 {
                let curr = gradient.stops[i];
                let next = gradient.stops[i + 1];
                if t >= curr.offset && t < next.offset {
                    let tt = (t - curr.offset) / (next.offset - curr.offset);
                    let color_curr = curr.color.to_vec4();
                    let color_next = next.color.to_vec4();
                    let d_color_curr = Vec4::new(
                        d_color.x * (1.0 - tt),
                        d_color.y * (1.0 - tt),
                        d_color.z * (1.0 - tt),
                        d_color.w * (1.0 - tt),
                    );
                    let d_color_next = Vec4::new(
                        d_color.x * tt,
                        d_color.y * tt,
                        d_color.z * tt,
                        d_color.w * tt,
                    );
                    if let Some(stop) = d_grad.stops.get_mut(i) {
                        stop.color.r += d_color_curr.x;
                        stop.color.g += d_color_curr.y;
                        stop.color.b += d_color_curr.z;
                        stop.color.a += d_color_curr.w;
                    }
                    if let Some(stop) = d_grad.stops.get_mut(i + 1) {
                        stop.color.r += d_color_next.x;
                        stop.color.g += d_color_next.y;
                        stop.color.b += d_color_next.z;
                        stop.color.a += d_color_next.w;
                    }
                    let diff = Vec4::new(
                        color_next.x - color_curr.x,
                        color_next.y - color_curr.y,
                        color_next.z - color_curr.z,
                        color_next.w - color_curr.w,
                    );
                    let d_tt = dot4(d_color, diff);
                    let denom_offset = next.offset - curr.offset;
                    if denom_offset.abs() > 0.0 {
                        let d_offset_next = -d_tt * tt / denom_offset;
                        let d_offset_curr = d_tt * (tt - 1.0) / denom_offset;
                        if let Some(stop) = d_grad.stops.get_mut(i) {
                            stop.offset += d_offset_curr;
                        }
                        if let Some(stop) = d_grad.stops.get_mut(i + 1) {
                            stop.offset += d_offset_next;
                        }
                    }
                    let d_t = d_tt / denom_offset;
                    let d_normalized = d_length(normalized, d_t);
                    let d_offset = Vec2::new(
                        d_normalized.x / gradient.radius.x,
                        d_normalized.y / gradient.radius.y,
                    );
                    let d_radius = Vec2::new(
                        -d_normalized.x * offset.x / (gradient.radius.x * gradient.radius.x),
                        -d_normalized.y * offset.y / (gradient.radius.y * gradient.radius.y),
                    );
                    let d_center = Vec2::new(-d_offset.x, -d_offset.y);
                    d_grad.center += d_center;
                    d_grad.radius += d_radius;
                    *d_translation += d_center;
                    return;
                }
            }
            if let Some(stop) = d_grad.stops.last_mut() {
                stop.color.r += d_color.x;
                stop.color.g += d_color.y;
                stop.color.b += d_color.z;
                stop.color.a += d_color.w;
            }
        }
        _ => {}
    }
}
