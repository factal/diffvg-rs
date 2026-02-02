use cubecl::prelude::*;
use crate::gpu::constants::*;
use super::{math::*, curves::*};

#[cube]
pub(super) fn shape_fill_coverage(
    shape_data: &Array<f32>,
    segment_data: &Array<f32>,
    curve_data: &Array<f32>,
    shape_index: u32,
    fill_rule: u32,
    px: f32,
    py: f32,
    aa: f32,
) -> f32 {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let base = (shape_index * SHAPE_STRIDE) as usize;
    let kind = shape_data[base] as u32;
    let seg_offset = shape_data[base + 1] as u32;
    let seg_count = shape_data[base + 2] as u32;
    let curve_offset = shape_data[base + 12] as u32;
    let curve_count = shape_data[base + 13] as u32;
    let use_distance_approx = shape_data[base + 14] > f32::new(0.5);
    let p0 = shape_data[base + 4];
    let p1 = shape_data[base + 5];
    let p2 = shape_data[base + 6];
    let p3 = shape_data[base + 7];

    let mut coverage = zero;

    if kind == SHAPE_KIND_CIRCLE {
        let dx = px - p0;
        let dy = py - p1;
        let radius = p2;
        let dist = (dx * dx + dy * dy).sqrt() - radius;
        coverage = sdf_coverage(dist, aa);
    } else if kind == SHAPE_KIND_ELLIPSE {
        let dx = (px - p0) / max_f32(p2.abs(), f32::new(1.0e-3));
        let dy = (py - p1) / max_f32(p3.abs(), f32::new(1.0e-3));
        let len = (dx * dx + dy * dy).sqrt();
        let scale = min_f32(p2.abs(), p3.abs());
        let dist = (len - one) * scale;
        coverage = sdf_coverage(dist, aa);
    } else if kind == SHAPE_KIND_RECT {
        let min_x = p0;
        let min_y = p1;
        let max_x = p2;
        let max_y = p3;
        let dx = max_f32(max_f32(min_x - px, zero), px - max_x);
        let dy = max_f32(max_f32(min_y - py, zero), py - max_y);
        let outside = (dx * dx + dy * dy).sqrt();
        let inside = min_f32(
            min_f32(px - min_x, max_x - px),
            min_f32(py - min_y, max_y - py),
        );
        let dist = if outside > zero { outside } else { -inside };
        coverage = sdf_coverage(dist, aa);
    } else if kind == SHAPE_KIND_PATH {
        if curve_count > 0 || seg_count > 0 {
            let min_x = p0;
            let min_y = p1;
            let max_x = p2;
            let max_y = p3;
            if px >= min_x - aa && px <= max_x + aa && py >= min_y - aa && py <= max_y + aa {
                let mut min_dist = f32::new(1.0e20);
                let mut winding = zero;
                let mut crossings = zero;

                if curve_count > 0 {
                    for s in 0..curve_count {
                        let seg_base = ((curve_offset + s) * CURVE_STRIDE) as usize;
                        let seg_kind = curve_data[seg_base] as u32;
                        let x0 = curve_data[seg_base + 1];
                        let y0 = curve_data[seg_base + 2];
                        let x1 = curve_data[seg_base + 3];
                        let y1 = curve_data[seg_base + 4];
                        let x2 = curve_data[seg_base + 5];
                        let y2 = curve_data[seg_base + 6];
                        let x3 = curve_data[seg_base + 7];
                        let y3 = curve_data[seg_base + 8];

                        let dist = if seg_kind == 0 {
                            distance_to_segment(px, py, x0, y0, x1, y1)
                        } else if seg_kind == 1 {
                            distance_to_quadratic(px, py, x0, y0, x1, y1, x2, y2, use_distance_approx)
                        } else {
                            distance_to_cubic(
                                px,
                                py,
                                x0,
                                y0,
                                x1,
                                y1,
                                x2,
                                y2,
                                x3,
                                y3,
                                use_distance_approx,
                            )
                        };
                        if dist < min_dist {
                            min_dist = dist;
                        }

                        if seg_kind == 0 {
                            winding_and_crossings_line(
                                px,
                                py,
                                x0,
                                y0,
                                x1,
                                y1,
                                fill_rule,
                                &mut winding,
                                &mut crossings,
                            );
                        } else if seg_kind == 1 {
                            winding_and_crossings_quadratic(
                                px,
                                py,
                                x0,
                                y0,
                                x1,
                                y1,
                                x2,
                                y2,
                                fill_rule,
                                &mut winding,
                                &mut crossings,
                            );
                        } else {
                            winding_and_crossings_cubic(
                                px,
                                py,
                                x0,
                                y0,
                                x1,
                                y1,
                                x2,
                                y2,
                                x3,
                                y3,
                                fill_rule,
                                &mut winding,
                                &mut crossings,
                            );
                        }
                    }
                } else {
                    for s in 0..seg_count {
                        let seg_base = ((seg_offset + s) * SEGMENT_STRIDE) as usize;
                        let x0 = segment_data[seg_base];
                        let y0 = segment_data[seg_base + 1];
                        let x1 = segment_data[seg_base + 2];
                        let y1 = segment_data[seg_base + 3];

                        let dist = distance_to_segment(px, py, x0, y0, x1, y1);
                        if dist < min_dist {
                            min_dist = dist;
                        }

                        winding_and_crossings_line(
                            px,
                            py,
                            x0,
                            y0,
                            x1,
                            y1,
                            fill_rule,
                            &mut winding,
                            &mut crossings,
                        );
                    }
                }

                let inside = if fill_rule == 1 {
                    crossings > zero
                } else {
                    winding != zero
                };
                let dist = if inside { -min_dist } else { min_dist };
                coverage = sdf_coverage(dist, aa);
            }
        }
    }

    coverage
}

#[cube]
pub(super) fn shape_stroke_coverage(
    shape_data: &Array<f32>,
    segment_data: &Array<f32>,
    shape_index: u32,
    px: f32,
    py: f32,
    aa: f32,
) -> f32 {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let base = (shape_index * SHAPE_STRIDE) as usize;
    let kind = shape_data[base] as u32;
    let seg_offset = shape_data[base + 1] as u32;
    let seg_count = shape_data[base + 2] as u32;
    let stroke_width = shape_data[base + 3];
    let has_thickness = shape_data[base + 8] > f32::new(0.5);
    let join_type = shape_data[base + 9] as u32;
    let cap_type = shape_data[base + 10] as u32;
    let miter_limit = shape_data[base + 11];

    let p0 = shape_data[base + 4];
    let p1 = shape_data[base + 5];
    let p2 = shape_data[base + 6];
    let p3 = shape_data[base + 7];

    let mut coverage = zero;

    if has_thickness || stroke_width > zero {
        if kind == SHAPE_KIND_CIRCLE {
            let dx = px - p0;
            let dy = py - p1;
            let radius = p2;
            let dist = (dx * dx + dy * dy).sqrt();
            let sdf = abs_f32(dist - radius) - stroke_width;
            coverage = sdf_coverage(sdf, aa);
        } else if kind == SHAPE_KIND_ELLIPSE {
            let dx = (px - p0) / max_f32(p2.abs(), f32::new(1.0e-3));
            let dy = (py - p1) / max_f32(p3.abs(), f32::new(1.0e-3));
            let len = (dx * dx + dy * dy).sqrt();
            let scale = min_f32(p2.abs(), p3.abs());
            let dist = (len - one) * scale;
            let sdf = abs_f32(dist) - stroke_width;
            coverage = sdf_coverage(sdf, aa);
        } else if kind == SHAPE_KIND_RECT {
            let min_x = p0;
            let min_y = p1;
            let max_x = p2;
            let max_y = p3;
            let dx = max_f32(max_f32(min_x - px, zero), px - max_x);
            let dy = max_f32(max_f32(min_y - py, zero), py - max_y);
            let outside = (dx * dx + dy * dy).sqrt();
            let inside = min_f32(
                min_f32(px - min_x, max_x - px),
                min_f32(py - min_y, max_y - py),
            );
            let dist = if outside > zero { outside } else { -inside };
            let sdf = abs_f32(dist) - stroke_width;
            coverage = sdf_coverage(sdf, aa);
        } else if kind == SHAPE_KIND_PATH {
            if seg_count > 0 {
                let mut min_sdf = f32::new(1.0e20);
                for s in 0..seg_count {
                    let seg_base = ((seg_offset + s) * SEGMENT_STRIDE) as usize;
                    let x0 = segment_data[seg_base];
                    let y0 = segment_data[seg_base + 1];
                    let x1 = segment_data[seg_base + 2];
                    let y1 = segment_data[seg_base + 3];
                    let r0 = segment_data[seg_base + 4];
                    let r1 = segment_data[seg_base + 5];
                    let prev_dx = segment_data[seg_base + 6];
                    let prev_dy = segment_data[seg_base + 7];
                    let next_dx = segment_data[seg_base + 8];
                    let next_dy = segment_data[seg_base + 9];
                    let start_cap = segment_data[seg_base + 10];
                    let end_cap = segment_data[seg_base + 11];

                    let sdf = segment_sdf_with_caps(
                        px,
                        py,
                        x0,
                        y0,
                        x1,
                        y1,
                        r0,
                        r1,
                        start_cap,
                        end_cap,
                        join_type,
                        cap_type,
                        has_thickness,
                        stroke_width,
                    );
                    if sdf < min_sdf {
                        min_sdf = sdf;
                    }

                    if join_type == JOIN_MITER {
                        let dir = normalize_vec(x1 - x0, y1 - y0);
                        if start_cap <= f32::new(0.5) {
                            let miter_sdf = miter_join_sdf(
                                px,
                                py,
                                x0,
                                y0,
                                prev_dx,
                                prev_dy,
                                dir[0],
                                dir[1],
                                r0,
                                miter_limit,
                            );
                            if miter_sdf < min_sdf {
                                min_sdf = miter_sdf;
                            }
                        }
                        if end_cap <= f32::new(0.5) {
                            let miter_sdf = miter_join_sdf(
                                px,
                                py,
                                x1,
                                y1,
                                dir[0],
                                dir[1],
                                next_dx,
                                next_dy,
                                r1,
                                miter_limit,
                            );
                            if miter_sdf < min_sdf {
                                min_sdf = miter_sdf;
                            }
                        }
                    }
                }
                coverage = sdf_coverage(min_sdf, aa);
            }
        }
    }

    coverage
}

#[cube]
pub(super) fn segment_sdf_with_caps(
    px: f32,
    py: f32,
    ax: f32,
    ay: f32,
    bx: f32,
    by: f32,
    r0_in: f32,
    r1_in: f32,
    start_cap: f32,
    end_cap: f32,
    join_type: u32,
    cap_type: u32,
    has_thickness: bool,
    stroke_width: f32,
) -> f32 {
    let zero = f32::new(0.0);
    let half = f32::new(0.5);
    let cap_round = f32::new(2.0);
    let cap_square = f32::new(1.0);
    let cap_butt = f32::new(0.0);

    let r0 = if has_thickness { r0_in } else { stroke_width };
    let r1 = if has_thickness { r1_in } else { stroke_width };

    let mut cap_start = cap_type as f32;
    if start_cap <= half {
        cap_start = if join_type == JOIN_ROUND { cap_round } else { cap_butt };
    }
    let mut cap_end = cap_type as f32;
    if end_cap <= half {
        cap_end = if join_type == JOIN_ROUND { cap_round } else { cap_butt };
    }

    let ext_start = if cap_start == cap_square { r0 } else { zero };
    let ext_end = if cap_end == cap_square { r1 } else { zero };

    let mut sdf = segment_rect_sdf(px, py, ax, ay, bx, by, r0, r1, ext_start, ext_end);

    if cap_start == cap_round {
        let cap_sdf = distance_to_point(px, py, ax, ay) - r0;
        sdf = min_f32(sdf, cap_sdf);
    }
    if cap_end == cap_round {
        let cap_sdf = distance_to_point(px, py, bx, by) - r1;
        sdf = min_f32(sdf, cap_sdf);
    }

    sdf
}

#[cube]
pub(super) fn segment_rect_sdf(
    px: f32,
    py: f32,
    ax: f32,
    ay: f32,
    bx: f32,
    by: f32,
    r0: f32,
    r1: f32,
    ext_start: f32,
    ext_end: f32,
) -> f32 {
    let zero = f32::new(0.0);
    let vx = bx - ax;
    let vy = by - ay;
    let len = (vx * vx + vy * vy).sqrt();
    let mut sdf = distance_to_point(px, py, ax, ay) - r0;
    if len > f32::new(1.0e-6) {
        let tx = vx / len;
        let ty = vy / len;
        let nx = -ty;
        let ny = tx;

        let dx = px - ax;
        let dy = py - ay;
        let s = dx * tx + dy * ty;
        let n = dx * nx + dy * ny;

        let s0 = -ext_start;
        let s1 = len + ext_end;

        let s_clamped = clamp_range(s, zero, len);
        let t = s_clamped / len;
        let radius = r0 + t * (r1 - r0);

        let mut outside_x = zero;
        if s < s0 {
            outside_x = s0 - s;
        } else if s > s1 {
            outside_x = s - s1;
        }
        let mut outside_y = abs_f32(n) - radius;
        if outside_y < zero {
            outside_y = zero;
        }

        let outside = (outside_x * outside_x + outside_y * outside_y).sqrt();
        if outside_x > zero || outside_y > zero {
            sdf = outside;
        } else {
            let inside_x = min_f32(s - s0, s1 - s);
            let inside_y = radius - abs_f32(n);
            sdf = -min_f32(inside_x, inside_y);
        }
    }

    sdf
}

#[cube]
pub(super) fn miter_join_sdf(
    px: f32,
    py: f32,
    vx: f32,
    vy: f32,
    dir0x: f32,
    dir0y: f32,
    dir1x: f32,
    dir1y: f32,
    radius: f32,
    miter_limit: f32,
) -> f32 {
    let zero = f32::new(0.0);
    let big = f32::new(1.0e20);
    let eps = f32::new(1.0e-5);
    let mut out = big;
    if radius > zero {
        let d0 = normalize_vec(dir0x, dir0y);
        let d1 = normalize_vec(dir1x, dir1y);
        if !(d0[0] == zero && d0[1] == zero) && !(d1[0] == zero && d1[1] == zero) {
            let cross = cross2(d0[0], d0[1], d1[0], d1[1]);
            if abs_f32(cross) >= eps {
                let sign = if cross > zero { f32::new(1.0) } else { f32::new(-1.0) };
                let n0x = -d0[1] * sign;
                let n0y = d0[0] * sign;
                let n1x = -d1[1] * sign;
                let n1y = d1[0] * sign;

                let p0x = vx + n0x * radius;
                let p0y = vy + n0y * radius;
                let p1x = vx + n1x * radius;
                let p1y = vy + n1y * radius;

                let denom = cross;
                let t = cross2(p1x - p0x, p1y - p0y, d1[0], d1[1]) / denom;
                let mx = p0x + d0[0] * t;
                let my = p0y + d0[1] * t;

                let miter_len = distance_to_point(mx, my, vx, vy);
                if miter_len <= miter_limit * radius {
                    out = distance_to_triangle(px, py, p0x, p0y, p1x, p1y, mx, my);
                }
            }
        }
    }
    out
}

#[cube]
pub(super) fn distance_to_triangle(
    px: f32,
    py: f32,
    ax: f32,
    ay: f32,
    bx: f32,
    by: f32,
    cx: f32,
    cy: f32,
) -> f32 {
    let zero = f32::new(0.0);
    let c1 = cross2(bx - ax, by - ay, px - ax, py - ay);
    let c2 = cross2(cx - bx, cy - by, px - bx, py - by);
    let c3 = cross2(ax - cx, ay - cy, px - cx, py - cy);
    let has_neg = c1 < zero || c2 < zero || c3 < zero;
    let has_pos = c1 > zero || c2 > zero || c3 > zero;
    let inside = !(has_neg && has_pos);

    let d0 = distance_to_segment(px, py, ax, ay, bx, by);
    let d1 = distance_to_segment(px, py, bx, by, cx, cy);
    let d2 = distance_to_segment(px, py, cx, cy, ax, ay);
    let min_dist = min_f32(d0, min_f32(d1, d2));
    if inside {
        -min_dist
    } else {
        min_dist
    }
}

#[cube]
pub(super) fn normalize_vec(x: f32, y: f32) -> Line<f32> {
    let zero = f32::new(0.0);
    let len = (x * x + y * y).sqrt();
    let mut out = Line::empty(2usize);
    if len > f32::new(1.0e-6) {
        out[0] = x / len;
        out[1] = y / len;
    } else {
        out[0] = zero;
        out[1] = zero;
    }
    out
}

#[cube]
pub(super) fn cross2(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    ax * by - ay * bx
}

#[cube]
pub(super) fn clamp_range(v: f32, min_v: f32, max_v: f32) -> f32 {
    if v < min_v {
        min_v
    } else if v > max_v {
        max_v
    } else {
        v
    }
}

#[cube]
pub(super) fn sdf_coverage(dist: f32, aa: f32) -> f32 {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let two = f32::new(2.0);
    let three = f32::new(3.0);
    let mut coverage = if dist < zero { one } else { zero };
    if aa > zero {
        let inv = one / (two * aa);
        let mut t = (dist + aa) * inv;
        t = clamp01(t);
        let smooth = t * t * (three - two * t);
        coverage = one - smooth;
    }
    coverage
}
