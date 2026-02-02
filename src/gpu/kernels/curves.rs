use cubecl::prelude::*;
use super::math::*;

#[cube]
pub(super) fn distance_to_point(px: f32, py: f32, ax: f32, ay: f32) -> f32 {
    let dx = px - ax;
    let dy = py - ay;
    (dx * dx + dy * dy).sqrt()
}

#[cube]
pub(super) fn distance_to_segment_with_t(
    px: f32,
    py: f32,
    ax: f32,
    ay: f32,
    bx: f32,
    by: f32,
) -> Line<f32> {
    let zero = f32::new(0.0);
    let vx = bx - ax;
    let vy = by - ay;
    let wx = px - ax;
    let wy = py - ay;
    let c1 = vx * wx + vy * wy;
    let c2 = vx * vx + vy * vy;
    let mut t = zero;
    if c2 > zero && c1 > zero {
        t = c1 / c2;
        t = clamp01(t);
    }
    let proj_x = ax + t * vx;
    let proj_y = ay + t * vy;
    let dist_sq = (px - proj_x) * (px - proj_x) + (py - proj_y) * (py - proj_y);
    let mut out = Line::empty(2usize);
    out[0] = dist_sq.sqrt();
    out[1] = t;
    out
}

#[cube]
pub(super) fn distance_to_segment(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let zero = f32::new(0.0);
    let vx = bx - ax;
    let vy = by - ay;
    let wx = px - ax;
    let wy = py - ay;

    let c1 = vx * wx + vy * wy;
    let c2 = vx * vx + vy * vy;

    let mut dist_sq = (px - ax) * (px - ax) + (py - ay) * (py - ay);
    if c1 > zero {
        if c2 <= c1 {
            dist_sq = (px - bx) * (px - bx) + (py - by) * (py - by);
        } else {
            let t = c1 / c2;
            let proj_x = ax + t * vx;
            let proj_y = ay + t * vy;
            dist_sq = (px - proj_x) * (px - proj_x) + (py - proj_y) * (py - proj_y);
        }
    }

    dist_sq.sqrt()
}

#[cube]
pub(super) fn winding_and_crossings_line(
    px: f32,
    py: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    fill_rule: u32,
    winding: &mut f32,
    crossings: &mut f32,
) {
    let one = f32::new(1.0);
    let y0_le = y0 <= py;
    let y1_le = y1 <= py;
    if y0_le != y1_le {
        let t = (py - y0) / (y1 - y0);
        let x_int = x0 + t * (x1 - x0);
        if x_int > px {
            if fill_rule == 1 {
                *crossings = one - *crossings;
            } else {
                let delta = if y1 > y0 { one } else { -one };
                *winding += delta;
            }
        }
    }
}

#[cube]
pub(super) fn winding_and_crossings_quadratic(
    px: f32,
    py: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    fill_rule: u32,
    winding: &mut f32,
    crossings: &mut f32,
) {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    // y(t) = (y0 - 2y1 + y2) t^2 + (-2y0 + 2y1) t + y0
    let ay = y0 - f32::new(2.0) * y1 + y2;
    let by = -f32::new(2.0) * y0 + f32::new(2.0) * y1;
    let cy = y0 - py;
    let roots = solve_quadratic(ay, by, cy);
    let count = roots[0] as u32;
    let mut i = u32::new(0);
    while i < count {
        let t = roots[(i + 1) as usize];
        if t >= zero && t <= one {
            let tt = one - t;
            let x = (tt * tt) * x0 + (f32::new(2.0) * tt * t) * x1 + (t * t) * x2;
            if x > px {
                if fill_rule == 1 {
                    *crossings = one - *crossings;
                } else {
                    let dy = f32::new(2.0) * ay * t + by;
                    let delta = if dy > zero { one } else { -one };
                    *winding += delta;
                }
            }
        }
        i += u32::new(1);
    }
}

#[cube]
pub(super) fn winding_and_crossings_cubic(
    px: f32,
    py: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    x3: f32,
    y3: f32,
    fill_rule: u32,
    winding: &mut f32,
    crossings: &mut f32,
) {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let a = -y0 + f32::new(3.0) * y1 - f32::new(3.0) * y2 + y3;
    let b = f32::new(3.0) * y0 - f32::new(6.0) * y1 + f32::new(3.0) * y2;
    let c = -f32::new(3.0) * y0 + f32::new(3.0) * y1;
    let d = y0 - py;
    let roots = solve_cubic(a, b, c, d);
    let count = roots[0] as u32;
    let mut i = u32::new(0);
    while i < count {
        let t = roots[(i + 1) as usize];
        if t >= zero && t <= one {
            let tt = one - t;
            let x = (tt * tt * tt) * x0
                + (f32::new(3.0) * tt * tt * t) * x1
                + (f32::new(3.0) * tt * t * t) * x2
                + (t * t * t) * x3;
            if x > px {
                if fill_rule == 1 {
                    *crossings = one - *crossings;
                } else {
                    let dy = f32::new(3.0) * a * t * t + f32::new(2.0) * b * t + c;
                    let delta = if dy > zero { one } else { -one };
                    *winding += delta;
                }
            }
        }
        i += u32::new(1);
    }
}

#[cube]
pub(super) fn closest_point_quadratic_with_t(
    px: f32,
    py: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    use_distance_approx: bool,
) -> Line<f32> {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let mut best_t = zero;
    let mut best_dist = distance_to_point(px, py, x0, y0);
    if use_distance_approx {
        let approx = quadratic_closest_pt_approx(px, py, x0, y0, x1, y1, x2, y2);
        best_dist = distance_to_point(px, py, approx[0], approx[1]);
        best_t = approx[2];
    } else {
        let dist_end = distance_to_point(px, py, x2, y2);
        if dist_end < best_dist {
            best_dist = dist_end;
            best_t = one;
        }

        let ax = x0 - f32::new(2.0) * x1 + x2;
        let ay = y0 - f32::new(2.0) * y1 + y2;
        let bx = -x0 + x1;
        let by = -y0 + y1;
        let cx = x0 - px;
        let cy = y0 - py;

        let a = ax * ax + ay * ay;
        let b = f32::new(3.0) * (ax * bx + ay * by);
        let c = f32::new(2.0) * (bx * bx + by * by) + (ax * cx + ay * cy);
        let d = bx * cx + by * cy;

        let roots = solve_cubic(a, b, c, d);
        let count = roots[0] as u32;
        let mut i = u32::new(0);
        while i < count {
            let t = roots[(i + 1) as usize];
            if t >= zero && t <= one {
                let tt = one - t;
                let x = (tt * tt) * x0 + (f32::new(2.0) * tt * t) * x1 + (t * t) * x2;
                let y = (tt * tt) * y0 + (f32::new(2.0) * tt * t) * y1 + (t * t) * y2;
                let dist = distance_to_point(px, py, x, y);
                if dist < best_dist {
                    best_dist = dist;
                    best_t = t;
                }
            }
            i += u32::new(1);
        }
    }

    let mut out = Line::empty(2usize);
    out[0] = best_dist;
    out[1] = best_t;
    out
}

#[cube]
pub(super) fn distance_to_quadratic(
    px: f32,
    py: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    use_distance_approx: bool,
) -> f32 {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let mut min_dist = distance_to_point(px, py, x0, y0);
    if use_distance_approx {
        let approx = quadratic_closest_pt_approx(px, py, x0, y0, x1, y1, x2, y2);
        min_dist = distance_to_point(px, py, approx[0], approx[1]);
    } else {
        let dist_end = distance_to_point(px, py, x2, y2);
        if dist_end < min_dist {
            min_dist = dist_end;
        }

        let ax = x0 - f32::new(2.0) * x1 + x2;
        let ay = y0 - f32::new(2.0) * y1 + y2;
        let bx = -x0 + x1;
        let by = -y0 + y1;
        let cx = x0 - px;
        let cy = y0 - py;

        let a = ax * ax + ay * ay;
        let b = f32::new(3.0) * (ax * bx + ay * by);
        let c = f32::new(2.0) * (bx * bx + by * by) + (ax * cx + ay * cy);
        let d = bx * cx + by * cy;

        let roots = solve_cubic(a, b, c, d);
        let count = roots[0] as u32;
        let mut i = u32::new(0);
        while i < count {
            let t = roots[(i + 1) as usize];
            if t >= zero && t <= one {
                let tt = one - t;
                let x = (tt * tt) * x0 + (f32::new(2.0) * tt * t) * x1 + (t * t) * x2;
                let y = (tt * tt) * y0 + (f32::new(2.0) * tt * t) * y1 + (t * t) * y2;
                let dist = distance_to_point(px, py, x, y);
                if dist < min_dist {
                    min_dist = dist;
                }
            }
            i += u32::new(1);
        }
    }

    min_dist
}

#[cube]
pub(super) fn distance_to_cubic(
    px: f32,
    py: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    x3: f32,
    y3: f32,
    use_distance_approx: bool,
) -> f32 {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let min_dist = if use_distance_approx {
        distance_to_cubic_approx(px, py, x0, y0, x1, y1, x2, y2, x3, y3)
    } else {
        let mut min_dist = distance_to_point(px, py, x0, y0);
        let dist_end = distance_to_point(px, py, x3, y3);
        if dist_end < min_dist {
            min_dist = dist_end;
        }

        let ax = -x0 + f32::new(3.0) * x1 - f32::new(3.0) * x2 + x3;
        let ay = -y0 + f32::new(3.0) * y1 - f32::new(3.0) * y2 + y3;
        let bx = f32::new(3.0) * x0 - f32::new(6.0) * x1 + f32::new(3.0) * x2;
        let by = f32::new(3.0) * y0 - f32::new(6.0) * y1 + f32::new(3.0) * y2;
        let cx = -f32::new(3.0) * x0 + f32::new(3.0) * x1;
        let cy = -f32::new(3.0) * y0 + f32::new(3.0) * y1;
        let dx = x0 - px;
        let dy = y0 - py;

        let a = f32::new(3.0) * (ax * ax + ay * ay);
        if a.abs() > f32::new(1.0e-8) {
            let b = f32::new(5.0) * (ax * bx + ay * by);
            let c = f32::new(4.0) * (ax * cx + ay * cy)
                + f32::new(2.0) * (bx * bx + by * by);
            let d = f32::new(3.0) * ((bx * cx + by * cy) + (ax * dx + ay * dy));
            let e = (cx * cx + cy * cy) + f32::new(2.0) * (dx * bx + dy * by);
            let f = dx * cx + dy * cy;

            let b = b / a;
            let c = c / a;
            let d = d / a;
            let e = e / a;
            let f = f / a;

            let p1a = f32::new(2.0 / 5.0) * c - f32::new(4.0 / 25.0) * b * b;
            let p1b = f32::new(3.0 / 5.0) * d - f32::new(3.0 / 25.0) * b * c;
            let p1c = f32::new(4.0 / 5.0) * e - f32::new(2.0 / 25.0) * b * d;
            let p1d = f - b * e / f32::new(25.0);

            let q_root = -b / f32::new(5.0);

            let p_roots = solve_cubic(p1a, p1b, p1c, p1d);
            let num_p = p_roots[0] as u32;

            let mut intervals = Line::empty(4usize);
            let mut num_intervals = u32::new(0);
            if q_root >= zero && q_root <= one {
                intervals[num_intervals as usize] = q_root;
                num_intervals += 1;
            }
            let mut i = u32::new(0);
            while i < num_p {
                intervals[num_intervals as usize] = p_roots[(i + 1) as usize];
                num_intervals += 1;
                i += u32::new(1);
            }

            // sort intervals
            let mut j = u32::new(1);
            while j < num_intervals {
                let mut k = j;
                while k > 0 {
                    let a = intervals[(k - 1) as usize];
                    let b = intervals[k as usize];
                    if a <= b {
                        break;
                    }
                    intervals[(k - 1) as usize] = b;
                    intervals[k as usize] = a;
                    k -= u32::new(1);
                }
                j += u32::new(1);
            }

            let mut lower = zero;
            let mut idx = u32::new(0);
            while idx <= num_intervals {
                if idx < num_intervals && intervals[idx as usize] < zero {
                    idx += u32::new(1);
                } else {
                    let upper = if idx < num_intervals {
                        min_f32(intervals[idx as usize], one)
                    } else {
                        one
                    };
                    let mut lb = lower;
                    let mut ub = upper;
                    let mut lb_eval = eval_quintic(lb, b, c, d, e, f);
                    let mut ub_eval = eval_quintic(ub, b, c, d, e, f);
                    if lb_eval * ub_eval > zero {
                        lower = upper;
                        idx += u32::new(1);
                    } else {
                        if lb_eval > ub_eval {
                            let tmp = lb;
                            lb = ub;
                            ub = tmp;
                            let tmp_eval = lb_eval;
                            lb_eval = ub_eval;
                            ub_eval = tmp_eval;
                        }
                        let mut t = (lb + ub) * f32::new(0.5);
                        let mut it = u32::new(0);
                        while it < u32::new(20) {
                            if t < lb || t > ub {
                                t = (lb + ub) * f32::new(0.5);
                            }
                            let value = eval_quintic(t, b, c, d, e, f);
                            if abs_f32(value) < f32::new(1.0e-5) || it == u32::new(19) {
                                break;
                            }
                            if value > zero {
                                ub = t;
                            } else {
                                lb = t;
                            }
                            let derivative = eval_quintic_deriv(t, b, c, d, e);
                            t = t - value / derivative;
                            it += u32::new(1);
                        }

                        if t >= zero && t <= one {
                            let tt = one - t;
                            let x = (tt * tt * tt) * x0
                                + (f32::new(3.0) * tt * tt * t) * x1
                                + (f32::new(3.0) * tt * t * t) * x2
                                + (t * t * t) * x3;
                            let y = (tt * tt * tt) * y0
                                + (f32::new(3.0) * tt * tt * t) * y1
                                + (f32::new(3.0) * tt * t * t) * y2
                                + (t * t * t) * y3;
                            let dist = distance_to_point(px, py, x, y);
                            if dist < min_dist {
                                min_dist = dist;
                            }
                        }

                        if upper >= one {
                            break;
                        }
                        lower = upper;
                        idx += u32::new(1);
                    }
                }
            }
        }
        min_dist
    };

    min_dist
}

#[cube]
pub(super) fn distance_to_cubic_approx_with_t(
    px: f32,
    py: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    x3: f32,
    y3: f32,
) -> Line<f32> {
    let steps = u32::new(8);
    let inv = f32::new(1.0) / f32::cast_from(steps);
    let mut prev_x = x0;
    let mut prev_y = y0;
    let mut prev_t = f32::new(0.0);
    let mut best_dist = distance_to_point(px, py, x0, y0);
    let mut best_t = prev_t;
    let mut i = u32::new(1);
    while i <= steps {
        let t = f32::cast_from(i) * inv;
        let tt = f32::new(1.0) - t;
        let tt2 = tt * tt;
        let t2 = t * t;
        let a = tt2 * tt;
        let b = f32::new(3.0) * tt2 * t;
        let c = f32::new(3.0) * tt * t2;
        let d = t2 * t;
        let cx = a * x0 + b * x1 + c * x2 + d * x3;
        let cy = a * y0 + b * y1 + c * y2 + d * y3;
        let seg = distance_to_segment_with_t(px, py, prev_x, prev_y, cx, cy);
        let dist = seg[0];
        let seg_t = seg[1];
        let local_t = prev_t + seg_t * (t - prev_t);
        if dist < best_dist {
            best_dist = dist;
            best_t = local_t;
        }
        prev_x = cx;
        prev_y = cy;
        prev_t = t;
        i += u32::new(1);
    }
    let mut out = Line::empty(2usize);
    out[0] = best_dist;
    out[1] = best_t;
    out
}

#[cube]
pub(super) fn closest_point_cubic_with_t(
    px: f32,
    py: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    x3: f32,
    y3: f32,
    use_distance_approx: bool,
) -> Line<f32> {
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let mut best_t = zero;
    let mut best_dist = distance_to_point(px, py, x0, y0);
    if use_distance_approx {
        let approx = distance_to_cubic_approx_with_t(px, py, x0, y0, x1, y1, x2, y2, x3, y3);
        best_dist = approx[0];
        best_t = approx[1];
    } else {
        let dist_end = distance_to_point(px, py, x3, y3);
        if dist_end < best_dist {
            best_dist = dist_end;
            best_t = one;
        }

        let ax = -x0 + f32::new(3.0) * x1 - f32::new(3.0) * x2 + x3;
        let ay = -y0 + f32::new(3.0) * y1 - f32::new(3.0) * y2 + y3;
        let bx = f32::new(3.0) * x0 - f32::new(6.0) * x1 + f32::new(3.0) * x2;
        let by = f32::new(3.0) * y0 - f32::new(6.0) * y1 + f32::new(3.0) * y2;
        let cx = -f32::new(3.0) * x0 + f32::new(3.0) * x1;
        let cy = -f32::new(3.0) * y0 + f32::new(3.0) * y1;
        let dx = x0 - px;
        let dy = y0 - py;

        let a = f32::new(3.0) * (ax * ax + ay * ay);
        if a.abs() > f32::new(1.0e-8) {
            let b = f32::new(5.0) * (ax * bx + ay * by);
            let c = f32::new(4.0) * (ax * cx + ay * cy)
                + f32::new(2.0) * (bx * bx + by * by);
            let d = f32::new(3.0) * ((bx * cx + by * cy) + (ax * dx + ay * dy));
            let e = (cx * cx + cy * cy) + f32::new(2.0) * (dx * bx + dy * by);
            let f = dx * cx + dy * cy;

            let b = b / a;
            let c = c / a;
            let d = d / a;
            let e = e / a;
            let f = f / a;

            let p1a = f32::new(2.0 / 5.0) * c - f32::new(4.0 / 25.0) * b * b;
            let p1b = f32::new(3.0 / 5.0) * d - f32::new(3.0 / 25.0) * b * c;
            let p1c = f32::new(4.0 / 5.0) * e - f32::new(2.0 / 25.0) * b * d;
            let p1d = f - b * e / f32::new(25.0);

            let q_root = -b / f32::new(5.0);

            let p_roots = solve_cubic(p1a, p1b, p1c, p1d);
            let num_p = p_roots[0] as u32;

            let mut intervals = Line::empty(4usize);
            let mut num_intervals = u32::new(0);
            if q_root >= zero && q_root <= one {
                intervals[num_intervals as usize] = q_root;
                num_intervals += 1;
            }
            let mut i = u32::new(0);
            while i < num_p {
                intervals[num_intervals as usize] = p_roots[(i + 1) as usize];
                num_intervals += 1;
                i += u32::new(1);
            }

            let mut j = u32::new(1);
            while j < num_intervals {
                let mut k = j;
                while k > 0 {
                    let a = intervals[(k - 1) as usize];
                    let b = intervals[k as usize];
                    if a <= b {
                        break;
                    }
                    intervals[(k - 1) as usize] = b;
                    intervals[k as usize] = a;
                    k -= u32::new(1);
                }
                j += u32::new(1);
            }

            let mut lower = zero;
            let mut idx = u32::new(0);
            while idx <= num_intervals {
                if idx < num_intervals && intervals[idx as usize] < zero {
                    idx += u32::new(1);
                } else {
                    let upper = if idx < num_intervals {
                        min_f32(intervals[idx as usize], one)
                    } else {
                        one
                    };
                    let mut lb = lower;
                    let mut ub = upper;
                    let mut lb_eval = eval_quintic(lb, b, c, d, e, f);
                    let mut ub_eval = eval_quintic(ub, b, c, d, e, f);
                    if lb_eval * ub_eval > zero {
                        lower = upper;
                        idx += u32::new(1);
                    } else {
                        if lb_eval > ub_eval {
                            let tmp = lb;
                            lb = ub;
                            ub = tmp;
                            let tmp_eval = lb_eval;
                            lb_eval = ub_eval;
                            ub_eval = tmp_eval;
                        }
                        let mut t = (lb + ub) * f32::new(0.5);
                        let mut it = u32::new(0);
                        while it < u32::new(20) {
                            if t < lb || t > ub {
                                t = (lb + ub) * f32::new(0.5);
                            }
                            let value = eval_quintic(t, b, c, d, e, f);
                            if abs_f32(value) < f32::new(1.0e-5) || it == u32::new(19) {
                                break;
                            }
                            if value > zero {
                                ub = t;
                            } else {
                                lb = t;
                            }
                            let derivative = eval_quintic_deriv(t, b, c, d, e);
                            t = t - value / derivative;
                            it += u32::new(1);
                        }

                        if t >= zero && t <= one {
                            let tt = one - t;
                            let x = (tt * tt * tt) * x0
                                + (f32::new(3.0) * tt * tt * t) * x1
                                + (f32::new(3.0) * tt * t * t) * x2
                                + (t * t * t) * x3;
                            let y = (tt * tt * tt) * y0
                                + (f32::new(3.0) * tt * tt * t) * y1
                                + (f32::new(3.0) * tt * t * t) * y2
                                + (t * t * t) * y3;
                            let dist = distance_to_point(px, py, x, y);
                            if dist < best_dist {
                                best_dist = dist;
                                best_t = t;
                            }
                        }

                        if upper >= one {
                            break;
                        }
                        lower = upper;
                        idx += u32::new(1);
                    }
                }
            }
        }
    }

    let mut out = Line::empty(2usize);
    out[0] = best_dist;
    out[1] = best_t;
    out
}

#[cube]
pub(super) fn distance_to_cubic_approx(
    px: f32,
    py: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    x3: f32,
    y3: f32,
) -> f32 {
    let steps = u32::new(8);
    let inv = f32::new(1.0) / f32::cast_from(steps);
    let mut prev_x = x0;
    let mut prev_y = y0;
    let mut min_dist = distance_to_point(px, py, x0, y0);
    let mut i = u32::new(1);
    while i <= steps {
        let t = f32::cast_from(i) * inv;
        let tt = f32::new(1.0) - t;
        let tt2 = tt * tt;
        let t2 = t * t;
        let a = tt2 * tt;
        let b = f32::new(3.0) * tt2 * t;
        let c = f32::new(3.0) * tt * t2;
        let d = t2 * t;
        let cx = a * x0 + b * x1 + c * x2 + d * x3;
        let cy = a * y0 + b * y1 + c * y2 + d * y3;
        let dist = distance_to_segment(px, py, prev_x, prev_y, cx, cy);
        if dist < min_dist {
            min_dist = dist;
        }
        prev_x = cx;
        prev_y = cy;
        i += u32::new(1);
    }
    min_dist
}

#[cube]
pub(super) fn eval_quintic(t: f32, b: f32, c: f32, d: f32, e: f32, f: f32) -> f32 {
    ((((t + b) * t + c) * t + d) * t + e) * t + f
}

#[cube]
pub(super) fn eval_quintic_deriv(t: f32, b: f32, c: f32, d: f32, e: f32) -> f32 {
    (((f32::new(5.0) * t + f32::new(4.0) * b) * t + f32::new(3.0) * c) * t
        + f32::new(2.0) * d)
        * t
        + e
}

#[cube]
pub(super) fn quadratic_closest_pt_approx(
    px: f32,
    py: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
) -> Line<f32> {
    let zero = f32::new(0.0);
    let two = f32::new(2.0);
    let b0x = x0 - px;
    let b0y = y0 - py;
    let b1x = x1 - px;
    let b1y = y1 - py;
    let b2x = x2 - px;
    let b2y = y2 - py;

    let a = det2(b0x, b0y, b2x, b2y);
    let b = two * det2(b1x, b1y, b0x, b0y);
    let d = two * det2(b2x, b2y, b1x, b1y);
    let f = b * d - a * a;

    let d21x = b2x - b1x;
    let d21y = b2y - b1y;
    let d10x = b1x - b0x;
    let d10y = b1y - b0y;
    let d20x = b2x - b0x;
    let d20y = b2y - b0y;

    let mut gfx = two * (b * d21x + d * d10x + a * d20x);
    let mut gfy = two * (b * d21y + d * d10y + a * d20y);
    // rotate 90 degrees
    let tmp = gfx;
    gfx = gfy;
    gfy = -tmp;

    let gf_dot = gfx * gfx + gfy * gfy;
    let mut t = zero;
    if gf_dot > f32::new(1.0e-8) {
        let ppx = -f * gfx / gf_dot;
        let ppy = -f * gfy / gf_dot;
        let d0px = b0x - ppx;
        let d0py = b0y - ppy;
        let ap = det2(d0px, d0py, d20x, d20y);
        let bp = two * det2(d10x, d10y, d0px, d0py);
        let denom = two * a + b + d;
        if denom.abs() > f32::new(1.0e-8) {
            t = clamp01((ap + bp) / denom);
        } else {
            t = clamp01((ap + bp) * f32::new(0.5));
        }
    }

    let one = f32::new(1.0);
    let tt = one - t;
    let mut out = Line::empty(4usize);
    out[0] = (tt * tt) * x0 + (two * tt * t) * x1 + (t * t) * x2;
    out[1] = (tt * tt) * y0 + (two * tt * t) * y1 + (t * t) * y2;
    out[2] = t;
    out[3] = f32::new(0.0);
    out
}

#[cube]
pub(super) fn det2(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    ax * by - ay * bx
}

#[cube]
pub(super) fn solve_quadratic(a: f32, b: f32, c: f32) -> Line<f32> {
    let mut out = Line::empty(4usize);
    let zero = f32::new(0.0);
    let discrim = b * b - f32::new(4.0) * a * c;
    if discrim < zero {
        out[0] = zero;
        out[1] = zero;
        out[2] = zero;
        out[3] = zero;
    } else {
        let root = discrim.sqrt();
        let half = f32::new(0.5);
        let q = if b < zero { -half * (b - root) } else { -half * (b + root) };
        let t0 = q / a;
        let t1 = c / q;
        let mut lo = t0;
        let mut hi = t1;
        if lo > hi {
            let tmp = lo;
            lo = hi;
            hi = tmp;
        }
        out[0] = f32::new(2.0);
        out[1] = lo;
        out[2] = hi;
        out[3] = zero;
    }
    out
}

#[cube]
pub(super) fn solve_cubic(a: f32, b: f32, c: f32, d: f32) -> Line<f32> {
    let mut out = Line::empty(4usize);
    let zero = f32::new(0.0);
    let eps = f32::new(1.0e-6);
    if abs_f32(a) < eps {
        let roots = solve_quadratic(b, c, d);
        out[0] = roots[0];
        out[1] = roots[1];
        out[2] = roots[2];
        out[3] = zero;
    } else {
        let bb = b / a;
        let cc = c / a;
        let dd = d / a;

        let third = f32::new(1.0 / 3.0);
        let q = (bb * bb - f32::new(3.0) * cc) / f32::new(9.0);
        let r = (f32::new(2.0) * bb * bb * bb - f32::new(9.0) * bb * cc + f32::new(27.0) * dd)
            / f32::new(54.0);
        let r2 = r * r;
        let q3 = q * q * q;
        if r2 < q3 {
            let theta = (r / q3.sqrt()).acos();
            let two = f32::new(2.0);
            let sqrt_q = q.sqrt();
            out[0] = f32::new(3.0);
            out[1] = -two * sqrt_q * (theta * third).cos() - bb * third;
            out[2] =
                -two * sqrt_q * ((theta + f32::new(2.0) * f32::new(3.14159265)) * third).cos()
                    - bb * third;
            out[3] =
                -two * sqrt_q * ((theta - f32::new(2.0) * f32::new(3.14159265)) * third).cos()
                    - bb * third;
        } else {
            let a_root = if r > zero {
                -cbrt(r + (r2 - q3).sqrt())
            } else {
                cbrt(-r + (r2 - q3).sqrt())
            };
            let b_root = if abs_f32(a_root) > eps { q / a_root } else { zero };
            out[0] = f32::new(1.0);
            out[1] = (a_root + b_root) - bb * third;
            out[2] = zero;
            out[3] = zero;
        }
    }

    out
}

#[cube]
pub(super) fn cbrt(x: f32) -> f32 {
    let zero = f32::new(0.0);
    let one_third = f32::new(1.0 / 3.0);
    if x > zero {
        x.powf(one_third)
    } else if x < zero {
        -(-x).powf(one_third)
    } else {
        zero
    }
}
