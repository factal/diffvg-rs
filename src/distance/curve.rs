//! Curve distance helpers for CPU evaluation.

use crate::math::Vec2;

pub(crate) fn distance_to_segment(pt: Vec2, a: Vec2, b: Vec2) -> (f32, Vec2, f32) {
    let ab = b - a;
    let denom = ab.dot(ab);
    if denom <= 1.0e-10 {
        let dist = (pt - a).length();
        return (dist, a, 0.0);
    }
    let t = ((pt - a).dot(ab) / denom).clamp(0.0, 1.0);
    let cp = a + ab * t;
    let dist = (pt - cp).length();
    (dist, cp, t)
}

pub(crate) fn closest_point_quadratic(
    pt: Vec2,
    p0: Vec2,
    p1: Vec2,
    p2: Vec2,
    use_distance_approx: bool,
) -> (Vec2, f32, f32) {
    if use_distance_approx {
        let (cp, t) = quadratic_closest_pt_approx(p0, p1, p2, pt);
        return (cp, t, (pt - cp).length());
    }

    let mut best_t = 0.0;
    let mut best_pt = p0;
    let mut best_dist = (pt - p0).length();
    let dist_end = (pt - p2).length();
    if dist_end < best_dist {
        best_dist = dist_end;
        best_pt = p2;
        best_t = 1.0;
    }

    let ax = (p0.x - 2.0 * p1.x + p2.x) as f64;
    let ay = (p0.y - 2.0 * p1.y + p2.y) as f64;
    let bx = (-p0.x + p1.x) as f64;
    let by = (-p0.y + p1.y) as f64;
    let cx = (p0.x - pt.x) as f64;
    let cy = (p0.y - pt.y) as f64;

    let a = ax * ax + ay * ay;
    let b = 3.0 * (ax * bx + ay * by);
    let c = 2.0 * (bx * bx + by * by) + (ax * cx + ay * cy);
    let d = bx * cx + by * cy;

    let mut roots = [0.0f64; 3];
    let num = solve_cubic(a, b, c, d, &mut roots);
    for i in 0..num {
        let t = roots[i];
        if t >= 0.0 && t <= 1.0 {
            let t32 = t as f32;
            let tt = 1.0 - t32;
            let cp = (p0 * (tt * tt)) + (p1 * (2.0 * tt * t32)) + (p2 * (t32 * t32));
            let dist = (pt - cp).length();
            if dist < best_dist {
                best_dist = dist;
                best_pt = cp;
                best_t = t32;
            }
        }
    }

    (best_pt, best_t, best_dist)
}

fn closest_point_cubic_approx(
    pt: Vec2,
    p0: Vec2,
    p1: Vec2,
    p2: Vec2,
    p3: Vec2,
) -> (Vec2, f32, f32) {
    let steps = 8usize;
    let inv = 1.0 / steps as f32;
    let mut prev = p0;
    let mut prev_t = 0.0f32;
    let mut best_pt = p0;
    let mut best_t = 0.0f32;
    let mut best_dist = (pt - p0).length();
    for i in 1..=steps {
        let t = i as f32 * inv;
        let tt = 1.0 - t;
        let tt2 = tt * tt;
        let t2 = t * t;
        let a = tt2 * tt;
        let b = 3.0 * tt2 * t;
        let c = 3.0 * tt * t2;
        let d = t2 * t;
        let curr = Vec2::new(
            a * p0.x + b * p1.x + c * p2.x + d * p3.x,
            a * p0.y + b * p1.y + c * p2.y + d * p3.y,
        );
        let (dist, cp, seg_t) = distance_to_segment(pt, prev, curr);
        let local_t = prev_t + seg_t * (t - prev_t);
        if dist < best_dist {
            best_dist = dist;
            best_pt = cp;
            best_t = local_t;
        }
        prev = curr;
        prev_t = t;
    }
    (best_pt, best_t, best_dist)
}

pub(crate) fn closest_point_cubic(
    pt: Vec2,
    p0: Vec2,
    p1: Vec2,
    p2: Vec2,
    p3: Vec2,
    use_distance_approx: bool,
) -> (Vec2, f32, f32) {
    if use_distance_approx {
        return closest_point_cubic_approx(pt, p0, p1, p2, p3);
    }
    let mut best_t = 0.0;
    let mut best_pt = p0;
    let mut best_dist = (pt - p0).length();
    let dist_end = (pt - p3).length();
    if dist_end < best_dist {
        best_dist = dist_end;
        best_pt = p3;
        best_t = 1.0;
    }

    let ax = (-p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x) as f64;
    let ay = (-p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y) as f64;
    let bx = (3.0 * p0.x - 6.0 * p1.x + 3.0 * p2.x) as f64;
    let by = (3.0 * p0.y - 6.0 * p1.y + 3.0 * p2.y) as f64;
    let cx = (-3.0 * p0.x + 3.0 * p1.x) as f64;
    let cy = (-3.0 * p0.y + 3.0 * p1.y) as f64;
    let dx = (p0.x - pt.x) as f64;
    let dy = (p0.y - pt.y) as f64;

    let a = 3.0 * (ax * ax + ay * ay);
    if a.abs() < 1.0e-10 {
        return (best_pt, best_t, best_dist);
    }
    let b = 5.0 * (ax * bx + ay * by);
    let c = 4.0 * (ax * cx + ay * cy) + 2.0 * (bx * bx + by * by);
    let d = 3.0 * ((bx * cx + by * cy) + (ax * dx + ay * dy));
    let e = (cx * cx + cy * cy) + 2.0 * (dx * bx + dy * by);
    let f = dx * cx + dy * cy;

    let b = b / a;
    let c = c / a;
    let d = d / a;
    let e = e / a;
    let f = f / a;

    let p1a = (2.0 / 5.0) * c - (4.0 / 25.0) * b * b;
    let p1b = (3.0 / 5.0) * d - (3.0 / 25.0) * b * c;
    let p1c = (4.0 / 5.0) * e - (2.0 / 25.0) * b * d;
    let p1d = f - b * e / 25.0;

    let q_root = -b / 5.0;
    let mut p_roots = [0.0f64; 3];
    let num_p = solve_cubic(p1a, p1b, p1c, p1d, &mut p_roots);

    let mut intervals = [0.0f64; 4];
    let mut num_intervals = 0usize;
    if q_root >= 0.0 && q_root <= 1.0 {
        intervals[num_intervals] = q_root;
        num_intervals += 1;
    }
    for i in 0..num_p {
        intervals[num_intervals] = p_roots[i];
        num_intervals += 1;
    }

    for j in 1..num_intervals {
        let mut k = j;
        while k > 0 && intervals[k - 1] > intervals[k] {
            intervals.swap(k - 1, k);
            k -= 1;
        }
    }

    // Root finding on the quintic derivative polynomial to refine closest t.
    let eval_poly = |t: f64| -> f64 { ((((t + b) * t + c) * t + d) * t + e) * t + f };
    let eval_poly_deriv = |t: f64| -> f64 {
        (((5.0 * t + 4.0 * b) * t + 3.0 * c) * t + 2.0 * d) * t + e
    };

    let mut lower = 0.0;
    for j in 0..=num_intervals {
        if j < num_intervals && intervals[j] < 0.0 {
            continue;
        }
        let upper = if j < num_intervals {
            intervals[j].min(1.0)
        } else {
            1.0
        };
        let mut lb = lower;
        let mut ub = upper;
        let mut lb_eval = eval_poly(lb);
        let mut ub_eval = eval_poly(ub);
        if lb_eval * ub_eval > 0.0 {
            lower = upper;
            continue;
        }
        if lb_eval > ub_eval {
            std::mem::swap(&mut lb, &mut ub);
            std::mem::swap(&mut lb_eval, &mut ub_eval);
        }
        let mut t = 0.5 * (lb + ub);
        for it in 0..20 {
            if !(t >= lb && t <= ub) {
                t = 0.5 * (lb + ub);
            }
            let value = eval_poly(t);
            if value.abs() < 1.0e-5 || it == 19 {
                break;
            }
            if value > 0.0 {
                ub = t;
            } else {
                lb = t;
            }
            let derivative = eval_poly_deriv(t);
            t -= value / derivative;
        }
        if t >= 0.0 && t <= 1.0 {
            let t32 = t as f32;
            let tt = 1.0 - t32;
            let cp = (p0 * (tt * tt * tt))
                + (p1 * (3.0 * tt * tt * t32))
                + (p2 * (3.0 * tt * t32 * t32))
                + (p3 * (t32 * t32 * t32));
            let dist = (pt - cp).length();
            if dist < best_dist {
                best_dist = dist;
                best_pt = cp;
                best_t = t32;
            }
        }
        if upper >= 1.0 {
            break;
        }
        lower = upper;
    }

    (best_pt, best_t, best_dist)
}

fn quadratic_closest_pt_approx(p0: Vec2, p1: Vec2, p2: Vec2, pt: Vec2) -> (Vec2, f32) {
    let b0 = p0 - pt;
    let b1 = p1 - pt;
    let b2 = p2 - pt;
    let a = b0.cross(b2);
    let b = 2.0 * b1.cross(b0);
    let d = 2.0 * b2.cross(b1);
    let f = b * d - a * a;
    let d21 = b2 - b1;
    let d10 = b1 - b0;
    let d20 = b2 - b0;
    let mut gf = (d21 * b) + (d10 * d) + (d20 * a);
    gf *= 2.0;
    gf = Vec2::new(gf.y, -gf.x);
    let mut t = 0.0;
    let denom = gf.dot(gf);
    if denom > 1.0e-8 {
        let pp = gf * (-f / denom);
        let d0p = b0 - pp;
        let ap = d0p.cross(d20);
        let bp = 2.0 * d10.cross(d0p);
        let denom2 = 2.0 * a + b + d;
        if denom2.abs() > 1.0e-8 {
            t = ((ap + bp) / denom2).clamp(0.0, 1.0);
        }
    }
    let tt = 1.0 - t;
    let cp = p0 * (tt * tt) + p1 * (2.0 * tt * t) + p2 * (t * t);
    (cp, t)
}

fn solve_quadratic(a: f64, b: f64, c: f64, t: &mut [f64; 2]) -> bool {
    let discrim = b * b - 4.0 * a * c;
    if discrim < 0.0 {
        return false;
    }
    let root = discrim.sqrt();
    let q = if b < 0.0 { -0.5 * (b - root) } else { -0.5 * (b + root) };
    t[0] = q / a;
    t[1] = c / q;
    if t[0] > t[1] {
        t.swap(0, 1);
    }
    true
}

fn solve_cubic(a: f64, b: f64, c: f64, d: f64, t: &mut [f64; 3]) -> usize {
    if a.abs() < 1.0e-6 {
        let mut roots = [0.0f64; 2];
        if solve_quadratic(b, c, d, &mut roots) {
            t[0] = roots[0];
            t[1] = roots[1];
            return 2;
        }
        return 0;
    }
    let b = b / a;
    let c = c / a;
    let d = d / a;
    let q = (b * b - 3.0 * c) / 9.0;
    let r = (2.0 * b * b * b - 9.0 * b * c + 27.0 * d) / 54.0;
    if r * r < q * q * q {
        let theta = (r / (q * q * q).sqrt()).acos();
        t[0] = -2.0 * q.sqrt() * (theta / 3.0).cos() - b / 3.0;
        t[1] = -2.0 * q.sqrt() * ((theta + 2.0 * std::f64::consts::PI) / 3.0).cos() - b / 3.0;
        t[2] = -2.0 * q.sqrt() * ((theta - 2.0 * std::f64::consts::PI) / 3.0).cos() - b / 3.0;
        return 3;
    }
    let a_root = if r > 0.0 {
        -cbrt(r + (r * r - q * q * q).sqrt())
    } else {
        cbrt(-r + (r * r - q * q * q).sqrt())
    };
    let b_root = if a_root.abs() > 1.0e-6 { q / a_root } else { 0.0 };
    t[0] = (a_root + b_root) - b / 3.0;
    1
}

fn cbrt(x: f64) -> f64 {
    if x > 0.0 {
        x.powf(1.0 / 3.0)
    } else if x < 0.0 {
        -(-x).powf(1.0 / 3.0)
    } else {
        0.0
    }
}
