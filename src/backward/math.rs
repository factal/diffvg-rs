use crate::math::{Mat3, Vec2, Vec4};

/// Return the dot product of two Vec4 values.
pub(super) fn dot4(a: Vec4, b: Vec4) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
}

/// Map a signed distance in [-1, 1] to a smoothstep weight in [0, 1].
pub(super) fn smoothstep(d: f32) -> f32 {
    let t = ((d + 1.0) * 0.5).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Backpropagate through `smoothstep`, returning the gradient w.r.t. `d`.
pub(super) fn d_smoothstep(d: f32, d_ret: f32) -> f32 {
    if d < -1.0 || d > 1.0 {
        return 0.0;
    }
    let t = (d + 1.0) * 0.5;
    let d_t = d_ret * (6.0 * t - 6.0 * t * t);
    d_t * 0.5
}

/// Backpropagate through distance(a, b), accumulating into `d_a` and `d_b`.
pub(super) fn d_distance(a: Vec2, b: Vec2, d_out: f32, d_a: &mut Vec2, d_b: &mut Vec2) {
    let d = d_length(b - a, d_out);
    *d_a -= d;
    *d_b += d;
}

/// Backpropagate through vector length, returning the gradient w.r.t. `v`.
pub(super) fn d_length(v: Vec2, d_l: f32) -> Vec2 {
    let l_sq = v.dot(v);
    let l = l_sq.sqrt();
    if l == 0.0 {
        return Vec2::ZERO;
    }
    let d_l_sq = 0.5 * d_l / l;
    v * (2.0 * d_l_sq)
}

/// Normalize a vector, returning zero when its length is zero.
pub(super) fn normalize(v: Vec2) -> Vec2 {
    let len = v.length();
    if len == 0.0 {
        Vec2::ZERO
    } else {
        v / len
    }
}

/// Backpropagate through normalization, returning the gradient w.r.t. `v`.
pub(super) fn d_normalize(v: Vec2, d_n: Vec2) -> Vec2 {
    let l = v.length();
    if l == 0.0 {
        return Vec2::ZERO;
    }
    let n = v / l;
    let mut d_v = d_n / l;
    let d_l = -d_n.dot(n) / l;
    d_v += d_length(v, d_l);
    d_v
}

/// Return a 3x3 zero matrix.
pub(super) fn zero_mat3() -> Mat3 {
    Mat3 { m: [[0.0; 3]; 3] }
}

/// Add two 3x3 matrices component-wise.
pub(super) fn mat3_add(a: Mat3, b: Mat3) -> Mat3 {
    let mut out = a;
    for r in 0..3 {
        for c in 0..3 {
            out.m[r][c] += b.m[r][c];
        }
    }
    out
}

/// Scale a 3x3 matrix by a scalar.
pub(super) fn mat3_scale(m: Mat3, s: f32) -> Mat3 {
    let mut out = m;
    for r in 0..3 {
        for c in 0..3 {
            out.m[r][c] *= s;
        }
    }
    out
}

/// Multiply two 3x3 matrices (`a * b`).
pub(super) fn mat3_mul(a: Mat3, b: Mat3) -> Mat3 {
    a.mul(b)
}

/// Transpose a 3x3 matrix.
pub(super) fn mat3_transpose(m: Mat3) -> Mat3 {
    let mut out = Mat3 { m: [[0.0; 3]; 3] };
    for r in 0..3 {
        for c in 0..3 {
            out.m[r][c] = m.m[c][r];
        }
    }
    out
}

/// Backpropagate through a homogeneous point transform, accumulating into `d_m` and `d_pt`.
pub(super) fn d_xform_pt(m: Mat3, pt: Vec2, d_out: Vec2, d_m: &mut Mat3, d_pt: &mut Vec2) {
    let t0 = m.m[0][0] * pt.x + m.m[0][1] * pt.y + m.m[0][2];
    let t1 = m.m[1][0] * pt.x + m.m[1][1] * pt.y + m.m[1][2];
    let t2 = m.m[2][0] * pt.x + m.m[2][1] * pt.y + m.m[2][2];
    let out = Vec2::new(t0 / t2, t1 / t2);
    let d_t0 = d_out.x / t2;
    let d_t1 = d_out.y / t2;
    let d_t2 = -(d_out.x * out.x + d_out.y * out.y) / t2;
    d_m.m[0][0] += d_t0 * pt.x;
    d_m.m[0][1] += d_t0 * pt.y;
    d_m.m[0][2] += d_t0;
    d_m.m[1][0] += d_t1 * pt.x;
    d_m.m[1][1] += d_t1 * pt.y;
    d_m.m[1][2] += d_t1;
    d_m.m[2][0] += d_t2 * pt.x;
    d_m.m[2][1] += d_t2 * pt.y;
    d_m.m[2][2] += d_t2;
    d_pt.x += d_t0 * m.m[0][0] + d_t1 * m.m[1][0] + d_t2 * m.m[2][0];
    d_pt.y += d_t0 * m.m[0][1] + d_t1 * m.m[1][1] + d_t2 * m.m[2][1];
}

/// Transform a point by the inverse of `m`, falling back to identity if singular.
pub(super) fn transform_point_inverse(m: Mat3, pt: Vec2) -> Vec2 {
    m.inverse()
        .unwrap_or(Mat3::identity())
        .transform_point(pt)
}


/// Transform a normal using the inverse-transpose of the 2x2 linear part and renormalize.
pub(super) fn xform_normal(m_inv: Mat3, n: Vec2) -> Vec2 {
    let x = m_inv.m[0][0] * n.x + m_inv.m[1][0] * n.y;
    let y = m_inv.m[0][1] * n.x + m_inv.m[1][1] * n.y;
    normalize(Vec2::new(x, y))
}
