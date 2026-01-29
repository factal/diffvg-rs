use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

const EPSILON: f32 = 1.0e-6;

#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y
    }

    pub fn cross(self, other: Self) -> f32 {
        self.x * other.y - self.y * other.x
    }

    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    pub fn min(self, other: Self) -> Self {
        Self::new(self.x.min(other.x), self.y.min(other.y))
    }

    pub fn max(self, other: Self) -> Self {
        Self::new(self.x.max(other.x), self.y.max(other.y))
    }

    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self::new(self.x + (other.x - self.x) * t, self.y + (other.y - self.y) * t)
    }
}

impl Add for Vec2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl AddAssign for Vec2 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl Sub for Vec2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl SubAssign for Vec2 {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl Mul<f32> for Vec2 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

impl MulAssign<f32> for Vec2 {
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl Div<f32> for Vec2 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs)
    }
}

impl DivAssign<f32> for Vec2 {
    fn div_assign(&mut self, rhs: f32) {
        self.x /= rhs;
        self.y /= rhs;
    }
}

#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 0.0,
    };

    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Mat3 {
    pub m: [[f32; 3]; 3],
}

impl Default for Mat3 {
    fn default() -> Self {
        Self::identity()
    }
}

impl Mat3 {
    pub const fn identity() -> Self {
        Self {
            m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    pub fn translate(tx: f32, ty: f32) -> Self {
        Self {
            m: [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]],
        }
    }

    pub fn scale(sx: f32, sy: f32) -> Self {
        Self {
            m: [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    pub fn rotate(radians: f32) -> Self {
        let (sin, cos) = radians.sin_cos();
        Self {
            m: [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    pub fn mul(self, other: Self) -> Self {
        let mut out = Self { m: [[0.0; 3]; 3] };
        for r in 0..3 {
            for c in 0..3 {
                out.m[r][c] = self.m[r][0] * other.m[0][c]
                    + self.m[r][1] * other.m[1][c]
                    + self.m[r][2] * other.m[2][c];
            }
        }
        out
    }

    pub fn transform_point(&self, v: Vec2) -> Vec2 {
        let x = self.m[0][0] * v.x + self.m[0][1] * v.y + self.m[0][2];
        let y = self.m[1][0] * v.x + self.m[1][1] * v.y + self.m[1][2];
        Vec2::new(x, y)
    }

    pub fn transform_vector(&self, v: Vec2) -> Vec2 {
        let x = self.m[0][0] * v.x + self.m[0][1] * v.y;
        let y = self.m[1][0] * v.x + self.m[1][1] * v.y;
        Vec2::new(x, y)
    }

    pub fn is_identity(&self) -> bool {
        (self.m[0][0] - 1.0).abs() < EPSILON
            && self.m[0][1].abs() < EPSILON
            && self.m[0][2].abs() < EPSILON
            && self.m[1][0].abs() < EPSILON
            && (self.m[1][1] - 1.0).abs() < EPSILON
            && self.m[1][2].abs() < EPSILON
            && self.m[2][0].abs() < EPSILON
            && self.m[2][1].abs() < EPSILON
            && (self.m[2][2] - 1.0).abs() < EPSILON
    }

    pub fn axis_aligned_scale_translate(&self) -> Option<(f32, f32, f32, f32)> {
        if self.m[0][1].abs() < EPSILON
            && self.m[1][0].abs() < EPSILON
            && self.m[2][0].abs() < EPSILON
            && self.m[2][1].abs() < EPSILON
            && (self.m[2][2] - 1.0).abs() < EPSILON
        {
            Some((self.m[0][0], self.m[1][1], self.m[0][2], self.m[1][2]))
        } else {
            None
        }
    }

    pub fn uniform_scale_translate(&self) -> Option<(f32, f32, f32)> {
        self.axis_aligned_scale_translate().and_then(|(sx, sy, tx, ty)| {
            if (sx - sy).abs() < EPSILON {
                Some((sx, tx, ty))
            } else {
                None
            }
        })
    }

    pub fn scale_factors(&self) -> (f32, f32) {
        let sx = (self.m[0][0] * self.m[0][0] + self.m[1][0] * self.m[1][0]).sqrt();
        let sy = (self.m[0][1] * self.m[0][1] + self.m[1][1] * self.m[1][1]).sqrt();
        (sx, sy)
    }

    pub fn average_scale(&self) -> f32 {
        let (sx, sy) = self.scale_factors();
        0.5 * (sx + sy)
    }

    pub fn max_scale(&self) -> f32 {
        let (sx, sy) = self.scale_factors();
        sx.max(sy)
    }

    pub fn inverse(&self) -> Option<Self> {
        let a = self.m[0][0];
        let b = self.m[0][1];
        let c = self.m[1][0];
        let d = self.m[1][1];
        let tx = self.m[0][2];
        let ty = self.m[1][2];
        let det = a * d - b * c;
        if det.abs() < EPSILON {
            return None;
        }
        let inv_det = 1.0 / det;
        let m00 = d * inv_det;
        let m01 = -b * inv_det;
        let m10 = -c * inv_det;
        let m11 = a * inv_det;
        let m02 = -(m00 * tx + m01 * ty);
        let m12 = -(m10 * tx + m11 * ty);
        Some(Self {
            m: [[m00, m01, m02], [m10, m11, m12], [0.0, 0.0, 1.0]],
        })
    }
}
