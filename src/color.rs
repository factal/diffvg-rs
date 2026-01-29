use crate::math::Vec4;

#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const TRANSPARENT: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };

    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub const fn opaque(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    pub fn with_alpha(self, a: f32) -> Self {
        Self { a, ..self }
    }

    pub fn clamp01(self) -> Self {
        Self {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
            a: self.a.clamp(0.0, 1.0),
        }
    }

    pub fn to_vec4(self) -> Vec4 {
        Vec4::new(self.r, self.g, self.b, self.a)
    }

    pub fn over(self, dst: Color) -> Color {
        let alpha = self.a;
        let inv = 1.0 - alpha;
        Color {
            r: self.r * alpha + dst.r * inv,
            g: self.g * alpha + dst.g * inv,
            b: self.b * alpha + dst.b * inv,
            a: alpha + dst.a * inv,
        }
    }
}
