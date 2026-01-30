//! Color utilities for scene construction and compositing.

use crate::math::Vec4;

/// Linear RGBA color (non-premultiplied).
#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    /// Fully transparent black.
    pub const TRANSPARENT: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };

    /// Create a color from linear RGBA components.
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Create an opaque color from linear RGB components.
    pub const fn opaque(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    /// Return the same color with a new alpha value.
    pub fn with_alpha(self, a: f32) -> Self {
        Self { a, ..self }
    }

    /// Clamp all components into [0, 1].
    pub fn clamp01(self) -> Self {
        Self {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
            a: self.a.clamp(0.0, 1.0),
        }
    }

    /// Convert to a Vec4 in RGBA order.
    pub fn to_vec4(self) -> Vec4 {
        Vec4::new(self.r, self.g, self.b, self.a)
    }

    /// Porter-Duff "over" compositing against a destination color.
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
