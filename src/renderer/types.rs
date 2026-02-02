//! Public render options and output types.

use crate::color::Color;
use cubecl::prelude::LaunchError;

use super::constants::DEFAULT_PATH_TOLERANCE;

/// Rendering configuration for rasterization and SDF evaluation.
#[derive(Debug, Copy, Clone)]
pub struct RenderOptions {
    /// Anti-aliasing half-width for SDF evaluation.
    pub aa: f32,
    /// Curve flattening tolerance for CPU and preprocessing.
    pub path_tolerance: f32,
    /// Number of samples along X per pixel.
    pub samples_x: u32,
    /// Number of samples along Y per pixel.
    pub samples_y: u32,
    /// Random seed used for jitter.
    pub seed: u32,
    /// Enable subpixel jitter when prefiltering is disabled.
    pub jitter: bool,
    /// Enable prefiltering via filter weights and smoothstep coverage.
    pub use_prefiltering: bool,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            aa: 1.0,
            path_tolerance: DEFAULT_PATH_TOLERANCE,
            samples_x: 2,
            samples_y: 2,
            seed: 0,
            jitter: true,
            use_prefiltering: true,
        }
    }
}

/// RGBA image in linear color space.
#[derive(Debug, Clone)]
pub struct Image {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Linear RGBA pixels, row-major, length = width * height * 4.
    pub pixels: Vec<f32>,
}

impl Image {
    /// Create a solid color image.
    pub fn solid(width: u32, height: u32, color: Color) -> Self {
        let mut pixels = vec![0.0; width as usize * height as usize * 4];
        for chunk in pixels.chunks_mut(4) {
            chunk[0] = color.r;
            chunk[1] = color.g;
            chunk[2] = color.b;
            chunk[3] = color.a;
        }
        Self {
            width,
            height,
            pixels,
        }
    }
}

/// Signed distance field image for the current scene.
#[derive(Debug, Clone)]
pub struct SdfImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Signed distance values, row-major, length = width * height.
    pub values: Vec<f32>,
}

/// Render-time error conditions.
#[derive(Debug)]
pub enum RenderError {
    /// Scene parameters are inconsistent or overflow the renderer limits.
    InvalidScene(&'static str),
    /// GPU kernel launch failed.
    Launch(LaunchError),
}
