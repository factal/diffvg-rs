//! CPU backward rendering path for diffvg-rs.

pub(crate) mod background;
pub(crate) mod boundary;
mod distance;
mod filters;
mod math;
mod paint;
mod render;
mod sampling;
mod types;

pub use render::{render_backward, render_backward_positions, BackwardOptions};
