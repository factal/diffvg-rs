//! GPU kernels for diffvg-rs rendering and distance evaluation.

mod backward;
mod binning;
mod curves;
mod distance;
mod entrypoints;
mod forward;
mod math;
mod rng;
mod sampling;
mod shape_sdf;

pub(crate) use entrypoints::*;
