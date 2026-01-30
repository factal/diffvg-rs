//! GPU kernels for diffvg-rs rendering and distance evaluation.

pub(crate) mod constants;
mod kernels;

pub(crate) use kernels::*;
