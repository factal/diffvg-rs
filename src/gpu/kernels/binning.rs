//! GPU tile binning kernels for group-level culling and scheduling.

use cubecl::prelude::*;
use crate::gpu::constants::*;
use super::math::*;

/// Per-group pass to count how many tiles each group touches.
/// Transforms group bounds to canvas space and atomically increments `tile_counts`.
#[cube(launch_unchecked)]
pub(crate) fn bin_tiles_count(
    group_bounds: &Array<f32>,
    group_shape_xform: &Array<f32>,
    num_groups: u32,
    tile_count_x: u32,
    tile_count_y: u32,
    tile_size: u32,
    width: u32,
    height: u32,
    tile_counts: &mut Array<Atomic<u32>>,
) {
    let idx = ABSOLUTE_POS;
    if idx >= num_groups as usize {
        terminate!();
    }

    let group_id = idx as u32;

    let bounds_base = (group_id * BOUNDS_STRIDE) as usize;
    let min_x = group_bounds[bounds_base];
    let min_y = group_bounds[bounds_base + 1];
    let max_x = group_bounds[bounds_base + 2];
    let max_y = group_bounds[bounds_base + 3];

    let mut valid = min_x <= max_x && min_y <= max_y;
    if tile_count_x == u32::new(0) || tile_count_y == u32::new(0) {
        valid = false;
    }

    if valid {
        let xform_base = (group_id * XFORM_STRIDE) as usize;
        let m00 = group_shape_xform[xform_base];
        let m01 = group_shape_xform[xform_base + 1];
        let m02 = group_shape_xform[xform_base + 2];
        let m10 = group_shape_xform[xform_base + 3];
        let m11 = group_shape_xform[xform_base + 4];
        let m12 = group_shape_xform[xform_base + 5];

        let x0 = m00 * min_x + m01 * min_y + m02;
        let y0 = m10 * min_x + m11 * min_y + m12;
        let x1 = m00 * max_x + m01 * min_y + m02;
        let y1 = m10 * max_x + m11 * min_y + m12;
        let x2 = m00 * max_x + m01 * max_y + m02;
        let y2 = m10 * max_x + m11 * max_y + m12;
        let x3 = m00 * min_x + m01 * max_y + m02;
        let y3 = m10 * min_x + m11 * max_y + m12;

        let min_cx = min_f32(min_f32(x0, x1), min_f32(x2, x3));
        let max_cx = max_f32(max_f32(x0, x1), max_f32(x2, x3));
        let min_cy = min_f32(min_f32(y0, y1), min_f32(y2, y3));
        let max_cy = max_f32(max_f32(y0, y1), max_f32(y2, y3));

        let zero = f32::new(0.0);
        let w = f32::cast_from(width);
        let h = f32::cast_from(height);
        let in_view = !(max_cx < zero || max_cy < zero || min_cx >= w || min_cy >= h);
        if in_view {
            let tile_size_f = f32::cast_from(tile_size);
            let mut min_tx = (min_cx / tile_size_f).floor() as i32;
            let mut max_tx = (max_cx / tile_size_f).floor() as i32;
            let mut min_ty = (min_cy / tile_size_f).floor() as i32;
            let mut max_ty = (max_cy / tile_size_f).floor() as i32;

            let max_tile_x = tile_count_x as i32 - 1;
            let max_tile_y = tile_count_y as i32 - 1;

            if min_tx < 0 {
                min_tx = 0;
            } else if min_tx > max_tile_x {
                min_tx = max_tile_x;
            }
            if max_tx < 0 {
                max_tx = 0;
            } else if max_tx > max_tile_x {
                max_tx = max_tile_x;
            }
            if min_ty < 0 {
                min_ty = 0;
            } else if min_ty > max_tile_y {
                min_ty = max_tile_y;
            }
            if max_ty < 0 {
                max_ty = 0;
            } else if max_ty > max_tile_y {
                max_ty = max_tile_y;
            }

            if min_tx <= max_tx && min_ty <= max_ty {
                let min_tx = min_tx as u32;
                let max_tx = max_tx as u32;
                let min_ty = min_ty as u32;
                let max_ty = max_ty as u32;
                for ty in min_ty..=max_ty {
                    let row = ty * tile_count_x;
                    for tx in min_tx..=max_tx {
                        let tile_id = (row + tx) as usize;
                        tile_counts[tile_id].fetch_add(u32::new(1));
                    }
                }
            }
        }
    }
}

/// Initialize prefix-sum input as an offsets array.
/// Writes `offsets[0]=0` and `offsets[i+1]=tile_counts[i]`.
#[cube(launch_unchecked)]
pub(crate) fn init_tile_offsets(
    tile_counts: &Array<Atomic<u32>>,
    num_tiles: u32,
    offsets: &mut Array<u32>,
) {
    let idx = ABSOLUTE_POS;
    if idx >= num_tiles as usize {
        terminate!();
    }

    if idx == 0 {
        offsets[0] = u32::new(0);
    }
    offsets[idx + 1] = tile_counts[idx].load();
}

/// Single scan step for tile offsets with the given stride.
/// Computes `offsets_out[i]=offsets_in[i]+offsets_in[i-stride]` when in range.
#[cube(launch_unchecked)]
pub(crate) fn scan_tile_offsets(
    offsets_in: &Array<u32>,
    offsets_out: &mut Array<u32>,
    num_entries: u32,
    stride: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= num_entries as usize {
        terminate!();
    }

    let idx_u32 = idx as u32;
    if idx_u32 >= stride {
        offsets_out[idx] = offsets_in[idx] + offsets_in[(idx_u32 - stride) as usize];
    } else {
        offsets_out[idx] = offsets_in[idx];
    }
}

/// Initialize atomic cursors from the computed tile offsets.
#[cube(launch_unchecked)]
pub(crate) fn init_tile_cursor(
    tile_offsets: &Array<u32>,
    num_tiles: u32,
    tile_cursor: &mut Array<Atomic<u32>>,
) {
    let idx = ABSOLUTE_POS;
    if idx >= num_tiles as usize {
        terminate!();
    }
    let value = tile_offsets[idx];
    tile_cursor[idx].fetch_add(value);
}

/// Per-group pass to write group ids into tile entries.
/// Uses `tile_cursor` to reserve write positions per tile.
#[cube(launch_unchecked)]
pub(crate) fn bin_tiles_write(
    group_bounds: &Array<f32>,
    group_shape_xform: &Array<f32>,
    num_groups: u32,
    tile_count_x: u32,
    tile_count_y: u32,
    tile_size: u32,
    width: u32,
    height: u32,
    tile_cursor: &mut Array<Atomic<u32>>,
    tile_entries: &mut Array<u32>,
) {
    let idx = ABSOLUTE_POS;
    if idx >= num_groups as usize {
        terminate!();
    }

    let group_id = idx as u32;

    let bounds_base = (group_id * BOUNDS_STRIDE) as usize;
    let min_x = group_bounds[bounds_base];
    let min_y = group_bounds[bounds_base + 1];
    let max_x = group_bounds[bounds_base + 2];
    let max_y = group_bounds[bounds_base + 3];

    let mut valid = min_x <= max_x && min_y <= max_y;
    if tile_count_x == u32::new(0) || tile_count_y == u32::new(0) {
        valid = false;
    }

    if valid {
        let xform_base = (group_id * XFORM_STRIDE) as usize;
        let m00 = group_shape_xform[xform_base];
        let m01 = group_shape_xform[xform_base + 1];
        let m02 = group_shape_xform[xform_base + 2];
        let m10 = group_shape_xform[xform_base + 3];
        let m11 = group_shape_xform[xform_base + 4];
        let m12 = group_shape_xform[xform_base + 5];

        let x0 = m00 * min_x + m01 * min_y + m02;
        let y0 = m10 * min_x + m11 * min_y + m12;
        let x1 = m00 * max_x + m01 * min_y + m02;
        let y1 = m10 * max_x + m11 * min_y + m12;
        let x2 = m00 * max_x + m01 * max_y + m02;
        let y2 = m10 * max_x + m11 * max_y + m12;
        let x3 = m00 * min_x + m01 * max_y + m02;
        let y3 = m10 * min_x + m11 * max_y + m12;

        let min_cx = min_f32(min_f32(x0, x1), min_f32(x2, x3));
        let max_cx = max_f32(max_f32(x0, x1), max_f32(x2, x3));
        let min_cy = min_f32(min_f32(y0, y1), min_f32(y2, y3));
        let max_cy = max_f32(max_f32(y0, y1), max_f32(y2, y3));

        let zero = f32::new(0.0);
        let w = f32::cast_from(width);
        let h = f32::cast_from(height);
        let in_view = !(max_cx < zero || max_cy < zero || min_cx >= w || min_cy >= h);
        if in_view {
            let tile_size_f = f32::cast_from(tile_size);
            let mut min_tx = (min_cx / tile_size_f).floor() as i32;
            let mut max_tx = (max_cx / tile_size_f).floor() as i32;
            let mut min_ty = (min_cy / tile_size_f).floor() as i32;
            let mut max_ty = (max_cy / tile_size_f).floor() as i32;

            let max_tile_x = tile_count_x as i32 - 1;
            let max_tile_y = tile_count_y as i32 - 1;

            if min_tx < 0 {
                min_tx = 0;
            } else if min_tx > max_tile_x {
                min_tx = max_tile_x;
            }
            if max_tx < 0 {
                max_tx = 0;
            } else if max_tx > max_tile_x {
                max_tx = max_tile_x;
            }
            if min_ty < 0 {
                min_ty = 0;
            } else if min_ty > max_tile_y {
                min_ty = max_tile_y;
            }
            if max_ty < 0 {
                max_ty = 0;
            } else if max_ty > max_tile_y {
                max_ty = max_tile_y;
            }

            if min_tx <= max_tx && min_ty <= max_ty {
                let min_tx = min_tx as u32;
                let max_tx = max_tx as u32;
                let min_ty = min_ty as u32;
                let max_ty = max_ty as u32;
                for ty in min_ty..=max_ty {
                    let row = ty * tile_count_x;
                    for tx in min_tx..=max_tx {
                        let tile_id = (row + tx) as usize;
                        let entry_index = tile_cursor[tile_id].fetch_add(u32::new(1));
                        let base = (entry_index * TILE_ENTRY_STRIDE) as usize;
                        tile_entries[base] = group_id;
                    }
                }
            }
        }
    }
}

/// In-place per-tile insertion sort of group ids for deterministic ordering.
/// No-op for tiles with zero or one entry.
#[cube(launch_unchecked)]
pub(crate) fn sort_tile_entries(
    tile_offsets: &Array<u32>,
    tile_entries: &mut Array<u32>,
    num_tiles: u32,
) {
    let tile_id = ABSOLUTE_POS;
    if tile_id >= num_tiles as usize {
        terminate!();
    }
    let start = tile_offsets[tile_id];
    let end = tile_offsets[tile_id + 1];
    let one = u32::new(1);
    if end <= start + one {
        terminate!();
    }
    let mut i = start + one;
    while i < end {
        let key = tile_entries[i as usize];
        let mut j = i;
        while j > start {
            let prev = tile_entries[(j - one) as usize];
            if prev <= key {
                break;
            }
            tile_entries[j as usize] = prev;
            j -= one;
        }
        tile_entries[j as usize] = key;
        i += one;
    }
}
