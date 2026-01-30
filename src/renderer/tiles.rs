//! Tile binning helpers for GPU rendering.

pub(crate) fn div_ceil(value: u32, divisor: u32) -> u32 {
    if divisor == 0 {
        return 0;
    }
    (value + divisor - 1) / divisor
}

pub(crate) fn sort_tile_entries(entries: &mut [u32], offsets: &[u32], num_tiles: usize) {
    if entries.is_empty() || offsets.is_empty() {
        return;
    }
    let mut cursor = Vec::with_capacity(num_tiles + 1);
    cursor.extend_from_slice(offsets);
    cursor.push(entries.len() as u32);
    for tile_id in 0..num_tiles {
        let start = cursor[tile_id] as usize;
        let end = cursor[tile_id + 1] as usize;
        if start >= end || end > entries.len() {
            continue;
        }
        entries[start..end].sort_unstable();
    }
}

pub(crate) fn build_tile_order(tile_counts: &[u32], num_tiles: u32) -> Vec<u32> {
    let mut order = (0..num_tiles).collect::<Vec<_>>();
    order.sort_by(|a, b| {
        let ca = tile_counts.get(*a as usize).copied().unwrap_or(0);
        let cb = tile_counts.get(*b as usize).copied().unwrap_or(0);
        cb.cmp(&ca)
    });
    order
}
