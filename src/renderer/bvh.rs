//! CPU-side BVH packing helpers for GPU rendering.

use crate::math::Vec2;
use crate::path_utils::bounds_from_points;

use super::constants::{BVH_LEAF_SIZE, BVH_NONE};
use super::path::CurveSegment;

/// Axis-aligned bounds used for BVH construction and packing.
#[derive(Debug, Copy, Clone)]
pub(crate) struct Bounds {
    /// Minimum corner.
    pub(crate) min: Vec2,
    /// Maximum corner.
    pub(crate) max: Vec2,
}

impl Bounds {
    /// Return an empty bounds that can be grown via `include`.
    pub(crate) fn empty() -> Self {
        Self {
            min: Vec2::new(f32::INFINITY, f32::INFINITY),
            max: Vec2::new(f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }

    /// Expand this bounds to include another.
    pub(crate) fn include(&mut self, other: Bounds) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    /// Center point of the bounds.
    pub(crate) fn center(&self) -> Vec2 {
        (self.min + self.max) * 0.5
    }

    /// Extent (max - min) of the bounds.
    pub(crate) fn extent(&self) -> Vec2 {
        self.max - self.min
    }
}

/// CPU-side BVH node used for packing GPU traversal data.
#[derive(Debug, Copy, Clone)]
pub(crate) struct BvhNode {
    /// Node bounds covering all child items.
    pub(crate) bounds: Bounds,
    /// Index of the left child, or `BVH_NONE` for leaves.
    pub(crate) left: u32,
    /// Index of the right child, or `BVH_NONE` for leaves.
    pub(crate) right: u32,
    /// Skip pointer for iterative traversal on the GPU.
    pub(crate) skip: u32,
    /// Start index into the leaf index array.
    pub(crate) start: u32,
    /// Number of items in the leaf (0 for interior nodes).
    pub(crate) count: u32,
}

/// Append a per-group BVH and return `[node_offset, node_count, index_offset, index_count]`.
pub(crate) fn append_group_bvh(
    group_shape_indices: &[u32],
    shape_bounds_list: &[Option<Bounds>],
    out_bounds: &mut Vec<f32>,
    out_nodes: &mut Vec<u32>,
    out_indices: &mut Vec<u32>,
) -> [u32; 4] {
    let node_offset = (out_nodes.len() / 4) as u32;
    let index_offset = out_indices.len() as u32;
    let mut items = Vec::new();
    for &shape_index in group_shape_indices {
        let bounds = shape_bounds_list
            .get(shape_index as usize)
            .copied()
            .flatten();
        if let Some(bounds) = bounds {
            items.push((shape_index, bounds));
        }
    }
    if items.is_empty() {
        return [0u32; 4];
    }
    let mut bounds_list = Vec::with_capacity(items.len());
    for item in &items {
        bounds_list.push(item.1);
    }
    let mut indices = (0..items.len()).collect::<Vec<_>>();
    let mut nodes = Vec::new();
    let index_len = indices.len();
    let root = build_bvh_node(&mut nodes, &bounds_list, &mut indices, 0, index_len);
    assign_bvh_skip(&mut nodes, root, BVH_NONE);
    for idx in &indices {
        out_indices.push(items[*idx].0);
    }
    for node in &nodes {
        out_bounds.extend_from_slice(&[
            node.bounds.min.x,
            node.bounds.min.y,
            node.bounds.max.x,
            node.bounds.max.y,
        ]);
        out_nodes.extend_from_slice(&[node.left, node.skip, node.start, node.count]);
    }
    let node_count = nodes.len() as u32;
    let index_count = index_len as u32;
    [node_offset, node_count, index_offset, index_count]
}

/// Append a per-path BVH and return `[node_offset, node_count, index_offset, index_count]`.
pub(crate) fn append_path_bvh(
    curve_segments: &[CurveSegment],
    curve_offset: u32,
    out_bounds: &mut Vec<f32>,
    out_nodes: &mut Vec<u32>,
    out_indices: &mut Vec<u32>,
) -> [u32; 4] {
    if curve_segments.is_empty() {
        return [0u32; 4];
    }
    let node_offset = (out_nodes.len() / 4) as u32;
    let index_offset = out_indices.len() as u32;
    let mut bounds_list = Vec::with_capacity(curve_segments.len());
    for seg in curve_segments {
        bounds_list.push(curve_segment_bounds(seg));
    }
    let mut indices = (0..bounds_list.len()).collect::<Vec<_>>();
    let mut nodes = Vec::new();
    let index_len = indices.len();
    let root = build_bvh_node(&mut nodes, &bounds_list, &mut indices, 0, index_len);
    assign_bvh_skip(&mut nodes, root, BVH_NONE);
    for idx in &indices {
        out_indices.push(curve_offset + *idx as u32);
    }
    for node in &nodes {
        out_bounds.extend_from_slice(&[
            node.bounds.min.x,
            node.bounds.min.y,
            node.bounds.max.x,
            node.bounds.max.y,
        ]);
        out_nodes.extend_from_slice(&[node.left, node.skip, node.start, node.count]);
    }
    let node_count = nodes.len() as u32;
    let index_count = index_len as u32;
    [node_offset, node_count, index_offset, index_count]
}

fn build_bvh_node(
    nodes: &mut Vec<BvhNode>,
    bounds: &[Bounds],
    indices: &mut [usize],
    start: usize,
    end: usize,
) -> u32 {
    let mut node_bounds = Bounds::empty();
    for idx in &indices[start..end] {
        node_bounds.include(bounds[*idx]);
    }
    let count = end - start;
    let node_index = nodes.len() as u32;
    nodes.push(BvhNode {
        bounds: node_bounds,
        left: BVH_NONE,
        right: BVH_NONE,
        skip: BVH_NONE,
        start: start as u32,
        count: count as u32,
    });
    if count <= BVH_LEAF_SIZE {
        return node_index;
    }
    let extent = node_bounds.extent();
    let axis = if extent.x >= extent.y { 0 } else { 1 };
    indices[start..end].sort_by(|a, b| {
        let ca = bounds[*a].center();
        let cb = bounds[*b].center();
        let va = if axis == 0 { ca.x } else { ca.y };
        let vb = if axis == 0 { cb.x } else { cb.y };
        va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mid = start + count / 2;
    let left = build_bvh_node(nodes, bounds, indices, start, mid);
    let right = build_bvh_node(nodes, bounds, indices, mid, end);
    let node = &mut nodes[node_index as usize];
    node.left = left;
    node.right = right;
    node.count = 0;
    node_index
}

fn assign_bvh_skip(nodes: &mut [BvhNode], node_index: u32, skip: u32) {
    if nodes.is_empty() || node_index == BVH_NONE {
        return;
    }
    let mut stack = Vec::new();
    stack.push((node_index, skip));
    while let Some((node_id, node_skip)) = stack.pop() {
        if node_id == BVH_NONE {
            continue;
        }
        let node = &mut nodes[node_id as usize];
        node.skip = node_skip;
        if node.count == 0 {
            let left = node.left;
            let right = node.right;
            if right != BVH_NONE {
                stack.push((right, node_skip));
            }
            if left != BVH_NONE {
                stack.push((left, right));
            }
        }
    }
}

fn curve_segment_bounds(seg: &CurveSegment) -> Bounds {
    let (min, max) = bounds_from_points(&[seg.p0, seg.p1, seg.p2, seg.p3]);
    let mut pad = seg.r0.abs().max(seg.r1.abs()).max(seg.r2.abs());
    pad = pad.max(seg.r3.abs());
    let pad = pad.max(0.0);
    Bounds {
        min: Vec2::new(min.x - pad, min.y - pad),
        max: Vec2::new(max.x + pad, max.y + pad),
    }
}
