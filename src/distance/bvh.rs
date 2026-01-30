//! BVH data structures and builders for CPU distance queries.

use crate::math::Vec2;
use crate::scene::{Scene, Shape, ShapeGeometry};
use crate::path_utils::path_point_radius;

use super::shape::{segment_bounds, shape_bounds};

pub(crate) const BVH_NONE: u32 = u32::MAX;
const BVH_LEAF_SIZE: usize = 8;

/// Per-group BVH acceleration structure for distance and winding queries.
#[derive(Debug, Clone)]
pub struct SceneBvh {
    pub(crate) groups: Vec<GroupBvh>,
}

impl SceneBvh {
    /// Build a BVH for each shape group in the scene.
    pub fn new(scene: &Scene) -> Self {
        let mut groups = Vec::with_capacity(scene.groups.len());
        for group in &scene.groups {
            groups.push(GroupBvh::build(scene, group));
        }
        Self { groups }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct GroupBvh {
    pub(crate) nodes: Vec<BvhNode>,
    pub(crate) indices: Vec<usize>,
    pub(crate) shapes: Vec<BvhShape>,
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct BvhNode {
    pub(crate) bounds: Bounds,
    pub(crate) left: u32,
    pub(crate) right: u32,
    pub(crate) start: u32,
    pub(crate) count: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct BvhShape {
    pub(crate) shape_index: usize,
    pub(crate) bounds: Bounds,
    pub(crate) path_bvh: Option<PathBvh>,
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct Bounds {
    pub(crate) min: Vec2,
    pub(crate) max: Vec2,
}

impl Bounds {
    pub(crate) fn empty() -> Self {
        Self {
            min: Vec2::new(f32::INFINITY, f32::INFINITY),
            max: Vec2::new(f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }

    pub(crate) fn include(&mut self, other: Bounds) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    pub(crate) fn center(&self) -> Vec2 {
        (self.min + self.max) * 0.5
    }

    pub(crate) fn extent(&self) -> Vec2 {
        self.max - self.min
    }

    pub(crate) fn contains(&self, pt: Vec2) -> bool {
        pt.x >= self.min.x && pt.x <= self.max.x && pt.y >= self.min.y && pt.y <= self.max.y
    }

    pub(crate) fn distance(&self, pt: Vec2) -> f32 {
        let dx = (self.min.x - pt.x).max(0.0).max(pt.x - self.max.x);
        let dy = (self.min.y - pt.y).max(0.0).max(pt.y - self.max.y);
        (dx * dx + dy * dy).sqrt()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PathBvh {
    pub(crate) nodes: Vec<BvhNode>,
    pub(crate) indices: Vec<usize>,
    pub(crate) segments: Vec<PathSegmentData>,
    pub(crate) use_distance_approx: bool,
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct PathSegmentData {
    pub(crate) segment_index: usize,
    pub(crate) kind: u8,
    pub(crate) p0: Vec2,
    pub(crate) p1: Vec2,
    pub(crate) p2: Vec2,
    pub(crate) p3: Vec2,
    pub(crate) r0: f32,
    pub(crate) r1: f32,
    pub(crate) r2: f32,
    pub(crate) r3: f32,
    pub(crate) bounds: Bounds,
}

impl GroupBvh {
    pub(crate) fn build(scene: &Scene, group: &crate::scene::ShapeGroup) -> Self {
        let mut shapes = Vec::new();
        for &shape_index in &group.shape_indices {
            let Some(shape) = scene.shapes.get(shape_index) else {
                continue;
            };
            if let Some(bounds) = shape_bounds(shape) {
                let path_bvh = build_path_bvh(shape);
                shapes.push(BvhShape {
                    shape_index,
                    bounds,
                    path_bvh,
                });
            }
        }

        let mut indices = (0..shapes.len()).collect::<Vec<_>>();
        let mut nodes = Vec::new();
        if !indices.is_empty() {
            let len = indices.len();
            build_bvh_node(&mut nodes, &shapes, &mut indices, 0, len);
        }

        Self {
            nodes,
            indices,
            shapes,
        }
    }
}

fn build_bvh_node(
    nodes: &mut Vec<BvhNode>,
    shapes: &[BvhShape],
    indices: &mut [usize],
    start: usize,
    end: usize,
) -> u32 {
    let mut bounds = Bounds::empty();
    for idx in &indices[start..end] {
        bounds.include(shapes[*idx].bounds);
    }

    let count = end - start;
    let node_index = nodes.len() as u32;
    nodes.push(BvhNode {
        bounds,
        left: BVH_NONE,
        right: BVH_NONE,
        start: start as u32,
        count: count as u32,
    });

    if count <= BVH_LEAF_SIZE {
        return node_index;
    }

    // Split along the longest axis to balance the tree.
    let extent = bounds.extent();
    let axis = if extent.x >= extent.y { 0 } else { 1 };

    indices[start..end].sort_by(|a, b| {
        let ca = shapes[*a].bounds.center();
        let cb = shapes[*b].bounds.center();
        let va = if axis == 0 { ca.x } else { ca.y };
        let vb = if axis == 0 { cb.x } else { cb.y };
        va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mid = start + count / 2;
    let left = build_bvh_node(nodes, shapes, indices, start, mid);
    let right = build_bvh_node(nodes, shapes, indices, mid, end);

    let node = &mut nodes[node_index as usize];
    node.left = left;
    node.right = right;
    node.count = 0;

    node_index
}

fn build_path_bvh(shape: &Shape) -> Option<PathBvh> {
    let ShapeGeometry::Path { path } = &shape.geometry else {
        return None;
    };

    if path.points.is_empty() {
        return None;
    }

    let default_radius = shape.stroke_width;
    let total_points = path.points.len();
    let mut point_id = 0usize;
    let mut segments = Vec::new();

    for (segment_index, &num_controls) in path.num_control_points.iter().enumerate() {
        match num_controls {
            0 => {
                let i0 = point_id;
                let i1 = point_id + 1;
                let (p0, r0) =
                    match path_point_radius(path, i0, total_points, default_radius, 1.0) {
                        Some(value) => value,
                        None => break,
                    };
                let (p1, r1) =
                    match path_point_radius(path, i1, total_points, default_radius, 1.0) {
                        Some(value) => value,
                        None => break,
                    };
                let bounds = segment_bounds(&[p0, p1], r0.max(r1));
                segments.push(PathSegmentData {
                    segment_index,
                    kind: 0,
                    p0,
                    p1,
                    p2: p1,
                    p3: p1,
                    r0,
                    r1,
                    r2: r1,
                    r3: r1,
                    bounds,
                });
                point_id += 1;
            }
            1 => {
                let i0 = point_id;
                let i1 = point_id + 1;
                let i2 = point_id + 2;
                let (p0, r0) =
                    match path_point_radius(path, i0, total_points, default_radius, 1.0) {
                        Some(value) => value,
                        None => break,
                    };
                let (p1, r1) =
                    match path_point_radius(path, i1, total_points, default_radius, 1.0) {
                        Some(value) => value,
                        None => break,
                    };
                let (p2, r2) =
                    match path_point_radius(path, i2, total_points, default_radius, 1.0) {
                        Some(value) => value,
                        None => break,
                    };
                let bounds = segment_bounds(&[p0, p1, p2], r0.max(r1).max(r2));
                segments.push(PathSegmentData {
                    segment_index,
                    kind: 1,
                    p0,
                    p1,
                    p2,
                    p3: p2,
                    r0,
                    r1,
                    r2,
                    r3: r2,
                    bounds,
                });
                point_id += 2;
            }
            2 => {
                let i0 = point_id;
                let i1 = point_id + 1;
                let i2 = point_id + 2;
                let i3 = point_id + 3;
                let (p0, r0) =
                    match path_point_radius(path, i0, total_points, default_radius, 1.0) {
                        Some(value) => value,
                        None => break,
                    };
                let (p1, r1) =
                    match path_point_radius(path, i1, total_points, default_radius, 1.0) {
                        Some(value) => value,
                        None => break,
                    };
                let (p2, r2) =
                    match path_point_radius(path, i2, total_points, default_radius, 1.0) {
                        Some(value) => value,
                        None => break,
                    };
                let (p3, r3) =
                    match path_point_radius(path, i3, total_points, default_radius, 1.0) {
                        Some(value) => value,
                        None => break,
                    };
                let bounds = segment_bounds(&[p0, p1, p2, p3], r0.max(r1).max(r2).max(r3));
                segments.push(PathSegmentData {
                    segment_index,
                    kind: 2,
                    p0,
                    p1,
                    p2,
                    p3,
                    r0,
                    r1,
                    r2,
                    r3,
                    bounds,
                });
                point_id += 3;
            }
            _ => break,
        }
    }

    if segments.is_empty() {
        return None;
    }

    let mut indices = (0..segments.len()).collect::<Vec<_>>();
    let mut nodes = Vec::new();
    let len = indices.len();
    build_bvh_node_segments(&mut nodes, &segments, &mut indices, 0, len);

    Some(PathBvh {
        nodes,
        indices,
        segments,
        use_distance_approx: path.use_distance_approx,
    })
}

fn build_bvh_node_segments(
    nodes: &mut Vec<BvhNode>,
    segments: &[PathSegmentData],
    indices: &mut [usize],
    start: usize,
    end: usize,
) -> u32 {
    let mut bounds = Bounds::empty();
    for idx in &indices[start..end] {
        bounds.include(segments[*idx].bounds);
    }

    let count = end - start;
    let node_index = nodes.len() as u32;
    nodes.push(BvhNode {
        bounds,
        left: BVH_NONE,
        right: BVH_NONE,
        start: start as u32,
        count: count as u32,
    });

    if count <= BVH_LEAF_SIZE {
        return node_index;
    }

    // Split along the dominant axis of the current bounds.
    let extent = bounds.extent();
    let axis = if extent.x >= extent.y { 0 } else { 1 };

    indices[start..end].sort_by(|a, b| {
        let ca = segments[*a].bounds.center();
        let cb = segments[*b].bounds.center();
        let va = if axis == 0 { ca.x } else { ca.y };
        let vb = if axis == 0 { cb.x } else { cb.y };
        va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mid = start + count / 2;
    let left = build_bvh_node_segments(nodes, segments, indices, start, mid);
    let right = build_bvh_node_segments(nodes, segments, indices, mid, end);

    let node = &mut nodes[node_index as usize];
    node.left = left;
    node.right = right;
    node.count = 0;

    node_index
}
