//! CPU-side distance queries and BVH acceleration.

use crate::geometry::{Path, StrokeSegment};
use crate::math::{Mat3, Vec2};
use crate::scene::{FillRule, Scene, Shape, ShapeGeometry, StrokeJoin};

/// Closest location along a path segment.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ClosestPathPoint {
    pub segment_index: usize,
    pub t: f32,
}

/// Closest point on any shape in a group.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ClosestPoint {
    pub shape_index: usize,
    pub point: Vec2,
    pub distance: f32,
    pub path: Option<ClosestPathPoint>,
}

/// Tuning options for distance evaluation.
#[derive(Debug, Copy, Clone)]
pub struct DistanceOptions {
    pub path_tolerance: f32,
}

impl Default for DistanceOptions {
    fn default() -> Self {
        Self { path_tolerance: 0.5 }
    }
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

impl Bounds {
    fn empty() -> Self {
        Self {
            min: Vec2::new(f32::INFINITY, f32::INFINITY),
            max: Vec2::new(f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }

    fn include(&mut self, other: Bounds) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    fn center(&self) -> Vec2 {
        (self.min + self.max) * 0.5
    }

    fn extent(&self) -> Vec2 {
        self.max - self.min
    }

    fn contains(&self, pt: Vec2) -> bool {
        pt.x >= self.min.x && pt.x <= self.max.x && pt.y >= self.min.y && pt.y <= self.max.y
    }

    fn distance(&self, pt: Vec2) -> f32 {
        let dx = (self.min.x - pt.x).max(0.0).max(pt.x - self.max.x);
        let dy = (self.min.y - pt.y).max(0.0).max(pt.y - self.max.y);
        (dx * dx + dy * dy).sqrt()
    }
}
/// Per-group BVH acceleration structure for distance and winding queries.
#[derive(Debug, Clone)]
pub struct SceneBvh {
    groups: Vec<GroupBvh>,
}

#[derive(Debug, Clone)]
struct GroupBvh {
    nodes: Vec<BvhNode>,
    indices: Vec<usize>,
    shapes: Vec<BvhShape>,
}

#[derive(Debug, Copy, Clone)]
struct BvhNode {
    bounds: Bounds,
    left: u32,
    right: u32,
    start: u32,
    count: u32,
}

#[derive(Debug, Clone)]
struct BvhShape {
    shape_index: usize,
    bounds: Bounds,
    path_bvh: Option<PathBvh>,
}

#[derive(Debug, Copy, Clone)]
struct Bounds {
    min: Vec2,
    max: Vec2,
}

#[derive(Debug, Clone)]
struct PathBvh {
    nodes: Vec<BvhNode>,
    indices: Vec<usize>,
    segments: Vec<PathSegmentData>,
    use_distance_approx: bool,
}

#[derive(Debug, Copy, Clone)]
struct PathSegmentData {
    segment_index: usize,
    kind: u8,
    p0: Vec2,
    p1: Vec2,
    p2: Vec2,
    p3: Vec2,
    r0: f32,
    r1: f32,
    r2: f32,
    r3: f32,
    bounds: Bounds,
}

const BVH_LEAF_SIZE: usize = 8;
const BVH_NONE: u32 = u32::MAX;

/// Check if any stroked shape in the group is within its stroke radius of `pt`.
pub fn within_distance(scene: &Scene, group_index: usize, pt: Vec2) -> bool {
    let Some(group) = scene.groups.get(group_index) else {
        return false;
    };
    let local_pt = group.canvas_to_shape.transform_point(pt);
    for &shape_index in &group.shape_indices {
        let Some(shape) = scene.shapes.get(shape_index) else {
            continue;
        };
        if shape.stroke_width <= 0.0 {
            continue;
        }
        let stroke_width = shape.stroke_width;
        if within_distance_shape(shape, local_pt, stroke_width) {
            return true;
        }
    }
    false
}

/// Compute the closest point to `pt` within a given group.
pub fn compute_distance(
    scene: &Scene,
    group_index: usize,
    pt: Vec2,
    max_radius: f32,
    options: DistanceOptions,
) -> Option<ClosestPoint> {
    let Some(group) = scene.groups.get(group_index) else {
        return None;
    };
    let local_pt = group.canvas_to_shape.transform_point(pt);
    let mut best: Option<ClosestPoint> = None;

    for &shape_index in &group.shape_indices {
        let Some(shape) = scene.shapes.get(shape_index) else {
            continue;
        };
        let local = closest_point_shape(shape, local_pt, options.path_tolerance);
        let Some((local_closest, path_info)) = local else {
            continue;
        };
        let closest = group.shape_to_canvas.transform_point(local_closest);
        let distance = (closest - pt).length();
        if distance <= max_radius {
            let update = match best {
                None => true,
                Some(prev) => distance < prev.distance,
            };
            if update {
                best = Some(ClosestPoint {
                    shape_index,
                    point: closest,
                    distance,
                    path: path_info,
                });
            }
        }
    }

    best
}

/// BVH-accelerated variant of `within_distance`.
pub fn within_distance_bvh(scene: &Scene, bvh: &SceneBvh, group_index: usize, pt: Vec2) -> bool {
    let Some(group) = scene.groups.get(group_index) else {
        return false;
    };
    let Some(group_bvh) = bvh.groups.get(group_index) else {
        return false;
    };
    if group_bvh.nodes.is_empty() {
        return within_distance(scene, group_index, pt);
    }

    let local_pt = group.canvas_to_shape.transform_point(pt);
    let mut stack = Vec::new();
    stack.push(0usize);

    while let Some(node_index) = stack.pop() {
        let node = &group_bvh.nodes[node_index];
        if !node.bounds.contains(local_pt) {
            continue;
        }
        if node.count > 0 {
            let start = node.start as usize;
            let end = start + node.count as usize;
            for i in start..end {
                let shape_slot = group_bvh.indices[i];
                let shape_meta = &group_bvh.shapes[shape_slot];
                let shape_index = shape_meta.shape_index;
                let Some(shape) = scene.shapes.get(shape_index) else {
                    continue;
                };
                if shape.stroke_width <= 0.0 {
                    continue;
                }
                let stroke_width = shape.stroke_width;
                if let Some(path_bvh) = &shape_meta.path_bvh {
                    let local_pt = transform_point_inverse(shape.transform, local_pt);
                    if within_distance_path_bvh(path_bvh, local_pt) {
                        return true;
                    }
                } else if within_distance_shape(shape, local_pt, stroke_width) {
                    return true;
                }
            }
        } else {
            let left = node.left as usize;
            let right = node.right as usize;
            if left != BVH_NONE as usize {
                stack.push(left);
            }
            if right != BVH_NONE as usize {
                stack.push(right);
            }
        }
    }

    false
}

/// BVH-accelerated variant of `compute_distance`.
pub fn compute_distance_bvh(
    scene: &Scene,
    bvh: &SceneBvh,
    group_index: usize,
    pt: Vec2,
    max_radius: f32,
    options: DistanceOptions,
) -> Option<ClosestPoint> {
    let Some(group) = scene.groups.get(group_index) else {
        return None;
    };
    let Some(group_bvh) = bvh.groups.get(group_index) else {
        return None;
    };
    if group_bvh.nodes.is_empty() {
        return compute_distance(scene, group_index, pt, max_radius, options);
    }

    let local_pt = group.canvas_to_shape.transform_point(pt);
    let mut best: Option<ClosestPoint> = None;
    let mut best_dist = max_radius;

    let mut stack = Vec::new();
    stack.push(0usize);

    while let Some(node_index) = stack.pop() {
        let node = &group_bvh.nodes[node_index];
        let node_dist = node.bounds.distance(local_pt);
        if node_dist > best_dist {
            continue;
        }

        if node.count > 0 {
            let start = node.start as usize;
            let end = start + node.count as usize;
            for i in start..end {
                let shape_slot = group_bvh.indices[i];
                let shape_meta = &group_bvh.shapes[shape_slot];
                let shape_index = shape_meta.shape_index;
                let Some(shape) = scene.shapes.get(shape_index) else {
                    continue;
                };
                let local = if let Some(path_bvh) = &shape_meta.path_bvh {
                    let shape_pt = transform_point_inverse(shape.transform, local_pt);
                    closest_point_path_bvh(path_bvh, shape_pt).map(|(shape_closest, path_info)| {
                        let group_closest = shape.transform.transform_point(shape_closest);
                        (group_closest, path_info)
                    })
                } else {
                    closest_point_shape(shape, local_pt, options.path_tolerance)
                };
                let Some((local_closest, path_info)) = local else {
                    continue;
                };
                let closest = group.shape_to_canvas.transform_point(local_closest);
                let distance = (closest - pt).length();
                if distance <= best_dist {
                    best_dist = distance;
                    best = Some(ClosestPoint {
                        shape_index,
                        point: closest,
                        distance,
                        path: path_info,
                    });
                }
            }
        } else {
            let left = node.left as usize;
            let right = node.right as usize;
            let mut left_dist = f32::INFINITY;
            let mut right_dist = f32::INFINITY;
            if left != BVH_NONE as usize {
                left_dist = group_bvh.nodes[left].bounds.distance(local_pt);
            }
            if right != BVH_NONE as usize {
                right_dist = group_bvh.nodes[right].bounds.distance(local_pt);
            }

            // Traverse the closer child first to improve pruning.
            if left_dist < right_dist {
                if right_dist <= best_dist {
                    stack.push(right);
                }
                if left_dist <= best_dist {
                    stack.push(left);
                }
            } else {
                if left_dist <= best_dist {
                    stack.push(left);
                }
                if right_dist <= best_dist {
                    stack.push(right);
                }
            }
        }
    }

    best
}

/// Determine whether a point is inside the group's filled region.
pub(crate) fn is_inside(scene: &Scene, group_index: usize, pt: Vec2) -> bool {
    let Some(group) = scene.groups.get(group_index) else {
        return false;
    };
    let local_pt = group.canvas_to_shape.transform_point(pt);
    let mut winding = 0i32;
    for &shape_index in &group.shape_indices {
        let Some(shape) = scene.shapes.get(shape_index) else {
            continue;
        };
        let shape_pt = transform_point_inverse(shape.transform, local_pt);
        winding += winding_number_shape(shape, None, shape_pt);
    }
    match group.fill_rule {
        FillRule::EvenOdd => winding.abs() % 2 == 1,
        FillRule::NonZero => winding != 0,
    }
}

/// BVH-accelerated inside test used for fill evaluation.
pub(crate) fn is_inside_bvh(scene: &Scene, bvh: &SceneBvh, group_index: usize, pt: Vec2) -> bool {
    let Some(group) = scene.groups.get(group_index) else {
        return false;
    };
    let Some(group_bvh) = bvh.groups.get(group_index) else {
        return false;
    };
    if group_bvh.nodes.is_empty() {
        return is_inside(scene, group_index, pt);
    }
    let local_pt = group.canvas_to_shape.transform_point(pt);
    if !group_bvh.nodes[0].bounds.contains(local_pt) {
        return false;
    }
    let mut winding = 0i32;
    let mut stack = Vec::new();
    stack.push(0usize);
    while let Some(node_index) = stack.pop() {
        let node = &group_bvh.nodes[node_index];
        if !node.bounds.contains(local_pt) {
            continue;
        }
        if node.count > 0 {
            let start = node.start as usize;
            let end = start + node.count as usize;
            for i in start..end {
                let shape_slot = group_bvh.indices[i];
                let shape_meta = &group_bvh.shapes[shape_slot];
                let shape_index = shape_meta.shape_index;
                let Some(shape) = scene.shapes.get(shape_index) else {
                    continue;
                };
                let shape_pt = transform_point_inverse(shape.transform, local_pt);
                winding +=
                    winding_number_shape(shape, shape_meta.path_bvh.as_ref(), shape_pt);
            }
        } else {
            let left = node.left as usize;
            let right = node.right as usize;
            if left != BVH_NONE as usize {
                stack.push(left);
            }
            if right != BVH_NONE as usize {
                stack.push(right);
            }
        }
    }
    match group.fill_rule {
        FillRule::EvenOdd => winding.abs() % 2 == 1,
        FillRule::NonZero => winding != 0,
    }
}

fn winding_number_shape(shape: &Shape, path_bvh: Option<&PathBvh>, pt: Vec2) -> i32 {
    match &shape.geometry {
        ShapeGeometry::Circle { center, radius } => {
            if (pt - *center).length_squared() < radius.abs() * radius.abs() {
                1
            } else {
                0
            }
        }
        ShapeGeometry::Ellipse { center, radius } => {
            let rx = radius.x.abs().max(1.0e-6);
            let ry = radius.y.abs().max(1.0e-6);
            let dx = (pt.x - center.x) / rx;
            let dy = (pt.y - center.y) / ry;
            if dx * dx + dy * dy < 1.0 {
                1
            } else {
                0
            }
        }
        ShapeGeometry::Rect { min, max } => {
            if pt.x > min.x && pt.x < max.x && pt.y > min.y && pt.y < max.y {
                1
            } else {
                0
            }
        }
        ShapeGeometry::Path { path } => {
            if let Some(path_bvh) = path_bvh {
                winding_number_path_bvh(path_bvh, pt)
            } else {
                winding_number_path(path, pt)
            }
        }
    }
}

fn winding_number_path(path: &Path, pt: Vec2) -> i32 {
    if path.points.is_empty() {
        return 0;
    }
    let total_points = path.points.len();
    let mut winding = 0i32;
    let mut point_id = 0usize;
    for &num_controls in &path.num_control_points {
        match num_controls {
            0 => {
                let i0 = point_id;
                let i1 = point_id + 1;
                let p0 = match path_point(path, i0, total_points) {
                    Some(value) => value,
                    None => break,
                };
                let p1 = match path_point(path, i1, total_points) {
                    Some(value) => value,
                    None => break,
                };
                winding += winding_number_segment(0, p0, p1, p1, p1, pt);
                point_id += 1;
            }
            1 => {
                let i0 = point_id;
                let i1 = point_id + 1;
                let i2 = point_id + 2;
                let p0 = match path_point(path, i0, total_points) {
                    Some(value) => value,
                    None => break,
                };
                let p1 = match path_point(path, i1, total_points) {
                    Some(value) => value,
                    None => break,
                };
                let p2 = match path_point(path, i2, total_points) {
                    Some(value) => value,
                    None => break,
                };
                winding += winding_number_segment(1, p0, p1, p2, p2, pt);
                point_id += 2;
            }
            2 => {
                let i0 = point_id;
                let i1 = point_id + 1;
                let i2 = point_id + 2;
                let i3 = point_id + 3;
                let p0 = match path_point(path, i0, total_points) {
                    Some(value) => value,
                    None => break,
                };
                let p1 = match path_point(path, i1, total_points) {
                    Some(value) => value,
                    None => break,
                };
                let p2 = match path_point(path, i2, total_points) {
                    Some(value) => value,
                    None => break,
                };
                let p3 = match path_point(path, i3, total_points) {
                    Some(value) => value,
                    None => break,
                };
                winding += winding_number_segment(2, p0, p1, p2, p3, pt);
                point_id += 3;
            }
            _ => break,
        }
    }
    winding
}

fn winding_number_path_bvh(path_bvh: &PathBvh, pt: Vec2) -> i32 {
    if path_bvh.nodes.is_empty() {
        return 0;
    }
    let mut winding = 0i32;
    let mut stack = Vec::new();
    stack.push(0usize);
    while let Some(node_index) = stack.pop() {
        let node = &path_bvh.nodes[node_index];
        if !ray_intersects_bounds(&node.bounds, pt) {
            continue;
        }
        if node.count > 0 {
            let start = node.start as usize;
            let end = start + node.count as usize;
            for i in start..end {
                let seg = &path_bvh.segments[path_bvh.indices[i]];
                winding += winding_number_segment(seg.kind, seg.p0, seg.p1, seg.p2, seg.p3, pt);
            }
        } else {
            let left = node.left as usize;
            let right = node.right as usize;
            if left != BVH_NONE as usize {
                stack.push(left);
            }
            if right != BVH_NONE as usize {
                stack.push(right);
            }
        }
    }
    winding
}

fn ray_intersects_bounds(bounds: &Bounds, pt: Vec2) -> bool {
    if pt.y < bounds.min.y || pt.y > bounds.max.y {
        return false;
    }
    if pt.x > bounds.max.x {
        return false;
    }
    true
}

fn winding_number_segment(kind: u8, p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, pt: Vec2) -> i32 {
    match kind {
        0 => {
            let dy = p1.y - p0.y;
            if dy.abs() <= 1.0e-8 {
                return 0;
            }
            let t = (pt.y - p0.y) / dy;
            if t >= 0.0 && t <= 1.0 {
                let tp = p0.x - pt.x + t * (p1.x - p0.x);
                if tp >= 0.0 {
                    return if dy > 0.0 { 1 } else { -1 };
                }
            }
            0
        }
        1 => {
            // Quadratic: solve y(t) = pt.y to count crossings.
            let a = (p0.y - 2.0 * p1.y + p2.y) as f64;
            let b = (-2.0 * p0.y + 2.0 * p1.y) as f64;
            let c = (p0.y - pt.y) as f64;
            let mut roots = [0.0f64; 2];
            let mut num_roots = 0usize;
            if a.abs() < 1.0e-12 {
                if b.abs() > 1.0e-12 {
                    roots[0] = -c / b;
                    num_roots = 1;
                }
            } else if solve_quadratic(a, b, c, &mut roots) {
                num_roots = 2;
            }
            let ax = (p0.x - 2.0 * p1.x + p2.x) as f64;
            let bx = (-2.0 * p0.x + 2.0 * p1.x) as f64;
            let cx = (p0.x - pt.x) as f64;
            let mut winding = 0i32;
            for i in 0..num_roots {
                let t = roots[i];
                if t >= 0.0 && t <= 1.0 {
                    let tp = ax * t * t + bx * t + cx;
                    if tp >= 0.0 {
                        let deriv = 2.0 * a * t + b;
                        winding += if deriv > 0.0 { 1 } else { -1 };
                    }
                }
            }
            winding
        }
        2 => {
            // Cubic: solve y(t) = pt.y and accumulate winding by derivative sign.
            let a = (-p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y) as f64;
            let b = (3.0 * p0.y - 6.0 * p1.y + 3.0 * p2.y) as f64;
            let c = (-3.0 * p0.y + 3.0 * p1.y) as f64;
            let d = (p0.y - pt.y) as f64;
            let mut roots = [0.0f64; 3];
            let num_roots = solve_cubic(a, b, c, d, &mut roots);
            let ax = (-p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x) as f64;
            let bx = (3.0 * p0.x - 6.0 * p1.x + 3.0 * p2.x) as f64;
            let cx = (-3.0 * p0.x + 3.0 * p1.x) as f64;
            let dx = (p0.x - pt.x) as f64;
            let mut winding = 0i32;
            for i in 0..num_roots {
                let t = roots[i];
                if t >= 0.0 && t <= 1.0 {
                    let tp = ax * t * t * t + bx * t * t + cx * t + dx;
                    if tp > 0.0 {
                        let deriv = 3.0 * a * t * t + 2.0 * b * t + c;
                        winding += if deriv > 0.0 { 1 } else { -1 };
                    }
                }
            }
            winding
        }
        _ => 0,
    }
}

fn within_distance_path_bvh(path_bvh: &PathBvh, pt: Vec2) -> bool {
    if path_bvh.nodes.is_empty() {
        return false;
    }

    let mut stack = Vec::new();
    stack.push(0usize);

    while let Some(node_index) = stack.pop() {
        let node = &path_bvh.nodes[node_index];
        if !node.bounds.contains(pt) {
            continue;
        }
        if node.count > 0 {
            let start = node.start as usize;
            let end = start + node.count as usize;
            for i in start..end {
                let seg = &path_bvh.segments[path_bvh.indices[i]];
                let (dist, t) = match seg.kind {
                    0 => {
                        let (dist, _cp, t) = distance_to_segment(pt, seg.p0, seg.p1);
                        (dist, t)
                    }
                    1 => {
                        let (_cp, t, dist) =
                            closest_point_quadratic(pt, seg.p0, seg.p1, seg.p2, path_bvh.use_distance_approx);
                        (dist, t)
                    }
                    _ => {
                        let (_cp, t, dist) =
                            closest_point_cubic(pt, seg.p0, seg.p1, seg.p2, seg.p3, path_bvh.use_distance_approx);
                        (dist, t)
                    }
                };

                let radius = match seg.kind {
                    0 => seg.r0 + t * (seg.r1 - seg.r0),
                    1 => {
                        let tt = 1.0 - t;
                        tt * tt * seg.r0 + 2.0 * tt * t * seg.r1 + t * t * seg.r2
                    }
                    _ => {
                        let tt = 1.0 - t;
                        tt * tt * tt * seg.r0
                            + 3.0 * tt * tt * t * seg.r1
                            + 3.0 * tt * t * t * seg.r2
                            + t * t * t * seg.r3
                    }
                };

                if dist < radius {
                    return true;
                }
            }
        } else {
            let left = node.left as usize;
            let right = node.right as usize;
            if left != BVH_NONE as usize {
                stack.push(left);
            }
            if right != BVH_NONE as usize {
                stack.push(right);
            }
        }
    }

    false
}

fn closest_point_path_bvh(
    path_bvh: &PathBvh,
    pt: Vec2,
) -> Option<(Vec2, Option<ClosestPathPoint>)> {
    if path_bvh.nodes.is_empty() {
        return None;
    }

    let mut best_dist = f32::INFINITY;
    let mut best_pt = Vec2::ZERO;
    let mut best_seg = 0usize;
    let mut best_t = 0.0;

    let mut stack = Vec::new();
    stack.push(0usize);

    while let Some(node_index) = stack.pop() {
        let node = &path_bvh.nodes[node_index];
        if node.bounds.distance(pt) > best_dist {
            continue;
        }
        if node.count > 0 {
            let start = node.start as usize;
            let end = start + node.count as usize;
            for i in start..end {
                let seg = &path_bvh.segments[path_bvh.indices[i]];
                let (cp, t, dist) = match seg.kind {
                    0 => {
                        let (dist, cp, t) = distance_to_segment(pt, seg.p0, seg.p1);
                        (cp, t, dist)
                    }
                    1 => closest_point_quadratic(
                        pt,
                        seg.p0,
                        seg.p1,
                        seg.p2,
                        path_bvh.use_distance_approx,
                    ),
                    _ => closest_point_cubic(pt, seg.p0, seg.p1, seg.p2, seg.p3, path_bvh.use_distance_approx),
                };
                if dist < best_dist {
                    best_dist = dist;
                    best_pt = cp;
                    best_seg = seg.segment_index;
                    best_t = t;
                }
            }
        } else {
            let left = node.left as usize;
            let right = node.right as usize;
            let mut left_dist = f32::INFINITY;
            let mut right_dist = f32::INFINITY;
            if left != BVH_NONE as usize {
                left_dist = path_bvh.nodes[left].bounds.distance(pt);
            }
            if right != BVH_NONE as usize {
                right_dist = path_bvh.nodes[right].bounds.distance(pt);
            }

            if left_dist < right_dist {
                if right_dist <= best_dist {
                    stack.push(right);
                }
                if left_dist <= best_dist {
                    stack.push(left);
                }
            } else {
                if left_dist <= best_dist {
                    stack.push(left);
                }
                if right_dist <= best_dist {
                    stack.push(right);
                }
            }
        }
    }

    if best_dist.is_finite() {
        Some((
            best_pt,
            Some(ClosestPathPoint {
                segment_index: best_seg,
                t: best_t,
            }),
        ))
    } else {
        None
    }
}

impl GroupBvh {
    fn build(scene: &Scene, group: &crate::scene::ShapeGroup) -> Self {
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

fn within_distance_shape(shape: &Shape, pt: Vec2, r: f32) -> bool {
    if r <= 0.0 {
        return false;
    }
    let local_pt = transform_point_inverse(shape.transform, pt);
    match &shape.geometry {
        ShapeGeometry::Circle { center, radius } => {
            let dist = (local_pt - *center).length() - radius.abs();
            dist.abs() < r
        }
        ShapeGeometry::Ellipse { center, radius } => {
            let dist = ellipse_signed_distance(local_pt, *center, Vec2::new(radius.x.abs(), radius.y.abs()));
            dist.abs() < r
        }
        ShapeGeometry::Rect { min, max } => {
            let dist = rect_signed_distance(local_pt, *min, *max);
            dist.abs() < r
        }
        ShapeGeometry::Path { path } => {
            let stroke_width = r;
            let total_points = path.points.len();
            if total_points == 0 {
                return false;
            }
            let mut point_id = 0usize;
            for &num_controls in &path.num_control_points {
                match num_controls {
                    0 => {
                        let i0 = point_id;
                        let i1 = point_id + 1;
                        let (p0, r0) = match path_point_radius(
                            path,
                            i0,
                            total_points,
                            stroke_width,
                            1.0,
                        ) {
                            Some(value) => value,
                            None => break,
                        };
                        let (p1, r1) = match path_point_radius(
                            path,
                            i1,
                            total_points,
                            stroke_width,
                            1.0,
                        ) {
                            Some(value) => value,
                            None => break,
                        };
                        let (dist, _cp, t) = distance_to_segment(local_pt, p0, p1);
                        let radius = r0 + t * (r1 - r0);
                        if dist < radius {
                            return true;
                        }
                        point_id += 1;
                    }
                    1 => {
                        let i0 = point_id;
                        let i1 = point_id + 1;
                        let i2 = point_id + 2;
                        let (p0, r0) = match path_point_radius(
                            path,
                            i0,
                            total_points,
                            stroke_width,
                            1.0,
                        ) {
                            Some(value) => value,
                            None => break,
                        };
                        let (p1, r1) = match path_point_radius(
                            path,
                            i1,
                            total_points,
                            stroke_width,
                            1.0,
                        ) {
                            Some(value) => value,
                            None => break,
                        };
                        let (p2, r2) = match path_point_radius(
                            path,
                            i2,
                            total_points,
                            stroke_width,
                            1.0,
                        ) {
                            Some(value) => value,
                            None => break,
                        };
                        let (_cp, t, dist) =
                            closest_point_quadratic(local_pt, p0, p1, p2, path.use_distance_approx);
                        let tt = 1.0 - t;
                        let radius = tt * tt * r0 + 2.0 * tt * t * r1 + t * t * r2;
                        if dist < radius {
                            return true;
                        }
                        point_id += 2;
                    }
                    2 => {
                        let i0 = point_id;
                        let i1 = point_id + 1;
                        let i2 = point_id + 2;
                        let i3 = point_id + 3;
                        let (p0, r0) = match path_point_radius(
                            path,
                            i0,
                            total_points,
                            stroke_width,
                            1.0,
                        ) {
                            Some(value) => value,
                            None => break,
                        };
                        let (p1, r1) = match path_point_radius(
                            path,
                            i1,
                            total_points,
                            stroke_width,
                            1.0,
                        ) {
                            Some(value) => value,
                            None => break,
                        };
                        let (p2, r2) = match path_point_radius(
                            path,
                            i2,
                            total_points,
                            stroke_width,
                            1.0,
                        ) {
                            Some(value) => value,
                            None => break,
                        };
                        let (p3, r3) = match path_point_radius(
                            path,
                            i3,
                            total_points,
                            stroke_width,
                            1.0,
                        ) {
                            Some(value) => value,
                            None => break,
                        };
                        let (_cp, t, dist) =
                            closest_point_cubic(local_pt, p0, p1, p2, p3, path.use_distance_approx);
                        let tt = 1.0 - t;
                        let radius = tt * tt * tt * r0
                            + 3.0 * tt * tt * t * r1
                            + 3.0 * tt * t * t * r2
                            + t * t * t * r3;
                        if dist < radius {
                            return true;
                        }
                        point_id += 3;
                    }
                    _ => break,
                }
            }
            false
        }
    }
}

fn shape_bounds(shape: &Shape) -> Option<Bounds> {
    let geom_bounds = shape_geom_bounds(shape)?;
    let pad = max_stroke_radius(shape);
    Some(inflate_bounds(geom_bounds, pad))
}

fn shape_geom_bounds(shape: &Shape) -> Option<Bounds> {
    match &shape.geometry {
        ShapeGeometry::Circle { center, radius } => {
            let transform = shape.transform;
            let center_t = transform.transform_point(*center);
            let r = radius.abs();
            let ext = ellipse_extents(transform, r, r);
            Some(Bounds {
                min: Vec2::new(center_t.x - ext.x, center_t.y - ext.y),
                max: Vec2::new(center_t.x + ext.x, center_t.y + ext.y),
            })
        }
        ShapeGeometry::Ellipse { center, radius } => {
            let transform = shape.transform;
            let center_t = transform.transform_point(*center);
            let rx = radius.x.abs();
            let ry = radius.y.abs();
            let ext = ellipse_extents(transform, rx, ry);
            Some(Bounds {
                min: Vec2::new(center_t.x - ext.x, center_t.y - ext.y),
                max: Vec2::new(center_t.x + ext.x, center_t.y + ext.y),
            })
        }
        ShapeGeometry::Rect { min, max } => {
            let transform = shape.transform;
            let corners = rect_corners(*min, *max)
                .into_iter()
                .map(|p| transform.transform_point(p))
                .collect::<Vec<_>>();
            let (min, max) = bounds_from_points(&corners);
            Some(Bounds { min, max })
        }
        ShapeGeometry::Path { path } => {
            let transform = shape.transform;
            let transformed = transform_path(path, transform);
            if transformed.points.is_empty() {
                return None;
            }
            let (min, max) = bounds_from_points(&transformed.points);
            Some(Bounds { min, max })
        }
    }
}

fn max_stroke_radius(shape: &Shape) -> f32 {
    let stroke_scale = shape.transform.max_scale().max(0.0);
    let mut radius = shape.stroke_width.abs() * stroke_scale;
    if let ShapeGeometry::Path { path } = &shape.geometry {
        if let Some(thickness) = &path.thickness {
            let mut max_thickness = 0.0f32;
            for &value in thickness {
                max_thickness = max_thickness.max(value.abs());
            }
            radius = radius.max(max_thickness * stroke_scale);
        }
    }
    if radius > 0.0 && shape.stroke_join == StrokeJoin::Miter {
        radius *= shape.stroke_miter_limit.max(1.0);
    }
    radius
}

fn ellipse_extents(transform: Mat3, rx: f32, ry: f32) -> Vec2 {
    let m00 = transform.m[0][0];
    let m01 = transform.m[0][1];
    let m10 = transform.m[1][0];
    let m11 = transform.m[1][1];
    let ex = ((m00 * rx) * (m00 * rx) + (m01 * ry) * (m01 * ry)).sqrt();
    let ey = ((m10 * rx) * (m10 * rx) + (m11 * ry) * (m11 * ry)).sqrt();
    Vec2::new(ex, ey)
}

fn inflate_bounds(bounds: Bounds, pad: f32) -> Bounds {
    let pad = pad.max(0.0);
    Bounds {
        min: Vec2::new(bounds.min.x - pad, bounds.min.y - pad),
        max: Vec2::new(bounds.max.x + pad, bounds.max.y + pad),
    }
}

fn segment_bounds(points: &[Vec2], pad: f32) -> Bounds {
    let (min, max) = bounds_from_points(points);
    inflate_bounds(Bounds { min, max }, pad)
}

fn closest_point_shape(
    shape: &Shape,
    pt: Vec2,
    tolerance: f32,
) -> Option<(Vec2, Option<ClosestPathPoint>)> {
    let local_pt = transform_point_inverse(shape.transform, pt);
    match &shape.geometry {
        ShapeGeometry::Circle { center, radius } => {
            let local_closest = closest_point_circle(*center, radius.abs(), local_pt);
            let closest = shape.transform.transform_point(local_closest);
            Some((closest, None))
        }
        ShapeGeometry::Ellipse { center, radius } => {
            let segments = ellipse_to_segments(*center, Vec2::new(radius.x.abs(), radius.y.abs()), Mat3::identity(), tolerance);
            let (local_closest, _path) = closest_point_segments(&segments, local_pt)?;
            let closest = shape.transform.transform_point(local_closest);
            Some((closest, None))
        }
        ShapeGeometry::Rect { min, max } => {
            let local_closest = closest_point_rect(*min, *max, local_pt);
            let closest = shape.transform.transform_point(local_closest);
            Some((closest, None))
        }
        ShapeGeometry::Path { path } => {
            let total_points = path.points.len();
            if total_points == 0 {
                return None;
            }
            let mut best_dist = f32::INFINITY;
            let mut best_pt = Vec2::ZERO;
            let mut best_seg = 0usize;
            let mut best_t = 0.0;
            let mut point_id = 0usize;
            for (seg_index, &num_controls) in path.num_control_points.iter().enumerate() {
                match num_controls {
                    0 => {
                        let i0 = point_id;
                        let i1 = point_id + 1;
                        let p0 = match path_point(path, i0, total_points) {
                            Some(value) => value,
                            None => break,
                        };
                        let p1 = match path_point(path, i1, total_points) {
                            Some(value) => value,
                            None => break,
                        };
                        let (dist, cp, t) = distance_to_segment(local_pt, p0, p1);
                        if dist < best_dist {
                            best_dist = dist;
                            best_pt = cp;
                            best_seg = seg_index;
                            best_t = t;
                        }
                        point_id += 1;
                    }
                    1 => {
                        let i0 = point_id;
                        let i1 = point_id + 1;
                        let i2 = point_id + 2;
                        let p0 = match path_point(path, i0, total_points) {
                            Some(value) => value,
                            None => break,
                        };
                        let p1 = match path_point(path, i1, total_points) {
                            Some(value) => value,
                            None => break,
                        };
                        let p2 = match path_point(path, i2, total_points) {
                            Some(value) => value,
                            None => break,
                        };
                        let (cp, t, dist) =
                            closest_point_quadratic(local_pt, p0, p1, p2, path.use_distance_approx);
                        if dist < best_dist {
                            best_dist = dist;
                            best_pt = cp;
                            best_seg = seg_index;
                            best_t = t;
                        }
                        point_id += 2;
                    }
                    2 => {
                        let i0 = point_id;
                        let i1 = point_id + 1;
                        let i2 = point_id + 2;
                        let i3 = point_id + 3;
                        let p0 = match path_point(path, i0, total_points) {
                            Some(value) => value,
                            None => break,
                        };
                        let p1 = match path_point(path, i1, total_points) {
                            Some(value) => value,
                            None => break,
                        };
                        let p2 = match path_point(path, i2, total_points) {
                            Some(value) => value,
                            None => break,
                        };
                        let p3 = match path_point(path, i3, total_points) {
                            Some(value) => value,
                            None => break,
                        };
                        let (cp, t, dist) =
                            closest_point_cubic(local_pt, p0, p1, p2, p3, path.use_distance_approx);
                        if dist < best_dist {
                            best_dist = dist;
                            best_pt = cp;
                            best_seg = seg_index;
                            best_t = t;
                        }
                        point_id += 3;
                    }
                    _ => break,
                }
            }
            if best_dist.is_finite() {
                let closest = shape.transform.transform_point(best_pt);
                Some((
                    closest,
                    Some(ClosestPathPoint {
                        segment_index: best_seg,
                        t: best_t,
                    }),
                ))
            } else {
                None
            }
        }
    }
}

fn closest_point_segments(
    segments: &[StrokeSegment],
    pt: Vec2,
) -> Option<(Vec2, Option<ClosestPathPoint>)> {
    if segments.is_empty() {
        return None;
    }
    let mut min_dist = f32::INFINITY;
    let mut best = Vec2::ZERO;
    let mut best_index = 0usize;
    let mut best_t = 0.0;
    for (i, seg) in segments.iter().enumerate() {
        let (dist, cp, t) = distance_to_segment(pt, seg.start, seg.end);
        if dist < min_dist {
            min_dist = dist;
            best = cp;
            best_index = i;
            best_t = t;
        }
    }
    Some((
        best,
        Some(ClosestPathPoint {
            segment_index: best_index,
            t: best_t,
        }),
    ))
}

fn within_distance_segments(segments: &[StrokeSegment], pt: Vec2, r: f32) -> bool {
    for seg in segments {
        let (dist, _cp, _t) = distance_to_segment(pt, seg.start, seg.end);
        if dist < r {
            return true;
        }
    }
    false
}

fn distance_to_segment(pt: Vec2, a: Vec2, b: Vec2) -> (f32, Vec2, f32) {
    let ab = b - a;
    let denom = ab.dot(ab);
    if denom <= 1.0e-10 {
        let dist = (pt - a).length();
        return (dist, a, 0.0);
    }
    let t = ((pt - a).dot(ab) / denom).clamp(0.0, 1.0);
    let cp = a + ab * t;
    let dist = (pt - cp).length();
    (dist, cp, t)
}

fn transform_point_inverse(transform: Mat3, pt: Vec2) -> Vec2 {
    if transform.is_identity() {
        return pt;
    }
    match transform.inverse() {
        Some(inv) => inv.transform_point(pt),
        None => pt,
    }
}

fn path_point(path: &Path, index: usize, total_points: usize) -> Option<Vec2> {
    if total_points == 0 {
        return None;
    }
    if path.is_closed {
        let idx = index % total_points;
        return Some(path.points[idx]);
    }
    path.points.get(index).copied()
}

fn path_point_radius(
    path: &Path,
    index: usize,
    total_points: usize,
    default_radius: f32,
    thickness_scale: f32,
) -> Option<(Vec2, f32)> {
    let p = path_point(path, index, total_points)?;
    let radius = path
        .thickness
        .as_ref()
        .and_then(|values| {
            if path.is_closed {
                let idx = index % total_points;
                values.get(idx).copied()
            } else {
                values.get(index).copied()
            }
        })
        .map(|value| value * thickness_scale)
        .unwrap_or(default_radius);
    Some((p, radius))
}

fn closest_point_quadratic(
    pt: Vec2,
    p0: Vec2,
    p1: Vec2,
    p2: Vec2,
    use_distance_approx: bool,
) -> (Vec2, f32, f32) {
    if use_distance_approx {
        let (cp, t) = quadratic_closest_pt_approx(p0, p1, p2, pt);
        return (cp, t, (pt - cp).length());
    }

    let mut best_t = 0.0;
    let mut best_pt = p0;
    let mut best_dist = (pt - p0).length();
    let dist_end = (pt - p2).length();
    if dist_end < best_dist {
        best_dist = dist_end;
        best_pt = p2;
        best_t = 1.0;
    }

    let ax = (p0.x - 2.0 * p1.x + p2.x) as f64;
    let ay = (p0.y - 2.0 * p1.y + p2.y) as f64;
    let bx = (-p0.x + p1.x) as f64;
    let by = (-p0.y + p1.y) as f64;
    let cx = (p0.x - pt.x) as f64;
    let cy = (p0.y - pt.y) as f64;

    let a = ax * ax + ay * ay;
    let b = 3.0 * (ax * bx + ay * by);
    let c = 2.0 * (bx * bx + by * by) + (ax * cx + ay * cy);
    let d = bx * cx + by * cy;

    let mut roots = [0.0f64; 3];
    let num = solve_cubic(a, b, c, d, &mut roots);
    for i in 0..num {
        let t = roots[i];
        if t >= 0.0 && t <= 1.0 {
            let t32 = t as f32;
            let tt = 1.0 - t32;
            let cp = (p0 * (tt * tt)) + (p1 * (2.0 * tt * t32)) + (p2 * (t32 * t32));
            let dist = (pt - cp).length();
            if dist < best_dist {
                best_dist = dist;
                best_pt = cp;
                best_t = t32;
            }
        }
    }

    (best_pt, best_t, best_dist)
}

fn closest_point_cubic_approx(
    pt: Vec2,
    p0: Vec2,
    p1: Vec2,
    p2: Vec2,
    p3: Vec2,
) -> (Vec2, f32, f32) {
    let steps = 8usize;
    let inv = 1.0 / steps as f32;
    let mut prev = p0;
    let mut prev_t = 0.0f32;
    let mut best_pt = p0;
    let mut best_t = 0.0f32;
    let mut best_dist = (pt - p0).length();
    for i in 1..=steps {
        let t = i as f32 * inv;
        let tt = 1.0 - t;
        let tt2 = tt * tt;
        let t2 = t * t;
        let a = tt2 * tt;
        let b = 3.0 * tt2 * t;
        let c = 3.0 * tt * t2;
        let d = t2 * t;
        let curr = Vec2::new(
            a * p0.x + b * p1.x + c * p2.x + d * p3.x,
            a * p0.y + b * p1.y + c * p2.y + d * p3.y,
        );
        let (dist, cp, seg_t) = distance_to_segment(pt, prev, curr);
        let local_t = prev_t + seg_t * (t - prev_t);
        if dist < best_dist {
            best_dist = dist;
            best_pt = cp;
            best_t = local_t;
        }
        prev = curr;
        prev_t = t;
    }
    (best_pt, best_t, best_dist)
}

fn closest_point_cubic(
    pt: Vec2,
    p0: Vec2,
    p1: Vec2,
    p2: Vec2,
    p3: Vec2,
    use_distance_approx: bool,
) -> (Vec2, f32, f32) {
    if use_distance_approx {
        return closest_point_cubic_approx(pt, p0, p1, p2, p3);
    }
    let mut best_t = 0.0;
    let mut best_pt = p0;
    let mut best_dist = (pt - p0).length();
    let dist_end = (pt - p3).length();
    if dist_end < best_dist {
        best_dist = dist_end;
        best_pt = p3;
        best_t = 1.0;
    }

    let ax = (-p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x) as f64;
    let ay = (-p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y) as f64;
    let bx = (3.0 * p0.x - 6.0 * p1.x + 3.0 * p2.x) as f64;
    let by = (3.0 * p0.y - 6.0 * p1.y + 3.0 * p2.y) as f64;
    let cx = (-3.0 * p0.x + 3.0 * p1.x) as f64;
    let cy = (-3.0 * p0.y + 3.0 * p1.y) as f64;
    let dx = (p0.x - pt.x) as f64;
    let dy = (p0.y - pt.y) as f64;

    let a = 3.0 * (ax * ax + ay * ay);
    if a.abs() < 1.0e-10 {
        return (best_pt, best_t, best_dist);
    }
    let b = 5.0 * (ax * bx + ay * by);
    let c = 4.0 * (ax * cx + ay * cy) + 2.0 * (bx * bx + by * by);
    let d = 3.0 * ((bx * cx + by * cy) + (ax * dx + ay * dy));
    let e = (cx * cx + cy * cy) + 2.0 * (dx * bx + dy * by);
    let f = dx * cx + dy * cy;

    let b = b / a;
    let c = c / a;
    let d = d / a;
    let e = e / a;
    let f = f / a;

    let p1a = (2.0 / 5.0) * c - (4.0 / 25.0) * b * b;
    let p1b = (3.0 / 5.0) * d - (3.0 / 25.0) * b * c;
    let p1c = (4.0 / 5.0) * e - (2.0 / 25.0) * b * d;
    let p1d = f - b * e / 25.0;

    let q_root = -b / 5.0;
    let mut p_roots = [0.0f64; 3];
    let num_p = solve_cubic(p1a, p1b, p1c, p1d, &mut p_roots);

    let mut intervals = [0.0f64; 4];
    let mut num_intervals = 0usize;
    if q_root >= 0.0 && q_root <= 1.0 {
        intervals[num_intervals] = q_root;
        num_intervals += 1;
    }
    for i in 0..num_p {
        intervals[num_intervals] = p_roots[i];
        num_intervals += 1;
    }

    for j in 1..num_intervals {
        let mut k = j;
        while k > 0 && intervals[k - 1] > intervals[k] {
            intervals.swap(k - 1, k);
            k -= 1;
        }
    }

    // Root finding on the quintic derivative polynomial to refine closest t.
    let eval_poly = |t: f64| -> f64 { ((((t + b) * t + c) * t + d) * t + e) * t + f };
    let eval_poly_deriv = |t: f64| -> f64 {
        (((5.0 * t + 4.0 * b) * t + 3.0 * c) * t + 2.0 * d) * t + e
    };

    let mut lower = 0.0;
    for j in 0..=num_intervals {
        if j < num_intervals && intervals[j] < 0.0 {
            continue;
        }
        let upper = if j < num_intervals {
            intervals[j].min(1.0)
        } else {
            1.0
        };
        let mut lb = lower;
        let mut ub = upper;
        let mut lb_eval = eval_poly(lb);
        let mut ub_eval = eval_poly(ub);
        if lb_eval * ub_eval > 0.0 {
            lower = upper;
            continue;
        }
        if lb_eval > ub_eval {
            std::mem::swap(&mut lb, &mut ub);
            std::mem::swap(&mut lb_eval, &mut ub_eval);
        }
        let mut t = 0.5 * (lb + ub);
        for it in 0..20 {
            if !(t >= lb && t <= ub) {
                t = 0.5 * (lb + ub);
            }
            let value = eval_poly(t);
            if value.abs() < 1.0e-5 || it == 19 {
                break;
            }
            if value > 0.0 {
                ub = t;
            } else {
                lb = t;
            }
            let derivative = eval_poly_deriv(t);
            t -= value / derivative;
        }
        if t >= 0.0 && t <= 1.0 {
            let t32 = t as f32;
            let tt = 1.0 - t32;
            let cp = (p0 * (tt * tt * tt))
                + (p1 * (3.0 * tt * tt * t32))
                + (p2 * (3.0 * tt * t32 * t32))
                + (p3 * (t32 * t32 * t32));
            let dist = (pt - cp).length();
            if dist < best_dist {
                best_dist = dist;
                best_pt = cp;
                best_t = t32;
            }
        }
        if upper >= 1.0 {
            break;
        }
        lower = upper;
    }

    (best_pt, best_t, best_dist)
}

fn quadratic_closest_pt_approx(p0: Vec2, p1: Vec2, p2: Vec2, pt: Vec2) -> (Vec2, f32) {
    let b0 = p0 - pt;
    let b1 = p1 - pt;
    let b2 = p2 - pt;
    let a = b0.cross(b2);
    let b = 2.0 * b1.cross(b0);
    let d = 2.0 * b2.cross(b1);
    let f = b * d - a * a;
    let d21 = b2 - b1;
    let d10 = b1 - b0;
    let d20 = b2 - b0;
    let mut gf = (d21 * b) + (d10 * d) + (d20 * a);
    gf *= 2.0;
    gf = Vec2::new(gf.y, -gf.x);
    let mut t = 0.0;
    let denom = gf.dot(gf);
    if denom > 1.0e-8 {
        let pp = gf * (-f / denom);
        let d0p = b0 - pp;
        let ap = d0p.cross(d20);
        let bp = 2.0 * d10.cross(d0p);
        let denom2 = 2.0 * a + b + d;
        if denom2.abs() > 1.0e-8 {
            t = ((ap + bp) / denom2).clamp(0.0, 1.0);
        }
    }
    let tt = 1.0 - t;
    let cp = p0 * (tt * tt) + p1 * (2.0 * tt * t) + p2 * (t * t);
    (cp, t)
}

fn solve_quadratic(a: f64, b: f64, c: f64, t: &mut [f64; 2]) -> bool {
    let discrim = b * b - 4.0 * a * c;
    if discrim < 0.0 {
        return false;
    }
    let root = discrim.sqrt();
    let q = if b < 0.0 { -0.5 * (b - root) } else { -0.5 * (b + root) };
    t[0] = q / a;
    t[1] = c / q;
    if t[0] > t[1] {
        t.swap(0, 1);
    }
    true
}

fn solve_cubic(a: f64, b: f64, c: f64, d: f64, t: &mut [f64; 3]) -> usize {
    if a.abs() < 1.0e-6 {
        let mut roots = [0.0f64; 2];
        if solve_quadratic(b, c, d, &mut roots) {
            t[0] = roots[0];
            t[1] = roots[1];
            return 2;
        }
        return 0;
    }
    let b = b / a;
    let c = c / a;
    let d = d / a;
    let q = (b * b - 3.0 * c) / 9.0;
    let r = (2.0 * b * b * b - 9.0 * b * c + 27.0 * d) / 54.0;
    if r * r < q * q * q {
        let theta = (r / (q * q * q).sqrt()).acos();
        t[0] = -2.0 * q.sqrt() * (theta / 3.0).cos() - b / 3.0;
        t[1] = -2.0 * q.sqrt() * ((theta + 2.0 * std::f64::consts::PI) / 3.0).cos() - b / 3.0;
        t[2] = -2.0 * q.sqrt() * ((theta - 2.0 * std::f64::consts::PI) / 3.0).cos() - b / 3.0;
        return 3;
    }
    let a_root = if r > 0.0 {
        -cbrt(r + (r * r - q * q * q).sqrt())
    } else {
        cbrt(-r + (r * r - q * q * q).sqrt())
    };
    let b_root = if a_root.abs() > 1.0e-6 { q / a_root } else { 0.0 };
    t[0] = (a_root + b_root) - b / 3.0;
    1
}

fn cbrt(x: f64) -> f64 {
    if x > 0.0 {
        x.powf(1.0 / 3.0)
    } else if x < 0.0 {
        -(-x).powf(1.0 / 3.0)
    } else {
        0.0
    }
}

fn closest_point_circle(center: Vec2, radius: f32, pt: Vec2) -> Vec2 {
    let d = pt - center;
    let len = d.length();
    if len <= 1.0e-8 {
        return Vec2::new(center.x + radius, center.y);
    }
    center + d * (radius / len)
}

fn closest_point_rect(min: Vec2, max: Vec2, pt: Vec2) -> Vec2 {
    let inside = pt.x >= min.x && pt.x <= max.x && pt.y >= min.y && pt.y <= max.y;
    if !inside {
        return Vec2::new(pt.x.clamp(min.x, max.x), pt.y.clamp(min.y, max.y));
    }
    let dl = pt.x - min.x;
    let dr = max.x - pt.x;
    let db = pt.y - min.y;
    let dt = max.y - pt.y;
    if dl <= dr && dl <= db && dl <= dt {
        Vec2::new(min.x, pt.y)
    } else if dr <= db && dr <= dt {
        Vec2::new(max.x, pt.y)
    } else if db <= dt {
        Vec2::new(pt.x, min.y)
    } else {
        Vec2::new(pt.x, max.y)
    }
}

fn rect_signed_distance(pt: Vec2, min: Vec2, max: Vec2) -> f32 {
    let dx = (min.x - pt.x).max(0.0).max(pt.x - max.x);
    let dy = (min.y - pt.y).max(0.0).max(pt.y - max.y);
    let outside = (dx * dx + dy * dy).sqrt();
    let inside = (pt.x - min.x)
        .min(max.x - pt.x)
        .min(pt.y - min.y)
        .min(max.y - pt.y);
    if outside > 0.0 {
        outside
    } else {
        -inside
    }
}

fn ellipse_signed_distance(pt: Vec2, center: Vec2, radius: Vec2) -> f32 {
    let rx = radius.x.abs().max(1.0e-6);
    let ry = radius.y.abs().max(1.0e-6);
    let dx = (pt.x - center.x) / rx;
    let dy = (pt.y - center.y) / ry;
    let len = (dx * dx + dy * dy).sqrt();
    let scale = rx.min(ry);
    (len - 1.0) * scale
}

fn rect_corners(min: Vec2, max: Vec2) -> [Vec2; 4] {
    [
        Vec2::new(min.x, min.y),
        Vec2::new(max.x, min.y),
        Vec2::new(max.x, max.y),
        Vec2::new(min.x, max.y),
    ]
}

fn rect_to_segments(min: Vec2, max: Vec2, transform: Mat3) -> Vec<StrokeSegment> {
    let corners = rect_corners(min, max)
        .into_iter()
        .map(|p| transform.transform_point(p))
        .collect::<Vec<_>>();
    let mut segs = Vec::with_capacity(4);
    for i in 0..4 {
        let a = corners[i];
        let b = corners[(i + 1) % 4];
        segs.push(StrokeSegment::new(a, b, 0.0, 0.0));
    }
    segs
}

fn ellipse_to_segments(center: Vec2, radius: Vec2, transform: Mat3, tolerance: f32) -> Vec<StrokeSegment> {
    let rx = radius.x.abs();
    let ry = radius.y.abs();
    if rx == 0.0 || ry == 0.0 {
        return Vec::new();
    }
    let circumference = 2.0 * core::f32::consts::PI * 0.5 * (rx + ry);
    let mut steps = (circumference / tolerance.max(0.01)).ceil() as usize;
    steps = steps.clamp(12, 256);

    let mut points = Vec::with_capacity(steps);
    for i in 0..steps {
        let angle = (i as f32) * (2.0 * core::f32::consts::PI / steps as f32);
        let (sin, cos) = angle.sin_cos();
        let point = Vec2::new(center.x + cos * rx, center.y + sin * ry);
        points.push(transform.transform_point(point));
    }

    let mut segs = Vec::with_capacity(steps);
    for i in 0..steps {
        let a = points[i];
        let b = points[(i + 1) % steps];
        segs.push(StrokeSegment::new(a, b, 0.0, 0.0));
    }
    segs
}

fn transform_path(path: &Path, transform: Mat3) -> Path {
    if transform.is_identity() {
        return path.clone();
    }
    let points = path
        .points
        .iter()
        .map(|p| transform.transform_point(*p))
        .collect::<Vec<_>>();
    Path {
        num_control_points: path.num_control_points.clone(),
        points,
        thickness: path.thickness.clone(),
        is_closed: path.is_closed,
        use_distance_approx: path.use_distance_approx,
    }
}

fn bounds_from_points(points: &[Vec2]) -> (Vec2, Vec2) {
    let mut min = Vec2::new(f32::INFINITY, f32::INFINITY);
    let mut max = Vec2::new(f32::NEG_INFINITY, f32::NEG_INFINITY);
    for p in points {
        min = min.min(*p);
        max = max.max(*p);
    }
    (min, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{FillRule, Paint, ShapeGroup};
    use crate::{Color, ShapeGeometry, Shape};

    #[test]
    fn test_compute_distance_circle() {
        let mut scene = Scene::new(100, 100);
        let circle = Shape::new(ShapeGeometry::Circle {
            center: Vec2::new(50.0, 50.0),
            radius: 10.0,
        });
        scene.shapes.push(circle);
        scene.groups.push(ShapeGroup::new(vec![0], Some(Paint::Solid(Color::opaque(1.0, 0.0, 0.0))), None));

        let pt = Vec2::new(70.0, 50.0);
        let hit = compute_distance(&scene, 0, pt, f32::INFINITY, DistanceOptions::default());
        assert!(hit.is_some());
        let hit = hit.unwrap();
        assert!((hit.distance - 10.0).abs() < 1.0e-3);
    }

    #[test]
    fn test_within_distance_path_stroke() {
        let mut scene = Scene::new(100, 100);
        let mut path = Path::from_segments(vec![
            crate::geometry::PathSegment::MoveTo(Vec2::new(10.0, 10.0)),
            crate::geometry::PathSegment::LineTo(Vec2::new(90.0, 10.0)),
        ]);
        path.is_closed = false;
        let mut shape = Shape::new(ShapeGeometry::Path { path });
        shape.stroke_width = 4.0;
        scene.shapes.push(shape);
        let mut group = ShapeGroup::new(vec![0], None, Some(Paint::Solid(Color::opaque(0.0, 0.0, 0.0))));
        group.fill_rule = FillRule::NonZero;
        scene.groups.push(group);

        let near = Vec2::new(50.0, 12.0);
        let far = Vec2::new(50.0, 20.0);
        assert!(within_distance(&scene, 0, near));
        assert!(!within_distance(&scene, 0, far));
    }

    #[test]
    fn test_bvh_compute_distance_matches_linear() {
        let mut scene = Scene::new(100, 100);
        let circle = Shape::new(ShapeGeometry::Circle {
            center: Vec2::new(20.0, 40.0),
            radius: 8.0,
        });
        let rect = Shape::new(ShapeGeometry::Rect {
            min: Vec2::new(60.0, 60.0),
            max: Vec2::new(80.0, 80.0),
        });
        scene.shapes.push(circle);
        scene.shapes.push(rect);
        scene.groups.push(ShapeGroup::new(
            vec![0, 1],
            Some(Paint::Solid(Color::opaque(1.0, 0.0, 0.0))),
            None,
        ));

        let bvh = SceneBvh::new(&scene);
        let pt = Vec2::new(30.0, 40.0);
        let linear = compute_distance(&scene, 0, pt, f32::INFINITY, DistanceOptions::default());
        let accel =
            compute_distance_bvh(&scene, &bvh, 0, pt, f32::INFINITY, DistanceOptions::default());
        assert!(linear.is_some());
        assert!(accel.is_some());
        let linear = linear.unwrap();
        let accel = accel.unwrap();
        assert!((linear.distance - accel.distance).abs() < 1.0e-3);
        assert_eq!(linear.shape_index, accel.shape_index);
    }

    #[test]
    fn test_bvh_within_distance_matches_linear() {
        let mut scene = Scene::new(100, 100);
        let mut shape = Shape::new(ShapeGeometry::Rect {
            min: Vec2::new(10.0, 10.0),
            max: Vec2::new(30.0, 30.0),
        });
        shape.stroke_width = 5.0;
        scene.shapes.push(shape);
        scene.groups.push(ShapeGroup::new(
            vec![0],
            None,
            Some(Paint::Solid(Color::opaque(0.0, 0.0, 0.0))),
        ));

        let bvh = SceneBvh::new(&scene);
        let near = Vec2::new(32.0, 20.0);
        let far = Vec2::new(50.0, 20.0);
        assert_eq!(within_distance(&scene, 0, near), within_distance_bvh(&scene, &bvh, 0, near));
        assert_eq!(within_distance(&scene, 0, far), within_distance_bvh(&scene, &bvh, 0, far));
    }

    #[test]
    fn test_bvh_path_distance_matches_linear() {
        let mut scene = Scene::new(120, 120);
        let path = Path::from_segments(vec![
            crate::geometry::PathSegment::MoveTo(Vec2::new(20.0, 20.0)),
            crate::geometry::PathSegment::QuadTo(Vec2::new(60.0, 80.0), Vec2::new(100.0, 20.0)),
            crate::geometry::PathSegment::CubicTo(
                Vec2::new(90.0, 90.0),
                Vec2::new(30.0, 90.0),
                Vec2::new(20.0, 20.0),
            ),
        ]);
        let shape = Shape::new(ShapeGeometry::Path { path });
        scene.shapes.push(shape);
        scene.groups.push(ShapeGroup::new(
            vec![0],
            Some(Paint::Solid(Color::opaque(0.2, 0.3, 0.4))),
            None,
        ));

        let bvh = SceneBvh::new(&scene);
        let pt = Vec2::new(60.0, 40.0);
        let linear = compute_distance(&scene, 0, pt, f32::INFINITY, DistanceOptions::default());
        let accel =
            compute_distance_bvh(&scene, &bvh, 0, pt, f32::INFINITY, DistanceOptions::default());
        assert!(linear.is_some());
        assert!(accel.is_some());
        let linear = linear.unwrap();
        let accel = accel.unwrap();
        assert!((linear.distance - accel.distance).abs() < 1.0e-3);
        assert_eq!(linear.shape_index, accel.shape_index);
    }
}
