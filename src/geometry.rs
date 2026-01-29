use crate::math::Vec2;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct LineSegment {
    pub start: Vec2,
    pub end: Vec2,
}

impl LineSegment {
    pub const fn new(start: Vec2, end: Vec2) -> Self {
        Self { start, end }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct StrokeSegment {
    pub start: Vec2,
    pub end: Vec2,
    pub r0: f32,
    pub r1: f32,
    pub prev_dir: Vec2,
    pub next_dir: Vec2,
    pub start_cap: bool,
    pub end_cap: bool,
}

impl StrokeSegment {
    pub const fn new(start: Vec2, end: Vec2, r0: f32, r1: f32) -> Self {
        Self {
            start,
            end,
            r0,
            r1,
            prev_dir: Vec2::ZERO,
            next_dir: Vec2::ZERO,
            start_cap: false,
            end_cap: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PathSegment {
    MoveTo(Vec2),
    LineTo(Vec2),
    QuadTo(Vec2, Vec2),
    CubicTo(Vec2, Vec2, Vec2),
    Close,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Path {
    pub num_control_points: Vec<u8>,
    pub points: Vec<Vec2>,
    pub thickness: Option<Vec<f32>>,
    pub is_closed: bool,
    pub use_distance_approx: bool,
}

impl Path {
    pub fn new(num_control_points: Vec<u8>, points: Vec<Vec2>) -> Self {
        Self {
            num_control_points,
            points,
            thickness: None,
            is_closed: false,
            use_distance_approx: false,
        }
    }

    pub fn from_segments(segments: Vec<PathSegment>) -> Self {
        let mut points = Vec::new();
        let mut num_control_points = Vec::new();
        let mut has_current = false;
        let mut is_closed = false;

        for seg in segments {
            match seg {
                PathSegment::MoveTo(p) => {
                    if !has_current {
                        points.push(p);
                        has_current = true;
                    }
                }
                PathSegment::LineTo(p) => {
                    if has_current {
                        num_control_points.push(0);
                        points.push(p);
                    } else {
                        points.push(p);
                        has_current = true;
                    }
                }
                PathSegment::QuadTo(c, p) => {
                    if has_current {
                        num_control_points.push(1);
                        points.push(c);
                        points.push(p);
                    } else {
                        points.push(p);
                        has_current = true;
                    }
                }
                PathSegment::CubicTo(c1, c2, p) => {
                    if has_current {
                        num_control_points.push(2);
                        points.push(c1);
                        points.push(c2);
                        points.push(p);
                    } else {
                        points.push(p);
                        has_current = true;
                    }
                }
                PathSegment::Close => {
                    if has_current {
                        is_closed = true;
                    }
                }
            }
        }

        Self {
            num_control_points,
            points,
            thickness: None,
            is_closed,
            use_distance_approx: false,
        }
    }

    pub fn with_thickness(mut self, thickness: Vec<f32>) -> Self {
        self.thickness = Some(thickness);
        self
    }

    pub fn with_closed(mut self, is_closed: bool) -> Self {
        self.is_closed = is_closed;
        self
    }

    pub fn with_distance_approx(mut self, use_distance_approx: bool) -> Self {
        self.use_distance_approx = use_distance_approx;
        self
    }

    pub fn is_empty(&self) -> bool {
        self.num_control_points.is_empty() || self.points.is_empty()
    }

    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    pub fn num_base_points(&self) -> usize {
        self.num_control_points.len()
    }

    pub fn flatten(&self, tolerance: f32) -> Vec<LineSegment> {
        self.flatten_with_thickness(tolerance, 0.0, 1.0)
            .into_iter()
            .map(|seg| LineSegment::new(seg.start, seg.end))
            .collect()
    }

    pub fn flatten_with_thickness(
        &self,
        tolerance: f32,
        default_radius: f32,
        thickness_scale: f32,
    ) -> Vec<StrokeSegment> {
        let mut out = Vec::new();
        if self.is_empty() {
            return out;
        }

        let mut point_id = 0usize;
        let total_points = self.points.len();

        for &num_controls in &self.num_control_points {
            match num_controls {
                0 => {
                    let i0 = point_id;
                    let i1 = point_id + 1;
                    let (p0, p1, r0, r1) = match self.segment_points(
                        i0,
                        i1,
                        default_radius,
                        thickness_scale,
                        total_points,
                    ) {
                        Some(values) => values,
                        None => break,
                    };
                    out.push(StrokeSegment::new(p0, p1, r0, r1));
                    point_id += 1;
                }
                1 => {
                    let i0 = point_id;
                    let i1 = point_id + 1;
                    let i2 = point_id + 2;
                    let (p0, p1, r0, r1) = match self.segment_points(
                        i0,
                        i1,
                        default_radius,
                        thickness_scale,
                        total_points,
                    ) {
                        Some(values) => values,
                        None => break,
                    };
                    let (p2, r2) = match self.point_and_radius(i2, default_radius, thickness_scale, total_points)
                    {
                        Some(values) => values,
                        None => break,
                    };
                    flatten_quad_thick(p0, p1, p2, r0, r1, r2, tolerance, 0, &mut out);
                    point_id += 2;
                }
                2 => {
                    let i0 = point_id;
                    let i1 = point_id + 1;
                    let i2 = point_id + 2;
                    let i3 = point_id + 3;
                    let (p0, p1, r0, r1) = match self.segment_points(
                        i0,
                        i1,
                        default_radius,
                        thickness_scale,
                        total_points,
                    ) {
                        Some(values) => values,
                        None => break,
                    };
                    let (p2, r2) = match self.point_and_radius(i2, default_radius, thickness_scale, total_points)
                    {
                        Some(values) => values,
                        None => break,
                    };
                    let (p3, r3) = match self.point_and_radius(i3, default_radius, thickness_scale, total_points)
                    {
                        Some(values) => values,
                        None => break,
                    };
                    flatten_cubic_thick(p0, p1, p2, p3, r0, r1, r2, r3, tolerance, 0, &mut out);
                    point_id += 3;
                }
                _ => {
                    break;
                }
            }
        }

        annotate_segments(&mut out, self.is_closed);
        out
    }

    fn segment_points(
        &self,
        i0: usize,
        i1: usize,
        default_radius: f32,
        thickness_scale: f32,
        total_points: usize,
    ) -> Option<(Vec2, Vec2, f32, f32)> {
        let (p0, r0) = self.point_and_radius(i0, default_radius, thickness_scale, total_points)?;
        let (p1, r1) = self.point_and_radius(i1, default_radius, thickness_scale, total_points)?;
        Some((p0, p1, r0, r1))
    }

    fn point_and_radius(
        &self,
        index: usize,
        default_radius: f32,
        thickness_scale: f32,
        total_points: usize,
    ) -> Option<(Vec2, f32)> {
        if total_points == 0 {
            return None;
        }
        if self.is_closed {
            let idx = index % total_points;
            let p = self.points[idx];
            let r = self
                .thickness
                .as_ref()
                .and_then(|values| values.get(idx).copied())
                .map(|value| value * thickness_scale)
                .unwrap_or(default_radius);
            return Some((p, r));
        }
        if index >= total_points {
            return None;
        }
        let p = self.points[index];
        let r = self
            .thickness
            .as_ref()
            .and_then(|values| values.get(index).copied())
            .map(|value| value * thickness_scale)
            .unwrap_or(default_radius);
        Some((p, r))
    }
}

const MAX_SUBDIVISION_DEPTH: u32 = 16;

fn flatten_quad_thick(
    p0: Vec2,
    p1: Vec2,
    p2: Vec2,
    r0: f32,
    r1: f32,
    r2: f32,
    tolerance: f32,
    depth: u32,
    out: &mut Vec<StrokeSegment>,
) {
    if depth >= MAX_SUBDIVISION_DEPTH || quad_flat_enough(p0, p1, p2, tolerance) {
        out.push(StrokeSegment::new(p0, p2, r0, r2));
        return;
    }

    let p01 = p0.lerp(p1, 0.5);
    let p12 = p1.lerp(p2, 0.5);
    let p012 = p01.lerp(p12, 0.5);

    let r01 = 0.5 * (r0 + r1);
    let r12 = 0.5 * (r1 + r2);
    let r012 = 0.5 * (r01 + r12);

    flatten_quad_thick(p0, p01, p012, r0, r01, r012, tolerance, depth + 1, out);
    flatten_quad_thick(p012, p12, p2, r012, r12, r2, tolerance, depth + 1, out);
}

fn flatten_cubic_thick(
    p0: Vec2,
    p1: Vec2,
    p2: Vec2,
    p3: Vec2,
    r0: f32,
    r1: f32,
    r2: f32,
    r3: f32,
    tolerance: f32,
    depth: u32,
    out: &mut Vec<StrokeSegment>,
) {
    if depth >= MAX_SUBDIVISION_DEPTH || cubic_flat_enough(p0, p1, p2, p3, tolerance) {
        out.push(StrokeSegment::new(p0, p3, r0, r3));
        return;
    }

    let p01 = p0.lerp(p1, 0.5);
    let p12 = p1.lerp(p2, 0.5);
    let p23 = p2.lerp(p3, 0.5);
    let p012 = p01.lerp(p12, 0.5);
    let p123 = p12.lerp(p23, 0.5);
    let p0123 = p012.lerp(p123, 0.5);

    let r01 = 0.5 * (r0 + r1);
    let r12 = 0.5 * (r1 + r2);
    let r23 = 0.5 * (r2 + r3);
    let r012 = 0.5 * (r01 + r12);
    let r123 = 0.5 * (r12 + r23);
    let r0123 = 0.5 * (r012 + r123);

    flatten_cubic_thick(p0, p01, p012, p0123, r0, r01, r012, r0123, tolerance, depth + 1, out);
    flatten_cubic_thick(p0123, p123, p23, p3, r0123, r123, r23, r3, tolerance, depth + 1, out);
}

fn annotate_segments(segments: &mut [StrokeSegment], is_closed: bool) {
    let count = segments.len();
    if count == 0 {
        return;
    }

    let mut dirs = Vec::with_capacity(count);
    for seg in segments.iter() {
        let v = seg.end - seg.start;
        let len = v.length();
        if len > 0.0 {
            dirs.push(v / len);
        } else {
            dirs.push(Vec2::ZERO);
        }
    }

    for i in 0..count {
        segments[i].start_cap = !is_closed && i == 0;
        segments[i].end_cap = !is_closed && i + 1 == count;

        let prev_dir = if i == 0 {
            if is_closed {
                let d = dirs[count - 1];
                Vec2::new(-d.x, -d.y)
            } else {
                Vec2::ZERO
            }
        } else {
            let d = dirs[i - 1];
            Vec2::new(-d.x, -d.y)
        };

        let next_dir = if i + 1 < count {
            dirs[i + 1]
        } else if is_closed {
            dirs[0]
        } else {
            Vec2::ZERO
        };

        segments[i].prev_dir = prev_dir;
        segments[i].next_dir = next_dir;
    }
}

fn quad_flat_enough(p0: Vec2, p1: Vec2, p2: Vec2, tolerance: f32) -> bool {
    let dist = distance_point_line(p1, p0, p2);
    dist <= tolerance
}

fn cubic_flat_enough(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, tolerance: f32) -> bool {
    let dist1 = distance_point_line(p1, p0, p3);
    let dist2 = distance_point_line(p2, p0, p3);
    dist1.max(dist2) <= tolerance
}

fn distance_point_line(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let ap = p - a;
    let area = ab.cross(ap).abs();
    let len = ab.length();
    if len == 0.0 {
        ap.length()
    } else {
        area / len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flatten_quadratic_produces_segments() {
        let path = Path::from_segments(vec![
            PathSegment::MoveTo(Vec2::new(0.0, 0.0)),
            PathSegment::QuadTo(Vec2::new(0.5, 1.0), Vec2::new(1.0, 0.0)),
        ]);
        let segments = path.flatten(0.01);
        assert!(!segments.is_empty());
        assert_eq!(segments.first().unwrap().start, Vec2::new(0.0, 0.0));
        assert_eq!(segments.last().unwrap().end, Vec2::new(1.0, 0.0));
    }
}
