use crate::math::Vec2;

#[derive(Clone, Copy)]
pub(super) struct Rgb {
    pub(super) r: f32,
    pub(super) g: f32,
    pub(super) b: f32,
}

impl Rgb {
    pub(super) fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    pub(super) fn dot(self, other: Self) -> f32 {
        self.r * other.r + self.g * other.g + self.b * other.b
    }

    pub(super) fn scale(self, s: f32) -> Self {
        Self::new(self.r * s, self.g * s, self.b * s)
    }

    pub(super) fn add(self, other: Self) -> Self {
        Self::new(self.r + other.r, self.g + other.g, self.b + other.b)
    }

    pub(super) fn sub(self, other: Self) -> Self {
        Self::new(self.r - other.r, self.g - other.g, self.b - other.b)
    }
}

#[derive(Clone, Copy)]
pub(super) struct EdgeQuery {
    pub(super) shape_group_id: usize,
    pub(super) shape_id: usize,
    pub(super) hit: bool,
}

#[derive(Clone, Copy)]
pub(super) struct PathInfo {
    pub(super) base_point_id: usize,
    pub(super) point_id: usize,
    pub(super) t: f32,
}

#[derive(Clone, Copy)]
pub(super) struct PathBoundaryData {
    pub(super) base_point_id: usize,
    pub(super) point_id: usize,
    pub(super) t: f32,
}

#[derive(Clone, Copy)]
pub(super) struct BoundaryData {
    pub(super) path: PathBoundaryData,
    pub(super) is_stroke: bool,
}

#[derive(Clone, Copy)]
pub(super) struct BoundarySample {
    pub(super) pt: Vec2,
    pub(super) local_pt: Vec2,
    pub(super) normal: Vec2,
    pub(super) shape_group_id: usize,
    pub(super) shape_id: usize,
    pub(super) t: f32,
    pub(super) data: BoundaryData,
    pub(super) pdf: f32,
}

#[derive(Clone, Copy)]
pub(super) struct Fragment {
    pub(super) color: Rgb,
    pub(super) alpha: f32,
    pub(super) group_id: usize,
    pub(super) is_stroke: bool,
}

#[derive(Clone, Copy)]
pub(super) struct PrefilterFragment {
    pub(super) color: Rgb,
    pub(super) alpha: f32,
    pub(super) group_id: usize,
    pub(super) shape_id: usize,
    pub(super) distance: f32,
    pub(super) closest_pt: Vec2,
    pub(super) path_info: Option<PathInfo>,
    pub(super) within_distance: bool,
    pub(super) is_stroke: bool,
}

#[derive(Clone)]
pub(super) struct ShapeCdf {
    pub(super) shape_ids: Vec<usize>,
    pub(super) group_ids: Vec<usize>,
    pub(super) cdf: Vec<f32>,
    pub(super) pmf: Vec<f32>,
}

#[derive(Clone)]
pub(super) struct PathCdf {
    pub(super) cdf: Vec<f32>,
    pub(super) pmf: Vec<f32>,
    pub(super) point_id_map: Vec<usize>,
    pub(super) length: f32,
}
