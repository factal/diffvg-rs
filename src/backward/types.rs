use crate::math::Vec2;

/// RGB triple used by backward sampling (alpha is stored separately).
#[derive(Clone, Copy)]
pub(super) struct Rgb {
    /// Red channel in linear space.
    pub(super) r: f32,
    /// Green channel in linear space.
    pub(super) g: f32,
    /// Blue channel in linear space.
    pub(super) b: f32,
}

impl Rgb {
    /// Construct an RGB triple without alpha.
    pub(super) fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    /// Return the dot product between two RGB values.
    pub(super) fn dot(self, other: Self) -> f32 {
        self.r * other.r + self.g * other.g + self.b * other.b
    }

    /// Scale each channel by a scalar.
    pub(super) fn scale(self, s: f32) -> Self {
        Self::new(self.r * s, self.g * s, self.b * s)
    }

    /// Add two RGB values component-wise.
    pub(super) fn add(self, other: Self) -> Self {
        Self::new(self.r + other.r, self.g + other.g, self.b + other.b)
    }

    /// Subtract two RGB values component-wise.
    pub(super) fn sub(self, other: Self) -> Self {
        Self::new(self.r - other.r, self.g - other.g, self.b - other.b)
    }
}

/// Tracks which shape/group was queried for inside/outside edge tests.
#[derive(Clone, Copy)]
pub(super) struct EdgeQuery {
    /// Shape group index being tested.
    pub(super) shape_group_id: usize,
    /// Shape index being tested.
    pub(super) shape_id: usize,
    /// Whether the query hit the sampled boundary.
    pub(super) hit: bool,
}

/// Identifies a location along a path segment for gradient attribution.
#[derive(Clone, Copy)]
pub(super) struct PathInfo {
    /// Segment index in the path control list.
    pub(super) base_point_id: usize,
    /// Starting point index for the segment in the flattened point list.
    pub(super) point_id: usize,
    /// Local parameter along the segment in [0, 1].
    pub(super) t: f32,
}

/// Metadata for a sampled boundary point that lies on a path segment.
#[derive(Clone, Copy)]
pub(super) struct PathBoundaryData {
    /// Segment index in the path control list.
    pub(super) base_point_id: usize,
    /// Starting point index for the segment in the flattened point list.
    pub(super) point_id: usize,
    /// Local parameter along the segment in [0, 1].
    pub(super) t: f32,
}

/// Extra data used to accumulate boundary gradients.
#[derive(Clone, Copy)]
pub(super) struct BoundaryData {
    /// Path segment metadata (unused for non-path shapes).
    pub(super) path: PathBoundaryData,
    /// True if the sample comes from the stroke boundary.
    pub(super) is_stroke: bool,
}

/// Fully specified boundary sample used in edge gradient accumulation.
#[derive(Clone, Copy)]
pub(super) struct BoundarySample {
    /// Normalized canvas-space position (0..1 range).
    pub(super) pt: Vec2,
    /// Boundary position in the shape's local space.
    pub(super) local_pt: Vec2,
    /// Outward normal in canvas space.
    pub(super) normal: Vec2,
    /// Shape group index associated with the sample.
    pub(super) shape_group_id: usize,
    /// Shape index associated with the sample.
    pub(super) shape_id: usize,
    /// Primary random sample in [0, 1].
    pub(super) t: f32,
    /// Shape/path metadata for gradient attribution.
    pub(super) data: BoundaryData,
    /// Combined probability density of selecting this boundary point.
    pub(super) pdf: f32,
}

/// Color contribution for a single shape evaluation.
#[derive(Clone, Copy)]
pub(super) struct Fragment {
    /// RGB color in linear space.
    pub(super) color: Rgb,
    /// Alpha channel in linear space.
    pub(super) alpha: f32,
    /// Shape group index for the fragment.
    pub(super) group_id: usize,
    /// True if the fragment belongs to a stroke.
    pub(super) is_stroke: bool,
}

/// Prefiltering payload for boundary-aware color evaluation.
#[derive(Clone, Copy)]
pub(super) struct PrefilterFragment {
    /// RGB color in linear space.
    pub(super) color: Rgb,
    /// Alpha channel in linear space.
    pub(super) alpha: f32,
    /// Shape group index for the fragment.
    pub(super) group_id: usize,
    /// Shape index for the fragment.
    pub(super) shape_id: usize,
    /// Signed distance from the sample to the shape boundary.
    pub(super) distance: f32,
    /// Closest point on the shape boundary (canvas space).
    pub(super) closest_pt: Vec2,
    /// Optional path segment metadata (when the closest point lies on a path).
    pub(super) path_info: Option<PathInfo>,
    /// True if the point is within the prefilter distance band.
    pub(super) within_distance: bool,
    /// True if the fragment is for a stroke boundary.
    pub(super) is_stroke: bool,
}

/// Discrete distribution over shapes for boundary sampling.
#[derive(Clone)]
pub(super) struct ShapeCdf {
    /// Shape indices aligned with the CDF entries.
    pub(super) shape_ids: Vec<usize>,
    /// Shape group indices aligned with the CDF entries.
    pub(super) group_ids: Vec<usize>,
    /// Cumulative distribution function (monotonic, ends at 1.0).
    pub(super) cdf: Vec<f32>,
    /// Probability mass function aligned with `cdf`.
    pub(super) pmf: Vec<f32>,
}

/// Per-path segment distribution used when sampling boundary points.
#[derive(Clone)]
pub(super) struct PathCdf {
    /// CDF over segments in the path.
    pub(super) cdf: Vec<f32>,
    /// PMF over segments in the path.
    pub(super) pmf: Vec<f32>,
    /// Maps CDF indices to starting point indices in the path data.
    pub(super) point_id_map: Vec<usize>,
    /// Total estimated length of the path boundary.
    pub(super) length: f32,
}
