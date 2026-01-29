use crate::geometry::{Path, StrokeSegment};
use crate::math::{Mat3, Vec2};
use crate::scene::{Paint, Scene, Shape, ShapeGeometry, StrokeJoin};
use crate::distance::{compute_distance_bvh, is_inside_bvh, DistanceOptions, SceneBvh};
use crate::{color::Color, gpu};
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use cubecl::features::TypeUsage;
use cubecl::ir::{ElemType, FloatKind, StorageType};

const SHAPE_STRIDE: usize = 15;
const GROUP_STRIDE: usize = 16;
const CURVE_STRIDE: usize = 13;
const GRADIENT_STRIDE: usize = 8;
const MAX_F32_INT: usize = 16_777_216;
const DEFAULT_PATH_TOLERANCE: f32 = 0.5;
const TILE_SIZE: u32 = 16;

#[derive(Debug, Copy, Clone)]
pub struct RenderOptions {
    pub aa: f32,
    pub path_tolerance: f32,
    pub samples_x: u32,
    pub samples_y: u32,
    pub seed: u32,
    pub jitter: bool,
    pub use_prefiltering: bool,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            aa: 1.0,
            path_tolerance: DEFAULT_PATH_TOLERANCE,
            samples_x: 2,
            samples_y: 2,
            seed: 0,
            jitter: true,
            use_prefiltering: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<f32>,
}

impl Image {
    pub fn solid(width: u32, height: u32, color: Color) -> Self {
        let mut pixels = vec![0.0; width as usize * height as usize * 4];
        for chunk in pixels.chunks_mut(4) {
            chunk[0] = color.r;
            chunk[1] = color.g;
            chunk[2] = color.b;
            chunk[3] = color.a;
        }
        Self {
            width,
            height,
            pixels,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SdfImage {
    pub width: u32,
    pub height: u32,
    pub values: Vec<f32>,
}

#[derive(Debug)]
pub enum RenderError {
    InvalidScene(&'static str),
    Launch(LaunchError),
}

pub struct Renderer {
    device: WgpuDevice,
}

impl Renderer {
    pub fn new() -> Self {
        Self {
            device: WgpuDevice::default(),
        }
    }

    pub fn with_device(device: WgpuDevice) -> Self {
        Self { device }
    }

    pub fn render(&self, scene: &Scene, options: RenderOptions) -> Result<Image, RenderError> {
        if scene.width == 0 || scene.height == 0 {
            return Ok(Image {
                width: scene.width,
                height: scene.height,
                pixels: Vec::new(),
            });
        }

        if scene.groups.is_empty() {
            if let Some(background) = scene.background_image.as_ref() {
                return Ok(Image {
                    width: scene.width,
                    height: scene.height,
                    pixels: background.clone(),
                });
            }
            return Ok(Image::solid(scene.width, scene.height, scene.background));
        }

        let prepared = prepare_scene(scene, &options)?;
        if prepared.num_groups == 0 {
            if let Some(background) = scene.background_image.as_ref() {
                return Ok(Image {
                    width: scene.width,
                    height: scene.height,
                    pixels: background.clone(),
                });
            }
            return Ok(Image::solid(scene.width, scene.height, scene.background));
        }

        let background_len = scene.width as usize * scene.height as usize * 4;
        let (background_image, has_background_image) = match scene.background_image.as_ref() {
            Some(image) => {
                if image.len() != background_len {
                    return Err(RenderError::InvalidScene("background image size mismatch"));
                }
                (image.clone(), 1u32)
            }
            None => (vec![0.0f32], 0u32),
        };

        let client = WgpuRuntime::client(&self.device);

        let shape_data = ensure_nonempty(prepared.shape_data, 0.0);
        let segment_data = ensure_nonempty(prepared.segment_data, 0.0);
        let shape_bounds = ensure_nonempty(prepared.shape_bounds, 0.0);
        let group_data = ensure_nonempty(prepared.group_data, 0.0);
        let group_xform = ensure_nonempty(prepared.group_xform, 0.0);
        let group_shape_xform = ensure_nonempty(prepared.group_shape_xform, 0.0);
        let group_inv_scale = ensure_nonempty(prepared.group_inv_scale, 1.0);
        let group_shapes = ensure_nonempty(prepared.group_shapes, 0.0);
        let shape_xform = ensure_nonempty(prepared.shape_xform, 0.0);
        let mut group_shape_pairs = prepared.group_shape_pairs;
        if group_shape_pairs.len() % 2 != 0 {
            return Err(RenderError::InvalidScene("group shape pair list is malformed"));
        }
        let num_pairs = (group_shape_pairs.len() / 2) as u32;
        if group_shape_pairs.is_empty() {
            group_shape_pairs = vec![0u32; 2];
        }
        let curve_data = ensure_nonempty(prepared.curve_data, 0.0);
        let gradient_data = ensure_nonempty(prepared.gradient_data, 0.0);
        let stop_offsets = ensure_nonempty(prepared.stop_offsets, 0.0);
        let stop_colors = ensure_nonempty(prepared.stop_colors, 0.0);

        let shape_handle = client.create_from_slice(f32::as_bytes(&shape_data));
        let segment_handle = client.create_from_slice(f32::as_bytes(&segment_data));
        let shape_bounds_handle = client.create_from_slice(f32::as_bytes(&shape_bounds));
        let group_handle = client.create_from_slice(f32::as_bytes(&group_data));
        let group_xform_handle = client.create_from_slice(f32::as_bytes(&group_xform));
        let group_shape_xform_handle = client.create_from_slice(f32::as_bytes(&group_shape_xform));
        let group_inv_scale_handle = client.create_from_slice(f32::as_bytes(&group_inv_scale));
        let group_shapes_handle = client.create_from_slice(f32::as_bytes(&group_shapes));
        let shape_xform_handle = client.create_from_slice(f32::as_bytes(&shape_xform));
        let group_shape_pairs_handle = client.create_from_slice(u32::as_bytes(&group_shape_pairs));
        let curve_handle = client.create_from_slice(f32::as_bytes(&curve_data));
        let gradient_handle = client.create_from_slice(f32::as_bytes(&gradient_data));
        let stop_offsets_handle = client.create_from_slice(f32::as_bytes(&stop_offsets));
        let stop_colors_handle = client.create_from_slice(f32::as_bytes(&stop_colors));

        let output_len = scene.width as usize * scene.height as usize * 4;
        let output_handle = client.empty(output_len * core::mem::size_of::<f32>());
        let background_handle = client.create_from_slice(f32::as_bytes(&background_image));

        let samples_x = options.samples_x.max(1);
        let samples_y = options.samples_y.max(1);
        let use_prefiltering = options.use_prefiltering;
        let jitter = if use_prefiltering {
            0u32
        } else if options.jitter {
            1u32
        } else {
            0u32
        };
        let filter_radius_i = scene.filter.radius.ceil().max(0.0) as u32;

        let total_samples = (scene.width as u64)
            .saturating_mul(scene.height as u64)
            .saturating_mul(samples_x as u64)
            .saturating_mul(samples_y as u64);
        if total_samples > u32::MAX as u64 {
            return Err(RenderError::InvalidScene("too many samples for 1d launch"));
        }
        let total_samples = total_samples as u32;

        let use_float_atomics = client
            .properties()
            .type_usage(StorageType::Atomic(ElemType::Float(FloatKind::F32)))
            .contains(TypeUsage::AtomicAdd);

        let weight_len = scene.width as usize * scene.height as usize;
        let color_len = weight_len * 4;
        let (weight_handle, color_handle) = if use_float_atomics {
            let weight_init = vec![0f32; weight_len];
            let color_init = vec![0f32; color_len];
            (
                client.create_from_slice(f32::as_bytes(&weight_init)),
                client.create_from_slice(f32::as_bytes(&color_init)),
            )
        } else {
            let weight_init = vec![0u32; weight_len];
            let color_init = vec![0u32; color_len];
            (
                client.create_from_slice(u32::as_bytes(&weight_init)),
                client.create_from_slice(u32::as_bytes(&color_init)),
            )
        };

        let tile_count_x = div_ceil(scene.width, TILE_SIZE);
        let tile_count_y = div_ceil(scene.height, TILE_SIZE);
        let num_tiles = tile_count_x as usize * tile_count_y as usize;
        let tile_pixels = (TILE_SIZE as u64) * (TILE_SIZE as u64);
        let samples_per_pixel = samples_x as u64 * samples_y as u64;
        let tile_samples = tile_pixels
            .checked_mul(samples_per_pixel)
            .ok_or(RenderError::InvalidScene("too many samples per tile"))?;
        let total_tile_samples = (num_tiles as u64)
            .checked_mul(tile_samples)
            .ok_or(RenderError::InvalidScene("too many tile samples"))?;
        if total_tile_samples > u32::MAX as u64 {
            return Err(RenderError::InvalidScene("too many tile samples for 1d launch"));
        }
        let total_tile_samples = total_tile_samples as u32;
        let num_entries = num_tiles as u32 + 1;
        let tile_counts_init = vec![0u32; num_tiles];
        let tile_offsets_init = vec![0u32; num_tiles + 1];
        let tile_counts_handle = client.create_from_slice(u32::as_bytes(&tile_counts_init));
        let tile_offsets_a_handle = client.create_from_slice(u32::as_bytes(&tile_offsets_init));
        let tile_offsets_b_handle = client.create_from_slice(u32::as_bytes(&tile_offsets_init));

        unsafe {
            let sample_dim = CubeDim::new_1d(256);
            let weight_count = CubeCount::new_1d(div_ceil(total_samples, sample_dim.x));
            if use_float_atomics {
                gpu::rasterize_weights_f32::launch_unchecked::<WgpuRuntime>(
                    &client,
                    weight_count,
                    sample_dim,
                    ScalarArg::new(scene.width),
                    ScalarArg::new(scene.height),
                    ScalarArg::new(scene.filter.filter_type.as_u32()),
                    ScalarArg::new(scene.filter.radius),
                    ScalarArg::new(filter_radius_i),
                    ScalarArg::new(samples_x),
                    ScalarArg::new(samples_y),
                    ScalarArg::new(options.seed),
                    ScalarArg::new(jitter),
                    ArrayArg::from_raw_parts::<f32>(&weight_handle, weight_len, 1),
                )
                .map_err(RenderError::Launch)?;
            } else {
                gpu::rasterize_weights::launch_unchecked::<WgpuRuntime>(
                    &client,
                    weight_count,
                    sample_dim,
                    ScalarArg::new(scene.width),
                    ScalarArg::new(scene.height),
                    ScalarArg::new(scene.filter.filter_type.as_u32()),
                    ScalarArg::new(scene.filter.radius),
                    ScalarArg::new(filter_radius_i),
                    ScalarArg::new(samples_x),
                    ScalarArg::new(samples_y),
                    ScalarArg::new(options.seed),
                    ScalarArg::new(jitter),
                    ArrayArg::from_raw_parts::<u32>(&weight_handle, weight_len, 1),
                )
                .map_err(RenderError::Launch)?;
            }

            let pair_dim = CubeDim::new_1d(256);
            if num_pairs > 0 {
                let pair_count = CubeCount::new_1d(div_ceil(num_pairs, pair_dim.x));
                gpu::bin_tiles_count::launch_unchecked::<WgpuRuntime>(
                    &client,
                    pair_count,
                    pair_dim,
                    ArrayArg::from_raw_parts::<f32>(&shape_bounds_handle, shape_bounds.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&group_shape_pairs_handle, group_shape_pairs.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&group_shape_xform_handle, group_shape_xform.len(), 1),
                    ScalarArg::new(num_pairs),
                    ScalarArg::new(tile_count_x),
                    ScalarArg::new(tile_count_y),
                    ScalarArg::new(TILE_SIZE),
                    ScalarArg::new(scene.width),
                    ScalarArg::new(scene.height),
                    ArrayArg::from_raw_parts::<u32>(&tile_counts_handle, num_tiles, 1),
                )
                .map_err(RenderError::Launch)?;
            }

            let tile_dim = CubeDim::new_1d(256);
            let tile_count = CubeCount::new_1d(div_ceil(num_tiles as u32, tile_dim.x));
            gpu::init_tile_offsets::launch_unchecked::<WgpuRuntime>(
                &client,
                tile_count,
                tile_dim,
                ArrayArg::from_raw_parts::<u32>(&tile_counts_handle, num_tiles, 1),
                ScalarArg::new(num_tiles as u32),
                ArrayArg::from_raw_parts::<u32>(&tile_offsets_a_handle, num_tiles + 1, 1),
            )
            .map_err(RenderError::Launch)?;

            let entry_count = CubeCount::new_1d(div_ceil(num_entries, tile_dim.x));
            let mut offsets_in_a = true;
            let mut stride = 1u32;
            while stride < num_entries {
                let (src, dst) = if offsets_in_a {
                    (&tile_offsets_a_handle, &tile_offsets_b_handle)
                } else {
                    (&tile_offsets_b_handle, &tile_offsets_a_handle)
                };
                gpu::scan_tile_offsets::launch_unchecked::<WgpuRuntime>(
                    &client,
                    entry_count.clone(),
                    tile_dim,
                    ArrayArg::from_raw_parts::<u32>(src, num_tiles + 1, 1),
                    ArrayArg::from_raw_parts::<u32>(dst, num_tiles + 1, 1),
                    ScalarArg::new(num_entries),
                    ScalarArg::new(stride),
                )
                .map_err(RenderError::Launch)?;
                offsets_in_a = !offsets_in_a;
                stride = stride.saturating_mul(2);
            }

            let offsets_handle = if offsets_in_a {
                tile_offsets_a_handle.clone()
            } else {
                tile_offsets_b_handle.clone()
            };
            let offsets_bytes = client.read_one(offsets_handle.clone());
            let offsets_vec = u32::from_bytes(&offsets_bytes).to_vec();
            let total_entries = *offsets_vec.get(num_tiles).unwrap_or(&0) as usize;
            let entry_len = total_entries
                .checked_mul(2)
                .ok_or(RenderError::InvalidScene("too many tile entries"))?;
            let tile_entries = ensure_nonempty_u32(vec![0u32; entry_len], 0);
            let tile_entries_handle = client.create_from_slice(u32::as_bytes(&tile_entries));
            let mut tile_cursor = if num_tiles > 0 {
                offsets_vec[..num_tiles].to_vec()
            } else {
                Vec::new()
            };
            if tile_cursor.is_empty() {
                tile_cursor.push(0);
            }
            let tile_cursor_handle = client.create_from_slice(u32::as_bytes(&tile_cursor));

            if num_pairs > 0 {
                let pair_count = CubeCount::new_1d(div_ceil(num_pairs, pair_dim.x));
                gpu::bin_tiles_write::launch_unchecked::<WgpuRuntime>(
                    &client,
                    pair_count,
                    pair_dim,
                    ArrayArg::from_raw_parts::<f32>(&shape_bounds_handle, shape_bounds.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&group_shape_pairs_handle, group_shape_pairs.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&group_shape_xform_handle, group_shape_xform.len(), 1),
                    ScalarArg::new(num_pairs),
                    ScalarArg::new(tile_count_x),
                    ScalarArg::new(tile_count_y),
                    ScalarArg::new(TILE_SIZE),
                    ScalarArg::new(scene.width),
                    ScalarArg::new(scene.height),
                    ArrayArg::from_raw_parts::<u32>(&tile_cursor_handle, tile_cursor.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&tile_entries_handle, tile_entries.len(), 1),
                )
                .map_err(RenderError::Launch)?;
            }

            let tile_entries_sorted_handle = if total_entries > 0 {
                let entries_bytes = client.read_one(tile_entries_handle.clone());
                let mut entries_vec = u32::from_bytes(&entries_bytes).to_vec();
                sort_tile_entries(&mut entries_vec, &offsets_vec, num_tiles);
                client.create_from_slice(u32::as_bytes(&entries_vec))
            } else {
                tile_entries_handle.clone()
            };

            let splat_count = CubeCount::new_1d(div_ceil(total_tile_samples, sample_dim.x));
            let use_prefiltering_flag = if use_prefiltering { 1u32 } else { 0u32 };
            if use_float_atomics {
                gpu::rasterize_splat_f32::launch_unchecked::<WgpuRuntime>(
                    &client,
                    splat_count,
                    sample_dim,
                    ArrayArg::from_raw_parts::<f32>(&shape_handle, shape_data.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&segment_handle, segment_data.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&shape_bounds_handle, shape_bounds.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&group_handle, group_data.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&group_xform_handle, group_xform.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&group_inv_scale_handle, group_inv_scale.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&group_shapes_handle, group_shapes.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&shape_xform_handle, shape_xform.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&curve_handle, curve_data.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&gradient_handle, gradient_data.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&stop_offsets_handle, stop_offsets.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&stop_colors_handle, stop_colors.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&offsets_handle, num_tiles + 1, 1),
                    ArrayArg::from_raw_parts::<u32>(&tile_entries_sorted_handle, tile_entries.len(), 1),
                    ScalarArg::new(tile_count_x),
                    ScalarArg::new(tile_count_y),
                    ScalarArg::new(TILE_SIZE),
                    ScalarArg::new(scene.width),
                    ScalarArg::new(scene.height),
                    ScalarArg::new(prepared.num_groups),
                    ArrayArg::from_raw_parts::<f32>(&background_handle, background_image.len(), 1),
                    ScalarArg::new(has_background_image),
                    ScalarArg::new(scene.background.r),
                    ScalarArg::new(scene.background.g),
                    ScalarArg::new(scene.background.b),
                    ScalarArg::new(scene.background.a),
                    ScalarArg::new(scene.filter.filter_type.as_u32()),
                    ScalarArg::new(scene.filter.radius),
                    ScalarArg::new(filter_radius_i),
                    ScalarArg::new(use_prefiltering_flag),
                    ScalarArg::new(samples_x),
                    ScalarArg::new(samples_y),
                    ScalarArg::new(options.seed),
                    ScalarArg::new(jitter),
                    ArrayArg::from_raw_parts::<f32>(&weight_handle, weight_len, 1),
                    ArrayArg::from_raw_parts::<f32>(&color_handle, color_len, 1),
                )
                .map_err(RenderError::Launch)?;
            } else {
                gpu::rasterize_splat::launch_unchecked::<WgpuRuntime>(
                    &client,
                    splat_count,
                    sample_dim,
                    ArrayArg::from_raw_parts::<f32>(&shape_handle, shape_data.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&segment_handle, segment_data.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&shape_bounds_handle, shape_bounds.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&group_handle, group_data.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&group_xform_handle, group_xform.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&group_inv_scale_handle, group_inv_scale.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&group_shapes_handle, group_shapes.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&shape_xform_handle, shape_xform.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&curve_handle, curve_data.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&gradient_handle, gradient_data.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&stop_offsets_handle, stop_offsets.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&stop_colors_handle, stop_colors.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&offsets_handle, num_tiles + 1, 1),
                    ArrayArg::from_raw_parts::<u32>(&tile_entries_sorted_handle, tile_entries.len(), 1),
                    ScalarArg::new(tile_count_x),
                    ScalarArg::new(tile_count_y),
                    ScalarArg::new(TILE_SIZE),
                    ScalarArg::new(scene.width),
                    ScalarArg::new(scene.height),
                    ScalarArg::new(prepared.num_groups),
                    ArrayArg::from_raw_parts::<f32>(&background_handle, background_image.len(), 1),
                    ScalarArg::new(has_background_image),
                    ScalarArg::new(scene.background.r),
                    ScalarArg::new(scene.background.g),
                    ScalarArg::new(scene.background.b),
                    ScalarArg::new(scene.background.a),
                    ScalarArg::new(scene.filter.filter_type.as_u32()),
                    ScalarArg::new(scene.filter.radius),
                    ScalarArg::new(filter_radius_i),
                    ScalarArg::new(use_prefiltering_flag),
                    ScalarArg::new(samples_x),
                    ScalarArg::new(samples_y),
                    ScalarArg::new(options.seed),
                    ScalarArg::new(jitter),
                    ArrayArg::from_raw_parts::<u32>(&weight_handle, weight_len, 1),
                    ArrayArg::from_raw_parts::<u32>(&color_handle, color_len, 1),
                )
                .map_err(RenderError::Launch)?;
            }

            let cube_dim = CubeDim::new_2d(8, 8);
            let cubes_x = div_ceil(scene.width, cube_dim.x);
            let cubes_y = div_ceil(scene.height, cube_dim.y);
            let cube_count = CubeCount::new_2d(cubes_x, cubes_y);
            let bg_r = scene.background.r * scene.background.a;
            let bg_g = scene.background.g * scene.background.a;
            let bg_b = scene.background.b * scene.background.a;
            if use_float_atomics {
                gpu::resolve_splat_f32::launch_unchecked::<WgpuRuntime>(
                    &client,
                    cube_count,
                    cube_dim,
                    ArrayArg::from_raw_parts::<f32>(&weight_handle, weight_len, 1),
                    ArrayArg::from_raw_parts::<f32>(&color_handle, color_len, 1),
                    ScalarArg::new(scene.width),
                    ScalarArg::new(scene.height),
                    ScalarArg::new(bg_r),
                    ScalarArg::new(bg_g),
                    ScalarArg::new(bg_b),
                    ScalarArg::new(scene.background.a),
                    ArrayArg::from_raw_parts::<f32>(&output_handle, output_len, 1),
                )
                .map_err(RenderError::Launch)?;
            } else {
                gpu::resolve_splat::launch_unchecked::<WgpuRuntime>(
                    &client,
                    cube_count,
                    cube_dim,
                    ArrayArg::from_raw_parts::<u32>(&weight_handle, weight_len, 1),
                    ArrayArg::from_raw_parts::<u32>(&color_handle, color_len, 1),
                    ScalarArg::new(scene.width),
                    ScalarArg::new(scene.height),
                    ScalarArg::new(bg_r),
                    ScalarArg::new(bg_g),
                    ScalarArg::new(bg_b),
                    ScalarArg::new(scene.background.a),
                    ArrayArg::from_raw_parts::<f32>(&output_handle, output_len, 1),
                )
                .map_err(RenderError::Launch)?;
            }
        }

        let bytes = client.read_one(output_handle);
        let pixels = f32::from_bytes(&bytes).to_vec();

        Ok(Image {
            width: scene.width,
            height: scene.height,
            pixels,
        })
    }

    pub fn render_sdf(
        &self,
        scene: &Scene,
        options: RenderOptions,
    ) -> Result<SdfImage, RenderError> {
        if scene.width == 0 || scene.height == 0 {
            return Ok(SdfImage {
                width: scene.width,
                height: scene.height,
                values: Vec::new(),
            });
        }
        let pixel_count = (scene.width as usize)
            .checked_mul(scene.height as usize)
            .ok_or(RenderError::InvalidScene("image size overflow"))?;
        let mut values = vec![0.0f32; pixel_count];
        if scene.groups.is_empty() {
            return Ok(SdfImage {
                width: scene.width,
                height: scene.height,
                values,
            });
        }

        let samples_x = options.samples_x.max(1);
        let samples_y = options.samples_y.max(1);
        let total_samples = (samples_x as f32) * (samples_y as f32);
        let weight = if total_samples > 0.0 {
            1.0 / total_samples
        } else {
            1.0
        };
        let use_jitter = options.jitter && !options.use_prefiltering;
        let bvh = SceneBvh::new(scene);
        let dist_options = DistanceOptions {
            path_tolerance: options.path_tolerance,
        };

        for y in 0..scene.height {
            for x in 0..scene.width {
                let mut accum = 0.0f32;
                for sy in 0..samples_y {
                    for sx in 0..samples_x {
                        let mut rx = 0.5f32;
                        let mut ry = 0.5f32;
                        if use_jitter {
                            let canonical = (((y as u64 * scene.width as u64) + x as u64)
                                * samples_y as u64
                                + sy as u64)
                                * samples_x as u64
                                + sx as u64;
                            let mut rng = Pcg32::new(canonical, options.seed as u64);
                            rx = rng.next_f32();
                            ry = rng.next_f32();
                        }
                        let px = x as f32 + (sx as f32 + rx) / samples_x as f32;
                        let py = y as f32 + (sy as f32 + ry) / samples_y as f32;
                        let dist = sample_distance(scene, &bvh, Vec2::new(px, py), dist_options);
                        accum += dist;
                    }
                }
                values[(y * scene.width + x) as usize] = accum * weight;
            }
        }

        Ok(SdfImage {
            width: scene.width,
            height: scene.height,
            values,
        })
    }

    pub fn eval_positions(
        &self,
        scene: &Scene,
        positions: &[Vec2],
        options: RenderOptions,
    ) -> Result<Vec<f32>, RenderError> {
        if positions.is_empty() {
            return Ok(Vec::new());
        }
        let bvh = SceneBvh::new(scene);
        let dist_options = DistanceOptions {
            path_tolerance: options.path_tolerance,
        };
        let mut values = Vec::with_capacity(positions.len());
        for &pt in positions {
            values.push(sample_distance(scene, &bvh, pt, dist_options));
        }
        Ok(values)
    }
}

struct PreparedScene {
    shape_data: Vec<f32>,
    segment_data: Vec<f32>,
    curve_data: Vec<f32>,
    shape_bounds: Vec<f32>,
    group_data: Vec<f32>,
    group_xform: Vec<f32>,
    group_shape_xform: Vec<f32>,
    group_inv_scale: Vec<f32>,
    group_shapes: Vec<f32>,
    shape_xform: Vec<f32>,
    group_shape_pairs: Vec<u32>,
    gradient_data: Vec<f32>,
    stop_offsets: Vec<f32>,
    stop_colors: Vec<f32>,
    num_groups: u32,
}

#[derive(Debug)]
struct PreparedShape {
    kind: u32,
    seg_offset: u32,
    seg_count: u32,
    curve_offset: u32,
    curve_count: u32,
    stroke_width: f32,
    params: [f32; 8],
    use_distance_approx: bool,
    bounds: Option<(Vec2, Vec2)>,
    max_stroke_radius: f32,
}

#[derive(Debug, Copy, Clone)]
struct PaintPack {
    kind: u32,
    gradient_index: u32,
    color: Color,
}


fn apply_stroke_meta(params: &mut [f32; 8], shape: &Shape, has_thickness: bool) {
    params[4] = if has_thickness { 1.0 } else { 0.0 };
    params[5] = shape.stroke_join.as_u32() as f32;
    params[6] = shape.stroke_cap.as_u32() as f32;
    params[7] = shape.stroke_miter_limit;
}

fn prepare_scene(scene: &Scene, options: &RenderOptions) -> Result<PreparedScene, RenderError> {
    let mut segments = Vec::new();
    let mut curves = Vec::new();
    let mut shape_data = Vec::new();
    let mut shape_bounds = Vec::new();
    let mut group_data = Vec::with_capacity(scene.groups.len() * GROUP_STRIDE);
    let mut group_xform = Vec::with_capacity(scene.groups.len() * 6);
    let mut group_shape_xform = Vec::with_capacity(scene.groups.len() * 6);
    let mut group_inv_scale = Vec::with_capacity(scene.groups.len());
    let mut group_shapes = Vec::new();
    let mut shape_xform = Vec::new();
    let mut group_shape_pairs = Vec::new();
    let mut gradient_data = Vec::new();
    let mut stop_offsets = Vec::new();
    let mut stop_colors = Vec::new();

    if scene.shapes.len() > MAX_F32_INT {
        return Err(RenderError::InvalidScene("too many shapes for f32 indexing"));
    }

    for (group_id, group) in scene.groups.iter().enumerate() {
        let shape_offset = group_shapes.len() as u32;
        let mut shape_count = 0u32;
        let group_scale = group.shape_to_canvas.average_scale().max(1.0e-6);
        let inv_scale = 1.0 / group_scale;
        let base_tol = options.path_tolerance.max(0.01);
        let aa_local = if options.use_prefiltering { inv_scale } else { 0.0 };

        let xform = group.canvas_to_shape;
        group_xform.extend_from_slice(&[
            xform.m[0][0],
            xform.m[0][1],
            xform.m[0][2],
            xform.m[1][0],
            xform.m[1][1],
            xform.m[1][2],
        ]);
        let group_shape = group.shape_to_canvas;
        group_shape_xform.extend_from_slice(&[
            group_shape.m[0][0],
            group_shape.m[0][1],
            group_shape.m[0][2],
            group_shape.m[1][0],
            group_shape.m[1][1],
            group_shape.m[1][2],
        ]);
        group_inv_scale.push(inv_scale);

        for &shape_idx in &group.shape_indices {
            if shape_idx >= scene.shapes.len() {
                return Err(RenderError::InvalidScene("group references invalid shape index"));
            }

            let shape = &scene.shapes[shape_idx];
            let total_scale = (group_scale * shape.transform.average_scale()).max(1.0e-6);
            let shape_tolerance = (base_tol / total_scale).max(1.0e-3);
            let prepared = prepare_shape(shape, &mut segments, &mut curves, shape_tolerance);
            let shape_inv = shape.transform.inverse().unwrap_or(Mat3::identity());
            shape_xform.extend_from_slice(&[
                shape_inv.m[0][0],
                shape_inv.m[0][1],
                shape_inv.m[0][2],
                shape_inv.m[1][0],
                shape_inv.m[1][1],
                shape_inv.m[1][2],
            ]);
            let shape_index = (shape_data.len() / SHAPE_STRIDE) as u32;
            group_shapes.push(shape_index as f32);
            group_shape_pairs.push(group_id as u32);
            group_shape_pairs.push(shape_index);

            shape_data.push(prepared.kind as f32);
            shape_data.push(prepared.seg_offset as f32);
            shape_data.push(prepared.seg_count as f32);
            shape_data.push(prepared.stroke_width);
            shape_data.extend_from_slice(&prepared.params);
            shape_data.push(prepared.curve_offset as f32);
            shape_data.push(prepared.curve_count as f32);
            shape_data.push(if prepared.use_distance_approx { 1.0 } else { 0.0 });

            let mut stroke_pad = prepared.max_stroke_radius.abs() * shape.transform.max_scale().max(0.0);
            if shape.stroke_join == StrokeJoin::Miter {
                stroke_pad *= shape.stroke_miter_limit.max(1.0);
            }
            let pad = aa_local + stroke_pad;
            let bounds_group = prepared
                .bounds
                .map(|bounds| transform_bounds(bounds, shape.transform));
            let bounds_local = inflate_bounds(bounds_group, pad);
            if let Some(bounds) = bounds_local {
                shape_bounds.extend_from_slice(&[bounds.0.x, bounds.0.y, bounds.1.x, bounds.1.y]);
            } else {
                shape_bounds.extend_from_slice(&[1.0, 1.0, 0.0, 0.0]);
            }

            shape_count += 1;
        }

        if group_shapes.len() > MAX_F32_INT {
            return Err(RenderError::InvalidScene(
                "too many group shape indices for f32 indexing",
            ));
        }

        let fill_pack = pack_paint(
            group.fill.as_ref(),
            &mut gradient_data,
            &mut stop_offsets,
            &mut stop_colors,
        );
        let stroke_pack = pack_paint(
            group.stroke.as_ref(),
            &mut gradient_data,
            &mut stop_offsets,
            &mut stop_colors,
        );

        group_data.push(shape_offset as f32);
        group_data.push(shape_count as f32);
        group_data.push(fill_pack.kind as f32);
        group_data.push(fill_pack.gradient_index as f32);
        group_data.push(stroke_pack.kind as f32);
        group_data.push(stroke_pack.gradient_index as f32);
        group_data.push(1.0);
        group_data.push(group.fill_rule.as_u32() as f32);
        group_data.push(fill_pack.color.r);
        group_data.push(fill_pack.color.g);
        group_data.push(fill_pack.color.b);
        group_data.push(fill_pack.color.a);
        group_data.push(stroke_pack.color.r);
        group_data.push(stroke_pack.color.g);
        group_data.push(stroke_pack.color.b);
        group_data.push(stroke_pack.color.a);
    }

    let segment_data = segments_to_f32(&segments);
    let curve_data = curve_segments_to_f32(&curves);

    if segment_data.len() / 12 > MAX_F32_INT {
        return Err(RenderError::InvalidScene(
            "too many segments for f32 indexing",
        ));
    }
    if curve_data.len() / CURVE_STRIDE > MAX_F32_INT {
        return Err(RenderError::InvalidScene(
            "too many curve segments for f32 indexing",
        ));
    }

    Ok(PreparedScene {
        shape_data,
        segment_data,
        curve_data,
        shape_bounds,
        group_data,
        group_xform,
        group_shape_xform,
        group_inv_scale,
        group_shapes,
        shape_xform,
        group_shape_pairs,
        gradient_data,
        stop_offsets,
        stop_colors,
        num_groups: scene.groups.len() as u32,
    })
}

fn pack_paint(
    paint: Option<&Paint>,
    gradient_data: &mut Vec<f32>,
    stop_offsets: &mut Vec<f32>,
    stop_colors: &mut Vec<f32>,
) -> PaintPack {
    match paint {
        None => PaintPack {
            kind: 0,
            gradient_index: 0,
            color: Color::TRANSPARENT,
        },
        Some(Paint::Solid(color)) => PaintPack {
            kind: 1,
            gradient_index: 0,
            color: *color,
        },
        Some(Paint::LinearGradient(gradient)) => {
            let index = push_linear_gradient(gradient_data, stop_offsets, stop_colors, gradient);
            PaintPack {
                kind: 2,
                gradient_index: index,
                color: Color::TRANSPARENT,
            }
        }
        Some(Paint::RadialGradient(gradient)) => {
            let index = push_radial_gradient(gradient_data, stop_offsets, stop_colors, gradient);
            PaintPack {
                kind: 3,
                gradient_index: index,
                color: Color::TRANSPARENT,
            }
        }
    }
}

fn push_linear_gradient(
    gradient_data: &mut Vec<f32>,
    stop_offsets: &mut Vec<f32>,
    stop_colors: &mut Vec<f32>,
    gradient: &crate::scene::LinearGradient,
) -> u32 {
    let start = gradient.start;
    let end = gradient.end;

    let stop_offset = stop_offsets.len() as u32;
    let stop_count = gradient.stops.len() as u32;

    for stop in &gradient.stops {
        stop_offsets.push(stop.offset);
        stop_colors.push(stop.color.r);
        stop_colors.push(stop.color.g);
        stop_colors.push(stop.color.b);
        stop_colors.push(stop.color.a);
    }

    let index = (gradient_data.len() / GRADIENT_STRIDE) as u32;
    gradient_data.push(0.0);
    gradient_data.push(start.x);
    gradient_data.push(start.y);
    gradient_data.push(end.x);
    gradient_data.push(end.y);
    gradient_data.push(stop_offset as f32);
    gradient_data.push(stop_count as f32);
    gradient_data.push(0.0);
    index
}

fn push_radial_gradient(
    gradient_data: &mut Vec<f32>,
    stop_offsets: &mut Vec<f32>,
    stop_colors: &mut Vec<f32>,
    gradient: &crate::scene::RadialGradient,
) -> u32 {
    let center = gradient.center;
    let radius = gradient.radius;

    let stop_offset = stop_offsets.len() as u32;
    let stop_count = gradient.stops.len() as u32;

    for stop in &gradient.stops {
        stop_offsets.push(stop.offset);
        stop_colors.push(stop.color.r);
        stop_colors.push(stop.color.g);
        stop_colors.push(stop.color.b);
        stop_colors.push(stop.color.a);
    }

    let index = (gradient_data.len() / GRADIENT_STRIDE) as u32;
    gradient_data.push(1.0);
    gradient_data.push(center.x);
    gradient_data.push(center.y);
    gradient_data.push(radius.x);
    gradient_data.push(radius.y);
    gradient_data.push(stop_offset as f32);
    gradient_data.push(stop_count as f32);
    gradient_data.push(0.0);
    index
}

fn prepare_shape(
    shape: &Shape,
    segments: &mut Vec<StrokeSegment>,
    curves: &mut Vec<CurveSegment>,
    tolerance: f32,
) -> PreparedShape {
    let stroke_width = shape.stroke_width;
    let mut max_stroke_radius = stroke_width.abs();

    match &shape.geometry {
        ShapeGeometry::Circle { center, radius } => {
            let r = radius.abs();
            let mut params = [center.x, center.y, r, 0.0, 0.0, 0.0, 0.0, 0.0];
            apply_stroke_meta(&mut params, shape, false);
            let bounds = Some((
                Vec2::new(center.x - r, center.y - r),
                Vec2::new(center.x + r, center.y + r),
            ));
            PreparedShape {
                kind: 0,
                seg_offset: 0,
                seg_count: 0,
                curve_offset: 0,
                curve_count: 0,
                stroke_width,
                params,
                use_distance_approx: false,
                bounds,
                max_stroke_radius,
            }
        }
        ShapeGeometry::Ellipse { center, radius } => {
            let rx = radius.x.abs();
            let ry = radius.y.abs();
            let mut params = [center.x, center.y, rx, ry, 0.0, 0.0, 0.0, 0.0];
            apply_stroke_meta(&mut params, shape, false);
            let bounds = Some((
                Vec2::new(center.x - rx, center.y - ry),
                Vec2::new(center.x + rx, center.y + ry),
            ));
            PreparedShape {
                kind: 3,
                seg_offset: 0,
                seg_count: 0,
                curve_offset: 0,
                curve_count: 0,
                stroke_width,
                params,
                use_distance_approx: false,
                bounds,
                max_stroke_radius,
            }
        }
        ShapeGeometry::Rect { min, max } => {
            let mut params = [min.x, min.y, max.x, max.y, 0.0, 0.0, 0.0, 0.0];
            apply_stroke_meta(&mut params, shape, false);
            PreparedShape {
                kind: 1,
                seg_offset: 0,
                seg_count: 0,
                curve_offset: 0,
                curve_count: 0,
                stroke_width,
                params,
                use_distance_approx: false,
                bounds: Some((*min, *max)),
                max_stroke_radius,
            }
        }
        ShapeGeometry::Path { path } => {
            let has_thickness = path.thickness.is_some();
            let path_tolerance = tolerance;
            let segs = path.flatten_with_thickness(path_tolerance, stroke_width, 1.0);
            for seg in &segs {
                max_stroke_radius = max_stroke_radius.max(seg.r0.abs()).max(seg.r1.abs());
            }
            let (seg_offset, seg_count, bounds) = push_segments(segments, &segs);
            let path_bounds = if path.points.is_empty() {
                bounds
            } else {
                Some(bounds_from_points(&path.points))
            };
            let (curve_offset, curve_count) = {
                let curve_segments = path_to_curve_segments(path, stroke_width);
                push_curve_segments(curves, &curve_segments)
            };
            let mut params = bounds_to_params(path_bounds);
            apply_stroke_meta(&mut params, shape, has_thickness);
            PreparedShape {
                kind: 2,
                seg_offset,
                seg_count,
                curve_offset,
                curve_count,
                stroke_width,
                params,
                use_distance_approx: path.use_distance_approx,
                bounds: path_bounds,
                max_stroke_radius,
            }
        }
    }
}

fn push_segments(
    segments: &mut Vec<StrokeSegment>,
    new_segments: &[StrokeSegment],
) -> (u32, u32, Option<(Vec2, Vec2)>) {
    let seg_offset = segments.len() as u32;
    let mut bounds: Option<(Vec2, Vec2)> = None;
    for seg in new_segments {
        bounds = Some(match bounds {
            None => (seg.start.min(seg.end), seg.start.max(seg.end)),
            Some((min, max)) => (min.min(seg.start).min(seg.end), max.max(seg.start).max(seg.end)),
        });
    }
    segments.extend_from_slice(new_segments);
    let seg_count = new_segments.len() as u32;
    (seg_offset, seg_count, bounds)
}

#[derive(Debug, Copy, Clone)]
struct CurveSegment {
    kind: u32,
    p0: Vec2,
    p1: Vec2,
    p2: Vec2,
    p3: Vec2,
    r0: f32,
    r1: f32,
    r2: f32,
    r3: f32,
}

impl CurveSegment {
    fn line(p0: Vec2, p1: Vec2, r0: f32, r1: f32) -> Self {
        Self {
            kind: 0,
            p0,
            p1,
            p2: p1,
            p3: p1,
            r0,
            r1,
            r2: r1,
            r3: r1,
        }
    }

    fn quad(p0: Vec2, p1: Vec2, p2: Vec2, r0: f32, r1: f32, r2: f32) -> Self {
        Self {
            kind: 1,
            p0,
            p1,
            p2,
            p3: p2,
            r0,
            r1,
            r2,
            r3: r2,
        }
    }

    fn cubic(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, r0: f32, r1: f32, r2: f32, r3: f32) -> Self {
        Self {
            kind: 2,
            p0,
            p1,
            p2,
            p3,
            r0,
            r1,
            r2,
            r3,
        }
    }
}

fn push_curve_segments(
    curves: &mut Vec<CurveSegment>,
    new_segments: &[CurveSegment],
) -> (u32, u32) {
    let offset = curves.len() as u32;
    curves.extend_from_slice(new_segments);
    let count = new_segments.len() as u32;
    (offset, count)
}

fn push_curve_lines(curves: &mut Vec<CurveSegment>, lines: &[StrokeSegment]) -> (u32, u32) {
    let offset = curves.len() as u32;
    for seg in lines {
        curves.push(CurveSegment::line(seg.start, seg.end, seg.r0, seg.r1));
    }
    let count = (curves.len() as u32).saturating_sub(offset);
    (offset, count)
}

fn path_to_curve_segments(path: &Path, stroke_width: f32) -> Vec<CurveSegment> {
    let mut out = Vec::new();
    if path.is_empty() {
        return out;
    }

    let mut point_id = 0usize;
    let total_points = path.points.len();

    for &num_controls in &path.num_control_points {
        match num_controls {
            0 => {
                let i0 = point_id;
                let i1 = point_id + 1;
                let (p0, r0) = match path_point_radius(path, i0, total_points, stroke_width) {
                    Some(values) => values,
                    None => break,
                };
                let (p1, r1) = match path_point_radius(path, i1, total_points, stroke_width) {
                    Some(values) => values,
                    None => break,
                };
                out.push(CurveSegment::line(p0, p1, r0, r1));
                point_id += 1;
            }
            1 => {
                let i0 = point_id;
                let i1 = point_id + 1;
                let i2 = point_id + 2;
                let (p0, r0) = match path_point_radius(path, i0, total_points, stroke_width) {
                    Some(values) => values,
                    None => break,
                };
                let (p1, r1) = match path_point_radius(path, i1, total_points, stroke_width) {
                    Some(value) => value,
                    None => break,
                };
                let (p2, r2) = match path_point_radius(path, i2, total_points, stroke_width) {
                    Some(value) => value,
                    None => break,
                };
                out.push(CurveSegment::quad(p0, p1, p2, r0, r1, r2));
                point_id += 2;
            }
            2 => {
                let i0 = point_id;
                let i1 = point_id + 1;
                let i2 = point_id + 2;
                let i3 = point_id + 3;
                let (p0, r0) = match path_point_radius(path, i0, total_points, stroke_width) {
                    Some(values) => values,
                    None => break,
                };
                let (p1, r1) = match path_point_radius(path, i1, total_points, stroke_width) {
                    Some(value) => value,
                    None => break,
                };
                let (p2, r2) = match path_point_radius(path, i2, total_points, stroke_width) {
                    Some(value) => value,
                    None => break,
                };
                let (p3, r3) = match path_point_radius(path, i3, total_points, stroke_width) {
                    Some(value) => value,
                    None => break,
                };
                out.push(CurveSegment::cubic(p0, p1, p2, p3, r0, r1, r2, r3));
                point_id += 3;
            }
            _ => break,
        }
    }

    out
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
    stroke_width: f32,
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
        .unwrap_or(stroke_width);
    Some((p, radius))
}

fn bounds_to_params(bounds: Option<(Vec2, Vec2)>) -> [f32; 8] {
    if let Some((min, max)) = bounds {
        [min.x, min.y, max.x, max.y, 0.0, 0.0, 0.0, 0.0]
    } else {
        [0.0; 8]
    }
}

fn transform_bounds(bounds: (Vec2, Vec2), transform: Mat3) -> (Vec2, Vec2) {
    let corners = rect_corners(bounds.0, bounds.1)
        .into_iter()
        .map(|p| transform.transform_point(p))
        .collect::<Vec<_>>();
    bounds_from_points(&corners)
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

fn ellipse_to_segments(
    center: Vec2,
    radius: Vec2,
    transform: Mat3,
    tolerance: f32,
) -> Vec<StrokeSegment> {
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

fn inflate_bounds(bounds: Option<(Vec2, Vec2)>, pad: f32) -> Option<(Vec2, Vec2)> {
    bounds.map(|(min, max)| {
        let pad = pad.max(0.0);
        (
            Vec2::new(min.x - pad, min.y - pad),
            Vec2::new(max.x + pad, max.y + pad),
        )
    })
}

fn segments_to_f32(segments: &[StrokeSegment]) -> Vec<f32> {
    let mut data = Vec::with_capacity(segments.len() * 12);
    for seg in segments {
        data.push(seg.start.x);
        data.push(seg.start.y);
        data.push(seg.end.x);
        data.push(seg.end.y);
        data.push(seg.r0);
        data.push(seg.r1);
        data.push(seg.prev_dir.x);
        data.push(seg.prev_dir.y);
        data.push(seg.next_dir.x);
        data.push(seg.next_dir.y);
        data.push(if seg.start_cap { 1.0 } else { 0.0 });
        data.push(if seg.end_cap { 1.0 } else { 0.0 });
    }
    data
}

fn curve_segments_to_f32(curves: &[CurveSegment]) -> Vec<f32> {
    let mut data = Vec::with_capacity(curves.len() * 13);
    for seg in curves {
        data.push(seg.kind as f32);
        data.push(seg.p0.x);
        data.push(seg.p0.y);
        data.push(seg.p1.x);
        data.push(seg.p1.y);
        data.push(seg.p2.x);
        data.push(seg.p2.y);
        data.push(seg.p3.x);
        data.push(seg.p3.y);
        data.push(seg.r0);
        data.push(seg.r1);
        data.push(seg.r2);
        data.push(seg.r3);
    }
    data
}

fn ensure_nonempty(mut data: Vec<f32>, filler: f32) -> Vec<f32> {
    if data.is_empty() {
        data.push(filler);
    }
    data
}

fn ensure_nonempty_u32(mut data: Vec<u32>, filler: u32) -> Vec<u32> {
    if data.is_empty() {
        data.push(filler);
    }
    data
}

fn div_ceil(value: u32, divisor: u32) -> u32 {
    (value + divisor - 1) / divisor
}

fn sort_tile_entries(entries: &mut [u32], offsets: &[u32], num_tiles: usize) {
    if num_tiles == 0 {
        return;
    }
    let max_tiles = num_tiles.min(offsets.len().saturating_sub(1));
    for tile in 0..max_tiles {
        let start = offsets[tile] as usize * 2;
        let end = offsets[tile + 1] as usize * 2;
        if end <= start || end > entries.len() {
            continue;
        }
        let mut pairs = Vec::with_capacity((end - start) / 2);
        let mut idx = start;
        while idx + 1 < end {
            pairs.push((entries[idx], entries[idx + 1]));
            idx += 2;
        }
        pairs.sort_by(|a, b| {
            let ord = a.0.cmp(&b.0);
            if ord == std::cmp::Ordering::Equal {
                a.1.cmp(&b.1)
            } else {
                ord
            }
        });
        for (i, (group_id, shape_id)) in pairs.into_iter().enumerate() {
            let base = start + i * 2;
            entries[base] = group_id;
            entries[base + 1] = shape_id;
        }
    }
}

fn sample_distance(scene: &Scene, bvh: &SceneBvh, pt: Vec2, options: DistanceOptions) -> f32 {
    let mut best_dist = f32::INFINITY;
    let mut best_group = None;
    for group_id in (0..scene.groups.len()).rev() {
        if let Some(hit) = compute_distance_bvh(scene, bvh, group_id, pt, best_dist, options) {
            if hit.distance < best_dist {
                best_dist = hit.distance;
                best_group = Some(group_id);
            }
        }
    }
    let Some(group_id) = best_group else {
        return 0.0;
    };
    let mut dist = best_dist;
    if scene.groups[group_id].fill.is_some() && is_inside_bvh(scene, bvh, group_id, pt) {
        dist = -dist;
    }
    dist
}

const PCG_MULT: u64 = 6364136223846793005;
const PCG_INIT: u64 = 0x853c49e6748fea9b;

struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    fn new(idx: u64, seed: u64) -> Self {
        let mut rng = Self {
            state: 0,
            inc: ((idx + 1) << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(PCG_INIT.wrapping_add(seed));
        rng.next_u32();
        rng
    }

    fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(PCG_MULT)
            .wrapping_add(self.inc | 1);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    fn next_f32(&mut self) -> f32 {
        let u = self.next_u32();
        let bits = (u >> 9) | 0x3f800000;
        f32::from_bits(bits) - 1.0
    }
}
