//! GPU renderer implementation and SDF helpers.

use crate::backward::BackwardOptions;
use crate::grad::SceneGrad;
use crate::math::Vec2;
use crate::scene::Scene;
use crate::gpu;
use cubecl::features::TypeUsage;
use cubecl::ir::{ElemType, FloatKind, StorageType};
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

use super::constants::TILE_SIZE;
use super::backward_gpu::render_backward_gpu;
use super::prepare::prepare_scene;
use super::tiles::{build_tile_order, div_ceil, sort_tile_entries};
use super::types::{Image, RenderError, RenderOptions, SdfImage};
use super::utils::{ensure_nonempty, ensure_nonempty_u32};

/// GPU-backed renderer targeting WGPU via CubeCL.
pub struct Renderer {
    device: WgpuDevice,
}

impl Renderer {
    /// Construct a renderer using the default WGPU device.
    pub fn new() -> Self {
        Self {
            device: WgpuDevice::default(),
        }
    }

    /// Construct a renderer with a caller-provided device.
    pub fn with_device(device: WgpuDevice) -> Self {
        Self { device }
    }

    /// Render a scene into an RGBA image.
    pub fn render(&self, scene: &Scene, options: RenderOptions) -> Result<Image, RenderError> {
        if let Some(background) = scene.background_image.as_ref() {
            let expected_len = (scene.width as usize)
                .checked_mul(scene.height as usize)
                .and_then(|v| v.checked_mul(4))
                .ok_or(RenderError::InvalidScene("image size overflow"))?;
            if background.len() != expected_len {
                return Err(RenderError::InvalidScene("background image size mismatch"));
            }
        }
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

        // CPU preprocessing: flatten paths, compute bounds, pack GPU buffers.
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

        // Ensure all GPU buffers are non-empty to satisfy WGPU binding requirements.
        let shape_data = ensure_nonempty(prepared.shape_data, 0.0);
        let segment_data = ensure_nonempty(prepared.segment_data, 0.0);
        let shape_bounds = ensure_nonempty(prepared.shape_bounds, 0.0);
        let group_bounds = ensure_nonempty(prepared.group_bounds, 0.0);
        let group_data = ensure_nonempty(prepared.group_data, 0.0);
        let group_xform = ensure_nonempty(prepared.group_xform, 0.0);
        let group_shape_xform = ensure_nonempty(prepared.group_shape_xform, 0.0);
        let group_inv_scale = ensure_nonempty(prepared.group_inv_scale, 1.0);
        let group_shapes = ensure_nonempty(prepared.group_shapes, 0.0);
        let shape_xform = ensure_nonempty(prepared.shape_xform, 0.0);
        let group_bvh_bounds = ensure_nonempty(prepared.group_bvh_bounds, 0.0);
        let group_bvh_nodes = ensure_nonempty_u32(prepared.group_bvh_nodes, 0);
        let group_bvh_indices = ensure_nonempty_u32(prepared.group_bvh_indices, 0);
        let group_bvh_meta = ensure_nonempty_u32(prepared.group_bvh_meta, 0);
        let path_bvh_bounds = ensure_nonempty(prepared.path_bvh_bounds, 0.0);
        let path_bvh_nodes = ensure_nonempty_u32(prepared.path_bvh_nodes, 0);
        let path_bvh_indices = ensure_nonempty_u32(prepared.path_bvh_indices, 0);
        let path_bvh_meta = ensure_nonempty_u32(prepared.path_bvh_meta, 0);
        let curve_data = ensure_nonempty(prepared.curve_data, 0.0);
        let gradient_data = ensure_nonempty(prepared.gradient_data, 0.0);
        let stop_offsets = ensure_nonempty(prepared.stop_offsets, 0.0);
        let stop_colors = ensure_nonempty(prepared.stop_colors, 0.0);

        let shape_handle = client.create_from_slice(f32::as_bytes(&shape_data));
        let segment_handle = client.create_from_slice(f32::as_bytes(&segment_data));
        let shape_bounds_handle = client.create_from_slice(f32::as_bytes(&shape_bounds));
        let group_bounds_handle = client.create_from_slice(f32::as_bytes(&group_bounds));
        let group_handle = client.create_from_slice(f32::as_bytes(&group_data));
        let group_xform_handle = client.create_from_slice(f32::as_bytes(&group_xform));
        let group_shape_xform_handle = client.create_from_slice(f32::as_bytes(&group_shape_xform));
        let group_inv_scale_handle = client.create_from_slice(f32::as_bytes(&group_inv_scale));
        let group_shapes_handle = client.create_from_slice(f32::as_bytes(&group_shapes));
        let shape_xform_handle = client.create_from_slice(f32::as_bytes(&shape_xform));
        let group_bvh_bounds_handle = client.create_from_slice(f32::as_bytes(&group_bvh_bounds));
        let group_bvh_nodes_handle = client.create_from_slice(u32::as_bytes(&group_bvh_nodes));
        let group_bvh_indices_handle = client.create_from_slice(u32::as_bytes(&group_bvh_indices));
        let group_bvh_meta_handle = client.create_from_slice(u32::as_bytes(&group_bvh_meta));
        let path_bvh_bounds_handle = client.create_from_slice(f32::as_bytes(&path_bvh_bounds));
        let path_bvh_nodes_handle = client.create_from_slice(u32::as_bytes(&path_bvh_nodes));
        let path_bvh_indices_handle = client.create_from_slice(u32::as_bytes(&path_bvh_indices));
        let path_bvh_meta_handle = client.create_from_slice(u32::as_bytes(&path_bvh_meta));
        let curve_handle = client.create_from_slice(f32::as_bytes(&curve_data));
        let gradient_handle = client.create_from_slice(f32::as_bytes(&gradient_data));
        let stop_offsets_handle = client.create_from_slice(f32::as_bytes(&stop_offsets));
        let stop_colors_handle = client.create_from_slice(f32::as_bytes(&stop_colors));
        let num_groups = prepared.num_groups;

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
        // Switch accumulation buffers based on float atomics support.
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
            // 1) Precompute filter weights for each subpixel sample.
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

            // 2) Bin shape groups into tiles (GPU prefix-sum + CPU sort).
            let group_dim = CubeDim::new_1d(256);
            if num_groups > 0 {
                let group_count = CubeCount::new_1d(div_ceil(num_groups, group_dim.x));
                gpu::bin_tiles_count::launch_unchecked::<WgpuRuntime>(
                    &client,
                    group_count,
                    group_dim,
                    ArrayArg::from_raw_parts::<f32>(&group_bounds_handle, group_bounds.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(
                        &group_shape_xform_handle,
                        group_shape_xform.len(),
                        1,
                    ),
                    ScalarArg::new(num_groups),
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
            let mut offsets_vec = u32::from_bytes(&offsets_bytes).to_vec();
            if offsets_vec.len() < num_tiles + 1 {
                offsets_vec.resize(num_tiles + 1, 0);
            }
            let total_entries = offsets_vec.get(num_tiles).copied().unwrap_or(0) as usize;
            if total_entries > u32::MAX as usize {
                return Err(RenderError::InvalidScene("too many tile entries"));
            }
            let tile_entries = ensure_nonempty_u32(vec![0u32; total_entries], 0);
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

            if num_groups > 0 {
                let group_count = CubeCount::new_1d(div_ceil(num_groups, group_dim.x));
                gpu::bin_tiles_write::launch_unchecked::<WgpuRuntime>(
                    &client,
                    group_count,
                    group_dim,
                    ArrayArg::from_raw_parts::<f32>(&group_bounds_handle, group_bounds.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(
                        &group_shape_xform_handle,
                        group_shape_xform.len(),
                        1,
                    ),
                    ScalarArg::new(num_groups),
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

            let mut tile_counts = Vec::with_capacity(num_tiles.max(1));
            if num_tiles > 0 {
                for tile_id in 0..num_tiles {
                    let start = offsets_vec.get(tile_id).copied().unwrap_or(0);
                    let end = offsets_vec.get(tile_id + 1).copied().unwrap_or(start);
                    tile_counts.push(end.saturating_sub(start));
                }
            } else {
                tile_counts.push(0);
            }
            // Interleave heavy/light tiles to reduce tail latency.
            let tile_order = build_tile_order(&tile_counts, num_tiles as u32);
            let tile_order_handle = client.create_from_slice(u32::as_bytes(&tile_order));

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
                    ArrayArg::from_raw_parts::<f32>(&group_bvh_bounds_handle, group_bvh_bounds.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&group_bvh_nodes_handle, group_bvh_nodes.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&group_bvh_indices_handle, group_bvh_indices.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&group_bvh_meta_handle, group_bvh_meta.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&path_bvh_bounds_handle, path_bvh_bounds.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&path_bvh_nodes_handle, path_bvh_nodes.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&path_bvh_indices_handle, path_bvh_indices.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&path_bvh_meta_handle, path_bvh_meta.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&offsets_handle, num_tiles + 1, 1),
                    ArrayArg::from_raw_parts::<u32>(&tile_entries_sorted_handle, tile_entries.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&tile_order_handle, tile_order.len(), 1),
                    ScalarArg::new(tile_count_x),
                    ScalarArg::new(tile_count_y),
                    ScalarArg::new(TILE_SIZE),
                    ScalarArg::new(scene.width),
                    ScalarArg::new(scene.height),
                    ScalarArg::new(num_groups),
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
                    ArrayArg::from_raw_parts::<f32>(&group_bvh_bounds_handle, group_bvh_bounds.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&group_bvh_nodes_handle, group_bvh_nodes.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&group_bvh_indices_handle, group_bvh_indices.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&group_bvh_meta_handle, group_bvh_meta.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&path_bvh_bounds_handle, path_bvh_bounds.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&path_bvh_nodes_handle, path_bvh_nodes.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&path_bvh_indices_handle, path_bvh_indices.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&path_bvh_meta_handle, path_bvh_meta.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&offsets_handle, num_tiles + 1, 1),
                    ArrayArg::from_raw_parts::<u32>(&tile_entries_sorted_handle, tile_entries.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&tile_order_handle, tile_order.len(), 1),
                    ScalarArg::new(tile_count_x),
                    ScalarArg::new(tile_count_y),
                    ScalarArg::new(TILE_SIZE),
                    ScalarArg::new(scene.width),
                    ScalarArg::new(scene.height),
                    ScalarArg::new(num_groups),
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

            // 3) Resolve splats into the final RGBA image.
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

    /// Backward pass for differentiable rendering (CPU reference implementation).
    pub fn render_backward(
        &self,
        scene: &Scene,
        options: RenderOptions,
        backward_options: BackwardOptions,
        d_render_image: Option<&[f32]>,
        d_sdf_image: Option<&[f32]>,
    ) -> Result<SceneGrad, RenderError> {
        render_backward_gpu(
            &self.device,
            scene,
            options,
            backward_options,
            d_render_image,
            d_sdf_image,
        )
    }

    /// Backward pass for SDF values evaluated at arbitrary positions.
    pub fn eval_positions_backward(
        &self,
        scene: &Scene,
        positions: &[Vec2],
        options: RenderOptions,
        backward_options: BackwardOptions,
        d_sdf_values: Option<&[f32]>,
    ) -> Result<SceneGrad, RenderError> {
        let _ = self;
        crate::backward::render_backward_positions(
            scene,
            options,
            backward_options,
            positions,
            d_sdf_values,
        )
    }

    /// Render a signed distance field for the scene.
    pub fn render_sdf(&self, scene: &Scene, options: RenderOptions) -> Result<SdfImage, RenderError> {
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
        let values = vec![0.0f32; pixel_count];
        if scene.groups.is_empty() {
            return Ok(SdfImage {
                width: scene.width,
                height: scene.height,
                values,
            });
        }
        let prepared = prepare_scene(scene, &options)?;
        if prepared.num_groups == 0 {
            return Ok(SdfImage {
                width: scene.width,
                height: scene.height,
                values,
            });
        }

        let mut shape_transform = Vec::with_capacity(scene.shapes.len() * 6);
        for shape in &scene.shapes {
            let t = shape.transform;
            shape_transform.extend_from_slice(&[
                t.m[0][0],
                t.m[0][1],
                t.m[0][2],
                t.m[1][0],
                t.m[1][1],
                t.m[1][2],
            ]);
        }

        let client = WgpuRuntime::client(&self.device);

        let shape_data = ensure_nonempty(prepared.shape_data, 0.0);
        let segment_data = ensure_nonempty(prepared.segment_data, 0.0);
        let shape_bounds = ensure_nonempty(prepared.shape_bounds, 0.0);
        let group_data = ensure_nonempty(prepared.group_data, 0.0);
        let group_xform = ensure_nonempty(prepared.group_xform, 0.0);
        let group_shape_xform = ensure_nonempty(prepared.group_shape_xform, 0.0);
        let group_shapes = ensure_nonempty(prepared.group_shapes, 0.0);
        let shape_xform = ensure_nonempty(prepared.shape_xform, 0.0);
        let shape_transform = ensure_nonempty(shape_transform, 0.0);
        let curve_data = ensure_nonempty(prepared.curve_data, 0.0);
        let group_bvh_bounds = ensure_nonempty(prepared.group_bvh_bounds, 0.0);
        let group_bvh_nodes = ensure_nonempty_u32(prepared.group_bvh_nodes, 0);
        let group_bvh_indices = ensure_nonempty_u32(prepared.group_bvh_indices, 0);
        let group_bvh_meta = ensure_nonempty_u32(prepared.group_bvh_meta, 0);
        let path_bvh_bounds = ensure_nonempty(prepared.path_bvh_bounds, 0.0);
        let path_bvh_nodes = ensure_nonempty_u32(prepared.path_bvh_nodes, 0);
        let path_bvh_indices = ensure_nonempty_u32(prepared.path_bvh_indices, 0);
        let path_bvh_meta = ensure_nonempty_u32(prepared.path_bvh_meta, 0);

        let shape_handle = client.create_from_slice(f32::as_bytes(&shape_data));
        let segment_handle = client.create_from_slice(f32::as_bytes(&segment_data));
        let shape_bounds_handle = client.create_from_slice(f32::as_bytes(&shape_bounds));
        let group_handle = client.create_from_slice(f32::as_bytes(&group_data));
        let group_xform_handle = client.create_from_slice(f32::as_bytes(&group_xform));
        let group_shape_xform_handle = client.create_from_slice(f32::as_bytes(&group_shape_xform));
        let group_shapes_handle = client.create_from_slice(f32::as_bytes(&group_shapes));
        let shape_xform_handle = client.create_from_slice(f32::as_bytes(&shape_xform));
        let shape_transform_handle = client.create_from_slice(f32::as_bytes(&shape_transform));
        let curve_handle = client.create_from_slice(f32::as_bytes(&curve_data));
        let group_bvh_bounds_handle = client.create_from_slice(f32::as_bytes(&group_bvh_bounds));
        let group_bvh_nodes_handle = client.create_from_slice(u32::as_bytes(&group_bvh_nodes));
        let group_bvh_indices_handle = client.create_from_slice(u32::as_bytes(&group_bvh_indices));
        let group_bvh_meta_handle = client.create_from_slice(u32::as_bytes(&group_bvh_meta));
        let path_bvh_bounds_handle = client.create_from_slice(f32::as_bytes(&path_bvh_bounds));
        let path_bvh_nodes_handle = client.create_from_slice(u32::as_bytes(&path_bvh_nodes));
        let path_bvh_indices_handle = client.create_from_slice(u32::as_bytes(&path_bvh_indices));
        let path_bvh_meta_handle = client.create_from_slice(u32::as_bytes(&path_bvh_meta));

        let output_handle = client.empty(pixel_count * core::mem::size_of::<f32>());

        let samples_x = options.samples_x.max(1);
        let samples_y = options.samples_y.max(1);
        let jitter = if options.use_prefiltering {
            0u32
        } else if options.jitter {
            1u32
        } else {
            0u32
        };

        unsafe {
            let cube_dim = CubeDim::new_2d(8, 8);
            let cubes_x = div_ceil(scene.width, cube_dim.x);
            let cubes_y = div_ceil(scene.height, cube_dim.y);
            let cube_count = CubeCount::new_2d(cubes_x, cubes_y);
            gpu::render_sdf_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<f32>(&shape_handle, shape_data.len(), 1),
                ArrayArg::from_raw_parts::<f32>(&segment_handle, segment_data.len(), 1),
                ArrayArg::from_raw_parts::<f32>(&shape_bounds_handle, shape_bounds.len(), 1),
                ArrayArg::from_raw_parts::<f32>(&group_handle, group_data.len(), 1),
                ArrayArg::from_raw_parts::<f32>(&group_xform_handle, group_xform.len(), 1),
                ArrayArg::from_raw_parts::<f32>(&group_shape_xform_handle, group_shape_xform.len(), 1),
                ArrayArg::from_raw_parts::<f32>(&group_shapes_handle, group_shapes.len(), 1),
                ArrayArg::from_raw_parts::<f32>(&shape_xform_handle, shape_xform.len(), 1),
                ArrayArg::from_raw_parts::<f32>(&shape_transform_handle, shape_transform.len(), 1),
                ArrayArg::from_raw_parts::<f32>(&curve_handle, curve_data.len(), 1),
                ArrayArg::from_raw_parts::<f32>(&group_bvh_bounds_handle, group_bvh_bounds.len(), 1),
                ArrayArg::from_raw_parts::<u32>(&group_bvh_nodes_handle, group_bvh_nodes.len(), 1),
                ArrayArg::from_raw_parts::<u32>(&group_bvh_indices_handle, group_bvh_indices.len(), 1),
                ArrayArg::from_raw_parts::<u32>(&group_bvh_meta_handle, group_bvh_meta.len(), 1),
                ArrayArg::from_raw_parts::<f32>(&path_bvh_bounds_handle, path_bvh_bounds.len(), 1),
                ArrayArg::from_raw_parts::<u32>(&path_bvh_nodes_handle, path_bvh_nodes.len(), 1),
                ArrayArg::from_raw_parts::<u32>(&path_bvh_indices_handle, path_bvh_indices.len(), 1),
                ArrayArg::from_raw_parts::<u32>(&path_bvh_meta_handle, path_bvh_meta.len(), 1),
                ScalarArg::new(scene.width),
                ScalarArg::new(scene.height),
                ScalarArg::new(prepared.num_groups),
                ScalarArg::new(samples_x),
                ScalarArg::new(samples_y),
                ScalarArg::new(options.seed),
                ScalarArg::new(jitter),
                ArrayArg::from_raw_parts::<f32>(&output_handle, pixel_count, 1),
            )
            .map_err(RenderError::Launch)?;
        }

        let bytes = client.read_one(output_handle);
        let values = f32::from_bytes(&bytes).to_vec();

        Ok(SdfImage {
            width: scene.width,
            height: scene.height,
            values,
        })
    }

    /// Evaluate SDF values at arbitrary positions.
    pub fn eval_positions(
        &self,
        scene: &Scene,
        positions: &[Vec2],
        options: RenderOptions,
    ) -> Result<Vec<f32>, RenderError> {
        if positions.is_empty() {
            return Ok(Vec::new());
        }
        if scene.groups.is_empty() {
            return Ok(vec![0.0f32; positions.len()]);
        }

        let prepared = prepare_scene(scene, &options)?;
        if prepared.num_groups == 0 {
            return Ok(vec![0.0f32; positions.len()]);
        }

        let mut shape_transform = Vec::with_capacity(scene.shapes.len() * 6);
        for shape in &scene.shapes {
            let t = shape.transform;
            shape_transform.extend_from_slice(&[
                t.m[0][0],
                t.m[0][1],
                t.m[0][2],
                t.m[1][0],
                t.m[1][1],
                t.m[1][2],
            ]);
        }

        let mut positions_flat = Vec::with_capacity(positions.len() * 2);
        for pt in positions {
            positions_flat.push(pt.x);
            positions_flat.push(pt.y);
        }
        if positions.len() > (u32::MAX as usize) {
            return Err(RenderError::InvalidScene("too many eval positions for 1d launch"));
        }
        let position_count = positions.len() as u32;

        let client = WgpuRuntime::client(&self.device);

        let shape_data = ensure_nonempty(prepared.shape_data, 0.0);
        let segment_data = ensure_nonempty(prepared.segment_data, 0.0);
        let shape_bounds = ensure_nonempty(prepared.shape_bounds, 0.0);
        let group_data = ensure_nonempty(prepared.group_data, 0.0);
        let group_xform = ensure_nonempty(prepared.group_xform, 0.0);
        let group_shape_xform = ensure_nonempty(prepared.group_shape_xform, 0.0);
        let group_shapes = ensure_nonempty(prepared.group_shapes, 0.0);
        let shape_xform = ensure_nonempty(prepared.shape_xform, 0.0);
        let shape_transform = ensure_nonempty(shape_transform, 0.0);
        let curve_data = ensure_nonempty(prepared.curve_data, 0.0);
        let group_bvh_bounds = ensure_nonempty(prepared.group_bvh_bounds, 0.0);
        let group_bvh_nodes = ensure_nonempty_u32(prepared.group_bvh_nodes, 0);
        let group_bvh_indices = ensure_nonempty_u32(prepared.group_bvh_indices, 0);
        let group_bvh_meta = ensure_nonempty_u32(prepared.group_bvh_meta, 0);
        let path_bvh_bounds = ensure_nonempty(prepared.path_bvh_bounds, 0.0);
        let path_bvh_nodes = ensure_nonempty_u32(prepared.path_bvh_nodes, 0);
        let path_bvh_indices = ensure_nonempty_u32(prepared.path_bvh_indices, 0);
        let path_bvh_meta = ensure_nonempty_u32(prepared.path_bvh_meta, 0);

        let shape_handle = client.create_from_slice(f32::as_bytes(&shape_data));
        let segment_handle = client.create_from_slice(f32::as_bytes(&segment_data));
        let shape_bounds_handle = client.create_from_slice(f32::as_bytes(&shape_bounds));
        let group_handle = client.create_from_slice(f32::as_bytes(&group_data));
        let group_xform_handle = client.create_from_slice(f32::as_bytes(&group_xform));
        let group_shape_xform_handle = client.create_from_slice(f32::as_bytes(&group_shape_xform));
        let group_shapes_handle = client.create_from_slice(f32::as_bytes(&group_shapes));
        let shape_xform_handle = client.create_from_slice(f32::as_bytes(&shape_xform));
        let shape_transform_handle = client.create_from_slice(f32::as_bytes(&shape_transform));
        let curve_handle = client.create_from_slice(f32::as_bytes(&curve_data));
        let group_bvh_bounds_handle = client.create_from_slice(f32::as_bytes(&group_bvh_bounds));
        let group_bvh_nodes_handle = client.create_from_slice(u32::as_bytes(&group_bvh_nodes));
        let group_bvh_indices_handle = client.create_from_slice(u32::as_bytes(&group_bvh_indices));
        let group_bvh_meta_handle = client.create_from_slice(u32::as_bytes(&group_bvh_meta));
        let path_bvh_bounds_handle = client.create_from_slice(f32::as_bytes(&path_bvh_bounds));
        let path_bvh_nodes_handle = client.create_from_slice(u32::as_bytes(&path_bvh_nodes));
        let path_bvh_indices_handle = client.create_from_slice(u32::as_bytes(&path_bvh_indices));
        let path_bvh_meta_handle = client.create_from_slice(u32::as_bytes(&path_bvh_meta));

        let positions_handle = client.create_from_slice(f32::as_bytes(&positions_flat));
        let output_handle = client.empty(positions.len() * core::mem::size_of::<f32>());

        unsafe {
            let sample_dim = CubeDim::new_1d(256);
            let sample_count = CubeCount::new_1d(div_ceil(position_count, sample_dim.x));
            gpu::eval_positions_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                sample_count,
                sample_dim,
                ArrayArg::from_raw_parts::<f32>(&shape_handle, shape_data.len(), 1),
                ArrayArg::from_raw_parts::<f32>(&segment_handle, segment_data.len(), 1),
                ArrayArg::from_raw_parts::<f32>(&shape_bounds_handle, shape_bounds.len(), 1),
                ArrayArg::from_raw_parts::<f32>(&group_handle, group_data.len(), 1),
                ArrayArg::from_raw_parts::<f32>(&group_xform_handle, group_xform.len(), 1),
                ArrayArg::from_raw_parts::<f32>(&group_shape_xform_handle, group_shape_xform.len(), 1),
                ArrayArg::from_raw_parts::<f32>(&group_shapes_handle, group_shapes.len(), 1),
                ArrayArg::from_raw_parts::<f32>(&shape_xform_handle, shape_xform.len(), 1),
                ArrayArg::from_raw_parts::<f32>(&shape_transform_handle, shape_transform.len(), 1),
                ArrayArg::from_raw_parts::<f32>(&curve_handle, curve_data.len(), 1),
                ArrayArg::from_raw_parts::<f32>(&group_bvh_bounds_handle, group_bvh_bounds.len(), 1),
                ArrayArg::from_raw_parts::<u32>(&group_bvh_nodes_handle, group_bvh_nodes.len(), 1),
                ArrayArg::from_raw_parts::<u32>(&group_bvh_indices_handle, group_bvh_indices.len(), 1),
                ArrayArg::from_raw_parts::<u32>(&group_bvh_meta_handle, group_bvh_meta.len(), 1),
                ArrayArg::from_raw_parts::<f32>(&path_bvh_bounds_handle, path_bvh_bounds.len(), 1),
                ArrayArg::from_raw_parts::<u32>(&path_bvh_nodes_handle, path_bvh_nodes.len(), 1),
                ArrayArg::from_raw_parts::<u32>(&path_bvh_indices_handle, path_bvh_indices.len(), 1),
                ArrayArg::from_raw_parts::<u32>(&path_bvh_meta_handle, path_bvh_meta.len(), 1),
                ArrayArg::from_raw_parts::<f32>(&positions_handle, positions_flat.len(), 1),
                ScalarArg::new(position_count),
                ScalarArg::new(prepared.num_groups),
                ArrayArg::from_raw_parts::<f32>(&output_handle, positions.len(), 1),
            )
            .map_err(RenderError::Launch)?;
        }

        let bytes = client.read_one(output_handle);
        Ok(f32::from_bytes(&bytes).to_vec())
    }
}
