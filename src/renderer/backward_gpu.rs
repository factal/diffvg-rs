//! GPU-backed backward pass for differentiable rendering.

use crate::backward::background::finalize_background_gradients;
use crate::backward::boundary::boundary_sampling;
use crate::backward::BackwardOptions;
use crate::grad::{DGradientStop, DLinearGradient, DPaint, DRadialGradient, SceneGrad};
use crate::math::{Mat3, Vec2};
use crate::scene::{Paint, Scene, ShapeGeometry};
use crate::gpu;
use crate::renderer::constants::{GRADIENT_STRIDE, GROUP_STRIDE};
use crate::renderer::prepare_backward::prepare_scene_backward;
use crate::renderer::types::{RenderError, RenderOptions};
use crate::renderer::utils::{ensure_nonempty, ensure_nonempty_u32};
use cubecl::features::TypeUsage;
use cubecl::ir::{ElemType, FloatKind, StorageType};
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

pub(crate) fn render_backward_gpu(
    device: &WgpuDevice,
    scene: &Scene,
    options: RenderOptions,
    backward_options: BackwardOptions,
    d_render_image: Option<&[f32]>,
    d_sdf_image: Option<&[f32]>,
) -> Result<SceneGrad, RenderError> {
    let width = scene.width as usize;
    let height = scene.height as usize;
    let pixel_count = width.saturating_mul(height);

    if let Some(d_render) = d_render_image {
        if d_render.len() != pixel_count.saturating_mul(4) {
            return Err(RenderError::InvalidScene("d_render_image size mismatch"));
        }
    }
    if let Some(d_sdf) = d_sdf_image {
        if d_sdf.len() != pixel_count {
            return Err(RenderError::InvalidScene("d_sdf_image size mismatch"));
        }
    }

    let include_background_image = d_render_image.is_some() && scene.background_image.is_some();
    let mut grads = SceneGrad::zeros_from_scene(
        scene,
        include_background_image,
        backward_options.compute_translation,
    );

    if scene.width == 0 || scene.height == 0 {
        return Ok(grads);
    }

    if d_render_image.is_none() && d_sdf_image.is_none() {
        return Ok(grads);
    }

    if scene.groups.is_empty() {
        return Ok(grads);
    }

    let prepared = prepare_scene_backward(scene, &options)?;
    if prepared.prepared.num_groups == 0 {
        return Ok(grads);
    }

    let client = WgpuRuntime::client(device);
    let use_float_atomics = client
        .properties()
        .type_usage(StorageType::Atomic(ElemType::Float(FloatKind::F32)))
        .contains(TypeUsage::AtomicAdd);
    if !use_float_atomics {
        return crate::backward::render_backward(
            scene,
            options,
            backward_options,
            d_render_image,
            d_sdf_image,
        );
    }

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

    let PreparedSceneBuffers {
        shape_data,
        segment_data,
        shape_bounds,
        group_data,
        group_xform,
        group_shape_xform,
        group_shapes,
        shape_xform,
        curve_data,
        gradient_data,
        stop_offsets,
        stop_colors,
        group_bvh_bounds,
        group_bvh_nodes,
        group_bvh_indices,
        group_bvh_meta,
        path_bvh_bounds,
        path_bvh_nodes,
        path_bvh_indices,
        path_bvh_meta,
        shape_transform,
        path_points,
        path_num_control_points,
        path_thickness,
        shape_path_offsets,
        shape_path_point_counts,
        shape_path_ctrl_offsets,
        shape_path_ctrl_counts: _shape_path_ctrl_counts,
        shape_path_thickness_offsets: _shape_path_thickness_offsets,
        shape_path_thickness_counts: _shape_path_thickness_counts,
        shape_path_is_closed,
    } = PreparedSceneBuffers::from_prepared(&prepared);

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

    let shape_handle = client.create_from_slice(f32::as_bytes(&shape_data));
    let segment_handle = client.create_from_slice(f32::as_bytes(&segment_data));
    let shape_bounds_handle = client.create_from_slice(f32::as_bytes(&shape_bounds));
    let group_handle = client.create_from_slice(f32::as_bytes(&group_data));
    let group_xform_handle = client.create_from_slice(f32::as_bytes(&group_xform));
    let group_shape_xform_handle = client.create_from_slice(f32::as_bytes(&group_shape_xform));
    let group_shapes_handle = client.create_from_slice(f32::as_bytes(&group_shapes));
    let shape_xform_handle = client.create_from_slice(f32::as_bytes(&shape_xform));
    let curve_handle = client.create_from_slice(f32::as_bytes(&curve_data));
    let gradient_handle = client.create_from_slice(f32::as_bytes(&gradient_data));
    let stop_offsets_handle = client.create_from_slice(f32::as_bytes(&stop_offsets));
    let stop_colors_handle = client.create_from_slice(f32::as_bytes(&stop_colors));
    let group_bvh_bounds_handle = client.create_from_slice(f32::as_bytes(&group_bvh_bounds));
    let group_bvh_nodes_handle = client.create_from_slice(u32::as_bytes(&group_bvh_nodes));
    let group_bvh_indices_handle = client.create_from_slice(u32::as_bytes(&group_bvh_indices));
    let group_bvh_meta_handle = client.create_from_slice(u32::as_bytes(&group_bvh_meta));
    let path_bvh_bounds_handle = client.create_from_slice(f32::as_bytes(&path_bvh_bounds));
    let path_bvh_nodes_handle = client.create_from_slice(u32::as_bytes(&path_bvh_nodes));
    let path_bvh_indices_handle = client.create_from_slice(u32::as_bytes(&path_bvh_indices));
    let path_bvh_meta_handle = client.create_from_slice(u32::as_bytes(&path_bvh_meta));
    let shape_transform_handle = client.create_from_slice(f32::as_bytes(&shape_transform));
    let path_points_handle = client.create_from_slice(f32::as_bytes(&path_points));
    let path_controls_handle = client.create_from_slice(u32::as_bytes(&path_num_control_points));
    let shape_path_offsets_handle = client.create_from_slice(u32::as_bytes(&shape_path_offsets));
    let shape_path_point_counts_handle = client.create_from_slice(u32::as_bytes(&shape_path_point_counts));
    let shape_path_ctrl_offsets_handle = client.create_from_slice(u32::as_bytes(&shape_path_ctrl_offsets));
    let shape_path_is_closed_handle = client.create_from_slice(u32::as_bytes(&shape_path_is_closed));

    let background_handle = client.create_from_slice(f32::as_bytes(&background_image));

    let render_grad_flag = if d_render_image.is_some() { 1u32 } else { 0u32 };
    let translation_flag = if backward_options.compute_translation { 1u32 } else { 0u32 };

    let d_render_image_handle = match d_render_image {
        Some(values) => client.create_from_slice(f32::as_bytes(values)),
        None => client.create_from_slice(f32::as_bytes(&[0.0f32])),
    };
    let d_render_len = d_render_image.map(|values| values.len()).unwrap_or(1);

    let weight_len = pixel_count.max(1);
    let weight_init = vec![0.0f32; weight_len];
    let weight_handle = client.create_from_slice(f32::as_bytes(&weight_init));

    let d_shape_params = vec![0.0f32; scene.shapes.len() * 8];
    let d_shape_points = vec![0.0f32; path_points.len().max(1)];
    let d_shape_thickness = vec![0.0f32; path_thickness.len().max(1)];
    let d_shape_stroke_width = vec![0.0f32; scene.shapes.len().max(1)];
    let d_shape_transform = vec![0.0f32; scene.shapes.len() * 9];
    let d_group_transform = vec![0.0f32; scene.groups.len() * 9];
    let d_group_data = vec![0.0f32; group_data.len()];
    let d_gradient_data = vec![0.0f32; gradient_data.len().max(1)];
    let d_stop_offsets = vec![0.0f32; stop_offsets.len().max(1)];
    let d_stop_colors = vec![0.0f32; stop_colors.len().max(1)];
    let d_filter_radius = vec![0.0f32; 1];
    let d_background = vec![0.0f32; 4];
    let d_background_image = vec![0.0f32; background_image.len().max(1)];
    let d_translation = vec![0.0f32; pixel_count.saturating_mul(2).max(1)];

    let d_shape_params_handle = client.create_from_slice(f32::as_bytes(&d_shape_params));
    let d_shape_points_handle = client.create_from_slice(f32::as_bytes(&d_shape_points));
    let d_shape_thickness_handle = client.create_from_slice(f32::as_bytes(&d_shape_thickness));
    let d_shape_stroke_width_handle = client.create_from_slice(f32::as_bytes(&d_shape_stroke_width));
    let d_shape_transform_handle = client.create_from_slice(f32::as_bytes(&d_shape_transform));
    let d_group_transform_handle = client.create_from_slice(f32::as_bytes(&d_group_transform));
    let d_group_data_handle = client.create_from_slice(f32::as_bytes(&d_group_data));
    let d_gradient_data_handle = client.create_from_slice(f32::as_bytes(&d_gradient_data));
    let d_stop_offsets_handle = client.create_from_slice(f32::as_bytes(&d_stop_offsets));
    let d_stop_colors_handle = client.create_from_slice(f32::as_bytes(&d_stop_colors));
    let d_filter_radius_handle = client.create_from_slice(f32::as_bytes(&d_filter_radius));
    let d_background_handle = client.create_from_slice(f32::as_bytes(&d_background));
    let d_background_image_handle = client.create_from_slice(f32::as_bytes(&d_background_image));
    let d_translation_handle = client.create_from_slice(f32::as_bytes(&d_translation));

    unsafe {
        if render_grad_flag != 0 {
            let sample_dim = CubeDim::new_1d(256);
            let weight_count = CubeCount::new_1d(crate::renderer::tiles::div_ceil(total_samples, sample_dim.x));
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
        }

        let sample_dim = CubeDim::new_1d(256);
        let sample_count = CubeCount::new_1d(crate::renderer::tiles::div_ceil(total_samples, sample_dim.x));
        if render_grad_flag != 0 {
            if use_prefiltering {
                gpu::render_backward_color_prefilter_kernel::launch_unchecked::<WgpuRuntime>(
                    &client,
                    sample_count.clone(),
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
                    ArrayArg::from_raw_parts::<f32>(&path_points_handle, path_points.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&path_controls_handle, path_num_control_points.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&shape_path_offsets_handle, shape_path_offsets.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&shape_path_point_counts_handle, shape_path_point_counts.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&shape_path_ctrl_offsets_handle, shape_path_ctrl_offsets.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&shape_path_is_closed_handle, shape_path_is_closed.len(), 1),
                    ScalarArg::new(scene.width),
                    ScalarArg::new(scene.height),
                    ScalarArg::new(prepared.prepared.num_groups),
                    ScalarArg::new(samples_x),
                    ScalarArg::new(samples_y),
                    ScalarArg::new(options.seed),
                    ScalarArg::new(jitter),
                    ScalarArg::new(scene.filter.filter_type.as_u32()),
                    ScalarArg::new(scene.filter.radius),
                    ArrayArg::from_raw_parts::<f32>(&background_handle, background_image.len(), 1),
                    ScalarArg::new(has_background_image),
                    ScalarArg::new(scene.background.r),
                    ScalarArg::new(scene.background.g),
                    ScalarArg::new(scene.background.b),
                    ScalarArg::new(scene.background.a),
                    ArrayArg::from_raw_parts::<f32>(&weight_handle, weight_len, 1),
                    ArrayArg::from_raw_parts::<f32>(&d_render_image_handle, d_render_len, 1),
                    ScalarArg::new(translation_flag),
                    ArrayArg::from_raw_parts::<f32>(&d_shape_params_handle, d_shape_params.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&d_shape_points_handle, d_shape_points.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&d_shape_stroke_width_handle, d_shape_stroke_width.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&d_shape_transform_handle, d_shape_transform.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&d_group_transform_handle, d_group_transform.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&d_group_data_handle, d_group_data.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&d_gradient_data_handle, d_gradient_data.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&d_stop_offsets_handle, d_stop_offsets.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&d_stop_colors_handle, d_stop_colors.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&d_filter_radius_handle, d_filter_radius.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&d_background_handle, d_background.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&d_background_image_handle, d_background_image.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&d_translation_handle, d_translation.len(), 1),
                )
                .map_err(RenderError::Launch)?;
            } else {
                gpu::render_backward_color_kernel::launch_unchecked::<WgpuRuntime>(
                    &client,
                    sample_count.clone(),
                    sample_dim,
                    ArrayArg::from_raw_parts::<f32>(&shape_handle, shape_data.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&segment_handle, segment_data.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&shape_bounds_handle, shape_bounds.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&group_handle, group_data.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&group_xform_handle, group_xform.len(), 1),
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
                    ScalarArg::new(scene.width),
                    ScalarArg::new(scene.height),
                    ScalarArg::new(prepared.prepared.num_groups),
                    ScalarArg::new(samples_x),
                    ScalarArg::new(samples_y),
                    ScalarArg::new(options.seed),
                    ScalarArg::new(jitter),
                    ScalarArg::new(scene.filter.filter_type.as_u32()),
                    ScalarArg::new(scene.filter.radius),
                    ArrayArg::from_raw_parts::<f32>(&background_handle, background_image.len(), 1),
                    ScalarArg::new(has_background_image),
                    ScalarArg::new(scene.background.r),
                    ScalarArg::new(scene.background.g),
                    ScalarArg::new(scene.background.b),
                    ScalarArg::new(scene.background.a),
                    ArrayArg::from_raw_parts::<f32>(&weight_handle, weight_len, 1),
                    ArrayArg::from_raw_parts::<f32>(&d_render_image_handle, d_render_len, 1),
                    ScalarArg::new(translation_flag),
                    ArrayArg::from_raw_parts::<f32>(&d_group_data_handle, d_group_data.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&d_gradient_data_handle, d_gradient_data.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&d_stop_offsets_handle, d_stop_offsets.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&d_stop_colors_handle, d_stop_colors.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&d_filter_radius_handle, d_filter_radius.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&d_background_handle, d_background.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&d_background_image_handle, d_background_image.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&d_translation_handle, d_translation.len(), 1),
                )
                .map_err(RenderError::Launch)?;
            }
        }

    }

    let d_shape_params = f32::from_bytes(&client.read_one(d_shape_params_handle)).to_vec();
    let d_shape_points = f32::from_bytes(&client.read_one(d_shape_points_handle)).to_vec();
    let d_shape_thickness = f32::from_bytes(&client.read_one(d_shape_thickness_handle)).to_vec();
    let d_shape_stroke_width =
        f32::from_bytes(&client.read_one(d_shape_stroke_width_handle)).to_vec();
    let d_shape_transform = f32::from_bytes(&client.read_one(d_shape_transform_handle)).to_vec();
    let d_group_transform = f32::from_bytes(&client.read_one(d_group_transform_handle)).to_vec();
    let d_group_data = f32::from_bytes(&client.read_one(d_group_data_handle)).to_vec();
    let d_gradient_data = f32::from_bytes(&client.read_one(d_gradient_data_handle)).to_vec();
    let d_stop_offsets = f32::from_bytes(&client.read_one(d_stop_offsets_handle)).to_vec();
    let d_stop_colors = f32::from_bytes(&client.read_one(d_stop_colors_handle)).to_vec();
    let d_filter_radius = f32::from_bytes(&client.read_one(d_filter_radius_handle)).to_vec();
    let d_background = f32::from_bytes(&client.read_one(d_background_handle)).to_vec();
    let d_background_image = f32::from_bytes(&client.read_one(d_background_image_handle)).to_vec();
    let d_translation = f32::from_bytes(&client.read_one(d_translation_handle)).to_vec();

    apply_gpu_grads(
        scene,
        &prepared,
        &d_shape_params,
        &d_shape_points,
        &d_shape_thickness,
        &d_shape_stroke_width,
        &d_shape_transform,
        &d_group_transform,
        &d_group_data,
        &d_gradient_data,
        &d_stop_offsets,
        &d_stop_colors,
        &d_filter_radius,
        &d_background,
        &d_background_image,
        &d_translation,
        &mut grads,
    );

    if !use_prefiltering && d_render_image.is_some() {
        let weight_bytes = client.read_one(weight_handle);
        let weight_image = f32::from_bytes(&weight_bytes).to_vec();
        let bvh = crate::distance::SceneBvh::new(scene);
        boundary_sampling(
            scene,
            &bvh,
            samples_x,
            samples_y,
            options.seed,
            d_render_image.unwrap(),
            &weight_image,
            &mut grads,
            backward_options.compute_translation,
        );
    }

    finalize_background_gradients(scene, &mut grads);

    if let Some(d_sdf) = d_sdf_image {
        let sdf_grads = crate::backward::render_backward(
            scene,
            options,
            backward_options,
            None,
            Some(d_sdf),
        )?;
        grads.accumulate_from(&sdf_grads);
    }
    Ok(grads)
}

struct PreparedSceneBuffers {
    shape_data: Vec<f32>,
    segment_data: Vec<f32>,
    shape_bounds: Vec<f32>,
    group_data: Vec<f32>,
    group_xform: Vec<f32>,
    group_shape_xform: Vec<f32>,
    group_shapes: Vec<f32>,
    shape_xform: Vec<f32>,
    curve_data: Vec<f32>,
    gradient_data: Vec<f32>,
    stop_offsets: Vec<f32>,
    stop_colors: Vec<f32>,
    group_bvh_bounds: Vec<f32>,
    group_bvh_nodes: Vec<u32>,
    group_bvh_indices: Vec<u32>,
    group_bvh_meta: Vec<u32>,
    path_bvh_bounds: Vec<f32>,
    path_bvh_nodes: Vec<u32>,
    path_bvh_indices: Vec<u32>,
    path_bvh_meta: Vec<u32>,
    shape_transform: Vec<f32>,
    path_points: Vec<f32>,
    path_num_control_points: Vec<u32>,
    path_thickness: Vec<f32>,
    shape_path_offsets: Vec<u32>,
    shape_path_point_counts: Vec<u32>,
    shape_path_ctrl_offsets: Vec<u32>,
    shape_path_ctrl_counts: Vec<u32>,
    shape_path_thickness_offsets: Vec<u32>,
    shape_path_thickness_counts: Vec<u32>,
    shape_path_is_closed: Vec<u32>,
}

impl PreparedSceneBuffers {
    fn from_prepared(prepared: &crate::renderer::prepare_backward::PreparedBackwardScene) -> Self {
        let base = &prepared.prepared;
        Self {
            shape_data: ensure_nonempty(base.shape_data.clone(), 0.0),
            segment_data: ensure_nonempty(base.segment_data.clone(), 0.0),
            shape_bounds: ensure_nonempty(base.shape_bounds.clone(), 0.0),
            group_data: ensure_nonempty(base.group_data.clone(), 0.0),
            group_xform: ensure_nonempty(base.group_xform.clone(), 0.0),
            group_shape_xform: ensure_nonempty(base.group_shape_xform.clone(), 0.0),
            group_shapes: ensure_nonempty(base.group_shapes.clone(), 0.0),
            shape_xform: ensure_nonempty(base.shape_xform.clone(), 0.0),
            curve_data: ensure_nonempty(base.curve_data.clone(), 0.0),
            gradient_data: ensure_nonempty(base.gradient_data.clone(), 0.0),
            stop_offsets: ensure_nonempty(base.stop_offsets.clone(), 0.0),
            stop_colors: ensure_nonempty(base.stop_colors.clone(), 0.0),
            group_bvh_bounds: ensure_nonempty(base.group_bvh_bounds.clone(), 0.0),
            group_bvh_nodes: ensure_nonempty_u32(base.group_bvh_nodes.clone(), 0),
            group_bvh_indices: ensure_nonempty_u32(base.group_bvh_indices.clone(), 0),
            group_bvh_meta: ensure_nonempty_u32(base.group_bvh_meta.clone(), 0),
            path_bvh_bounds: ensure_nonempty(base.path_bvh_bounds.clone(), 0.0),
            path_bvh_nodes: ensure_nonempty_u32(base.path_bvh_nodes.clone(), 0),
            path_bvh_indices: ensure_nonempty_u32(base.path_bvh_indices.clone(), 0),
            path_bvh_meta: ensure_nonempty_u32(base.path_bvh_meta.clone(), 0),
            shape_transform: ensure_nonempty(prepared.shape_transform.clone(), 0.0),
            path_points: ensure_nonempty(prepared.path_points.clone(), 0.0),
            path_num_control_points: ensure_nonempty_u32(prepared.path_num_control_points.clone(), 0),
            path_thickness: ensure_nonempty(prepared.path_thickness.clone(), 0.0),
            shape_path_offsets: ensure_nonempty_u32(prepared.shape_path_offsets.clone(), 0),
            shape_path_point_counts: ensure_nonempty_u32(prepared.shape_path_point_counts.clone(), 0),
            shape_path_ctrl_offsets: ensure_nonempty_u32(prepared.shape_path_ctrl_offsets.clone(), 0),
            shape_path_ctrl_counts: ensure_nonempty_u32(prepared.shape_path_ctrl_counts.clone(), 0),
            shape_path_thickness_offsets: ensure_nonempty_u32(prepared.shape_path_thickness_offsets.clone(), 0),
            shape_path_thickness_counts: ensure_nonempty_u32(prepared.shape_path_thickness_counts.clone(), 0),
            shape_path_is_closed: ensure_nonempty_u32(prepared.shape_path_is_closed.clone(), 0),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_gpu_grads(
    scene: &Scene,
    prepared: &crate::renderer::prepare_backward::PreparedBackwardScene,
    d_shape_params: &[f32],
    d_shape_points: &[f32],
    d_shape_thickness: &[f32],
    d_shape_stroke_width: &[f32],
    d_shape_transform: &[f32],
    d_group_transform: &[f32],
    d_group_data: &[f32],
    d_gradient_data: &[f32],
    d_stop_offsets: &[f32],
    d_stop_colors: &[f32],
    d_filter_radius: &[f32],
    d_background: &[f32],
    d_background_image: &[f32],
    d_translation: &[f32],
    grads: &mut SceneGrad,
) {
    for (shape_index, shape) in scene.shapes.iter().enumerate() {
        let d_shape = &mut grads.shapes[shape_index];
        if let Some(value) = d_shape_stroke_width.get(shape_index) {
            d_shape.stroke_width = *value;
        }

        let t_base = shape_index * 9;
        if t_base + 8 < d_shape_transform.len() {
            d_shape.transform = Mat3 {
                m: [
                    [
                        d_shape_transform[t_base],
                        d_shape_transform[t_base + 1],
                        d_shape_transform[t_base + 2],
                    ],
                    [
                        d_shape_transform[t_base + 3],
                        d_shape_transform[t_base + 4],
                        d_shape_transform[t_base + 5],
                    ],
                    [
                        d_shape_transform[t_base + 6],
                        d_shape_transform[t_base + 7],
                        d_shape_transform[t_base + 8],
                    ],
                ],
            };
        }

        match &shape.geometry {
            ShapeGeometry::Circle { .. } => {
                let base = shape_index * 8;
                if base + 2 < d_shape_params.len() {
                    if let crate::grad::DShapeGeometry::Circle { center, radius } =
                        &mut d_shape.geometry
                    {
                        *center = Vec2::new(d_shape_params[base], d_shape_params[base + 1]);
                        *radius = d_shape_params[base + 2];
                    }
                }
            }
            ShapeGeometry::Ellipse { .. } => {
                let base = shape_index * 8;
                if base + 3 < d_shape_params.len() {
                    if let crate::grad::DShapeGeometry::Ellipse { center, radius } =
                        &mut d_shape.geometry
                    {
                        *center = Vec2::new(d_shape_params[base], d_shape_params[base + 1]);
                        *radius = Vec2::new(d_shape_params[base + 2], d_shape_params[base + 3]);
                    }
                }
            }
            ShapeGeometry::Rect { .. } => {
                let base = shape_index * 8;
                if base + 3 < d_shape_params.len() {
                    if let crate::grad::DShapeGeometry::Rect { min, max } = &mut d_shape.geometry {
                        *min = Vec2::new(d_shape_params[base], d_shape_params[base + 1]);
                        *max = Vec2::new(d_shape_params[base + 2], d_shape_params[base + 3]);
                    }
                }
            }
            ShapeGeometry::Path { .. } => {
                let point_offset = prepared.shape_path_offsets.get(shape_index).copied().unwrap_or(0);
                let point_count = prepared.shape_path_point_counts.get(shape_index).copied().unwrap_or(0);
                let thickness_offset = prepared
                    .shape_path_thickness_offsets
                    .get(shape_index)
                    .copied()
                    .unwrap_or(0);
                let thickness_count = prepared
                    .shape_path_thickness_counts
                    .get(shape_index)
                    .copied()
                    .unwrap_or(0);

                if let crate::grad::DShapeGeometry::Path { points, thickness } =
                    &mut d_shape.geometry
                {
                    for i in 0..(point_count as usize) {
                        let base = (point_offset as usize + i) * 2;
                        if base + 1 < d_shape_points.len() && i < points.len() {
                            points[i] = Vec2::new(d_shape_points[base], d_shape_points[base + 1]);
                        }
                    }
                    if let Some(thickness) = thickness.as_mut() {
                        for i in 0..(thickness_count as usize) {
                            let idx = thickness_offset as usize + i;
                            if idx < d_shape_thickness.len() && i < thickness.len() {
                                thickness[i] = d_shape_thickness[idx];
                            }
                        }
                    }
                }
            }
        }
    }

    for (group_index, group) in scene.groups.iter().enumerate() {
        let d_group = &mut grads.shape_groups[group_index];
        let base = group_index * 9;
        if base + 8 < d_group_transform.len() {
            d_group.shape_to_canvas = Mat3 {
                m: [
                    [
                        d_group_transform[base],
                        d_group_transform[base + 1],
                        d_group_transform[base + 2],
                    ],
                    [
                        d_group_transform[base + 3],
                        d_group_transform[base + 4],
                        d_group_transform[base + 5],
                    ],
                    [
                        d_group_transform[base + 6],
                        d_group_transform[base + 7],
                        d_group_transform[base + 8],
                    ],
                ],
            };
        }

        let group_base = group_index * GROUP_STRIDE;
        let fill_kind = prepared.prepared.group_data[group_base + 2] as u32;
        let stroke_kind = prepared.prepared.group_data[group_base + 4] as u32;
        let fill_grad = prepared.prepared.group_data[group_base + 3] as u32;
        let stroke_grad = prepared.prepared.group_data[group_base + 5] as u32;

        if let Some(fill) = group.fill.as_ref() {
            d_group.fill = Some(build_paint_grad(
                fill,
                fill_kind,
                fill_grad,
                group_base,
                d_group_data,
                &prepared.prepared.gradient_data,
                d_gradient_data,
                d_stop_offsets,
                d_stop_colors,
            ));
        }

        if let Some(stroke) = group.stroke.as_ref() {
            d_group.stroke = Some(build_paint_grad(
                stroke,
                stroke_kind,
                stroke_grad,
                group_base + 4,
                d_group_data,
                &prepared.prepared.gradient_data,
                d_gradient_data,
                d_stop_offsets,
                d_stop_colors,
            ));
        }
    }

    if let Some(value) = d_filter_radius.get(0) {
        grads.filter.radius = *value;
    }

    if d_background.len() >= 4 {
        grads.background = crate::color::Color {
            r: d_background[0],
            g: d_background[1],
            b: d_background[2],
            a: d_background[3],
        };
    }

    if let Some(bg) = grads.background_image.as_mut() {
        let len = bg.len();
        if d_background_image.len() >= len {
            bg.copy_from_slice(&d_background_image[..len]);
        }
    }

    if let Some(trans) = grads.translation.as_mut() {
        let len = trans.len();
        if d_translation.len() >= len {
            trans.copy_from_slice(&d_translation[..len]);
        }
    }
}

fn build_paint_grad(
    paint: &Paint,
    kind: u32,
    grad_index: u32,
    group_base: usize,
    d_group_data: &[f32],
    gradient_data: &[f32],
    d_gradient_data: &[f32],
    d_stop_offsets: &[f32],
    d_stop_colors: &[f32],
) -> DPaint {
    match paint {
        Paint::Solid(_) => {
            let base = group_base + 8;
            let r = d_group_data.get(base).copied().unwrap_or(0.0);
            let g = d_group_data.get(base + 1).copied().unwrap_or(0.0);
            let b = d_group_data.get(base + 2).copied().unwrap_or(0.0);
            let a = d_group_data.get(base + 3).copied().unwrap_or(0.0);
            DPaint::Solid(crate::color::Color { r, g, b, a })
        }
        Paint::LinearGradient(_) | Paint::RadialGradient(_) => {
            if kind == 0 {
                return DPaint::Solid(crate::color::Color::TRANSPARENT);
            }
            let base = grad_index as usize * GRADIENT_STRIDE;
            let grad_type = gradient_data.get(base).copied().unwrap_or(0.0) as u32;
            let p0 = d_gradient_data.get(base + 1).copied().unwrap_or(0.0);
            let p1 = d_gradient_data.get(base + 2).copied().unwrap_or(0.0);
            let p2 = d_gradient_data.get(base + 3).copied().unwrap_or(0.0);
            let p3 = d_gradient_data.get(base + 4).copied().unwrap_or(0.0);
            let stop_offset = gradient_data.get(base + 5).copied().unwrap_or(0.0) as usize;
            let stop_count = gradient_data.get(base + 6).copied().unwrap_or(0.0) as usize;
            let mut stops = Vec::with_capacity(stop_count);
            for i in 0..stop_count {
                let o_idx = stop_offset + i;
                let c_idx = o_idx * 4;
                let offset = d_stop_offsets.get(o_idx).copied().unwrap_or(0.0);
                let r = d_stop_colors.get(c_idx).copied().unwrap_or(0.0);
                let g = d_stop_colors.get(c_idx + 1).copied().unwrap_or(0.0);
                let b = d_stop_colors.get(c_idx + 2).copied().unwrap_or(0.0);
                let a = d_stop_colors.get(c_idx + 3).copied().unwrap_or(0.0);
                stops.push(DGradientStop {
                    offset,
                    color: crate::color::Color { r, g, b, a },
                });
            }
            if grad_type == 0 {
                DPaint::LinearGradient(DLinearGradient {
                    start: Vec2::new(p0, p1),
                    end: Vec2::new(p2, p3),
                    stops,
                })
            } else {
                DPaint::RadialGradient(DRadialGradient {
                    center: Vec2::new(p0, p1),
                    radius: Vec2::new(p2, p3),
                    stops,
                })
            }
        }
    }
}
