use crate::color::Color;
use crate::distance::{compute_distance_bvh, within_distance_bvh, DistanceOptions, SceneBvh};
use crate::grad::SceneGrad;
use crate::math::{Vec2, Vec4};
use crate::scene::FillRule;

use super::background::accumulate_background_grad;
use super::distance::{d_compute_distance, path_info_from_closest};
use super::math::{d_smoothstep, smoothstep, transform_point_inverse};
use super::paint::{d_sample_paint, paint_color};
use super::types::{EdgeQuery, Fragment, PathInfo, PrefilterFragment, Rgb};

pub(super) fn sample_distance(
    scene: &crate::scene::Scene,
    bvh: &SceneBvh,
    pt: Vec2,
    weight: f32,
    d_dist: f32,
    grads: &mut SceneGrad,
    translation_index: Option<usize>,
    options: DistanceOptions,
) {
    let mut best_dist = f32::INFINITY;
    let mut best_group = None;
    let mut best_shape = None;
    let mut best_point = Vec2::ZERO;
    let mut best_path: Option<PathInfo> = None;

    for group_id in (0..scene.groups.len()).rev() {
        if let Some(hit) = compute_distance_bvh(scene, bvh, group_id, pt, best_dist, options) {
            if hit.distance < best_dist {
                best_dist = hit.distance;
                best_group = Some(group_id);
                best_shape = Some(hit.shape_index);
                best_point = hit.point;
                best_path = path_info_from_closest(scene, hit.shape_index, hit.path);
            }
        }
    }

    let Some(group_id) = best_group else {
        return;
    };
    let shape_id = best_shape.unwrap_or(0);
    let mut signed_dist = best_dist * weight;
    let inside = scene.groups[group_id].fill.is_some() && is_inside_bvh(scene, bvh, group_id, pt);
    if inside {
        signed_dist = -signed_dist;
    }

    let d_abs_dist = if inside { -d_dist } else { d_dist };
    let d_pt = {
        let d_shape = &mut grads.shapes[shape_id];
        let d_group = &mut grads.shape_groups[group_id];
        d_compute_distance(
            &scene.groups[group_id],
            &scene.shapes[shape_id],
            pt,
            best_point,
            best_path,
            d_abs_dist,
            d_shape,
            d_group,
        )
    };
    accumulate_translation(grads, translation_index, d_pt * -1.0);
}

fn is_inside_bvh(scene: &crate::scene::Scene, bvh: &SceneBvh, group_id: usize, pt: Vec2) -> bool {
    crate::distance::is_inside_bvh(scene, bvh, group_id, pt)
}

pub(super) fn sample_color(
    scene: &crate::scene::Scene,
    bvh: &SceneBvh,
    screen_pt: Vec2,
    background: Option<Vec4>,
    d_color: Option<Vec4>,
    mut edge_query: Option<&mut EdgeQuery>,
    grads: &mut SceneGrad,
    translation_index: Option<usize>,
    pixel_index: usize,
) -> Vec4 {
    if let Some(eq) = edge_query.as_deref_mut() {
        eq.hit = false;
    }

    let pt = Vec2::new(screen_pt.x * scene.width as f32, screen_pt.y * scene.height as f32);
    let mut fragments: Vec<Fragment> = Vec::new();
    for group_id in 0..scene.groups.len() {
        let group = &scene.groups[group_id];
        if group.stroke.is_some() {
            let hit = within_distance_edge(scene, bvh, group_id, pt, edge_query.as_deref_mut());
            if hit {
                let color = paint_color(group.stroke.as_ref().unwrap(), pt);
                fragments.push(Fragment {
                    color: Rgb::new(color.x, color.y, color.z),
                    alpha: color.w,
                    group_id,
                    is_stroke: true,
                });
            }
        }
        if group.fill.is_some() {
            let hit = is_inside_edge(scene, bvh, group_id, pt, edge_query.as_deref_mut());
            if hit {
                let color = paint_color(group.fill.as_ref().unwrap(), pt);
                fragments.push(Fragment {
                    color: Rgb::new(color.x, color.y, color.z),
                    alpha: color.w,
                    group_id,
                    is_stroke: false,
                });
            }
        }
    }

    if fragments.is_empty() {
        if let Some(bg) = background {
            if let Some(d) = d_color {
                accumulate_background_grad(scene, grads, pixel_index, d);
            }
            return bg;
        }
        return Vec4::ZERO;
    }

    fragments.sort_by_key(|f| f.group_id);

    let mut accum_color: Vec<Rgb> = Vec::with_capacity(fragments.len());
    let mut accum_alpha: Vec<f32> = Vec::with_capacity(fragments.len());
    let mut prev_color = Rgb::new(0.0, 0.0, 0.0);
    let mut prev_alpha = 0.0f32;
    if let Some(bg) = background {
        prev_color = Rgb::new(bg.x, bg.y, bg.z);
        prev_alpha = bg.w;
    }

    for fragment in &fragments {
        let new_color = fragment.color;
        let new_alpha = fragment.alpha;
        if let Some(eq) = edge_query.as_deref_mut() {
            if new_alpha >= 1.0 && eq.hit {
                eq.hit = false;
            }
            if eq.shape_group_id == fragment.group_id {
                eq.hit = true;
            }
        }
        let blended = prev_color.scale(1.0 - new_alpha).add(new_color.scale(new_alpha));
        let blended_alpha = prev_alpha * (1.0 - new_alpha) + new_alpha;
        accum_color.push(blended);
        accum_alpha.push(blended_alpha);
        prev_color = blended;
        prev_alpha = blended_alpha;
    }

    let mut final_color = accum_color[accum_color.len() - 1];
    let final_alpha = accum_alpha[accum_alpha.len() - 1];
    if final_alpha > 1.0e-6 {
        final_color = final_color.scale(1.0 / final_alpha);
    }

    if let Some(d) = d_color {
        let mut d_curr_color = Rgb::new(d.x, d.y, d.z);
        let mut d_curr_alpha = d.w;
        if final_alpha > 1.0e-6 {
            d_curr_color = d_curr_color.scale(1.0 / final_alpha);
            d_curr_alpha -= d_curr_color.dot(final_color) / final_alpha;
        }
        for i in (0..fragments.len()).rev() {
            let prev_alpha = if i > 0 {
                accum_alpha[i - 1]
            } else {
                background.map(|b| b.w).unwrap_or(0.0)
            };
            let prev_color = if i > 0 {
                accum_color[i - 1]
            } else {
                background
                    .map(|b| Rgb::new(b.x, b.y, b.z))
                    .unwrap_or(Rgb::new(0.0, 0.0, 0.0))
            };
            let fragment = fragments[i];
            let d_prev_alpha = d_curr_alpha * (1.0 - fragment.alpha);
            let mut d_alpha_i = d_curr_alpha * (1.0 - prev_alpha);
            d_alpha_i += d_curr_color.dot(fragment.color.sub(prev_color));
            let d_prev_color = d_curr_color.scale(1.0 - fragment.alpha);
            let d_color_i = d_curr_color.scale(fragment.alpha);

            let d_vec4 = Vec4::new(d_color_i.r, d_color_i.g, d_color_i.b, d_alpha_i);
            let mut translation_delta = Vec2::ZERO;
            let group = &scene.groups[fragment.group_id];
            let d_group = &mut grads.shape_groups[fragment.group_id];
            if fragment.is_stroke {
                if let (Some(paint), Some(d_paint)) = (group.stroke.as_ref(), d_group.stroke.as_mut()) {
                    d_sample_paint(paint, d_paint, pt, d_vec4, &mut translation_delta);
                }
            } else if let (Some(paint), Some(d_paint)) = (group.fill.as_ref(), d_group.fill.as_mut()) {
                d_sample_paint(paint, d_paint, pt, d_vec4, &mut translation_delta);
            }
            accumulate_translation(grads, translation_index, translation_delta);

            d_curr_color = d_prev_color;
            d_curr_alpha = d_prev_alpha;
        }

        if background.is_some() {
            let d_bg = Color {
                r: d_curr_color.r,
                g: d_curr_color.g,
                b: d_curr_color.b,
                a: d_curr_alpha,
            };
            accumulate_background_grad(scene, grads, pixel_index, d_bg.to_vec4());
        }
    }

    Vec4::new(final_color.r, final_color.g, final_color.b, final_alpha)
}

pub(super) fn sample_color_prefiltered(
    scene: &crate::scene::Scene,
    bvh: &SceneBvh,
    screen_pt: Vec2,
    background: Option<Vec4>,
    d_color: Option<Vec4>,
    grads: &mut SceneGrad,
    translation_index: Option<usize>,
    pixel_index: usize,
    dist_options: DistanceOptions,
) -> Vec4 {
    let pt = Vec2::new(screen_pt.x * scene.width as f32, screen_pt.y * scene.height as f32);
    let mut fragments: Vec<PrefilterFragment> = Vec::new();
    for group_id in 0..scene.groups.len() {
        let group = &scene.groups[group_id];
        if group.stroke.is_some() {
            if let Some(hit) = compute_distance_bvh(scene, bvh, group_id, pt, f32::INFINITY, dist_options) {
                let shape = &scene.shapes[hit.shape_index];
                let d = hit.distance;
                let w = smoothstep(d.abs() + shape.stroke_width) - smoothstep(d.abs() - shape.stroke_width);
                if w > 0.0 {
                    let mut color = paint_color(group.stroke.as_ref().unwrap(), pt);
                    color.w *= w;
                    fragments.push(PrefilterFragment {
                        color: Rgb::new(color.x, color.y, color.z),
                        alpha: color.w,
                        group_id,
                        shape_id: hit.shape_index,
                        distance: d,
                        closest_pt: hit.point,
                        is_stroke: true,
                        path_info: path_info_from_closest(scene, hit.shape_index, hit.path),
                        within_distance: true,
                    });
                }
            }
        }
        if group.fill.is_some() {
            let hit = compute_distance_bvh(scene, bvh, group_id, pt, 1.0, dist_options);
            let inside = is_inside_bvh(scene, bvh, group_id, pt);
            if hit.is_some() || inside {
                let (shape_id, closest_pt, path_info, mut d) = if let Some(hit) = hit {
                    (hit.shape_index, hit.point, path_info_from_closest(scene, hit.shape_index, hit.path), hit.distance)
                } else {
                    (0usize, Vec2::ZERO, None, 0.0)
                };
                if !inside {
                    d = -d;
                }
                let w = smoothstep(d);
                if w > 0.0 {
                    let mut color = paint_color(group.fill.as_ref().unwrap(), pt);
                    color.w *= w;
                    fragments.push(PrefilterFragment {
                        color: Rgb::new(color.x, color.y, color.z),
                        alpha: color.w,
                        group_id,
                        shape_id,
                        distance: d,
                        closest_pt,
                        is_stroke: false,
                        path_info,
                        within_distance: hit.is_some(),
                    });
                }
            }
        }
    }

    if fragments.is_empty() {
        if let Some(bg) = background {
            if let Some(d) = d_color {
                accumulate_background_grad(scene, grads, pixel_index, d);
            }
            return bg;
        }
        return Vec4::ZERO;
    }

    fragments.sort_by_key(|f| f.group_id);

    let mut accum_color: Vec<Rgb> = Vec::with_capacity(fragments.len());
    let mut accum_alpha: Vec<f32> = Vec::with_capacity(fragments.len());
    let mut prev_color = Rgb::new(0.0, 0.0, 0.0);
    let mut prev_alpha = 0.0f32;
    if let Some(bg) = background {
        prev_color = Rgb::new(bg.x, bg.y, bg.z);
        prev_alpha = bg.w;
    }

    for fragment in &fragments {
        let new_color = fragment.color;
        let new_alpha = fragment.alpha;
        let blended = prev_color.scale(1.0 - new_alpha).add(new_color.scale(new_alpha));
        let blended_alpha = prev_alpha * (1.0 - new_alpha) + new_alpha;
        accum_color.push(blended);
        accum_alpha.push(blended_alpha);
        prev_color = blended;
        prev_alpha = blended_alpha;
    }

    let mut final_color = accum_color[accum_color.len() - 1];
    let final_alpha = accum_alpha[accum_alpha.len() - 1];
    if final_alpha > 1.0e-6 {
        final_color = final_color.scale(1.0 / final_alpha);
    }

    if let Some(d) = d_color {
        let mut d_curr_color = Rgb::new(d.x, d.y, d.z);
        let mut d_curr_alpha = d.w;
        if final_alpha > 1.0e-6 {
            d_curr_color = d_curr_color.scale(1.0 / final_alpha);
            d_curr_alpha -= d_curr_color.dot(final_color) / final_alpha;
        }
        for i in (0..fragments.len()).rev() {
            let prev_alpha = if i > 0 {
                accum_alpha[i - 1]
            } else {
                background.map(|b| b.w).unwrap_or(0.0)
            };
            let prev_color = if i > 0 {
                accum_color[i - 1]
            } else {
                background
                    .map(|b| Rgb::new(b.x, b.y, b.z))
                    .unwrap_or(Rgb::new(0.0, 0.0, 0.0))
            };
            let fragment = fragments[i];
            let d_prev_alpha = d_curr_alpha * (1.0 - fragment.alpha);
            let mut d_alpha_i = d_curr_alpha * (1.0 - prev_alpha);
            d_alpha_i += d_curr_color.dot(fragment.color.sub(prev_color));
            let d_prev_color = d_curr_color.scale(1.0 - fragment.alpha);
            let d_color_i = d_curr_color.scale(fragment.alpha);

            if fragment.is_stroke {
                let shape = &scene.shapes[fragment.shape_id];
                let d = fragment.distance;
                let abs_plus = d.abs() + shape.stroke_width;
                let abs_minus = d.abs() - shape.stroke_width;
                let w = smoothstep(abs_plus) - smoothstep(abs_minus);
                if w != 0.0 {
                    let d_w = if w > 0.0 { (fragment.alpha / w) * d_alpha_i } else { 0.0 };
                    let mut d_alpha_i = d_alpha_i * w;

                    let mut translation_delta = Vec2::ZERO;
                    let group = &scene.groups[fragment.group_id];
                    let d_group = &mut grads.shape_groups[fragment.group_id];
                    if let (Some(paint), Some(d_paint)) = (group.stroke.as_ref(), d_group.stroke.as_mut()) {
                        d_sample_paint(
                            paint,
                            d_paint,
                            pt,
                            Vec4::new(d_color_i.r, d_color_i.g, d_color_i.b, d_alpha_i),
                            &mut translation_delta,
                        );
                    }
                    accumulate_translation(grads, translation_index, translation_delta);

                    let d_abs_plus = d_smoothstep(abs_plus, d_w);
                    let d_abs_minus = -d_smoothstep(abs_minus, d_w);
                    let mut d_d = d_abs_plus + d_abs_minus;
                    if d < 0.0 {
                        d_d = -d_d;
                    }
                    let d_stroke_width = d_abs_plus - d_abs_minus;

                    let d_shape = &mut grads.shapes[fragment.shape_id];
                    d_shape.stroke_width += d_stroke_width;

                    if d_d.abs() > 1.0e-10 {
                        let d_pt = {
                            let d_group = &mut grads.shape_groups[fragment.group_id];
                            d_compute_distance(
                                &scene.groups[fragment.group_id],
                                shape,
                                pt,
                                fragment.closest_pt,
                                fragment.path_info,
                                d_d,
                                d_shape,
                                d_group,
                            )
                        };
                        accumulate_translation(grads, translation_index, d_pt * -1.0);
                    }
                }
            } else {
                let d = fragment.distance;
                let w = smoothstep(d);
                if w != 0.0 {
                    let d_w = if w > 0.0 { (fragment.alpha / w) * d_alpha_i } else { 0.0 };
                    let mut d_alpha_i = d_alpha_i * w;

                    let mut translation_delta = Vec2::ZERO;
                    let group = &scene.groups[fragment.group_id];
                    let d_group = &mut grads.shape_groups[fragment.group_id];
                    if let (Some(paint), Some(d_paint)) = (group.fill.as_ref(), d_group.fill.as_mut()) {
                        d_sample_paint(
                            paint,
                            d_paint,
                            pt,
                            Vec4::new(d_color_i.r, d_color_i.g, d_color_i.b, d_alpha_i),
                            &mut translation_delta,
                        );
                    }
                    accumulate_translation(grads, translation_index, translation_delta);

                    let mut d_d = d_smoothstep(d, d_w);
                    if d < 0.0 {
                        d_d = -d_d;
                    }

                    if d_d.abs() > 1.0e-10 && fragment.within_distance {
                        let d_pt = {
                            let d_shape = &mut grads.shapes[fragment.shape_id];
                            let d_group = &mut grads.shape_groups[fragment.group_id];
                            d_compute_distance(
                                &scene.groups[fragment.group_id],
                                &scene.shapes[fragment.shape_id],
                                pt,
                                fragment.closest_pt,
                                fragment.path_info,
                                d_d,
                                d_shape,
                                d_group,
                            )
                        };
                        accumulate_translation(grads, translation_index, d_pt * -1.0);
                    }
                }
            }

            d_curr_color = d_prev_color;
            d_curr_alpha = d_prev_alpha;
        }

        if background.is_some() {
            let d_bg = Color {
                r: d_curr_color.r,
                g: d_curr_color.g,
                b: d_curr_color.b,
                a: d_curr_alpha,
            };
            accumulate_background_grad(scene, grads, pixel_index, d_bg.to_vec4());
        }
    }

    Vec4::new(final_color.r, final_color.g, final_color.b, final_alpha)
}

fn within_distance_edge(
    scene: &crate::scene::Scene,
    bvh: &SceneBvh,
    group_id: usize,
    pt: Vec2,
    edge_query: Option<&mut EdgeQuery>,
) -> bool {
    let hit = within_distance_bvh(scene, bvh, group_id, pt);
    let Some(eq) = edge_query else {
        return hit;
    };
    if eq.shape_group_id != group_id {
        return hit;
    }
    let Some(group) = scene.groups.get(group_id) else {
        return hit;
    };
    let Some(shape) = scene.shapes.get(eq.shape_id) else {
        return hit;
    };
    if shape.stroke_width <= 0.0 {
        return hit;
    }
    let local_pt = group.canvas_to_shape.transform_point(pt);
    if crate::distance::closest::within_distance_shape(shape, local_pt, shape.stroke_width) {
        eq.hit = true;
    }
    hit
}

fn is_inside_edge(
    scene: &crate::scene::Scene,
    bvh: &SceneBvh,
    group_id: usize,
    pt: Vec2,
    edge_query: Option<&mut EdgeQuery>,
) -> bool {
    let inside = is_inside_bvh(scene, bvh, group_id, pt);
    let Some(eq) = edge_query else {
        return inside;
    };
    if eq.shape_group_id != group_id {
        return inside;
    }
    let Some(group) = scene.groups.get(group_id) else {
        return inside;
    };
    let Some(shape) = scene.shapes.get(eq.shape_id) else {
        return inside;
    };
    let local_pt = group.canvas_to_shape.transform_point(pt);
    let shape_pt = transform_point_inverse(shape.transform, local_pt);
    let path_bvh = bvh
        .groups
        .get(group_id)
        .and_then(|gbvh| gbvh.shapes.iter().find(|s| s.shape_index == eq.shape_id))
        .and_then(|shape_bvh| shape_bvh.path_bvh.as_ref());
    let winding = crate::distance::winding::winding_number_shape(shape, path_bvh, shape_pt);
    let hit = match group.fill_rule {
        FillRule::EvenOdd => winding.abs() % 2 == 1,
        FillRule::NonZero => winding != 0,
    };
    if hit {
        eq.hit = true;
    }
    inside
}

fn accumulate_translation(grads: &mut SceneGrad, translation_index: Option<usize>, delta: Vec2) {
    if let Some(index) = translation_index {
        if let Some(trans) = grads.translation.as_mut() {
            let idx = index * 2;
            trans[idx] += delta.x;
            trans[idx + 1] += delta.y;
        }
    }
}
