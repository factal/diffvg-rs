use crate::color::Color;
use crate::grad::SceneGrad;
use crate::math::Vec4;
use crate::scene::Scene;

pub(super) fn sample_background(scene: &Scene, pixel_index: usize) -> Vec4 {
    if let Some(background_image) = scene.background_image.as_ref() {
        let base = pixel_index * 4;
        let r = background_image[base];
        let g = background_image[base + 1];
        let b = background_image[base + 2];
        let a = background_image[base + 3];
        Vec4::new(r * a, g * a, b * a, a)
    } else {
        Vec4::new(
            scene.background.r * scene.background.a,
            scene.background.g * scene.background.a,
            scene.background.b * scene.background.a,
            scene.background.a,
        )
    }
}

pub(super) fn finalize_background_gradients(scene: &Scene, grads: &mut SceneGrad) {
    if let Some(image) = scene.background_image.as_ref() {
        if let Some(d_bg) = grads.background_image.as_mut() {
            for (i, chunk) in d_bg.chunks_mut(4).enumerate() {
                let base = i * 4;
                let r = image[base];
                let g = image[base + 1];
                let b = image[base + 2];
                let a = image[base + 3];
                let dr = chunk[0] * a;
                let dg = chunk[1] * a;
                let db = chunk[2] * a;
                let da = chunk[0] * r + chunk[1] * g + chunk[2] * b + chunk[3];
                chunk[0] = dr;
                chunk[1] = dg;
                chunk[2] = db;
                chunk[3] = da;
            }
        }
    } else {
        let r = scene.background.r;
        let g = scene.background.g;
        let b = scene.background.b;
        let a = scene.background.a;
        let d_pre = grads.background;
        grads.background = Color {
            r: d_pre.r * a,
            g: d_pre.g * a,
            b: d_pre.b * a,
            a: d_pre.r * r + d_pre.g * g + d_pre.b * b + d_pre.a,
        };
    }
}

pub(super) fn accumulate_background_grad(
    scene: &Scene,
    grads: &mut SceneGrad,
    pixel_index: usize,
    d_color: Vec4,
) {
    if scene.background_image.is_some() {
        if let Some(d_image) = grads.background_image.as_mut() {
            let base = pixel_index * 4;
            if base + 3 < d_image.len() {
                d_image[base] += d_color.x;
                d_image[base + 1] += d_color.y;
                d_image[base + 2] += d_color.z;
                d_image[base + 3] += d_color.w;
            }
        }
        return;
    }

    let d_bg = Color {
        r: d_color.x,
        g: d_color.y,
        b: d_color.z,
        a: d_color.w,
    };
    grads.background = add_color(grads.background, d_bg);
}

fn add_color(a: Color, b: Color) -> Color {
    Color {
        r: a.r + b.r,
        g: a.g + b.g,
        b: a.b + b.b,
        a: a.a + b.a,
    }
}
