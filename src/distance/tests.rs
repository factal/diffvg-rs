use super::*;
use crate::scene::{FillRule, Paint, ShapeGroup, Scene};
use crate::{Color, Shape, ShapeGeometry};
use crate::math::Vec2;
use crate::geometry::Path;

#[test]
fn test_compute_distance_circle() {
    let mut scene = Scene::new(100, 100);
    let circle = Shape::new(ShapeGeometry::Circle {
        center: Vec2::new(50.0, 50.0),
        radius: 10.0,
    });
    scene.shapes.push(circle);
    scene.groups.push(ShapeGroup::new(
        vec![0],
        Some(Paint::Solid(Color::opaque(1.0, 0.0, 0.0))),
        None,
    ));

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
    let mut group = ShapeGroup::new(
        vec![0],
        None,
        Some(Paint::Solid(Color::opaque(0.0, 0.0, 0.0))),
    );
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
    let accel = compute_distance_bvh(&scene, &bvh, 0, pt, f32::INFINITY, DistanceOptions::default());
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
    let accel = compute_distance_bvh(&scene, &bvh, 0, pt, f32::INFINITY, DistanceOptions::default());
    assert!(linear.is_some());
    assert!(accel.is_some());
    let linear = linear.unwrap();
    let accel = accel.unwrap();
    assert!((linear.distance - accel.distance).abs() < 1.0e-3);
    assert_eq!(linear.shape_index, accel.shape_index);
}
