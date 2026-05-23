import argparse
import csv
import math
import os
import random

import carla
import numpy as np


def choose_blueprint(blueprints, vehicle_filter):
    # Prefer a specific vehicle if available; otherwise use any vehicle blueprint.
    matches = list(blueprints.filter(vehicle_filter))
    if matches:
        return random.choice(matches)
    fallback = list(blueprints.filter("vehicle.*"))
    if not fallback:
        raise RuntimeError("No vehicle blueprints available.")
    return random.choice(fallback)


def spawn_vehicle(world, blueprint):
    # Try random spawn points until a valid actor spawn succeeds.
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points found in current map.")
    random.shuffle(spawn_points)
    for transform in spawn_points:
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is not None:
            return actor, transform
    raise RuntimeError("Failed to spawn vehicle at any spawn point.")


def make_static_roadside_transform(world, vehicle_transform):
    # Build a world-space camera transform near the nearest lane.
    # This is useful for "traffic cam" style viewpoints (not attached to vehicle).
    road_wp = world.get_map().get_waypoint(vehicle_transform.location, project_to_road=True)
    road_tf = road_wp.transform
    yaw_rad = math.radians(road_tf.rotation.yaw)
    right_x = math.cos(yaw_rad + math.pi / 2.0)
    right_y = math.sin(yaw_rad + math.pi / 2.0)

    offset = max(road_wp.lane_width * 0.5, 1.5) + 3.0
    location = carla.Location(
        x=road_tf.location.x + right_x * offset,
        y=road_tf.location.y + right_y * offset,
        z=3.0,
    )
    rotation = carla.Rotation(
        pitch=-10.0,
        yaw=road_tf.rotation.yaw - 90.0,
        roll=0.0,
    )
    return carla.Transform(location, rotation)


def build_camera_transform(args):
    # Vehicle-relative camera placement presets.
    presets = {
        "front_roof": carla.Transform(carla.Location(x=1.8, z=1.6), carla.Rotation(pitch=-5.0)),
        "rear_roof": carla.Transform(carla.Location(x=-1.8, z=1.7), carla.Rotation(yaw=180.0, pitch=-5.0)),
        "left_mirror": carla.Transform(carla.Location(x=0.6, y=-0.9, z=1.3), carla.Rotation(yaw=-95.0)),
        "right_mirror": carla.Transform(carla.Location(x=0.6, y=0.9, z=1.3), carla.Rotation(yaw=95.0)),
        "birdseye": carla.Transform(carla.Location(x=0.0, z=18.0), carla.Rotation(pitch=-90.0)),
    }
    base = presets[args.camera_mount]
    base.location.x += args.camera_x
    base.location.y += args.camera_y
    base.location.z += args.camera_z
    base.rotation.pitch += args.camera_pitch
    base.rotation.yaw += args.camera_yaw
    base.rotation.roll += args.camera_roll
    return base


def process_camera_image(image):
    # CARLA RGB camera raw_data layout is BGRA uint8.
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    # Convert BGRA -> RGB for easier processing.
    rgb = array[:, :, :3][:, :, ::-1]

    # Example processing: luminance map and two lightweight summary features.
    grayscale = np.dot(rgb[:, :, :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    mean_luma = float(grayscale.mean())
    center = grayscale[image.height // 2, image.width // 2]
    return mean_luma, int(center)


def process_radar_measurement(radar_measurement):
    # Each detection has azimuth/altitude/depth/velocity.
    # Here we compute compact frame-level statistics.
    count = len(radar_measurement)
    if count == 0:
        return 0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    depths = [det.depth for det in radar_measurement]
    velocities = [det.velocity for det in radar_measurement]
    azimuths = [math.degrees(det.azimuth) for det in radar_measurement]

    return (
        count,
        float(min(depths)),
        float(sum(depths) / count),
        float(max(velocities)),
        float(sum(abs(v) for v in velocities) / count),
        float(sum(abs(a) for a in azimuths) / count),
    )


def trim_buffer(buffer, max_size=240):
    # Sensor callbacks are asynchronous; cap memory growth if frames drift.
    while len(buffer) > max_size:
        buffer.pop(next(iter(buffer)))


def main():
    parser = argparse.ArgumentParser(
        description="Example CARLA script for camera and radar placement, capture, and processing."
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port.")
    parser.add_argument("--frames", type=int, default=300, help="Number of synchronous ticks to run.")
    parser.add_argument("--delta-seconds", type=float, default=0.05, help="Fixed delta in synchronous mode.")
    parser.add_argument(
        "--vehicle-filter",
        default="vehicle.tesla.model3",
        help="Vehicle blueprint filter. Falls back to random vehicle.* if no match.",
    )

    parser.add_argument(
        "--camera-mount",
        choices=["front_roof", "rear_roof", "left_mirror", "right_mirror", "birdseye", "static_roadside"],
        default="front_roof",
        help="Camera placement preset. static_roadside places camera in world space near the road.",
    )
    parser.add_argument("--camera-x", type=float, default=0.0, help="Extra camera X offset from preset.")
    parser.add_argument("--camera-y", type=float, default=0.0, help="Extra camera Y offset from preset.")
    parser.add_argument("--camera-z", type=float, default=0.0, help="Extra camera Z offset from preset.")
    parser.add_argument("--camera-pitch", type=float, default=0.0, help="Extra camera pitch from preset.")
    parser.add_argument("--camera-yaw", type=float, default=0.0, help="Extra camera yaw from preset.")
    parser.add_argument("--camera-roll", type=float, default=0.0, help="Extra camera roll from preset.")
    parser.add_argument("--camera-width", type=int, default=1280, help="Camera image width.")
    parser.add_argument("--camera-height", type=int, default=720, help="Camera image height.")
    parser.add_argument("--camera-fov", type=float, default=90.0, help="Camera field of view in degrees.")

    parser.add_argument("--radar-range", type=float, default=60.0, help="Radar range in meters.")
    parser.add_argument("--radar-hfov", type=float, default=35.0, help="Radar horizontal FoV in degrees.")
    parser.add_argument("--radar-vfov", type=float, default=20.0, help="Radar vertical FoV in degrees.")
    parser.add_argument("--radar-pps", type=int, default=1500, help="Radar points per second.")

    parser.add_argument("--save-images", action="store_true", help="Save camera images to disk.")
    parser.add_argument("--output-dir", default="output/sensor_demo", help="Output directory.")
    args = parser.parse_args()

    # Prepare output paths up front.
    os.makedirs(args.output_dir, exist_ok=True)
    image_dir = os.path.join(args.output_dir, "images")
    if args.save_images:
        os.makedirs(image_dir, exist_ok=True)

    # Connect to CARLA world and blueprint library.
    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    vehicle = None
    camera = None
    radar = None
    original_settings = world.get_settings()

    camera_by_frame = {}
    radar_by_frame = {}

    camera_csv_path = os.path.join(args.output_dir, "camera_processed.csv")
    radar_csv_path = os.path.join(args.output_dir, "radar_processed.csv")

    try:
        # Run in synchronous mode so each tick corresponds to deterministic sensor frames.
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.delta_seconds
        world.apply_settings(settings)

        # Spawn and enable autopilot so the car moves without manual control.
        vehicle_bp = choose_blueprint(bp_lib, args.vehicle_filter)
        vehicle, spawn_tf = spawn_vehicle(world, vehicle_bp)
        vehicle.set_autopilot(True)

        # Configure RGB camera intrinsics.
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(args.camera_width))
        cam_bp.set_attribute("image_size_y", str(args.camera_height))
        cam_bp.set_attribute("fov", str(args.camera_fov))

        if args.camera_mount == "static_roadside":
            # World-space camera (acts like fixed roadside infrastructure).
            camera_tf = make_static_roadside_transform(world, spawn_tf)
            camera = world.spawn_actor(cam_bp, camera_tf)
        else:
            # Vehicle-attached camera using selected preset + user offsets.
            camera_tf = build_camera_transform(args)
            camera = world.spawn_actor(cam_bp, camera_tf, attach_to=vehicle)

        # Configure and attach radar near front bumper.
        radar_bp = bp_lib.find("sensor.other.radar")
        radar_bp.set_attribute("range", str(args.radar_range))
        radar_bp.set_attribute("horizontal_fov", str(args.radar_hfov))
        radar_bp.set_attribute("vertical_fov", str(args.radar_vfov))
        radar_bp.set_attribute("points_per_second", str(args.radar_pps))

        radar_tf = carla.Transform(carla.Location(x=2.0, z=1.0))
        radar = world.spawn_actor(radar_bp, radar_tf, attach_to=vehicle)

        def on_camera(image):
            # Store latest frame so main loop can process/write in tick order.
            camera_by_frame[image.frame] = image
            trim_buffer(camera_by_frame)
            if args.save_images:
                image.save_to_disk(os.path.join(image_dir, f"{image.frame:06d}.png"))

        def on_radar(measurement):
            # Same strategy for radar measurements.
            radar_by_frame[measurement.frame] = measurement
            trim_buffer(radar_by_frame)

        camera.listen(on_camera)
        radar.listen(on_radar)

        with open(camera_csv_path, "w", newline="", encoding="utf-8") as camera_csv, open(
            radar_csv_path, "w", newline="", encoding="utf-8"
        ) as radar_csv:
            cam_writer = csv.writer(camera_csv)
            radar_writer = csv.writer(radar_csv)

            cam_writer.writerow(
                ["frame", "sim_time_s", "mean_luma", "center_pixel_luma", "width", "height"]
            )
            radar_writer.writerow(
                [
                    "frame",
                    "sim_time_s",
                    "point_count",
                    "min_depth_m",
                    "mean_depth_m",
                    "max_rel_velocity_mps",
                    "mean_abs_rel_velocity_mps",
                    "mean_abs_azimuth_deg",
                ]
            )

            print("Running synchronous capture loop...")
            for _ in range(args.frames):
                # Advance simulation one fixed step.
                snapshot = world.tick()
                frame = snapshot.frame
                sim_time = snapshot.timestamp.elapsed_seconds

                # Process camera data if callback delivered this frame.
                image = camera_by_frame.pop(frame, None)
                if image is not None:
                    mean_luma, center_luma = process_camera_image(image)
                    cam_writer.writerow(
                        [frame, sim_time, mean_luma, center_luma, image.width, image.height]
                    )

                # Process radar data if callback delivered this frame.
                measurement = radar_by_frame.pop(frame, None)
                if measurement is not None:
                    (
                        point_count,
                        min_depth,
                        mean_depth,
                        max_velocity,
                        mean_abs_velocity,
                        mean_abs_azimuth,
                    ) = process_radar_measurement(measurement)
                    radar_writer.writerow(
                        [
                            frame,
                            sim_time,
                            point_count,
                            min_depth,
                            mean_depth,
                            max_velocity,
                            mean_abs_velocity,
                            mean_abs_azimuth,
                        ]
                    )

        print("Completed.")
        print(f"Camera processed data: {camera_csv_path}")
        print(f"Radar processed data:  {radar_csv_path}")
        if args.save_images:
            print(f"Saved camera frames to: {image_dir}")

    finally:
        # Stop sensors, destroy actors, and restore world settings.
        if camera is not None:
            camera.stop()
            camera.destroy()
        if radar is not None:
            radar.stop()
            radar.destroy()
        if vehicle is not None:
            vehicle.destroy()
        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()
