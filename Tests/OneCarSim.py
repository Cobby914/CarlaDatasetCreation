import csv
import math
import os
import random

import carla


def make_roadside_transform(world, vehicle_spawn):
    """
    Build a roadside camera transform near the vehicle spawn:
    - take nearest waypoint
    - move sideways to road shoulder
    - look slightly downward toward road
    """
    m = world.get_map()
    wp = m.get_waypoint(vehicle_spawn.location, project_to_road=True)

    road_tf = wp.transform
    yaw_rad = math.radians(road_tf.rotation.yaw)

    # Right vector in XY plane
    right_x = math.cos(yaw_rad + math.pi / 2.0)
    right_y = math.sin(yaw_rad + math.pi / 2.0)

    # Approx shoulder offset: lane width / 2 + extra meters
    side_offset = (wp.lane_width * 0.5) + 3.0

    cam_loc = carla.Location(
        x=road_tf.location.x + right_x * side_offset,
        y=road_tf.location.y + right_y * side_offset,
        z=3.0,
    )

    # Face toward road / vehicle direction
    cam_rot = carla.Rotation(
        pitch=-10.0,
        yaw=road_tf.rotation.yaw - 90.0,
        roll=0.0,
    )

    return carla.Transform(cam_loc, cam_rot)


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    vehicle = None
    camera = None
    original_settings = world.get_settings()

    os.makedirs("output/images", exist_ok=True)
    csv_path = "output/vehicle_data.csv"

    try:
        # Synchronous mode = clean frame alignment
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # Spawn vehicle
        vehicle_bp = random.choice(bp_lib.filter("vehicle.*"))
        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_autopilot(True)

        # Spawn roadside camera (static, not attached)
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "1280")
        cam_bp.set_attribute("image_size_y", "720")
        cam_bp.set_attribute("fov", "90")

        cam_tf = make_roadside_transform(world, spawn_point)
        camera = world.spawn_actor(cam_bp, cam_tf)

        # Image callback stores latest frame
        image_buffer = {}

        def on_image(image):
            image.save_to_disk(f"output/images/{image.frame:06d}.png")
            image_buffer[image.frame] = True

        camera.listen(on_image)

        # CSV logging
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame", "sim_time",
                "car_x", "car_y", "car_z",
                "yaw", "pitch", "roll",
                "speed_mps",
                "vel_x", "vel_y", "vel_z",
                "acc_x", "acc_y", "acc_z",
                "throttle", "steer", "brake", "hand_brake", "reverse", "gear",
            ])

            for _ in range(400):  # ~20 seconds at 0.05s tick
                world.tick()
                frame = world.get_snapshot().frame

                # Only log when corresponding image frame exists
                if frame not in image_buffer:
                    continue

                tf = vehicle.get_transform()
                vel = vehicle.get_velocity()
                acc = vehicle.get_acceleration()
                ctrl = vehicle.get_control()
                speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

                writer.writerow([
                    frame, world.get_snapshot().timestamp.elapsed_seconds,
                    tf.location.x, tf.location.y, tf.location.z,
                    tf.rotation.yaw, tf.rotation.pitch, tf.rotation.roll,
                    speed,
                    vel.x, vel.y, vel.z,
                    acc.x, acc.y, acc.z,
                    ctrl.throttle, ctrl.steer, ctrl.brake, ctrl.hand_brake, ctrl.reverse, ctrl.gear
                ])

        print("Done. Images in output/images and telemetry in output/vehicle_data.csv")

    finally:
        if camera is not None:
            camera.stop()
            camera.destroy()
        if vehicle is not None:
            vehicle.destroy()
        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()