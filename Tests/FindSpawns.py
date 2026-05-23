import carla
import time


client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

world = None
for attempt in range(1, 6):
    try:
        world = client.get_world()
        break
    except RuntimeError:
        if attempt == 5:
            raise RuntimeError(
                "Could not connect to CARLA at localhost:2000. "
                "Make sure CarlaUE4 is running and fully loaded."
            )
        time.sleep(2)

current_map = world.get_map()

# spawn_transforms will be a list of carla.Transform
spawn_transforms = current_map.get_spawn_points()
print(f"Found {len(spawn_transforms)} spawn points on map: {current_map.name}")

vehicle_output_path = "spawn_points.txt"
camera_output_path = "camera_spawn_points.txt"
radar_output_path = "radar_spawn_points.txt"

# Camera points: place near traffic lights so they represent infrastructure cameras.
camera_spawn_transforms = []
traffic_light_bbs = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
for bbox in traffic_light_bbs:
    camera_location = bbox.location + carla.Location(z=bbox.extent.z + 0.8)
    road_wp = current_map.get_waypoint(
        camera_location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    camera_yaw = road_wp.transform.rotation.yaw if road_wp else 0.0
    camera_spawn_transforms.append(
        carla.Transform(
            camera_location,
            carla.Rotation(pitch=-15.0, yaw=camera_yaw, roll=0.0),
        )
    )

# Radar points: sample pedestrian navigation locations and keep sidewalk-only points.
radar_spawn_transforms = []
seen_sidewalk_points = set()
target_radar_points = len(spawn_transforms)
max_attempts = target_radar_points * 40
for _ in range(max_attempts):
    if len(radar_spawn_transforms) >= target_radar_points:
        break

    nav_location = world.get_random_location_from_navigation()
    if nav_location is None:
        continue

    sidewalk_wp = current_map.get_waypoint(
        nav_location,
        project_to_road=False,
        lane_type=carla.LaneType.Sidewalk,
    )
    if sidewalk_wp is None:
        continue

    snapped_location = sidewalk_wp.transform.location
    point_key = (round(snapped_location.x, 1), round(snapped_location.y, 1))
    if point_key in seen_sidewalk_points:
        continue
    seen_sidewalk_points.add(point_key)

    nearest_road_wp = current_map.get_waypoint(
        snapped_location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    radar_yaw = nearest_road_wp.transform.rotation.yaw if nearest_road_wp else 0.0
    radar_spawn_transforms.append(
        carla.Transform(
            snapped_location + carla.Location(z=1.0),
            carla.Rotation(pitch=0.0, yaw=radar_yaw, roll=0.0),
        )
    )

with open(vehicle_output_path, "w", encoding="utf-8") as f:
    f.write(f"Map: {current_map.name}\n")
    f.write(f"Vehicle spawn points: {len(spawn_transforms)}\n\n")
    for i, transform in enumerate(spawn_transforms):
        f.write(f"[{i}] {transform}\n")

with open(camera_output_path, "w", encoding="utf-8") as f:
    f.write(f"Map: {current_map.name}\n")
    f.write(f"Camera spawn points: {len(camera_spawn_transforms)}\n")
    f.write("Note: camera points are derived from traffic light locations.\n\n")
    for i, transform in enumerate(camera_spawn_transforms):
        f.write(f"[{i}] {transform}\n")

with open(radar_output_path, "w", encoding="utf-8") as f:
    f.write(f"Map: {current_map.name}\n")
    f.write(f"Radar spawn points: {len(radar_spawn_transforms)}\n")
    f.write("Note: radar points are sampled from sidewalk navigation locations.\n\n")
    for i, transform in enumerate(radar_spawn_transforms):
        f.write(f"[{i}] {transform}\n")

print(f"Saved vehicle spawn points to {vehicle_output_path}")
print(f"Saved camera spawn points to {camera_output_path}")
print(f"Saved radar spawn points to {radar_output_path}")

origin_location = carla.Location(x=0.0, y=0.0, z=2.0)
world.debug.draw_string(
    origin_location,
    "ORIGIN (0,0,0)",
    draw_shadow=True,
    color=carla.Color(255, 255, 0),
    life_time=120.0,
    persistent_lines=False,
)

for i, transform in enumerate(spawn_transforms):
    print(f"[{i}] {transform}")
    label_location = transform.location + carla.Location(z=1.5)
    world.debug.draw_string(
        label_location,
        str(i),
        draw_shadow=False,
        color=carla.Color(255, 0, 0),
        life_time=120.0,
        persistent_lines=False,
    )

for i, camera_transform in enumerate(camera_spawn_transforms):
    camera_label_location = camera_transform.location
    world.debug.draw_string(
        camera_label_location,
        f"C{i}",
        draw_shadow=False,
        color=carla.Color(0, 120, 255),
        life_time=120.0,
        persistent_lines=False,
    )

for i, radar_transform in enumerate(radar_spawn_transforms):
    radar_label_location = radar_transform.location
    world.debug.draw_string(
        radar_label_location,
        f"R{i}",
        draw_shadow=False,
        color=carla.Color(0, 180, 255),
        life_time=120.0,
        persistent_lines=False,
    )

print("Spawn labels drawn: red=vehicle, blue C#=camera, blue R#=radar (120 seconds).")
time.sleep(5)