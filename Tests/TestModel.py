import argparse
import math
import random
import time

import carla


PERIMETER_STEP_M = 50.0


def distance_2d(a, b):
    """Return planar (x,y) distance between two CARLA locations."""
    dx = a.x - b.x
    dy = a.y - b.y
    return math.hypot(dx, dy)


def right_vector_from_yaw(yaw_deg):
    """Compute right-facing unit vector in XY plane from yaw angle (degrees)."""
    yaw_rad = math.radians(yaw_deg)
    return math.cos(yaw_rad + math.pi / 2.0), math.sin(yaw_rad + math.pi / 2.0)


def choose_forward_candidate(current_wp, candidates):
    """Pick the candidate whose heading deviates least from current waypoint."""
    if not candidates:
        return None
    current_yaw = current_wp.transform.rotation.yaw
    scored = []
    for candidate in candidates:
        delta = abs((candidate.transform.rotation.yaw - current_yaw + 180.0) % 360.0 - 180.0)
        scored.append((delta, candidate))
    scored.sort(key=lambda item: item[0])
    return scored[0][1]


def build_road_stretch_near_spectator(world, step=2.0, max_length=650.0):
    """
    Build a 1-lane road corridor centered at spectator position.

    Tunable:
    - step: waypoint sampling step in meters (smaller = denser/smoother)
    - max_length: target total road length in meters
    """
    world_map = world.get_map()
    spectator_loc = world.get_spectator().get_transform().location
    seed_wp = world_map.get_waypoint(
        spectator_loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if seed_wp is None:
        raise RuntimeError("Could not project spectator location onto a driving lane.")

    road_id = seed_wp.road_id
    lane_id = seed_wp.lane_id

    backward = []
    current = seed_wp
    traveled = 0.0
    while traveled < max_length * 0.5:
        prev_candidates = [
            wp
            for wp in current.previous(step)
            if wp.road_id == road_id and wp.lane_id == lane_id and not wp.is_junction
        ]
        next_wp = choose_forward_candidate(current, prev_candidates)
        if next_wp is None:
            break
        seg = distance_2d(current.transform.location, next_wp.transform.location)
        traveled += seg
        backward.append(next_wp)
        current = next_wp

    forward = []
    current = seed_wp
    traveled = 0.0
    while traveled < max_length * 0.5:
        next_candidates = [
            wp
            for wp in current.next(step)
            if wp.road_id == road_id and wp.lane_id == lane_id and not wp.is_junction
        ]
        next_wp = choose_forward_candidate(current, next_candidates)
        if next_wp is None:
            break
        seg = distance_2d(current.transform.location, next_wp.transform.location)
        traveled += seg
        forward.append(next_wp)
        current = next_wp

    waypoints = list(reversed(backward)) + [seed_wp] + forward
    if len(waypoints) < 20:
        raise RuntimeError(
            "Road stretch is too short near spectator. Move spectator onto a longer straight road and retry."
        )
    return waypoints


def boundary_distance(location, min_x, max_x, min_y, max_y):
    """Return how close a point is to any map boundary edge in XY."""
    return min(
        abs(location.x - min_x),
        abs(location.x - max_x),
        abs(location.y - min_y),
        abs(location.y - max_y),
    )


def yaw_delta_deg(target_yaw, current_yaw):
    """Signed shortest angle difference (degrees) in range [-180, 180]."""
    return (target_yaw - current_yaw + 180.0) % 360.0 - 180.0


def pick_perimeter_start(waypoints):
    """Pick a deterministic outer-corner-ish seed waypoint for perimeter traversal."""
    return min(waypoints, key=lambda wp: (wp.transform.location.x + wp.transform.location.y))


def build_outer_perimeter_loop(world_map, step_m=PERIMETER_STEP_M, max_steps=450):
    """
    Build a loop that follows the drivable outer perimeter of the map.

    Tunable:
    - step_m: waypoint spacing used for perimeter graph traversal
    - max_steps: loop-search cap; raise for larger maps
    """
    all_waypoints = world_map.generate_waypoints(step_m)
    driving_waypoints = [
        wp
        for wp in all_waypoints
        if wp.lane_type == carla.LaneType.Driving and wp.is_junction is False
    ]
    if not driving_waypoints:
        raise RuntimeError("No non-junction driving waypoints found for perimeter route.")

    xs = [wp.transform.location.x for wp in driving_waypoints]
    ys = [wp.transform.location.y for wp in driving_waypoints]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    start_wp = pick_perimeter_start(driving_waypoints)
    loop = [start_wp]
    current = start_wp
    visited_keys = {
        (round(start_wp.transform.location.x, 1), round(start_wp.transform.location.y, 1))
    }

    for _ in range(max_steps):
        candidates = current.next(step_m)
        if not candidates:
            break

        current_yaw = current.transform.rotation.yaw
        scored = []
        for candidate in candidates:
            if candidate.lane_type != carla.LaneType.Driving:
                continue
            loc = candidate.transform.location
            edge_dist = boundary_distance(loc, min_x, max_x, min_y, max_y)
            turn_penalty = abs(yaw_delta_deg(candidate.transform.rotation.yaw, current_yaw))
            scored.append((edge_dist, turn_penalty, candidate))

        if not scored:
            break

        scored.sort(key=lambda item: (item[0], item[1]))
        next_wp = scored[0][2]
        next_key = (round(next_wp.transform.location.x, 1), round(next_wp.transform.location.y, 1))

        if len(loop) > 12 and next_key in visited_keys:
            loop.append(next_wp)
            break

        loop.append(next_wp)
        visited_keys.add(next_key)
        current = next_wp

    if len(loop) < 12:
        raise RuntimeError(
            "Could not build a usable perimeter loop. Try increasing max_steps or adjusting step_m."
        )
    return loop


def cumulative_distances(waypoints):
    """Return cumulative arc-length list and total length for waypoint path."""
    if not waypoints:
        return [], 0.0
    dists = [0.0]
    total = 0.0
    for i in range(1, len(waypoints)):
        total += distance_2d(waypoints[i - 1].transform.location, waypoints[i].transform.location)
        dists.append(total)
    return dists, total


def poisson_positions_fixed_count(count, length_m, edge_buffer=10.0):
    """
    Generate sorted 1D Poisson-like positions along a segment for fixed actor count.

    Tunable:
    - edge_buffer: keep spawns away from both segment ends (meters)
    """
    usable = max(0.0, length_m - 2.0 * edge_buffer)
    if usable <= 0.0 or count <= 0:
        return []

    gaps = [random.expovariate(1.0) for _ in range(count + 1)]
    total_gap = sum(gaps)
    positions = []
    running = 0.0
    for i in range(count):
        running += gaps[i]
        u = running / total_gap
        positions.append(edge_buffer + u * usable)
    return positions


def waypoint_at_distance(waypoints, cum_dists, distance_m):
    """Return waypoint nearest to the requested cumulative distance."""
    if not waypoints:
        return None
    idx = 0
    while idx + 1 < len(cum_dists) and cum_dists[idx + 1] < distance_m:
        idx += 1
    return waypoints[idx]


def waypoint_index_at_distance(cum_dists, distance_m):
    """Return index into waypoint list at requested cumulative distance."""
    idx = 0
    while idx + 1 < len(cum_dists) and cum_dists[idx + 1] < distance_m:
        idx += 1
    return idx


def set_all_traffic_lights_red(world, red_seconds=10000.0):
    """
    Force every traffic light in the world to red and freeze it.

    Tunable:
    - red_seconds: red phase duration (large value keeps them effectively red forever)
    """
    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    for light in lights:
        light.set_state(carla.TrafficLightState.Red)
        light.set_red_time(red_seconds)
        light.set_yellow_time(0.1)
        light.set_green_time(0.1)
        light.freeze(True)
    return len(lights)


def nearest_traffic_light(world, ref_location):
    """Find the closest traffic light to a reference location."""
    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    if not lights:
        return None
    return min(lights, key=lambda actor: distance_2d(actor.get_location(), ref_location))


def spawn_camera_on_light(world, traffic_light):
    """
    Attach RGB camera to a traffic light mast.

    Tunable:
    - image_size_x/image_size_y: output resolution
    - fov: camera field of view
    - relative transform: camera mounting height/orientation on the light
    """
    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "1280")
    cam_bp.set_attribute("image_size_y", "720")
    cam_bp.set_attribute("fov", "90")

    rel_tf = carla.Transform(
        carla.Location(x=0.0, y=0.0, z=3.0),
        carla.Rotation(pitch=-15.0, yaw=0.0, roll=0.0),
    )
    return world.spawn_actor(cam_bp, rel_tf, attach_to=traffic_light)


def build_radar_transforms(waypoints, spacing_m=35.0, side_offset_m=4.5, height_m=4.0):
    """
    Generate fixed roadside radar transforms along the selected road stretch.

    Tunable:
    - spacing_m: distance between radar nodes along the road
    - side_offset_m: lateral offset from lane center to roadside
    - height_m: sensor mounting height above roadway
    """
    if not waypoints:
        return []
    cum_dists, length_m = cumulative_distances(waypoints)
    if length_m <= 0.0:
        return []

    transforms = []
    d = 0.0
    while d <= length_m:
        wp = waypoint_at_distance(waypoints, cum_dists, d)
        if wp is None:
            break
        tf = wp.transform
        rx, ry = right_vector_from_yaw(tf.rotation.yaw)
        loc = carla.Location(
            x=tf.location.x + rx * side_offset_m,
            y=tf.location.y + ry * side_offset_m,
            z=tf.location.z + height_m,
        )
        rot = carla.Rotation(
            pitch=-3.0,
            yaw=tf.rotation.yaw - 90.0,
            roll=0.0,
        )
        transforms.append(carla.Transform(loc, rot))
        d += spacing_m
    return transforms


def spawn_radars(world, radar_transforms):
    """
    Spawn radar sensors at given transforms and keep listeners active.

    Tunable radar blueprint attributes:
    - horizontal_fov
    - vertical_fov
    - range
    - points_per_second
    """
    bp_lib = world.get_blueprint_library()
    radar_bp = bp_lib.find("sensor.other.radar")
    radar_bp.set_attribute("horizontal_fov", "30")
    radar_bp.set_attribute("vertical_fov", "20")
    radar_bp.set_attribute("range", "60")
    radar_bp.set_attribute("points_per_second", "2000")

    radars = []
    for tf in radar_transforms:
        radar = world.try_spawn_actor(radar_bp, tf)
        if radar is None:
            continue
        radar.listen(lambda data: None)
        radars.append(radar)
    return radars


def spawn_poisson_vehicles_on_stretch(world, traffic_manager, waypoints, count):
    """
    Spawn vehicles along a waypoint path using Poisson-like spacing.

    Tunable:
    - count: requested number of vehicles
    - edge_buffer passed into poisson_positions_fixed_count
    - neighborhood search deltas for fallback spawn spots
    - TM following distance / lane-change behavior
    """
    bp_lib = world.get_blueprint_library()
    vehicle_bps = [
        bp
        for bp in bp_lib.filter("vehicle.*")
        if bp.has_attribute("number_of_wheels")
        and int(bp.get_attribute("number_of_wheels").as_int()) == 4
    ]
    if not vehicle_bps:
        raise RuntimeError("No 4-wheel vehicle blueprints found.")

    cum_dists, length_m = cumulative_distances(waypoints)
    # Adjust edge_buffer to keep vehicles further from corridor ends.
    target_positions = poisson_positions_fixed_count(count, length_m, edge_buffer=10.0)

    vehicles = []
    used_wp_indices = set()
    for d in target_positions:
        center_idx = waypoint_index_at_distance(cum_dists, d)
        idx_candidates = [center_idx]
        # Expand this tuple if you want more aggressive nearby fallback attempts.
        for delta in (1, -1, 2, -2, 3, -3, 5, -5):
            idx = center_idx + delta
            if 0 <= idx < len(waypoints):
                idx_candidates.append(idx)

        spawned = False
        for idx in idx_candidates:
            if idx in used_wp_indices:
                continue
            wp = waypoints[idx]
            tf = carla.Transform(
                carla.Location(
                    x=wp.transform.location.x,
                    y=wp.transform.location.y,
                    z=wp.transform.location.z + 0.5,
                ),
                wp.transform.rotation,
            )
            bp = random.choice(vehicle_bps)
            if bp.has_attribute("color"):
                bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))

            vehicle = world.try_spawn_actor(bp, tf)
            if vehicle is None:
                continue

            vehicle.set_autopilot(True, traffic_manager.get_port())
            traffic_manager.auto_lane_change(vehicle, False)
            traffic_manager.ignore_lights_percentage(vehicle, 0.0)
            # Increase/decrease range for tighter or looser platoons.
            traffic_manager.distance_to_leading_vehicle(vehicle, random.uniform(3.0, 6.0))
            vehicles.append(vehicle)
            used_wp_indices.add(idx)
            spawned = True
            break
        if not spawned:
            continue
    return vehicles, length_m


def in_bounds(location, bounds):
    """Check if location falls inside rectangular XY bounds."""
    return (
        bounds["min_x"] <= location.x <= bounds["max_x"]
        and bounds["min_y"] <= location.y <= bounds["max_y"]
    )


def stretch_bounds(waypoints, margin=8.0):
    """
    Create XY bounding box around selected road stretch.

    Tunable:
    - margin: expands box around lane to include sidewalk/navmesh for pedestrians
    """
    xs = [wp.transform.location.x for wp in waypoints]
    ys = [wp.transform.location.y for wp in waypoints]
    return {
        "min_x": min(xs) - margin,
        "max_x": max(xs) + margin,
        "min_y": min(ys) - margin,
        "max_y": max(ys) + margin,
    }


def random_nav_location_in_bounds(world, bounds, z=0.3, attempts=80):
    """
    Sample a random navmesh location constrained to bounds.

    Tunable:
    - z: spawn elevation
    - attempts: retries before giving up
    """
    for _ in range(attempts):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        if in_bounds(loc, bounds):
            loc.z = z
            return loc
    return None


def spawn_pedestrians(world, count, bounds, speed_min=0.9, speed_max=1.7):
    """
    Spawn pedestrians within road-corridor bounds and assign walk targets.

    Tunable:
    - count: requested number of walkers
    - speed_min/speed_max: walking speed range (m/s)
    - nav sampling attempts for spawn/target points
    """
    bp_lib = world.get_blueprint_library()
    walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
    if not walker_bps:
        raise RuntimeError("No pedestrian blueprints found.")
    controller_bp = bp_lib.find("controller.ai.walker")

    walkers = []
    controllers = []
    for _ in range(count):
        spawn_loc = random_nav_location_in_bounds(world, bounds, z=0.3, attempts=120)
        target_loc = random_nav_location_in_bounds(world, bounds, z=0.3, attempts=120)
        if spawn_loc is None or target_loc is None:
            continue

        walker_bp = random.choice(walker_bps)
        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")

        walker = world.try_spawn_actor(walker_bp, carla.Transform(spawn_loc))
        if walker is None:
            continue

        controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
        speed = random.uniform(speed_min, speed_max)
        controller.start()
        controller.set_max_speed(speed)
        controller.go_to_location(target_loc)

        walkers.append(walker)
        controllers.append(controller)

    return walkers, controllers


def main():
    """Entry point: assemble scenario, run for configured time, and clean up."""
    parser = argparse.ArgumentParser(
        description=(
            "Long-road scenario: all-red lights, Poisson vehicle spacing, "
            "camera on traffic light, roadside radars, and pedestrians."
        )
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--tm-port", type=int, default=8000, help="Traffic Manager port")
    # Primary scenario knobs:
    parser.add_argument("--vehicle-count", type=int, default=30, help="Number of vehicles")
    parser.add_argument("--ped-count", type=int, default=10, help="Number of pedestrians")
    parser.add_argument("--radar-spacing", type=float, default=35.0, help="Meters between radar nodes")
    parser.add_argument("--run-seconds", type=float, default=120.0, help="Scenario run time")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    client = carla.Client(args.host, args.port)
    client.set_timeout(12.0)
    world = client.get_world()
    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_global_distance_to_leading_vehicle(3.0)
    traffic_manager.global_percentage_speed_difference(0.0)

    vehicles = []
    walkers = []
    controllers = []
    radars = []
    camera = None
    try:
        # Local corridor used for radar, camera proximity, and pedestrian region.
        # Tune road detection scope here if your map has shorter/longer straight segments.
        waypoints = build_road_stretch_near_spectator(world, step=2.0, max_length=650.0)
        # Dedicated outer perimeter loop used for vehicle spawning.
        perimeter_waypoints = build_outer_perimeter_loop(
            world.get_map(),
            step_m=PERIMETER_STEP_M,
            max_steps=450,
        )
        # Margin controls how far pedestrian sampling can spread from the lane centerline.
        bounds = stretch_bounds(waypoints, margin=10.0)

        # Tune red_seconds if you need shorter red-cycle persistence.
        red_count = set_all_traffic_lights_red(world, red_seconds=10000.0)
        print(f"Traffic lights forced red: {red_count}")

        vehicles, stretch_len = spawn_poisson_vehicles_on_stretch(
            world=world,
            traffic_manager=traffic_manager,
            waypoints=perimeter_waypoints,
            count=args.vehicle_count,
        )
        print(
            f"Spawned vehicles: {len(vehicles)}/{args.vehicle_count} "
            f"on outer perimeter (~{stretch_len:.1f} m path)"
        )

        center_loc = waypoints[len(waypoints) // 2].transform.location
        tl = nearest_traffic_light(world, center_loc)
        if tl is not None:
            camera = spawn_camera_on_light(world, tl)
            camera.listen(lambda image: None)
            print(f"Camera attached to traffic light id={tl.id}")
        else:
            print("No traffic light found near stretch center; camera not spawned.")

        radar_tfs = build_radar_transforms(
            waypoints=waypoints,
            spacing_m=args.radar_spacing,
            # Roadside radar mounting geometry:
            side_offset_m=4.5,
            height_m=4.0,
        )
        radars = spawn_radars(world, radar_tfs)
        print(f"Spawned radar nodes: {len(radars)}")

        # Pedestrian speed range can also be changed by editing function defaults.
        walkers, controllers = spawn_pedestrians(world, args.ped_count, bounds)
        print(f"Spawned pedestrians: {len(walkers)}/{args.ped_count}")

        end_time = time.time() + args.run_seconds
        while time.time() < end_time:
            # Keep lights locked red even if other scripts alter state.
            if int(end_time - time.time()) % 5 == 0:
                for light in world.get_actors().filter("traffic.traffic_light*"):
                    light.set_state(carla.TrafficLightState.Red)
                    light.freeze(True)
            time.sleep(0.2)

    finally:
        if camera is not None and camera.is_alive:
            camera.stop()
            camera.destroy()
        for radar in radars:
            if radar.is_alive:
                radar.stop()
                radar.destroy()
        for controller in controllers:
            if controller.is_alive:
                controller.stop()
        for actor in controllers + walkers + vehicles:
            if actor.is_alive:
                actor.destroy()
        # Unfreeze lights so world is not left permanently modified.
        for light in world.get_actors().filter("traffic.traffic_light*"):
            light.freeze(False)
        print(
            "Cleaned up actors: "
            f"vehicles={len(vehicles)}, walkers={len(walkers)}, "
            f"controllers={len(controllers)}, radars={len(radars)}"
        )


if __name__ == "__main__":
    main()
