import argparse
import math
import random
import time

import carla


ROAD_LENGTH_M = 50.0


def boundary_distance(location, min_x, max_x, min_y, max_y):
    return min(
        abs(location.x - min_x),
        abs(location.x - max_x),
        abs(location.y - min_y),
        abs(location.y - max_y),
    )


def yaw_delta_deg(target_yaw, current_yaw):
    delta = (target_yaw - current_yaw + 180.0) % 360.0 - 180.0
    return delta


def pick_perimeter_start(waypoints):
    return min(waypoints, key=lambda wp: (wp.transform.location.x + wp.transform.location.y))


def build_perimeter_loop(world_map, max_steps=400):
    all_waypoints = world_map.generate_waypoints(ROAD_LENGTH_M)
    driving_waypoints = [
        wp for wp in all_waypoints
        if wp.lane_type == carla.LaneType.Driving and wp.is_junction is False
    ]
    if not driving_waypoints:
        raise RuntimeError("No driving waypoints found in current map.")

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
        candidates = current.next(ROAD_LENGTH_M)
        if not candidates:
            break

        # Bias toward waypoints nearest outer map boundaries, while preferring
        # gentler heading changes to avoid oscillations at junction-like areas.
        current_yaw = current.transform.rotation.yaw
        scored = []
        for candidate in candidates:
            if candidate.lane_type != carla.LaneType.Driving:
                continue
            candidate_loc = candidate.transform.location
            edge_dist = boundary_distance(candidate_loc, min_x, max_x, min_y, max_y)
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
            "Could not build a usable perimeter route. Try a larger map or adjust logic."
        )

    return loop


def spawn_vehicles_on_loop(world, vehicle_count, loop_waypoints):
    blueprints = [
        bp
        for bp in world.get_blueprint_library().filter("vehicle.*")
        if bp.has_attribute("number_of_wheels")
        and int(bp.get_attribute("number_of_wheels").as_int()) == 4
    ]
    if not blueprints:
        raise RuntimeError("No 4-wheel vehicle blueprints available.")

    vehicles = []
    loop_size = len(loop_waypoints)
    step = max(1, loop_size // max(1, vehicle_count))

    for index in range(vehicle_count):
        wp = loop_waypoints[(index * step) % loop_size]
        transform = wp.transform
        transform.location.z += 0.5

        blueprint = random.choice(blueprints)
        if blueprint.has_attribute("color"):
            color = random.choice(blueprint.get_attribute("color").recommended_values)
            blueprint.set_attribute("color", color)

        vehicle = world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            vehicles.append(vehicle)

    return vehicles


def configure_traffic_manager_for_perimeter(
    traffic_manager, vehicles, loop_waypoints, desired_speed_kmh
):
    path_locations = [wp.transform.location for wp in loop_waypoints]
    speed_diff_pct = 0.0
    if desired_speed_kmh > 0.0:
        # TM uses percentage speed difference from speed limit:
        # positive => slower than limit, negative => faster than limit.
        speed_limit_guess = 50.0
        speed_diff_pct = ((speed_limit_guess - desired_speed_kmh) / speed_limit_guess) * 100.0

    for vehicle in vehicles:
        vehicle.set_autopilot(True, traffic_manager.get_port())
        traffic_manager.auto_lane_change(vehicle, False)
        traffic_manager.ignore_lights_percentage(vehicle, 0.0)
        traffic_manager.ignore_signs_percentage(vehicle, 0.0)
        traffic_manager.ignore_vehicles_percentage(vehicle, 0.0)
        traffic_manager.distance_to_leading_vehicle(vehicle, 5.0)
        traffic_manager.vehicle_percentage_speed_difference(vehicle, speed_diff_pct)

        # Keep each vehicle on the perimeter route when the API is available.
        if hasattr(traffic_manager, "set_path"):
            traffic_manager.set_path(vehicle, path_locations)


def run_perimeter_drive(vehicles, run_seconds):
    if not vehicles:
        print("No vehicles spawned.")
        return

    end_time = time.time() + run_seconds
    while time.time() < end_time:
        time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser(
        description="Drive vehicles only on the outside perimeter using 50m road segments."
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--tm-port", type=int, default=8000, help="Traffic Manager port")
    parser.add_argument("--count", type=int, default=12, help="Number of vehicles to spawn")
    parser.add_argument("--run-seconds", type=float, default=60.0, help="Simulation runtime")
    parser.add_argument(
        "--speed-kmh",
        type=float,
        default=35.0,
        help="Target speed in km/h for perimeter loop driving",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    traffic_manager = client.get_trafficmanager(args.tm_port)

    vehicles = []
    try:
        print(f"Building perimeter route with road length {ROAD_LENGTH_M:.0f}m...")
        perimeter_loop = build_perimeter_loop(world.get_map())
        print(f"Perimeter route points: {len(perimeter_loop)}")

        vehicles = spawn_vehicles_on_loop(world, args.count, perimeter_loop)
        print(f"Spawned {len(vehicles)} vehicles on perimeter.")
        configure_traffic_manager_for_perimeter(
            traffic_manager=traffic_manager,
            vehicles=vehicles,
            loop_waypoints=perimeter_loop,
            desired_speed_kmh=args.speed_kmh,
        )

        run_perimeter_drive(vehicles=vehicles, run_seconds=args.run_seconds)
    finally:
        for vehicle in vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
        print(f"Destroyed {len(vehicles)} vehicles.")


if __name__ == "__main__":
    main()
