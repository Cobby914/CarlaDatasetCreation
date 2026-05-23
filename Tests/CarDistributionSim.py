import argparse
import math
import random
import time

import carla


def count_total_vehicles(world):
    return len(world.get_actors().filter("vehicle.*"))


def weighted_shuffle(items, weights):
    # Weighted random ordering without replacement.
    # Higher weight means item appears earlier on average.
    keys = []
    for item, weight in zip(items, weights):
        safe_weight = max(weight, 1e-6) 
        key = random.random() ** (1.0 / safe_weight)
        keys.append((key, item))
    keys.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in keys]


def compute_map_center(spawn_points):
    avg_x = sum(t.location.x for t in spawn_points) / len(spawn_points)
    avg_y = sum(t.location.y for t in spawn_points) / len(spawn_points)
    return avg_x, avg_y


def build_spawn_list(world_map, mode, args):
    spawn_points = world_map.get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points found in current map.")

    if mode == "uniform":
        random.shuffle(spawn_points)
        return spawn_points

    if mode == "gaussian":
        if args.center_x is None or args.center_y is None:
            center_x, center_y = compute_map_center(spawn_points)
        else:
            center_x, center_y = args.center_x, args.center_y

        sigma = max(args.sigma, 1e-3)
        two_sigma_sq = 2.0 * sigma * sigma
        weights = []
        for transform in spawn_points:
            dx = transform.location.x - center_x
            dy = transform.location.y - center_y
            distance_sq = dx * dx + dy * dy
            weight = math.exp(-distance_sq / two_sigma_sq)
            weights.append(weight)
        return weighted_shuffle(spawn_points, weights)

    if mode == "fourier":
        weights = []
        for transform in spawn_points:
            x = transform.location.x
            y = transform.location.y
            phase_value = (args.kx * x) + (args.ky * y) + args.phase
            signal = math.sin(phase_value)
            weight = args.offset + (args.amplitude * signal)
            weights.append(max(weight, 1e-3))
        return weighted_shuffle(spawn_points, weights)

    if mode == "linear":
        weights = []
        for transform in spawn_points:
            x = transform.location.x
            y = transform.location.y
            linear_value = (args.linear_ax * x) + (args.linear_ay * y) + args.linear_bias
            if args.linear_abs:
                linear_value = abs(linear_value)
            weights.append(max(linear_value, 1e-3))
        return weighted_shuffle(spawn_points, weights)

    # Heuristic: points near intersections are ranked first.
    ranked = []
    for transform in spawn_points:
        wp = world_map.get_waypoint(transform.location, project_to_road=True)
        rank = 0 if wp.is_junction else 1
        ranked.append((rank, random.random(), transform))

    ranked.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ranked]


def try_spawn_vehicles(world, traffic_manager, count, mode, safe_mode, args):
    blueprints = list(world.get_blueprint_library().filter("vehicle.*"))
    if safe_mode:
        # These generally avoid odd edge-case actors.
        blueprints = [bp for bp in blueprints if bp.has_attribute("number_of_wheels")]
        blueprints = [
            bp for bp in blueprints if int(bp.get_attribute("number_of_wheels").as_int()) == 4
        ]

    if not blueprints:
        raise RuntimeError("No vehicle blueprints found for spawning.")

    spawn_points = build_spawn_list(world.get_map(), mode, args)
    random.shuffle(blueprints)
    vehicles = []
    print(f"Spawn progress: 0/{count}")

    for spawn_point in spawn_points:
        if len(vehicles) >= count:
            break

        bp = random.choice(blueprints)
        if bp.has_attribute("color"):
            color = random.choice(bp.get_attribute("color").recommended_values)
            bp.set_attribute("color", color)
        if bp.has_attribute("driver_id"):
            driver_id = random.choice(bp.get_attribute("driver_id").recommended_values)
            bp.set_attribute("driver_id", driver_id)

        vehicle = world.try_spawn_actor(bp, spawn_point)
        if vehicle is None:
            continue

        vehicle.set_autopilot(True, traffic_manager.get_port())
        traffic_manager.auto_lane_change(vehicle, True)
        traffic_manager.distance_to_leading_vehicle(vehicle, random.uniform(2.5, 5.0))
        vehicles.append(vehicle)
        print(f"Spawn progress: {len(vehicles)}/{count}")
        if args.spawn_interval > 0:
            time.sleep(args.spawn_interval)

    return vehicles


def main():
    parser = argparse.ArgumentParser(
        description="Spawn vehicles with configurable spatial distribution."
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument(
        "--tm-port", type=int, default=8000, help="Traffic Manager port"
    )
    parser.add_argument(
        "--count", type=int, default=40, help="Number of vehicles to spawn"
    )
    parser.add_argument(
        "--mode",
        choices=["uniform", "junction-biased", "gaussian", "fourier", "linear"],
        default="uniform",
        help="How spawn points are selected",
    )
    parser.add_argument(
        "--center-x",
        type=float,
        default=None,
        help="Gaussian center X (defaults to map center if omitted)",
    )
    parser.add_argument(
        "--center-y",
        type=float,
        default=None,
        help="Gaussian center Y (defaults to map center if omitted)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=50.0,
        help="Gaussian spread in meters (used only with --mode gaussian)",
    )
    parser.add_argument(
        "--kx",
        type=float,
        default=0.05,
        help="Fourier wave number for X (used only with --mode fourier)",
    )
    parser.add_argument(
        "--ky",
        type=float,
        default=0.05,
        help="Fourier wave number for Y (used only with --mode fourier)",
    )
    parser.add_argument(
        "--phase",
        type=float,
        default=0.0,
        help="Fourier phase offset in radians (used only with --mode fourier)",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.8,
        help="Fourier amplitude (used only with --mode fourier)",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=1.0,
        help="Fourier baseline offset (used only with --mode fourier)",
    )
    parser.add_argument(
        "--linear-ax",
        type=float,
        default=1.0,
        help="Linear X coefficient (used only with --mode linear)",
    )
    parser.add_argument(
        "--linear-ay",
        type=float,
        default=0.0,
        help="Linear Y coefficient (used only with --mode linear)",
    )
    parser.add_argument(
        "--linear-bias",
        type=float,
        default=1.0,
        help="Linear bias/intercept (used only with --mode linear)",
    )
    parser.add_argument(
        "--linear-abs",
        action="store_true",
        help="Use absolute value for linear weights (used only with --mode linear)",
    )
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=30.0,
        help="How long to keep simulation running after spawn",
    )
    parser.add_argument(
        "--safe-mode",
        action="store_true",
        help="Spawn only standard 4-wheel vehicles",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducible spawns"
    )
    parser.add_argument(
        "--spawn-interval",
        type=float,
        default=0.0,
        help="Seconds to wait after each successful spawn (for visible ramp-up)",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.global_percentage_speed_difference(0.0)

    vehicles = []
    try:
        vehicles = try_spawn_vehicles(
            world=world,
            traffic_manager=traffic_manager,
            count=args.count,
            mode=args.mode,
            safe_mode=args.safe_mode,
            args=args,
        )
        print(
            f"Spawned {len(vehicles)} vehicles "
            f"(requested {args.count}, mode={args.mode})."
        )
        print(f"Total vehicles in world now: {count_total_vehicles(world)}")

        # Keep script alive so Traffic Manager controls active vehicles.
        end_time = time.time() + args.run_seconds
        next_report_time = time.time()
        while time.time() < end_time:
            now = time.time()
            if now >= next_report_time:
                alive_from_script = sum(1 for vehicle in vehicles if vehicle.is_alive)
                total_vehicles = count_total_vehicles(world)
                print(
                    "Vehicle count | "
                    f"spawned_by_script_alive={alive_from_script}, "
                    f"total_in_world={total_vehicles}"
                )
                next_report_time = now + 1.0
            time.sleep(0.2)

    finally:
        for vehicle in vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
        print(f"Destroyed {len(vehicles)} vehicles.")


if __name__ == "__main__":
    main()
