import argparse
import random
import time

import carla


def parse_xyz(text, arg_name):
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"{arg_name} must be 'x,y,z'")
    return float(parts[0]), float(parts[1]), float(parts[2])


def parse_box(text):
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 5:
        raise ValueError("--spawn-box/--target-box must be 'xmin,xmax,ymin,ymax,z'")
    xmin, xmax, ymin, ymax, z = [float(part) for part in parts]
    if xmin > xmax or ymin > ymax:
        raise ValueError("Box bounds must satisfy xmin<=xmax and ymin<=ymax")
    return xmin, xmax, ymin, ymax, z


def parse_route(text):
    if ":" not in text:
        raise ValueError("--route must be 'sx,sy,sz:tx,ty,tz'")
    spawn_text, target_text = text.split(":", 1)
    spawn_xyz = parse_xyz(spawn_text, "--route spawn")
    target_xyz = parse_xyz(target_text, "--route target")
    return spawn_xyz, target_xyz


def in_box(location, box):
    xmin, xmax, ymin, ymax, _ = box
    return xmin <= location.x <= xmax and ymin <= location.y <= ymax


def random_nav_location(world, box=None, attempts=80):
    for _ in range(attempts):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        if box is None or in_box(loc, box):
            if box is not None:
                loc.z = box[4]
            return loc
    return None


def spawn_pedestrian(world, walker_bp, controller_bp, spawn_location, target_location, speed):
    walker_tf = carla.Transform(spawn_location, carla.Rotation())
    walker = world.try_spawn_actor(walker_bp, walker_tf)
    if walker is None:
        return None, None

    controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
    controller.start()
    controller.set_max_speed(speed)
    controller.go_to_location(target_location)
    return walker, controller


def main():
    # Command-line controls:
    # - Use --route for exact spawn->target pairs.
    # - Use --spawn-box/--target-box for random walkers in bounded areas.
    parser = argparse.ArgumentParser(
        description="Spawn and control pedestrians with fixed routes or random navigation."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="CARLA server host/IP (e.g. 127.0.0.1 or localhost).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
        help="CARLA server TCP port (usually 2000).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Client API timeout in seconds. Must be > 0.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Total walkers to control. Recommended >= 1.",
    )
    parser.add_argument(
        "--route",
        action="append",
        default=[],
        help=(
            "Fixed route: 'sx,sy,sz:tx,ty,tz'. "
            "Can be repeated. If routes exceed --count, all routes still spawn."
        ),
    )
    parser.add_argument(
        "--spawn-box",
        default=None,
        help=(
            "Random spawn area for extra walkers: 'xmin,xmax,ymin,ymax,z'. "
            "Use world coordinates. Requires xmin<=xmax and ymin<=ymax."
        ),
    )
    parser.add_argument(
        "--target-box",
        default=None,
        help=(
            "Random destination area: 'xmin,xmax,ymin,ymax,z'. "
            "If omitted, random targets can be anywhere on nav mesh."
        ),
    )
    parser.add_argument(
        "--retarget-seconds",
        type=float,
        default=0.0,
        help=(
            "Retarget interval for random walkers. "
            "0 disables retargeting. Values > 0 retarget every N seconds."
        ),
    )
    parser.add_argument(
        "--speed-min",
        type=float,
        default=1.0,
        help="Minimum walker speed in m/s. Must be > 0 and <= --speed-max.",
    )
    parser.add_argument(
        "--speed-max",
        type=float,
        default=1.8,
        help="Maximum walker speed in m/s. Must be > 0 and >= --speed-min.",
    )
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=30.0,
        help="How long the script keeps walkers alive. Recommended > 0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible random walker placement/speeds.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.speed_min <= 0 or args.speed_max <= 0:
        raise ValueError("Speed values must be positive.")
    if args.speed_min > args.speed_max:
        raise ValueError("--speed-min must be <= --speed-max")

    # Parsed control values used throughout the script.
    # routes: list of ((spawn_x,spawn_y,spawn_z), (target_x,target_y,target_z))
    # spawn_box/target_box: (xmin, xmax, ymin, ymax, z) or None.
    routes = [parse_route(route_text) for route_text in args.route]
    spawn_box = parse_box(args.spawn_box) if args.spawn_box else None
    target_box = parse_box(args.target_box) if args.target_box else None

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    walker_blueprints = list(bp_lib.filter("walker.pedestrian.*"))
    if not walker_blueprints:
        raise RuntimeError("No walker blueprints found.")
    controller_bp = bp_lib.find("controller.ai.walker")

    walkers = []
    controllers = []
    random_controller_indices = []

    try:
        # total_to_spawn controls final pedestrian count:
        # - At least --count
        # - Also at least number of explicit --route entries
        total_to_spawn = max(args.count, len(routes))
        print(f"Spawning up to {total_to_spawn} pedestrians...")

        # 1) Spawn all fixed routes first.
        for route_index, (spawn_xyz, target_xyz) in enumerate(routes):
            if len(walkers) >= total_to_spawn:
                break
            spawn_loc = carla.Location(*spawn_xyz)
            target_loc = carla.Location(*target_xyz)
            speed = random.uniform(args.speed_min, args.speed_max)
            walker_bp = random.choice(walker_blueprints)

            # Keep walkers mortal to avoid odd interactions when testing collisions.
            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")

            walker, controller = spawn_pedestrian(
                world, walker_bp, controller_bp, spawn_loc, target_loc, speed
            )
            if walker is None:
                print(f"[route {route_index}] spawn failed at {spawn_xyz}")
                continue

            walkers.append(walker)
            controllers.append(controller)
            print(
                f"[route {route_index}] walker_id={walker.id} "
                f"spawn={spawn_xyz} target={target_xyz} speed={speed:.2f}"
            )

        # 2) Fill remaining walkers with random nav spawn + random destination.
        while len(walkers) < total_to_spawn:
            spawn_loc = random_nav_location(world, spawn_box)
            if spawn_loc is None:
                print("No valid random spawn location found in nav mesh.")
                break

            target_loc = random_nav_location(world, target_box)
            if target_loc is None:
                print("No valid random target location found in nav mesh.")
                break

            speed = random.uniform(args.speed_min, args.speed_max)
            walker_bp = random.choice(walker_blueprints)
            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")

            walker, controller = spawn_pedestrian(
                world, walker_bp, controller_bp, spawn_loc, target_loc, speed
            )
            if walker is None:
                continue

            walkers.append(walker)
            controllers.append(controller)
            random_controller_indices.append(len(controllers) - 1)
            print(
                f"[random] walker_id={walker.id} "
                f"spawn=({spawn_loc.x:.1f},{spawn_loc.y:.1f},{spawn_loc.z:.1f}) "
                f"target=({target_loc.x:.1f},{target_loc.y:.1f},{target_loc.z:.1f}) "
                f"speed={speed:.2f}"
            )

        print(f"Spawned {len(walkers)} pedestrians.")

        end_time = time.time() + args.run_seconds
        next_retarget = time.time() + args.retarget_seconds if args.retarget_seconds > 0 else None

        while time.time() < end_time:
            now = time.time()
            if (
                next_retarget is not None
                and now >= next_retarget
                and len(random_controller_indices) > 0
            ):
                for index in random_controller_indices:
                    if index >= len(controllers):
                        continue
                    controller = controllers[index]
                    if not controller.is_alive:
                        continue
                    new_target = random_nav_location(world, target_box)
                    if new_target is not None:
                        controller.go_to_location(new_target)
                next_retarget = now + args.retarget_seconds

            time.sleep(0.2)

    finally:
        for controller in controllers:
            if controller.is_alive:
                controller.stop()
        for actor in controllers + walkers:
            if actor.is_alive:
                actor.destroy()
        print(f"Destroyed {len(walkers)} walkers and {len(controllers)} controllers.")


if __name__ == "__main__":
    main()
