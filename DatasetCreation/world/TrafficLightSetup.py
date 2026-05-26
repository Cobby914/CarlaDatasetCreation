import importlib.util
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dc_entry", _root / "_entry.py")
_dc_entry = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dc_entry)
_dc_entry.bootstrap(__file__)

import os
import time

import carla

from carla_connect import get_world


PERIMETER_MARGIN_M = 25.0
TURN_ANGLE_THRESHOLD_DEG = 35.0
TURN_LOOKAHEAD_M = 12.0
# Anchor near CAM C10 from your dataset setup.
C10_ANCHOR_LOCATION = carla.Location(x=-45.741562, y=-68.056618, z=0.0)
LABEL_REFRESH_S = 0.25
LABEL_LIFETIME_S = 1.0
LABEL_Z_OFFSET_M = 2.5
# Standard phase lengths so unfrozen lights actually cycle in CARLA.
DEFAULT_GREEN_TIME_S = 12.0
DEFAULT_YELLOW_TIME_S = 2.0
DEFAULT_RED_TIME_S = 12.0


def yaw_delta_deg(target_yaw, current_yaw):
    return (target_yaw - current_yaw + 180.0) % 360.0 - 180.0


def get_driving_bounds(world_map, waypoint_step=8.0):
    waypoints = world_map.generate_waypoints(waypoint_step)
    if not waypoints:
        raise RuntimeError("No driving waypoints found; cannot compute map bounds.")

    xs = [wp.transform.location.x for wp in waypoints]
    ys = [wp.transform.location.y for wp in waypoints]
    return min(xs), max(xs), min(ys), max(ys)


def is_perimeter_light(light, min_x, max_x, min_y, max_y, margin_m):
    """
    Treat a light as perimeter if any affected lane lies close to the map edge.

    If CARLA returns no stop waypoints for this light, do not classify as inner (that would force
    red on every such light and effectively kill the junction network).
    """
    stop_wps = light.get_stop_waypoints()
    if not stop_wps:
        return True

    for wp in stop_wps:
        loc = wp.transform.location
        if (
            loc.x <= (min_x + margin_m)
            or loc.x >= (max_x - margin_m)
            or loc.y <= (min_y + margin_m)
            or loc.y >= (max_y - margin_m)
        ):
            return True
    return False


def light_controls_turning_movement(
    light,
    angle_threshold_deg=TURN_ANGLE_THRESHOLD_DEG,
    lookahead_m=TURN_LOOKAHEAD_M,
):
    """
    Return True if any stop waypoint controlled by this light diverges into a turn.
    """
    stop_wps = light.get_stop_waypoints()
    for stop_wp in stop_wps:
        # At/after the stop line inside junction, next() often contains route branches.
        for next_wp in stop_wp.next(lookahead_m):
            delta = abs(yaw_delta_deg(next_wp.transform.rotation.yaw, stop_wp.transform.rotation.yaw))
            if angle_threshold_deg <= delta <= (180.0 - angle_threshold_deg):
                return True
    return False


def distance_sq(loc_a, loc_b):
    dx = loc_a.x - loc_b.x
    dy = loc_a.y - loc_b.y
    dz = loc_a.z - loc_b.z
    return dx * dx + dy * dy + dz * dz


def reset_light_phase_times(
    light,
    *,
    green_s=DEFAULT_GREEN_TIME_S,
    yellow_s=DEFAULT_YELLOW_TIME_S,
    red_s=DEFAULT_RED_TIME_S,
):
    light.set_green_time(green_s)
    light.set_yellow_time(yellow_s)
    light.set_red_time(red_s)


def set_light_near_location_always_green(lights, anchor_location):
    if not lights:
        return None

    selected = min(lights, key=lambda light: distance_sq(light.get_location(), anchor_location))
    selected.set_state(carla.TrafficLightState.Green)
    selected.set_green_time(99999.0)
    selected.set_yellow_time(0.1)
    selected.set_red_time(0.1)
    selected.freeze(True)
    return selected


def color_for_light_state(state):
    if state == carla.TrafficLightState.Red:
        return carla.Color(255, 80, 80)
    if state == carla.TrafficLightState.Yellow:
        return carla.Color(255, 220, 0)
    if state == carla.TrafficLightState.Green:
        return carla.Color(80, 255, 80)
    return carla.Color(180, 180, 180)


def label_all_traffic_lights(world, always_green_light_id=None):
    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    for light in lights:
        state = light.get_state()
        extra = " | ALWAYS_GREEN" if light.id == always_green_light_id else ""
        world.debug.draw_string(
            light.get_location() + carla.Location(z=LABEL_Z_OFFSET_M),
            f"TL {light.id} | {state}{extra}",
            draw_shadow=False,
            color=color_for_light_state(state),
            life_time=LABEL_LIFETIME_S,
            persistent_lines=False,
        )
    return len(lights)


def automatic_traffic_lights_from_env() -> bool:
    """When true, unfreeze all lights. Default off — legacy perimeter rules work better in Town10HD."""
    raw = os.environ.get("DATASET_AUTOMATIC_TRAFFIC_LIGHTS", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def configure_traffic_lights_free_automatic(world):
    """
    Let CARLA run the traffic-light state machine (unfreeze all lights).
    Optionally keeps one light near the dataset anchor always green for corridor flow.
    """
    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    if not lights:
        print("No traffic lights found.")
        return None

    for light in lights:
        reset_light_phase_times(light)
        light.freeze(False)

    always_green_light = set_light_near_location_always_green(lights, C10_ANCHOR_LOCATION)
    print(
        f"Automatic traffic lights: unfroze {len(lights)} lights "
        "(CARLA cycles red/yellow/green)."
    )
    if always_green_light is not None:
        light_loc = always_green_light.get_location()
        print(
            "Dataset corridor override (always green):",
            f"id={always_green_light.id} "
            f"at ({light_loc.x:.2f}, {light_loc.y:.2f}, {light_loc.z:.2f})",
        )
        return always_green_light.id
    return None


def configure_traffic_lights(world, margin_m=PERIMETER_MARGIN_M):
    world_map = world.get_map()
    min_x, max_x, min_y, max_y = get_driving_bounds(world_map)

    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    if not lights:
        print("No traffic lights found.")
        return

    perimeter_lights = []
    perimeter_turn_lights = []
    forced_red_lights = []

    for light in lights:
        is_perimeter = is_perimeter_light(light, min_x, max_x, min_y, max_y, margin_m)
        is_turn_light = light_controls_turning_movement(light)

        if is_perimeter and not is_turn_light:
            perimeter_lights.append(light)
            # Let perimeter lights run normally with sane phase times.
            reset_light_phase_times(light)
            light.freeze(False)
        else:
            if is_perimeter and is_turn_light:
                perimeter_turn_lights.append(light)
            forced_red_lights.append(light)
            light.set_state(carla.TrafficLightState.Red)
            light.set_red_time(99999.0)
            light.set_yellow_time(0.1)
            light.set_green_time(0.1)
            light.freeze(True)

    print(
        f"Traffic lights total={len(lights)} | "
        f"perimeter_straight={len(perimeter_lights)} | "
        f"perimeter_turn_forced_red={len(perimeter_turn_lights)} | "
        f"forced_red_total={len(forced_red_lights)}"
    )
    print(f"Perimeter margin used: {margin_m:.1f} m")
    print(
        f"Turn detection: threshold={TURN_ANGLE_THRESHOLD_DEG:.1f} deg, "
        f"lookahead={TURN_LOOKAHEAD_M:.1f} m"
    )
    if perimeter_lights:
        print("Perimeter straight light IDs:", ", ".join(str(light.id) for light in perimeter_lights))
    if perimeter_turn_lights:
        print(
            "Perimeter turn light IDs (forced red):",
            ", ".join(str(light.id) for light in perimeter_turn_lights),
        )

    always_green_light = set_light_near_location_always_green(lights, C10_ANCHOR_LOCATION)
    if always_green_light is not None:
        light_loc = always_green_light.get_location()
        print(
            "Always-green override light:",
            f"id={always_green_light.id} "
            f"at ({light_loc.x:.2f}, {light_loc.y:.2f}, {light_loc.z:.2f})",
        )
        return always_green_light.id
    return None


def main():
    _, world = get_world()

    if automatic_traffic_lights_from_env():
        always_green_light_id = configure_traffic_lights_free_automatic(world)
    else:
        always_green_light_id = configure_traffic_lights(
            world=world, margin_m=PERIMETER_MARGIN_M
        )

    print("Drawing traffic light labels. Press Ctrl+C to stop.")
    try:
        while True:
            count = label_all_traffic_lights(world, always_green_light_id=always_green_light_id)
            print(f"Labeled traffic lights: {count}", end="\r", flush=True)
            time.sleep(LABEL_REFRESH_S)
    except KeyboardInterrupt:
        print("\nStopped traffic light labeling.")


if __name__ == "__main__":
    main()
