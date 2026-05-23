import csv
import datetime
import math
import os
import sys
import threading
import traceback
from pathlib import Path
import time

import carla
import msvcrt

NEARBY_DISTANCE_M = 35.0
# Default radar sensor limits (overridden per actor when attributes are present).
RADAR_MAX_RANGE_M = 35.0
RADAR_HORIZONTAL_FOV_DEG = 120.0
# CARLA default points_per_second is 1500; raise for denser returns (CPU cost scales up).
RADAR_POINTS_PER_SECOND_DEFAULT = 4500
# 0.0 = emit every simulation step (fastest); >0 throttles callback rate.
RADAR_SENSOR_TICK_S = 0.0
# Extra range beyond reported depth when building per-detection candidates.
RADAR_CANDIDATE_DEPTH_MARGIN_M = 3.0
# Extra horizontal tolerance (deg) for beam vs actor bearing / OBB angular width.
RADAR_CANDIDATE_AZIMUTH_MARGIN_DEG = 8.0
# Pre-filter: actor must be within this OBB margin (m) of the hit to count as a candidate.
# Rejects beam-only FPs (road return + car in same direction). None = beam gate only.
RADAR_CANDIDATE_HIT_MAX_BBOX_MARGIN_M = 5.0
# Legacy wide bubble (reports only).
RADAR_ACTOR_PROXIMITY_M = 40.0
RADAR_VEHICLE_PROXIMITY_M = RADAR_ACTOR_PROXIMITY_M
# Inflate each actor OBB extent when computing margin (m per axis).
BBOX_MATCH_EXTENT_INFLATION_M = 0.75
# Max distance from hit to OBB surface for a primary match (m).
RADAR_HIT_MATCH_MAX_MARGIN_M = 2.0
# Looser margin when exactly one actor is in the depth/azimuth gate.
RADAR_SINGLE_CANDIDATE_MAX_MARGIN_M = 4.0
# Backward-compatible alias for reports / CLI (near-surface threshold, not extent inflation).
RADAR_HIT_MATCH_MAX_DISTANCE_M = RADAR_HIT_MATCH_MAX_MARGIN_M
# Min |radial velocity| (m/s) to score a return. Default 0 includes parked/stalled actors.
# Set > 0 (e.g. 0.5) to exclude near-static clutter from match stats.
RADAR_LABELABLE_MIN_SPEED_MPS = 0.0
# CARLA does not simulate electromagnetic RCS; `rcs_proxy_m2` is a geometric OBB silhouette.
SENSOR_WAIT_TIMEOUT_S = 30.0
SENSOR_WAIT_POLL_S = 0.5
DATASET_RADAR_ROLE_PREFIX = "dataset_radar_"
DATASET_CAMERA_ROLE_PREFIX = "dataset_camera_"


def _expected_radar_count_from_env() -> int:
    raw = os.environ.get("DATASET_EXPECTED_RADAR_COUNT", "12")
    try:
        n = int(raw)
    except ValueError:
        return 12
    return max(1, min(n, 64))


EXPECTED_RADAR_LABELS = {f"R{i}" for i in range(1, _expected_radar_count_from_env() + 1)}


def vehicle_class_from_type_id(type_id):
    type_lower = type_id.lower()
    if any(token in type_lower for token in ("firetruck", "ambulance", "truck")):
        return "truck"
    if "bus" in type_lower:
        return "bus"
    if any(token in type_lower for token in ("motorcycle", "vespa", "yamaha", "kawasaki", "harley")):
        return "motorcycle"
    if any(token in type_lower for token in ("bicycle", "bike", "crossbike")):
        return "bicycle"
    if "van" in type_lower:
        return "van"
    return "car"


def make_output_paths(base_dir):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"sensor_capture_{timestamp}")
    camera_dir = os.path.join(run_dir, "camera_frames")
    os.makedirs(camera_dir, exist_ok=True)
    radar_csv = os.path.join(run_dir, "radar_data.csv")
    camera_csv = os.path.join(run_dir, "camera_data.csv")
    return run_dir, camera_dir, radar_csv, camera_csv


def setup_radar_writer(path):
    """RadarDetection fields + pose; actor match via OBB margin; rcs_proxy_m2 from OBB geometry."""
    file_handle = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(file_handle)
    writer.writerow(
        [
            "sensor_id",
            "sensor_label",
            "frame",
            "timestamp",
            "detection_index",
            "depth_m",
            "azimuth_rad",
            "altitude_rad",
            "velocity_mps",
            "sensor_world_x_m",
            "sensor_world_y_m",
            "sensor_world_z_m",
            "sensor_pitch_deg",
            "sensor_yaw_deg",
            "sensor_roll_deg",
            "matched_actor_id",
            "matched_actor_kind",
            "matched_actor_type_id",
            "matched_actor_class",
            "matched_actor_bbox_margin_m",
            "matched_vehicle_id",
            "matched_vehicle_type_id",
            "matched_vehicle_class",
            "matched_vehicle_distance_m",
            "rcs_proxy_m2",
            "had_actor_candidates",
            "label_scored",
            "nearest_actor_bbox_margin_m",
        ]
    )
    return file_handle, writer


def setup_camera_writer(path):
    file_handle = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(file_handle)
    writer.writerow(
        [
            "sensor_id",
            "sensor_label",
            "frame",
            "timestamp",
            "width",
            "height",
            "image_path",
            "nearest_actor_id",
            "nearest_actor_kind",
            "nearest_actor_type_id",
            "nearest_actor_class",
            "nearest_actor_distance_m",
            "nearby_actor_ids",
            "nearby_actor_kinds",
            "nearby_actor_classes",
            "nearest_vehicle_id",
            "nearest_vehicle_type_id",
            "nearest_vehicle_class",
            "nearest_vehicle_distance_m",
            "nearby_vehicle_ids",
            "nearby_vehicle_classes",
            "nearest_pedestrian_id",
            "nearest_pedestrian_type_id",
            "nearest_pedestrian_class",
            "nearest_pedestrian_distance_m",
            "nearby_pedestrian_ids",
            "nearby_pedestrian_classes",
        ]
    )
    return file_handle, writer


def sensor_label_from_role_name(role_name, prefix):
    if role_name.startswith(prefix):
        return role_name[len(prefix) :]
    return ""


def list_radar_actors(world):
    """All radar sensors in the world (robust type_id match across CARLA builds)."""
    return [a for a in world.get_actors() if "sensor.other.radar" in a.type_id]


def filter_tagged_sensors(world, actor_pattern, role_prefix, allowed_labels=None):
    filtered = []
    if actor_pattern == "sensor.other.radar":
        actors = list_radar_actors(world)
    else:
        actors = world.get_actors().filter(actor_pattern)
    for actor in actors:
        role_name = actor.attributes.get("role_name", "")
        if not role_name.startswith(role_prefix):
            continue
        if allowed_labels is not None:
            label = sensor_label_from_role_name(role_name, role_prefix)
            if label not in allowed_labels:
                continue
        filtered.append(actor)
    return filtered


def select_one_sensor_per_label(sensors, role_prefix, allowed_labels):
    """
    When multiple actors share the same role label (e.g. leftover radars from a prior run),
    keep the newest actor id per label.
    """
    best_by_label = {}
    for actor in sensors:
        label = sensor_label_from_role_name(actor.attributes.get("role_name", ""), role_prefix)
        if not label or label not in allowed_labels:
            continue
        prev = best_by_label.get(label)
        if prev is None or actor.id > prev.id:
            best_by_label[label] = actor
    return [best_by_label[label] for label in sorted(allowed_labels) if label in best_by_label]


def destroy_dataset_radars(world):
    """Remove stale dataset radars before a fresh RadarCameraSetup spawn."""
    removed = 0
    for actor in list_radar_actors(world):
        role_name = actor.attributes.get("role_name", "")
        if not role_name.startswith(DATASET_RADAR_ROLE_PREFIX):
            continue
        try:
            actor.destroy()
            removed += 1
        except RuntimeError:
            pass
    return removed


def wait_for_sensors(world, timeout_s, log_progress=False):
    deadline = time.time() + timeout_s
    last_log = 0.0
    last_radar_sensors: list = []
    last_camera_sensors: list = []
    expected = len(EXPECTED_RADAR_LABELS)

    while time.time() < deadline:
        all_radars = list_radar_actors(world)
        radar_sensors = filter_tagged_sensors(
            world,
            "sensor.other.radar",
            DATASET_RADAR_ROLE_PREFIX,
            EXPECTED_RADAR_LABELS,
        )
        camera_sensors = filter_tagged_sensors(
            world,
            "sensor.camera.rgb",
            DATASET_CAMERA_ROLE_PREFIX,
        )
        radar_sensors = select_one_sensor_per_label(
            radar_sensors, DATASET_RADAR_ROLE_PREFIX, EXPECTED_RADAR_LABELS
        )
        last_radar_sensors = radar_sensors
        last_camera_sensors = camera_sensors

        if len(radar_sensors) == expected:
            return radar_sensors, camera_sensors

        if log_progress and time.time() - last_log >= 5.0:
            unique_labels = len(radar_sensors)
            print(
                f"  Waiting for radars: {len(all_radars)} in world, "
                f"{unique_labels}/{expected} unique labels ready",
                flush=True,
            )
            last_log = time.time()

        time.sleep(SENSOR_WAIT_POLL_S)

    last_radar_sensors = select_one_sensor_per_label(
        last_radar_sensors, DATASET_RADAR_ROLE_PREFIX, EXPECTED_RADAR_LABELS
    )
    return last_radar_sensors, last_camera_sensors


def pedestrian_class_from_type_id(type_id):
    return "pedestrian"


def get_vehicle_snapshots(world):
    return [s for s in get_radar_target_snapshots(world) if s["kind"] == "vehicle"]


def get_radar_target_snapshots(world):
    """Vehicles and pedestrians (walkers) eligible for radar point labeling."""
    snapshots = []
    for vehicle in world.get_actors().filter("vehicle.*"):
        vehicle_type_id = vehicle.type_id
        snapshots.append(
            {
                "id": vehicle.id,
                "kind": "vehicle",
                "type_id": vehicle_type_id,
                "class_label": vehicle_class_from_type_id(vehicle_type_id),
                "location": vehicle.get_transform().location,
            }
        )
    for walker in world.get_actors().filter("walker.pedestrian.*"):
        walker_type_id = walker.type_id
        snapshots.append(
            {
                "id": walker.id,
                "kind": "pedestrian",
                "type_id": walker_type_id,
                "class_label": pedestrian_class_from_type_id(walker_type_id),
                "location": walker.get_transform().location,
            }
        )
    return snapshots


def normalize_angle_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def radar_detection_is_labelable(
    velocity_mps: float,
    *,
    min_speed_mps: float = RADAR_LABELABLE_MIN_SPEED_MPS,
    had_candidates: bool = False,
) -> bool:
    """
    True when this return should be scored.

    With had_candidates=True, parked actors in the beam are included even if |v| is low.
    Static clutter with no actor in the beam is excluded unless |v| exceeds min_speed_mps.
    """
    if had_candidates:
        return True
    return abs(velocity_mps) >= min_speed_mps


def should_score_radar_return(
    velocity_mps: float,
    had_candidates: bool,
    *,
    min_speed_mps: float = RADAR_LABELABLE_MIN_SPEED_MPS,
) -> bool:
    return radar_detection_is_labelable(
        velocity_mps, min_speed_mps=min_speed_mps, had_candidates=had_candidates
    )


def radar_candidate_hit_max_bbox_margin_m() -> float | None:
    """
    Candidate hit proximity (m). Override via DATASET_RADAR_CANDIDATE_HIT_MAX_BBOX_MARGIN_M;
    set to 'none'/'off'/'disable' for beam-only candidacy (legacy behavior).
    """
    raw = os.environ.get("DATASET_RADAR_CANDIDATE_HIT_MAX_BBOX_MARGIN_M", "").strip().lower()
    if raw in ("none", "off", "disable"):
        return None
    if raw:
        try:
            value = float(raw)
            return None if value <= 0 else min(max(value, 1.0), 25.0)
        except ValueError:
            pass
    return RADAR_CANDIDATE_HIT_MAX_BBOX_MARGIN_M


def radar_points_per_second_from_env() -> int:
    """Override via DATASET_RADAR_POINTS_PER_SECOND (e.g. 6000). Clamped 500–20000."""
    raw = os.environ.get("DATASET_RADAR_POINTS_PER_SECOND", "").strip()
    if raw:
        try:
            return max(500, min(int(raw), 20000))
        except ValueError:
            pass
    return RADAR_POINTS_PER_SECOND_DEFAULT


def configure_dataset_radar_blueprint(radar_bp) -> int:
    """
    Apply shared dataset radar settings to a CARLA sensor.other.radar blueprint.
    Returns the points_per_second value applied (for logging).
    """
    if radar_bp.has_attribute("range"):
        radar_bp.set_attribute("range", str(int(RADAR_MAX_RANGE_M)))
    if radar_bp.has_attribute("horizontal_fov"):
        radar_bp.set_attribute("horizontal_fov", str(int(RADAR_HORIZONTAL_FOV_DEG)))
    pps = radar_points_per_second_from_env()
    if radar_bp.has_attribute("points_per_second"):
        radar_bp.set_attribute("points_per_second", str(pps))
    if radar_bp.has_attribute("sensor_tick"):
        radar_bp.set_attribute("sensor_tick", str(RADAR_SENSOR_TICK_S))
    return pps


def radar_sensor_limits(radar_actor):
    """Read range (m) and horizontal FOV (deg) from a spawned radar actor."""
    attrs = radar_actor.attributes
    range_m = RADAR_MAX_RANGE_M
    hfov_deg = RADAR_HORIZONTAL_FOV_DEG
    if attrs.get("range"):
        try:
            range_m = float(attrs["range"])
        except ValueError:
            pass
    if attrs.get("horizontal_fov"):
        try:
            hfov_deg = float(attrs["horizontal_fov"])
        except ValueError:
            pass
    return range_m, hfov_deg


def _planar_range_bearing_deg(sensor_location, target_location):
    dx = target_location.x - sensor_location.x
    dy = target_location.y - sensor_location.y
    return math.hypot(dx, dy), math.degrees(math.atan2(dy, dx))


def _detection_beam_yaw_deg(sensor_transform, detection):
    return normalize_angle_deg(
        sensor_transform.rotation.yaw + math.degrees(float(detection.azimuth))
    )


def _actor_bbox_world_center_and_extent(world, actor_snapshot):
    try:
        actor = world.get_actor(actor_snapshot["id"])
    except RuntimeError:
        return None, None
    bbox = actor.bounding_box
    actor_tf = actor.get_transform()
    center = actor_tf.transform(bbox.location)
    return center, bbox.extent


def actor_snapshot_in_sensor_fov(
    sensor_transform, actor_location, max_distance_m, horizontal_fov_deg
):
    """True when actor center is within horizontal FOV and planar range of the sensor."""
    sensor_location = sensor_transform.location
    distance, bearing_deg = _planar_range_bearing_deg(sensor_location, actor_location)
    if distance > max_distance_m:
        return False
    sensor_yaw = sensor_transform.rotation.yaw
    yaw_delta = abs(normalize_angle_deg(bearing_deg - sensor_yaw))
    return yaw_delta <= horizontal_fov_deg * 0.5


def actor_visible_in_detection_beam(
    world,
    sensor_transform,
    detection,
    actor_snapshot,
    *,
    max_range_m=RADAR_MAX_RANGE_M,
    horizontal_fov_deg=RADAR_HORIZONTAL_FOV_DEG,
    depth_margin_m=RADAR_CANDIDATE_DEPTH_MARGIN_M,
    azimuth_margin_deg=RADAR_CANDIDATE_AZIMUTH_MARGIN_DEG,
):
    """
    True when the actor OBB is plausibly illuminated by this detection (depth + bearing gate).
    Uses bbox center range and angular extent, not only the actor origin.
    """
    if world is None:
        return actor_snapshot_in_sensor_fov(
            sensor_transform,
            actor_snapshot["location"],
            min(max_range_m, float(detection.depth) + depth_margin_m),
            horizontal_fov_deg,
        )

    center, extent = _actor_bbox_world_center_and_extent(world, actor_snapshot)
    if center is None or extent is None:
        return False

    sensor_loc = sensor_transform.location
    range_m, bearing_deg = _planar_range_bearing_deg(sensor_loc, center)
    max_extent_m = math.hypot(extent.x, extent.y)
    depth = float(detection.depth)
    depth_min = max(0.0, depth - depth_margin_m - max_extent_m)
    depth_max = min(max_range_m, depth + depth_margin_m + max_extent_m)
    if range_m < depth_min or range_m > depth_max:
        return False

    beam_yaw = _detection_beam_yaw_deg(sensor_transform, detection)
    angular_half_deg = math.degrees(math.atan2(max_extent_m, max(range_m, 0.5)))
    yaw_delta = abs(normalize_angle_deg(bearing_deg - beam_yaw))
    half_fov = horizontal_fov_deg * 0.5
    return yaw_delta <= half_fov + azimuth_margin_deg or yaw_delta <= angular_half_deg + azimuth_margin_deg


def actor_snapshots_near_sensor(sensor_location, actor_snapshots, max_distance_m):
    """Actors whose transform location is within max_distance_m (3D) of the sensor."""
    out = []
    for actor in actor_snapshots:
        if sensor_location.distance(actor["location"]) <= max_distance_m:
            out.append(actor)
    return out


_HIT_MARGIN_DEFAULT = object()


def actor_snapshots_for_radar_detection(
    sensor_transform,
    detection,
    actor_snapshots,
    world=None,
    *,
    max_range_m=RADAR_MAX_RANGE_M,
    horizontal_fov_deg=RADAR_HORIZONTAL_FOV_DEG,
    depth_margin_m=RADAR_CANDIDATE_DEPTH_MARGIN_M,
    azimuth_margin_deg=RADAR_CANDIDATE_AZIMUTH_MARGIN_DEG,
    hit_max_bbox_margin_m=_HIT_MARGIN_DEFAULT,
):
    """
    Per-detection candidates: actor OBB in the detection beam, optionally near the hit.

    When hit_max_bbox_margin_m is set, actors only qualify if the reconstructed hit is within
    that distance of their OBB (reduces beam-only false candidates).
    """
    if hit_max_bbox_margin_m is _HIT_MARGIN_DEFAULT:
        hit_max_bbox_margin_m = radar_candidate_hit_max_bbox_margin_m()
    candidates = []
    for actor in actor_snapshots:
        if not actor_visible_in_detection_beam(
            world,
            sensor_transform,
            detection,
            actor,
            max_range_m=max_range_m,
            horizontal_fov_deg=horizontal_fov_deg,
            depth_margin_m=depth_margin_m,
            azimuth_margin_deg=azimuth_margin_deg,
        ):
            continue
        candidates.append(actor)

    if not candidates or world is None or hit_max_bbox_margin_m is None:
        return candidates

    hit_loc = radar_detection_world_location(sensor_transform, detection)
    near_hit = []
    for actor in candidates:
        margin = actor_bbox_margin_m(world, hit_loc, actor)
        if margin is not None and margin <= hit_max_bbox_margin_m:
            near_hit.append(actor)
    return near_hit


def _actors_in_depth_azimuth_gate(
    world,
    sensor_transform,
    detection,
    candidate_actors,
    *,
    max_range_m=RADAR_MAX_RANGE_M,
    depth_margin_m=RADAR_CANDIDATE_DEPTH_MARGIN_M,
    azimuth_margin_deg=RADAR_CANDIDATE_AZIMUTH_MARGIN_DEG,
):
    if world is None:
        return list(candidate_actors)
    gated = []
    for actor in candidate_actors:
        if actor_visible_in_detection_beam(
            world,
            sensor_transform,
            detection,
            actor,
            max_range_m=max_range_m,
            horizontal_fov_deg=180.0,
            depth_margin_m=depth_margin_m,
            azimuth_margin_deg=azimuth_margin_deg,
        ):
            gated.append(actor)
    return gated


def vehicle_snapshots_near_sensor(sensor_location, vehicle_snapshots, max_distance_m):
    return actor_snapshots_near_sensor(sensor_location, vehicle_snapshots, max_distance_m)


def radar_detection_world_location_legacy(sensor_transform, detection):
    """Previous spherical conversion (kept for TestRadarLabeling.py comparison)."""
    forward_depth = detection.depth * math.cos(detection.azimuth) * math.cos(detection.altitude)
    right_depth = detection.depth * math.sin(detection.azimuth) * math.cos(detection.altitude)
    up_depth = detection.depth * math.sin(detection.altitude)
    sensor_location = sensor_transform.location
    sensor_rotation = sensor_transform.rotation
    yaw = math.radians(sensor_rotation.yaw)
    pitch = math.radians(sensor_rotation.pitch)
    roll = math.radians(sensor_rotation.roll)

    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cr = math.cos(roll)
    sr = math.sin(roll)

    x = forward_depth
    y = right_depth
    z = up_depth

    wx = cy * cp * x + (cy * sp * sr - sy * cr) * y + (cy * sp * cr + sy * sr) * z
    wy = sy * cp * x + (sy * sp * sr + cy * cr) * y + (sy * sp * cr - cy * sr) * z
    wz = -sp * x + cp * sr * y + cp * cr * z

    return carla.Location(
        x=sensor_location.x + wx,
        y=sensor_location.y + wy,
        z=sensor_location.z + wz,
    )


def radar_detection_world_location(sensor_transform, detection):
    """
    World-space hit point using CARLA's radar convention (see PythonAPI/examples/manual_control.py):
    depth along sensor forward, with azimuth/altitude applied as yaw/pitch offsets in degrees.
    """
    rot = sensor_transform.rotation
    beam_rot = carla.Rotation(
        pitch=rot.pitch + math.degrees(detection.altitude),
        yaw=rot.yaw + math.degrees(detection.azimuth),
        roll=rot.roll,
    )
    offset = carla.Transform(carla.Location(), beam_rot).transform(
        carla.Vector3D(x=detection.depth)
    )
    loc = sensor_transform.location
    return carla.Location(loc.x + offset.x, loc.y + offset.y, loc.z + offset.z)


def _world_offset_in_actor_frame(world_offset, actor_rotation):
    """Rotate a world-space offset into the actor's local frame."""
    inv_rot = carla.Rotation(
        pitch=-actor_rotation.pitch,
        yaw=-actor_rotation.yaw,
        roll=-actor_rotation.roll,
    )
    return carla.Transform(carla.Location(), inv_rot).transform(world_offset)


def actor_bbox_margin_m(world, hit_location, actor_snapshot, inflation_m=BBOX_MATCH_EXTENT_INFLATION_M):
    """
    Signed margin to the actor OBB in meters: 0 if inside (with optional inflation),
    otherwise the shortest distance from the hit to the box surface.
    """
    if world is None:
        return None
    try:
        actor = world.get_actor(actor_snapshot["id"])
    except RuntimeError:
        return None

    bbox = actor.bounding_box
    actor_tf = actor.get_transform()
    center_world = actor_tf.transform(bbox.location)
    delta = carla.Location(
        hit_location.x - center_world.x,
        hit_location.y - center_world.y,
        hit_location.z - center_world.z,
    )
    local = _world_offset_in_actor_frame(delta, actor_tf.rotation)
    if bbox.rotation:
        local = _world_offset_in_actor_frame(local, bbox.rotation)

    ex = bbox.extent.x + inflation_m
    ey = bbox.extent.y + inflation_m
    ez = bbox.extent.z + inflation_m

    dx = max(0.0, abs(local.x) - ex)
    dy = max(0.0, abs(local.y) - ey)
    dz = max(0.0, abs(local.z) - ez)
    if dx == 0.0 and dy == 0.0 and dz == 0.0:
        return 0.0
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def vehicle_hit_distance_m(world, hit_location, vehicle_snapshot):
    """Deprecated sphere proxy; prefer actor_bbox_margin_m."""
    margin = actor_bbox_margin_m(world, hit_location, vehicle_snapshot, inflation_m=0.0)
    if margin is not None:
        return margin
    return hit_location.distance(vehicle_snapshot["location"])


def actor_rcs_proxy_projected_area_m2(world, actor_id, sensor_location):
    """
    Sum of (face area × cos θ) for OBB faces visible from the sensor direction — a geometric
    RCS surrogate (m²). Not physical radar cross section; empty if the actor is unavailable.
    """
    try:
        actor = world.get_actor(int(actor_id))
    except (RuntimeError, ValueError, OverflowError):
        return ""
    if actor is None:
        return ""

    bbox = actor.bounding_box
    actor_tf = actor.get_transform()
    extent = bbox.extent
    ex, ey, ez = extent.x, extent.y, extent.z
    face_specs = [
        ((1.0, 0.0, 0.0), 4.0 * ey * ez),
        ((-1.0, 0.0, 0.0), 4.0 * ey * ez),
        ((0.0, 1.0, 0.0), 4.0 * ex * ez),
        ((0.0, -1.0, 0.0), 4.0 * ex * ez),
        ((0.0, 0.0, 1.0), 4.0 * ex * ey),
        ((0.0, 0.0, -1.0), 4.0 * ex * ey),
    ]

    center_world = actor_tf.transform(bbox.location)
    vx = sensor_location.x - center_world.x
    vy = sensor_location.y - center_world.y
    vz = sensor_location.z - center_world.z
    vl = math.sqrt(vx * vx + vy * vy + vz * vz)
    if vl < 1e-6:
        return ""
    ux, uy, uz = vx / vl, vy / vl, vz / vl

    projected = 0.0
    for (lx, ly, lz), area in face_specs:
        loc_n = carla.Location(x=lx, y=ly, z=lz)
        n_bbox = carla.Transform(carla.Location(), bbox.rotation).transform(loc_n)
        n_world = carla.Transform(carla.Location(), actor_tf.rotation).transform(n_bbox)
        nx, ny, nz = n_world.x, n_world.y, n_world.z
        nl = math.sqrt(nx * nx + ny * ny + nz * nz)
        if nl < 1e-9:
            continue
        nx, ny, nz = nx / nl, ny / nl, nz / nl
        dot = nx * ux + ny * uy + nz * uz
        if dot > 0:
            projected += area * dot

    return f"{projected:.6f}"


def vehicle_rcs_proxy_projected_area_m2(world, vehicle_actor_id, sensor_location):
    return actor_rcs_proxy_projected_area_m2(world, vehicle_actor_id, sensor_location)


def match_detection_to_actor(
    hit_location,
    candidate_actors,
    world=None,
    *,
    max_margin_m=RADAR_HIT_MATCH_MAX_MARGIN_M,
    extent_inflation_m=BBOX_MATCH_EXTENT_INFLATION_M,
):
    """
    Pick the actor with the smallest OBB margin to hit_location within max_margin_m.
    """
    if world is None or not candidate_actors:
        return None, None

    best_actor = None
    best_margin = None
    best_center_d = None
    for actor in candidate_actors:
        margin = actor_bbox_margin_m(
            world, hit_location, actor, inflation_m=extent_inflation_m
        )
        if margin is None:
            continue
        center_d = hit_location.distance(actor["location"])
        if margin > max_margin_m:
            continue
        if (
            best_margin is None
            or margin < best_margin
            or (margin == best_margin and (best_center_d is None or center_d < best_center_d))
        ):
            best_margin = margin
            best_center_d = center_d
            best_actor = actor

    if best_actor is None:
        return None, None
    return best_actor, best_margin


def nearest_actor_bbox_margin_m(
    hit_location,
    candidate_actors,
    world=None,
    *,
    extent_inflation_m=BBOX_MATCH_EXTENT_INFLATION_M,
):
    """Smallest OBB margin among candidates (no accept threshold). For labeling failure diagnostics."""
    if world is None or not candidate_actors:
        return None
    best_margin = None
    for actor in candidate_actors:
        margin = actor_bbox_margin_m(
            world, hit_location, actor, inflation_m=extent_inflation_m
        )
        if margin is None:
            continue
        if best_margin is None or margin < best_margin:
            best_margin = margin
    return best_margin


def match_radar_detection_to_actor(
    sensor_transform,
    detection,
    candidate_actors,
    world=None,
    *,
    max_margin_m=RADAR_HIT_MATCH_MAX_MARGIN_M,
    single_candidate_max_margin_m=RADAR_SINGLE_CANDIDATE_MAX_MARGIN_M,
    extent_inflation_m=BBOX_MATCH_EXTENT_INFLATION_M,
    use_legacy_hit_fallback=True,
):
    """
    Match a radar return to an actor: primary hit, legacy hit, then single-target fallbacks.
    """
    if world is None or not candidate_actors:
        return None, None

    hit_loc = radar_detection_world_location(sensor_transform, detection)
    ma, margin = match_detection_to_actor(
        hit_loc,
        candidate_actors,
        world,
        max_margin_m=max_margin_m,
        extent_inflation_m=extent_inflation_m,
    )
    if ma is not None:
        return ma, margin

    if use_legacy_hit_fallback:
        legacy_hit = radar_detection_world_location_legacy(sensor_transform, detection)
        ma, margin = match_detection_to_actor(
            legacy_hit,
            candidate_actors,
            world,
            max_margin_m=max_margin_m,
            extent_inflation_m=extent_inflation_m,
        )
        if ma is not None:
            return ma, margin

    gated = _actors_in_depth_azimuth_gate(world, sensor_transform, detection, candidate_actors)
    pool = gated if gated else candidate_actors

    if len(pool) == 1:
        actor = pool[0]
        margin = actor_bbox_margin_m(
            world, hit_loc, actor, inflation_m=extent_inflation_m
        )
        if margin is not None and margin <= single_candidate_max_margin_m:
            return actor, margin

    best_actor = None
    best_margin = None
    best_center_d = None
    for actor in pool:
        margin = actor_bbox_margin_m(
            world, hit_loc, actor, inflation_m=extent_inflation_m
        )
        if margin is None:
            continue
        center_d = hit_loc.distance(actor["location"])
        if margin > single_candidate_max_margin_m:
            continue
        if (
            best_margin is None
            or margin < best_margin
            or (margin == best_margin and (best_center_d is None or center_d < best_center_d))
        ):
            best_margin = margin
            best_center_d = center_d
            best_actor = actor

    if best_actor is None and len(pool) == 1:
        actor = pool[0]
        margin = actor_bbox_margin_m(
            world, hit_loc, actor, inflation_m=extent_inflation_m
        )
        if margin is not None and margin <= single_candidate_max_margin_m:
            return actor, margin

    if best_actor is None:
        return None, None
    return best_actor, best_margin


def match_detection_to_vehicle(hit_location, candidate_vehicles, world=None, **kwargs):
    return match_detection_to_actor(hit_location, candidate_vehicles, world, **kwargs)


def get_nearby_actors_in_fov(sensor_transform, actor_snapshots, max_distance, horizontal_fov_deg):
    """Vehicles and pedestrians within camera horizontal FOV and range."""
    in_fov = []
    for actor in actor_snapshots:
        actor_location = actor["location"]
        if not actor_snapshot_in_sensor_fov(
            sensor_transform, actor_location, max_distance, horizontal_fov_deg
        ):
            continue
        sensor_location = sensor_transform.location
        distance = math.hypot(
            actor_location.x - sensor_location.x,
            actor_location.y - sensor_location.y,
        )
        in_fov.append(
            {
                "id": actor["id"],
                "kind": actor["kind"],
                "type_id": actor["type_id"],
                "class_label": actor["class_label"],
                "location": actor_location,
                "distance": distance,
            }
        )

    in_fov.sort(key=lambda item: item["distance"])
    return in_fov


def get_nearby_vehicles_in_fov(sensor_transform, vehicles, max_distance, horizontal_fov_deg):
    return get_nearby_actors_in_fov(sensor_transform, vehicles, max_distance, horizontal_fov_deg)


def evaluate_radar_detection_label(
    world,
    sensor_transform,
    detection,
    actors,
    *,
    range_m,
    hfov_deg,
    labelable_min_speed_mps=RADAR_LABELABLE_MIN_SPEED_MPS,
    compare_legacy=False,
):
    """
    Shared radar labeling path (TestRadarLabeling + CaptureRadarCameraData).

    Scores returns with an actor in the detection beam or |velocity| above threshold.
    Matching uses beam/depth candidates, primary + legacy hit, and single-target fallbacks.
    """
    velocity_mps = float(detection.velocity)
    candidate_hit_m = radar_candidate_hit_max_bbox_margin_m()
    match_candidates = actor_snapshots_for_radar_detection(
        sensor_transform,
        detection,
        actors,
        world,
        max_range_m=range_m,
        horizontal_fov_deg=hfov_deg,
        hit_max_bbox_margin_m=candidate_hit_m,
    )
    had_candidates = bool(match_candidates)
    scored = should_score_radar_return(
        velocity_mps,
        had_candidates,
        min_speed_mps=labelable_min_speed_mps,
    )

    matched = False
    legacy_matched = None
    actor_id = None
    actor_kind = ""
    actor_type_id = ""
    actor_class = ""
    match_bbox_margin_m = None
    nearest_bbox_margin_m = None

    if had_candidates:
        hit_loc = radar_detection_world_location(sensor_transform, detection)
        nearest_bbox_margin_m = nearest_actor_bbox_margin_m(
            hit_loc, match_candidates, world
        )
        ma, margin = match_radar_detection_to_actor(
            sensor_transform,
            detection,
            match_candidates,
            world,
        )
        if ma is not None:
            matched = True
            actor_id = ma["id"]
            actor_kind = ma["kind"]
            actor_type_id = ma["type_id"]
            actor_class = ma["class_label"]
            match_bbox_margin_m = margin
            nearest_bbox_margin_m = margin

        if compare_legacy:
            legacy_hit = radar_detection_world_location_legacy(sensor_transform, detection)
            legacy_ma, _ = match_detection_to_actor(
                legacy_hit, match_candidates, world
            )
            legacy_matched = legacy_ma is not None

    return {
        "scored": scored,
        "had_candidates": had_candidates,
        "matched": matched,
        "legacy_matched": legacy_matched,
        "actor_id": actor_id,
        "actor_kind": actor_kind,
        "actor_type_id": actor_type_id,
        "actor_class": actor_class,
        "match_bbox_margin_m": match_bbox_margin_m,
        "nearest_bbox_margin_m": nearest_bbox_margin_m,
        "velocity_mps": velocity_mps,
    }


def labelable_min_speed_from_env() -> float:
    raw = os.environ.get("DATASET_LABELABLE_MIN_SPEED_MPS", "").strip()
    if not raw:
        return RADAR_LABELABLE_MIN_SPEED_MPS
    try:
        return max(0.0, float(raw))
    except ValueError:
        return RADAR_LABELABLE_MIN_SPEED_MPS


def write_capture_labeling_report(
    collector,
    run_dir: str,
    *,
    labelable_min_speed_mps: float,
) -> None:
    """Write TestRadarLabeling-style QA plots/CSVs into the capture folder."""
    out = Path(os.path.normpath(run_dir)) / "radar_labeling_qa"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    try:
        from RadarLabelingTestReport import write_report
    except ImportError as exc:
        print(f"Labeling QA report skipped (import failed): {exc}", file=sys.stderr, flush=True)
        return

    report_kwargs = {
        "min_match_rate": 0.05,
        "expected_radar_labels": EXPECTED_RADAR_LABELS,
        "proximity_m": RADAR_VEHICLE_PROXIMITY_M,
        "hit_match_m": RADAR_HIT_MATCH_MAX_DISTANCE_M,
        "hit_match_max_margin_m": RADAR_HIT_MATCH_MAX_MARGIN_M,
        "bbox_extent_inflation_m": BBOX_MATCH_EXTENT_INFLATION_M,
        "labelable_min_speed_mps": labelable_min_speed_mps,
        "candidate_max_range_m": RADAR_MAX_RANGE_M,
        "candidate_horizontal_fov_deg": RADAR_HORIZONTAL_FOV_DEG,
        "candidate_depth_margin_m": RADAR_CANDIDATE_DEPTH_MARGIN_M,
        "candidate_azimuth_margin_deg": RADAR_CANDIDATE_AZIMUTH_MARGIN_DEG,
        "candidate_hit_max_bbox_margin_m": radar_candidate_hit_max_bbox_margin_m(),
        "single_candidate_max_margin_m": RADAR_SINGLE_CANDIDATE_MAX_MARGIN_M,
    }
    try:
        write_report(collector, out, **report_kwargs)
        print(f"Radar labeling QA report: {out.resolve()}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"Labeling QA report failed: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()


def _run_dataset_extrinsic_exports(world, run_dir: str) -> None:
    """
    After CSVs are closed, write camera_extrinsics.* and sensor_extrinsics.* into run_dir.
    Runs in this process (same Python + CARLA as the recorder) so a subprocess is not
    used — that was failing silently when a different python.exe could not import carla.
    """
    out = Path(os.path.normpath(run_dir))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    try:
        from ExportCameraExtrinsics import write_camera_extrinsics_to_dataset_dir
        from ExportRadarExtrinsics import write_radar_extrinsics_live_to_dataset_dir
    except ImportError as e:
        print(f"Extrinsic export: could not import export modules: {e}", file=sys.stderr, flush=True)
        return
    print("Exporting camera + radar extrinsics into the capture folder...", flush=True)
    try:
        ok_c = write_camera_extrinsics_to_dataset_dir(world, out)
        ok_r = write_radar_extrinsics_live_to_dataset_dir(world, out)
    except Exception as e:
        print(f"Extrinsic export failed: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return
    if ok_c and ok_r:
        print(
            f"Done. Extrinsic files are in: {out}",
            flush=True,
        )
    else:
        print(
            "Extrinsic export incomplete (see messages above). "
            "Keep CARLA and RadarCameraSetup* running when you stop recording with Enter.",
            file=sys.stderr,
            flush=True,
        )


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    print(f"Waiting up to {SENSOR_WAIT_TIMEOUT_S:.1f}s for tagged radar/camera sensors...")
    radar_sensors, camera_sensors = wait_for_sensors(world, SENSOR_WAIT_TIMEOUT_S)

    if not radar_sensors and not camera_sensors:
        print("No tagged dataset sensors found in the world.")
        print("Run RadarCameraSetup.py first, then run this script again.")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    capture_parent = os.environ.get("DATASET_CAPTURE_BASE_DIR", "").strip()
    if capture_parent:
        capture_parent = os.path.normpath(capture_parent)
    else:
        capture_parent = script_dir
    run_dir, camera_dir, radar_csv, camera_csv = make_output_paths(capture_parent)

    # Lets Start.py / export scripts target this run on shutdown (not just "latest" by mtime).
    pointer_path = os.path.join(script_dir, ".last_dataset_capture_dir")
    try:
        with open(pointer_path, "w", encoding="utf-8") as pointer_f:
            pointer_f.write(os.path.normpath(run_dir) + "\n")
    except OSError as e:
        print(f"Warning: could not write {pointer_path}: {e}", file=sys.stderr)

    radar_file, radar_writer = setup_radar_writer(radar_csv)
    camera_file, camera_writer = setup_camera_writer(camera_csv)

    script_dir_for_import = os.path.dirname(os.path.abspath(__file__))
    if script_dir_for_import not in sys.path:
        sys.path.insert(0, script_dir_for_import)
    from RadarLabelingTestReport import DetectionRecord, LabelingStatsCollector

    labelable_min_speed_mps = labelable_min_speed_from_env()
    labeling_collector = LabelingStatsCollector(labelable_min_speed_mps=labelable_min_speed_mps)

    lock = threading.Lock()
    counts = {
        "radar_messages": 0,
        "radar_detections": 0,
        "radar_scored": 0,
        "radar_matched": 0,
        "camera_frames": 0,
    }

    vehicle_count = len(world.get_actors().filter("vehicle.*"))
    pedestrian_count = len(world.get_actors().filter("walker.pedestrian.*"))
    print(f"Recording output directory: {run_dir}")
    print(
        f"World actors: {vehicle_count} vehicles, {pedestrian_count} pedestrians "
        f"(radar labels vehicles + pedestrians via OBB)"
    )
    print(f"Radars found: {len(radar_sensors)}")
    print(f"RGB cameras found: {len(camera_sensors)}")
    if radar_sensors:
        found_radar_labels = {
            sensor_label_from_role_name(r.attributes.get("role_name", ""), DATASET_RADAR_ROLE_PREFIX)
            for r in radar_sensors
        }
        missing_radar_labels = sorted(EXPECTED_RADAR_LABELS - found_radar_labels)
        radar_summary = [
            f"{sensor_label_from_role_name(r.attributes.get('role_name', ''), DATASET_RADAR_ROLE_PREFIX)}:{r.id}"
            for r in radar_sensors
        ]
        radar_summary.sort()
        print(f"Tagged radars (label:actor_id): {', '.join(radar_summary)}")
        if missing_radar_labels:
            print(
                "Warning: Missing expected radars: "
                + ", ".join(missing_radar_labels)
            )
    if camera_sensors:
        camera_summary = [
            f"{sensor_label_from_role_name(c.attributes.get('role_name', ''), DATASET_CAMERA_ROLE_PREFIX)}:{c.id}"
            for c in camera_sensors
        ]
        print(f"Tagged cameras (label:actor_id): {', '.join(camera_summary)}")

    try:
        for radar in radar_sensors:
            sensor_id = radar.id
            sensor_label = sensor_label_from_role_name(
                radar.attributes.get("role_name", ""), DATASET_RADAR_ROLE_PREFIX
            )

            def radar_callback(
                measurement,
                sid=sensor_id,
                slabel=sensor_label,
                radar_actor=radar,
            ):
                sensor_transform = measurement.transform
                loc = sensor_transform.location
                rot = sensor_transform.rotation

                actors = get_radar_target_snapshots(world)
                range_m, hfov_deg = radar_sensor_limits(radar_actor)

                rows = []
                qa_records: list[DetectionRecord] = []
                for idx, detection in enumerate(measurement):
                    label = evaluate_radar_detection_label(
                        world,
                        sensor_transform,
                        detection,
                        actors,
                        range_m=range_m,
                        hfov_deg=hfov_deg,
                        labelable_min_speed_mps=labelable_min_speed_mps,
                    )

                    matched_actor_id = ""
                    matched_actor_kind = ""
                    matched_actor_type_id = ""
                    matched_actor_class = ""
                    matched_actor_bbox_margin = ""
                    matched_vehicle_id = ""
                    matched_vehicle_type_id = ""
                    matched_vehicle_class = ""
                    matched_vehicle_distance = ""
                    nearest_margin_str = ""
                    if label["nearest_bbox_margin_m"] is not None:
                        nearest_margin_str = f"{label['nearest_bbox_margin_m']:.6f}"

                    if label["matched"] and label["actor_id"] is not None:
                        matched_actor_id = str(label["actor_id"])
                        matched_actor_kind = label["actor_kind"]
                        matched_actor_type_id = label["actor_type_id"]
                        matched_actor_class = label["actor_class"]
                        if label["match_bbox_margin_m"] is not None:
                            matched_actor_bbox_margin = f"{label['match_bbox_margin_m']:.6f}"
                        if label["actor_kind"] == "vehicle":
                            matched_vehicle_id = matched_actor_id
                            matched_vehicle_type_id = matched_actor_type_id
                            matched_vehicle_class = matched_actor_class
                            matched_vehicle_distance = matched_actor_bbox_margin

                    rcs_proxy_m2 = ""
                    if matched_actor_id:
                        rcs_proxy_m2 = actor_rcs_proxy_projected_area_m2(
                            world, matched_actor_id, loc
                        )

                    rows.append(
                        [
                            sid,
                            slabel,
                            measurement.frame,
                            f"{measurement.timestamp:.6f}",
                            idx,
                            f"{detection.depth:.6f}",
                            f"{detection.azimuth:.6f}",
                            f"{detection.altitude:.6f}",
                            f"{detection.velocity:.6f}",
                            f"{loc.x:.6f}",
                            f"{loc.y:.6f}",
                            f"{loc.z:.6f}",
                            f"{rot.pitch:.6f}",
                            f"{rot.yaw:.6f}",
                            f"{rot.roll:.6f}",
                            matched_actor_id,
                            matched_actor_kind,
                            matched_actor_type_id,
                            matched_actor_class,
                            matched_actor_bbox_margin,
                            matched_vehicle_id,
                            matched_vehicle_type_id,
                            matched_vehicle_class,
                            matched_vehicle_distance,
                            rcs_proxy_m2,
                            "1" if label["had_candidates"] else "0",
                            "1" if label["scored"] else "0",
                            nearest_margin_str,
                        ]
                    )
                    if label["scored"]:
                        qa_records.append(
                            DetectionRecord(
                                sensor_label=slabel,
                                frame=int(measurement.frame),
                                had_candidates=label["had_candidates"],
                                matched=label["matched"],
                                depth_m=float(detection.depth),
                                velocity_mps=label["velocity_mps"],
                                azimuth_rad=float(detection.azimuth),
                                actor_id=label["actor_id"],
                                actor_kind=label["actor_kind"],
                                actor_class=label["actor_class"],
                                match_bbox_margin_m=label["match_bbox_margin_m"],
                                nearest_bbox_margin_m=label["nearest_bbox_margin_m"],
                            )
                        )
                if not rows:
                    return
                with lock:
                    for row in rows:
                        radar_writer.writerow(row)
                    counts["radar_messages"] += 1
                    counts["radar_detections"] += len(rows)
                    counts["radar_scored"] += len(qa_records)
                    counts["radar_matched"] += sum(1 for r in qa_records if r.matched)
                    labeling_collector.record_message()
                    for rec in qa_records:
                        labeling_collector.record_detection(rec)

            radar.listen(radar_callback)

        for camera in camera_sensors:
            sensor_id = camera.id
            sensor_label = sensor_label_from_role_name(
                camera.attributes.get("role_name", ""), DATASET_CAMERA_ROLE_PREFIX
            )
            sensor_folder = os.path.join(camera_dir, f"camera_{sensor_id}")
            os.makedirs(sensor_folder, exist_ok=True)
            camera_hfov = float(camera.attributes.get("fov", "90.0"))

            def camera_callback(
                image,
                sid=sensor_id,
                slabel=sensor_label,
                folder=sensor_folder,
                sensor_hfov=camera_hfov,
            ):
                sensor_transform = image.transform
                actors = get_radar_target_snapshots(world)
                nearby_actors = get_nearby_actors_in_fov(
                    sensor_transform,
                    actors,
                    NEARBY_DISTANCE_M,
                    sensor_hfov,
                )
                if not nearby_actors:
                    return

                nearest = nearby_actors[0]
                nearby_ids = ";".join(str(a["id"]) for a in nearby_actors)
                nearby_kinds = ";".join(a["kind"] for a in nearby_actors)
                nearby_classes = ";".join(a["class_label"] for a in nearby_actors)

                nearby_vehicles = [a for a in nearby_actors if a["kind"] == "vehicle"]
                nearby_peds = [a for a in nearby_actors if a["kind"] == "pedestrian"]
                nearest_vehicle = nearby_vehicles[0] if nearby_vehicles else None
                nearest_ped = nearby_peds[0] if nearby_peds else None

                def _actor_fields(actor):
                    if actor is None:
                        return ("", "", "", "")
                    return (
                        actor["id"],
                        actor["type_id"],
                        actor["class_label"],
                        f"{actor['distance']:.6f}",
                    )

                nv_id, nv_type, nv_class, nv_dist = _actor_fields(nearest_vehicle)
                np_id, np_type, np_class, np_dist = _actor_fields(nearest_ped)
                veh_ids = ";".join(str(v["id"]) for v in nearby_vehicles)
                veh_classes = ";".join(v["class_label"] for v in nearby_vehicles)
                ped_ids = ";".join(str(p["id"]) for p in nearby_peds)
                ped_classes = ";".join(p["class_label"] for p in nearby_peds)

                image_name = f"frame_{image.frame:08d}.png"
                image_path = os.path.join(folder, image_name)
                image.save_to_disk(image_path)

                with lock:
                    camera_writer.writerow(
                        [
                            sid,
                            slabel,
                            image.frame,
                            f"{image.timestamp:.6f}",
                            image.width,
                            image.height,
                            image_path,
                            nearest["id"],
                            nearest["kind"],
                            nearest["type_id"],
                            nearest["class_label"],
                            f"{nearest['distance']:.6f}",
                            nearby_ids,
                            nearby_kinds,
                            nearby_classes,
                            nv_id,
                            nv_type,
                            nv_class,
                            nv_dist,
                            veh_ids,
                            veh_classes,
                            np_id,
                            np_type,
                            np_class,
                            np_dist,
                            ped_ids,
                            ped_classes,
                        ]
                    )
                    counts["camera_frames"] += 1

            camera.listen(camera_callback)

        print("Listening to sensors...")
        print("Press Enter to stop recording.")

        last_print = time.time()
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getwch()
                if key in ("\r", "\n"):
                    break

            now = time.time()
            if now - last_print >= 2.0:
                with lock:
                    snap = labeling_collector.snapshot()
                    wc = snap.get("with_candidates", 0)
                    rate_c = snap.get("match_rate_given_candidates", 0.0)
                    print(
                        "Status | "
                        f"radar_msgs={counts['radar_messages']} "
                        f"radar_detections={counts['radar_detections']} "
                        f"radar_scored={counts['radar_scored']} "
                        f"radar_matched={counts['radar_matched']} "
                        f"label_rate={100 * rate_c:.1f}% ({snap['matched_detections']}/{wc} w/ cand) "
                        f"camera_frames={counts['camera_frames']}"
                    )
                last_print = now

            time.sleep(0.05)

    finally:
        # Export world poses while sensor actors are still present. Some CARLA builds
        # may leave radars out of get_actors() after sensor.stop(); and we no longer
        # read the full multi-GB radar_data.csv for this step.
        _run_dataset_extrinsic_exports(world, run_dir)
        write_capture_labeling_report(
            labeling_collector,
            run_dir,
            labelable_min_speed_mps=labelable_min_speed_mps,
        )
        for sensor in radar_sensors + camera_sensors:
            try:
                sensor.stop()
            except RuntimeError:
                pass

        with lock:
            radar_file.flush()
            camera_file.flush()
        radar_file.close()
        camera_file.close()

        print("Recording stopped.")
        print(f"Radar file: {radar_csv}")
        print(f"Camera file: {camera_csv}")
        print(f"Camera frames: {camera_dir}")


if __name__ == "__main__":
    main()
