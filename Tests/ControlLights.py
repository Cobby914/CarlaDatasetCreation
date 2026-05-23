import random
import time

import carla


def get_or_spawn_vehicle(world):
    """
    Return a vehicle actor:
    - If a vehicle already exists in the world, use the first one found.
    - Otherwise spawn a random vehicle at a random spawn point.
    """
    vehicles = world.get_actors().filter("vehicle.*")
    if len(vehicles) > 0:
        return vehicles[0], False  # False = we did not spawn it here

    bp_lib = world.get_blueprint_library()
    vehicle_bp = random.choice(bp_lib.filter("vehicle.*"))
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    return vehicle, True  # True = cleanup later because we created it


def main():
    # ---------------------------
    # Connection variables
    # ---------------------------
    HOST = "localhost"  # CARLA server address
    PORT = 2000  # CARLA server port (default 2000)
    TIMEOUT_SECONDS = 10.0  # Client timeout for network/API calls

    # ---------------------------
    # Timing/control variables
    # ---------------------------
    STEP_SLEEP_SECONDS = 2.0
    # How long each demo state stays active before switching to the next one.

    ENABLE_AUTOPILOT = True
    # If True, vehicle drives itself while lights are changed.
    # If False, vehicle stays where it is unless some other script controls it.

    client = carla.Client(HOST, PORT)
    client.set_timeout(TIMEOUT_SECONDS)
    world = client.get_world()

    vehicle = None
    spawned_here = False

    try:
        vehicle, spawned_here = get_or_spawn_vehicle(world)

        if ENABLE_AUTOPILOT:
            vehicle.set_autopilot(True)

        print(f"Using vehicle id={vehicle.id}, type={vehicle.type_id}")

        # ============================================================
        # VehicleLightState values you can set (bit flags)
        # ============================================================
        # You can combine flags using bitwise OR (|).
        #
        # carla.VehicleLightState.NONE        -> All lights off
        # carla.VehicleLightState.Position    -> Position/parking lights
        # carla.VehicleLightState.LowBeam     -> Low beam headlights
        # carla.VehicleLightState.HighBeam    -> High beam headlights
        # carla.VehicleLightState.Brake       -> Brake lights
        # carla.VehicleLightState.RightBlinker-> Right turn signal
        # carla.VehicleLightState.LeftBlinker -> Left turn signal
        # carla.VehicleLightState.Reverse     -> Reverse lights
        # carla.VehicleLightState.Fog         -> Fog lights
        # carla.VehicleLightState.Interior    -> Interior cabin lights
        # carla.VehicleLightState.Special1    -> Vehicle-specific special light 1
        # carla.VehicleLightState.Special2    -> Vehicle-specific special light 2
        # carla.VehicleLightState.All         -> Every supported light on
        #
        # IMPORTANT:
        # Some light channels (like Brake/Reverse) are often controlled by physics
        # state too (braking/reversing). You can still set flags manually for demo.

        # Preset demonstrations (label, combined light state)
        light_presets = [
            ("All OFF", carla.VehicleLightState.NONE),
            ("Position only", carla.VehicleLightState.Position),
            (
                "Low beam + Position",
                carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam,
            ),
            (
                "High beam + Position",
                carla.VehicleLightState.Position | carla.VehicleLightState.HighBeam,
            ),
            (
                "Fog + Low beam + Position",
                carla.VehicleLightState.Position
                | carla.VehicleLightState.LowBeam
                | carla.VehicleLightState.Fog,
            ),
            ("Left blinker", carla.VehicleLightState.LeftBlinker),
            ("Right blinker", carla.VehicleLightState.RightBlinker),
            (
                "Hazard style (left+right blinkers)",
                carla.VehicleLightState.LeftBlinker
                | carla.VehicleLightState.RightBlinker,
            ),
            ("Interior", carla.VehicleLightState.Interior),
            ("Special1 + Special2", carla.VehicleLightState.Special1 | carla.VehicleLightState.Special2),
            ("All lights", carla.VehicleLightState.All),
        ]

        print("\nCycling through light presets...")
        for label, state in light_presets:
            vehicle.set_light_state(carla.VehicleLightState(state))
            print(f" -> {label}")
            time.sleep(STEP_SLEEP_SECONDS)

        # End in OFF state so script exits cleanly.
        vehicle.set_light_state(carla.VehicleLightState.NONE)
        print("\nDone. Final state set to NONE (all off).")

    finally:
        # Only destroy if this script spawned the vehicle.
        if vehicle is not None and spawned_here:
            vehicle.destroy()


if __name__ == "__main__":
    main()
