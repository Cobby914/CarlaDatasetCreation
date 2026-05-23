import carla


def despawn_pedestrians(world):
    destroyed = 0
    for pattern in ("walker.pedestrian.*", "controller.ai.walker"):
        for actor in world.get_actors().filter(pattern):
            try:
                if actor.is_alive and actor.destroy():
                    destroyed += 1
            except RuntimeError:
                continue
    return destroyed


def despawn_all_cars(world):
    destroyed_cars = 0
    skipped_motorcycles = 0

    for actor in world.get_actors().filter("vehicle.*"):
        wheels_str = actor.attributes.get("number_of_wheels", "4")
        try:
            wheels = int(wheels_str)
        except ValueError:
            wheels = 4

        if wheels == 2:
            skipped_motorcycles += 1
            continue

        if actor.destroy():
            destroyed_cars += 1

    return destroyed_cars, skipped_motorcycles


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    destroyed_cars, skipped_motorcycles = despawn_all_cars(world)
    destroyed_peds = despawn_pedestrians(world)
    print(f"Destroyed cars: {destroyed_cars}")
    print(f"Skipped motorcycles: {skipped_motorcycles}")
    print(f"Destroyed pedestrians/controllers: {destroyed_peds}")


if __name__ == "__main__":
    main()
