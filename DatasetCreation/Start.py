import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from time import sleep, time

# Only one of these runs; others are excluded from the auto-launch list.
RADAR_SETUP_SCRIPTS = frozenset(
    {
        "RadarCameraSetup12.py",
        "RadarCameraSetup4.py",
        "RadarCameraSetup8.py",
        "RadarCameraSetup14.py",
    }
)

# Never auto-launched with the full dataset stack (run explicitly or via test mode).
MANUAL_ONLY_SCRIPTS = frozenset(
    {
        "TestRadarLabeling.py",
        "RadarLabelingTestReport.py",
        "FetchActorSizing.py",
        "PrintRadarLayoutExtrinsics.py",
    }
)

# Preferred order for full pipeline (after radar setup, before capture).
FULL_PIPELINE_MIDDLE_SCRIPTS = (
    "ClearParkedCarsAndMotorcycles.py",
    "ClearTrashCansAndMailboxes.py",
    "SpawnCarsAtPosition14.py",
    "SpawnPedestriansAcrossMap.py",
    "TrafficLightSetup.py",
    "TrafficLightControl.py",
)

DEFAULT_PEDESTRIAN_COUNT = 30

# Radar count -> setup script (each layout is mutually exclusive).
RADAR_COUNT_TO_SETUP = {
    4: "RadarCameraSetup4.py",
    8: "RadarCameraSetup8.py",
    12: "RadarCameraSetup12.py",
    14: "RadarCameraSetup14.py",
}

# Test runs labeling after setup has had time to spawn sensors; traffic spawns in parallel.
TEST_MODE_SCRIPTS_AFTER_SETUP = (
    "SpawnCarsAtPosition14.py",
    "SpawnPedestriansAcrossMap.py",
    "TrafficLightSetup.py",
    "TrafficLightControl.py",
    "TestRadarLabeling.py",
)


def prompt_radar_count() -> int:
    print("How many radars should the dataset use?")
    print("  1) 4   -> RadarCameraSetup4.py")
    print("  2) 8   -> RadarCameraSetup8.py")
    print("  3) 12  -> RadarCameraSetup12.py")
    print("  For 14 -> RadarCameraSetup14.py: type 14 at the prompt.")
    print("Enter menu 1-3, or type the radar count: 4, 8, 12, or 14.")
    allowed = frozenset({4, 8, 12, 14})
    menu = {"1": 4, "2": 8, "3": 12}
    while True:
        choice = input("Choice (default 1): ").strip() or "1"
        try:
            n = int(choice)
        except ValueError:
            n = None
        if n is not None and n in allowed:
            return n
        if choice in menu:
            return menu[choice]
        print("Please enter 1-3, or the radar count 4, 8, 12, or 14.")


def prompt_run_mode() -> str:
    print("\nRun mode:")
    print("  1) Full dataset pipeline (all scripts + CaptureRadarCameraData.py)")
    print(
        "  2) Test radar labeling only "
        "(setup + spawn cars/pedestrians + TestRadarLabeling.py)"
    )
    while True:
        choice = input("Choice (default 1): ").strip() or "1"
        if choice in ("1", "full", "dataset"):
            return "full"
        if choice in ("2", "test", "test-labeling"):
            return "test"
        print("Please enter 1 or 2.")


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch dataset creation scripts or radar labeling test mode.",
    )
    parser.add_argument(
        "--test-labeling",
        action="store_true",
        help="Test mode: RadarCameraSetup + SpawnCars + TestRadarLabeling (no full capture).",
    )
    parser.add_argument(
        "--radar-count",
        type=int,
        choices=[4, 8, 12, 14],
        help="Radar layout (skips interactive prompt when set).",
    )
    return parser.parse_args()


def get_scripts_to_start(directory: Path, radar_setup_filename: str) -> list[Path]:
    """Return sorted Python scripts in this directory, excluding this file."""
    this_file = Path(__file__).resolve()
    # Export* scripts are run from the Ctrl+C handler, not as parallel children.
    excluded_scripts = {
        "DespawnAllCars.py",
        "ExportRadarExtrinsics.py",
        "ExportCameraExtrinsics.py",
    } | MANUAL_ONLY_SCRIPTS
    scripts = []
    for script in directory.glob("*.py"):
        if script.resolve() == this_file:
            continue
        if script.name in excluded_scripts:
            continue
        if script.name in RADAR_SETUP_SCRIPTS:
            continue
        scripts.append(script)
    sorted_scripts = sorted(scripts, key=lambda path: path.name.lower())
    by_name = {s.name: s for s in sorted_scripts}

    setup_path = directory / radar_setup_filename
    if not setup_path.exists():
        raise FileNotFoundError(f"Radar setup script not found: {setup_path}")

    preferred_last = {"CaptureRadarCameraData.py"}
    ordered_middle: list[Path] = []
    used = set()

    for name in FULL_PIPELINE_MIDDLE_SCRIPTS:
        path = by_name.get(name)
        if path is not None:
            ordered_middle.append(path)
            used.add(name)

    for script in sorted_scripts:
        if script.name in preferred_last or script.name in used:
            continue
        ordered_middle.append(script)

    last = [by_name[name] for name in preferred_last if name in by_name]
    return [setup_path] + ordered_middle + last


def get_scripts_for_test_mode(directory: Path, radar_setup_filename: str) -> list[Path]:
    """Minimal stack to validate radar labeling (vehicles + pedestrians)."""
    setup_path = directory / radar_setup_filename
    if not setup_path.exists():
        raise FileNotFoundError(f"Radar setup script not found: {setup_path}")

    scripts = [setup_path]
    for name in TEST_MODE_SCRIPTS_AFTER_SETUP:
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(f"Test mode script not found: {path}")
        scripts.append(path)
    return scripts


def resolve_dataset_export_dir(script_dir: Path) -> Path | None:
    """
    Active capture run from .last_dataset_capture_dir (written by CaptureRadarCameraData.py),
    else the newest sensor_capture_* that has both radar and camera metadata CSVs.
    """
    pointer = script_dir / ".last_dataset_capture_dir"
    if pointer.is_file():
        try:
            raw = pointer.read_text(encoding="utf-8").strip()
            p = Path(raw)
            if not p.is_absolute():
                p = (script_dir / p).resolve()
            else:
                p = p.resolve()
            if (
                p.is_dir()
                and (p / "radar_data.csv").exists()
                and (p / "camera_data.csv").exists()
            ):
                return p
        except OSError:
            pass
    candidates = [
        p
        for p in script_dir.glob("sensor_capture_*")
        if p.is_dir()
        and (p / "radar_data.csv").exists()
        and (p / "camera_data.csv").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


REQUEST_STOP_FILENAME = ".request_stop"
REPORT_COMPLETE_FILENAME = ".report_complete"
TEST_LABELING_WAIT_S = 120.0
LIVE_STATS_POLL_INTERVAL_S = 2.0


def resolve_last_test_output_dir(script_dir: Path) -> Path | None:
    pointer = script_dir / ".last_radar_labeling_test_dir"
    if not pointer.is_file():
        return None
    raw = pointer.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = (script_dir / path).resolve()
    return path if path.is_dir() else None


def format_live_stats_console_line(data: dict) -> str:
    wc = data.get("with_candidates", 0)
    rate_c = data.get("match_rate_given_candidates", 0.0)
    return (
        "[LiveStats] "
        f"scored={data.get('total_detections', 0):,} | "
        f"matched={data.get('matched_detections', 0):,} | "
        f"w/ candidates={wc:,} → {100 * rate_c:.1f}% | "
        f"updated={data.get('updated_at', '?')} (#{data.get('seq', 0)})"
    )


def tick_live_stats_display(
    script_dir: Path, last_key: tuple[int, str] | None
) -> tuple[int, str] | None:
    """Print when live_stats.json changes (polled from Start.py)."""
    out_dir = resolve_last_test_output_dir(script_dir)
    if out_dir is None:
        return last_key
    path = out_dir / "live_stats.json"
    if not path.is_file():
        return last_key
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return last_key
    key = (int(data.get("seq", 0)), str(data.get("updated_at", "")))
    if key == last_key:
        return last_key
    print(format_live_stats_console_line(data), flush=True)
    return key


def request_test_labeling_stop(script_dir: Path) -> Path | None:
    out_dir = resolve_last_test_output_dir(script_dir)
    if out_dir is None:
        return None
    (out_dir / REQUEST_STOP_FILENAME).write_text("", encoding="utf-8")
    return out_dir


def wait_for_test_labeling_export(out_dir: Path, timeout_s: float = TEST_LABELING_WAIT_S) -> bool:
    """Wait until TestRadarLabeling finishes plots, summary.txt, and .report_complete."""
    deadline = time() + timeout_s
    meta_path = out_dir / "run_meta.json"
    complete_path = out_dir / REPORT_COMPLETE_FILENAME
    while time() < deadline:
        if complete_path.is_file() and (out_dir / "summary.txt").is_file():
            return True
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
            if meta.get("status") in ("completed", "failed") and (out_dir / "summary.json").is_file():
                return True
        sleep(1.0)
    return False


def print_test_labeling_summaries(out_dir: Path) -> None:
    """Echo final test summaries to this console after TestRadarLabeling exits."""
    print("\n" + "=" * 60)
    print("RADAR LABELING TEST — END OF SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Folder: {out_dir.resolve()}\n")

    summary_txt = out_dir / "summary.txt"
    if summary_txt.is_file():
        print(summary_txt.read_text(encoding="utf-8"), end="")
    else:
        print("(summary.txt not found — run may have been interrupted early)")

    summary_json = out_dir / "summary.json"
    if summary_json.is_file():
        try:
            data = json.loads(summary_json.read_text(encoding="utf-8"))
            snap = data.get("summary", {})
            if snap:
                wc = snap.get("with_candidates", 0)
                rate_c = snap.get("match_rate_given_candidates", 0.0)
                print(
                    f"\nKey metrics: scored={snap.get('total_detections', 0):,}, "
                    f"matched={snap.get('matched_detections', 0):,}, "
                    f"w/ candidates={wc:,}, "
                    f"match among candidates={100 * rate_c:.1f}%, "
                    f"PASS={data.get('pass', '?')}"
                )
        except json.JSONDecodeError:
            pass

    print("\nArtifacts:")
    for name in (
        "radar_labeling_summary.png",
        "busiest_frame_summary.png",
        "per_sensor_summary.csv",
        "per_vehicle_summary.csv",
        "per_pedestrian_summary.csv",
        "per_frame_summary.csv",
        "labeling_failure_samples.csv",
        "vehicle_radar_matrix.csv",
        "live_stats.json",
    ):
        path = out_dir / name
        print(f"  [{'ok' if path.is_file() else '—'}] {name}")
    print("=" * 60 + "\n", flush=True)


def stop_all(processes: list[subprocess.Popen], timeout_s: float = 5.0) -> None:
    """Stop all started child scripts."""
    if not processes:
        return

    print("\nStopping all scripts...")

    # First ask each script to terminate gracefully.
    for process in processes:
        if process.poll() is None:
            process.terminate()

    # Wait briefly for graceful exits.
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            pass

    # Force kill any script still running after timeout.
    for process in processes:
        if process.poll() is None:
            print(f"  - Force killing PID {process.pid}")
            process.kill()
            process.wait()

    print("All scripts stopped.")


def export_dataset_extrinsics_in_process(
    script_dir: Path, dataset_dir: Path | None
) -> None:
    """
    Same in-process path as CaptureRadarCameraData: avoids spawning python.exe, which
    often failed to import `carla` and produced no extrinsic files.
    """
    target = dataset_dir
    if target is None:
        target = resolve_dataset_export_dir(script_dir)
    if target is None:
        print(
            "No capture folder found for extrinsics (need sensor_capture_* with both CSVs).",
            file=sys.stderr,
        )
        return
    sd = str(script_dir)
    if sd not in sys.path:
        sys.path.insert(0, sd)
    try:
        import carla
        from ExportCameraExtrinsics import write_camera_extrinsics_to_dataset_dir
        from ExportRadarExtrinsics import write_radar_extrinsics_live_to_dataset_dir
    except ImportError as e:
        print(f"Extrinsics: could not import CARLA/export modules: {e}", file=sys.stderr)
        return

    print(
        "Exporting camera + radar extrinsics (keep CARLA and sensor scripts running)...",
        flush=True,
    )
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        write_camera_extrinsics_to_dataset_dir(world, target)
        write_radar_extrinsics_live_to_dataset_dir(world, target)
    except Exception as e:  # noqa: BLE001
        print(f"Extrinsics export error: {e}", file=sys.stderr)


def despawn_all_cars(script_dir: Path) -> None:
    """Run the dedicated car-despawn script."""
    despawn_script = script_dir / "DespawnAllCars.py"
    if not despawn_script.exists():
        print(f"Despawn script not found: {despawn_script}")
        return

    print("Despawning cars...")
    completed = subprocess.run(
        [sys.executable, str(despawn_script)],
        check=False,
        cwd=str(script_dir),
    )
    if completed.returncode == 0:
        print("Car despawn completed.")
    else:
        print(f"Car despawn exited with code {completed.returncode}.")


def launch_scripts(
    scripts: list[Path],
    current_dir: Path,
    child_env: dict[str, str],
    *,
    test_mode: bool,
) -> list[subprocess.Popen]:
    processes: list[subprocess.Popen] = []
    for script in scripts:
        print(f"  - {script.name}")
        popen_kwargs: dict = {
            "env": child_env,
            "cwd": str(current_dir),
        }
        # Tk GUI in its own console on Windows so it stays visible beside Start.py.
        if script.name == "TrafficLightControl.py" and sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
        process = subprocess.Popen(
            [sys.executable, str(script)],
            **popen_kwargs,
        )
        processes.append(process)
        if script.name in RADAR_SETUP_SCRIPTS:
            sleep(12.0 if test_mode else 2.0)
        elif script.name == "TrafficLightSetup.py":
            sleep(4.0)
        elif script.name == "TrafficLightControl.py":
            sleep(2.0)
        elif script.name == "SpawnPedestriansAcrossMap.py":
            sleep(3.0)
        elif script.name == "SpawnCarsAtPosition14.py":
            sleep(5.0)
        elif test_mode and script.name == "TestRadarLabeling.py":
            sleep(2.0)
    return processes


def main() -> None:
    cli = parse_cli_args()
    current_dir = Path(__file__).resolve().parent

    radar_count = cli.radar_count if cli.radar_count is not None else prompt_radar_count()
    run_mode = "test" if cli.test_labeling else prompt_run_mode()
    radar_setup_name = RADAR_COUNT_TO_SETUP[radar_count]

    if run_mode == "test":
        scripts = get_scripts_for_test_mode(current_dir, radar_setup_name)
    else:
        scripts = get_scripts_to_start(current_dir, radar_setup_name)

    if not scripts:
        print("No scripts found to start.")
        return

    test_mode = run_mode == "test"
    child_env = os.environ.copy()
    child_env["DATASET_EXPECTED_RADAR_COUNT"] = str(radar_count)
    child_env["DATASET_PEDESTRIAN_COUNT"] = str(DEFAULT_PEDESTRIAN_COUNT)
    child_env["DATASET_KEEP_PEDESTRIANS_RUNNING"] = "1"
    child_env["DATASET_KEEP_TRAFFIC_RUNNING"] = "1"
    child_env["DATASET_FREE_VEHICLE_DRIVING"] = "1"
    # Legacy perimeter cycling (proven); set DATASET_AUTOMATIC_TRAFFIC_LIGHTS=1 to unfreeze all.
    child_env["DATASET_AUTOMATIC_TRAFFIC_LIGHTS"] = "0"
    child_env["DATASET_TRAFFIC_LIGHT_GUI_AUTOCONNECT"] = "1"
    if test_mode:
        # Prevent RadarCameraSetup from exiting on Enter (shared console with TestRadarLabeling).
        child_env["DATASET_KEEP_SENSORS_RUNNING"] = "1"
    mode_label = "TEST (radar labeling)" if test_mode else "FULL dataset"

    print(f"Mode: {mode_label}")
    print(f"Radar layout: {radar_count} sensors via {radar_setup_name}")
    print(f"Starting {len(scripts)} scripts from {current_dir}:")
    if test_mode:
        print(
            "Radar test: [RadarTest] + [LiveStats] every ~2s (live_stats.json); "
            "autosave every 90s. Press Enter or Ctrl+C to stop — full summary at end "
            f"(waits up to {TEST_LABELING_WAIT_S:.0f}s for plots + summary.txt)."
        )
    else:
        print(
            "Traffic: TrafficLightSetup (perimeter lights cycle) + TrafficLightControl GUI "
            "(auto-connects). Free-roaming TM vehicles and navmesh pedestrians. "
            "On stop: extrinsics + radar_labeling_qa/ under sensor_capture_*."
        )
        print(
            "On Ctrl+C: camera + radar extrinsics are exported into the active sensor_capture_* folder, "
            "then scripts stop. Keep CARLA running until you see the export messages."
        )

    processes = launch_scripts(scripts, current_dir, child_env, test_mode=test_mode)

    print("All scripts started. Press Ctrl+C to stop all.")

    is_test_mode = run_mode == "test"
    live_stats_key: tuple[int, str] | None = None
    last_live_poll = 0.0
    try:
        while True:
            if is_test_mode:
                now = time()
                if now - last_live_poll >= LIVE_STATS_POLL_INTERVAL_S:
                    live_stats_key = tick_live_stats_display(current_dir, live_stats_key)
                    last_live_poll = now
            sleep(0.25)
    except KeyboardInterrupt:
        if is_test_mode:
            out_dir = request_test_labeling_stop(current_dir)
            if out_dir is not None:
                print(
                    f"\nRequested TestRadarLabeling to stop and save to:\n  {out_dir}",
                    flush=True,
                )
                print(
                    f"Waiting up to {TEST_LABELING_WAIT_S:.0f}s for final report...",
                    flush=True,
                )
                labeling_proc = next(
                    (
                        p
                        for p in processes
                        if p.poll() is None and p.args and "TestRadarLabeling" in p.args[-1]
                    ),
                    None,
                )
                if wait_for_test_labeling_export(out_dir, timeout_s=TEST_LABELING_WAIT_S):
                    if labeling_proc is not None:
                        try:
                            labeling_proc.wait(timeout=15.0)
                        except subprocess.TimeoutExpired:
                            pass
                    print_test_labeling_summaries(out_dir)
                else:
                    print(
                        "Timed out waiting for final report. Partial outputs may exist; "
                        "check live_stats.json and the folder above.",
                        flush=True,
                    )
                    print_test_labeling_summaries(out_dir)
            stop_all(processes, timeout_s=12.0)
        else:
            export_dir = resolve_dataset_export_dir(current_dir)
            if export_dir is not None:
                print(f"Target capture folder for extrinsics: {export_dir}")
            else:
                print(
                    "No capture folder with radar_data.csv + camera_data.csv found; "
                    "extrinsics will try each script's default (latest) path."
                )
            export_dataset_extrinsics_in_process(current_dir, export_dir)
            stop_all(processes)
        despawn_all_cars(current_dir)


if __name__ == "__main__":
    main()
