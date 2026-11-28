"""Search over line-shortening combinations at a fixed angle of attack."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from threading import Lock

from sector_pilot import GROUPS as PILOT_GROUPS
from sector_pilot import ROOT, command

GROUPS = {
    "front3": (24, 1, 2),
    "rear3": (12, 13, 14),
    "front6": PILOT_GROUPS["front"],
    "rear6": PILOT_GROUPS["rear"],
}
STROKES = (200, 100, 400)  # Basis points of nominal line length.
WIDTHS = (3, 6)
NOMINAL_LINE_M = math.sqrt(24 / math.pi)
CONTROL_PATCHES = "nos6,cords,sector_pull,sector_mixed_pull"
SLURM_WORKERS = 4
CASE_RANKS = 36


class BudgetReached(Exception):
    pass


@dataclass(frozen=True, order=True)
class Actuation:
    front_width: int = 0
    rear_width: int = 0
    front_bp: int = 0
    rear_bp: int = 0

    @classmethod
    def from_strokes(cls, front_width, rear_width, front_bp, rear_bp):
        return cls(front_width if front_bp else 0, rear_width if rear_bp else 0,
                   front_bp, rear_bp)

    @property
    def family(self):
        return self.front_width, self.rear_width

    def lines(self):
        return [(line, stroke)
                for side, width, stroke in (("front", self.front_width, self.front_bp),
                                            ("rear", self.rear_width, self.rear_bp))
                for line in GROUPS.get(f"{side}{width}", ())]


NEUTRAL = Actuation()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def case_name(actuation, angle):
    return (f"f{actuation.front_width}_{actuation.front_bp:04d}"
            f"_r{actuation.rear_width}_{actuation.rear_bp:04d}_a{angle:.6f}")


def fixed_angle_command(angle, actuation, seed=None, passes=12):
    args = command(angle, (), seed, passes=passes)
    args[args.index("--patch=nos6,cords,sector_pull")] = f"--patch={CONTROL_PATCHES}"
    args.append("--line-take-up-bp=" + ",".join(f"{line}:{bp}" for line, bp in actuation.lines()))
    return args


def radius(values, center):
    return max(abs(value - center) for value in values)


def slurm_case_command(args):
    solver = [item + ",sector_local_mpi" if item.startswith("--patch=") else item
              for item in args[1:]]
    worker = Path(__file__).with_name("sector_worker.py").resolve()
    return ["srun", "--nodes=1", "--ntasks=1", f"--cpus-per-task={CASE_RANKS}",
            "--threads-per-core=1", "--cpu-bind=threads", "--exact", "--exclusive",
            "--mpi=none", "--mem=0", "--input=none", "--export=ALL",
            args[0], "-B", "-u", str(worker), *solver]


def measure(work, angle, actuation, rules):
    """Measure the final three coupled passes and their variation."""
    out = work / "out"
    history_file = out / "nos6_held.json"
    if not history_file.is_file():
        return None
    history = read_json(history_file)
    last = history[-3:]
    status = out / "nos6_held.done"
    if len(last) != 3 or not status.is_file() or read_json(status)["pass"] != last[-1]["pass"]:
        return None
    line_inputs = actuation.lines()
    ids = [line for line, _ in line_inputs]
    strokes = [stroke for _, stroke in line_inputs]
    samples = []
    for row in last:
        if (row["hold"] != angle or row["controlled_line_ids"] != ids
                or row["controlled_take_up_bp"] != strokes):
            raise ValueError(f"Case identity differs from requested inputs: {work}")
        if not row["relaxed"] or row["canopy_model"] != "wall" or row["cx"] <= 0:
            return None
        reference = 0.5 * row["rho"] * row["V"]**2 * row["S0"]
        force = row["aero_force_xyz_N"]
        moment = [value / (reference * row["D0"]) for value in row["aero_moment_xyz_Nm"]]
        tension = row["settled_line_tensions_N"]
        if len(tension) != 24 or len(force) != 3 or len(moment) != 3:
            raise ValueError(f"Missing vector or line loads: {work}")
        if not math.isclose(moment[1], row["mz"], abs_tol=1e-10):
            raise ValueError(f"Pitch moment convention differs from expected y axis: {work}")
        if not math.isclose(force[0] / reference, row["cy"], abs_tol=1e-10):
            raise ValueError(f"Lift convention differs from expected x axis: {work}")
        # The last ten SIMPLE iterations are stored separately from the outer history.
        tail_force = row["force_last_window_range"]
        if row["force_window_samples"] < 10:
            return None
        signed_k = row["cy"] / row["cx"]
        values = dict(quality=abs(signed_k), signed_quality=signed_k,
                      drag_coefficient=row["cx"], lift_coefficient=row["cy"],
                      side_force_coefficient=force[1] / reference,
                      force_tail_range=max(tail_force), shape_gap=row["gap"])
        if not all(map(math.isfinite, [*values.values(), *moment, *tension])):
            return None
        samples.append(dict(**values, moment=moment, tension=tension))
    force_keys = ("quality", "signed_quality", "drag_coefficient", "lift_coefficient", "side_force_coefficient")
    force = {key: median(sample[key] for sample in samples) for key in force_keys}
    k_radius = radius((sample["quality"] for sample in samples), force["quality"])
    if (k_radius > rules.k_drift or max(sample["force_tail_range"] for sample in samples) > rules.cfd_drift
            or max(sample["shape_gap"] for sample in samples) > rules.shape_tol):
        return None
    moments = list(zip(*(sample["moment"] for sample in samples)))
    moment_medians = list(map(median, moments))
    tensions = [sample["tension"] for sample in samples]
    return dict(angle_deg=angle, **force, quality_spread=k_radius,
                moment_coefficients=moment_medians,
                moment_iteration_spread=[radius(series, center)
                                         for series, center in zip(moments, moment_medians)],
                line_tensions_N=list(map(median, zip(*tensions))),
                peak_line_N=max(map(max, tensions)), passes=last[-1]["pass"], case=work.name)


class Cases:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.reservation_lock = Lock()
        self.directory = root / "cases"
        self.directory.mkdir(exist_ok=True)
        sources = [Path(__file__), Path(__file__).with_name("sector_pilot.py")]
        patches = CONTROL_PATCHES
        if args.slurm_steps:
            sources.append(Path(__file__).with_name("sector_worker.py"))
            patches += ",sector_local_mpi"
        sys.path.insert(0, str(ROOT / "coupled"))
        from run_guard import source_hashes
        signature = dict(
            options={key: value for key, value in vars(args).items()
                     if key not in {"work_root", "seed", "max_cases", "dry_run"}},
            groups={key: list(value) for key, value in GROUPS.items()},
            openfoam=os.environ.get("WM_PROJECT_VERSION"),
            mpi_ranks=CASE_RANKS if args.slurm_steps else os.environ.get("SLURM_NTASKS"),
            seed_sha256=hashlib.sha256(args.seed.read_bytes()).hexdigest() if args.seed else None,
            sources={**source_hashes(ROOT / "coupled", patches),
                     **{path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}})
        path = root / "settings.json"
        if path.exists():
            if read_json(path) != signature:
                raise ValueError("Settings, seed, solver or MPI ranks changed. Use a new --work-root.")
        else:
            write_json(path, signature)
        self.count = sum(item.is_dir() for item in self.directory.iterdir())

    def __call__(self, actuation):
        angle = self.args.angle
        work = self.directory / case_name(actuation, angle)
        record = work / "sample.json"
        if record.is_file():
            return read_json(record)["sample"]
        with self.reservation_lock:
            if not work.is_dir():
                if self.count >= self.args.max_cases:
                    raise BudgetReached
                (work / "pics").mkdir(parents=True)
                self.count += 1
            reserved_count = self.count
        out = work / "out"
        history_file = out / "nos6_held.json"
        history = read_json(history_file) if history_file.is_file() else []
        shape = out / "nos6_held_shape.npy"
        if history:
            # History is committed after the per-pass frame. A wall-time kill
            # can leave the rolling shape one pass ahead of committed history.
            frame = out / "кадры" / f"nos6_held_проход{history[-1]['pass']:02d}.npy"
            committed = frame.read_bytes()
            if not shape.is_file() or shape.read_bytes() != committed:
                temp = shape.with_suffix(".tmp")
                temp.write_bytes(committed)
                temp.replace(shape)
        elif shape.is_file():
            # Preserve an uncommitted first pass and allow a clean retry.
            orphan = shape.with_name("nos6_held_shape_uncommitted.npy")
            if orphan.exists():
                raise RuntimeError(f"A second uncommitted initial shape exists in {work}")
            shape.rename(orphan)
        while True:
            sample = measure(work, angle, actuation, self.args)
            if sample is not None:
                write_json(record, {"status": "accepted", "sample": sample})
                return sample
            completed = len(read_json(history_file)) if history_file.is_file() else 0
            if completed >= self.args.max_passes:
                write_json(record, {"status": "pass_limit_unresolved", "sample": None})
                return None
            args = fixed_angle_command(angle, actuation, self.args.seed,
                                       passes=min(self.args.max_passes, max(12, completed + 2)))
            args.append(f"--shape-tol={self.args.shape_tol}")
            if completed >= 6:
                args.append("--all-passes")
            if self.args.slurm_steps:
                args = slurm_case_command(args)
            print(f"cases reserved {reserved_count}/{self.args.max_cases}: {work.name}, completed passes {completed}",
                  flush=True)
            with (work / "driver.log").open("a", encoding="utf-8") as stream:
                ret = subprocess.run(args, cwd=work, stdout=stream, stderr=subprocess.STDOUT).returncode
            new_count = len(read_json(history_file)) if history_file.is_file() else 0
            if new_count <= completed or (out / "nos6_held.failed").exists() or (ret and new_count < self.args.max_passes):
                raise RuntimeError(f"Solver interrupted or failed. See {work / 'driver.log'}")


def single_sector_commands(stroke):
    return [Actuation(front_width=width, front_bp=stroke) for width in WIDTHS] + [
        Actuation(rear_width=width, rear_bp=stroke) for width in WIDTHS]


def quality_rank(actuation, results):
    return (results[actuation]["sample"]["quality"], -(actuation.front_bp + actuation.rear_bp),
            -actuation.front_width - actuation.rear_width, actuation)


def best_in_family(results, family):
    valid = [command for command, row in results.items()
             if command.family == family and row["status"] == "accepted"]
    return max(valid, key=lambda command: quality_rank(command, results), default=None)


def neighbors(actuation, step):
    front_offsets = (-step, 0, step) if actuation.front_width else (0,)
    rear_offsets = (-step, 0, step) if actuation.rear_width else (0,)
    for front_offset, rear_offset in product(front_offsets, rear_offsets):
        front = actuation.front_bp + front_offset
        rear = actuation.rear_bp + rear_offset
        if 0 <= front <= max(STROKES) and 0 <= rear <= max(STROKES):
            yield Actuation.from_strokes(actuation.front_width, actuation.rear_width, front, rear)


def optimize(evaluate, rules, update, workers=1):
    results = {}
    update(results, "running")

    def batch(pool, commands):
        pending = {pool.submit(evaluate, actuation): actuation
                   for actuation in dict.fromkeys(commands) if actuation not in results}
        errors = []
        for future in as_completed(pending):
            actuation = pending[future]
            try:
                sample = future.result()
            except Exception as error:
                errors.append(error)
                continue
            if sample is None:
                status = "insufficient_flow_convergence"
            elif rules.max_tension is not None and sample["peak_line_N"] > rules.max_tension:
                status = "line_load_exceeded"
            else:
                status = "accepted"
            results[actuation] = dict(status=status, sample=sample)
            update(results, "running")
        if errors:
            raise next((error for error in errors if not isinstance(error, BudgetReached)), errors[0])

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            batch(pool, [NEUTRAL, *single_sector_commands(STROKES[0])])
            for stroke in STROKES[1:]:
                batch(pool, single_sector_commands(stroke))

            pairs = list(product(WIDTHS, repeat=2))
            batch(pool, [Actuation(front, rear, 200, 200) for front, rear in pairs])
            initial_pairs = []
            for front, rear in pairs:
                front_best = best_in_family(results, (front, 0))
                rear_best = best_in_family(results, (0, rear))
                initial_pairs.append(Actuation(front, rear, front_best.front_bp if front_best else 200,
                                                rear_best.rear_bp if rear_best else 200))
            batch(pool, initial_pairs)

            families = sorted({actuation.family for actuation in results if actuation != NEUTRAL})
            candidates = [actuation for family in families
                          if (actuation := best_in_family(results, family)) is not None]
            candidates.sort(key=lambda actuation: quality_rank(actuation, results), reverse=True)
            finalists = [actuation.family for actuation in candidates[:2]]
            step = 100
            while step >= rules.stroke_step_bp:
                while True:
                    centers = [best_in_family(results, family) for family in finalists]
                    batch(pool, [candidate for center in centers for candidate in neighbors(center, step)])
                    if centers == [best_in_family(results, family) for family in finalists]:
                        break
                step //= 2
    except BudgetReached:
        update(results, "case_budget_reached")
        return
    except Exception:
        update(results, "interrupted")
        raise
    if not any(row["status"] == "accepted" for row in results.values()):
        status = "no_feasible_cases"
    elif any(row["status"] == "insufficient_flow_convergence" for row in results.values()):
        status = "search_complete_with_unresolved_points"
    else:
        status = "search_complete"
    update(results, status)


def command_record(actuation, result):
    def sector(side, width, stroke):
        return dict(width=width, line_ids=list(GROUPS.get(f"{side}{width}", ())),
                    shortening_percent=stroke / 100,
                    shortening_mm=1000 * NOMINAL_LINE_M * stroke / 10000)

    return dict(front=sector("front", actuation.front_width, actuation.front_bp),
                rear=sector("rear", actuation.rear_width, actuation.rear_bp), **result)


def summary(results, status, rules, count):
    results = dict(sorted(results.items()))
    neutral = results.get(NEUTRAL)
    accepted = [actuation for actuation, row in results.items() if row["status"] == "accepted"]
    best_actuation = max(accepted, key=lambda actuation: quality_rank(actuation, results), default=None)
    best = None
    if best_actuation is not None:
        best = command_record(best_actuation, results[best_actuation])
        if neutral and neutral["sample"] is not None:
            sample, base = best["sample"], neutral["sample"]
            variation = sample["quality_spread"] + base["quality_spread"]
            gain = sample["quality"] - base["quality"]
            best.update(quality_gain=gain, pair_iteration_spread=variation,
                        gain_exceeds_iteration_spread=gain > variation)
    return dict(status=status, angle_deg=rules.angle, case_count=count, max_cases=rules.max_cases,
                line_load_limit_N=rules.max_tension, line_load_limit_checked=rules.max_tension is not None,
                best_measured=best, neutral=command_record(NEUTRAL, neutral) if neutral else None,
                commands=[command_record(actuation, row) for actuation, row in results.items()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--max-cases", type=int, default=60)
    parser.add_argument("--max-passes", type=int, default=16)
    parser.add_argument("--angle", type=float, default=30., help="same prescribed angle of attack for all cases")
    parser.add_argument("--stroke-step-bp", type=int, choices=(25, 50, 100), default=50)
    parser.add_argument("--k-drift", type=float, default=.006)
    parser.add_argument("--cfd-drift", type=float, default=.003)
    parser.add_argument("--shape-tol", type=float, default=.003)
    parser.add_argument("--max-tension", type=float)
    parser.add_argument("--slurm-steps", action="store_true", help="dispatch four 36-CPU steps inside the two-node allocation")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    limits = [args.k_drift, args.cfd_drift, args.shape_tol]
    if args.max_tension is not None:
        limits.append(args.max_tension)
    if (not 0 <= args.angle <= 50 or not all(math.isfinite(x) and x > 0 for x in limits)
            or args.max_cases < 1 or args.max_passes < 12):
        parser.error("Invalid search bounds, convergence tolerances or budget")
    args.angle = round(args.angle, 6)
    if args.seed:
        args.seed = args.seed.resolve()
        if not args.seed.is_file():
            parser.error(f"Seed file not found: {args.seed}")
    if args.dry_run:
        examples = [NEUTRAL, *single_sector_commands(STROKES[0]), Actuation(3, 6, 100, 200)]
        commands = [fixed_angle_command(args.angle, actuation, args.seed) for actuation in examples]
        print(json.dumps(dict(groups=GROUPS, coarse_stroke_bp=STROKES,
                              angle_deg=args.angle,
                              launch_examples=[slurm_case_command(args) for args in commands]
                              if args.slurm_steps else commands), indent=2, ensure_ascii=False))
        return
    allocation = (os.environ.get("SLURM_JOB_NUM_NODES"), os.environ.get("SLURM_NTASKS"),
                  os.environ.get("SLURM_CPUS_PER_TASK"))
    if args.slurm_steps and (not os.environ.get("SLURM_JOB_ID") or allocation != ("2", "4", "36")):
        parser.error("--slurm-steps requires two allocated nodes and four 36-CPU slots")

    import fcntl
    root = args.work_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evaluator = Cases(root, args)
        def update(results, status):
            write_json(root / "summary.json", summary(results, status, args, evaluator.count))
        optimize(evaluator, args, update, workers=SLURM_WORKERS if args.slurm_steps else 1)


if __name__ == "__main__":
    main()
