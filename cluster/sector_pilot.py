"""List the pilot matrix or execute one case in an isolated working directory."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Six-line quadrants share a +7.5 degree offset from the coordinate axes.
FRONT = (23, 24, 1, 2, 3, 4)

def rotate(steps):
    return tuple((line - 1 + steps) % 24 + 1 for line in FRONT)

GROUPS = {"neutral": (), "front": FRONT, "rear": rotate(12),
          "left": rotate(6), "right": rotate(18),
          "sides": rotate(6) + rotate(18), "all": tuple(range(1, 25))}

def cases():
    for angle in (0, 30, 45, 50):
        for name, ids in GROUPS.items():
            if angle != 50 or name == "neutral":
                yield angle, name, ids

def command(angle, ids, seed=None, *, take_up=0.02, passes=12):
    positional = [sys.executable, str(ROOT / "coupled/cs.py"), "nos6_held", "0", "0", "2000", str(passes)]
    if seed is not None:
        positional.append(str(seed))
    return [*positional,
            "--patch=nos6,cords,sector_pull", "--cord-mm=3.2", f"--угол={angle}",
            "--полотнищ=96", "--шаг=4", "--сдвиг=0.02", "--V=30", "--nu=1.46e-5",
            "--line-ids=" + ",".join(map(str, ids)), "--take-up=" + (str(take_up) if ids else "0")]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=int, help="execute case number (zero based)")
    parser.add_argument("--work-root", type=Path, default=ROOT / "run/sector_pilot",
                        help="parent directory for isolated case workspaces")
    parser.add_argument("--seed", type=Path, help="initial structural shape (.npy)")
    args = parser.parse_args()
    matrix = list(cases())
    if args.run is None:
        for index, (angle, name, ids) in enumerate(matrix):
            print(index, angle, name, list(ids))
    else:
        if not 0 <= args.run < len(matrix):
            parser.error("case number outside matrix")
        angle, name, ids = matrix[args.run]
        seed = args.seed.resolve() if args.seed else None
        if seed is not None and not seed.is_file():
            parser.error(f"seed does not exist: {seed}")
        work = args.work_root.resolve() / f"a{angle:02}_{name}"
        (work / "pics").mkdir(parents=True, exist_ok=True)
        subprocess.run(command(angle, ids, seed), cwd=work, check=True)
