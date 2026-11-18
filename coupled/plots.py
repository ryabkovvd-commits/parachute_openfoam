"""Картинки по посчитанному случаю связки -- без ParaView, только matplotlib.

    python plots.py ИМЯ [--run=ПАПКА] [--pass=N] [--out=...] [--fields=...] [--V=...] [--wake=0|1]
    python plots.py ИМЯ --run=ПАПКА --gif=1 --pass=12 --compare-pass=1
    python plots.py ИМЯ --run=ПАПКА --gif=all --wake=1

ПАПКА -- где лежат out/ и fields/ (на кластере run/cc, run/s1, run/mks3 ...;
по умолчанию текущая).  Проход -- последний сохранённый, если не задан.
Скорость потока берётся из журнала (строка «V = ... м/с»), иначе 9 м/с.

Пишет в ПАПКА/pics/:
    ИМЯ_история.png      c_x, c_y, m_z и Dp/D0 по проходам
    ИМЯ_купол.png        перепад ΔCp по куполу, три вида
    ИМЯ_давление.png     поле Cp в плоскости через ось купола
    ИМЯ_скорость.png     поле |U|/V в той же плоскости
    ИМЯ_линии_тока.png   линии тока и поле полной скорости, включая след для nos6_held
    ИМЯ_форма.png        форма по проходам, проекция на плоскость наклона
Поток идёт снизу вверх (+z), коуш в нуле, наклон -- в плоскости xz.
"""
import json
import re
import sys
from pathlib import Path
import shutil
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
from matplotlib import patheffects  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vtp  # noqa: E402

_arrays = vtp.arrays


def _mean(path):
    got = _arrays(path)
    if "pMean" in got:
        got["p"] = got["pMean"]
    return got


vtp.arrays = _mean

FLAG = {a.split("=")[0]: a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--") and "=" in a}
NAME = [a for a in sys.argv[1:] if not a.startswith("--")][0]
RUN = Path(FLAG.get("--run", ".")).resolve()
OUT = Path(FLAG.get("--out", RUN / "out"))
FIELDS = Path(FLAG.get("--fields", RUN / "fields")) / NAME
PICS = Path(FLAG.get("--pics", RUN / "pics"))
PICS.mkdir(parents=True, exist_ok=True)

journal = (OUT / f"{NAME}.txt").read_text(encoding="utf-8") if (OUT / f"{NAME}.txt").exists() else ""
found = re.search(r"V = ([\d.]+) м/с", journal)
V = float(FLAG.get("--V", found.group(1) if found else 9.0))
KIN = 0.5 * V**2
passes = sorted(FIELDS.glob("pass*"))
FACES = None   # грани сетки на месте ткани -- «настоящий» купол, из surfacePressure
PASS = FIELDS / f"pass{int(FLAG['--pass']):02d}" if "--pass" in FLAG else (passes[-1] if passes else None)
story = OUT / f"{NAME}.json"
HISTORY = json.loads(story.read_text(encoding="utf-8")) if story.is_file() else []


def last(folder):
    """Последний по времени подкаталог."""
    steps = [d for d in folder.iterdir() if d.name.replace(".", "").isdigit()] if folder.exists() else []
    return max(steps, key=lambda d: float(d.name)) if steps else None


def polys(path):
    """Точки, многоугольники (списки номеров) и данные файла vtp."""
    got = vtp.arrays(path)
    pts = np.asarray(got["Points"], dtype=float)
    conn, offs = np.asarray(got["connectivity"]), np.asarray(got["offsets"])
    starts = np.concatenate(([0], offs[:-1]))
    return pts, [conn[a:b] for a, b in zip(starts, offs)], got


def fan(cells):
    """Многоугольники -> треугольники веером."""
    return np.array([[c[0], c[k], c[k + 1]] for c in cells for k in range(1, len(c) - 1)])


def save(fig, suffix):
    path = PICS / f"{NAME}_{suffix}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(path)


def history():
    if not HISTORY:
        return
    n = np.array([int(r["pass"]) for r in HISTORY])
    ticks = n[np.linspace(0, len(n) - 1, min(6, len(n)), dtype=int)]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, key, label in zip(axes, ("cx", "cy", "mz", "ratio"), ("c_x", "c_y", "m_z", "Dp/D0")):
        ax.plot(n, [r[key] for r in HISTORY], "o-", color="#3b6ea8", ms=4)
        ax.set_title(label)
        ax.set_xlabel("проход")
        ax.set_xticks(ticks)
        ax.grid(alpha=0.3)
    fig.suptitle("Сходимость параметров")
    fig.tight_layout()
    save(fig, "история")


def canopy_pressure():
    step = last(PASS / "postProcessing" / "surfacePressure") if PASS else None
    if step is None:
        return None
    fig = plt.figure(figsize=(15, 5))
    if (step / "master.vtp").exists():
        pts, cells, got = polys(step / "master.vtp")
        centre, area, unit, dcp = vtp.delta_cp(step, kin=KIN)
        faces = [pts[c] for c in cells]
        global FACES
        FACES = faces
    else:  # сток: грани ткани из fabric*.dat, рисуем точками
        rows = np.vstack([np.loadtxt(f, ndmin=2) for f in step.glob("fabric*.dat") if f.stat().st_size])
        centre, dcp, faces = rows[:, :3], rows[:, 6] / KIN, None
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-0.5, vmax=3.0)
    cmap = plt.get_cmap("RdBu_r")
    mid, span = centre.mean(axis=0), np.ptp(centre, axis=0).max() * 0.6
    for k, (elev, azim) in enumerate(((5, -90), (90, -90), (35, -60))):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        if faces is not None:
            ax.add_collection3d(Poly3DCollection(faces, facecolors=cmap(norm(dcp)),
                                                 edgecolors=(0, 0, 0, 0.18), linewidths=0.2))
        else:
            ax.scatter(*centre.T, c=cmap(norm(dcp)), s=4)
        for axis, set_lim in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
            set_lim(mid[axis] - span, mid[axis] + span)
        ax.set_box_aspect((1, 1, 1), zoom=1.45)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=fig.axes, shrink=0.7)
    bar.set_label("ΔCp = (p_наветр − p_подветр) / (ρV²/2)")
    side = centre[:, 0] > centre[:, 0].mean()
    fig.suptitle("Перепад давления на куполе")
    save(fig, "купол")
    if faces is not None:
        force = dcp[:, None] * area[:, None] * unit
        total = force.sum(axis=0)
        _, _, _, pm = vtp.faces_of(step / "master.vtp")
        cs, _, _, ps = vtp.faces_of(step / "slave.vtp")
        ps = ps[vtp.nearest(centre, cs)]
        print(f"dCp {NAME}: средний по площади {np.average(dcp, weights=area):.3f}, по граням {dcp.mean():.3f}")
        print("область     граней  площадь   dCp  доля Fx  доля Fz")
        for label, pick in (("x+", side), ("x-", ~side), ("|dCp|>4", np.abs(dcp) > 4)):
            if pick.any():
                f = force[pick].sum(axis=0) / total
                print(f"{label:9s} {pick.sum():7d} {area[pick].sum():8.2f} {np.average(dcp[pick], weights=area[pick]):5.2f}"
                      f" {f[0]:8.2f} {f[2]:8.2f}")
        edges = np.linspace(centre[:, 0].min(), centre[:, 0].max(), 9)
        for a, b in zip(edges[:-1], edges[1:]):
            pick = (centre[:, 0] >= a) & (centre[:, 0] <= b)
            print(f"x {a:6.2f} .. {b:6.2f}: dCp {np.average(dcp[pick], weights=area[pick]):5.2f}"
                  f" Cp1 {np.average(pm[pick], weights=area[pick]) / KIN:6.2f}"
                  f" Cp2 {np.average(ps[pick], weights=area[pick]) / KIN:6.2f}")
    return centre


def window(ax, centre):
    """Окно вокруг купола: 1.2 размера вбок, 1 вниз и 2 вверх по потоку."""
    lo, hi = centre.min(axis=0), centre.max(axis=0)
    size = np.ptp(centre, axis=0).max()
    ax.set_xlim(lo[0] - 1.2 * size, hi[0] + 1.2 * size)
    ax.set_ylim(lo[2] - 1.0 * size, hi[2] + 2.0 * size)


def cut_segments(faces, axis=1, level=0.0):
    """Разрез граней плоскостью axis = level: отрезки в координатах (x, z)."""
    segs = []
    for f in faces:
        d = f[:, axis] - level
        hit = []
        for k in range(len(f)):
            a, b, da, db = f[k], f[(k + 1) % len(f)], d[k], d[(k + 1) % len(f)]
            if da == 0.0:
                hit.append(a)
            elif da * db < 0.0:
                hit.append(a + (b - a) * da / (da - db))
        if len(hit) >= 2:
            segs.append([hit[0][[0, 2]], hit[1][[0, 2]]])
    return segs


def trace(ax, centre):
    """Купол в плоскости y = 0: разрез настоящих граней ткани (или точки для стока)."""
    if FACES is not None:
        ax.add_collection(LineCollection(cut_segments(FACES), colors="k", linewidths=2.0))
    elif centre is not None:
        near = np.abs(centre[:, 1]) < 0.02 * np.ptp(centre[:, 0])
        ax.plot(centre[near, 0], centre[near, 2], ".", color="k", ms=1.5)


def attack_angle_label():
    if PASS is not None:
        number = int(PASS.name.removeprefix("pass"))
        for row in HISTORY:
            if int(row["pass"]) == number:
                if row.get("hold") is not None:
                    return f" при α={float(row['hold']):g}°"
                if row.get("tilt") is not None and row.get("dtilt") is not None:
                    angle = float(row["tilt"]) - float(row["dtilt"])
                    return f" при α={angle:g}°"
                break
    match = re.search(r"(?:^|_)a(-?\d+(?:\.\d+)?)$", RUN.name) or re.search(
        r"(?:^|_)a(-?\d+(?:\.\d+)?)$", NAME)
    return f" при α={float(match[1]):g}°" if match else ""


def section_field(path):
    pts, cells, got = polys(path)
    tri = mtri.Triangulation(pts[:, 0], pts[:, 2], fan(cells))
    return pts, tri, got


def section(centre):
    step = last(PASS / "postProcessing" / "cut") if PASS else None
    if step is None:
        return
    pts, tri, got = section_field(step / "plane.vtp")
    speed = np.linalg.norm(np.asarray(got["U"]).reshape(-1, 3), axis=1) / V
    cp = np.asarray(got["p"]).ravel() / KIN
    total = cp + speed**2
    far = pts[:, 2] < pts[:, 2].min() + 0.05 * np.ptp(pts[:, 2])
    print(f"сечение {NAME}: вдали перед куполом Cp {cp[far].mean():.3f} Cp0 {total[far].mean():.3f},"
          f" наибольшие Cp {cp.max():.3f} Cp0 {total.max():.3f}")
    k = cp.argmax()
    print(f"наибольший Cp в точке x {pts[k, 0]:.2f} z {pts[k, 2]:.2f}, |U|/V там {speed[k]:.2f}")
    angle = attack_angle_label()
    for suffix, title, val, cmap, label, norm in (
            ("скорость", "Распределение скорости", speed, "viridis", "|U| / V", None),
            ("давление", "Распределение давления", cp, "RdBu_r", "Cp = p / (ρV²/2)",
             TwoSlopeNorm(0.0, vmin=min(cp.min(), -1), vmax=max(cp.max(), 1)))):
        fig, ax = plt.subplots(figsize=(7, 7))
        art = ax.tripcolor(tri, val, cmap=cmap, norm=norm, shading="gouraud")
        trace(ax, centre)
        if centre is not None:
            window(ax, centre)
        ax.set_aspect("equal")
        ax.set_xlabel("x, м")
        ax.set_ylabel("z, м")
        ax.set_title(title + angle)
        fig.colorbar(art, ax=ax, shrink=0.8, label=label)
        fig.tight_layout()
        save(fig, suffix)
    (PICS / f"{NAME}_сечение.png").unlink(missing_ok=True)


def read_tracks(files):
    tracks = []
    for path in files:
        data = vtp.arrays(path)
        if not {"Points", "U", "connectivity", "offsets"}.issubset(data):
            continue
        points = np.asarray(data["Points"], dtype=float)
        velocity = np.asarray(data["U"]).reshape(-1, 3)
        conn, offsets = np.asarray(data["connectivity"]), np.asarray(data["offsets"])
        for start, end in zip(np.r_[0, offsets[:-1]], offsets):
            idx = conn[start:end]
            if len(idx) > 1:
                tracks.append((points[idx], velocity[idx]))
    return tracks


def hit_triangle(point, triangles, lo, hi, tol):
    ids = np.flatnonzero(np.all((lo - tol <= point) & (point <= hi + tol), axis=1))
    if not len(ids):
        return None
    a, b, c = triangles[ids].transpose(1, 0, 2)
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n, axis=1)
    keep = norm > 0
    ids, a, b, c, n = ids[keep], a[keep], b[keep], c[keep], n[keep] / norm[keep, None]
    distance = np.abs(np.sum((point - a) * n, axis=1))
    inside = distance <= tol
    for v, w in ((a, b), (b, c), (c, a)):
        edge = w - v
        inside &= np.sum(np.cross(edge, point - v) * n, axis=1) >= -tol * np.linalg.norm(edge, axis=1)
    hits = np.flatnonzero(inside)
    return ids[hits[np.argmin(distance[hits])]] if len(hits) else None


def paired_fabric(step):
    master, slave = step / "master.vtp", step / "slave.vtp"
    if not master.is_file() or not slave.is_file():
        return None
    centre, area, normal, pm = vtp.faces_of(master)
    other, _, _, ps = vtp.faces_of(slave)
    if not len(centre) or not len(other):
        return None
    mate = vtp.nearest(centre, other)
    paired = ((np.linalg.norm(centre - other[mate], axis=1) <= 1e-4)
              & (vtp.nearest(other, centre)[mate] == np.arange(len(centre))))
    return area, normal, pm - ps[mate], paired


def fabric_transfers(step, tracks):
    fabric = paired_fabric(step)
    if fabric is None:
        return {}, 0.0
    area, normal, drop, paired = fabric
    points, cells, _ = polys(step / "master.vtp")
    # Нормаль master направлена в сторону slave. Знак Δp определяет сторону выхода.
    direction = np.sign(drop)[:, None] * normal
    triangles = points[fan(cells)]
    face_ids = np.repeat(np.arange(len(cells)), [len(cell) - 2 for cell in cells])
    lo, hi = triangles.min(axis=1), triangles.max(axis=1)
    tol = 1e-5 * max(np.ptp(points, axis=0).max(), 1.0)
    transfers = {}
    for i, (path, _) in enumerate(tracks):
        hit = path[-1]
        triangle = hit_triangle(hit, triangles, lo, hi, tol)
        if triangle is None:
            continue
        face = face_ids[triangle]
        previous = np.flatnonzero(np.linalg.norm(path[:-1] - hit, axis=1) > tol)
        if paired[face] and len(previous) and np.dot(hit - path[previous[-1]], direction[face]) > 0:
            # Малое смещение помещает начальную точку в ячейку на выходной стороне.
            offset = max(4 * tol, 0.015 * np.sqrt(area[face]))
            transfers[i] = hit + offset * direction[face]
    return transfers, tol


def recirculation_seeds(points, velocity, lo, hi, width):
    """Облако точек с Uz < 0 в следе, в центральной плоскости купола."""
    xz = points[:, [0, 2]]
    x, z = xz.T
    wake = ((x >= lo[0] - 0.3 * width) & (x <= hi[0] + 0.3 * width) &
            (z >= hi[2]) & (z <= hi[2] + 2.0 * width))
    uz = velocity[:, 2]
    reverse = xz[wake & (uz <= -0.005 * V)]
    candidates = xz[wake & (uz <= -0.015 * V)]
    if not len(candidates):
        return [], hi[2] + 2.0 * width
    low, high = reverse.min(axis=0), reverse.max(axis=0)
    xgrid, zgrid = np.meshgrid(np.arange(low[0] + 0.05 * width, high[0], width / 9),
                              np.arange(low[1] + 0.045 * width, high[1], width / 9))
    sites = np.c_[xgrid.ravel(), zgrid.ravel()]
    distances = np.linalg.norm(sites[:, None] - candidates[None], axis=2)
    closest = distances.argmin(axis=1)
    chosen = np.unique(closest[distances[np.arange(len(sites)), closest] < width / 20])
    chosen_x, chosen_z = candidates[chosen].T
    return np.c_[chosen_x, np.zeros(len(chosen)), chosen_z], high[1]


def trace_saved_tracks(step, requests, *, run_dir=RUN, pass_dir=PASS, case_name=NAME):
    """Трассировка по сохранённому полю U во временном кейсе."""
    requests = {name: (points, direction) for name, (points, direction) in requests.items() if len(points)}
    if not requests:
        return {}
    tool = shutil.which("simpleFoam")
    case = run_dir / "case"
    if tool is None or not (pass_dir / "polyMesh").is_dir() or not (pass_dir / step.name / "U").is_file():
        print("след: нужны simpleFoam, polyMesh и U того же прохода, что и surfacePressure")
        return {}
    functions = []
    for name, (targets, direction) in requests.items():
        points = " ".join("(" + " ".join(f"{v:.10g}" for v in point) + ")" for point in targets)
        functions.append(f"""
    {name}
    {{
        type streamLine; libs (fieldFunctionObjects); cloud {name}Tracks;
        writeControl writeTime; setFormat vtk; fields (U); U U;
        direction {direction}; lifeTime 12000; nSubCycle 5;
        seedSampleSet {{ type cloud; axis xyz; points ({points}); }}
    }}""")
    streamlines_dict = "\n".join(functions)
    log = PICS / f"{case_name}_{pass_dir.name}_след.log"
    try:
        with tempfile.TemporaryDirectory(prefix="nos6_wake_") as tmp:
            work = Path(tmp)
            (work / "system").mkdir()
            (work / "constant").mkdir()
            (work / step.name).mkdir()
            shutil.copytree(pass_dir / "polyMesh", work / "constant" / "polyMesh")
            for name in ("U", "p", "phi", "k", "omega", "nut"):
                value = pass_dir / step.name / name
                if value.is_file():
                    shutil.copy2(value, work / step.name / name)
            for name in ("system/fvSchemes", "system/fvSolution", "constant/transportProperties",
                         "constant/turbulenceProperties"):
                shutil.copy2(case / name, work / name)
            (work / "system" / "controlDict").write_text(f"""
FoamFile {{ version 2.0; format ascii; class dictionary; object controlDict; }}
application simpleFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime {step.name}; deltaT 1;
writeControl timeStep; writeInterval 1; writeFormat binary;
functions
{{
    {streamlines_dict}
}}
""", encoding="ascii")
            with log.open("w", encoding="utf-8") as stream:
                subprocess.run([tool, "-postProcess", "-case", str(work), "-time", step.name],
                               stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=180)
            sets = work / "postProcessing" / "sets"
            result = {}
            for name in requests:
                paths = sorted((sets / name).rglob("*.vtp"))
                for path in paths:
                    label = {"continuation": "продолжение", "recirculation": "рециркуляция"}.get(name, name)
                    shutil.copy2(path, PICS / f"{case_name}_{pass_dir.name}_{label}_{path.name}")
                result[name] = read_tracks(paths)
            if not any(result.values()):
                print(f"след: OpenFOAM не записал траектории, см. {log}")
            return result
    except (OSError, subprocess.SubprocessError) as error:
        print(f"след: {error}, см. {log}")
        return {}


def join_transfers(tracks, transfers, continuations, tol):
    if not continuations:
        return 0
    distance = np.linalg.norm(np.asarray(list(transfers.values()))[:, None, :]
                              - np.asarray([points[0] for points, _ in continuations]), axis=2)
    incoming = list(transfers)
    claimed_in, claimed_out = set(), set()
    for k in np.argsort(distance, axis=None):
        i, j = np.unravel_index(k, distance.shape)
        if distance[i, j] > 4 * tol:
            break
        if i in claimed_in or j in claimed_out:
            continue
        points, velocity = tracks[incoming[i]]
        tail_points, tail_velocity = continuations[j]
        tracks[incoming[i]] = (np.vstack((points, tail_points)), np.vstack((velocity, tail_velocity)))
        claimed_in.add(i)
        claimed_out.add(j)
    return len(claimed_in)


def draw_recirculation(ax, tracks):
    curves = [points[:, [0, 2]] for points, _ in tracks]
    lines = LineCollection(curves, colors="#122536", linewidths=0.65, alpha=0.82, zorder=3)
    lines.set_path_effects([patheffects.withStroke(linewidth=1.45, foreground="white", alpha=0.65)])
    ax.add_collection(lines)
    for points, velocity in tracks[::6]:
        if len(points) < 10:
            continue
        middle = len(points) // 2
        a, b = points[[middle - 2, middle + 2]][:, [0, 2]]
        if np.linalg.norm(b - a) > 0.01:
            if np.dot(b - a, velocity[middle, [0, 2]]) < 0:
                a, b = b, a
            ax.annotate("", xy=b, xytext=a,
                        arrowprops=dict(arrowstyle="-|>", color="#122536", lw=0.8, mutation_scale=8), zorder=4)


def streamlines(centre):
    step = last(PASS / "postProcessing" / "sets" / "streamlines") if PASS else None
    if step is None:
        return
    tracks = read_tracks(sorted(step.glob("*.vtp")))
    if not tracks:
        return
    cut_file = PASS / "postProcessing" / "cut" / step.name / "plane.vtp"
    field = section_field(cut_file) if cut_file.is_file() else None
    draw_wake = FLAG.get("--wake", "1" if NAME == "nos6_held" else "0") == "1"
    pressure_step = PASS / "postProcessing" / "surfacePressure" / step.name
    shape = np.concatenate(FACES, axis=0) if FACES is not None else centre
    wake_top = None
    if shape is not None:
        lo, hi = shape.min(axis=0), shape.max(axis=0)
        width = hi[0] - lo[0]
        wake_top = hi[2] + 2 * width
    wake = []
    if draw_wake:
        transfers, tol = fabric_transfers(pressure_step, tracks)
        seeds = []
        if field is not None and wake_top is not None and width > 0:
            seeds, wake_top = recirculation_seeds(field[0], field[2]["U"].reshape(-1, 3), lo, hi, width)
        traced = trace_saved_tracks(pressure_step, {
            "continuation": (list(transfers.values()), "forward"),
            "recirculation": (seeds, "bidirectional"),
        })
        wake = traced.get("recirculation", [])
        if len(seeds):
            print(f"след: {len(seeds)} начальных точек, {len(wake)} траекторий")
        if transfers:
            joined = join_transfers(tracks, transfers, traced.get("continuation", []), tol)
            print(f"линии тока: продолжено через ткань {joined} из {len(transfers)}")
    fig, ax = plt.subplots(figsize=(7.5, 10.4), constrained_layout=True)
    vmax = 7 * V / 6
    if field is not None:
        speed = np.linalg.norm(field[2]["U"].reshape(-1, 3), axis=1)
        color = ax.tripcolor(field[1], speed, cmap="viridis", vmin=0, vmax=vmax, shading="gouraud")
        ax.add_collection(LineCollection([points[:, [0, 2]] for points, _ in tracks],
                                         colors="#ecf5f6", linewidths=0.8, alpha=0.9, zorder=2))
    else:
        segs, vals = [], []
        for points, velocity in tracks:
            xz = points[:, [0, 2]]
            speed = np.linalg.norm(velocity, axis=1)
            segs.append(np.stack([xz[:-1], xz[1:]], axis=1))
            vals.append(0.5 * (speed[:-1] + speed[1:]))
        color = LineCollection(np.concatenate(segs), array=np.concatenate(vals), cmap="viridis", linewidths=0.8,
                               norm=plt.Normalize(0, vmax))
        ax.add_collection(color)
    if wake:
        draw_recirculation(ax, wake)
    ax.autoscale_view()
    trace(ax, centre)
    if wake_top is not None and width > 0:
        ax.set_xlim(lo[0] - 0.64 * width, hi[0] + 0.64 * width)
        ax.set_ylim(lo[2] - 1.08 * width, wake_top + 0.17 * width)
    ax.set_aspect("equal")
    ax.set_xlabel("x, м")
    ax.set_ylabel("z, м")
    ax.set_title("Линии тока" + attack_angle_label())
    fig.colorbar(color, ax=ax, shrink=0.84, extend="max", label="V, м/с")
    save(fig, "линии_тока")


def shapes():
    frames = sorted((OUT / "кадры").glob(f"{NAME}_проход*.npy"))
    if not frames or not HISTORY:
        return
    first = HISTORY[0]
    g, r = first["gores"], first["rings"]
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = plt.get_cmap("Blues")(np.linspace(0.35, 1.0, len(frames)))
    for f, c in zip(frames, colors):
        pos = np.load(f)
        grid = pos[:(r + 1) * g].reshape(r + 1, g, 3)
        # два меридиана в плоскости наклона (0° и 180°) -- профиль купола
        prof = np.vstack([grid[::-1, g // 2], grid[:, 0]])
        ax.plot(prof[:, 0], prof[:, 2], "-", color=c, lw=1.5, label=f.stem.split("_")[-1])
    ax.set_aspect("equal")
    ax.set_xlabel("x, м")
    ax.set_ylabel("z, м")
    ax.set_title("Изменение профиля купола")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "форма")

def fabric_crossings(step, points, cells, level, coefficients):
    """Связать стороны ткани там, где её парные грани пересекают сечение."""
    fabric = paired_fabric(step)
    if fabric is None:
        return [], []
    area, normal, drop, paired = fabric
    d, cw = coefficients
    un = 2.0 * drop / (d + np.sqrt(d * d + 4.0 * cw * np.abs(drop)))
    width = np.ptp(points[:, 0])
    tolerance = 1e-5 * max(np.ptp(points, axis=0).max(), 1.0)
    locations, markers, seeds = [], [], []
    for face in np.flatnonzero(paired & (np.abs(un) >= 0.05)):
        segments = cut_segments([points[cells[face]]], level=level)
        if not segments:
            continue
        x, z = np.mean(segments[0], axis=0)
        middle = np.array([x, level, z])
        if any(np.linalg.norm(middle - point) < 0.06 * width for point in locations):
            continue
        locations.append(middle)
        direction = np.sign(un[face]) * normal[face]
        velocity = np.tile(un[face] * normal[face], (2, 1))
        half_length = 0.038 * width * direction
        offset = max(4 * tolerance, 0.015 * np.sqrt(area[face])) * direction
        markers.append((np.stack((middle - half_length, middle + half_length)), velocity))
        seeds.append((middle - offset, middle + offset))
    return markers, seeds


def join_fabric_crossings(tracks, seeds, upstream, downstream, tolerance):
    """Присоединить сохранённый переход к трассам с обеих сторон ткани."""
    if not upstream or not downstream:
        return 0
    joined = 0
    for index, (a, b) in enumerate(seeds):
        ends = []
        for target, candidates in ((a, upstream), (b, downstream)):
            points, velocities = min(candidates, key=lambda track: min(
                np.linalg.norm(track[0][0] - target), np.linalg.norm(track[0][-1] - target)))
            if np.linalg.norm(points[0] - target) > np.linalg.norm(points[-1] - target):
                points, velocities = points[::-1], velocities[::-1]
            if np.linalg.norm(points[0] - target) > tolerance:
                break
            ends.append((points, velocities))
        if len(ends) == 2:
            (before, u_before), (after, u_after) = ends
            tracks[index] = (np.vstack((before[::-1], after)), np.vstack((u_before[::-1], u_after)))
            joined += 1
    return joined


def animation_data(run, name, pass_dir, case_journal):
    if pass_dir is None:
        raise ValueError(f"{name}: нет сохранённых проходов в {run}")
    step = last(pass_dir / "postProcessing" / "sets" / "streamlines")
    if step is None:
        raise ValueError(f"Нет сохранённых линий тока в {pass_dir}")
    cut_file = pass_dir / "postProcessing" / "cut" / step.name / "plane.vtp"
    pressure_step = pass_dir / "postProcessing" / "surfacePressure" / step.name
    surface = pressure_step / "master.vtp"
    missing = next((path for path in (cut_file, surface) if not path.is_file()), None)
    if missing is not None:
        raise ValueError(f"Нет файла {missing}")

    points, triangles, fields = section_field(cut_file)
    cut_velocity = fields["U"].reshape(-1, 3)
    surface_points, cells, _ = polys(surface)
    level = float(np.median(points[:, 1]))
    outline = cut_segments([surface_points[cell] for cell in cells], level=level)
    tracks = read_tracks(sorted(step.glob("*.vtp")))
    if not tracks:
        raise ValueError(f"В {step} нет пригодных траекторий с полем U")
    fabric_line = re.search(r"(?:ВДУВОМ-ОТСОСОМ|Проницаемая ткань):[^\n]*?D\s*=\s*([\d.eE+\-]+)"
                            r"[^\n]*?Cw\s*=\s*([\d.eE+\-]+)", case_journal)
    crossings, seed_pairs = [], []
    if fabric_line:
        crossings, seed_pairs = fabric_crossings(pressure_step, surface_points, cells, level,
                                                  tuple(map(float, fabric_line.groups())))
        print(f"{name} {pass_dir.name}: локальных переходов через ткань {len(crossings)}")
    if FLAG.get("--wake", "1" if name == "nos6_held" else "0") == "1":
        lo, hi = surface_points.min(axis=0), surface_points.max(axis=0)
        seeds, _ = recirculation_seeds(points, cut_velocity, lo, hi, hi[0] - lo[0])
        extra = trace_saved_tracks(pressure_step, {
            "recirculation": (seeds, "bidirectional"),
            "fabricUpstream": ([pair[0] for pair in seed_pairs], "backward"),
            "fabricDownstream": ([pair[1] for pair in seed_pairs], "forward"),
        }, run_dir=run, pass_dir=pass_dir, case_name=name)
        if seed_pairs:
            tol = 4e-5 * max(np.ptp(surface_points, axis=0).max(), 1.0)
            joined = join_fabric_crossings(crossings, seed_pairs, extra.get("fabricUpstream", []),
                                           extra.get("fabricDownstream", []), tol)
            print(f"{name} {pass_dir.name}: продолжено через ткань {joined} из {len(seed_pairs)}")
        tracks.extend(extra.get("recirculation", []))
    label = f"{name} · проход {int(pass_dir.name[4:])}"
    data = points, triangles.triangles, np.linalg.norm(cut_velocity, axis=1), surface_points, outline, tracks, crossings
    return label, data


def animated_tracks(tracks, lo, hi):
    """Prepare advection along each saved 3D polyline before projecting to xz."""
    rng = np.random.default_rng(42)
    curves = []
    for points, velocity in tracks:
        in_view = np.all((lo <= points[:, [2, 0]]) & (points[:, [2, 0]] <= hi), axis=1)
        hits = np.flatnonzero(in_view)
        if len(hits) < 2:
            continue
        start, stop = max(0, hits[0] - 1), hits[-1] + 2
        points, velocity = points[start:stop], velocity[start:stop]
        tangent = np.diff(points, axis=0)
        mean_u = 0.5 * (velocity[1:] + velocity[:-1])
        distance = np.linalg.norm(tangent, axis=1)
        speed = np.linalg.norm(mean_u, axis=1)
        if np.sum(tangent * mean_u) < 0.0:
            points, distance, speed = points[::-1], distance[::-1], speed[::-1]
        keep = distance > 1e-8
        if not keep.any():
            continue
        elapsed = np.r_[0.0, np.cumsum(distance[keep] / np.maximum(speed[keep], 1e-3))]
        period = elapsed[-1]
        count = min(16, max(1, int(period / 0.12)))
        phases = (np.arange(count) + rng.random()) * period / count
        curves.append((elapsed, points[np.r_[True, keep]][:, [2, 0]], phases))
    return curves


def moving_dashes(curves, tick, trail=0.027):
    segments = []
    for elapsed, points, phases in curves:
        heads = (tick + phases) % elapsed[-1]
        tails = np.maximum(heads - trail, 0.0)
        samples = np.linspace(tails, heads, num=4, axis=1)
        projected = [np.interp(samples, elapsed, coordinate) for coordinate in points.T]
        segments.extend(np.stack(projected, axis=-1))
    return segments


def animated_panels(all_passes):
    if all_passes:
        if any(key in FLAG for key in ("--pass", "--compare-pass", "--compare-run", "--compare-name")):
            raise ValueError("--gif=all выбирает все проходы одного расчёта")
        panels = []
        folders = [path for path in passes if path.is_dir() and path.name[4:].isdigit()]
        for pass_dir in sorted(folders, key=lambda path: int(path.name[4:])):
            try:
                panels.append(animation_data(RUN, NAME, pass_dir, journal))
            except (OSError, ValueError) as error:
                print(f"{pass_dir.name}: пропущен. {error}", file=sys.stderr)
        if not panels:
            raise ValueError(f"{NAME}: нет проходов с полями для анимации в {FIELDS}")
        print("В GIF вошли проходы " + ", ".join(label.rsplit(" ", 1)[-1] for label, _ in panels))
        return [panels]

    items = [(RUN, NAME, PASS, journal)]
    if "--compare-pass" in FLAG or "--compare-run" in FLAG:
        other = Path(FLAG.get("--compare-run", RUN)).resolve()
        name = FLAG.get("--compare-name", NAME)
        folder = other / "fields" / name
        step = FLAG.get("--compare-pass")
        pass_dir = folder / f"pass{int(step):02d}" if step else max(folder.glob("pass*"), default=None)
        other_log = other / "out" / f"{name}.txt"
        case_journal = other_log.read_text(encoding="utf-8") if other_log.is_file() else ""
        items.insert(0, (other, name, pass_dir, case_journal))

    return [[animation_data(*item)] for item in items]


def flow_gif(all_passes):
    from matplotlib.animation import FuncAnimation, PillowWriter

    panels = animated_panels(all_passes)
    surface = np.concatenate([panel[3][:, [2, 0]] for group in panels for _, panel in group])
    width = max(np.ptp(surface[:, 1]), np.ptp(surface[:, 0]))
    lo = surface.min(axis=0) - width * np.array([0.7, 0.48])
    hi = surface.max(axis=0) + width * np.array([2.0, 0.48])

    background, ink = "#10151d", "#f3f7fb"
    fig = plt.figure(figsize=(5.4, 9.6) if len(panels) == 2 else (7.2, 4.8), facecolor=background)
    grid = fig.add_gridspec(len(panels), 1, left=0.035, right=0.965,
                           bottom=0.055, top=0.74, hspace=0.23)
    cax = fig.add_axes([0.055, 0.85, 0.89, 0.031])
    title = "Обтекание купола по проходам" if all_passes else "Обтекание купола"
    fig.suptitle(title, color=ink, fontsize=17, fontweight="bold")
    layers = []
    for index, group in enumerate(panels):
        ax = fig.add_subplot(grid[index, 0])
        frames = []
        for label, (points, triangles, speed, _, outline, tracks, crossings) in group:
            mesh = ax.tripcolor(points[:, 2], points[:, 0], triangles, speed, shading="gouraud",
                                cmap="viridis", vmin=0, vmax=7 * V / 6, visible=False)
            contour = LineCollection(np.asarray(outline)[..., ::-1] if outline else [],
                                     colors=ink, linewidths=1.6, zorder=3, visible=False)
            dashes = LineCollection([], colors=ink, linewidths=0.9, alpha=0.80, zorder=2, visible=False)
            cross_dashes = LineCollection([], colors="#ff77b6", linewidths=2.2, zorder=4, visible=False)
            cross_dashes.set_path_effects([patheffects.withStroke(linewidth=3, foreground=background)])
            ax.add_collection(contour)
            ax.add_collection(dashes)
            ax.add_collection(cross_dashes)
            frames.append((label, (mesh, contour, dashes, cross_dashes),
                           animated_tracks(tracks, lo, hi), animated_tracks(crossings, lo, hi)))
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect="equal", facecolor=background,
               xticks=[], yticks=[])
        panel_label = ax.set_title("", loc="left", color=ink, fontsize=11, fontweight="bold")
        layers.append((panel_label, frames))
        for spine in ax.spines.values():
            spine.set_color("#485467")
    bar = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="max")
    bar.set_label("Модуль скорости, м/с", color=ink, fontsize=9)
    bar.ax.tick_params(colors=ink, labelsize=8)

    fps = 12
    frames_per_pass = 10
    previous_index = -1

    def draw(number):
        nonlocal previous_index
        index = min(number // frames_per_pass, len(panels[0]) - 1)
        for panel_label, frames in layers:
            label, artists, curves, crossings = frames[index]
            if index != previous_index:
                if previous_index >= 0:
                    for artist in frames[previous_index][1]:
                        artist.set_visible(False)
                for artist in artists:
                    artist.set_visible(True)
                panel_label.set_text(label)
            artists[-2].set_segments(moving_dashes(curves, number * 0.047))
            artists[-1].set_segments(moving_dashes(crossings, number * 0.047, trail=0.20))
        previous_index = index

    suffix = "flow_passes" if all_passes else "flow"
    path = PICS / f"{NAME}_{suffix}.gif"
    frame_count = frames_per_pass * len(panels[0]) + fps if all_passes else 72
    movie = FuncAnimation(fig, draw, frames=frame_count, cache_frame_data=False)
    movie.save(path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    print(path)


if FLAG.get("--gif") in {"1", "all"}:
    try:
        flow_gif(FLAG["--gif"] == "all")
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
else:
    history()
    if PASS is None:
        print(f"нет сохранённых полей в {FIELDS}")
    else:
        c = canopy_pressure()
        section(c)
        streamlines(c)
    shapes()
