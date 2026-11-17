"""Картинки по посчитанному случаю связки -- без ParaView, только matplotlib.

    python plots.py ИМЯ [--run=ПАПКА] [--pass=N] [--out=...] [--fields=...] [--V=...] [--wake=0|1]

ПАПКА -- где лежат out/ и fields/ (на кластере run/cc, run/s1, run/mks3 ...;
по умолчанию текущая).  Проход -- последний сохранённый, если не задан.
Скорость потока берётся из журнала (строка «V = ... м/с»), иначе 9 м/с.

Пишет в ПАПКА/pics/:
    ИМЯ_история.png      c_x, c_y, m_z и Dp/D0 по проходам
    ИМЯ_купол.png        перепад ΔCp по куполу, три вида
    ИМЯ_давление.png     поле Cp в плоскости через ось купола
    ИМЯ_скорость.png     поле |U|/V в той же плоскости
    ИМЯ_линии_тока.png   проекция линий тока, включая след для nos6_held
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


def section(centre):
    step = last(PASS / "postProcessing" / "cut") if PASS else None
    if step is None:
        return
    pts, cells, got = polys(step / "plane.vtp")
    tri = mtri.Triangulation(pts[:, 0], pts[:, 2], fan(cells))
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


def fabric_wake_seeds(step, width):
    master, slave = step / "master.vtp", step / "slave.vtp"
    if not master.is_file() or not slave.is_file():
        return ""
    centre, area, normal, pm = vtp.faces_of(master)
    other, _, _, ps = vtp.faces_of(slave)
    mate = vtp.nearest(centre, other)
    paired = np.linalg.norm(centre - other[mate], axis=1) <= 1e-4
    near_section = np.abs(centre[:, 1]) <= 0.65 * np.sqrt(area)
    eligible = np.flatnonzero(paired & near_section & (pm > ps[mate]) & (area > 0))
    chosen = []
    xz = centre[:, [0, 2]]
    for i in eligible[np.argsort(centre[eligible, 0])]:
        if all(np.linalg.norm(xz[i] - xz[j]) >= width / 16 for j in chosen):
            chosen.append(i)
    if not chosen:
        return ""
    # Нормаль master направлена в сторону slave, куда выходит воздух при Δp > 0.
    seeds = centre[chosen] + 0.2 * np.sqrt(area[chosen])[:, None] * normal[chosen]
    points = " ".join("(" + " ".join(f"{v:.9g}" for v in point) + ")" for point in seeds)
    return f"""
    fabricWake
    {{
        type streamLine; libs (fieldFunctionObjects); cloud fabricWakeTracks;
        writeControl writeTime; setFormat vtk; fields (U); U U;
        direction forward; lifeTime 12000; nSubCycle 5;
        seedSampleSet {{ type cloud; axis xyz; points ({points}); }}
    }}
"""


def wake_tracks(centre):
    """Сохранённое поле U и сетка одного прохода во временном кейсе OpenFOAM."""
    tool = shutil.which("simpleFoam")
    step = last(PASS / "postProcessing" / "surfacePressure")
    case = RUN / "case"
    if tool is None or step is None or not (PASS / "polyMesh").is_dir() or not (PASS / step.name / "U").is_file():
        print("след: нужны simpleFoam, polyMesh и U того же прохода, что и surfacePressure")
        return []

    points = np.concatenate(FACES, axis=0) if FACES is not None else centre
    lo, hi = points.min(axis=0), points.max(axis=0)
    width = hi[0] - lo[0]
    start, end, z = lo[0] - 0.12 * width, hi[0] + 0.12 * width, hi[2] + 0.04 * width
    fabric_seeds = fabric_wake_seeds(step, width)
    log = PICS / f"{NAME}_{PASS.name}_след.log"
    try:
        with tempfile.TemporaryDirectory(prefix="nos6_wake_") as tmp:
            work = Path(tmp)
            (work / "system").mkdir()
            (work / "constant").mkdir()
            (work / step.name).mkdir()
            shutil.copytree(PASS / "polyMesh", work / "constant" / "polyMesh")
            for name in ("U", "p", "phi", "k", "omega", "nut"):
                value = PASS / step.name / name
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
    wake
    {{
        type streamLine; libs (fieldFunctionObjects); cloud wakeTracks;
        writeControl writeTime; setFormat vtk; fields (U); U U;
        direction forward; lifeTime 12000; nSubCycle 5;
        seedSampleSet
        {{
            type uniform; axis x; nPoints 45;
            start ({start:.9g} 0 {z:.9g}); end ({end:.9g} 0 {z:.9g});
        }}
    }}
    {fabric_seeds}
}}
""", encoding="ascii")
            with log.open("w", encoding="utf-8") as stream:
                subprocess.run([tool, "-postProcess", "-case", str(work), "-time", step.name],
                               stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=180)
            sets = work / "postProcessing" / "sets"
            paths = [(kind, path) for kind in ("wake", "fabricWake")
                     for path in sorted((sets / kind).rglob("*.vtp"))]
            if not paths:
                print(f"след: OpenFOAM не записал траектории, см. {log}")
                return []
            files = []
            for kind, path in paths:
                label = "след" if kind == "wake" else "выход_из_ткани"
                output = PICS / f"{NAME}_{PASS.name}_{label}_{path.name}"
                shutil.copy2(path, output)
                files.append(output)
            return files
    except (OSError, subprocess.SubprocessError) as error:
        print(f"след: {error}, см. {log}")
        return []


def streamlines(centre):
    step = last(PASS / "postProcessing" / "sets" / "streamlines") if PASS else None
    files = sorted(step.glob("*.vtp")) if step else []
    draw_wake = FLAG.get("--wake", "1" if NAME == "nos6_held" else "0") == "1"
    wake_files = wake_tracks(centre) if draw_wake and centre is not None else []
    if not files and not wake_files:
        return
    fig, ax = plt.subplots(figsize=(7, 9))
    flow = None
    for f in files + wake_files:
        got = vtp.arrays(f)
        pts = np.asarray(got["Points"], dtype=float)
        conn, offs = np.asarray(got["connectivity"]), np.asarray(got["offsets"])
        speed = np.linalg.norm(np.asarray(got["U"]).reshape(-1, 3), axis=1) / V
        segs, vals = [], []
        for a, b in zip(np.concatenate(([0], offs[:-1])), offs):
            idx = conn[a:b]
            xz = pts[idx][:, [0, 2]]
            segs += list(np.stack([xz[:-1], xz[1:]], axis=1))
            vals += list(0.5 * (speed[idx][:-1] + speed[idx][1:]))
        if segs:
            flow = LineCollection(segs, array=np.array(vals), cmap="viridis", linewidths=0.8,
                                  norm=plt.Normalize(0, 1.5))
            ax.add_collection(flow)
        ax.autoscale_view()
    trace(ax, centre)
    if centre is not None:
        window(ax, centre)
    ax.set_aspect("equal")
    ax.set_xlabel("x, м")
    ax.set_ylabel("z, м")
    ax.set_title("Линии тока")
    if flow is not None:
        fig.colorbar(flow, ax=ax, shrink=0.8, label="|U| / V")
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

history()
if PASS is None:
    print(f"нет сохранённых полей в {FIELDS}")
else:
    c = canopy_pressure()
    section(c)
    streamlines(c)
shapes()
