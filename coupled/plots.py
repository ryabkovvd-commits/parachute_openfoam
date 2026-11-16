"""Картинки по посчитанному случаю связки -- без ParaView, только matplotlib.

    python plots.py ИМЯ [--run=ПАПКА] [--pass=N] [--out=...] [--fields=...] [--V=...]

ПАПКА -- где лежат out/ и fields/ (на кластере run/cc, run/s1, run/mks3 ...;
по умолчанию текущая).  Проход -- последний сохранённый, если не задан.
Скорость потока берётся из журнала (строка «V = ... м/с»), иначе 9 м/с.

Пишет в ПАПКА/pics/:
    ИМЯ_история.png      c_x, c_y, m_z, Dp/D0 и невязка по проходам
    ИМЯ_купол.png        перепад ΔCp по куполу, три вида
    ИМЯ_сечение.png      скорость и давление в плоскости через ось купола
    ИМЯ_линии_тока.png   линии тока в той же плоскости
    ИМЯ_форма.png        форма по проходам, проекция на плоскость наклона
Поток идёт снизу вверх (+z), коуш в нуле, наклон -- в плоскости xz.
"""
import json
import re
import sys
from pathlib import Path

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
    story = OUT / f"{NAME}.json"
    if not story.exists():
        return
    rows = json.loads(story.read_text(encoding="utf-8"))
    n = [r["pass"] for r in rows]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, key, label in zip(axes, ("cx", "cy", "mz", "ratio"), ("c_x", "c_y", "m_z", "Dp/D0")):
        ax.plot(n, [r[key] for r in rows], "o-", color="#3b6ea8", ms=4)
        ax.set_title(label)
        ax.set_xlabel("проход")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{NAME}: по проходам")
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
    for k, (elev, azim, title) in enumerate(((5, -90, "сбоку (плоскость наклона)"),
                                             (90, -90, "по оси потока, с подветренной стороны"),
                                             (35, -60, "наискось"))):
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
        ax.set_title(title)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=fig.axes, shrink=0.7)
    bar.set_label("ΔCp = (p_наветр − p_подветр) / (ρV²/2)")
    side = centre[:, 0] > centre[:, 0].mean()
    fig.suptitle(f"{NAME}, {PASS.name}, итерация/время {step.name}: перепад по куполу, "
                 f"средний ΔCp = {dcp.mean():.3f}, x+ {dcp[side].mean():.3f}, x- {dcp[~side].mean():.3f}")
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
    print(f"наибольший Cp в точке x {pts[k, 0]:.2f} z {pts[k, 2]:.2f}, |U|/V там {speed[k]:.2f}; купол x {centre[:, 0].min():.2f}"
          f" .. {centre[:, 0].max():.2f} z {centre[:, 2].min():.2f} .. {centre[:, 2].max():.2f}")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, val, cmap, label, norm in (
            (axes[0], speed, "viridis", "|U| / V", None),
            (axes[1], cp, "RdBu_r", "Cp = p / (ρV²/2)", TwoSlopeNorm(0.0, vmin=min(cp.min(), -1), vmax=max(cp.max(), 1)))):
        art = ax.tripcolor(tri, val, cmap=cmap, norm=norm, shading="gouraud")
        trace(ax, centre)
        if centre is not None:
            window(ax, centre)
        ax.set_aspect("equal")
        ax.set_xlabel("x, м")
        ax.set_ylabel("z, м (поток вверх)")
        fig.colorbar(art, ax=ax, shrink=0.8, label=label)
    fig.suptitle(f"{NAME}, {PASS.name}, {step.name}: сечение y = 0")
    fig.tight_layout()
    save(fig, "сечение")


def streamlines(centre):
    step = last(PASS / "postProcessing" / "sets" / "streamlines") if PASS else None
    files = sorted(step.glob("*.vtp")) if step else []
    if not files:
        return
    fig, ax = plt.subplots(figsize=(7, 9))
    for f in files:
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
        ax.add_collection(LineCollection(segs, array=np.array(vals), cmap="viridis", linewidths=0.8,
                                         norm=plt.Normalize(0, 1.5)))
        ax.autoscale_view()
    trace(ax, centre)
    if centre is not None:
        window(ax, centre)
    ax.set_aspect("equal")
    ax.set_xlabel("x, м")
    ax.set_ylabel("z, м (поток вверх)")
    ax.set_title(f"{NAME}, {PASS.name}, {step.name}: линии тока, цвет |U|/V")
    save(fig, "линии_тока")


def shapes():
    frames = sorted((OUT / "кадры").glob(f"{NAME}_проход*.npy"))
    story = OUT / f"{NAME}.json"
    if not frames or not story.exists():
        return
    first = json.loads(story.read_text(encoding="utf-8"))[0]
    g, r = first["gores"], first["rings"]
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = plt.get_cmap("Blues")(np.linspace(0.35, 1.0, len(frames)))
    for f, c in zip(frames, colors):
        pos = np.load(f)
        grid = pos[:(r + 1) * g].reshape(r + 1, g, 3)
        # два меридиана в плоскости наклона (0° и 180°) -- профиль купола
        prof = np.vstack([grid[::-1, g // 2], grid[:, 0]])
        ax.plot(prof[:, 0], prof[:, 2], "-", color=c, lw=1.5, label=f.stem.split("_")[-1])
        skirt = grid[r]
        ax.plot(skirt[:, 0], skirt[:, 2], ".", color=c, ms=1)
    ax.set_aspect("equal")
    ax.set_xlabel("x, м")
    ax.set_ylabel("z, м (поток вверх)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"{NAME}: профиль купола по проходам (плоскость наклона)")
    save(fig, "форма")

history()
if PASS is None:
    print(f"нет сохранённых полей в {FIELDS}")
else:
    c = canopy_pressure()
    section(c)
    streamlines(c)
shapes()
