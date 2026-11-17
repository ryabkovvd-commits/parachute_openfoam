"""Inspect cord attachments, confluence loads and segmented cords in saved cases."""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from cords import SAMPLES
from nos6_geometry import closed_circular
from parafoil.inflation.materials import Webbing
from parafoil.inflation.relax import _rod_tensions


def stl_skirt(path, rings, gores, subdivision=12):
    """Recover original skirt nodes from the fixed triangle order of export_surface."""
    # Quad corner 3 is vertex 1 of the second triangle. It lies on the skirt.
    first = (rings * subdivision - 1) * gores * subdivision
    offsets = 6 * (first + np.arange(gores) * subdivision) + 4
    nodes = []
    index = 0
    with path.open(encoding="ascii") as stream:
        for line in stream:
            line = line.lstrip()
            if not line.startswith("vertex "):
                continue
            if index == offsets[len(nodes)]:
                nodes.append([float(value) for value in line.split()[1:]])
                if len(nodes) == gores:
                    return np.asarray(nodes)
            index += 1
    raise ValueError(f"Cannot recover all {gores} skirt nodes from {path}")


def save_rows(path, header, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def confluence_loads(work, name, records, output):
    """Recover elastic suspension loads on the common node from saved shapes."""
    if not records:
        raise ValueError("No completed passes")
    config = records[0]
    diameter = np.sqrt(24 / np.pi)
    mesh = closed_circular(
        diameter, config["lines"], gores=config["gores"], rings=config["rings"],
        line_stride=config["stride"], shear_ratio=config["shear"],
    )
    lines = mesh.line_rods
    frames = work / "out" / "кадры"
    rows = []
    for record in records:
        step = record["pass"]
        frame = frames / f"{name}_проход{step:02d}.npy"
        if not frame.is_file():
            print(f"Pass {step}: missing frame {frame}", file=sys.stderr)
            continue
        position = np.load(frame, allow_pickle=False)
        if position.shape != mesh.position.shape or not np.isfinite(position).all():
            raise ValueError(f"Invalid shape in {frame}")
        tension, direction = _rod_tensions(mesh, position)
        # Each line rod ends at the confluence; pull there is toward the skirt.
        force = -(tension[lines, None] * direction[lines]).sum(axis=0)
        rows.append((step, *force, np.linalg.norm(force), tension[lines].max()))
    if not rows:
        raise ValueError(f"No saved pass frames in {frames}")

    output.mkdir(parents=True, exist_ok=True)
    table = output / "confluence_loads.csv"
    save_rows(table, ("pass", "Fx_N", "Fy_N", "Fz_N", "node_load_N", "max_cord_N"), rows)
    print("pass  node_load_N  max_cord_N")
    for step, _, _, _, load, maximum in rows:
        print(f"{step:4d}  {load:11.1f}  {maximum:10.1f}")
    print(table)


def plot_motion(initial, final, anchor, junctions, output):
    start, end = initial[junctions], final[junctions]
    shifts = np.linalg.norm(end - start, axis=1)
    line_ids = np.arange(1, len(start) + 1)
    rows = [(i, *a, *b, 1000 * shift) for i, a, b, shift in zip(line_ids, start, end, shifts)]
    save_rows(output / "cord_motion.csv",
              ("cord", "x_start_m", "y_start_m", "z_start_m", "x_end_m", "y_end_m", "z_end_m", "shift_mm"), rows)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    for ax, columns, xlabel, ylabel in ((axes[0], (0, 2), "x, м", "z, м"),
                                         (axes[1], (0, 1), "x, м", "y, м")):
        for edge, ends, color, label in ((initial, start, "#818891", "Начало"),
                                          (final, end, "#066f93", "Конец")):
            ax.plot(*np.vstack((edge, edge[0]))[:, columns].T, color=color, lw=1.2, label=label)
            ax.add_collection(LineCollection(np.stack((np.broadcast_to(anchor, ends.shape), ends), axis=1)[:, :, columns],
                                             colors=color, linewidths=0.65, alpha=0.7))
            ax.scatter(*ends[:, columns].T, color=color, s=10, zorder=3)
        ax.set(xlabel=xlabel, ylabel=ylabel, aspect="equal")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
    axes[2].bar(line_ids, 1000 * shifts, color="#066f93")
    axes[2].set(xlabel="Номер стропы", ylabel="Перемещение крепления, мм")
    axes[2].set_xticks(line_ids[::2])
    axes[2].grid(axis="y", alpha=0.22)
    fig.suptitle("Положение строп в начале и конце расчёта")
    fig.savefig(output / "cord_motion.png", dpi=150)
    plt.close(fig)
    largest = shifts.argmax()
    print(f"Смещение креплений, мин / медиана / макс: {1000 * shifts.min():.2f} / "
          f"{1000 * np.median(shifts):.2f} / {1000 * shifts.max():.2f} мм. Максимум у стропы {largest + 1}.")


def read_sample_forces(folder, count, rho):
    files = sorted(path for path in folder.glob("*.dat") if path.stat().st_size)
    if not files:
        return None
    rows = np.vstack([np.loadtxt(path, ndmin=2) for path in files])
    ids = rows[:, 0].astype(int)
    if rows.shape != (count * SAMPLES, 4) or not np.array_equal(np.sort(ids), np.arange(len(rows))):
        raise ValueError(f"Incomplete or duplicate cord force samples in {folder}")
    return rho * rows[np.argsort(ids), 1:].reshape(count, -1, 3)


def cord_equilibrium(start, end, samples, segments, length, ea):
    """Elastic tension-only cable with fixed ends and a frozen load in material coordinates."""
    forces = np.zeros((segments + 1, 3))
    positions = (np.arange(len(samples)) + 0.5) * (segments / len(samples))
    left = positions.astype(int)
    right_weight = positions - left
    np.add.at(forces, left, (1 - right_weight)[:, None] * samples)
    np.add.at(forces, left + 1, right_weight[:, None] * samples)

    # At an interior node T_right - T_left + F_node = 0. Only the first
    # tension is unknown. The other tensions depend on the cumulative loads.
    cumulative = np.vstack((np.zeros(3), np.cumsum(forces[1:-1], axis=0)))
    chord = end - start
    chord_length = np.linalg.norm(chord)
    chord_axis = chord / chord_length
    load_scale = max(np.linalg.norm(samples, axis=1).sum(), 1e-4)
    first = cumulative.mean(axis=0) + max(ea * (chord_length / length - 1), 0.25 * load_scale) * chord_axis
    unloaded_segment = length / segments
    eye = np.eye(3)
    floor = 1e-10 * load_scale

    def state(tension_start):
        tension = tension_start - cumulative
        norm = np.sqrt(np.sum(tension * tension, axis=1) + floor**2)
        direction = tension / norm[:, None]
        edges = unloaded_segment * (direction + tension / ea)
        return edges.sum(axis=0) - chord, edges, norm, direction

    for _ in range(160):
        residual, edges, tensions, direction = state(first)
        error = np.linalg.norm(residual)
        if error <= 1e-9 * length:
            path = start + np.vstack((np.zeros(3), np.cumsum(edges, axis=0)))
            offset = path - start
            transverse = offset - (offset @ chord_axis)[:, None] * chord_axis
            return np.linalg.norm(transverse, axis=1).max(), tensions[0]
        jacobian = unloaded_segment * (
            ((eye - direction[:, :, None] * direction[:, None, :]) / tensions[:, None, None]).sum(axis=0)
            + (segments / ea) * eye)
        correction = np.linalg.solve(jacobian, residual)
        for damping in 0.5 ** np.arange(30):
            trial = first - damping * correction
            if np.linalg.norm(state(trial)[0]) < error:
                first = trial
                break
        else:
            break
    raise ValueError("Cord equilibrium did not converge")


def compare(work, name, pass_number, rings, gores, junctions, anchor, length, rho, output):
    folder = work / "fields" / name / f"pass{pass_number:02d}"
    times = folder / "postProcessing" / "cords"
    folders = [path for path in times.glob("*") if path.is_dir() and path.name.replace(".", "", 1).isdigit()]
    if not folders:
        print(f"Сравнение 20/40 требует сохранённых сил в {times}")
        return
    source = max(folders, key=lambda path: float(path.name))
    loads = read_sample_forces(source, len(junctions), rho)
    if loads is None:
        print(f"Нет файлов сил строп в {source}")
        return
    # The distributed loads belong to the geometry that OpenFOAM saw in this pass.
    attachments = stl_skirt(folder / "canopy.stl", rings, gores)[junctions]
    ea = Webbing.suspension_cord().rod().stiffness
    result = []
    for index, (point, samples) in enumerate(zip(attachments, loads), start=1):
        try:
            shift20, tension20 = cord_equilibrium(point, anchor, samples, 20, length, ea)
            shift40, tension40 = cord_equilibrium(point, anchor, samples, 40, length, ea)
        except (ValueError, np.linalg.LinAlgError) as error:
            raise ValueError(f"Cord {index}: {error}") from error
        result.append((index, 1000 * shift20, 1000 * shift40, tension20, tension40))
    save_rows(output / "cord_20_40.csv",
              ("cord", "bow20_mm", "bow40_mm", "skirt_tension20_N", "skirt_tension40_N"), result)
    rows = np.asarray(result)
    print(f"Проход {pass_number}, поле сил {source.name}. Максимальный прогиб 20 / 40: "
          f"{rows[:, 1].max():.2f} / {rows[:, 2].max():.2f} мм. Максимальная разница по стропам: "
          f"{np.max(np.abs(rows[:, 1] - rows[:, 2])):.2f} мм.")
    print(f"Максимальное натяжение у кромки 20 / 40: {rows[:, 3].max():.2f} / {rows[:, 4].max():.2f} Н.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--name", default="nos6_held")
    parser.add_argument("--loads", action="store_true",
                        help="recover confluence loads from saved Nosarev 6 m² shapes")
    parser.add_argument("--initial-npy", type=Path, help="saved CFD input geometry instead of pass01/canopy.stl")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rho", type=float, default=1.225)
    args = parser.parse_args()
    work, name = args.work, args.name
    if args.loads:
        try:
            records = json.loads((work / "out" / f"{name}.json").read_text(encoding="utf-8"))
            confluence_loads(work, name, records, args.output or work / "out")
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))
        return

    output = args.output or work / "pics"
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((work / "out" / f"{name}.json").read_text(encoding="utf-8"))
    config = rows[-1]
    rings, gores, stride = (config[key] for key in ("rings", "gores", "stride"))
    skirt = slice(rings * gores, (rings + 1) * gores)
    final = np.load(work / "out" / f"{name}_shape.npy", allow_pickle=False)
    anchor = final[(rings + 1) * gores]
    if args.initial_npy:
        initial = np.load(args.initial_npy, allow_pickle=False)[skirt]
    else:
        initial = stl_skirt(work / "fields" / name / "pass01" / "canopy.stl", rings, gores)
    junctions = np.arange(0, gores, stride)
    plot_motion(initial, final[skirt], anchor, junctions, output)
    compare(work, name, int(config["pass"]), rings, gores, junctions, anchor,
            float(config["lines"]), args.rho, output)


if __name__ == "__main__":
    main()
