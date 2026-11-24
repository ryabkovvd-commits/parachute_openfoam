"""Cluster cs.py patch for cord drag and STL export."""

from pathlib import Path
from shutil import copyfile

import numpy as np

SAMPLES = 64


def write_cords(mesh, case, head, width, cd):
    cords = mesh.position[mesh.rods[mesh.line_rods]].reshape(-1, 3)
    text = "\n".join("(%.12g %.12g %.12g)" % tuple(p) for p in cords)
    (case / "constant" / "cords").write_text(f"{len(cords)}\n(\n{text}\n)\n", encoding="ascii")
    template = Path(__file__).with_name("cordsFv.txt").read_text(encoding="ascii")
    flags = {"CORD_WIDTH": width, "CORD_CD": cd, "CORD_SAMPLES": SAMPLES}
    for key, value in flags.items():
        template = template.replace(key, str(value))
    (case / "system" / "fvOptions").write_text(
        head("dictionary", "fvOptions", "system") + template, encoding="ascii")


def read_cord_load(mesh, folder, rho):
    folder = folder.parent.parent / "cords" / folder.name
    rows = [np.loadtxt(path, ndmin=2) for path in folder.glob("*.dat") if path.stat().st_size]
    rods = mesh.rods[mesh.line_rods]
    how_many = len(rods) * SAMPLES
    if not rows:
        raise ValueError(f"Missing cord forces: {folder}")
    values = np.vstack(rows)
    ids = values[:, 0].astype(int)
    if len(ids) != how_many or not np.array_equal(np.sort(ids), np.arange(how_many)):
        raise ValueError(f"Missing or duplicate cord samples: {folder}")
    weight = ((ids % SAMPLES + 0.5) / SAMPLES)[:, None]
    forces = rho * values[:, 1:]
    ends = rods[ids // SAMPLES]
    load = np.zeros_like(mesh.position)
    np.add.at(load, ends[:, 0], (1 - weight)*forces)
    np.add.at(load, ends[:, 1], weight*forces)
    return load


def append_cords_stl(mesh, width, path):
    cords = mesh.position[mesh.rods[mesh.line_rods]]
    angles = np.arange(12) * (2 * np.pi / 12)
    facets = []
    for start, end in cords:
        axis = end - start
        axis /= np.linalg.norm(axis)
        right = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))])
        right /= np.linalg.norm(right)
        rim = width/2 * (np.outer(np.cos(angles), right)
                         + np.outer(np.sin(angles), np.cross(axis, right)))
        for a, b in zip(rim, np.roll(rim, -1, axis=0)):
            p, q, r, s = start+a, start+b, end+b, end+a
            for x, y, z in ((p, q, r), (p, r, s), (start, q, p), (end, s, r)):
                normal = np.cross(y - x, z - x)
                normal /= np.linalg.norm(normal)
                facets.append("  facet normal %.9g %.9g %.9g\n    outer loop" % tuple(normal))
                for vertex in (x, y, z):
                    facets.append("      vertex %.9g %.9g %.9g" % tuple(vertex))
                facets.append("    endloop\n  endfacet")
    footer = b"endsolid canopy"
    with Path(path).open("r+b") as stream:
        stream.seek(-len(footer), 2)
        if stream.read() != footer:
            raise ValueError(f"Expected closed canopy STL: {path}")
        stream.seek(-len(footer), 2)
        stream.write(("\n".join(facets) + "\n").encode("ascii") + footer)


if __name__ == "__main__":
    width = float(FLAG["--cord-mm"]) / 1000
    cd = float(FLAG.get("--cord-cd", 1.2))
    cx_tol = float(FLAG.get("--cx-tol", 0.003))
    if not all(np.isfinite(value) and value > 0 for value in (width, cd, cx_tol)):
        raise ValueError("Cord width, Cd and Cx tolerance must be positive")
    sub("import vtp\n", "import vtp\nfrom cords import write_cords, read_cord_load, copyfile\n")
    sub("for step in range(len(history) + 1, PASSES + 1):\n",
        "for step in range(len(history) + 1, PASSES + 1):\n"
        f"    write_cords(mesh, CASE, head, {width!r}, {cd!r})\n")
    sub("    load, total, dcp = nodal_load(mesh, folder)\n",
        "    load, total, dcp = nodal_load(mesh, folder)\n"
        "    cord_load = read_cord_load(mesh, folder, RHO)\n"
        "    load += cord_load\n    total += cord_load.sum(axis=0)\n"
        "    r = mesh.position - mesh.position[mesh.confluence[0]]\n"
        "    cord_mz = float(np.cross(r, cord_load).sum(axis=0) @ PIVOT / (Q * S0 * D0))\n"
        "    say(f'cords: dCx={cord_load.sum(axis=0) @ ONSET / (Q * S0):.6f}, dMz={cord_mz:.6f}')\n")
    sub('    row["cx"] = row["drag"] / (Q * S0)\n',
        '    row["cx"] = row["drag"] / (Q * S0)\n'
        '    row["cord_cx"] = float(cord_load.sum(axis=0) @ ONSET / (Q * S0))\n'
        '    row["cord_cy"] = float(cord_load.sum(axis=0) @ SIDE / (Q * S0))\n'
        '    row["cord_mz"] = cord_mz\n'
        f'    row["cord_width_m"] = {width!r}\n'
        f'    row["cord_cd"] = {cd!r}\n')
    sub('    row["cx"] = row["drag"] / (Q * S0)\n',
        '    row["cx"] = row["drag"] / (Q * S0)\n'
        '    row["cord_anchor_force_N"] = cord_load[mesh.confluence].sum(axis=0).tolist()\n')
    sub("    np.save(SHAPE, mesh.position)\n",
        '    copyfile(CASE / "constant/triSurface/canopy.stl", FIELDS / NAME / f"pass{step:02d}/canopy.stl")\n'
        "    np.save(SHAPE, mesh.position)\n")
    MODE.append(f"Cord drag: b={width*1000:g} mm, Cd={cd:g}")
    MODE.append(f"Maximum Cx change: {cx_tol:g} across two consecutive passes")
