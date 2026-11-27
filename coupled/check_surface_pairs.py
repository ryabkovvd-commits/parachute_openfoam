"""Read-only diagnosis of a failed pressure transfer; never starts a solver.

python coupled/check_surface_pairs.py run/nos6wall_refine/JOB/a50
"""
import argparse
import json
from pathlib import Path

import numpy as np
import vtp

DISTANCE_TOLERANCE = 1e-4


def nearest(source, target):
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        # Bound temporary distance-array memory for large saved surfaces.
        block = max(1, min(400, 32_000_000 // max(32 * len(target), 1)))
        return vtp.nearest(source, target, block=block)
    return cKDTree(target).query(source)[1]


def diagnose(master, slave):
    c, a, n, p = master
    cs, areas, normals, ps = slave
    if not len(c) or not len(cs):
        raise ValueError('An exported surface is empty')
    if not all(np.isfinite(x).all() for x in [c, cs, a, areas, n, normals, p, ps]):
        raise ValueError('Non-finite surface geometry or pressure')
    forward, back = nearest(c, cs), nearest(cs, c)
    distance = np.linalg.norm(c - cs[forward], axis=1)
    reverse_distance = np.linalg.norm(cs - c[back], axis=1)
    mutual = back[forward] == np.arange(len(c))
    paired = (distance <= DISTANCE_TOLERANCE) & mutual
    paired_slave = np.zeros(len(cs), dtype=bool)
    paired_slave[forward[paired]] = True
    same_count = len(c) == len(cs)
    areas_agree = bool(np.all(a > 0) and np.allclose(a, areas[forward], rtol=1e-5, atol=1e-12))
    normals_agree = bool(np.all(np.einsum('ij,ij->i', n, normals[forward]) <= -.99))
    return dict(master_faces=len(c), slave_faces=len(cs), same_face_count=same_count,
                tolerance_m=DISTANCE_TOLERANCE,
                max_master_to_slave_m=float(distance.max()),
                max_slave_to_master_m=float(reverse_distance.max()),
                master_too_far=int((distance > DISTANCE_TOLERANCE).sum()),
                slave_too_far=int((reverse_distance > DISTANCE_TOLERANCE).sum()),
                non_mutual_master=int((~mutual).sum()),
                master_unpaired=int((~paired).sum()), slave_unpaired=int((~paired_slave).sum()),
                paired_master_area_fraction=float(a[paired].sum() / a.sum()) if a.sum() > 0 else None,
                pair_geometry_pass=bool(same_count and paired.all()),
                paired_areas_agree=areas_agree, paired_normals_opposite=normals_agree)


def pressure_folder(work, number, name):
    root = work/'fields'/name/f'pass{number:02d}'/'postProcessing/surfacePressure'
    candidates = []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        try:
            time = float(folder.name)
        except ValueError:
            continue
        if np.isfinite(time) and any((folder/f).exists() for f in ['master.vtp', 'slave.vtp']):
            candidates.append((time, folder))
    if not candidates:
        raise FileNotFoundError(f'No saved pressure surfaces in {root}')
    return max(candidates, key=lambda item: item[0])[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    parser.add_argument('--pass', dest='number', type=int)
    parser.add_argument('--name', default='nos6_held')
    args = parser.parse_args(argv)
    work = args.work.resolve()
    number = args.number
    failure = work/'out'/f'{args.name}.failed'
    if number is None:
        if not failure.is_file():
            parser.error('No failure marker; specify --pass NUMBER')
        record = json.loads(failure.read_text(encoding='utf-8'))
        number = int(record['last_completed_pass']) + 1
    if number < 1:
        parser.error('Pass must be positive')
    try:
        folder = pressure_folder(work, number, args.name)
        report = diagnose(vtp.faces_of(folder/'master.vtp'), vtp.faces_of(folder/'slave.vtp'))
    except (OSError, ValueError, KeyError) as error:
        print(f'Could not inspect pass {number}: {error}')
        return 2
    print(f'Pass {number}; saved time {folder.name}')
    print(f'Folder: {folder}')
    print(f'Faces master/slave: {report["master_faces"]} / {report["slave_faces"]}')
    print(f'Distance tolerance: {DISTANCE_TOLERANCE:.1e} m')
    print(f'Max distance master->slave / slave->master: {report["max_master_to_slave_m"]:.6g} / {report["max_slave_to_master_m"]:.6g} m')
    print(f'Too far master/slave: {report["master_too_far"]} / {report["slave_too_far"]}')
    print(f'Non-mutual master matches: {report["non_mutual_master"]}')
    print(f'Unpaired master/slave: {report["master_unpaired"]} / {report["slave_unpaired"]}')
    fraction = report['paired_master_area_fraction']
    print('Paired master area: ' + (f'{100*fraction:.6f}%' if fraction is not None else 'undefined'))
    print(f'Pair geometry / areas / normals: {report["pair_geometry_pass"]} / {report["paired_areas_agree"]} / {report["paired_normals_opposite"]}')
    print('Read-only inspection; no CFD, history changes or load acceptance.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
