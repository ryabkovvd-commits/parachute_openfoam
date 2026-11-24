"""Чтение .vtp, которые пишет функция surfacePressure в OpenFOAM.

Отдельный модуль, а не копия внутри каждого инструмента.  Раньше эти три
функции лежали в `stiffness.py`, `shape_trace.py` и `follower.py`, и попытка
взять их оттуда импортом запускала весь счёт этого файла -- на этом однажды
случайно стартовал четырёхмиллионный прогон.  Здесь исполняемого кода нет.
"""
import base64
import re
from pathlib import Path

import numpy as np

#: Скоростной напор в кинематическом виде: OpenFOAM для несжимаемой задачи
#: пишет p/rho, поэтому Cp = p/(V^2/2), без плотности.
RHO, V = 1.225, 9.0
Q, KIN = 0.5 * RHO * V**2, 0.5 * V**2

KINDS = {"Float32": np.float32, "Float64": np.float64,
         "Int32": np.int32, "Int64": np.int64, "UInt8": np.uint8}


def arrays(path):
    """Все DataArray файла: имя -> массив."""
    text = Path(path).read_text(errors="replace")
    out = {}
    header_bytes = 4 if re.search(r'''header_type=["']UInt32["']''', text) else 8
    for m in re.finditer(r"<DataArray([^>]*)>(.*?)</DataArray>", text, re.S):
        attrs, body = m.group(1), m.group(2)
        name = re.search(r'''Name=["']([^"']*)["']''', attrs)
        kind = re.search(r'''type=["']([^"']*)["']''', attrs)
        if not name or not kind or kind.group(1) not in KINDS:
            continue
        if re.search(r'''format=["']ascii["']''', attrs):
            data = np.fromstring(body, sep=" ", dtype=KINDS[kind.group(1)])
        else:
            blob = base64.b64decode("".join(body.split()))
            data = np.frombuffer(blob[header_bytes:], dtype=KINDS[kind.group(1)])
        comps = re.search(r'''NumberOfComponents=["'](\d+)["']''', attrs)
        if comps and int(comps.group(1)) > 1:
            data = data.reshape(-1, int(comps.group(1)))
        out.setdefault(name.group(1), data)
    return out


def faces_of(path):
    """Центры, площади, единичные нормали и давление на гранях."""
    got = arrays(path)
    points = np.asarray(got["Points"], dtype=np.float64)
    conn = np.asarray(got["connectivity"], dtype=np.int64)
    offs = np.asarray(got["offsets"], dtype=np.int64)
    starts = np.concatenate(([0], offs[:-1]))
    centre = np.empty((len(offs), 3))
    normal = np.empty((len(offs), 3))
    for k, (s0, s1) in enumerate(zip(starts, offs)):
        poly = points[conn[s0:s1]]
        mid = poly.mean(axis=0)
        normal[k] = 0.5 * np.cross(poly - mid,
                                   np.roll(poly, -1, axis=0) - mid).sum(axis=0)
        centre[k] = mid
    area = np.linalg.norm(normal, axis=1)
    return centre, area, normal / np.maximum(area, 1e-30)[:, None], \
        np.asarray(got["p"], dtype=np.float64).ravel()


def nearest(source, target, block=400):
    """Для каждой точки source -- номер ближайшей в target."""
    idx = np.empty(len(source), dtype=int)
    for s0 in range(0, len(source), block):
        chunk = source[s0:s0 + block]
        idx[s0:s0 + block] = np.linalg.norm(
            chunk[:, None, :] - target[None, :, :], axis=2).argmin(axis=1)
    return idx


def delta_cp(folder, kin=KIN, master="master.vtp", slave="slave.vtp", *, strict=True):
    """Перепад по граням: наветренная сторона минус подветренная.

    Стороны сопоставляются по центрам граней, а не по номерам.
    """
    centre_m, area_m, unit_m, p_m = faces_of(Path(folder) / master)
    centre_s, area_s, unit_s, p_s = faces_of(Path(folder) / slave)
    if not len(centre_m) or not len(centre_s) or kin<=0 or not np.isfinite(kin):
        raise ValueError('Empty fabric surface or invalid kinetic pressure')
    mate=nearest(centre_m,centre_s)
    if not np.isfinite(np.r_[p_m,p_s]).all():
        raise ValueError('Non-finite fabric pressure')
    if strict:
        back=nearest(centre_s,centre_m)
        if len(centre_m)!=len(centre_s) or np.any(np.linalg.norm(centre_m-centre_s[mate],axis=1)>1e-4) or np.any(back[mate]!=np.arange(len(centre_m))):
            raise ValueError('Unpaired fabric faces; reject load transfer and inspect mesh')
        if np.any(area_m<=0) or not np.allclose(area_m,area_s[mate],rtol=1e-5,atol=1e-12) or np.any(np.einsum('ij,ij->i',unit_m,unit_s[mate])>-.99):
            raise ValueError('Fabric pair areas/normals disagree')
    return (centre_m, area_m, unit_m,
            (p_m - p_s[mate]) / kin)


def nodal_load(mesh, folder, q=Q, kin=KIN,
               master="master.vtp", slave="slave.vtp", *, strict=True):
    """Узловые силы стержневой модели по перепаду на гранях потока.

    Сила грани dCp*S*q*n уходит на четыре угла ближайшей панели с весами,
    обратными расстоянию.  Равнодействующая сохраняется точно, момент --
    приближённо.  Коуш (mesh.confluence[0]) закреплён и нагрузки не несёт.

    Возвращает (узловые силы, равнодействующая по граням, dCp по граням).
    """
    centre_m, area_m, unit_m, dcp = delta_cp(folder, kin, master, slave,strict=strict)
    face_force = (dcp * area_m * q)[:, None] * unit_m
    centroids = mesh.position[mesh.quads].mean(axis=1)
    near = nearest(centre_m, centroids)
    corners = mesh.quads[near]
    legs = mesh.position[corners] - centre_m[:, None, :]
    weight = 1.0 / np.maximum(np.linalg.norm(legs, axis=2), 1e-6)
    weight /= weight.sum(axis=1, keepdims=True)
    load = np.zeros_like(mesh.position)
    np.add.at(load, corners.ravel(),
              (weight[:, :, None] * face_force[:, None, :]).reshape(-1, 3))
    load[mesh.confluence[0]] = 0.0
    return load, face_force.sum(axis=0), dcp
