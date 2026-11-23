"""Meridian pressures and profiles from saved nos6_held results; no CFD run.

python coupled/pressure_table.py run/nos6held/8881/a00 --V=30
python coupled/pressure_table.py RUN0 RUN15 RUN40 --V=30 --pics=pics/pressure_compare

The compact terminal table has 11 positions. CSV and figures use the full
meridians of the smoothed input geometry, with independently sampled pressures
on both CFD patches. A missing input shape is an error, never replaced by the
post-relaxation shape from the same pass. See README for sampling limitations.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import vtp
from parafoil.inflation.exchange import _catmull_rom


def numeric_dirs(folder):
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.iterdir() if re.fullmatch(r'\d+(?:\.\d+)?', p.name)),
                  key=lambda p: float(p.name))


def locate(args, run):
    out = args.out_dir or run / 'out'
    history = json.loads((out / (args.name + '.json')).read_text(encoding='utf-8'))
    rows = {int(row['pass']): row for row in history}
    if args.surface_dir:
        if args.pass_number is None:
            raise ValueError('--surface-dir requires --pass')
        step, number = args.surface_dir, args.pass_number
    else:
        fields = (args.fields_dir or run / 'fields') / args.name
        found = []
        for folder in fields.glob('pass*'):
            match = re.fullmatch(r'pass(\d+)', folder.name)
            if not match:
                continue
            number = int(match[1])
            if args.pass_number is not None and number != args.pass_number:
                continue
            for step in numeric_dirs(folder / 'postProcessing/surfacePressure'):
                if (step/'master.vtp').is_file() and (step/'slave.vtp').is_file() and number in rows:
                    found.append((number, float(step.name), step))
        if not found:
            raise FileNotFoundError('No matching history entry and pair of surfacePressure VTP files')
        number, _, step = max(found)
    if number not in rows:
        raise ValueError('Selected CFD pass is absent from the JSON history')
    row = rows[number]
    if args.shape:
        shape = args.shape
    else:
        # Current driver stores input of pass N as the relaxed frame of N-1.
        # An exact input snapshot, if archived, is preferable.
        exact = step.parents[2] / 'input_shape.npy'
        candidates = [exact] if exact.is_file() else []
        if not candidates and number > 1:
            suffix = re.compile(r'(?:проход|pass|frame)' + f'{number-1:02d}' + r'\.npy$')
            candidates = [p for p in out.rglob(args.name + '_*.npy') if suffix.search(p.name)]
        if len(candidates) != 1:
            raise FileNotFoundError(
                f'Need the CFD INPUT shape for pass {number}. Found {len(candidates)} candidates. '
                'Use --shape=PATH to the actual input shape; the final shape is unsuitable.')
        shape = candidates[0]
    journal_path = out / (args.name + '.txt')
    journal = journal_path.read_text(encoding='utf-8') if journal_path.is_file() else ''
    speed_match = re.search(r'\bV\s*=\s*([0-9.eE+-]+)', journal)
    velocity = args.V if args.V is not None else (float(speed_match[1]) if speed_match else None)
    if velocity is None or velocity <= 0:
        raise ValueError('Provide --V with the actual CFD inlet speed')
    return row, step, shape, velocity


def ray(grid, pole, azimuth, vent, subdivide):
    """Sample the existing exporter's angular/radial Catmull-Rom geometry."""
    g = grid.shape[1]
    parameter = azimuth * g / (2*np.pi)
    j, t = int(np.floor(parameter)), parameter % 1
    p0, p1, p2, p3 = [grid[:, k % g] for k in (j-1, j, j+1, j+2)]
    control = .5*(2*p1 + (p2-p0)*t + (2*p0-5*p1+4*p2-p3)*t*t
                  + (-p0+3*p1-3*p2+p3)*t*t*t)
    curve = _catmull_rom(control, subdivide, periodic=False)
    radius = np.linspace(vent, 1., len(curve))
    if pole is not None:
        # Closed-pole exporter uses straight fan triangles, not a radial spline.
        fan = np.linspace(pole, curve[0], subdivide+1)[:-1]
        curve = np.vstack((fan, curve))
        radius = np.r_[np.linspace(0., vent, subdivide+1)[:-1], radius]
    return radius, curve


def ray_normal(grid, pole, azimuth, vent, subdivide, anchor):
    radius, curve = ray(grid,pole,azimuth,vent,subdivide)
    delta = 1e-4
    _, plus = ray(grid,pole,azimuth+delta,vent,subdivide)
    _, minus = ray(grid,pole,azimuth-delta,vent,subdivide)
    normal = np.cross(np.gradient(curve,axis=0),plus-minus)
    length = np.linalg.norm(normal,axis=1)
    good = length>1e-12
    normal[good] /= length[good,None]
    fallback = grid[0].mean(0)-anchor
    fallback /= np.linalg.norm(fallback)
    normal[~good] = fallback
    normal[np.einsum('ij,ij->i',normal,curve-anchor)<0] *= -1
    return radius,curve,normal


def normal_diagnostics(args, run):
    """Report the radial-normal assumption without assigning physical sides.

    A minority of radial signs can arise from cut facets or non-star-shaped
    geometry. Neither cause can be identified from this sign test alone.
    """
    row, step, shape, velocity = locate(args,run)
    pos = np.load(shape,allow_pickle=False)
    count = (int(row['rings'])+1)*int(row['gores'])
    expected = count+(2 if args.pole=='closed' else 1)
    if pos.shape!=(expected,3) or not np.isfinite(pos).all():
        raise ValueError('Input shape does not match the selected topology')
    anchor = pos[count]
    theta = np.radians(float(row['hold']))
    axis = np.array([np.sin(theta),0.,np.cos(theta)])
    e1 = np.array([np.cos(theta),0.,-np.sin(theta)])
    folder = args.pics or run/'pics'
    folder.mkdir(parents=True,exist_ok=True)
    report = dict(run=str(run),pass_number=int(row['pass']),angle_deg=float(row['hold']),
                  input_shape=str(shape),surface=str(step),patches={},
                  note='Radial signs relative to confluence, not a proof of inside/outside or self-intersection. No pressure sampling or CFD changes.')
    fig,axes = plt.subplots(1,2,figsize=(11,5),sharex=True,sharey=True)
    stem = f"{args.name}_normals_pass{int(row['pass']):02d}"
    stem += '_'+re.sub(r'[^\w.-]+','_',run.parent.name+'_'+run.name)
    for ax,name in zip(axes,['master','slave']):
        centres,area,normals,_ = vtp.faces_of(step/(name+'.vtp'))
        if area.sum()<=0 or not np.isfinite(centres).all() or not np.isfinite(normals).all():
            raise ValueError(f'{name}: invalid face geometry')
        projection = np.einsum('ij,ij->i',normals,centres-anchor)
        positive = projection>0
        fraction = float(area[positive].sum()/area.sum())
        majority_positive = fraction>=.5
        minority = (positive!=majority_positive) | (projection==0) | (area<=0)
        relative = centres-anchor
        xb, hb = relative@e1/args.diameter, relative@axis/args.diameter
        stats = dict(faces=len(area),positive_area_fraction=fraction,
                     minority_faces=int(minority.sum()),minority_area_fraction=float(area[minority].sum()/area.sum()),
                     minority_x_positive_faces=int((minority & (xb>=0)).sum()),
                     minority_x_negative_faces=int((minority & (xb<0)).sum()))
        report['patches'][name] = stats
        table = folder/(stem+'_'+name+'.csv')
        with table.open('w',encoding='utf-8',newline='') as stream:
            writer=csv.writer(stream)
            writer.writerow(['face','minority_radial_sign','area_m2','n_dot_r_m','x_body_D0','h_from_confluence_D0','x_m','y_m','z_m'])
            writer.writerows((i,int(minority[i]),area[i],projection[i],xb[i],hb[i],*centres[i]) for i in range(len(area)))
        stats['face_table']=str(table)
        ax.scatter(xb[~minority],hb[~minority],s=1,color='.65',rasterized=True,label='Преобладающий радиальный знак')
        ax.scatter(xb[minority],hb[minority],s=4,color='#b2182b',rasterized=True,label='Противоположный знак / нулевая грань')
        ax.set_title(f"{name}: противоположная площадь {100*stats['minority_area_fraction']:.2f}%")
        ax.set_xlabel('x′ / D₀');ax.grid(alpha=.2);ax.set_aspect('equal',adjustable='box')
        print(f"{name}: faces={len(area)}, positive_area={fraction:.6f}, minority_faces={stats['minority_faces']}, minority_area={stats['minority_area_fraction']:.6f}")
    axes[0].set_ylabel('h от коуша / D₀');axes[0].legend(fontsize=7)
    fig.suptitle(f"Проверка радиального знака нормалей · α = {report['angle_deg']:g}°")
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(folder/(stem+'.png'),dpi=160);plt.close(fig)
    (folder/(stem+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Диагностика сохранена: {folder/stem}')
    return report


def sample_patch(path, query, query_r, query_normal, centre, axis, meridian, args, velocity):
    centres, area, normals, pressure = vtp.faces_of(path)
    if args.pressure_field != 'p':
        data = vtp.arrays(path)
        if args.pressure_field not in data:
            raise ValueError(f'{path}: field {args.pressure_field} is missing')
        pressure = np.asarray(data[args.pressure_field]).ravel()
    if area.sum()<=0 or not np.isfinite(pressure).all():
        raise ValueError(f'{path}: zero surface area or non-finite pressure')
    projection = np.einsum('ij,ij->i', normals, centres-centre)
    positive = float(area[projection > 0].sum()/area.sum())
    if .01 < positive < .99:
        raise ValueError(f'{path}: ambiguous inside/outside normals ({positive:.1%} positive); '
                         'run --diagnose-normals to inspect this geometric assumption. Do not swap pressure sides by guessing.')
    inside = positive >= .99
    valid = ((projection > 0) == inside) & (area > 0)
    rejected = int((~valid).sum())
    radial = centres-centre
    e1 = np.array([axis[2], 0., -axis[0]])
    azimuth = np.arctan2(radial[:, 1], radial @ e1)
    values, distances = [], []
    for point, r, expected_normal in zip(query, query_r, query_normal):
        phi = meridian if r < 0 else meridian+np.pi
        delta = np.angle(np.exp(1j*(azimuth-phi)))
        keep = (np.abs(delta) <= np.radians(args.sector)) & valid
        if abs(r) < 1e-12:
            keep = valid.copy()
        # Closest centres may lie on small cut facets whose normals are almost
        # tangential to the intended canopy surface. Match the local surface too.
        alignment = (normals @ expected_normal)*(1 if inside else -1)
        keep &= alignment>=np.cos(np.radians(args.normal_angle))
        candidates = np.flatnonzero(keep)
        if len(candidates) < args.neighbours:
            raise ValueError('Not enough compatible faces near the meridian; inspect geometry or adjust --sector/--normal-angle')
        distance = np.linalg.norm(centres[candidates]-point, axis=1)
        order = np.argsort(distance)[:args.neighbours]
        selected, ds = candidates[order], distance[order]
        weight = 1/np.maximum(ds, 1e-12)
        values.append(float(np.average(pressure[selected], weights=weight)))
        distances.append(float(ds[0]))
    return inside, (np.array(values)-args.p_inf)/(.5*velocity**2), np.array(distances), positive, rejected


def extract(args, run):
    row, step, shape_path, velocity = locate(args, run)
    pos = np.load(shape_path, allow_pickle=False)
    g, r = int(row['gores']), int(row['rings'])
    count = (r+1)*g
    expected = count+2 if args.pole == 'closed' else count+1
    if len(pos) != expected or pos.shape[1:] != (3,):
        raise ValueError(f'Shape {shape_path} does not match the selected topology ({expected} nodes)')
    if not np.isfinite(pos).all():
        raise ValueError('Input shape contains non-finite coordinates')
    grid, anchor = pos[:count].reshape(r+1,g,3), pos[count]
    angle = row.get('hold')
    if angle is None:
        raise ValueError('This comparison requires a held angle recorded in JSON')
    theta = np.radians(float(angle))
    axis = np.array([np.sin(theta), 0., np.cos(theta)])
    actual = pos[:count].mean(0)-anchor
    actual /= np.linalg.norm(actual)
    error = np.degrees(np.arccos(np.clip(actual @ axis, -1., 1.)))
    if error > .1:
        raise ValueError(f'Input shape axis differs from held angle by {error:.3g} degrees')
    pole = pos[-1] if args.pole == 'closed' else None
    vent = args.vent_fraction if args.vent_fraction is not None else (1/(r+1) if pole is not None else .05)
    azimuth = np.radians(args.azimuth)
    first = ray_normal(grid, pole, azimuth, vent, args.subdivide, anchor)
    second = ray_normal(grid, pole, azimuth+np.pi, vent, args.subdivide, anchor)
    # Upstream skirt has smaller global z because the flow is in +z.
    if first[1][-1,2] > second[1][-1,2]+1e-8:
        first, second = second, first
        azimuth += np.pi
    fr, fp, fn = first
    br, bp, bn = second
    radius, query = np.r_[-fr[::-1], br[1:] if pole is not None else br], np.vstack((fp[::-1], bp[1:] if pole is not None else bp))
    query_normal = np.vstack((fn[::-1],bn[1:] if pole is not None else bn))
    pressures, diagnostics = {}, {}
    for name in ['master', 'slave']:
        inside, value, distance, fraction, rejected = sample_patch(step/(name+'.vtp'), query, radius, query_normal,
                                                                  anchor, axis, azimuth, args, velocity)
        side = 'in' if inside else 'out'
        if side in pressures:
            raise ValueError('Both patches were classified as the same side of the fabric')
        pressures[side] = value
        diagnostics[name] = dict(side=side, normal_positive_area_fraction=fraction,
                                 excluded_opposite_normal_faces=rejected,
                                 nearest_face_distance_max_D0=float(distance.max()/args.diameter))
    centre = grid[-1].mean(0)
    relative = query-centre
    e1 = np.array([np.cos(theta),0.,-np.sin(theta)])
    x, h = relative @ e1 / args.diameter, relative @ axis / args.diameter
    table = np.c_[radius, pressures['in'], pressures['out'], pressures['in']-pressures['out'], x, h]
    compact_r = np.array([-1.,-.85,-.6,-.4,-.2,0.,.2,.4,.6,.85,1.])
    compact = np.column_stack([compact_r]+[np.interp(compact_r, radius, table[:,j]) for j in range(1,6)])
    if pole is None:
        compact[np.abs(compact_r)<vent, 1:] = np.nan
    metadata = dict(run=str(run), name=args.name, pass_number=int(row['pass']), angle_deg=float(angle),
                    velocity_m_s=velocity, diameter_m=args.diameter, surface=str(step), input_shape=str(shape_path),
                    pressure_field=args.pressure_field, pressure_reference_kinematic=args.p_inf,
                    pole=args.pole, vent_fraction=vent, meridian_azimuth_deg=float(np.degrees(azimuth)%360),
                    sector_half_width_deg=args.sector, neighbours=args.neighbours, diagnostics=diagnostics,
                    local_normal_angle_limit_deg=args.normal_angle,
                    sampling='Inverse-distance interpolation within each meridian sector, with local surface-normal compatibility; no ring averaging',
                    geometry='Smoothed CFD input surface; profile relative to mean skirt centre',
                    normal_filter='Requires >=99% consistent area orientation; opposite-oriented residual faces excluded only from pressure sampling. Folded/ambiguous surfaces require another method.',
                    compact_note='Compact table interpolated from full samples; terminal values rounded only for display')
    return SimpleNamespace(table=table, compact=compact, metadata=metadata, run=run)


def write_table(path, values):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['R','Cp_in','Cp_out','delta_Cp','x_body_D0','h_body_D0'])
        writer.writerows(values)


def reference_data(path, angle):
    if path is None:
        return None
    with path.open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(line for line in stream if not line.startswith('#')))
    rows = [r for r in rows if abs(abs(float(r['alpha_deg']))-abs(angle)) < 1e-6]
    if not rows:
        return None
    result = np.array([[float(r[k]) for k in ['R','p_in','p_out','dp']] for r in rows])
    if abs(angle) < 1e-6 and np.all(result[:,0]>=0):
        result = np.vstack((result, result[result[:,0]>0]*np.array([-1,1,1,1])))
        result = result[np.argsort(result[:,0])]
    return result


def plot_results(results, args, folder):
    fig, axes = plt.subplots(len(results), 2, figsize=(11, 3.8*len(results)), squeeze=False)
    profile, ax_shape = plt.subplots(figsize=(7,5.5))
    for number, result in enumerate(results):
        data, meta = result.table, result.metadata
        case_label = result.run.parent.name+'/'+result.run.name
        ax_p, ax_dp = axes[number]
        keep = [data[:,0]<=0, data[:,0]>=0] if meta['pole']=='open' else [np.ones(len(data),dtype=bool)]
        for branch_index, mask in enumerate(keep):
            for j, label, color in [(1,'Внутри','#2166ac'),(2,'Снаружи','#b2182b')]:
                ax_p.plot(data[mask,0],data[mask,j],color=color,label=label if branch_index==0 else None)
            ax_dp.plot(data[mask,0],data[mask,3],color='#333333',label='OpenFOAM' if branch_index==0 else None)
            ax_shape.plot(data[mask,4],data[mask,5],label=f"{case_label}: α = {meta['angle_deg']:g}°" if branch_index==0 else None,
                          color=plt.get_cmap('tab10')(number%10))
        ref = reference_data(args.reference,meta['angle_deg'])
        if ref is not None:
            for j,color in [(1,'#2166ac'),(2,'#b2182b')]:
                ax_p.errorbar(ref[:,0],ref[:,j],yerr=.05,fmt='o',mfc='white',color=color,ms=4,capsize=2,
                              label='Носарев: '+('внутри' if j==1 else 'снаружи'))
            ax_dp.errorbar(ref[:,0],ref[:,3],yerr=.05,fmt='o',mfc='white',color='#dd8d00',ms=4,capsize=2,
                           label='Носарев, предварительная оцифровка')
        ax_p.set_title(f"{case_label} · α = {meta['angle_deg']:g}°, проход {meta['pass_number']}")
        ax_dp.set_title(f"Перепад давления · {case_label} · α = {meta['angle_deg']:g}°")
        for ax in [ax_p, ax_dp]:
            ax.set_xlabel('R: −1 — передняя кромка; 0 — полюс; +1 — задняя')
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        ax_p.set_ylabel('Cp = (p − p∞) / q')
        ax_dp.set_ylabel('ΔCp = Cp внутри − Cp снаружи')
    settings = {(r.metadata['diameter_m'],r.metadata['velocity_m_s']) for r in results}
    if len(settings)==1:
        diameter, velocity = next(iter(settings))
        fig.suptitle(f'D₀ = {diameter:g} м · V = {velocity:g} м/с',fontsize=11)
        fig.tight_layout(rect=(0,0,1,.96))
    else:
        fig.tight_layout()
    fig.savefig(folder/'pressure_meridians.png',dpi=160)
    plt.close(fig)
    ax_shape.set(xlabel='x′ / D₀',ylabel='h′ / D₀',title='Профили купола в связанной системе координат')
    ax_shape.set_aspect('equal',adjustable='datalim')
    ax_shape.grid(alpha=.25)
    ax_shape.legend()
    profile.tight_layout()
    profile.savefig(folder/'canopy_profiles.png',dpi=160)
    plt.close(profile)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--name',default='nos6_held')
    parser.add_argument('--pass',dest='pass_number',type=int)
    parser.add_argument('--V',type=float)
    parser.add_argument('--diameter',type=float,default=np.sqrt(24/np.pi),help='Default: 6 m2 circular canopy')
    parser.add_argument('--p-inf',type=float,default=0.,help='Far-field p/rho in m2/s2')
    parser.add_argument('--pressure-field',default='p')
    parser.add_argument('--neighbours',type=int,default=4)
    parser.add_argument('--sector',type=float,default=6.,help='Half width in degrees; never a complete ring')
    parser.add_argument('--normal-angle',type=float,default=60.,help='Maximum mismatch between CFD face and intended local surface normal')
    parser.add_argument('--azimuth',type=float,default=0.,help='Meridian around the canopy axis; 0 passes through line 0')
    parser.add_argument('--subdivide',type=int,default=12)
    parser.add_argument('--pole',choices=['closed','open'],default='closed')
    parser.add_argument('--vent-fraction',type=float)
    parser.add_argument('--shape',type=Path)
    parser.add_argument('--surface-dir',type=Path)
    parser.add_argument('--out-dir',type=Path)
    parser.add_argument('--fields-dir',type=Path)
    parser.add_argument('--pics',type=Path)
    parser.add_argument('--reference',type=Path,help='Optional preliminary digitization CSV, not raw experimental measurements')
    parser.add_argument('--diagnose-normals',action='store_true',help='Report radial normal signs and their location; do not sample pressure or infer physical sides')
    args = parser.parse_args(argv)
    if not np.isfinite(args.diameter) or args.diameter<=0 or args.neighbours<1 or not 0<args.sector<90 or args.subdivide<1 or not 0<args.normal_angle<90:
        parser.error('Invalid diameter or sampling parameters')
    if args.vent_fraction is not None and not 0<args.vent_fraction<1:
        parser.error('--vent-fraction must be between zero and one')
    if len(args.runs)>1 and any([args.shape,args.surface_dir,args.out_dir,args.fields_dir]):
        parser.error('Explicit shape/surface/out/fields paths require a single run')
    results, errors = [], []
    for run in args.runs:
        try:
            if args.diagnose_normals:
                normal_diagnostics(args,run)
                continue
            result = extract(args,run)
        except (OSError,ValueError,KeyError) as error:
            errors.append(f'{run}: {error}')
            print(errors[-1],file=sys.stderr)
            continue
        folder = args.pics or run/'pics'
        folder.mkdir(parents=True,exist_ok=True)
        suffix = re.sub(r'[^\w.-]+','_',run.parent.name+'_'+run.name) if len(args.runs)>1 else ''
        stem = f"{args.name}_a{result.metadata['angle_deg']:g}_pass{result.metadata['pass_number']:02d}"+(f'_{suffix}' if suffix else '')
        write_table(folder/(stem+'_full.csv'),result.table)
        write_table(folder/(stem+'_compact.csv'),result.compact)
        (folder/(stem+'_metadata.json')).write_text(json.dumps(result.metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f"\n{run}: pass {result.metadata['pass_number']}, α = {result.metadata['angle_deg']:g}°")
        print('R Cp_in Cp_out delta_Cp x_body/D0 h_body/D0')
        np.savetxt(sys.stdout,result.compact,fmt='%+.3f')
        removed = sum(d['excluded_opposite_normal_faces'] for d in result.metadata['diagnostics'].values())
        if removed:
            print(f'Диагностика: исключено из выборки {removed} граней с противоположной ориентацией; подробности в metadata.json')
        print(f'CSV и параметры: {folder}')
        results.append(result)
    if results:
        plot_results(results,args,args.pics or results[0].run/'pics')
    if errors:
        parser.exit(1,'Some cases could not be processed; valid-case figures were saved where available.\n')
    return results


if __name__=='__main__':
    main()
