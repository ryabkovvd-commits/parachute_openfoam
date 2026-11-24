"""Unified post-processing of saved CFD cases; no solver or relaxation.
python coupled/nos6_results.py case RUN
python coupled/nos6_results.py batch run/nos6wall/JOB
python coupled/nos6_results.py polar pics/adh_compare run/nos6held run/nos6wall
"""
import argparse
import contextlib
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import aero_diagnostics as diagnostics
import pressure_table

NAME = 'nos6_held'


def read_records(work,name=NAME):
    rows = json.loads((work/'out'/f'{name}.json').read_text(encoding='utf-8'))
    if not rows:
        raise ValueError('No completed passes')
    if len({int(r['pass']) for r in rows})!=len(rows):
        raise ValueError('Duplicate pass numbers')
    return sorted(rows,key=lambda r:int(r['pass']))


def saved_status(work,step,name=NAME):
    try:
        failed=json.loads((work/'out'/f'{name}.failed').read_text(encoding='utf-8'))
        if int(failed['last_completed_pass'])==int(step):
            return 'not_converged'
    except (OSError,ValueError,KeyError,TypeError):
        pass
    matches=[]
    for suffix,status in [('НЕСОШЁЛСЯ','not_converged'),('done','converged')]:
        try:
            record = json.loads((work/'out'/f'{name}.{suffix}').read_text(encoding='utf-8'))
            if int(record.get('pass',-1))==int(step):
                matches.append(status)
        except (OSError,ValueError,TypeError):
            pass
    return matches[0] if len(matches)==1 else 'unconfirmed'


def clean_json(value):
    if isinstance(value,dict):
        return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value,(float,np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer):
        return int(value)
    return value


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(clean_json(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)
    return str(path)


def settings(work,row,args):
    journal = work/'out'/f'{args.name}.txt'
    text = journal.read_text(encoding='utf-8') if journal.is_file() else ''
    found = re.search(r'\bV\s*=\s*([0-9.eE+-]+)',text)
    speed = args.speed if args.speed is not None else row.get('V',float(found[1]) if found else None)
    diameter = args.diameter if args.diameter is not None else row.get('D0',np.sqrt(24/np.pi) if args.name==NAME else None)
    rho = args.density if args.density is not None else row.get('rho',1.225)
    if any(v is None or not np.isfinite(v) or v<=0 for v in [speed,diameter,rho]):
        raise ValueError('Provide actual positive --speed and --diameter when absent from the archive')
    return float(diameter),float(speed),float(rho)


def canopy_model(work,row,name=NAME):
    if row.get('canopy_model'):
        return row['canopy_model']
    journal = work/'out'/f'{name}.txt'
    text = journal.read_text(encoding='utf-8') if journal.is_file() else ''
    if 'Проницаемая ткань:' in text or 'СТЕНКА С ВДУВОМ-ОТСОСОМ' in text:
        return 'fabric'
    if 'nos6wall' in work.parts:
        return 'wall_inferred_from_directory'
    return 'unconfirmed'


def csv_records(path,rows,preferred):
    columns = list(preferred)+sorted({key for row in rows for key in row}-set(preferred))
    values = [[json.dumps(row.get(k),ensure_ascii=False) if isinstance(row.get(k),(list,dict)) else row.get(k)
               for k in columns] for row in rows]
    return diagnostics.save_csv(path,columns,values)


def history_outputs(work,records,name):
    rows = []
    for r in records:
        row = dict(r)
        denominator = r.get('drag',r['cx'])
        row['K'] = r.get('side',r['cy'])/denominator if denominator else None
        row['status'] = saved_status(work,r['pass'],name)
        rows.append(row)
    columns = ['pass','hold','cx','cy','mz','K','drag','side','moment','ratio_seen','ratio','gap','shift',
               'cx_change','dload','omega','relaxed','status','cord_cx','cord_cy','cord_mz']
    path = csv_records(work/'out/adh_history.csv',rows,columns)
    last = rows[-1]
    summary = dict(last,case=str(work),canopy_model=canopy_model(work,last,name))
    for key in ['cx','cy','mz','ratio']:
        finite = [float(r[key]) for r in rows[-3:] if r.get(key) is not None]
        if finite:
            summary[key+'_last3_range'] = max(finite)-min(finite)
        if len(rows)>1 and rows[-2].get(key) is not None:
            summary[key+'_last_change_abs'] = abs(float(last[key])-float(rows[-2][key]))
    summary['iteration_note'] = 'Quasi-static passes are not physical time. Last-three-pass ranges measure iteration drift, not pulsations.'
    summary['status_scope']='Outer coupling criteria only; CFD, moment and grid convergence are separate checks.'
    final = csv_records(work/'out/adh_summary.csv',[summary],columns)
    summary_json = write_json(work/'out/adh_summary.json',summary)
    n = [r['pass'] for r in rows]
    fig,axes = plt.subplots(2,3,figsize=(12,7),sharex=True)
    for ax,key,label in zip(axes.flat[:3],['cx','cy','mz'],['Cx','Cy','mz']):
        ax.plot(n,[r[key] for r in rows],'o-',ms=3)
        ax.set_ylabel(label)
    axes[1,0].plot(n,[r.get('gap',np.nan) for r in rows],'o-',ms=3,label='Невязка формы gap')
    axes[1,0].plot(n,[r.get('shift',np.nan) for r in rows],':',label='Принятый сдвиг shift')
    axes[1,0].axhline(last.get('shape_tolerance',.03),color='gray',ls='--',label='Допуск формы')
    axes[1,0].set_ylabel('Доля D₀')
    axes[1,0].legend(fontsize=7)
    if all(r.get('gap') is None for r in rows):
        axes[1,0].text(.03,.82,'gap отсутствует в архиве',transform=axes[1,0].transAxes,fontsize=8)
    axes[1,1].plot(n,np.r_[np.nan,np.abs(np.diff([r['cx'] for r in rows]))],'o-',ms=3)
    axes[1,1].axhline(last.get('cx_tol',.003),color='gray',ls='--')
    axes[1,1].set_ylabel('|ΔCx| между проходами')
    axes[1,2].plot(n,[100*r['dload'] if r.get('dload') is not None else np.nan for r in rows],'o-',ms=3)
    axes[1,2].set_ylabel('Невязка нагрузки, %')
    if all(r.get('dload') is None for r in rows):
        axes[1,2].text(.1,.5,'Нет данных dload',transform=axes[1,2].transAxes,fontsize=9)
    for ax in axes.flat:
        ax.grid(alpha=.23)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in axes[-1]:
        ax.set_xlabel('Номер внешнего прохода')
    status_label = {'converged':'внешние критерии выполнены','not_converged':'внешние критерии не выполнены','unconfirmed':'статус не подтверждён'}[last['status']]
    fig.suptitle(f"Сходимость параметров · α = {last.get('hold',0):g}° · {status_label}")
    fig.tight_layout()
    image = work/'pics/convergence_criteria.png'
    fig.savefig(image,dpi=150)
    plt.close(fig)
    return [path,final,summary_json,str(image)]


def geometry_outputs(work,name,row,diameter,pole,vent_fraction):
    from cords import append_cords_stl
    from nos6_geometry import export_closed_surface
    from parafoil.inflation.exchange import export_surface
    mesh = diagnostics.make_mesh(row,diameter,pole,vent_fraction)
    final = np.load(work/'out'/f'{name}_shape.npy',allow_pickle=False)
    if final.shape!=mesh.position.shape or not np.isfinite(final).all():
        raise ValueError('Invalid relaxed final shape')
    exporter = export_closed_surface if pole=='closed' else export_surface
    mesh.position = final
    path = exporter(mesh,work/'pics/parachute.stl',subdivide=4)
    if row.get('cord_width_m'):
        append_cords_stl(mesh,row['cord_width_m'],path)
    artifacts = [str(path)]
    try:
        source = diagnostics.input_shape(work,name,row)
    except FileNotFoundError as error:
        return dict(status='partial',artifacts=artifacts,reason=str(error))
    mesh.position = np.load(source,allow_pickle=False)
    cfd = exporter(mesh,work/'pics/parachute_last_cfd.stl',subdivide=12)
    if row.get('cord_width_m'):
        append_cords_stl(mesh,row['cord_width_m'],cfd)
    artifacts.append(str(cfd))
    metrics = []
    theta = np.radians(float(row.get('hold') or 0.))
    e1,axis = np.array([np.cos(theta),0.,-np.sin(theta)]),np.array([np.sin(theta),0.,np.cos(theta)])
    for state,position in [('last_cfd_input',mesh.position),('after_relaxation',final)]:
        skirt = position[mesh.skirt]
        centre = skirt.mean(0)
        canopy = position[:(mesh.rings+1)*mesh.gores]
        metrics.append([state,np.ptp((skirt-centre)@e1)/diameter,np.ptp(skirt[:,1])/diameter,
                        np.max((canopy-centre)@axis)/diameter,np.ptp((skirt-centre)@axis)/diameter])
    artifacts.append(diagnostics.save_csv(work/'out/geometry_metrics.csv',
                     ['state','rim_width_xbody_D0','rim_width_y_D0','height_above_mean_rim_D0','rim_warp_D0'],metrics))
    return artifacts


def discover(roots,name=NAME):
    found = set()
    # Do not traverse polyMesh, processor*, or the complete cluster filesystem.
    patterns = [f'out/{name}.json',f'a*/out/{name}.json',f'*/a*/out/{name}.json',f'*/*/a*/out/{name}.json']
    for root in roots:
        for pattern in patterns:
            found.update(p.parent.parent.resolve() for p in root.glob(pattern))
    return sorted(found)


@contextlib.contextmanager
def summary_lock(output):
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.postprocess.lock').open('a+b') as stream:
        try:
            import fcntl
        except ImportError:
            fcntl = None  # Local Windows checks are serial; Linux cluster uses flock.
        if fcntl:
            fcntl.flock(stream,fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl:
                fcntl.flock(stream,fcntl.LOCK_UN)


def polar(output,roots,name=NAME):
    with summary_lock(output):
        return _polar_locked(output,roots,name)


def _polar_locked(output,roots,name=NAME):
    records,errors = [],[]
    for work in discover(roots,name):
        try:
            row = dict(read_records(work,name)[-1])
            row['angle'] = float(row['hold'])
            denominator = row.get('drag',row['cx'])
            row['K'] = row.get('side',row['cy'])/denominator if denominator else None
            row['status'] = saved_status(work,row['pass'],name)
            row['canopy_model'] = canopy_model(work,row,name)
            row['case'],row['series'] = str(work),str(work.parent)
            records.append(row)
        except (OSError,ValueError,KeyError,TypeError) as error:
            errors.append(dict(case=str(work),error=str(error)))
    if not records:
        raise ValueError('No valid held-angle results found')
    records.sort(key=lambda r:(r['series'],r['angle'],r['case']))
    table = csv_records(output/'polar.csv',records,
                        ['angle','series','case','canopy_model','pass','status','cx','cy','mz','K','gap','cx_change'])
    groups = {}
    for row in records:
        groups.setdefault(row['series'],[]).append(row)
    fig,axes = plt.subplots(2,2,figsize=(11,8),sharex=True)
    for ax,key,label in zip(axes.flat,['cx','cy','mz','K'],['Cx','Cy','mz','K = Cy/Cx']):
        for series,rows in groups.items():
            angles,values = [r['angle'] for r in rows],[r.get(key,np.nan) for r in rows]
            line, = ax.plot(angles,values,'-' if len(set(angles))==len(angles) else 'None',
                            label=Path(series).parent.name+'/'+Path(series).name,alpha=.6)
            for r,value in zip(rows,values):
                style = {'converged':'o','not_converged':'x','unconfirmed':'^'}[r['status']]
                ax.plot(r['angle'],value,style,color=line.get_color())
        ax.axhline(0,color='.65',lw=.7)
        ax.grid(alpha=.23)
        ax.set_ylabel(label)
    axes[0,0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Угол атаки, град')
    fig.suptitle('АДХ · ○ внешние критерии выполнены · × не выполнены · △ статус неизвестен')
    fig.tight_layout()
    fig.savefig(output/'polar.png',dpi=150)
    plt.close(fig)
    status = write_json(output/'polar_status.json',dict(cases=len(records),unreadable=errors,
                        note='Array jobs remain separate series. Only solver status markers confirm convergence.'))
    return [table,str(output/'polar.png'),status]


class CaseReport:
    def __init__(self,work,row,name):
        self.work = work
        self.data = dict(case=str(work),name=name,last_pass=int(row['pass']),
                         solver_status=saved_status(work,row['pass'],name),
                         started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),stages=[])

    def stage(self,name,action,optional=True):
        record = dict(name=name,status='running',artifacts=[])
        self.data['stages'].append(record)
        try:
            result = action()
            if isinstance(result,dict):
                record.update(result)
            else:
                record.update(status='ok',artifacts=result or [])
        except FileNotFoundError as error:
            record.update(status='skipped' if optional else 'failed',reason=str(error))
        except Exception as error:
            record.update(status='failed',reason=f'{type(error).__name__}: {error}')
        print(f"[{record['status']}] {name}: {record.get('reason','')}",flush=True)
        write_json(self.work/'out/postprocess_report.json',self.data)

    def finish(self):
        statuses = [s['status'] for s in self.data['stages']]
        self.data['status'] = 'failed' if 'failed' in statuses else 'partial' if any(s in statuses for s in ['skipped','partial']) else 'ok'
        self.data['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(self.work/'out/postprocess_report.json',self.data)
        return self.data


def case(work,args):
    records = read_records(work,args.name)
    row = records[-1]
    diameter,speed,rho = settings(work,row,args)
    (work/'pics').mkdir(exist_ok=True)
    report = CaseReport(work,row,args.name)
    report.data['configuration'] = dict(diameter_m=diameter,speed_m_s=speed,density_kg_m3=rho,
                                         canopy_model=canopy_model(work,row,args.name),
                                         standard_figures_requested=not args.skip_plots,gif_requested=args.gif)
    report.stage('adh_tables_and_convergence',lambda:history_outputs(work,records,args.name),optional=False)
    report.stage('final_and_cfd_input_stl',lambda:geometry_outputs(work,args.name,row,diameter,args.pole,args.vent_fraction))

    def run_plots(gif=False):
        script = Path(__file__).with_name('plots.py')
        if not script.is_file():
            raise FileNotFoundError(script)
        command = [sys.executable,str(script),args.name,f'--run={work}',f'--V={speed:g}',f'--wake={args.wake}']
        command += [f'--gif={args.gif}',f'--jobs={args.jobs}'] if gif else [f"--pass={int(row['pass'])}"]
        before = {p:p.stat().st_mtime_ns for p in (work/'pics').glob('*') if p.is_file()}
        done = subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace')
        log = work/'out'/('gif_postprocess.log' if gif else 'plots_postprocess.log')
        log.write_text(done.stdout+'\n'+done.stderr,encoding='utf-8')
        if done.returncode:
            raise ValueError(f'plots.py exit {done.returncode}; see {log}')
        changed = [str(p) for p in (work/'pics').glob('*') if p.is_file() and p.stat().st_mtime_ns!=before.get(p)]
        if not changed:
            raise ValueError(f'No new figures; see {log}')
        expected = [f'{args.name}_история.png',f'{args.name}_форма.png']
        if not gif:
            expected += [f'{args.name}_{label}.png' for label in ['купол','давление','скорость','линии_тока']]
        missing = [p for p in expected if str(work/'pics'/p) not in changed] if not gif else []
        warnings = [line for line in done.stdout.splitlines()
                    if ('след:' in line and ('нужны' in line or '0 траекторий' in line))
                    or (gif and args.gif=='all' and 'пропущен' in line)]
        reasons = (['Missing regenerated figures: '+', '.join(missing)] if missing else [])+warnings
        return dict(status='partial' if reasons else 'ok',artifacts=[str(log)]+changed,reason='; '.join(reasons))

    if not args.skip_plots:
        report.stage('standard_figures',run_plots)

    def meridians():
        options = SimpleNamespace(name=args.name,out_dir=None,fields_dir=None,surface_dir=None,
                                  pass_number=int(row['pass']),shape=None,V=speed,diameter=diameter,
                                  pole=args.pole,vent_fraction=args.vent_fraction,azimuth=0.,subdivide=12,
                                  sector=6.,normal_angle=60.,neighbours=4,p_inf=0.,pressure_field='p')
        folder = diagnostics.pressure_folder(work,args.name,row)
        if 'pMean' in diagnostics.vtp.arrays(folder/'master.vtp'):
            options.pressure_field = 'pMean'
        result = pressure_table.extract(options,work)
        prefix = work/'pics'/f'{args.name}_meridian_pass{int(row["pass"]):02d}'
        pressure_table.write_table(Path(str(prefix)+'_full.csv'),result.table)
        pressure_table.write_table(Path(str(prefix)+'_compact.csv'),result.compact)
        sampling = pressure_table.write_sampling(prefix,result)
        write_json(Path(str(prefix)+'_metadata.json'),result.metadata)
        pressure_table.plot_results([result],SimpleNamespace(reference=args.reference),work/'pics')
        return [str(Path(str(prefix)+s)) for s in ['_full.csv','_compact.csv','_metadata.json']]+[
                str(work/'pics/pressure_meridians.png'),str(work/'pics/canopy_profiles.png')]+sampling
    report.stage('meridian_pressure_and_profile',meridians)

    def surface_stage():
        artifacts,meta = diagnostics.surface(work,args.name,row,diameter,speed,rho,args.pole,args.vent_fraction)
        return dict(status='partial' if meta['paired_area_fraction']<.99 else 'ok',artifacts=artifacts,
                    reason=f"Paired area: {100*meta['paired_area_fraction']:.4f}%")
    report.stage('pressure_surface_moment_and_transfer',surface_stage)
    report.stage('wake_vorticity_and_reverse_flow',lambda:diagnostics.wake_cut(work,args.name,row,diameter,speed))

    def cord_loads():
        from cord_study import confluence_loads
        if row.get('fixed_shape'):
            raise FileNotFoundError('Fixed-shape probe: inherited structural tensions are not a response to the new flow')
        if args.pole!='closed' or not np.isclose(diameter,np.sqrt(24/np.pi)):
            raise FileNotFoundError('Confluence helper requires the closed 6 m2 model')
        load_info=confluence_loads(work,args.name,records,work/'out')
        artifacts = [str(work/'out/confluence_loads.csv'),str(work/'out/confluence_loads.png')]
        missing = [int(r['pass']) for r in records
                   if not (work/'out/кадры'/f"{args.name}_проход{int(r['pass']):02d}.npy").is_file()]
        missing_direct=load_info.get('missing_direct_passes',[])
        return dict(status='partial' if missing or missing_direct else 'ok',artifacts=artifacts,
                    reason=f'Missing frames: {missing}; direct confluence forces unavailable: {missing_direct}' if missing or missing_direct else '')
    report.stage('confluence_loads',cord_loads)
    if args.gif!='off':
        report.stage('flow_gif',lambda:run_plots(True))
    if not args.no_summary:
        report.stage('array_summary',lambda:polar(work.parent/'summary',[work.parent],args.name))
    return report.finish()


def add_case_options(parser):
    parser.add_argument('--name',default=NAME)
    parser.add_argument('--speed',type=float)
    parser.add_argument('--diameter',type=float)
    parser.add_argument('--density',type=float)
    parser.add_argument('--pole',choices=['closed','open'],default='closed')
    parser.add_argument('--vent-fraction',type=float)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--wake',type=int,choices=[0,1],default=1)
    parser.add_argument('--gif',choices=['off','1','all'],default='off')
    parser.add_argument('--jobs',type=int,default=1)
    parser.add_argument('--skip-plots',action='store_true',help='Skip only legacy plots.py')
    parser.add_argument('--no-summary',action='store_true')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode',required=True)
    single = modes.add_parser('case')
    single.add_argument('work',type=Path)
    add_case_options(single)
    batch = modes.add_parser('batch')
    batch.add_argument('roots',nargs='+',type=Path)
    add_case_options(batch)
    batch.add_argument('--summary',type=Path,help='Combined summary output for the selected roots')
    combined = modes.add_parser('polar')
    combined.add_argument('output',type=Path)
    combined.add_argument('roots',nargs='+',type=Path)
    combined.add_argument('--name',default=NAME)
    surface = modes.add_parser('pressure')
    surface.add_argument('work',type=Path)
    add_case_options(surface)
    args = parser.parse_args(argv)
    if args.mode=='polar':
        for output in polar(args.output.resolve(),[r.resolve() for r in args.roots],args.name):
            print(output)
        return 0
    if args.jobs<1:
        parser.error('--jobs must be positive')
    if args.mode=='pressure':
        work = args.work.resolve()
        row = read_records(work,args.name)[-1]
        diameter,speed,rho = settings(work,row,args)
        print(diagnostics.surface(work,args.name,row,diameter,speed,rho,args.pole,args.vent_fraction)[1])
        return 0
    cases = [args.work.resolve()] if args.mode=='case' else discover([r.resolve() for r in args.roots],args.name)
    if args.mode=='batch':
        potential = set(cases)
        for root in args.roots:
            for pattern in ['a[0-9]*','*/a[0-9]*']:
                potential.update(p.resolve() for p in root.glob(pattern) if p.is_dir())
            if re.fullmatch(r'a\d+',root.name) and root.is_dir():
                potential.add(root.resolve())
        cases = sorted(potential)
    if not cases:
        parser.error('No saved cases found')
    reports = []
    for work in cases:
        try:
            reports.append(case(work,args))
        except (OSError,ValueError,KeyError) as error:
            reports.append(dict(case=str(work),status='failed',reason=str(error)))
            if work.is_dir():
                write_json(work/'out/postprocess_report.json',reports[-1])
            print(f'{work}: {error}',file=sys.stderr)
    if args.mode=='batch':
        try:
            summary_output = args.summary or args.roots[0]/'summary'
            polar(summary_output,[r.resolve() for r in args.roots],args.name)
        except (OSError,ValueError,KeyError) as error:
            reports.append(dict(case='combined_summary',status='failed',reason=str(error)))
        for root in args.roots:
            write_json(root/'postprocess_batch_report.json',dict(cases=reports))
    return int(any(r['status']=='failed' for r in reports))


if __name__=='__main__':
    raise SystemExit(main())
