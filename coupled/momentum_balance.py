"""Audit saved steady permeability probes; never run a CFD solver.

Build tools/momentumAudit with wmake, then:
python coupled/momentum_balance.py run/permeability_models/8996
"""
import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import permeability_models as permeability


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def vector_columns(rows, names):
    result=np.array([[float(row[k]) for k in names] for row in rows],dtype=float)
    if result.shape!=(len(rows),len(names)) or not np.isfinite(result).all():
        raise ValueError('Invalid numerical export')
    return result


def integrate_boundary(rows,rho,origin):
    """Outward normals are relative to the fluid; pressure/viscous terms act on bodies."""
    if not rows or not np.isfinite(rho) or rho<=0:
        raise ValueError('Empty boundary or invalid density')
    c=vector_columns(rows,['cx','cy','cz'])-np.asarray(origin)
    s=vector_columns(rows,['sx','sy','sz'])
    u=vector_columns(rows,['Ux','Uy','Uz'])
    phi=vector_columns(rows,['phi']).ravel()
    p=vector_columns(rows,['p_kin']).ravel()
    tau=vector_columns(rows,['tauSfx','tauSfy','tauSfz'])
    area=np.linalg.norm(s,axis=1)
    if np.any(area<=0): raise ValueError('Non-positive boundary face area')
    normals=s/area[:,None]
    normal_u=np.einsum('ij,ij->i',u,normals)[:,None]*normals
    terms=dict(pressure=rho*p[:,None]*s,viscous=-rho*tau,
               convection=rho*phi[:,None]*u,
               convection_normal=rho*phi[:,None]*normal_u,
               convection_tangential=rho*phi[:,None]*(u-normal_u))
    groups={}
    names=np.array([row['patch'] for row in rows])
    if not {'canopy','canopy_slave'}.issubset(set(names)):
        raise ValueError('Both canopy and canopy_slave patches are required')
    if not np.any(~np.isin(names,['canopy','canopy_slave'])):
        raise ValueError('No external boundary patches found')
    selections={name:names==name for name in sorted(set(names))}
    selections.update(interface=np.isin(names,['canopy','canopy_slave']),
                      outer=~np.isin(names,['canopy','canopy_slave']),all=np.ones(len(rows),dtype=bool))
    for name,mask in selections.items():
        item=dict(face_count=int(mask.sum()),area_m2=float(area[mask].sum()),
                  volume_flux_m3_s=float(phi[mask].sum()),abs_flux_m3_s=float(np.abs(phi[mask]).sum()),
                  phi_minus_U_dot_Sf_L1_m3_s=float(np.abs(phi[mask]-np.einsum('ij,ij->i',u[mask],s[mask])).sum()))
        for key,force in terms.items():
            item[key+'_force_N']=force[mask].sum(0).tolist()
            item[key+'_moment_Nm']=np.cross(c[mask],force[mask]).sum(0).tolist()
        groups[name]=item
    return groups


def cord_terms(mapping,archive,rho,origin):
    ids=np.array([int(r['sample']) for r in mapping])
    if not np.array_equal(np.sort(ids),np.arange(len(ids))):
        raise ValueError('Invalid cord mapping IDs')
    if archive.shape!=(len(ids),4) or not np.isfinite(archive).all():
        raise ValueError('Missing or invalid applied cord forces')
    order=np.argsort(archive[:,0])
    applied=archive[order]
    if not np.array_equal(applied[:,0],np.arange(len(ids))):
        raise ValueError('Missing/duplicate cord force IDs')
    applied=rho*applied[ids,1:4]
    cells=vector_columns(mapping,['cx','cy','cz'])-np.asarray(origin)
    points=vector_columns(mapping,['x','y','z'])-np.asarray(origin)
    recomputed=rho*vector_columns(mapping,['rx_kin','ry_kin','rz_kin'])
    return dict(sample_count=len(ids),force_N=applied.sum(0).tolist(),
                moment_at_cells_Nm=np.cross(cells,applied).sum(0).tolist(),
                moment_at_samples_Nm=np.cross(points,applied).sum(0).tolist(),
                force_recomputed_final_U_N=recomputed.sum(0).tolist(),
                moment_at_cells_recomputed_final_U_Nm=np.cross(cells,recomputed).sum(0).tolist(),
                force_recomputed_difference_relative=float(np.linalg.norm((recomputed-applied).sum(0))/max(np.linalg.norm(applied.sum(0)),1e-30)),
                per_sample_recomputed_difference_relative=float(np.linalg.norm(recomputed-applied)/max(np.linalg.norm(applied),1e-30)),
                cell_mapping_note='Sample cells are reconstructed serially. Ambiguous points on cell faces can differ from the original parallel ownership. Applied forces come from saved source logs; final-U recomputation is separate.')


def form_balance(groups,cords,q,s0,diameter):
    """Steady CV: outer inferred load = interface traction + interface convection + cord reaction."""
    result={}
    for quantity,scale in [('force_N',q*s0),('moment_Nm',q*s0*diameter)]:
        def term(group,name): return np.array(groups[group][name+'_'+quantity])
        pressure=term('interface','pressure')
        viscous=term('interface','viscous')
        convection=term('interface','convection')
        cord=np.array(cords['force_N' if quantity=='force_N' else 'moment_at_cells_Nm'])
        outer=-sum((term('outer',name) for name in ['pressure','viscous','convection']),np.zeros(3))
        inferred=pressure+viscous+convection+cord
        result[quantity]=dict(interface_pressure=pressure.tolist(),interface_viscous=viscous.tolist(),
                              interface_convection=convection.tolist(),cords_at_cells=cord.tolist(),
                              interface_balance_with_cords=inferred.tolist(),outer_inferred_load=outer.tolist(),
                              residual=(inferred-outer).tolist(),normalization=scale)
    return result


def native_force_record(path,time):
    if not path.is_file(): return None
    records=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.lstrip().startswith('#'): continue
        values=[float(x) for x in line.replace('(',' ').replace(')',' ').split()]
        if len(values)>=10 and np.isclose(values[0],time,rtol=0,atol=1e-8): records.append(values)
    if len(records)!=1: return None
    a=records[0]
    return dict(time=a[0],total=a[1:4],pressure=a[4:7],viscous=a[7:10],
                porous=a[10:13] if len(a)>=13 else [0.,0.,0.])


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def present(folder,name):
    for suffix in ['', '.gz']:
        path=folder/(name+suffix)
        if path.is_file() and path.stat().st_size: return path
    raise FileNotFoundError(f'Required saved field/file is missing: {folder/name}')


def snapshot_sources(work,row):
    if not row.get('fixed_shape') or int(row['pass'])!=1:
        raise ValueError('This version audits the single-pass fixed-shape probes only')
    case=work/'case'
    control=(case/'system/controlDict').read_text(encoding='utf-8')
    if not re.search(r'\bapplication\s+simpleFoam\s*;',control):
        raise ValueError('Only steady simpleFoam snapshots are supported; unsteady storage is not included')
    time=str(int(row['iters']))
    archive=work/'fields'/permeability.NAME/'pass01'
    options=[archive/time,case/time]
    complete=[]
    for folder in options:
        try: complete.append({name:present(folder,name) for name in ['U','p','phi','nut']})
        except FileNotFoundError: pass
    if not complete:
        raise FileNotFoundError(f'Need U, p, phi, nut at {time} in {options}. No reconstruction of phi from U is substituted.')
    selected=complete[0]
    # Detect conflicting archive/current fields, rather than silently mixing snapshots.
    if len(complete)>1 and any(sha(complete[0][n])!=sha(complete[1][n]) for n in selected):
        raise ValueError('Saved archive and current case fields differ; select/repair provenance before auditing')
    extras=dict(mesh=case/'constant/polyMesh',cords=present(case/'constant','cords'),
                schemes=case/'system/fvSchemes',solution=case/'system/fvSolution')
    permeability.fingerprint(extras['mesh'])
    return time,selected,extras,archive


def run_case(work,binary):
    config,manifest,row=permeability.verify(work)
    time,fields,extra,archive=snapshot_sources(work,row)
    turbulence=work/'case/constant/turbulenceProperties'
    if not re.search(r'\bRASModel\s+kOmegaSST\s*;',turbulence.read_text(encoding='utf-8')):
        raise ValueError('This stress reconstruction is restricted to the current kOmegaSST cases')
    rho,speed,diameter=float(row['rho']),float(row['V']),float(row['D0'])
    if not np.isclose(float(row['S0']),np.pi*diameter**2/4): raise ValueError('Unexpected reference area')
    options=(work/'case/system/fvOptions').read_text(encoding='utf-8')
    count=re.search(r'\bconst\s+label\s+n\s*=\s*(\d+)\s*;',options)
    if not count or options.count('type vectorCodedSource;')!=1 or 'name cordDrag;' not in options:
        raise ValueError('Expected the known cordDrag source only')
    known=dict(CORD_SAMPLES=int(count[1]),CORD_WIDTH=float(row['cord_width_m']),CORD_CD=float(row['cord_cd']))
    template=Path(__file__).with_name('cordsFv.txt').read_text(encoding='ascii')
    for key,value in known.items(): template=template.replace(key,str(value))
    # Ensure the force-law implementation matches the audited source, not just its name.
    if re.sub(r'\s+','',template) not in re.sub(r'\s+','',options):
        raise ValueError('fvOptions differs from the known cord source; its applied force must be audited explicitly')
    cord_dir=archive/'postProcessing/cords'/time
    if not cord_dir.is_dir(): cord_dir=work/'case/postProcessing/cords'/time
    files=sorted(cord_dir.glob('*.dat'))
    if not files: raise FileNotFoundError(f'No applied cord source records: {cord_dir}')
    applied=np.vstack([np.loadtxt(p,ndmin=2) for p in files if p.stat().st_size])
    watched=list(fields.values())+[extra['cords'],extra['schemes'],extra['solution'],turbulence,work/'case/system/fvOptions']+files
    before={str(p):sha(p) for p in watched}
    mesh_before=permeability.fingerprint(extra['mesh'])
    output=work/'out/momentum_balance'
    output.mkdir(parents=True,exist_ok=True)
    status=output/'report.json'
    if status.exists(): status.unlink() # Never leave a stale successful report after a failed rerun.
    with tempfile.TemporaryDirectory(prefix='nos6-momentum-') as temp:
        stage=Path(temp)
        (stage/'system').mkdir()
        (stage/'constant').mkdir()
        (stage/time).mkdir()
        shutil.copytree(extra['mesh'],stage/'constant/polyMesh')
        shutil.copyfile(extra['cords'],stage/'constant/cords')
        shutil.copyfile(extra['schemes'],stage/'system/fvSchemes')
        shutil.copyfile(extra['solution'],stage/'system/fvSolution')
        for source in fields.values(): shutil.copyfile(source,stage/time/source.name)
        (stage/'system/controlDict').write_text(
            'FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n'
            f'application nos6MomentumAudit; startFrom startTime; startTime {time}; '
            f'stopAt endTime; endTime {time}; deltaT 1; writeControl timeStep; writeInterval 1; '
            'libs (); functions {}\n',encoding='ascii')
        command=[str(binary),'-case',str(stage),'-audit-out',str(output),'-nu',str(row['nu']),
                 '-cord-width',str(row['cord_width_m']),'-cord-cd',str(row['cord_cd']),
                 '-cord-samples',str(known['CORD_SAMPLES'])]
        done=subprocess.run(command,cwd=stage,capture_output=True,text=True,encoding='utf-8',errors='replace')
        (output/'native.log').write_text(done.stdout+done.stderr,encoding='utf-8')
        if done.returncode: raise RuntimeError(f'Exporter failed ({done.returncode}): {output}/native.log')
    unchanged=before=={str(p):sha(p) for p in watched} and mesh_before==permeability.fingerprint(extra['mesh'])
    if not unchanged: raise ValueError('Source files changed during the audit')
    groups=integrate_boundary(read_csv(output/'boundary.csv'),rho,[0.,0.,0.])
    cords=cord_terms(read_csv(output/'cord_mapping.csv'),applied,rho,[0.,0.,0.])
    q=.5*rho*speed**2
    balance=form_balance(groups,cords,q,float(row['S0']),diameter)
    volume={r['quantity']:[float(r[k]) for k in ['x','y','z']] for r in read_csv(output/'volume.csv')}
    for key in ['bounded_force_kin','bounded_moment_kin']: volume[key]=[rho*v for v in volume[key]]
    force=native_force_record(archive/'postProcessing/forces/0/force.dat',float(time))
    moment=native_force_record(archive/'postProcessing/forces/0/moment.dat',float(time))
    if force is None: force=native_force_record(work/'case/postProcessing/forces/0/force.dat',float(time))
    if moment is None: moment=native_force_record(work/'case/postProcessing/forces/0/moment.dat',float(time))
    report=dict(model=config['name'],time=float(time),rho=rho,speed=speed,diameter=diameter,
                origin=[0.,0.,0.],groups=groups,cords=cords,balance=balance,volume=volume,
                native_forces_function=dict(force=force,moment=moment),
                reference_history=dict(mz=row['mz'],cx=row['cx'],cy=row['cy'],cord_mz=row['cord_mz']),
                input_sha256=before,mesh_sha256=mesh_before,inputs_unchanged=unchanged,
                stress_method='rho*(nu+nut)*dev(grad(U)+grad(U).T), frozen stored nut and velocity values, original fvSchemes',
                limitations=['Steady snapshot: time storage is excluded.',
                             'Pressure is the incompressible solver pressure; no extra isotropic k term is inserted.',
                             'Boundary gradient reconstruction and angular-momentum discretization can leave a residual.',
                             'Cell mapping of cord samples is reconstructed; applied/final-U differences are reported.',
                             'An interface convective term is a diagnostic of this boundary model, not automatically a validated correction to fabric force.'],
                CFD_convergence='unconfirmed')
    status.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return compact(report)


def compact(report):
    b=report['balance']['moment_Nm']
    scale=b['normalization']
    def coeff(key): return b[key][1]/scale
    g=report['groups']
    native=report['native_forces_function']['moment']
    return dict(model=report['model'],mz_history=report['reference_history']['mz'],
                mz_pressure=coeff('interface_pressure'),mz_viscous=coeff('interface_viscous'),
                mz_convection=coeff('interface_convection'),
                mz_convection_tangential=g['interface']['convection_tangential_moment_Nm'][1]/scale,
                mz_cords_at_cells=coeff('cords_at_cells'),
                mz_cords_at_samples=report['cords']['moment_at_samples_Nm'][1]/scale,
                mz_interface_balance_with_cords=coeff('interface_balance_with_cords'),
                mz_outer=coeff('outer_inferred_load'),mz_balance_residual=coeff('residual'),
                mz_pressure_minus_native=(coeff('interface_pressure')-native['pressure'][1]/scale) if native else None,
                mz_viscous_minus_native=(coeff('interface_viscous')-native['viscous'][1]/scale) if native else None,
                Q_canopy=g['canopy']['volume_flux_m3_s'],Q_slave=g['canopy_slave']['volume_flux_m3_s'],
                Q_outer=g['outer']['volume_flux_m3_s'],Q_all=g['all']['volume_flux_m3_s'],
                Q_mismatch_U_phi=g['all']['phi_minus_U_dot_Sf_L1_m3_s'],
                continuity_L1=report['volume']['continuity_l1'][0],
                cord_recomputed_force_difference=report['cords']['force_recomputed_difference_relative'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--binary',type=Path,default=shutil.which('nos6MomentumAudit'))
    parser.add_argument('--model',choices=['df','darcy','quadratic'],action='append')
    args=parser.parse_args(argv)
    if args.binary is None or not args.binary.is_file(): parser.error('Build nos6MomentumAudit with wmake first')
    root=args.root.resolve()
    summary=root/'momentum_balance.csv'
    if summary.exists(): summary.unlink()
    rows=[]
    failed=False
    for model in args.model or ['df','darcy','quadratic']:
        previous=root/model/'out/momentum_balance/report.json'
        if previous.is_file(): previous.unlink()
        try:
            row=run_case(root/model,args.binary.resolve())
            rows.append(row)
            print(f"{model}: mz_p={row['mz_pressure']:+.6f} mz_visc={row['mz_viscous']:+.6f} "
                  f"mz_conv={row['mz_convection']:+.6f} mz_outer={row['mz_outer']:+.6f} "
                  f"residual={row['mz_balance_residual']:+.3e}",flush=True)
            print(f"  Q canopy/slave/all: {row['Q_canopy']:+.6g} {row['Q_slave']:+.6g} {row['Q_all']:+.3e}",flush=True)
        except (OSError,ValueError,KeyError,RuntimeError) as error:
            failed=True
            print(f'{model}: ERROR: {error}',flush=True)
    if rows:
        path=summary
        with path.open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        print(path)
    return int(failed)


if __name__=='__main__':
    raise SystemExit(main())
