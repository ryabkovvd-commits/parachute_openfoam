#!/bin/bash
#SBATCH -J fixedcontinue
#SBATCH -p compute1
#SBATCH -N 1
#SBATCH -n 36
#SBATCH -t 1-00:00:00
#SBATCH -o %x-%j.out
# Submit from project root:
# sbatch cluster/fixed_cfd_continue.slurm run/nos6wall_refine/9105/a49 6000
# Isolated restart of an existing valid CFD snapshot. No meshing or structure solve.
set -euo pipefail
B=${SLURM_SUBMIT_DIR:?Submit from project root}
SOURCE=$(realpath "${1:?Supply a completed CFD case directory with out/ and case/}")
END=${2:-6000}
P=${NOS6_DIAG_PYTHON:-/home/016/apps/dashboard/anaconda3/bin/python}
test -x "$P"
CORES=${SLURM_NTASKS:?}
W="$B/run/fixed_cfd_continue/${SLURM_JOB_ID:?}"
module load "${NOS6_OPENFOAM_MODULE:-openfoam/2212}"
export OMP_NUM_THREADS=1
"$P" - "$SOURCE" "$W" "$END" "$CORES" <<'PY_PREP'
from pathlib import Path
import hashlib,json,math,re,shutil,sys
source,work=map(lambda p:Path(p).resolve(),sys.argv[1:3])
end=float(sys.argv[3]);cores=int(sys.argv[4])
if not math.isfinite(end) or end%1000 or end>20000 or cores<1:
    raise ValueError('Expected end iteration divisible by 1000, <=20000, and a positive core count')
if work==source or source in work.parents or work in source.parents:
    raise ValueError('Restart output must be separate from the source')
if work.exists():raise FileExistsError('Keep existing controls; submit with a new job ID')
case=source/'case'
rows=json.loads((source/'out/nos6_held.json').read_text(encoding='utf-8'))
last=rows[-1]
diag=json.loads((source/'out'/('surface_diagnostics_pass%02d.json'%last['pass'])).read_text(encoding='utf-8'))
if diag['paired_area_fraction']<1-1e-12 or diag['unpaired_master_faces'] or diag['unpaired_slave_faces']:
    raise ValueError('Source pass does not have a validated complete pressure pair snapshot')
if diag.get('axes',{}).get('pitch')!='+Y' or diag['axes'].get('drag')!='+Z':
    raise ValueError('This diagnostic expects the existing native +Z-flow, +Y-pitch cases')
times=[]
for p in case.iterdir():
    if p.is_dir():
        try:t=float(p.name)
        except ValueError:continue
        if math.isfinite(t) and t>0:times.append((t,p))
if not times:raise ValueError('No saved serial solution in source case')
start,time_folder=max(times,key=lambda item:item[0])
if start!=float(diag['time']) or end<=start:
    raise ValueError('Latest saved solution must match pressure snapshot and precede end iteration')
for name in ['U','p','phi','k','omega','nut']:
    if not any((time_folder/(name+s)).is_file() for s in ['', '.gz']):
        raise ValueError('Missing restart field: '+name)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def mesh_hashes(root):
    result={}
    for name in ['points','faces','owner','neighbour','boundary']:
        files=[root/(name+s) for s in ['', '.gz'] if (root/(name+s)).is_file()]
        if len(files)!=1:raise ValueError('Missing or ambiguous mesh file '+name)
        result[files[0].name]=sha(files[0])
    return result
before=mesh_hashes(case/'constant/polyMesh')
# Preserve exact source bytes except for top-level run controls and processor count.
# Nested force/sampling writeInterval entries must remain unchanged.
def edit_root(text,values):
    masked=re.sub(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"',lambda m:' '*len(m[0]),text,flags=re.S)
    depth=0;spans={};skip_until=0
    for m in re.finditer(r'[{}]|[A-Za-z_]\w*',masked):
        if m.start()<skip_until:continue
        token=m[0]
        if token=='{':depth+=1
        elif token=='}':depth-=1
        elif depth==0 and token in values:
            if token in spans:raise ValueError('Duplicate top-level control '+token)
            stop=masked.find(';',m.end())
            if stop<0 or re.search(r'[{}]',masked[m.end():stop]):raise ValueError('Invalid control '+token)
            spans[token]=(m.start(),stop+1)
            skip_until=stop+1
    if depth!=0:raise ValueError('Unbalanced dictionary braces')
    for token,(a,b) in sorted(spans.items(),key=lambda item:item[1][0],reverse=True):
        text=text[:a]+token+' '+str(values[token])+';'+text[b:]
    text+='\n'+''.join(k+' '+str(v)+';\n' for k,v in values.items() if k not in spans)
    return text
work.mkdir(parents=True)
for name in ['system','constant','0']:
    if (case/name).is_dir():shutil.copytree(case/name,work/name)
shutil.copytree(time_folder,work/time_folder.name)
control=work/'system/controlDict'
control.write_text(edit_root(control.read_text(encoding='utf-8'),dict(startFrom='latestTime',startTime=format(start,'g'),stopAt='endTime',endTime=format(end,'g'),writeControl='timeStep',writeInterval=1000,purgeWrite=0,runTimeModifiable='false')),encoding='utf-8')
decomp=work/'system/decomposeParDict'
decomp.write_text(edit_root(decomp.read_text(encoding='utf-8'),{'numberOfSubdomains':cores}),encoding='utf-8')
baseline=work/'baseline';baseline.mkdir()
for name in ['force.dat','moment.dat']:
    shutil.copy2(case/'postProcessing/forces/0'/name,baseline/name)
if mesh_hashes(work/'constant/polyMesh')!=before:raise ValueError('Copied mesh does not match source')
report=dict(source_run=str(source),source_case=str(case),source_pass=last['pass'],start_iteration=start,end_iteration=end,cores=cores,
    source_time_folder=time_folder.name,source_mesh_sha256=before,source_control_sha256=sha(case/'system/controlDict'),
    source_restart_field_sha256={p.name:sha(p) for p in sorted(time_folder.iterdir()) if p.is_file()},
    source_STL_sha256=sha(case/'constant/triSurface/canopy.stl'),qS0=.5*last['rho']*last['V']**2*last['S0'],D0=last['D0'],
    scope='Fixed source CFD geometry and mesh, restart from saved fields. No structural update, no claim of convergence.')
(work/'continuation_input.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('Fixed CFD continuation:',work,flush=True)
print('Iterations:',start,'->',end,'; no mesh generation',flush=True)
PY_PREP
cd "$W"
ret=0
decomposePar -force -latestTime > log.decomposePar 2>&1 || ret=$?
if (( ret == 0 )); then mpirun -np "$CORES" simpleFoam -parallel > log.simpleFoam 2>&1 || ret=$?; fi
if (( ret == 0 )); then reconstructPar -latestTime > log.reconstruct 2>&1 || ret=$?; fi
"$P" - "$W" "$ret" <<'PY_POST'
from pathlib import Path
import csv,hashlib,json,math,re,sys
w=Path(sys.argv[1]);status=int(sys.argv[2]);meta=json.loads((w/'continuation_input.json').read_text())
mesh=w/'constant/polyMesh'
mesh_same=all(hashlib.sha256((mesh/n).read_bytes()).hexdigest()==s for n,s in meta['source_mesh_sha256'].items())
source=Path(meta['source_case'])
source_mesh_same=all(hashlib.sha256((source/'constant/polyMesh'/n).read_bytes()).hexdigest()==s for n,s in meta['source_mesh_sha256'].items())
stl_same=hashlib.sha256((w/'constant/triSurface/canopy.stl').read_bytes()).hexdigest()==meta['source_STL_sha256']
def read_table(p):
    result=[]
    for line in p.read_text(encoding='utf-8',errors='replace').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):continue
        values=[float(x) for x in line.replace('(',' ').replace(')',' ').split()]
        if len(values)!=10 or not all(math.isfinite(v) for v in values):raise ValueError('Invalid force history '+str(p))
        result.append(values)
    return result
outputs={};windows={}
for filename,scale in [('force.dat',meta['qS0']),('moment.dat',meta['qS0']*meta['D0'])]:
    records={row[0]:row for row in read_table(w/'baseline'/filename)}
    new_files=sorted((w/'postProcessing/forces').glob('*/'+filename))
    for f in new_files:
        for row in read_table(f):records[row[0]]=row
    rows=[records[t] for t in sorted(records)];outputs[filename]=rows
    axis=3 if filename=='force.dat' else 2
    windows[filename]=[]
    for boundary in [meta['start_iteration']]+list(range(int(meta['start_iteration'])+1000,int(meta['end_iteration'])+1,1000)):
        recent=[r for r in rows if boundary-225<=r[0]<=boundary]
        if recent:
            vals=[r[axis]/scale for r in recent]
            windows[filename].append(dict(iteration=boundary,samples=len(vals),min=min(vals),max=max(vals),range=max(vals)-min(vals)))
    with (w/('combined_'+filename)).open('w',encoding='utf-8') as f:
        for row in rows:f.write(' '.join(format(v,'.12g') for v in row)+'\n')
last_iteration=min(outputs['force.dat'][-1][0],outputs['moment.dat'][-1][0])
log=(w/'log.simpleFoam').read_text(encoding='utf-8',errors='replace') if (w/'log.simpleFoam').is_file() else ''
complete=bool(re.search(r'^\s*End\s*$',log,re.M)) and 'FOAM FATAL' not in log and not re.search(r'\b(?:nan|infinity)\b',log,re.I)
report=dict(meta,solver_pipeline_exit_code=status,last_iteration=last_iteration,
    mesh_unchanged=mesh_same,source_mesh_unchanged=source_mesh_same,STL_unchanged=stl_same,complete_finite_log=complete,
    windows=windows,convergence='not automatically certified; inspect iteration histories',scope='Forces/moments are CFD surface pressure plus viscous traction; direct cord forces are not added to these histories.')
(w/'continuation_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
lines=['SOURCE: '+meta['source_case'],'SOURCE PASS: '+str(meta['source_pass']),
       'ITERATIONS: '+str(meta['start_iteration'])+' -> '+str(last_iteration),
       'MESH / SOURCE MESH / STL UNCHANGED: '+str([mesh_same,source_mesh_same,stl_same]),
       'SOLVER PIPELINE / COMPLETE LOG: '+str([status,complete]),report['scope']]
for name,values in windows.items():
    lines.append(name+' NORMALIZED WINDOWS (last 10 samples when stride=25):')
    lines.extend(str(v) for v in values)
lines.append('No structural update; convergence requires review of histories, not just reaching endTime.')
(w/'summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines),flush=True)
if status:raise SystemExit(status)
if not(mesh_same and source_mesh_same and stl_same and complete and last_iteration==meta['end_iteration']):raise SystemExit(3)
PY_POST
