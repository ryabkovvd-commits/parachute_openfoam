#!/bin/bash
#SBATCH -J fixednumerics
#SBATCH -p compute1
#SBATCH -N 1
#SBATCH -n 36
#SBATCH -t 1-00:00:00
#SBATCH --array=0-2
#SBATCH -o %x-%A_%a.out
# Submit from project root; this one file has no repository-script dependencies:
# sbatch cluster/fixed_numerics_control.slurm run/fixed_cfd_continue/9121
# Same frozen 49-degree form, volume mesh and iteration-2000 restart as reference 9121.
# Tasks: 0 relaxation 0.5; 1 relTol 0; 2 both. Reference 9121 is the fourth factorial cell.
set -euo pipefail
B=${SLURM_SUBMIT_DIR:?Submit from project root}
SOURCE=$(realpath "${1:?Supply the completed fixed-CFD reference directory}")
END=${2:-6000}
P=${NOS6_DIAG_PYTHON:-/home/016/apps/dashboard/anaconda3/bin/python}
test -x "$P"
CORES=${SLURM_NTASKS:?}
MODES=(relax05 tight relax05_tight)
TASK=${SLURM_ARRAY_TASK_ID:?Submit this file with sbatch as an array}
[[ "$TASK" =~ ^[0-2]$ ]] || { echo 'Expected array task 0, 1 or 2' >&2; exit 2; }
MODE=${MODES[$TASK]}
W="$B/run/fixed_numerics/${SLURM_ARRAY_JOB_ID:?}/$MODE"
module load "${NOS6_OPENFOAM_MODULE:-openfoam/2212}"
export OMP_NUM_THREADS=1
"$P" - "$SOURCE" "$W" "$END" "$CORES" "$MODE" <<'PY_PREP'
from pathlib import Path
import hashlib,json,math,re,shutil,sys
reference,work=map(lambda p:Path(p).resolve(),sys.argv[1:3])
end=float(sys.argv[3]);cores=int(sys.argv[4]);mode=sys.argv[5]
modes={'relax05':(0.5,False),'tight':(0.9,True),'relax05_tight':(0.5,True)}
if mode not in modes:raise ValueError('Unknown numerical control')
alpha,tight=modes[mode]
if not math.isfinite(end) or end%1000 or end>20000 or cores<1:
    raise ValueError('Expected a finite end iteration divisible by 1000, <=20000, and positive cores')
if work==reference or reference in work.parents or work in reference.parents:
    raise ValueError('Output must be separate from the reference case')
if work.exists():raise FileExistsError('Keep existing controls; submit with a new array job ID')
meta=json.loads((reference/'continuation_report.json').read_text(encoding='utf-8'))
if not (meta['solver_pipeline_exit_code']==0 and meta['complete_finite_log'] and
        meta['mesh_unchanged'] and meta['source_mesh_unchanged'] and meta['STL_unchanged']):
    raise ValueError('Reference must be an intact completed fixed-CFD continuation')
start=float(meta['start_iteration'])
if start!=2000 or float(meta['last_iteration'])!=6000 or end!=float(meta['end_iteration']) or cores!=meta['cores']:
    raise ValueError('Control must match the reference: common restart 2000, end 6000, same core count')
time_folder=reference/meta['source_time_folder']
if time_folder.name!='2000':raise ValueError('Expected the common original 2000 restart, not reference final fields')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def mesh_hashes(root):
    result={}
    for name in ['points','faces','owner','neighbour','boundary']:
        files=[root/(name+s) for s in ['', '.gz'] if (root/(name+s)).is_file()]
        if len(files)!=1:raise ValueError('Missing or ambiguous mesh '+name)
        result[files[0].name]=sha(files[0])
    return result
before=mesh_hashes(reference/'constant/polyMesh')
if before!=meta['source_mesh_sha256']:raise ValueError('Reference mesh differs from its recorded snapshot')
if sha(reference/'constant/triSurface/canopy.stl')!=meta['source_STL_sha256']:
    raise ValueError('Reference STL changed')
restart_hashes={p.name:sha(p) for p in sorted(time_folder.iterdir()) if p.is_file()}
if restart_hashes!=meta['source_restart_field_sha256']:
    raise ValueError('Common restart fields differ from the baseline record')
for name in ['U','p','phi','k','omega','nut']:
    if not any((time_folder/(name+s)).is_file() for s in ['', '.gz']):raise ValueError('Missing restart '+name)
solution=(reference/'system/fvSolution').read_text(encoding='utf-8')
body=re.sub(r'/\*.*?\*/|//[^\n]*','',solution,flags=re.S)
body=re.sub(r'FoamFile\s*\{[^{}]*\}','',body,count=1)
canonical=re.sub(r'\s+','',body)
expected='solvers\n{\n p { solver GAMG; tolerance 1e-6; relTol 0.01; smoother GaussSeidel; }\n "(U|k|omega)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; }\n}\nSIMPLE { nNonOrthogonalCorrectors 0; consistent yes; }\nrelaxationFactors { equations { ".*" 0.9; } }\n'
if canonical!=re.sub(r'\s+','',expected):
    raise ValueError('Unexpected reference fvSolution; review the actual settings before applying controls')
# Alter only the two controlled parameter families; preserve SIMPLEC, all schemes and physics.
new_solution=solution
if alpha!=0.9:
    new_solution,count=re.subn(r'("\.\*"\s+)0\.9\s*;',lambda m:m[1]+str(alpha)+';',new_solution)
    if count!=1:raise ValueError('Expected exactly one equation relaxation setting')
if tight:
    new_solution,count=re.subn(r'\brelTol\s+(?:0\.01|0\.1)\s*;','relTol 0;',new_solution)
    if count!=2:raise ValueError('Expected two relative linear tolerances')
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
    if (reference/name).is_dir():shutil.copytree(reference/name,work/name)
shutil.copytree(time_folder,work/time_folder.name)
control=work/'system/controlDict'
control.write_text(edit_root(control.read_text(encoding='utf-8'),dict(startFrom='latestTime',startTime=format(start,'g'),
    stopAt='endTime',endTime=format(end,'g'),writeControl='timeStep',writeInterval=1000,purgeWrite=0,runTimeModifiable='false')),encoding='utf-8')
decomp=work/'system/decomposeParDict'
decomp.write_text(edit_root(decomp.read_text(encoding='utf-8'),{'numberOfSubdomains':cores}),encoding='utf-8')
(work/'system/fvSolution').write_text(new_solution,encoding='utf-8')
baseline=work/'baseline';baseline.mkdir()
for name in ['force.dat','moment.dat']:
    src=reference/'baseline'/name
    for line in src.read_text(encoding='utf-8').splitlines():
        if line.strip() and not line.lstrip().startswith('#') and float(line.split()[0])>start:
            raise ValueError('Baseline history includes samples after the common restart')
    shutil.copy2(src,baseline/name)
if mesh_hashes(work/'constant/polyMesh')!=before:raise ValueError('Copied mesh mismatch')
settings={p.relative_to(reference).as_posix():sha(p) for name in ['constant','system']
    for p in (reference/name).rglob('*') if p.is_file()}
changed=['system/fvSolution','system/controlDict','system/decomposeParDict']
if any(sha(work/p)!=h for p,h in settings.items() if p not in changed):
    raise ValueError('Uncontrolled case setting changed during staging')
report=dict(source_case=str(reference),original_coupled_case=meta['source_case'],source_pass=meta['source_pass'],
    start_iteration=start,end_iteration=end,cores=cores,source_time_folder=time_folder.name,
    source_mesh_sha256=before,source_STL_sha256=meta['source_STL_sha256'],
    source_restart_field_sha256=restart_hashes,source_setting_sha256=settings,
    qS0=meta['qS0'],D0=meta['D0'],reference_case=str(reference),reference_end_iteration=meta['last_iteration'],
    numerical_control=mode,equation_relaxation=alpha,linear_relTol=(0 if tight else 'reference'),
    linear_absolute_tolerances='unchanged: p=1e-6; U,k,omega=1e-8',
    unchanged='SIMPLEC, nonorthogonal correctors, fvSchemes, mesh, STL, material/flow settings, common 2000 restart',
    scope='Frozen CFD case: numerical sensitivity only, no structural solve or physical-time analysis.')
(work/'continuation_input.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('CONTROL:',mode,'; common restart:',start,'->',end,'; output:',work,flush=True)

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
def sha256_file(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read_table(p):
    result=[]
    for line in p.read_text(encoding='utf-8',errors='replace').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):continue
        values=[float(x) for x in line.replace('(',' ').replace(')',' ').split()]
        if len(values)!=10 or not all(math.isfinite(v) for v in values):raise ValueError('Invalid force history '+str(p))
        result.append(values)
    return result
outputs={};windows={}
common_start=float(meta['start_iteration'])
for filename,scale in [('force.dat',meta['qS0']),('moment.dat',meta['qS0']*meta['D0'])]:
    records={row[0]:row for row in read_table(w/'baseline'/filename)}
    new_files=sorted((w/'postProcessing/forces').glob('*/'+filename))
    for f in new_files:
        for row in read_table(f):
            if row[0]>common_start:records[row[0]]=row
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
last_iteration=min((rows[-1][0] if rows else common_start) for rows in outputs.values())
log=(w/'log.simpleFoam').read_text(encoding='utf-8',errors='replace') if (w/'log.simpleFoam').is_file() else ''
complete=bool(re.search(r'^\s*End\s*$',log,re.M)) and 'FOAM FATAL' not in log and not re.search(r'\b(?:nan|infinity)\b',log,re.I)
solver_iterations=[float(t) for t in re.findall(r'^Time = ([\deE+.-]+)\s*$',log,re.M)]
expected_iterations=list(range(int(meta['start_iteration'])+1,int(meta['end_iteration'])+1))
complete=complete and solver_iterations==expected_iterations

source_settings_same=all(sha256_file(source/p)==h for p,h in meta['source_setting_sha256'].items())
copied_settings_same=all(sha256_file(w/p)==h for p,h in meta['source_setting_sha256'].items()
    if p not in ['system/fvSolution','system/controlDict','system/decomposeParDict'])
restart_same=all(sha256_file(w/meta['source_time_folder']/p)==h for p,h in meta['source_restart_field_sha256'].items())
source_restart_same=all(sha256_file(source/meta['source_time_folder']/p)==h for p,h in meta['source_restart_field_sha256'].items())
residuals={};bounding={}
for field,initial,final,count in re.findall(r'Solving for (\w+), Initial residual = ([\deE+.-]+), Final residual = ([\deE+.-]+), No Iterations (\d+)',log):
    residuals[field]=dict(initial=float(initial),final=float(final),linear_iterations=int(count))
for field,minimum in re.findall(r'^bounding (\w+), min:\s*([\deE+.-]+)',log,re.M):
    b=bounding.setdefault(field,dict(messages=0,lowest_prebounding_minimum=0.))
    b['messages']+=1;b['lowest_prebounding_minimum']=min(b['lowest_prebounding_minimum'],float(minimum))

report=dict(meta,source_settings_unchanged=source_settings_same,
    unchanged_settings_preserved=copied_settings_same,common_restart_unchanged=restart_same,source_restart_unchanged=source_restart_same,
    last_residuals=residuals,bounding=bounding,solver_pipeline_exit_code=status,last_iteration=last_iteration,
    mesh_unchanged=mesh_same,source_mesh_unchanged=source_mesh_same,STL_unchanged=stl_same,complete_finite_log=complete,
    windows=windows,convergence='not automatically certified; inspect iteration histories',scope='Forces/moments are CFD surface pressure plus viscous traction; direct cord forces are not added to these histories.')
(w/'continuation_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
lines=['NUMERICAL CONTROL: '+meta['numerical_control'],'REFERENCE SETTINGS / COPIED SETTINGS / COMMON RESTART / SOURCE RESTART: '+str([source_settings_same,copied_settings_same,restart_same,source_restart_same]),'SOURCE: '+meta['source_case'],'SOURCE PASS: '+str(meta['source_pass']),
       'ITERATIONS: '+str(meta['start_iteration'])+' -> '+str(last_iteration),
       'MESH / SOURCE MESH / STL UNCHANGED: '+str([mesh_same,source_mesh_same,stl_same]),
       'SOLVER PIPELINE / COMPLETE LOG: '+str([status,complete]),report['scope']]
for name,values in windows.items():
    lines.append(name+' NORMALIZED WINDOWS (last 10 samples when stride=25):')
    lines.extend(str(v) for v in values)
lines.append('LAST RESIDUALS: '+str(residuals))
lines.append('BOUNDING: '+str(bounding))
lines.append('No structural update; convergence requires review of histories, not just reaching endTime.')
(w/'summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines),flush=True)
if status:raise SystemExit(status)
if not(mesh_same and source_mesh_same and stl_same and source_settings_same and copied_settings_same and restart_same and source_restart_same and complete and last_iteration==meta['end_iteration']):raise SystemExit(3)
PY_POST
