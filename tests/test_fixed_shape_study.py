"""Check geometry provenance and isolation; synthetic fixtures are not CFD results."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess
import os
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'coupled'))
import fixed_shape_study as fs
from nos6_geometry import closed_circular
from nosarev_frame import to_speed


def fixture(folder, angle=0, mode='legacy', frame='native'):
    folder=Path(folder)
    folder.joinpath('out').mkdir(parents=True)
    archive=folder/'fields/nos6_held/pass06'
    archive.mkdir(parents=True)
    diameter=np.sqrt(24/np.pi)
    row=dict(pass_number=6,hold=angle,D0=diameter,lines=diameter,gores=96,rings=12,
             stride=4,shear=.02,V=30.,nu=1.46e-5,rho=1.225,canopy_model='wall',
             fabric_coefficients=None,attitude_mode=mode,cx=.6,cy=.2,mz=.03,
             cord_width_m=.0032,cord_cd=1.2,pressure_gradient='gauss',
             turbulence_length_m=1.24,turbulence_intensity=.01)
    row['pass']=row.pop('pass_number')
    mesh=closed_circular(diameter,diameter,gores=96,rings=12,line_stride=4,shear_ratio=.02)
    mesh.position-=mesh.position[mesh.confluence[0]].copy()
    positions=fs.MeanAxisConstraint(mesh,angle).align_initial(mesh.position)
    if frame=='nosarev_speed':
        row['coordinate_frame']=frame
        row['cy']=-row['cy']
        positions=to_speed(positions)
    np.save(archive/'input_shape.npy',positions)
    # Deliberately different output: preparation must use the actual CFD input.
    np.save(folder/'out/nos6_held_shape.npy',positions+42)
    fs.save(folder/'out/nos6_held.json',[row])
    fs.save(folder/'out/nos6_held_config.json',dict(source_sha256={'core':'test'},
        physics={'flags':{'--attitude':mode,'--pressure-gradient':'gauss'}}))
    return row,positions


class FixedShapeTests(unittest.TestCase):
    def test_matrix_deduplicates_zero_and_includes_fifty(self):
        self.assertEqual(len(fs.plan(fs.PILOT)),7)
        self.assertEqual(len(fs.plan(fs.FULL)),21)
        self.assertEqual(sum(t['angle_deg']==0 for t in fs.plan(fs.FULL)),1)
        self.assertEqual(fs.plan(fs.FULL)[-1]['angle_deg'],50)

    def test_cli_defaults_to_all_twenty_one_states(self):
        r=subprocess.run([sys.executable,str(ROOT/'coupled/fixed_shape_study.py'),'plan'],
                         capture_output=True,text=True,encoding='utf-8',check=True)
        self.assertEqual(len(json.loads(r.stdout)),21)

    def test_full_series_collects_twenty_one_cases_and_eleven_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'8928'
            source_rows={}
            for angle in fs.FULL:
                source_rows[angle]=fixture(root/f'a{angle:02d}',angle)[0]
            setup=Path(tmp)/'setup';data=fs.prepare(root,setup,fs.FULL)
            run=Path(tmp)/'run'
            for index,task in enumerate(data['tasks']):
                work=run/task['id'];(work/'out').mkdir(parents=True)
                (work/'fields/nos6_held/pass01').mkdir(parents=True)
                for name in fs.NUMERICS:
                    p=work/'case'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('same setting')
                seed=np.load(setup/task['seed_file'])
                np.save(work/'fields/nos6_held/pass01/input_shape.npy',seed)
                np.save(work/'out/nos6_held_shape.npy',seed)
                fs.save(work/'study_case.json',dict(manifest=str(setup/'manifest.json'),task_index=index))
                row=dict(source_rows[task['angle_deg']]);row.update(fixed_shape=True,iters=2000,S0=6.,cells=1)
                row['pass']=1
                if task['family']=='angle_shape':row['cx']+=.01
                fs.save(work/'out/nos6_held.json',[row])
            report=fs.collect(run,setup/'manifest.json')
            self.assertTrue(report['complete'])
            self.assertEqual(report['completed_count'],21)
            self.assertEqual(len(report['delta_angle_shape_minus_reference']),11)
            self.assertAlmostEqual(report['delta_angle_shape_minus_reference'][1]['cx'],.01)
            self.assertTrue((run/'fixed_shape_polar.png').exists())
            # A corrupted case is recorded as missing, never silently counted as success.
            np.save(run/'angle_shape_a50/out/nos6_held_shape.npy',seed+1)
            report=fs.collect(run,setup/'manifest.json')
            self.assertFalse(report['complete'])
            self.assertIn('angle_shape_a50',report['missing_cases'])
            self.assertEqual(report['completed_count'],20)

    def test_completely_failed_series_still_produces_a_failure_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'8928';fixture(root/'a00')
            setup=Path(tmp)/'setup';fs.prepare(root,setup,[0])
            report=fs.collect(Path(tmp)/'empty',setup/'manifest.json')
            self.assertFalse(report['complete'])
            self.assertEqual(report['completed_count'],0)
            self.assertEqual(report['missing_cases'],['reference_a00'])

    @unittest.skipUnless(os.name=='posix' and shutil.which('bash'),'Bash submission integration runs in Linux CI')
    def test_submission_prepares_full_series_and_schedules_afterany_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp=Path(tmp);source=temp/'8928'
            for angle in fs.FULL:fixture(source/f'a{angle:02d}',angle)
            project=temp/'project with spaces';project.mkdir()
            (project/'coupled').symlink_to(ROOT/'coupled',target_is_directory=True)
            (project/'cluster').mkdir()
            for name in ['submit_fixed_shape_study.sh','fixed_shape_study.slurm','collect_fixed_shape_study.slurm']:
                shutil.copyfile(ROOT/'cluster'/name,project/'cluster'/name)
            binpath=temp/'bin';binpath.mkdir();calls=temp/'calls.json'
            fake=binpath/'sbatch'
            fake.write_text('#!'+sys.executable+'\nimport json,os,sys\nfrom pathlib import Path\n'
                'p=Path(os.environ["SBATCH_CALLS"]);rows=json.loads(p.read_text()) if p.exists() else []\n'
                'rows.append(sys.argv[1:]);p.write_text(json.dumps(rows));print(70000+len(rows))\n')
            fake.chmod(0o755)
            env=dict(os.environ,PATH=str(binpath)+os.pathsep+os.environ.get('PATH',''),
                     NOS6_PYTHON=sys.executable,SBATCH_CALLS=str(calls))
            result=subprocess.run(['bash','cluster/submit_fixed_shape_study.sh',str(source),'full21'],
                                  cwd=project,env=env,capture_output=True,text=True,timeout=45)
            self.assertEqual(result.returncode,0,result.stderr)
            invocations=json.loads(calls.read_text())
            self.assertIn('--array=0-20%2',invocations[0])
            self.assertIn('--dependency=afterany:70001',invocations[1])
            manifest=json.loads((project/'run/fixed_shape_study/setups/full21/manifest.json').read_text())
            self.assertEqual(len(manifest['tasks']),21)
            submission=json.loads((project/'run/fixed_shape_study/setups/full21/submission.json').read_text())
            self.assertEqual(submission['collection_job'],'70002')

    def test_preparation_preserves_geometry_and_exact_input_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'8928'
            row,original=fixture(root/'a00')
            fixture(root/'a30',30)
            setup=Path(tmp)/'setup'
            data=fs.prepare(root,setup,[0,30])
            task=next(t for t in data['tasks'] if t['id']=='reference_a30')
            transformed=np.load(setup/task['seed_file'])
            edges=np.array([[0,1],[0,96],[3,455],[1248,15]])
            def lengths(x): return np.linalg.norm(x[edges[:,0]]-x[edges[:,1]],axis=1)
            np.testing.assert_allclose(lengths(transformed),lengths(original),rtol=1e-12,atol=1e-12)
            axis=transformed[:1248].mean(0);axis/=np.linalg.norm(axis)
            np.testing.assert_allclose(axis,[.5,0,np.sqrt(3)/2],atol=1e-12)
            self.assertEqual(task['source_pass'],6)
            self.assertIn('pass06',task['source_shape'])
            with self.assertRaises(FileExistsError): fs.prepare(root,setup,[0,30])

    def test_rejects_permeable_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'8928'
            row,_=fixture(root/'a00')
            row['canopy_model']='fabric'
            fs.save(root/'a00/out/nos6_held.json',[row])
            with self.assertRaises(ValueError): fs.prepare(root,Path(tmp)/'setup',[0])

    def test_old_8928_schema_without_diameter_or_physics_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'8928';row,_=fixture(root/'a00')
            for key in ['D0','V','nu','rho','canopy_model','fabric_coefficients','attitude_mode',
                        'cord_width_m','cord_cd','pressure_gradient','turbulence_length_m','turbulence_intensity']:
                row.pop(key,None)
            fs.save(root/'a00/out/nos6_held.json',[row])
            data=fs.prepare(root,Path(tmp)/'setup',[0])
            task=data['tasks'][0]
            self.assertAlmostEqual(task['topology']['D0'],np.sqrt(24/np.pi))
            self.assertIn('D0',task['inferred_parameters'])
            self.assertIn('declared source series',task['wall_provenance'])
            row['lines']=19.51
            fs.save(root/'a00/out/nos6_held.json',[row])
            with self.assertRaises(ValueError):fs.prepare(root,Path(tmp)/'wrong_scale',[0])

    def test_command_is_wall_and_frozen_and_rejects_seed_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'8928';fixture(root/'a00')
            setup=Path(tmp)/'setup';fs.prepare(root,setup,[0])
            cmd,_,_=fs.command(setup/'manifest.json',0)
            self.assertIn('--fixed-shape',cmd)
            self.assertIn('--pressure-gradient=gauss',cmd)
            self.assertFalse(any(x.startswith('--fabric=') for x in cmd))
            self.assertEqual(cmd[6],'1')
            np.save(setup/'reference_a00.npy',np.zeros((1250,3)))
            with self.assertRaises(ValueError): fs.command(setup/'manifest.json',0)

    def test_frozen_result_verification_rejects_structural_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'8928';row,_=fixture(root/'a00')
            setup=Path(tmp)/'setup';data=fs.prepare(root,setup,[0])
            work=Path(tmp)/'run/reference_a00'
            (work/'fields/nos6_held/pass01').mkdir(parents=True)
            (work/'out').mkdir()
            for name in fs.NUMERICS:
                p=work/'case'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('same setting')
            seed=np.load(setup/data['tasks'][0]['seed_file'])
            np.save(work/'fields/nos6_held/pass01/input_shape.npy',seed)
            np.save(work/'out/nos6_held_shape.npy',seed)
            fs.save(work/'study_case.json',dict(manifest=str(setup/'manifest.json'),task_index=0))
            row.update(fixed_shape=True,iters=2000,S0=6.,cells=1)
            row['pass']=1
            fs.save(work/'out/nos6_held.json',[row])
            fs.verify(work)
            fs.collect(work.parent)
            self.assertEqual(json.loads((work.parent/'geometry_comparison.json').read_text())['missing_cases'],[])
            np.save(work/'out/nos6_held_shape.npy',seed+1e-6)
            with self.assertRaises(ValueError): fs.verify(work)

    def test_holding_comparison_normalizes_axes_and_checks_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            left,right=Path(tmp)/'legacy',Path(tmp)/'inner'
            fixture(left,30)
            row,_=fixture(right,30,'inner','nosarev_speed')
            output=Path(tmp)/'comparison.json'
            fs.compare_holding(left,right,output)
            result=json.loads(output.read_text())
            np.testing.assert_allclose(list(result['native_delta'].values()),0,atol=1e-14)
            self.assertTrue(result['controlled_holding_comparison'])
            row['fabric_coefficients']='33.2,28.6'
            fs.save(right/'out/nos6_held.json',[row])
            with self.assertRaises(ValueError): fs.compare_holding(left,right,Path(tmp)/'mismatch.json')

    def test_a_fixed_run_cannot_measure_holding_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            left,right=Path(tmp)/'legacy',Path(tmp)/'inner'
            fixture(left,30)
            row,_=fixture(right,30,'inner');row['fixed_shape']=True
            fs.save(right/'out/nos6_held.json',[row])
            with self.assertRaises(ValueError): fs.compare_holding(left,right,Path(tmp)/'invalid.json')

    def test_actual_generated_driver_accepts_frozen_wall_settings(self):
        from test_case_configuration import ConfigurationTests
        with tempfile.TemporaryDirectory() as tmp:
            seed=Path(tmp)/'seed.npy';np.save(seed,np.zeros((1250,3)))
            source=ConfigurationTests().generated(False,extra=[
                '--fixed-shape','--attitude=legacy','--mesh=1.25',
                '--pressure-gradient=gauss','--turbulence-length=1.24',
                '--turbulence-intensity=.01'],passes=1,seed=seed)
            self.assertIn('range(0 if FIXED_SHAPE else CHUNKS)',source)
            self.assertIn('and not FIXED_SHAPE',source)


if __name__=='__main__': unittest.main()
