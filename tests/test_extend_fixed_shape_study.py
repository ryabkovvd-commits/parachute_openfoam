"""Continuation/provenance checks with synthetic fields and a mocked solver."""
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from test_fixed_shape_study import fixture, fs, ROOT

spec = importlib.util.spec_from_file_location('extend_study', ROOT/'cluster/extend_fixed_shape_study.py')
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)


def study(base):
    source = base/'8928'
    originals = {angle: fixture(source/f'a{angle:02d}', angle)[0] for angle in fs.FULL}
    setup = base/'old_setup'
    data = fs.prepare(source, setup, fs.FULL, iterations=200)
    root = base/'9016'
    work = root/'reference_a00'
    (work/'out').mkdir(parents=True)
    archive = work/'fields/nos6_held/pass01'
    archive.mkdir(parents=True)
    seed = setup/data['tasks'][0]['seed_file']
    shutil.copyfile(seed, archive/'input_shape.npy')
    shutil.copyfile(seed, work/'out/nos6_held_shape.npy')
    row = dict(originals[0], fixed_shape=True, iters=200, S0=6., cells=12, minutes=1.)
    row['pass'] = 1
    fs.save(work/'out/nos6_held.json', [row])
    fs.save(work/'study_case.json', dict(manifest=str(setup/'manifest.json'), task_index=0))
    for name in fs.NUMERICS:
        path = work/'case'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('same solver setting')
    for name in ['points','faces','owner','neighbour','boundary']:
        for folder in [work/'case/constant/polyMesh', archive/'polyMesh']:
            folder.mkdir(parents=True, exist_ok=True)
            (folder/name).write_text('same mesh')
    control = ('application simpleFoam; startFrom startTime; startTime 0;\n'
               'stopAt endTime; endTime 200; deltaT 1;\n'
               'writeControl timeStep; writeInterval 25;\n'
               'functions\n{ forces { writeInterval 25; } }\n')
    (work/'case/system/controlDict').write_text(control)
    (work/'case/log.simpleFoam').write_text('Time = 200\nEnd\n')
    saved(work, 200)
    output = base/'extension'
    ext.prepare(root, output)
    return root, work, output/'manifest.json'


def saved(work, iteration):
    for name in ['U','p','phi','nut','k','omega']:
        path = work/'case'/str(iteration)/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('saved field')
    for filename in ['force.dat','moment.dat']:
        path = work/'case/postProcessing/forces'/('0' if iteration == 200 else '200')/filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'{iteration} 1 2 3\n')
    (work/'case/postProcessing/surfacePressure'/str(iteration)).mkdir(parents=True, exist_ok=True)


class ExtensionTests(unittest.TestCase):
    def test_inline_controls_and_function_write_interval(self):
        source = ('application simpleFoam; startFrom startTime; startTime 0;\n'
                  'stopAt endTime; endTime 200; deltaT 1;\n'
                  'writeControl timeStep; writeInterval 25;\n'
                  'functions\n{ forces { writeInterval 25; } }')
        result = ext.control_text(source)
        self.assertIn('startFrom latestTime;', result)
        self.assertIn('endTime 2000;', result)
        self.assertIn('writeInterval 250;', result)
        self.assertIn('forces { writeInterval 25;', result)
        with self.assertRaises(ValueError):
            ext.control_text(source.replace('simpleFoam', 'pimpleFoam'))

    def test_preparation_preserves_original_manifest_and_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, work, manifest = study(Path(tmp))
            data = ext.read(manifest)
            original = Path(data['extension']['original_manifest'])
            self.assertEqual(ext.read(original)['iterations'], 200)
            self.assertEqual(data['iterations'], 2000)
            self.assertEqual(len(data['tasks']), 21)
            self.assertEqual(data['code_sha256'], fs.code_fingerprint())
            for task in data['tasks']:
                self.assertEqual(fs.sha(manifest.parent/task['seed_file']), task['seed_sha256'])
            self.assertEqual(ext.read(work/'out/nos6_held.json')[0]['iters'], 200)
            with self.assertRaises(FileExistsError):
                ext.prepare(root, manifest.parent)

    def test_complete_continuation_reuses_mesh_and_refreshes_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, work, manifest = study(Path(tmp))
            seed = np.load(work/'out/nos6_held_shape.npy')
            load = np.zeros_like(seed)
            cord = np.zeros_like(seed)
            cord[0] = [1., 0., 2.]
            def fake_execute(command, case, filename):
                (case/filename).write_text('Time = 2000\nEnd\n')
                if 'simpleFoam' in command:
                    saved(work, 2000)
            with patch.object(ext, 'locked', return_value=contextlib.nullcontext()), \
                 patch.object(ext, 'execute', side_effect=fake_execute) as solver, \
                 patch.object(ext.shutil, 'which', return_value='/mock/tool'), \
                 patch.object(ext.vtp, 'nodal_load', return_value=(load,np.array([3.,0.,2000.]),np.array([.2]))), \
                 patch.object(ext, 'read_cord_load', return_value=cord), \
                 patch.object(ext.subprocess, 'run', return_value=subprocess.CompletedProcess([],0)):
                self.assertEqual(ext.run(manifest, 0, root), 0)
            row = ext.read(work/'out/nos6_held.json')[0]
            self.assertEqual(row['iters'], 2000)
            self.assertEqual(row['CFD_restart_iteration'], 200)
            self.assertEqual(row['CFD_additional_iterations'], 1800)
            self.assertAlmostEqual(row['drag'], 2002.)
            self.assertAlmostEqual(row['side'], 4.)
            self.assertEqual(row['CFD_convergence'], 'unconfirmed')
            self.assertEqual(ext.read(work/'resume_2000_backup/out/nos6_held.json')[0]['iters'], 200)
            self.assertEqual(ext.read(work/'extension_2000.json')['state'], 'completed')
            self.assertEqual(len(solver.call_args_list), 3)
            self.assertIn('-latestTime', solver.call_args_list[0].args[0])
            self.assertTrue((work/'case/200/U').exists())
            self.assertTrue((work/'fields/nos6_held/pass01/2000/U').exists())
            np.testing.assert_array_equal(ext.table(work/'case/postProcessing/forces/0/force.dat')[:,0], [200,2000])
            fs.verify(work)

    def test_failure_preserves_old_result_and_records_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, work, manifest = study(Path(tmp))
            with patch.object(ext, 'locked', return_value=contextlib.nullcontext()), \
                 patch.object(ext.shutil, 'which', return_value='/mock/tool'), \
                 patch.object(ext, 'execute', side_effect=RuntimeError('solver failed')):
                with self.assertRaisesRegex(RuntimeError, 'solver failed'):
                    ext.run(manifest, 0, root)
            self.assertEqual(ext.read(work/'extension_2000.json')['state'], 'failed')
            self.assertEqual(ext.read(work/'out/nos6_held.json')[0]['iters'], 200)
            self.assertEqual(ext.read(work/'study_case.json')['manifest'], str(Path(tmp)/'old_setup/manifest.json'))

    def test_incomplete_latest_fields_and_changed_mesh_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, work, manifest = study(Path(tmp))
            (work/'case/200/p').unlink()
            with self.assertRaisesRegex(ValueError, 'Incomplete latest'):
                ext.latest(work/'case')
            saved(work, 200)
            (work/'case/constant/polyMesh/points').write_text('changed mesh')
            with patch.object(ext, 'locked', return_value=contextlib.nullcontext()), \
                 patch.object(ext, 'execute') as solver:
                with self.assertRaisesRegex(ValueError, 'meshes differ'):
                    ext.run(manifest, 0, root)
                solver.assert_not_called()

    @unittest.skipUnless(os.name == 'posix' and shutil.which('bash'), 'Linux Bash integration')
    def test_submission_waits_for_array_parent_and_schedules_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root, _, _ = study(base)
            project = base/'project'; project.mkdir()
            (project/'coupled').symlink_to(ROOT/'coupled', target_is_directory=True)
            (project/'cluster').symlink_to(ROOT/'cluster', target_is_directory=True)
            bins = base/'bin'; bins.mkdir()
            calls = base/'calls.json'
            (bins/'squeue').write_text('#!/bin/bash\nprintf "9016\\n9016\\n"\n')
            (bins/'sbatch').write_text('#!'+sys.executable+'\nimport json,os,sys\nfrom pathlib import Path\n'
                'p=Path(os.environ["SBATCH_CALLS"]);a=json.loads(p.read_text()) if p.exists() else []\n'
                'a.append(sys.argv[1:]);p.write_text(json.dumps(a));print(8000+len(a))\n')
            for p in bins.iterdir(): p.chmod(0o755)
            env = dict(os.environ, PATH=str(bins)+os.pathsep+os.environ.get('PATH',''),
                       NOS6_PYTHON=sys.executable, SBATCH_CALLS=str(calls), USER='test')
            done = subprocess.run(['bash', str(ROOT/'cluster/submit_extend_fixed_shape_study.sh'), str(root)],
                                  cwd=project, env=env, capture_output=True, text=True, timeout=45)
            self.assertEqual(done.returncode, 0, done.stderr)
            rows = json.loads(calls.read_text())
            self.assertIn('--dependency=afterany:9016', rows[0])
            self.assertIn('--array=0-20%2', rows[0])
            self.assertIn('--dependency=afterany:8001', rows[1])


if __name__ == '__main__':
    unittest.main()
