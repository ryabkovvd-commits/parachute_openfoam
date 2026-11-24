"""Exercise the embedded mesh-probe preparation and reporting on real files."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace


SCRIPT=Path(__file__).resolve().parents[1]/'cluster/mesh_probe.slurm'


def snippets():
    source=SCRIPT.read_text(encoding='utf-8')
    prepare=source.split('"$P" - "$S" "$W" "$FACTOR" <<\'PY\'\n')[1].split('\nPY\n')[0]
    report=source.split('"$P" - "$W" <<\'PY\'\n')[1].split('\nPY\n')[0]
    return prepare,report


def execute(code,args):
    old=sys.argv[:]
    try:
        sys.argv=['embedded',*map(str,args)]
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code,'mesh-probe','exec'),{})
    finally: sys.argv=old


class MeshProbeTests(unittest.TestCase):
    def prepare(self,root):
        source,work=root/'source',root/'work'
        for folder in [source,work]:
            (folder/'system').mkdir(parents=True)
            (folder/'constant/triSurface').mkdir(parents=True)
            (folder/'system/blockMeshDict').write_text(
                'blocks (hex (0 1 2 3 4 5 6 7) (40 40 64) simpleGrading (1 1 1));')
            (folder/'system/snappyHexMeshDict').write_text('geometry unchanged;')
            (folder/'system/controlDict').write_text('original control;')
            (folder/'constant/triSurface/canopy.stl').write_text('fixed geometry fixture')
        return source,work

    def test_only_block_counts_change_and_source_is_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            source,work=self.prepare(Path(temp))
            execute(snippets()[0],[source,work,1.25])
            self.assertIn('(50 50 80)',(work/'system/blockMeshDict').read_text())
            self.assertIn('(40 40 64)',(source/'system/blockMeshDict').read_text())
            self.assertEqual((source/'system/controlDict').read_text(),'original control;')
            data=json.loads((work/'probe_input.json').read_text())
            self.assertEqual(data['source_stl_sha256'],data['copied_stl_sha256'])
            self.assertEqual(data['source_snappy_sha256'],data['copied_snappy_sha256'])
            self.assertIn('functions {}',(work/'system/controlDict').read_text())

    def test_counts_before_and_after_deletion_are_not_conflated(self):
        with tempfile.TemporaryDirectory() as temp:
            source,work=self.prepare(Path(temp))
            execute(snippets()[0],[source,work,1])
            for stage,count in [('beforeSplit',6694),('afterSplit',6682)]:
                (work/('boundary.'+stage)).write_text(
                    f'canopy {{ type wall; nFaces 6694; }}\n'
                    f'canopy_slave {{ type wall; nFaces {count}; }}')
            (work/'mesh_exit_code.txt').write_text('0')
            execute(snippets()[1],[work])
            data=json.loads((work/'mesh_probe.json').read_text())
            self.assertEqual(data['beforeSplit']['difference'],0)
            self.assertEqual(data['afterSplit']['difference'],12)

    def test_missing_mesh_is_reported_as_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            source,work=self.prepare(Path(temp))
            execute(snippets()[0],[source,work,1])
            (work/'mesh_exit_code.txt').write_text('2')
            execute(snippets()[1],[work])
            data=json.loads((work/'mesh_probe.json').read_text())
            self.assertIsNone(data['beforeSplit'])
            self.assertIsNone(data['afterSplit'])
            self.assertEqual(data['utility_exit_code'],2)

    def test_fixed_cfd_uses_probed_geometry_and_preserves_physical_flags(self):
        import hashlib
        import shutil
        import numpy as np
        sys.path.insert(0,str(SCRIPT.parents[1]/'coupled'))
        from nos6_geometry import closed_circular,export_closed_surface
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';work=root/'work';probe=root/'probe'
            for p in [source/'out',work,probe]: p.mkdir(parents=True)
            d=np.sqrt(24/np.pi)
            mesh=closed_circular(d,d,gores=8,rings=2,line_stride=2,shear_ratio=.02)
            np.save(source/'out/nos6_held_shape.npy',mesh.position)
            exported=export_closed_surface(mesh,probe/'canopy.stl',subdivide=12)
            digest=hashlib.sha256(exported.read_bytes()).hexdigest()
            meta=dict(utility_exit_code=0,afterSplit=dict(canopy=4,canopy_slave=4,difference=0),
                      source_case=str(source/'case'),copied_stl_sha256=digest)
            record=dict(D0=d,lines=d,gores=8,rings=2,stride=2,shear=.02,
                        attitude_mode='inner',**{'pass':4})
            flags={'--patch':'nos6,cords','--V':'30','--fabric':'33.2,28.6',
                   '--attitude':'inner','--mesh':'1.25','--угол':'30'}
            for path,data in [(probe/'mesh_probe.json',meta),(source/'out/nos6_held.json',[record]),
                              (source/'out/nos6_held_config.json',dict(physics=dict(flags=flags)))]:
                path.write_text(json.dumps(data))
            script=(SCRIPT.parent/'fixed_mesh_check.slurm').read_text(encoding='utf-8')
            code=script.split('<<\'PY\'\n')[1].split('\nPY\n')[0]
            def fake_run(command,cwd):
                target=Path(cwd)/'case/constant/triSurface/canopy.stl'
                target.parent.mkdir(parents=True)
                shutil.copyfile(Path(cwd)/'input_surface.stl',target)
                return SimpleNamespace(returncode=0)
            with patch('subprocess.run',side_effect=fake_run) as solver:
                with self.assertRaises(SystemExit) as done:
                    execute(code,[SCRIPT.parents[1],probe,work])
            self.assertEqual(done.exception.code,0)
            command=solver.call_args.args[0]
            self.assertIn('--fixed-shape',command)
            self.assertIn('--attitude=legacy',command)
            self.assertIn('--V=30',command)
            self.assertIn('--fabric=33.2,28.6',command)
            self.assertNotIn('--mesh=1.25',command)
            self.assertEqual(command[5:7],['200','1'])
            self.assertIn('--mesh-from='+str(probe/'constant/polyMesh'),command)
            result=json.loads((work/'fixed_mesh_input.json').read_text(encoding='utf-8'))
            self.assertEqual(result['input_stl_sha256'],digest)
            meta['copied_stl_sha256']='wrong'
            (probe/'mesh_probe.json').write_text(json.dumps(meta))
            with patch('subprocess.run') as solver:
                with self.assertRaisesRegex(ValueError,'exact probed STL'):
                    execute(code,[SCRIPT.parents[1],probe,work])
                solver.assert_not_called()


if __name__=='__main__': unittest.main()
