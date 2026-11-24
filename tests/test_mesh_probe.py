"""Exercise the embedded mesh-probe preparation and reporting on real files."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


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


if __name__=='__main__': unittest.main()
