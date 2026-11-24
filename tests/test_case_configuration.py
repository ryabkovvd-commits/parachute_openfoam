"""Compile generated cluster cases without executing the CFD driver."""
import runpy
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


class ConfigurationTests(unittest.TestCase):
    def generated(self,fabric,extra=(),passes=12,seed=None):
        base = Path(__file__).resolve().parents[1]/'coupled'
        captured = {}
        class StopBeforeExecution(Exception):
            pass
        def trace(frame,event,arg):
            if event=='call' and frame.f_code.co_filename==str(base/'coupled.py'):
                captured['source'] = frame.f_back.f_globals['src']
                captured['boundary'] = frame.f_globals['FABRIC_U']
                raise StopBeforeExecution()
            return trace
        old_argv,old_path,old_trace = sys.argv[:],sys.path[:],sys.gettrace()
        sys.argv = [str(base/'cs.py'),'nos6_held','0','0','2000',str(passes)]+([str(seed)] if seed else [])+['--patch=nos6,cords',
                    '--cord-mm=3.2','--угол=30','--полотнищ=96','--шаг=4','--сдвиг=0.02',
                    '--V=30','--nu=1.46e-5']+(['--fabric=33.2,28.6'] if fabric else [])+list(extra)
        sys.settrace(trace)
        try:
            runpy.run_path(str(base/'cs.py'),run_name='__main__')
        except StopBeforeExecution:
            pass
        finally:
            sys.settrace(old_trace)
            sys.argv,sys.path = old_argv,old_path
        source = captured['source']
        compile(source,'generated-cluster-case','exec')
        minimum=1 if '--fixed-shape' in extra else 6
        for text in ['input_shape.npy',f'if steady >= 2 and step >= {minimum}:',
                     'SHIFT_TOL = ','cord_load = read_cord_load',
                     '"canopy_model"','"D0": D0']:
            self.assertTrue(text in source,f'Missing generated configuration element: {text}')
        self.assertEqual(bool(captured['boundary']),fabric)
        return source

    def test_wall_configuration(self):
        self.assertIn('".*canopy.*" {{ type noSlip; }}',self.generated(False))

    def test_fabric_configuration(self):
        self.assertIn('{FABRIC_U}',self.generated(True))

    def test_gauss_override_changes_only_pressure_gradient_scheme(self):
        default=self.generated(True)
        gauss=self.generated(True,['--pressure-gradient=gauss'])
        def schemes(source):
            return source.split('write("system/fvSchemes",',1)[1].split('write("system/fvSolution",',1)[0]
        expected=schemes(default).replace('gradSchemes { default cellLimited Gauss linear 1; }',
            'gradSchemes { default cellLimited Gauss linear 1; grad(p) Gauss linear; }')
        self.assertEqual(schemes(gauss),expected)
        self.assertNotIn('grad(p) Gauss linear;',schemes(default))
        self.assertIn('"pressure_gradient": \'gauss\'',gauss)
        self.assertIn('"pressure_gradient": \'default\'',default)

    def test_invalid_pressure_gradient_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.generated(True,['--pressure-gradient=typo'])

    def test_slurm_control_rejects_unrelated_numerical_changes(self):
        script=(Path(__file__).resolve().parents[1]/'cluster/permeability_models.slurm').read_text(encoding='utf-8')
        code=script.split('"$P" - "$W" "$BASELINE" <<\'PY\'\n',1)[1].split('\nPY\n',1)[0]
        with tempfile.TemporaryDirectory() as temp:
            base,work=Path(temp)/'baseline',Path(temp)/'control'
            for p in [base/'case/system',work/'case/system',work/'out',base/'out',base/'case/constant',work/'case/constant']: p.mkdir(parents=True)
            original='header\ngradSchemes { default cellLimited Gauss linear 1; }\ndivSchemes { default none; }\n'
            modified=original.replace('gradSchemes { default cellLimited Gauss linear 1; }',
                'gradSchemes { default cellLimited Gauss linear 1; grad(p) Gauss linear; }')
            (base/'case/system/fvSchemes').write_text(original,encoding='utf-8')
            (work/'case/system/fvSchemes').write_text(modified,encoding='utf-8')
            record=dict(pressure_gradient='gauss',V=30.,fabric_coefficients='33.2,28.6')
            (work/'out/nos6_held.json').write_text(json.dumps([record]),encoding='utf-8')
            (base/'out/nos6_held.json').write_text(json.dumps([dict(record,pressure_gradient='default')]),encoding='utf-8')
            for relative in ['system/fvSolution','constant/transportProperties','constant/turbulenceProperties','constant/cords','system/fvOptions']:
                for directory in [base,work]: (directory/'case'/relative).write_text('same settings',encoding='utf-8')
            old=sys.argv[:]
            sys.argv=['check',str(work),str(base)]
            try:
                with contextlib.redirect_stdout(io.StringIO()): exec(compile(code,'slurm-check','exec'),{})
                (work/'case/system/fvSchemes').write_text(modified.replace('default none;','default Gauss linear;'),encoding='utf-8')
                with self.assertRaisesRegex(SystemExit,'beyond the pressure-gradient'):
                    exec(compile(code,'slurm-check','exec'),{})
                (work/'case/system/fvSchemes').write_text(modified,encoding='utf-8')
                record['V']=31.
                (work/'out/nos6_held.json').write_text(json.dumps([record]),encoding='utf-8')
                with self.assertRaisesRegex(SystemExit,'Physical settings differ'):
                    exec(compile(code,'slurm-check','exec'),{})
            finally:
                sys.argv=old


if __name__=='__main__':
    unittest.main()
