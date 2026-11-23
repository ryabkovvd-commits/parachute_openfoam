"""Compile generated cluster cases without executing the CFD driver."""
import runpy
import sys
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
                     'SHIFT_TOL, LOAD_TOL = 0.03, None','cord_load = read_cord_load',
                     '"canopy_model"','"D0": D0']:
            self.assertIn(text,source)
        self.assertEqual(bool(captured['boundary']),fabric)
        return source

    def test_wall_configuration(self):
        self.assertIn('".*canopy.*" {{ type noSlip; }}',self.generated(False))

    def test_fabric_configuration(self):
        self.assertIn('{FABRIC_U}',self.generated(True))


if __name__=='__main__':
    unittest.main()
