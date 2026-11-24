"""Conservation, vortex-gradient, status and complete-pipeline checks."""
import base64
import csv
import json
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import test_pressure_table as fixtures
from test_pressure_table import vtp_file
import aero_diagnostics as diagnostics
import nos6_results as results


class PostprocessingTests(unittest.TestCase):
    def setUp(self):
        fixtures.MeridianTests.setUp(self)
        self.diameter = np.sqrt(24/np.pi)
        path = self.run/'out/nos6_held.json'
        rows = json.loads(path.read_text())
        for row in rows:
            row.update(lines=self.diameter,stride=4,shear=.02,cx=.5,cy=0.,mz=0.,ratio=.6,
                       ratio_seen=.61,gap=.01,shift=.005,dload=.02,omega=.5,relaxed=True,
                       drag=100.,side=0.,moment=0.,canopy_model='wall',D0=self.diameter,V=30.,rho=1.225)
        path.write_text(json.dumps(rows))
        np.save(self.run/'out/nos6_held_shape.npy',self.points)
        np.save(self.run/'out/кадры/nos6_held_проход12.npy',self.points)
        np.save(self.run/'out/кадры/nos6_held_проход06.npy',self.points)
        (self.run/'pics').mkdir()

    def tearDown(self):
        fixtures.MeridianTests.tearDown(self)

    def test_surface_loads_and_transfer_conservation(self):
        row = results.read_records(self.run)[-1]
        paths,report = diagnostics.surface(self.run,'nos6_held',row,self.diameter,30.,1.225)
        expected = 1.225*700*.5*self.g*.8**2*np.sin(2*np.pi/self.g)
        np.testing.assert_allclose(report['face_force_N'],[0,0,expected],atol=1e-9)
        self.assertLess(report['force_transfer_relative'],1e-12)
        self.assertLess(abs(report['moment_transfer_error_mz']),1e-12)
        self.assertEqual(report['paired_area_fraction'],1.)
        self.assertTrue(all(Path(p).is_file() for p in paths))
        tree = ET.parse(next(self.run.glob('pics/*deltaCp.vtp')))
        flags = tree.find(".//DataArray[@Name='paired']").text.split()
        self.assertTrue(all(flag=='1' for flag in flags))
        bands=np.genfromtxt(self.run/'out/moment_bands_pass12.csv',delimiter=',',names=True,encoding='utf-8-sig')
        np.testing.assert_allclose(bands['Cp_master']-bands['Cp_slave'],bands['delta_Cp'],atol=1e-12)
        np.testing.assert_allclose(bands['dmz_zFx']+bands['dmz_minus_xFz'],bands['dmz'],atol=1e-12)
        with (self.run/'out/moment_halves_pass12.csv').open(encoding='utf-8-sig') as stream:
            halves=list(csv.DictReader(stream))
        self.assertEqual(len(halves),2)
        self.assertAlmostEqual(sum(float(r['dmz']) for r in halves),sum(bands['dmz']),places=12)
        self.assertAlmostEqual(report['moment_terms_paired_mz']['zFx'],sum(bands['dmz_zFx']),places=12)

    def write_linear_cut(self):
        coords = np.array([[x,.001,z] for z in np.linspace(0,2,5) for x in np.linspace(-1,1,5)])
        triangles = []
        for j in range(4):
            for i in range(4):
                a = j*5+i
                triangles.extend([[a,a+1,a+6],[a,a+6,a+5]])
        path = self.run/'fields/nos6_held/pass12/postProcessing/cut/2000/plane.vtp'
        path.parent.mkdir(parents=True)
        vtp_file(path,coords,np.array(triangles),np.zeros(len(triangles)))
        tree = ET.parse(path)
        values = np.c_[2*coords[:,2],np.zeros(len(coords)),-3*coords[:,0]].astype(np.float64)
        data = values.tobytes()
        payload = base64.b64encode(np.array([len(data)],dtype=np.uint64).tobytes()+data).decode()
        block = ET.SubElement(tree.find('.//Piece'),'PointData')
        array = ET.SubElement(block,'DataArray',Name='U',type='Float64',NumberOfComponents='3',format='binary')
        array.text = payload
        tree.write(path,encoding='utf-8')

    def test_analytical_vorticity(self):
        self.write_linear_cut()
        diagnostics.wake_cut(self.run,'nos6_held',results.read_records(self.run)[-1],3.,30.)
        table = np.genfromtxt(self.run/'out/wake_cut.csv',delimiter=',',skip_header=1)
        # Ux=2z, Uz=-3x => omega_y=5; D0/V = 3/30.
        np.testing.assert_allclose(table[:,2],.5,atol=1e-12)

    def test_stale_status_is_not_convergence(self):
        (self.run/'out/nos6_held.done').write_text('{"pass":11}')
        self.assertEqual(results.saved_status(self.run,12),'unconfirmed')
        (self.run/'out/nos6_held.НЕСОШЁЛСЯ').write_text('{"pass":12}')
        self.assertEqual(results.saved_status(self.run,12),'not_converged')

    def test_failure_does_not_hide_other_stages(self):
        report = results.CaseReport(self.run,{'pass':12},'nos6_held')
        def fail():
            raise ValueError('broken input')
        report.stage('bad',fail)
        report.stage('good',lambda:[])
        data = report.finish()
        self.assertEqual(data['status'],'failed')
        self.assertEqual(data['stages'][1]['status'],'ok')

    def test_batch_discovery_and_summary(self):
        self.assertEqual(results.discover([self.run]),[self.run.resolve()])
        artifacts = results.polar(self.run/'summary',[self.run])
        with (self.run/'summary/polar.csv').open(encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['status'],'unconfirmed')
        self.assertTrue(all(Path(p).exists() for p in artifacts))

    def test_pipeline_without_solver(self):
        self.write_linear_cut()
        code = results.main(['case',str(self.run),'--skip-plots','--no-summary'])
        self.assertEqual(code,0)
        report = json.loads((self.run/'out/postprocess_report.json').read_text(encoding='utf-8'))
        self.assertTrue(all(stage['status']=='ok' for stage in report['stages']))
        self.assertTrue((self.run/'out/confluence_loads.csv').is_file())


if __name__=='__main__':
    unittest.main()
