"""Coordinate covariance and actual dictionary/geometry transforms."""
from pathlib import Path
import base64
import json
import sys
import unittest
import xml.etree.ElementTree as ET
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'coupled'))
import nosarev_frame as frame
from nos6_geometry import closed_circular
from parafoil.inflation.relax import inflate
import test_case_configuration as configurations
import test_postprocessing as post
import pressure_table
import aero_diagnostics
import vtp
from nosarev_view import material_view


class NosarevFrameTests(unittest.TestCase):
    def mesh(self):
        return dict(diameter=np.sqrt(24/np.pi),length=np.sqrt(24/np.pi),
                    gores=8,rings=2,line_stride=2,shear_ratio=.02)

    def test_proper_rotation_cross_products_and_force_conventions(self):
        q=frame.ROTATION
        np.testing.assert_allclose(q@q.T,np.eye(3))
        self.assertAlmostEqual(np.linalg.det(q),1.)
        a=np.array([1.,2.,3.]);b=np.array([4.,-5.,6.])
        np.testing.assert_allclose(np.cross(frame.to_speed(a),frame.to_speed(b)),frame.to_speed(np.cross(a,b)))
        np.testing.assert_allclose(frame.to_speed([0,0,30]),[-30,0,0])
        f=np.array([2.,3.,10.]);m=np.array([4.,-5.,6.])
        self.assertEqual(frame.to_speed(f)@frame.FLOW,10.)
        self.assertEqual(frame.to_speed(f)@frame.LIFT,-2.)
        self.assertEqual(frame.to_speed(m)@frame.PITCH,-5.)

    def test_geometry_and_projected_diameter_rotate_with_the_project(self):
        native=closed_circular(**self.mesh())
        speed=frame.speed_circular(**self.mesh())
        np.testing.assert_allclose(speed.position,frame.to_speed(native.position))
        np.testing.assert_allclose(speed.flat,frame.to_speed(native.flat))
        self.assertAlmostEqual(speed.projected_diameter(),native.projected_diameter())
        np.testing.assert_array_equal(speed.rods,native.rods)
        np.testing.assert_array_equal(speed.rest_length,native.rest_length)

    def test_structural_relaxation_is_covariant_with_inner_attitude(self):
        native=closed_circular(**self.mesh());speed=frame.speed_circular(**self.mesh())
        f=np.zeros_like(native.position);f[:-2]=[.3,0.,2.]
        a=inflate(native,node_force=f,max_steps=10,attitude_deg=30)
        b=frame.speed_inflate(speed,node_force=frame.to_speed(f),max_steps=10,attitude_deg=30)
        np.testing.assert_allclose(frame.to_native(speed.position),native.position,atol=1e-12)
        np.testing.assert_allclose(b.tension,a.tension,atol=1e-10)
        np.testing.assert_allclose(b.confluence_load,frame.to_speed(a.confluence_load),atol=1e-10)
        np.testing.assert_allclose(b.constraint_force,frame.to_speed(a.constraint_force),atol=1e-10)
        self.assertAlmostEqual(b.residual,a.residual,places=12)

    def test_dictionary_rotates_domain_boxes_and_cut_without_corrupting_counts(self):
        original='vertices ((-1 -2 -3) (4 5 6)); blocks (hex (0 1 2 3 4 5 6 7) (40 40 64) simpleGrading (1 1 1));'
        rotated=frame.rotate_dictionary('system/blockMeshDict',original)
        self.assertIn('(3 1 -2)',rotated)
        self.assertIn('(-6 -4 5)',rotated)
        self.assertIn('(40 40 64)',rotated)
        self.assertIn('simpleGrading (1 1 1)',rotated)
        box=frame.rotate_dictionary('system/snappyHexMeshDict',
            'min (-1 -2 -3); max (4 5 6); centre (0 0 2); locationInMesh (0 0 -5);')
        self.assertIn('min (-6 -4 -2); max (3 1 5);',box)
        self.assertIn('centre (-2 0 0)',box)
        self.assertIn('locationInMesh (5 0 0)',box)
        cut=frame.rotate_dictionary('system/controlDict','point (0 0.001 0); normal (0 1 0); start (-1 0 -2);')
        self.assertIn('point (0 0 0.001)',cut)
        self.assertIn('normal (0 0 1)',cut)
        self.assertIn('start (2 1 0)',cut)

    def test_generated_new_profile_is_explicit_and_reuses_existing_driver(self):
        source=configurations.ConfigurationTests().generated(True,
            ['--patch=nos6,cords,nosarev_axes','--seed-frame=native'])
        self.assertIn('install_speed_frame(globals())',source)
        self.assertIn('"coordinate_frame": "nosarev_speed"',source)
        self.assertIn('coordinate_frame="nosarev_speed"',source)
        self.assertIn('if False:  # no implicit native-coordinate cache',source)

    def test_seed_frame_and_signed_mean_axis_are_explicit(self):
        mesh=closed_circular(**self.mesh())
        g=dict(AZIMUTH=0.,ALPHA=0.,write=lambda *a:None,set_attitude=lambda *a:None,
               refine_ball=lambda *a:None,source_hashes=lambda *a:{},CANOPY=24,HOLD=30.)
        frame.install_speed_frame(g)
        rotated=g['speed_seed'](mesh.position,'native')
        np.testing.assert_allclose(rotated,frame.to_speed(mesh.position))
        with self.assertRaisesRegex(ValueError,'hemisphere'):
            g['speed_seed'](mesh.position,'nosarev')
        for angle in [-30.,0.,30.]:
            position=rotated.copy()
            t=np.radians(angle)
            position[:24]=[-np.cos(t),-np.sin(t),0.]
            position[24]=0.
            self.assertAlmostEqual(g['tilt_of'](position),angle,places=12)


class NosarevPostprocessingTests(unittest.TestCase):
    def setUp(self):
        post.PostprocessingTests.setUp(self)

    def tearDown(self):
        post.PostprocessingTests.tearDown(self)

    def rotate_fixture(self):
        for path in self.run.rglob('*.npy'):
            values=np.load(path)
            np.save(path,frame.to_speed(values))
        for path in self.run.rglob('*.vtp'):
            tree=ET.parse(path)
            for block in tree.findall('.//DataArray'):
                if block.get('Name') not in {'Points','U'}: continue
                raw=base64.b64decode(block.text)
                values=np.frombuffer(raw[8:],dtype=np.float64).reshape(-1,3)
                payload=frame.to_speed(values).astype(np.float64).tobytes()
                block.text=base64.b64encode(np.array([len(payload)],dtype=np.uint64).tobytes()+payload).decode()
            tree.write(path,encoding='utf-8')
        path=self.run/'out/nos6_held.json'
        rows=json.loads(path.read_text())
        for row in rows: row.update(coordinate_frame='nosarev_speed',cy=-row['cy'])
        path.write_text(json.dumps(rows))

    def test_surface_meridian_and_wake_use_actual_global_frame(self):
        post.PostprocessingTests.write_linear_cut(self)
        row=post.results.read_records(self.run)[-1]
        _,a=aero_diagnostics.surface(self.run,'nos6_held',row,self.diameter,30,1.225)
        pa=pressure_table.extract(self.args,self.run)
        self.rotate_fixture()
        row=post.results.read_records(self.run)[-1]
        _,b=aero_diagnostics.surface(self.run,'nos6_held',row,self.diameter,30,1.225)
        pb=pressure_table.extract(self.args,self.run)
        np.testing.assert_allclose(b['face_force_N'],frame.to_speed(a['face_force_N']),atol=1e-9)
        self.assertEqual(b['axes']['pitch'],'+Z')
        self.assertAlmostEqual(a['cx_pressure_recomputed'],b['cx_pressure_recomputed'])
        self.assertAlmostEqual(a['cy_pressure_recomputed'],-b['cy_pressure_recomputed'])
        self.assertAlmostEqual(a['mz_pressure_recomputed'],b['mz_pressure_recomputed'])
        np.testing.assert_allclose(pa.table,pb.table,atol=1e-12)
        np.testing.assert_allclose(pb.sampling[:,1:4],frame.to_speed(pa.sampling[:,1:4]),atol=1e-12)
        aero_diagnostics.wake_cut(self.run,'nos6_held',row,3.,30.)
        wake=np.loadtxt(self.run/'out/wake_cut.csv',delimiter=',',skiprows=1)
        np.testing.assert_allclose(wake[:,2],.5,atol=1e-12)
        meta=json.loads((self.run/'out/wake_diagnostics.json').read_text())
        self.assertEqual(meta['omega_component'],'dUy/dx - dUx/dy')

    def test_material_view_restores_functions_and_rotates_solver_seeds(self):
        load,arrays,write=np.load,vtp.arrays,Path.write_text
        path=self.run/'controlDict'
        with self.assertRaisesRegex(ValueError,'intentional'):
            with material_view(sampling=True):
                path.write_text('seedSampleSet {points ((1 2 3) (4 5 6));}')
                self.assertIn('(-3 -1 2)',path.read_text())
                raise ValueError('intentional')
        self.assertIs(np.load,load)
        self.assertIs(vtp.arrays,arrays)
        self.assertIs(Path.write_text,write)


if __name__=='__main__': unittest.main()
