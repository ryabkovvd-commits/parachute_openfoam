"""Numerical checks for pressure normalization, geometry and archive selection."""
import base64
import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'coupled'))
import pressure_table as target


def vtp_file(path, points, triangles, pressure):
    def array(name, values, kind, width=1):
        raw = values.tobytes()
        payload = base64.b64encode(np.array([len(raw)],dtype=np.uint64).tobytes()+raw).decode()
        components = f" NumberOfComponents='{width}'" if width>1 else ''
        return f"<DataArray Name='{name}' type='{kind}'{components} format='binary'>{payload}</DataArray>"
    path.write_text(
        "<VTKFile type='PolyData' byte_order='LittleEndian' header_type='UInt64'><PolyData>"
        f"<Piece NumberOfPoints='{len(points)}' NumberOfPolys='{len(triangles)}'>"
        '<Points>'+array('Points',points.astype(np.float64),'Float64',3)+'</Points>'
        '<Polys>'+array('connectivity',triangles.ravel().astype(np.int32),'Int32')
        +array('offsets',np.arange(3,3*len(triangles)+1,3,dtype=np.int32),'Int32')+'</Polys>'
        '<CellData>'+array('p',pressure.astype(np.float64),'Float64')+'</CellData>'
        '</Piece></PolyData></VTKFile>', encoding='ascii')


class MeridianTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run = Path(self.temp.name)
        self.g, self.r = 64, 3  # Intentionally different from the 96 x 12 production mesh.
        radii = np.linspace(.25,1,self.r+1)
        phi = np.arange(self.g)*2*np.pi/self.g
        points = np.array([[.8*rr*np.cos(p),.8*rr*np.sin(p),2+.6*(1-rr*rr)]
                           for rr in radii for p in phi])
        points = np.vstack((points,[0,0,0],[0,0,2.6]))
        triangles = []
        for i in range(self.r):
            for j in range(self.g):
                a,b = i*self.g+j,i*self.g+(j+1)%self.g
                c,d = a+self.g,b+self.g
                triangles.extend([[a,b,d],[a,d,c]])
        pole = len(points)-1
        triangles += [[pole,j,(j+1)%self.g] for j in range(self.g)]
        triangles = np.array(triangles)
        for tri in triangles:
            p = points[tri]
            if np.cross(p[1]-p[0],p[2]-p[0]) @ p.mean(0)<0:
                tri[1],tri[2] = tri[2],tri[1]
        frames = self.run/'out/кадры'
        frames.mkdir(parents=True)
        for number in [5,11]:
            np.save(frames/f'nos6_held_проход{number:02d}.npy',points)
        np.save(frames/'nos6_held_проход12.npy',points+10) # Must never be used for CFD pass 12.
        row = dict(gores=self.g,rings=self.r,hold=0.)
        (self.run/'out/nos6_held.json').write_text(json.dumps([dict(row,pass_number=12,**{'pass':12}),
                                                                dict(row,**{'pass':6})]))
        (self.run/'out/nos6_held.txt').write_text('Поток V = 30 м/с',encoding='utf-8')
        for number,time in [(6,'2000'),(12,'250'),(12,'2000')]:
            folder = self.run/f'fields/nos6_held/pass{number:02d}/postProcessing/surfacePressure/{time}'
            folder.mkdir(parents=True)
            vtp_file(folder/'master.vtp',points,triangles,np.full(len(triangles),650. if time=='2000' else 999.))
            vtp_file(folder/'slave.vtp',points,triangles[:,::-1],np.full(len(triangles),-50.))
        self.args = SimpleNamespace(name='nos6_held',out_dir=None,fields_dir=None,surface_dir=None,
                                    pass_number=None,shape=None,V=None,diameter=3.,pole='closed',vent_fraction=None,
                                    azimuth=0.,subdivide=4,sector=6.,normal_angle=60.,neighbours=4,p_inf=50.,pressure_field='p')
        self.points,self.triangles = points,triangles

    def tearDown(self):
        self.temp.cleanup()

    def test_constant_pressure_and_geometry(self):
        result = target.extract(self.args,self.run)
        np.testing.assert_allclose(result.table[:,1],600/450,atol=1e-12)
        np.testing.assert_allclose(result.table[:,2],-100/450,atol=1e-12)
        np.testing.assert_allclose(result.table[:,3],700/450,atol=1e-12)
        pole = result.compact[5]
        np.testing.assert_allclose(pole[[0,4,5]],[0,0,.2],atol=1e-12)
        self.assertEqual(result.metadata['pass_number'],12)
        self.assertTrue(result.metadata['input_shape'].endswith('проход11.npy'))
        self.assertTrue(result.metadata['surface'].endswith('2000'))

    def test_patch_names_do_not_determine_the_sides(self):
        folder = self.run/'fields/nos6_held/pass12/postProcessing/surfacePressure/2000'
        a,b = (folder/'master.vtp').read_bytes(),(folder/'slave.vtp').read_bytes()
        (folder/'master.vtp').write_bytes(b)
        (folder/'slave.vtp').write_bytes(a)
        result = target.extract(self.args,self.run)
        np.testing.assert_allclose(result.table[:,3],700/450,atol=1e-12)

    def test_small_oppositely_oriented_face_cannot_contaminate_pressure(self):
        folder = self.run/'fields/nos6_held/pass12/postProcessing/surfacePressure/2000'
        extra = np.array([[.4,-.0001,2.45],[.4001,0,2.45],[.4,.0001,2.45]])
        points = np.vstack((self.points,extra))
        bad = np.arange(len(self.points),len(points))
        triangles = np.vstack((self.triangles[:,::-1],bad))
        pressure = np.r_[np.full(len(self.triangles),-50.),999999.]
        vtp_file(folder/'slave.vtp',points,triangles,pressure)
        result = target.extract(self.args,self.run)
        np.testing.assert_allclose(result.table[:,3],700/450,atol=1e-12)
        self.assertEqual(result.metadata['diagnostics']['slave']['excluded_opposite_normal_faces'],1)

    def test_missing_input_frame_is_an_error(self):
        (self.run/'out/кадры/nos6_held_проход11.npy').unlink()
        with self.assertRaisesRegex(FileNotFoundError,'INPUT shape'):
            target.extract(self.args,self.run)

    def test_nearby_tangential_cut_face_cannot_replace_canopy_pressure(self):
        folder = self.run/'fields/nos6_held/pass12/postProcessing/surfacePressure/2000'
        e = 1e-4
        extra = np.array([[e,-e,2.6-e],[e,-e,2.6+e],[e,e,2.6]])
        points = np.vstack((self.points,extra))
        triangles = np.vstack((self.triangles[:,::-1],np.arange(len(self.points),len(points))))
        pressure = np.r_[np.full(len(self.triangles),-50.),999999.]
        vtp_file(folder/'slave.vtp',points,triangles,pressure)
        result = target.extract(self.args,self.run)
        np.testing.assert_allclose(result.table[:,3],700/450,atol=1e-12)
        self.assertEqual(result.metadata['diagnostics']['slave']['excluded_opposite_normal_faces'],0)
        pole = result.sampling[np.argmin(np.abs(result.sampling[:,0]))]
        self.assertLess(pole[15],pole[12]) # Raw closest facet is not the compatible closest one.
        self.assertLess(pole[17],np.cos(np.radians(self.args.normal_angle)))

    def test_sampling_trace_and_compact_bracketing_distances(self):
        result = target.extract(self.args,self.run)
        full = result.sampling
        self.assertEqual(full.shape[1],18)
        self.assertTrue(np.all(full[:,12]+1e-12>=full[:,15]))
        self.assertTrue(np.all(full[:,13]+1e-12>=full[:,12]))
        for diagnostic in result.metadata['diagnostics'].values():
            nearest_col,raw_col=(5,8) if diagnostic['side']=='in' else (12,15)
            index=int(np.argmax(full[:,nearest_col]))
            self.assertAlmostEqual(diagnostic['nearest_face_distance_max_R'],full[index,0])
            self.assertAlmostEqual(diagnostic['raw_nearest_at_max_m'],full[index,raw_col])
        for row in result.sampling_compact:
            mask=(full[:,0]>=row[1]) & (full[:,0]<=row[2])
            self.assertTrue(mask.any())
            np.testing.assert_allclose(row[3:],[full[mask,5].max(),full[mask,12].max(),
                                              full[mask,15].max(),full[mask,13].max()])
        paths=target.write_sampling(self.run/'sample',result)
        self.assertTrue(all(Path(p).is_file() for p in paths))

    def test_sampling_report_cli_prints_metres_and_preserves_pressure(self):
        captured=io.StringIO()
        with contextlib.redirect_stdout(captured):
            result=target.main([str(self.run),'--V','30','--diameter','3','--subdivide','4',
                                '--sampling-report','--pics',str(self.run/'sampling_figures')])[0]
        lines=[s for s in captured.getvalue().splitlines() if s.startswith('sample ')]
        self.assertEqual(len(lines),12) # Header plus eleven compact locations.
        np.testing.assert_allclose(result.table[:,3],700/450,atol=1e-12)
        self.assertIn('near_out_m',lines[0])
        self.assertEqual(len([s for s in captured.getvalue().splitlines() if s.startswith('sample_max ')]),2)

    def test_explicit_pass_selection(self):
        self.args.pass_number = 6
        result = target.extract(self.args,self.run)
        self.assertEqual(result.metadata['pass_number'],6)
        self.assertTrue(result.metadata['input_shape'].endswith('проход05.npy'))

    def mixed_patch(self,run):
        folder = run/'fields/nos6_held/pass12/postProcessing/surfacePressure/2000'
        triangles = self.triangles.copy()
        corners = self.points[triangles]
        area = .5*np.linalg.norm(np.cross(corners[:,1]-corners[:,0],corners[:,2]-corners[:,0]),axis=1)
        number = int(np.searchsorted(np.cumsum(area),.05*area.sum())+1)
        triangles[:number] = triangles[:number,::-1]
        vtp_file(folder/'master.vtp',self.points,triangles,np.full(len(triangles),650.))

    def test_diagnostic_does_not_relax_ambiguous_side_check(self):
        self.mixed_patch(self.run)
        with self.assertRaisesRegex(ValueError,'ambiguous inside/outside'):
            target.extract(self.args,self.run)
        self.args.pics = self.run/'normal_report'
        with contextlib.redirect_stdout(io.StringIO()):
            report = target.normal_diagnostics(self.args,self.run)
        stats = report['patches']['master']
        self.assertTrue(.90<stats['positive_area_fraction']<.99)
        self.assertGreater(stats['minority_faces'],0)
        self.assertTrue(Path(stats['face_table']).is_file())
        self.assertTrue(list(self.args.pics.glob('*.png')))
        self.assertNotIn('side',stats)

    def test_cli_preserves_successful_figures_when_other_case_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            bad = Path(temp)/'bad'
            shutil.copytree(self.run,bad)
            self.mixed_patch(bad)
            pics = Path(temp)/'comparison'
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    target.main([str(self.run),str(bad),'--V=30','--diameter=3','--subdivide=4',f'--pics={pics}'])
            self.assertEqual(caught.exception.code,1)
            self.assertTrue((pics/'pressure_meridians.png').is_file())
            self.assertTrue((pics/'canopy_profiles.png').is_file())


if __name__=='__main__':
    unittest.main()
