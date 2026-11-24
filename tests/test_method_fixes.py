import ast
from dataclasses import replace
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'coupled'))
import run_guard
import vtp
import cord_study
import nos6_results
import method_compare
from nos6_geometry import closed_circular,export_closed_surface
from parafoil.inflation.constraints import MeanAxisConstraint
from parafoil.inflation.relax import inflate,_rod_tensions
import test_case_configuration as config_tests


def mesh():
    return closed_circular(2.,2.,gores=8,rings=2,line_stride=2,shear_ratio=.02)


class MethodFixTests(unittest.TestCase):
    def test_effective_requested_speed_viscosity_tolerance_and_inner(self):
        source=config_tests.ConfigurationTests().generated(True,['--V=20','--nu=0.000103','--shape-tol=0.01','--attitude=inner'])
        assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple)
                        and [getattr(t,'id',None) for t in n.targets[0].elts]==['D0','LINES','V','RHO','NU'])
        values=ast.literal_eval(assignment.value)
        self.assertEqual(values[2],20.)
        self.assertEqual(values[4],.000103)
        self.assertTrue('SHIFT_TOL = 0.01' in source)

    def test_unknown_option_is_rejected(self):
        with self.assertRaises(SystemExit):
            config_tests.ConfigurationTests().generated(True,['--fabrci=33.2,28.6'])

    def test_collapsed_geometry_rejected_without_radial_normal_assumption(self):
        original=mesh()
        self.assertGreater(run_guard.geometry_quality(original)['surface_area_m2'],0)
        with self.assertRaisesRegex(ValueError,'Collapsed'):
            run_guard.geometry_quality(replace(original,position=np.zeros_like(original.position)))
        reflected=original.position.copy();reflected[:,0]*=-1
        run_guard.geometry_quality(replace(original,position=reflected))

    def test_stl_normals_match_triangle_winding(self):
        original=mesh()
        original.position[original.skirt,2]+=np.sin(np.arange(8))*0.1
        with tempfile.TemporaryDirectory() as tmp:
            path=export_closed_surface(original,Path(tmp)/'canopy.stl',subdivide=2)
            text=path.read_text(encoding='ascii')
            points=np.array([[float(x) for x in v.split()] for v in re.findall(r'\bvertex\s+([^\n]+)',text)]).reshape(-1,3,3)
            normals=np.array([[float(x) for x in v.split()] for v in re.findall(r'facet normal\s+([^\n]+)',text)])
            cross=np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0])
            self.assertTrue(np.all(np.einsum('ij,ij->i',cross,normals)>0))

    def test_zero_load_does_not_fabricate_equilibrium(self):
        original=mesh();original.position*=1.1
        result=inflate(original,node_force=np.zeros_like(original.position),max_steps=3,drift_tolerance=1e-6)
        self.assertFalse(result.converged)
        self.assertGreater(result.residual,1.)

    def test_returned_tensions_and_direct_anchor_load_are_same_state(self):
        original=mesh();load=np.zeros_like(original.position);load[:-2,2]=10
        load[original.confluence[0]]=[2,3,4]
        result=inflate(original,node_force=load,max_steps=1)
        tension,direction=_rod_tensions(original,original.position)
        np.testing.assert_allclose(result.tension,tension)
        expected=-(tension[original.line_rods,None]*direction[original.line_rods]).sum(0)+load[original.confluence].sum(0)
        np.testing.assert_allclose(result.confluence_load,expected)

    def test_constraint_single_rod_has_correct_analytical_equilibrium(self):
        axis=np.array([.5,0,np.sqrt(3)/2])
        length=1+100*(np.sqrt(3)/2)/1000
        toy=SimpleNamespace(position=np.array([axis*length,[0.,0.,0.]]),mass=np.array([1.,np.inf]),
            rings=0,gores=1,confluence=np.array([1]))
        constraint=MeanAxisConstraint(toy,30)
        force=np.array([[0.,0.,100.],[0.,0.,0.]])
        force[0]-=1000*(length-1)*axis
        projected,_=constraint.project_force(force)
        np.testing.assert_allclose(projected,0,atol=1e-10)

    def test_constraint_preserves_axis_anchor_and_zero_virtual_work(self):
        original=mesh();constraint=MeanAxisConstraint(original,30)
        position=constraint.align_initial(original.position);velocity=np.zeros_like(position)
        rng=np.random.default_rng(2)
        force=rng.normal(size=position.shape)
        projected,reaction=constraint.project_force(force)
        acceleration=projected[constraint.ids]/original.mass[constraint.ids,None]
        np.testing.assert_allclose(constraint.transverse(acceleration.mean(0)),0,atol=1e-10)
        variation=rng.normal(size=position.shape);variation[constraint.ids]-=constraint.transverse(variation[constraint.ids].mean(0))
        self.assertAlmostEqual(float(np.sum(reaction*variation)),0.,places=10)
        anchor=position[original.confluence[0]].copy()
        for _ in range(5):
            velocity[constraint.ids]+=.001*acceleration
            position+=velocity*.001
            constraint.project_state(position,velocity)
        mean=position[constraint.ids].mean(0)-anchor
        self.assertAlmostEqual(np.degrees(np.arctan2(mean[0],mean[2])),30.,places=10)
        np.testing.assert_allclose(position[original.confluence[0]],anchor)

    def test_inflate_inner_keeps_mean_axis_while_deforming(self):
        original=mesh();force=np.zeros_like(original.position);force[:-2,2]=2.
        initial=original.position.copy()
        for _ in range(3):
            inflate(original,node_force=force,max_steps=3,attitude_deg=30)
            mean=original.position[:24].mean(0)-original.position[24]
            self.assertAlmostEqual(np.degrees(np.arctan2(mean[0],mean[2])),30.,places=10)
        self.assertGreater(np.linalg.norm(original.position-initial),.01)

    def test_resume_refuses_legacy_and_changed_physics(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);shape=root/'shape.npy';np.save(shape,np.zeros((1,3)))
            history=[{'pass':1}]
            with self.assertRaisesRegex(ValueError,'Legacy'):
                run_guard.begin_run(root,'case',{'V':30},history,shape,sources={})
            shape.unlink()
            run_guard.begin_run(root,'case',{'V':30},[],shape,sources={})
            np.save(shape,np.zeros((1,3)))
            run_guard.begin_run(root,'case',{'V':30},history,shape,sources={})
            with self.assertRaisesRegex(ValueError,'Incompatible'):
                run_guard.begin_run(root,'case',{'V':20},history,shape,sources={})

    def test_status_writer_removes_opposite_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            run_guard.mark_status(root,'case','НЕСОШЁЛСЯ',{'pass':6})
            run_guard.mark_status(root,'case','done',{'pass':6})
            self.assertFalse((root/'case.НЕСОШЁЛСЯ').exists())
            self.assertTrue((root/'case.done').is_file())

    def test_failed_continuation_revokes_completed_status(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);root=work/'out';root.mkdir()
            run_guard.atomic_json(root/'case.json',[{'pass':6}])
            run_guard.mark_status(root,'case','done',{'pass':6})
            run_guard.record_failure(root,'case',ValueError('failed pass 7'))
            self.assertFalse((root/'case.done').exists())
            self.assertEqual(nos6_results.saved_status(work,6,'case'),'not_converged')

    def test_confluence_load_includes_direct_force(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);frames=work/'out/кадры';frames.mkdir(parents=True)
            diameter=np.sqrt(24/np.pi)
            original=closed_circular(diameter,diameter,gores=8,rings=2,line_stride=2,shear_ratio=.02)
            np.save(frames/'case_проход01.npy',original.position)
            row=dict(pass_number=1,lines=diameter,gores=8,rings=2,stride=2,shear=.02,
                     cord_width_m=.0032,cord_anchor_force_N=[3,4,0])
            row['pass']=row.pop('pass_number')
            with patch.object(cord_study,'plot_confluence_loads'):
                info=cord_study.confluence_loads(work,'case',[row],work/'out')
            result=np.loadtxt(work/'out/confluence_loads.csv',delimiter=',',skiprows=1,encoding='utf-8-sig')
            self.assertEqual(info['missing_direct_passes'],[])
            np.testing.assert_allclose(result[1:4],result[6:9]+[3,4,0])
            self.assertAlmostEqual(result[4],np.linalg.norm(result[1:4]))
            row.pop('cord_anchor_force_N')
            with patch.object(cord_study,'plot_confluence_loads'):
                info=cord_study.confluence_loads(work,'case',[row],work/'out')
            self.assertEqual(info['missing_direct_passes'],[1])
            result=np.loadtxt(work/'out/confluence_loads.csv',delimiter=',',skiprows=1,encoding='utf-8-sig')
            self.assertTrue(np.isnan(result[4]))

    def fixture(self,root):
        case,archive=root/'case',root/'archive'
        for folder in [case/'constant/polyMesh',archive/'polyMesh',archive/'300',case/'300',case/'postProcessing/forces/0']:
            folder.mkdir(parents=True)
        for name in ['points','faces','owner','neighbour','boundary']:
            for folder in [case/'constant/polyMesh',archive/'polyMesh']: (folder/name).write_text('test mesh')
        for name in ['U','p','phi','nut']:
            for folder in [case/'300',archive/'300']: (folder/name).write_text('test field')
        (case/'log.simpleFoam').write_text('End\n')
        for filename in ['force.dat','moment.dat']:
            (case/'postProcessing/forces/0'/filename).write_text('300 (1 0 0) (1 0 0) (0 0 0)\n')
        return case,archive

    def test_cfd_guard_rejects_incomplete_archive_and_growth(self):
        with tempfile.TemporaryDirectory() as temp:
            case,archive=self.fixture(Path(temp))
            result=run_guard.check_cfd(case,archive,'300',1.,1.,1.)
            self.assertEqual(result['CFD_convergence'],'unconfirmed')
            path=case/'postProcessing/forces/0/moment.dat'
            path.write_text('300 (1e42 0 0) (1e42 0 0) (0 0 0)\n')
            with self.assertRaisesRegex(ValueError,'growth'):
                run_guard.check_cfd(case,archive,'300',1.,1.,1.)
            path.write_text('300 (1 0 0) (1 0 0) (0 0 0)\n')
            (archive/'300/U').write_text('wrong archive')
            with self.assertRaisesRegex(ValueError,'Archive fields'):
                run_guard.check_cfd(case,archive,'300',1.,1.,1.)

    def test_unpaired_face_does_not_receive_nearest_pressure(self):
        def faces(path):
            if Path(path).name=='master.vtp':
                return np.array([[0,0,1],[100,0,1]]),np.ones(2),np.array([[0,0,1],[0,0,1]]),np.array([1.,2.])
            return np.array([[0,0,1]]),np.ones(1),np.array([[0,0,-1]]),np.zeros(1)
        with patch.object(vtp,'faces_of',faces):
            with self.assertRaisesRegex(ValueError,'Unpaired'):
                vtp.delta_cp(Path('unused'),kin=1.)

    def test_method_compare_matches_equal_passes_and_initial_shape(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for mode,n in [('legacy',2),('inner',1)]:
                work=root/mode/'a30'
                archive=work/'fields/nos6_held/pass01'
                archive.mkdir(parents=True)
                np.save(archive/'input_shape.npy',np.zeros((4,3)))
                run_guard.atomic_json(work/'out/nos6_held.json',[
                    dict(pass_number=i,hold=30,cx=.5,cy=.2,mz=-.1,**{'pass':i}) for i in range(1,n+1)])
            with patch.object(method_compare,'polar'):
                rows=method_compare.compare(root)
            self.assertEqual(len(rows),1)
            self.assertTrue(rows[0]['same_initial_shape'])
            self.assertTrue(rows[0]['same_first_coefficients'])
            self.assertEqual(rows[0]['pass_number'],1)


if __name__=='__main__': unittest.main()
