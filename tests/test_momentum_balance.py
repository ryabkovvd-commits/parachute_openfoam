import contextlib
import csv
import io
import json
import sys
import tempfile
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'coupled'))
import momentum_balance as audit


def face(name,centre,area,u,phi,p=0.,tau=(0.,0.,0.)):
    return dict(zip(['patch','face','cx','cy','cz','sx','sy','sz','p_kin','Ux','Uy','Uz','phi','tauSfx','tauSfy','tauSfz'],
                    [name,0,*centre,*area,p,*u,phi,*tau]))


class MomentumTests(unittest.TestCase):
    def test_tangential_impulse_and_moment_sign(self):
        rows=[face('canopy',(.4,0,1),(0,0,1),(2,0,3),3,p=2),
              face('canopy_slave',(.4,0,1),(0,0,-1),(0,0,3),-3,p=1),
              face('farfield',(0,0,0),(1,0,0),(0,0,0),0)]
        g=audit.integrate_boundary(rows,2.,[0,0,0])
        np.testing.assert_allclose(g['interface']['pressure_force_N'],[0,0,2])
        np.testing.assert_allclose(g['interface']['pressure_moment_Nm'],[0,-.8,0])
        np.testing.assert_allclose(g['interface']['convection_force_N'],[12,0,0])
        np.testing.assert_allclose(g['interface']['convection_moment_Nm'],[0,12,0])
        np.testing.assert_allclose(g['interface']['convection_normal_force_N'],0)
        self.assertEqual(g['interface']['volume_flux_m3_s'],0.)
        self.assertEqual(g['all']['phi_minus_U_dot_Sf_L1_m3_s'],0.)
        moved=audit.integrate_boundary(rows,2.,[0,0,1])
        np.testing.assert_allclose(moved['interface']['convection_moment_Nm'],0)

    def test_closed_balance_and_viscous_sign(self):
        rows=[face('canopy',(0,0,1),(0,0,1),(0,0,0),0,p=3,tau=(2,0,0)),
              face('canopy_slave',(0,0,1),(0,0,-1),(0,0,0),0,p=1),
              face('farfield',(0,0,1),(0,0,-1),(0,0,0),0,p=2,tau=(-2,0,0))]
        groups=audit.integrate_boundary(rows,1.,[0,0,0])
        np.testing.assert_allclose(groups['interface']['viscous_force_N'],[-2,0,0])
        cords=dict(force_N=[0,0,0],moment_at_cells_Nm=[0,0,0])
        result=audit.form_balance(groups,cords,1.,1.,1.)
        for quantity in ['force_N','moment_Nm']:
            np.testing.assert_allclose(result[quantity]['residual'],0,atol=1e-14)

    def test_unequal_boundary_flux_is_detected(self):
        rows=[face('canopy',(0,0,1),(0,0,1),(0,0,3),3),
              face('canopy_slave',(0,0,1),(0,0,-1),(0,0,3),-2),
              face('farfield',(0,0,0),(1,0,0),(0,0,0),0)]
        groups=audit.integrate_boundary(rows,1.,[0,0,0])
        self.assertEqual(groups['all']['volume_flux_m3_s'],1.)
        self.assertEqual(groups['all']['phi_minus_U_dot_Sf_L1_m3_s'],1.)

    def test_cord_source_cell_and_structural_point_moments_differ(self):
        mapping=[dict(sample=0,cell=1,cx=0,cy=0,cz=2,x=0,y=0,z=1,rx_kin=1,ry_kin=0,rz_kin=0)]
        result=audit.cord_terms(mapping,np.array([[0,1,0,0.]]),2.,[0,0,0])
        np.testing.assert_allclose(result['force_N'],[2,0,0])
        np.testing.assert_allclose(result['moment_at_cells_Nm'],[0,4,0])
        np.testing.assert_allclose(result['moment_at_samples_Nm'],[0,2,0])
        self.assertEqual(result['force_recomputed_difference_relative'],0.)
        with self.assertRaises(ValueError):
            audit.cord_terms(mapping,np.array([[1,1,0,0.]]),2.,[0,0,0])

    def test_missing_phi_cannot_be_silently_reconstructed(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp)
            (work/'case/system').mkdir(parents=True)
            (work/'case/system/controlDict').write_text('application simpleFoam;')
            folder=work/'case/2000'
            folder.mkdir()
            for field in ['U','p','nut']: (folder/field).write_text('fixture')
            with self.assertRaisesRegex(FileNotFoundError,'No reconstruction of phi'):
                audit.snapshot_sources(work,dict(fixed_shape=True,**{'pass':1,'iters':2000}))

    def test_force_history_parser_keeps_pressure_and_viscosity_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'force.dat'
            path.write_text('# header\n2000 (1 2 3) (4 5 6) (7 8 9)\n')
            r=audit.native_force_record(path,2000.)
            self.assertEqual(r['total'],[1,2,3])
            self.assertEqual(r['pressure'],[4,5,6])
            self.assertEqual(r['viscous'],[7,8,9])

    def test_failed_rerun_cannot_leave_a_stale_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            binary=root/'binary'
            binary.touch()
            summary=root/'momentum_balance.csv'
            summary.write_text('old success')
            old_report=root/'df/out/momentum_balance/report.json'
            old_report.parent.mkdir(parents=True)
            old_report.write_text('{"old":"success"}')
            with patch.object(audit,'run_case',side_effect=ValueError('invalid snapshot')),contextlib.redirect_stdout(io.StringIO()):
                status=audit.main([str(root),'--binary',str(binary)])
            self.assertEqual(status,1)
            self.assertFalse(summary.exists())
            self.assertFalse(old_report.exists())

    def test_wrapper_stages_inputs_and_produces_report_without_changing_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp)/'df'
            case=work/'case'
            for directory in ['system','constant/polyMesh','2000','postProcessing/cords/2000']:
                (case/directory).mkdir(parents=True)
            for n in audit.permeability.CORE: (case/'constant/polyMesh'/n).write_text('mesh fixture')
            for n in ['U','p','phi','nut']: (case/'2000'/n).write_text('stored field '+n)
            (case/'system/controlDict').write_text('application simpleFoam; functions { MUST_NOT_COPY; }')
            (case/'system/fvSchemes').write_text('original schemes')
            (case/'system/fvSolution').write_text('original solution settings')
            (case/'constant/turbulenceProperties').write_text('RAS { RASModel kOmegaSST; }')
            (case/'constant/cords').write_text('2 ((0 0 0)(0 0 1))')
            template=Path(audit.__file__).with_name('cordsFv.txt').read_text()
            for k,v in dict(CORD_SAMPLES=64,CORD_WIDTH=.0032,CORD_CD=1.2).items(): template=template.replace(k,str(v))
            (case/'system/fvOptions').write_text(template)
            np.savetxt(case/'postProcessing/cords/2000/0.dat',np.c_[np.arange(64),np.zeros((64,3))])
            row=dict(fixed_shape=True,**{'pass':1,'iters':2000,'rho':1.225,'V':30.,'D0':float(np.sqrt(24/np.pi)),
                     'S0':6.,'nu':1.46e-5,'cord_width_m':.0032,'cord_cd':1.2,'mz':0.,'cx':0.,'cy':0.,'cord_mz':0.})
            def exporter(command,**kwargs):
                stage=Path(kwargs['cwd'])
                self.assertIn('functions {}',(stage/'system/controlDict').read_text())
                self.assertEqual((stage/'2000/U').read_text(),'stored field U')
                output=Path(command[command.index('-audit-out')+1])
                rows=[face('canopy',(.4,0,1),(0,0,1),(0,0,3),3),
                      face('canopy_slave',(.4,0,1),(0,0,-1),(0,0,3),-3),
                      face('farfield',(0,0,0),(1,0,0),(0,0,0),0)]
                with (output/'boundary.csv').open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                (output/'volume.csv').write_text('quantity,x,y,z\ncontinuity_l1,0,0,0\ncontinuity_net,0,0,0\nbounded_force_kin,0,0,0\nbounded_moment_kin,0,0,0\n')
                with (output/'cord_mapping.csv').open('w',newline='') as stream:
                    writer=csv.writer(stream)
                    writer.writerow(['sample','cell','cx','cy','cz','x','y','z','rx_kin','ry_kin','rz_kin'])
                    writer.writerows([[i,0,0,0,.5,0,0,.5,0,0,0] for i in range(64)])
                return subprocess.CompletedProcess(command,0,'exported','')
            with patch.object(audit.permeability,'verify',return_value=({'name':'df'},None,row)),patch.object(audit.subprocess,'run',side_effect=exporter):
                result=audit.run_case(work,Path('/dummy/exporter'))
            self.assertEqual(result['mz_balance_residual'],0.)
            report=json.loads((work/'out/momentum_balance/report.json').read_text())
            self.assertTrue(report['inputs_unchanged'])
            self.assertIn('MUST_NOT_COPY',(case/'system/controlDict').read_text())


if __name__=='__main__': unittest.main()
