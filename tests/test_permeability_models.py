import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'coupled'))
import permeability_models as pm
import test_case_configuration as configuration


def cpp_test_source():
    """Compile the actual inversion block from fabricU, rather than a copied formula."""
    text=(ROOT/'coupled/fabricU.txt').read_text(encoding='utf-8')
    block=text[text.index('        scalar target = 0.0;'):text.index('        Un[i] +=')]
    return r'''
#include <cmath>
#include <initializer_list>
#include <cassert>
#include <cfenv>
using scalar=double;
namespace Foam { using std::sqrt; }
double mag(double x) { return std::abs(x); }
double sqr(double x) { return x*x; }
double sign(double x) { return (x>0)-(x<0); }
double invert(double jump, double Dc, double Cw) {
    double dp[]={jump}; int i=0;
'''+block+r'''
    return target;
}
int main() {
#ifdef __linux__
    feenableexcept(FE_DIVBYZERO|FE_INVALID|FE_OVERFLOW);
#endif
    for (double d: {0.,33.2,55.}) for (double c: {0.,28.6,70.}) {
        if (d+c==0) continue;
        for (double dp: {-1000.,-1.,-1e-12,0.,1e-12,1.,1000.}) {
            double u=invert(dp,d,c);
            assert(std::isfinite(u));
            assert(std::abs(d*u+c*std::abs(u)*u-dp)<1e-10*(1+std::abs(dp)));
            assert(u==-invert(-dp,d,c));
        }
    }
}
'''


class PermeabilityTests(unittest.TestCase):
    def test_common_reference_and_limits(self):
        rows=pm.models()
        reference=[float(pm.speed(49,r['D'],r['Cw'])) for r in rows]
        np.testing.assert_allclose(reference,reference[0],rtol=1e-14)
        self.assertGreater(pm.speed(500,rows[1]['D'],0),pm.speed(500,33.2,28.6))
        self.assertLess(pm.speed(500,0,rows[2]['Cw']),pm.speed(500,33.2,28.6))
        for r in rows:
            p=np.array([-900.,-1e-12,0.,1e-12,900.])
            u=pm.speed(p,r['D'],r['Cw'])
            np.testing.assert_allclose(1.225*(r['D']*u+r['Cw']*np.abs(u)*u),p,rtol=1e-14,atol=1e-20)
        with self.assertRaises(ValueError): pm.speed(49,0,0)

    def test_cpp_actual_boundary_inverse(self):
        compiler=shutil.which('g++')
        if not compiler: self.skipTest('g++ unavailable; actual boundary block also tested in Linux CI')
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'inverse.cpp'
            program=Path(temp)/'inverse'
            source.write_text(cpp_test_source(),encoding='utf-8')
            subprocess.run([compiler,'-std=c++11','-O2',str(source),'-o',str(program)],check=True,capture_output=True)
            subprocess.run([str(program)],check=True,capture_output=True)

    def test_generated_frozen_mesh_case_and_no_relaxation(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            for name in pm.CORE: (folder/name).write_text('fixture',encoding='ascii')
            seed=folder/'shape.npy'
            np.save(seed,np.zeros((2,3)))
            source=configuration.ConfigurationTests().generated(True,
                ['--fixed-shape','--mesh-from='+str(folder),'--fabric=0,73.6'],passes=1,seed=seed)
            self.assertIn('cp -r {quote(str(MESH_SOURCE))} constant/polyMesh',source)
            self.assertIn("log.checkMesh.fixed | head -1",source)
            start=source.index('    relaxed = FIXED_SHAPE\n')
            end=source.index('    last_residual = residual\n',start)+len('    last_residual = residual\n')
            original=np.arange(6,dtype=float).reshape(2,3)
            def forbidden(*args,**kwargs): raise AssertionError('Frozen geometry was updated')
            scope=dict(FIXED_SHAPE=True,mesh=SimpleNamespace(position=original.copy()),previous=original.copy(),
                       load=np.ones((2,3)),HOLD=30,HOLDER=False,CHUNKS=9,CHUNK=1,np=np,
                       sane=lambda mesh:True,inflate=forbidden,set_attitude=forbidden,aitken_factor=forbidden,
                       last_residual=None,omega=.5)
            exec(compile('for step in [1]:\n'+source[start:end],'frozen-block','exec'),scope)
            np.testing.assert_array_equal(scope['mesh'].position,original)
            self.assertEqual(scope['omega'],0.)
            self.assertTrue(scope['relaxed'])

    def test_prepare_and_verify_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source=root/'source'
            archive=source/'fields'/pm.NAME/'pass06'
            archive.mkdir(parents=True)
            (source/'out').mkdir()
            d=float(np.sqrt(24/np.pi))
            row=dict(pass_=6,gores=96,rings=12,stride=4,lines=d,D0=d,hold=30)
            row['pass']=row.pop('pass_')
            mesh=pm.diagnostics.make_mesh(row,d)
            shape=np.zeros_like(mesh.position)
            shape[:13*96]=[.5,0,np.sqrt(3)/2]
            np.save(archive/'input_shape.npy',shape)
            meshdir=archive/'polyMesh'
            meshdir.mkdir()
            for n in pm.CORE: (meshdir/n).write_text('mesh fixture '+n,encoding='ascii')
            (source/'out'/f'{pm.NAME}.json').write_text(json.dumps([row]),encoding='utf-8')
            setup=root/'setup'
            manifest=pm.prepare(source,setup)
            self.assertEqual(manifest['mesh_sha256'],pm.fingerprint(meshdir))
            with self.assertRaises(FileExistsError): pm.prepare(source,setup)
            work=root/'df'
            (work/'out').mkdir(parents=True)
            target=work/'fields'/pm.NAME/'pass01'
            target.mkdir(parents=True)
            shutil.copytree(setup/'polyMesh',work/'case/constant/polyMesh')
            np.save(target/'input_shape.npy',shape)
            np.save(work/'out'/f'{pm.NAME}_shape.npy',shape)
            config=dict(name='df',D=33.2,Cw=28.6,setup=str(setup))
            pm.write_json(work/'model.json',config)
            pm.write_json(work/'out'/f'{pm.NAME}.json',[dict(fixed_shape=True,fabric_coefficients='33.2,28.6')])
            pm.verify(work)
            (work/'case/constant/polyMesh/points').write_text('changed',encoding='ascii')
            with self.assertRaisesRegex(ValueError,'mesh differs'): pm.verify(work)

    def test_invalid_coefficients_rejected_before_solver(self):
        for value in ['0,0','-1,2','nan,1','1,2,3']:
            with self.subTest(value=value), self.assertRaises(SystemExit):
                configuration.ConfigurationTests().generated(True,['--fabric='+value])


if __name__=='__main__': unittest.main()
