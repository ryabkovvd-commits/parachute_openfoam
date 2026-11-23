"""Optional native OpenFOAM smoke test: prepare DIR, run blockMesh/exporter, check DIR.

No CFD solver is used. The deliberately invalid saved BC must never execute.
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'coupled'))
import momentum_balance as audit


def header(cls,name,fmt='ascii'):
    return f'FoamFile {{ version 2.0; format {fmt}; class {cls}; object {name}; arch "LSB;label=32;scalar=64"; }}\n'


def prepare(root):
    for name,tangent,nx in [('uniform',0.,1),('tangential',2.,1),('pressure_limiter',0.,2)]:
        case=root/name
        if case.exists(): raise FileExistsError(case)
        (case/'system').mkdir(parents=True)
        (case/'constant').mkdir()
        (case/'2000').mkdir()
        block='''
scale 1;
vertices ((0 0 0)(1 0 0)(1 1 0)(0 1 0)(0 0 1)(1 0 1)(1 1 1)(0 1 1));
blocks (hex (0 1 2 3 4 5 6 7) (1 1 1) simpleGrading (1 1 1));
edges ();
boundary (
canopy {type patch; faces ((4 5 6 7));}
canopy_slave {type patch; faces ((0 3 2 1));}
farfield {type patch; faces ((0 4 7 3)(1 2 6 5)(0 1 5 4)(3 7 6 2));}
);
mergePatchPairs ();
'''
        (case/'system/blockMeshDict').write_text(header('dictionary','blockMeshDict')+block.replace('(1 1 1) simpleGrading',f'({nx} 1 1) simpleGrading'),encoding='ascii')
        (case/'system/controlDict').write_text(header('dictionary','controlDict')+
            'application nos6MomentumAudit; startFrom startTime; startTime 2000; endTime 2000; '
            'stopAt endTime; deltaT 1; writeControl timeStep; writeInterval 1; functions {}\n')
        (case/'system/fvSchemes').write_text(header('dictionary','fvSchemes')+
            'ddtSchemes {default steadyState;} gradSchemes {default '+('cellLimited Gauss linear 1;' if name=='pressure_limiter' else 'Gauss linear;')+'} '
            'divSchemes {default Gauss linear;} laplacianSchemes {default Gauss linear corrected;} '
            'interpolationSchemes {default linear;} snGradSchemes {default corrected;}')
        (case/'system/fvSolution').write_text(header('dictionary','fvSolution')+'solvers {}')
        (case/'constant/cords').write_text('2\n((0.3 0.4 0.2)(0.3 0.4 0.6))\n')
        # Binary internalField verifies that parsing is not an ASCII-only shortcut.
        vz=0. if name=='pressure_limiter' else 3.
        data=(header('volVectorField','U','binary')+'dimensions [0 1 -1 0 0 0 0];\n'
              f'internalField nonuniform List<vector>\n{nx}\n(').encode('ascii')+struct.pack('<'+str(3*nx)+'d',*([0.,0.,vz]*nx))+b');\n'
        data+=(f'''boundaryField {{
canopy {{ type codedFixedValue; name mustNeverExecute; value uniform ({tangent} 0 {vz});
code #{{
#error SAVED_BOUNDARY_CONDITION_MUST_NOT_BE_EXECUTED
#}};
}}
canopy_slave {{type fixedValue; value uniform (0 0 {vz});}}
farfield {{type fixedValue; value uniform (0 0 {vz});}}
}}
''').encode('ascii')
        (case/'2000/U').write_bytes(data)
        for field,dims,value in [('p','0 2 -2 0 0 0 0',5.123456789012345),('nut','0 2 -1 0 0 0 0',1e-7)]:
            values=[0.,10.] if field=='p' and name=='pressure_limiter' else [value]*nx
            boundary_value=11. if field=='p' and name=='pressure_limiter' else value
            data=(header('volScalarField',field,'binary')+f'dimensions [{dims}]; '
                  f'internalField nonuniform List<scalar>\n{nx}\n(').encode('ascii')
            data+=struct.pack('<'+str(nx)+'d',*values)+b');\n'
            data+=('boundaryField {canopy {type zeroGradient;} canopy_slave {type zeroGradient;} '
                   f'farfield {{type fixedValue; value uniform {boundary_value};}}}}').encode('ascii')
            (case/'2000'/field).write_bytes(data)
        (case/'2000/phi').write_text(header('surfaceScalarField','phi')+
            'dimensions [0 3 -1 0 0 0 0]; internalField uniform 0; boundaryField {'
            f'canopy {{type calculated; value uniform {vz};}} canopy_slave {{type calculated; value uniform {-vz};}}'
            'farfield {type calculated; value uniform 0;}}')
        hashes={str(p.relative_to(case)):audit.sha(p) for p in (case/'2000').iterdir()}
        (case/'input_hashes.json').write_text(json.dumps(hashes))


def check(root):
    for name,tangent in [('uniform',0.),('tangential',2.),('pressure_limiter',0.)]:
        case=root/name
        expected=json.loads((case/'input_hashes.json').read_text())
        assert all(audit.sha(case/p)==v for p,v in expected.items())
        assert not (case/'dynamicCode').exists(), 'Saved coded BC was instantiated'
        rows=audit.read_csv(case/'export/boundary.csv')
        if name!='pressure_limiter':
            np.testing.assert_allclose([float(r['p_kin']) for r in rows],5.123456789012345,rtol=1e-15)
        groups=audit.integrate_boundary(rows,1.,[0,0,0])
        np.testing.assert_allclose(groups['interface']['convection_force_N'],[3*tangent,0,0],atol=1e-12)
        np.testing.assert_allclose(groups['interface']['convection_moment_Nm'],[0,3*tangent,-1.5*tangent],atol=1e-12)
        np.testing.assert_allclose(groups['all']['pressure_force_N'],0,atol=1e-12)
        np.testing.assert_allclose(groups['all']['pressure_moment_Nm'],0,atol=1e-12)
        assert groups['all']['volume_flux_m3_s']==0.
        assert groups['all']['phi_minus_U_dot_Sf_L1_m3_s']==0.
        if not tangent:
            np.testing.assert_allclose(groups['all']['viscous_force_N'],0,atol=1e-12)
            np.testing.assert_allclose(groups['all']['convection_moment_Nm'],0,atol=1e-12)
        operators=audit.read_operators(case/'export/operators.csv',1.)
        if name=='pressure_limiter':
            np.testing.assert_allclose(operators['pressure_gauss']['force_N'],0,atol=1e-12)
            np.testing.assert_allclose(operators['pressure_limited']['force_N'],[2,0,0],atol=1e-12)
            np.testing.assert_allclose(operators['pressure_limited']['moment_Nm'],[0,1,-1],atol=1e-12)
        print(name+': binary fields, frozen BC, fluxes, force/moment signs and unchanged inputs: PASS')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','check'])
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    (prepare if args.mode=='prepare' else check)(args.root.resolve())
