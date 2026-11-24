"""Actual global Nosarev speed axes for geometry, CFD, loads and saved shapes.

Native to speed coordinates: (x,y,z) = (-Znative,-Xnative,+Ynative).
This proper rotation has determinant +1. The air velocity is along -x,
drag is -Fx, signed lift is Fy, and pitching moment is Mz about the coosh.
The unchanged structural kernel is evaluated in its material working frame;
all input/output states of the coupled driver stay in the global speed frame.
"""
from dataclasses import fields,replace
import hashlib
from pathlib import Path
import re

import numpy as np

from nos6_geometry import closed_circular
from parafoil.inflation.mesh import CanopyMesh
from parafoil.inflation.relax import inflate

FRAME_NAME='nosarev_speed'
ROTATION=np.array([[0.,0.,-1.],[-1.,0.,0.],[0.,1.,0.]])
FLOW=np.array([-1.,0.,0.])
LIFT=np.array([0.,1.,0.])
PITCH=np.array([0.,0.,1.])
TILT=np.array([0.,-1.,0.])


def to_speed(values):
    return np.asarray(values,dtype=float)@ROTATION.T


def to_native(values):
    return np.asarray(values,dtype=float)@ROTATION


def _mesh_values(mesh):
    return {f.name:getattr(mesh,f.name) for f in fields(CanopyMesh)}


class SpeedMesh(CanopyMesh):
    def projected_diameter(self):
        skirt=to_native(self.position[self.skirt])
        centre=skirt.mean(0)
        return 2*float(np.linalg.norm((skirt-centre)[:,:2],axis=1).mean())


def speed_circular(*args,**kwargs):
    mesh=closed_circular(*args,**kwargs)
    return SpeedMesh(**dict(_mesh_values(mesh),position=to_speed(mesh.position),flat=to_speed(mesh.flat)))


def speed_inflate(mesh,*args,**kwargs):
    """Use the same mechanical equations, with proper vector transformations."""
    local=CanopyMesh(**dict(_mesh_values(mesh),position=to_native(mesh.position),flat=to_native(mesh.flat)))
    if kwargs.get('node_force') is not None:
        kwargs['node_force']=to_native(kwargs['node_force'])
    result=inflate(local,*args,**kwargs)
    mesh.position=to_speed(local.position)
    rotate=lambda value:None if value is None else to_speed(value)
    return replace(result,mesh=mesh,confluence_load=rotate(result.confluence_load),
                   pressure_resultant=rotate(result.pressure_resultant),
                   other_applied_resultant=rotate(result.other_applied_resultant),
                   constraint_force=rotate(result.constraint_force))


_VECTOR=r'\(([^()]*)\)'


def _values(text):
    values=np.fromstring(text,sep=' ')
    if values.shape!=(3,): raise ValueError('Expected a three-component point/vector')
    return values


def _format(values):
    return '('+' '.join(format(0. if abs(v)<1e-15 else v,'.15g') for v in values)+')'


def _vectors(text):
    return re.sub(_VECTOR,lambda m:_format(to_speed(_values(m[1]))),text)


def rotate_dictionary(name,text):
    """Rotate only physical points/vectors; keep hex counts and grading local."""
    if name=='system/blockMeshDict':
        text,count=re.subn(r'(\bvertices\s*\()(.*?)(\)\s*;)',
                          lambda m:m[1]+_vectors(m[2])+m[3],text,flags=re.S)
        if count!=1: raise ValueError('Expected one vertices section')
    elif name=='system/snappyHexMeshDict':
        def box(m):
            lo,hi=_values(m[1]),_values(m[2])
            corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
            points=to_speed(corners)
            return 'min '+_format(points.min(0))+'; max '+_format(points.max(0))+';'
        text=re.sub(r'\bmin\s*\(([^()]*)\)\s*;\s*max\s*\(([^()]*)\)\s*;',box,text)
        text=re.sub(r'(\b(?:centre|point1|point2|locationInMesh)\s*)'+_VECTOR,
                    lambda m:m[1]+_format(to_speed(_values(m[2]))),text)
    elif name=='system/controlDict':
        text=re.sub(r'(\b(?:CofR|point|normal|start|end)\s*)'+_VECTOR,
                    lambda m:m[1]+_format(to_speed(_values(m[2]))),text)
    return text


def rotate_sampling_points(text):
    return re.sub(r'(\bpoints\s*\()(.*?)(\)\s*;)',
                  lambda m:m[1]+_vectors(m[2])+m[3],text,flags=re.S)


def install_speed_frame(g):
    """Install the isolated new profile after the existing functions are defined."""
    if g['AZIMUTH']!=0 or g['ALPHA']!=0:
        raise ValueError('Nosarev profile requires a fixed speed frame and the x-y attack plane')
    write_native=g['write'];attitude_native=g['set_attitude'];ball_native=g['refine_ball']
    hashes_native=g['source_hashes']
    def write(name,text):
        write_native(name,rotate_dictionary(name,text))
    def attitude(position,degrees,confluence):
        return to_speed(attitude_native(to_native(position),degrees,confluence))
    def tilt(position):
        axis=position[:g['CANOPY']].mean(0)-position[g['CANOPY']]
        norm=float(np.linalg.norm(axis))
        if norm<=1e-12 or not np.isfinite(norm): raise ValueError('Undefined canopy axis')
        if g['HOLD'] is not None:
            return float(np.degrees(np.arctan2(axis@TILT,axis@FLOW)))
        return float(np.degrees(np.arccos(np.clip(axis@FLOW/norm,-1,1))))
    def seed(position,frame):
        if frame not in {'native','nosarev'}: raise ValueError('--seed-frame must be native or nosarev')
        value=to_speed(position) if frame=='native' else np.asarray(position)
        axis=value[:g['CANOPY']].mean(0)-value[g['CANOPY']]
        if axis@FLOW<=1e-10:
            raise ValueError('Seed is not in the -X hemisphere; use --seed-frame=native for old data')
        return value
    def hashes(folder,patches):
        values=hashes_native(folder,patches)
        values['nosarev_frame.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return values
    g.update(write=write,closed_circular=speed_circular,inflate=speed_inflate,
             set_attitude=attitude,tilt_of=tilt,
             refine_ball=lambda pos:ball_native(to_native(pos)),
             UP=FLOW.copy(),ONSET=FLOW.copy(),TILT=TILT.copy(),SIDE=LIFT.copy(),PIVOT=PITCH.copy(),
             speed_seed=seed,source_hashes=hashes)

