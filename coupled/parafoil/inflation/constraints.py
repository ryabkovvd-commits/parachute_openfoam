"""Explicit experimental constraint on the mean canopy-node axis.

This is a kinematic comparison model, not the sliding-pole experimental stand.
The two linear constraints set the mean transverse coordinates to zero.
Mass-metric projection gives their Lagrange reaction and projects velocities.
"""
import numpy as np


class MeanAxisConstraint:
    def __init__(self,mesh,degrees):
        if not np.isfinite(degrees) or abs(degrees)>=90:
            raise ValueError('Inner attitude control requires a finite angle strictly between -90 and 90 degrees')
        theta=np.radians(degrees)
        self.axis=np.array([np.sin(theta),0.,np.cos(theta)])
        self.ids=np.arange((mesh.rings+1)*mesh.gores)
        self.anchor=mesh.position[mesh.confluence[0]].copy()
        self.weights=1/mesh.mass[self.ids]
        if not np.isfinite(self.weights).all() or np.any(self.weights<=0):
            raise ValueError('Mean-axis constraint requires finite positive canopy masses')
        self.sum_weights=float(self.weights.sum())

    def transverse(self,vector):
        return vector-(vector@self.axis)*self.axis

    def align_initial(self,position):
        rel=position-self.anchor
        current=rel[self.ids].mean(0)
        norm=np.linalg.norm(current)
        if not np.isfinite(norm) or norm<=1e-12:
            raise ValueError('Undefined initial constraint axis')
        current/=norm
        cross=np.cross(current,self.axis);sine=np.linalg.norm(cross)
        cosine=np.clip(current@self.axis,-1,1)
        if sine<1e-12:
            if cosine>0: return position.copy()
            axis=np.cross(current,np.eye(3)[np.argmin(np.abs(current))]);axis/=np.linalg.norm(axis)
            rotation=2*np.outer(axis,axis)-np.eye(3)
        else:
            x,y,z=cross
            skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
            rotation=np.eye(3)+skew+skew@skew*((1-cosine)/sine**2)
        return self.anchor+rel@rotation.T

    def project_force(self,force):
        reaction=np.zeros_like(force)
        transverse=self.transverse((force[self.ids]*self.weights[:,None]).sum(0))
        reaction[self.ids]=-transverse/self.sum_weights
        return force+reaction,reaction

    def project_state(self,position,velocity):
        mean=(position[self.ids]-self.anchor).mean(0)
        correction=self.transverse(mean)
        if mean@self.axis<=0:
            raise ValueError('Canopy axis crossed the confluence; invalid constrained state')
        factors=self.weights*len(self.ids)/self.sum_weights
        position[self.ids]-=factors[:,None]*correction
        mean_velocity=velocity[self.ids].mean(0)
        velocity[self.ids]-=factors[:,None]*self.transverse(mean_velocity)
        return float(np.linalg.norm(correction))
