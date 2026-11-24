"""Explicit material-view adapter for existing plot and sampling algorithms.

CFD files and saved states remain in Nosarev global coordinates. Only in-memory
views are rotated; plotted ticks are global y (horizontal) and x (vertical).
Seeds sent back to OpenFOAM are transformed to the actual global coordinates.
"""
from contextlib import contextmanager
from pathlib import Path
import re

import numpy as np
import vtp
from nosarev_frame import to_native,rotate_sampling_points


@contextmanager
def material_view(*,render=False,sampling=False):
    load_original=np.load
    arrays_original=vtp.arrays
    write_original=Path.write_text
    def load(path,*args,**kwargs):
        result=load_original(path,*args,**kwargs)
        if str(path).endswith('.npy') and isinstance(result,np.ndarray) and result.ndim==2 and result.shape[1]==3:
            return to_native(result)
        return result
    def arrays(path):
        result=dict(arrays_original(path))
        for key in ['Points','U']:
            if key in result:
                result[key]=to_native(np.asarray(result[key]).reshape(-1,3))
        return result
    def write(path,text,*args,**kwargs):
        if sampling and path.name=='controlDict' and 'seedSampleSet' in text:
            text=rotate_sampling_points(text)
        return write_original(path,text,*args,**kwargs)
    np.load=load;vtp.arrays=arrays;Path.write_text=write
    save_original=None
    if render:
        from matplotlib.figure import Figure
        from matplotlib.ticker import FuncFormatter
        save_original=Figure.savefig
        def negate(value,position):
            return format(0. if abs(value)<1e-12 else -value,'g')
        def save(fig,*args,**kwargs):
            for ax in fig.axes:
                if getattr(ax,'_nosarev_global_labels',False): continue
                if hasattr(ax,'get_zlabel'):
                    # The camera sees native coordinates (-y,z,-x).
                    if ax.get_xlabel().startswith('x'):
                        ax.set_xlabel(re.sub(r'^x','y',ax.get_xlabel()))
                        ax.xaxis.set_major_formatter(FuncFormatter(negate))
                    if ax.get_ylabel().startswith('y'):
                        ax.set_ylabel(re.sub(r'^y','z',ax.get_ylabel()))
                    if ax.get_zlabel().startswith('z'):
                        ax.set_zlabel(re.sub(r'^z','x',ax.get_zlabel()))
                        ax.zaxis.set_major_formatter(FuncFormatter(negate))
                elif ax.get_xlabel().startswith('x,') and ax.get_ylabel().startswith('z,'):
                    ax.set_xlabel(ax.get_xlabel().replace('x,','y,',1))
                    ax.set_ylabel(ax.get_ylabel().replace('z,','x,',1))
                    ax.xaxis.set_major_formatter(FuncFormatter(negate))
                    ax.yaxis.set_major_formatter(FuncFormatter(negate))
                ax.set_title(ax.get_title().replace('Uz < 0','Ux > 0'))
                ax.set_ylabel(ax.get_ylabel().replace('ωy','ωz'))
                ax._nosarev_global_labels=True
            return save_original(fig,*args,**kwargs)
        Figure.savefig=save
    try: yield
    finally:
        np.load=load_original;vtp.arrays=arrays_original;Path.write_text=write_original
        if save_original is not None:
            Figure.savefig=save_original

