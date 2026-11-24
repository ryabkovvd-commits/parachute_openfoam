"""Use the established figure/GIF pipeline on actual Nosarev-axis CFD data."""
from pathlib import Path
import runpy
from nosarev_view import material_view

if __name__=='__main__':
    with material_view(render=True,sampling=True):
        runpy.run_path(str(Path(__file__).with_name('plots.py')),run_name='__main__',
                       init_globals={'_NOSAREV_PLOT_ADAPTER':True})

