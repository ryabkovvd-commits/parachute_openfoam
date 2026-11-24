"""Compare equal-pass legacy/inner attitude runs without conflating convergence."""
import argparse
import json
from pathlib import Path

import numpy as np

from nos6_results import csv_records, polar, read_records, saved_status, summary_lock, write_json


def compare(root):
    root=Path(root)
    output=root/'comparison'
    with summary_lock(output):
        rows=[]
        available={}
        for mode in ['legacy','inner']:
            available[mode]={p.name:p for p in (root/mode).glob('a*')
                             if (p/'out/nos6_held.json').is_file()}
        matched=sorted(set(available['legacy']) & set(available['inner']))
        for angle in matched:
            left,right=(available[m][angle] for m in ['legacy','inner'])
            a,b=read_records(left),read_records(right)
            initial=[p/'fields/nos6_held/pass01/input_shape.npy' for p in [left,right]]
            same_shape=all(p.is_file() for p in initial) and np.array_equal(*[np.load(p,allow_pickle=False) for p in initial])
            same_first=all(np.isclose(a[0][k],b[0][k],rtol=1e-6,atol=1e-9) for k in ['cx','cy','mz'])
            right_by_pass={int(r['pass']):r for r in b}
            for x in a:
                y=right_by_pass.get(int(x['pass']))
                if y is None: continue
                row=dict(angle=float(x['hold']),pass_number=int(x['pass']),
                         same_initial_shape=bool(same_shape),same_first_coefficients=bool(same_first),
                         delta_mz=float(y['mz']-x['mz']))
                for mode,work,r in [('legacy',left,x),('inner',right,y)]:
                    row.update({mode+'_'+k:r.get(k) for k in ['cx','cy','mz','ratio_seen','ratio','gap','inner_residual']})
                    row[mode+'_status']=saved_status(work,r['pass'])
                rows.append(row)
        csv_records(output/'method_comparison.csv',rows,
                    ['angle','pass_number','same_initial_shape','same_first_coefficients','delta_mz'])
        write_json(output/'comparison_status.json',dict(available={m:sorted(v) for m,v in available.items()},
                    matched_angles=matched,matched_passes=len(rows),
                    note='Compare equal passes and both input/output shapes. Outer criteria do not validate CFD or grid convergence.'))
    if matched:
        polar(root/'summary',[root])
    print(output/'method_comparison.csv')
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    compare(parser.parse_args().root)
