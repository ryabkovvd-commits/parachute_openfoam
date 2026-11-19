"""Saved-field diagnostics: pressure loads, transfer conservation and wake cuts.

No solver is started. Global OpenFOAM axes: flow +Z, side force +X, pitching
moment +Y about the confluence. This module does not change CFD results.
"""
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection

import vtp
import pressure_table
from nos6_geometry import closed_circular
from parafoil.inflation import flat_circular


def save_csv(path, columns, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)
    return str(path)


def make_mesh(row, diameter, pole='closed', vent_fraction=None):
    options = dict(gores=int(row['gores']),rings=int(row['rings']),
                   line_stride=int(row['stride']),shear_ratio=row.get('shear',.02))
    if pole=='closed':
        return closed_circular(diameter,float(row['lines']),**options)
    return flat_circular(diameter,float(row['lines']),vent_fraction=vent_fraction or .05,**options)


def input_shape(work, name, row):
    number = int(row['pass'])
    exact = work/'fields'/name/f'pass{number:02d}'/'input_shape.npy'
    if exact.is_file():
        return exact
    frame = work/'out/кадры'/f'{name}_проход{number-1:02d}.npy'
    if number>1 and frame.is_file():
        return frame
    raise FileNotFoundError(f'No CFD input shape for pass {number}; final relaxed shape cannot replace it')


def pressure_folder(work, name, row):
    root = work/'fields'/name/f"pass{int(row['pass']):02d}"/'postProcessing/surfacePressure'
    folders = [d for d in pressure_table.numeric_dirs(root)
               if (d/'master.vtp').is_file() and (d/'slave.vtp').is_file()]
    if not folders:
        raise FileNotFoundError(f'No pressure pair in {root}')
    return folders[-1]


def surface(work, name, row, diameter, speed, rho, pole='closed', vent_fraction=None):
    folder = pressure_folder(work,name,row)
    arrays = vtp.arrays
    common = arrays(folder/'master.vtp').keys() & arrays(folder/'slave.vtp').keys()
    field = next((key for key in ('pMean','p') if key in common),None)
    if field is None:
        raise ValueError('No common pressure field')

    def reader(path):
        data = arrays(path)
        data['p'] = data[field]
        return data

    # Reuse the canonical face geometry and load transfer implementation.
    try:
        vtp.arrays = reader
        c,a,n,pm = vtp.faces_of(folder/'master.vtp')
        cs,as_,_,ps = vtp.faces_of(folder/'slave.vtp')
        forward,back = vtp.nearest(c,cs),vtp.nearest(cs,c)
        distance = np.linalg.norm(c-cs[forward],axis=1)
        tolerance = 1e-4 # Same geometric tolerance as fabricU's squared 1e-8 m2.
        paired = (distance<=tolerance) & (back[forward]==np.arange(len(c)))
        paired_slave = np.zeros(len(cs),dtype=bool)
        paired_slave[forward[paired]] = True
        kin, q = .5*speed**2, .5*rho*speed**2
        area0 = np.pi*diameter**2/4
        delta = (pm-ps[forward])/kin
        if not np.isfinite(delta).all() or a.sum()<=0:
            raise ValueError('Invalid pressure or zero surface area')
        shape = input_shape(work,name,row)
        mesh = make_mesh(row,diameter,pole,vent_fraction)
        mesh.position = np.load(shape,allow_pickle=False)
        if mesh.position.shape!=mesh.flat.shape or not np.isfinite(mesh.position).all():
            raise ValueError('CFD input shape does not match structural topology')
        anchor = mesh.position[mesh.confluence[0]]
        forces = (delta*a*q)[:,None]*n
        moments = np.cross(c-anchor,forces)
        load,_,_ = vtp.nodal_load(mesh,folder,q=q,kin=kin)
    finally:
        vtp.arrays = arrays
    f_face, f_node = forces.sum(0),load.sum(0)
    m_face,m_node = moments.sum(0),np.cross(mesh.position-anchor,load).sum(0)
    scale = q*area0*diameter
    coverage = float(a[paired].sum()/a.sum())
    report = dict(pass_number=int(row['pass']),time=float(folder.name),pressure_field=field,
                  input_shape=str(shape),pair_tolerance_m=tolerance,
                  paired_area_fraction=coverage,unpaired_master_faces=int((~paired).sum()),
                  unpaired_slave_faces=int((~paired_slave).sum()),
                  pair_distance_max_m=float(distance.max()),
                  face_force_N=f_face.tolist(),node_force_N=f_node.tolist(),
                  face_moment_Nm=m_face.tolist(),node_moment_Nm=m_node.tolist(),
                  paired_force_N=forces[paired].sum(0).tolist(),paired_moment_Nm=moments[paired].sum(0).tolist(),
                  force_transfer_relative=float(np.linalg.norm(f_face-f_node)/max(np.linalg.norm(f_face),1e-30)),
                  moment_transfer_error_mz=float((m_node[1]-m_face[1])/scale),
                  moment_transfer_error_vector_Nm=(m_node-m_face).tolist(),
                  axes=dict(drag='+Z',side='+X',pitch='+Y',origin=anchor.tolist()),
                  note='Raw-nearest integration reproduces the current solver transfer. Paired-only values exclude unmatched faces; they are not the complete CFD load. Pressure loads exclude cord drag and viscous surface traction.')
    for key,value,cord in [('cx',f_node[2]/(q*area0),'cord_cx'),
                           ('cy',f_node[0]/(q*area0),'cord_cy'),('mz',m_node[1]/scale,'cord_mz')]:
        report[key+'_pressure_recomputed'] = float(value)
        if key in row:
            report[key+'_pressure_from_history'] = float(row[key]-row.get(cord,0.))
            report[key+'_difference_from_history'] = float(value-row[key]+row.get(cord,0.))
    pics,out = work/'pics',work/'out'
    pics.mkdir(exist_ok=True); out.mkdir(exist_ok=True)
    number = int(row['pass'])
    meta = out/f'surface_diagnostics_pass{number:02d}.json'
    meta.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    table = save_csv(out/f'surface_unpaired_pass{number:02d}.csv',
                     ['side','face','x_m','y_m','z_m','area_m2'],
                     [(side,int(i),*centres[i],areas[i])
                      for side,centres,areas,valid in [('master',c,a,paired),('slave',cs,as_,paired_slave)]
                      for i in np.flatnonzero(~valid)])
    # Export on the exact CFD polygons. Bad pairs are NaN, never fabricated values.
    tree = ET.parse(folder/'master.vtp')
    piece = tree.find('./PolyData/Piece')
    if piece is None or int(piece.attrib['NumberOfPolys'])!=len(c):
        raise ValueError('Invalid VTP polygon count')
    cells = piece.find('CellData')
    if cells is None:
        cells = ET.SubElement(piece,'CellData')
    cells.set('Scalars','delta_Cp_master_minus_slave')
    exported = [dict(name='delta_Cp_master_minus_slave',kind='Float64',values=np.where(paired,delta,np.nan)),
                dict(name='delta_p_Pa',kind='Float64',values=np.where(paired,delta*q,np.nan)),
                dict(name='paired',kind='UInt8',values=paired.astype(int)),
                dict(name='dmz_about_Y',kind='Float64',values=np.where(paired,moments[:,1]/scale,np.nan))]
    for item in exported:
        entry = ET.SubElement(cells,'DataArray',type=item['kind'],Name=item['name'],format='ascii')
        entry.text = '\n'+' '.join(format(v,'.12g') for v in item['values'])+'\n'
    vtp_path = pics/f'{name}_pass{number:02d}_deltaCp.vtp'
    tree.write(vtp_path,encoding='utf-8',xml_declaration=True)
    theta = np.radians(float(row.get('hold') or 0.))
    e1 = np.array([np.cos(theta),0.,-np.sin(theta)])
    rim_centre = mesh.position[mesh.skirt].mean(0)
    x = (c-rim_centre)@e1
    half_width = max(float(np.max(np.abs((mesh.position[mesh.skirt]-rim_centre)@e1))),1e-12)
    xi = x/half_width
    extent = max(1.05,float(np.max(np.abs(xi[paired])))*1.001) if paired.any() else 1.05
    bounds = np.linspace(-extent,extent,9)
    bands = []
    for left,right in zip(bounds[:-1],bounds[1:]):
        mask = paired & (xi>=left) & (xi<(right if right<bounds[-1] else right+1e-10))
        bands.append([left,right,int(mask.sum()),float(a[mask].sum()),
                      float(np.average(delta[mask],weights=a[mask])) if mask.any() else np.nan,
                      float(forces[mask,2].sum()/(q*area0)),float(forces[mask,0].sum()/(q*area0)),
                      float(moments[mask,1].sum()/scale)])
    bands = np.array(bands)
    band_csv = save_csv(out/f'moment_bands_pass{number:02d}.csv',
                       ['x_body_Rrim_min','x_body_Rrim_max','faces','area_m2','delta_Cp','dCx','dCy','dmz'],bands)
    fig,axes = plt.subplots(1,2,figsize=(11,4))
    mid = bands[:,:2].mean(1)
    axes[0].bar(mid,bands[:,-1],width=np.diff(bounds)*.85,color=np.where(bands[:,-1]>=0,'#2166ac','#b2182b'))
    axes[0].set(title='Вклад участков купола в момент',ylabel='Δmz относительно коуша, ось +Y')
    axes[1].plot(mid,np.cumsum(bands[:,-1]),'o-',color='#333333',ms=4)
    axes[1].set(title='Накопленный момент давления',ylabel='Сумма Δmz')
    for ax in axes:
        ax.axhline(0,color='.5',lw=.7); ax.grid(alpha=.2)
        ax.set_xlabel('x′ / R кромки — геометрическая координата')
    fig.suptitle(f"α = {row.get('hold',0):g}° · согласованные пары: {100*coverage:.3f}% площади")
    fig.tight_layout()
    plot = pics/'moment_contributions.png'
    fig.savefig(plot,dpi=150); plt.close(fig)
    return [str(meta),table,str(vtp_path),band_csv,str(plot)],report


def wake_cut(work,name,row,diameter,speed):
    root = work/'fields'/name/f"pass{int(row['pass']):02d}"/'postProcessing/cut'
    candidates = [d/'plane.vtp' for d in pressure_table.numeric_dirs(root) if (d/'plane.vtp').is_file()]
    if not candidates:
        raise FileNotFoundError('No saved velocity cut for this pass')
    data = vtp.arrays(candidates[-1])
    points,velocity = np.asarray(data['Points']),np.asarray(data['U']).reshape(-1,3)
    if len(points)!=len(velocity):
        raise ValueError('Wake diagnostics require point-interpolated U in plane.vtp')
    conn,offset = data['connectivity'].ravel(),data['offsets'].ravel()
    starts = np.r_[0,offset[:-1]]
    triangles = np.array([[conn[a],conn[j],conn[j+1]]
                          for a,b in zip(starts,offset) for j in range(int(a)+1,int(b)-1)])
    tri = mtri.Triangulation(points[:,0],points[:,2],triangles)
    xz = points[triangles][:,:,[0,2]]
    edge1,edge2 = xz[:,1]-xz[:,0],xz[:,2]-xz[:,0]
    twice_area = np.abs(edge1[:,0]*edge2[:,1]-edge1[:,1]*edge2[:,0])
    tri.set_mask(twice_area<=diameter**2*1e-14)
    centres = xz.mean(1)
    # Local linear gradients on the saved polygons; no Delaunay interpolation
    # across the canopy or holes, and no global triangle-finder requirement.
    determinant = edge1[:,0]*edge2[:,1]-edge1[:,1]*edge2[:,0]
    u = velocity[triangles]
    with np.errstate(divide='ignore',invalid='ignore'):
        dux_dz = (edge1[:,0]*(u[:,2,0]-u[:,0,0])-edge2[:,0]*(u[:,1,0]-u[:,0,0]))/determinant
        duz_dx = ((u[:,1,2]-u[:,0,2])*edge2[:,1]-(u[:,2,2]-u[:,0,2])*edge1[:,1])/determinant
    # omega_y = dUx/dz - dUz/dx; this is one component, not |curl U|.
    with np.errstate(invalid='ignore'):
        omega = (dux_dz-duz_dx)*diameter/speed
    omega[twice_area<=diameter**2*1e-14] = np.nan
    finite = omega[np.isfinite(omega)]
    if not len(finite):
        raise ValueError('No non-degenerate velocity triangles')
    bound = max(float(np.percentile(np.abs(finite),98)),1e-6)
    fig,axes = plt.subplots(1,2,figsize=(11,5))
    vort = axes[0].tripcolor(tri,facecolors=omega,cmap='RdBu_r',vmin=-bound,vmax=bound)
    fig.colorbar(vort,ax=axes[0],label='ωy D₀ / V')
    speed_image = axes[1].tripcolor(tri,np.linalg.norm(velocity,axis=1)/speed,cmap='viridis',shading='gouraud')
    fig.colorbar(speed_image,ax=axes[1],label='V местная / V потока')
    if velocity[:,2].min()<0<velocity[:,2].max():
        axes[1].tricontourf(tri,velocity[:,2]/speed,
                           levels=[float(velocity[:,2].min()/speed)-1e-9,0],colors=[(1,.35,0,.16)])
        axes[1].tricontour(tri,velocity[:,2]/speed,levels=[0],colors='#e56b00',linewidths=1.3)
    axes[0].set_title('Завихренность в плоскости наклона')
    axes[1].set_title('Область Uz < 0 и её граница' if velocity[:,2].min()<0 else 'Обратное течение не обнаружено')
    # Crop and outline from the actual CFD surface, not the relaxed next shape.
    try:
        cp = pressure_folder(work,name,row)
        c,_,_,_ = vtp.faces_of(cp/'master.vtp')
        lo,hi = c.min(0),c.max(0)
        size = np.ptp(c,axis=0).max()
        surface_data = vtp.arrays(cp/'master.vtp')
        vertices = surface_data['Points']
        connectivity,ends = surface_data['connectivity'].ravel(),surface_data['offsets'].ravel()
        segments = []
        level = float(np.median(points[:,1]))
        for first,last in zip(np.r_[0,ends[:-1]],ends):
            polygon = vertices[connectivity[first:last]]
            if polygon[:,1].min()>level or polygon[:,1].max()<level:
                continue
            hits = []
            for p1,p2 in zip(polygon,np.roll(polygon,-1,axis=0)):
                d1,d2 = p1[1]-level,p2[1]-level
                if abs(d1)<1e-10:
                    hits.append(p1[[0,2]])
                elif d1*d2<0:
                    hits.append((p1+(p2-p1)*d1/(d1-d2))[[0,2]])
            if len(hits)>=2:
                segments.append([hits[0],hits[-1]])
        for ax in axes:
            ax.add_collection(LineCollection(segments,colors='k',linewidths=1.2))
            ax.set_xlim(lo[0]-size,hi[0]+size)
            ax.set_ylim(lo[2]-size,hi[2]+2*size)
    except FileNotFoundError:
        pass
    for ax in axes:
        ax.set_aspect('equal'); ax.set_xlabel('x, м'); ax.set_ylabel('z, м')
    fig.tight_layout()
    output = work/'pics/wake_diagnostics.png'
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output,dpi=150); plt.close(fig)
    table = save_csv(work/'out/wake_cut.csv',['x_m','z_m','omega_y_D0_V'],
                     np.c_[centres[:,0],centres[:,1],omega])
    info = dict(source=str(candidates[-1]),omega_component='dUx/dz - dUz/dx',
                reverse_flow_point_fraction=float(np.mean(velocity[:,2]<0)),
                note='Instantaneous/steady saved plane, not a time statistic. Reverse-flow fraction is a point-count diagnostic, not a volume fraction. Colour limits use the 98th percentile; CSV keeps all values.')
    path = work/'out/wake_diagnostics.json'
    path.write_text(json.dumps(info,indent=2),encoding='utf-8')
    return [str(output),table,str(path)]
