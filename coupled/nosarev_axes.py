"""cs.py profile: real Nosarev global speed axes, after nos6 and cords patches."""
if __name__=='__main__':
    if FLAG.get('--mesh-from') and FLAG.get('--mesh-frame')!='nosarev':
        raise ValueError('An external mesh must be rotated first and marked --mesh-frame=nosarev')
    sub('mesh = closed_circular(D0, LINES,',
        'from nosarev_frame import install_speed_frame\ninstall_speed_frame(globals())\n'
        'mesh = closed_circular(D0, LINES,')
    sub('    mesh.position = np.load(SEED)\n',
        '    mesh.position = speed_seed(np.load(SEED), FLAGS.get("--seed-frame","nosarev"))\n')
    sub('    if old.exists():\n','    if False:  # no implicit native-coordinate cache in this project\n')
    sub('_config=dict(D0=D0,','_config=dict(coordinate_frame="nosarev_speed",D0=D0,')
    sub('        "fixed_shape": FIXED_SHAPE,',
        '        "coordinate_frame": "nosarev_speed",\n'
        '        "force_global_N": total.tolist(), "moment_global_Nm": torque.tolist(),\n'
        '        "fixed_shape": FIXED_SHAPE,')
    MODE.append('Скоростные оси Носарева: U=(-V,0,0); Cx=-Fx/(qS), Cy=Fy/(qS), mz=Mz/(qSD0)')
