# Closed circular canopy at the Nosarev experimental scale.
diameter = (24 / np.pi) ** 0.5
scale = diameter / 19.51
start = float(FLAG.get("--start-angle", "0"))
line("D0, LINES, V, RHO, NU =", f"D0, LINES, V, RHO, NU = {diameter!r}, {diameter!r}, 30.0, 1.225, 1.46e-5")
line("AZIMUTH = -146.3", "AZIMUTH = 0.0")
line("MOMENT_TOL = 2000.0", f"MOMENT_TOL = {2000 * scale**3!r}")
line("return pos[:CANOPY].mean(axis=0), 11.0", f"    return pos[:CANOPY].mean(axis=0), {11 * scale!r}")
sub("scale 1;", f"scale {scale:.9f};")
sub("min (-26 -26 -10); max (26 26 75);",
    f"min ({-26*scale:.6f} {-26*scale:.6f} {-10*scale:.6f}); "
    f"max ({26*scale:.6f} {26*scale:.6f} {75*scale:.6f});")
sub("locationInMesh (0 0 -36);", f"locationInMesh (0 0 {-36*scale:.6f});")
sub("start (-12 0 -5);  end (22 0 -5);",
    f"start ({-12*scale:.6f} 0 {-5*scale:.6f});  end ({22*scale:.6f} 0 {-5*scale:.6f});")
sub("\nif PULLED:\n",
    f"\nif HOLD is None and not history and {start!r}:\n"
    f"    mesh.position = set_attitude(mesh.position, {start!r}, mesh.confluence[0])\n"
    "\nif PULLED:\n")
sub("static_dicts(refine_ball(mesh.position) if HOLD is not None else None)",
    "static_dicts(refine_ball(mesh.position))")
sub("import vtp\n", "import vtp\nfrom nos6_geometry import closed_circular, export_closed_surface\n")
sub("mesh = flat_circular(D0, LINES,", "mesh = closed_circular(D0, LINES,")
sub("export_surface(mesh,", "export_closed_surface(mesh,", count=2)
MODE.append(f"Nosarev 6 m2: closed pole, common confluence, initial tilt {start:g} deg")
