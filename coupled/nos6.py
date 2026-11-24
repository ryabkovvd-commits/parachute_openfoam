# Closed circular canopy at the Nosarev experimental scale.
diameter = (24 / np.pi) ** 0.5
scale = diameter / 19.51
start = float(FLAG.get("--start-angle", "0"))
line("D0, LINES, V, RHO, NU =", f"D0, LINES, V, RHO, NU = {diameter!r}, {diameter!r}, {float(FLAG.get('--V',30))!r}, 1.225, {float(FLAG.get('--nu',1.46e-5))!r}")
line("SHIFT_TOL = ", f"SHIFT_TOL = {float(FLAG.get('--shape-tol',.03))!r}")
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

if sys.argv[1] == "nos6_held":
    min_passes = 1 if '--fixed-shape' in sys.argv else 6
    # The series criteria take precedence over the cord patch's defaults.
    header_start = src.index("# ---- случай ")
    header_end = src.index("mesh = closed_circular(", header_start)
    sub(src[header_start:header_end], '''# ---- случай ----
angle_text = f"α = {HOLD:g}°" if HOLD is not None else "свободный наклон"
say(f"\\n{NAME} | купол 6 м², полюс закрыт | {angle_text} | "
    f"{time.strftime('%d.%m %H:%M:%S')}")
say(f"Поток V = {V:g} м/с, ν = {NU:g} м²/с | {ITERS} итераций CFD на проход")
say(f"Оболочка {GORES} полотнищ, {RINGS} колец | {GORES // STRIDE} строп по {LINES:.3f} м "
    f"(l = {LINES / D0:g})")
if SHEAR not in (None, 0.02):
    say(f"Доля сдвига для диагоналей {SHEAR:g}")
if PULLED:
    say(f"Затяжка {PULLED} строп на {STROKE:g} м")
if HOLDER:
    say("Учитывается реакция держателя")
if MORPH:
    say(f"Деформация сетки с повторным расчётом за {MORPH} итераций")
for _m in MODE:
    if not _m.startswith(("V =", "nu =", "Порог невязки формы:",
                          "Maximum Cx change:", "Ранняя остановка по допуску формы")):
        say(_m.replace("ТКАНЬ -- СТЕНКА С ВДУВОМ-ОТСОСОМ:", "Проницаемая ткань:")
              .replace("Cord drag: b=", "Сопротивление строп: d = ")
              .replace(" mm, Cd=", " мм, Cd = "))
''' + (
        'say("Фиксированная форма: один CFD-проход без структурной релаксации")\n' if '--fixed-shape' in sys.argv else
        'say(f"Выполняем все {PASSES} проходов")\n' if "--all-passes" in sys.argv else
        f'say(f"Проходы {min_passes}–{{PASSES}} | два прохода подряд: '
        'невязка формы < {SHIFT_TOL:.0e} от D0'
        + (f", |ΔCx| < {float(FLAG.get('--cx-tol', 0.003)):g}"
           if "cords" in FLAG.get("--patch", "").split(",") else "")
        + ', релаксация сошлась")\n'
    ) + "\n")
    sub('        say(f"затравки нет -- надуваю СИММЕТРИЧНЫЙ купол перепадом "\n'
        '            f"{PRELOAD_DCP}, затяжку прикладываю после")\n',
        '        say(f"Преднаполнение при ΔCp = {PRELOAD_DCP:g}")\n')
    sub('        say(f"  наполнен: наклон {tilt_of(mesh.position):.2f}°, "\n'
        '            f"Dp/D0 = {mesh.projected_diameter() / D0:.4f}")\n', "")
    if "--all-passes" not in sys.argv:
        sub("    if steady >= 2:\n",
            f"    if steady >= 2 and step >= {min_passes}:\n")
    sub('    settled = settled and row.get("relaxed", True)\n',
        '    settled = settled and row.get("relaxed", True)\n'
        f'    settled = settled and steady >= 2 and row["pass"] >= {min_passes}\n')
    sub("        why = []\n",
        "        why = []\n"
        f'        if last["pass"] < {min_passes}:\n'
        f'            why.append("выполнено меньше {min_passes} проходов")\n'
        '        elif steady < 2:\n'
        '            why.append("нет двух проходов подряд с выполнением всех условий")\n')
else:
    MODE.append(f"Nosarev 6 m2: closed pole, common confluence, initial tilt {start:g} deg")
