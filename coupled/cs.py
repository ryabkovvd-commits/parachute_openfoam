"""Связка на кластере: coupled.py плюс сток, густота сетки, нестационарный счёт.

    python cs.py ИМЯ ХОД СТРОП ИТЕРАЦИЙ ПРОХОДОВ [ЗАТРАВКА.npy] [ключи]

Ключи coupled.py (--угол, --шаг, --сдвиг ...) работают как прежде.  Свои:
    --sink=D,CW   ткань -- объёмный сток в ОДНУ ячейку (владелец грани),
                  закон dp/rho = D*Un + CW*|Un|*Un; D = 5000, CW = 0 -- глухая
    --mesh=R      все ячейки мельче в R раз по каждому направлению
    --shape-tol=E порог внешней невязки формы в долях D0 при заданном угле
    --all-passes  выполнить все заданные проходы без ранней остановки по допуску
    --load=rings  осреднить перепад по поясам только для механики оболочки
    --fabric-tan=K доля касательной скорости со стороны отсоса на стороне выдува
    --nu=NU       вязкость, м2/с (1.46e-5); 1.03e-4 -- Re модели Носарева Dп 2.76 м
    --urans=T     pimpleFoam на T секунд; силы -- среднее по второй половине
    --patch=имя   дополнительные правки из coupled/имя.py (mks3 -- купол опыта 1980)
    --fabric=D,CW ткань -- стенка с «вдувом-отсосом»: нормальная скорость сквозь
                  ткань по закону dp/rho = D*Un + CW*|Un|*Un (--fabric-relax=0.2).
                  Проверено дома 28.09: глухой предел -0.12 % к стенке.
                  ИСПОЛЬЗОВАТЬ ЕГО, а не --sink: сток на куполе теряет ~50 % силы.

coupled.py не меняется: правки вносятся заменой текста при запуске, и каждая
обязана найтись ровно один раз -- иначе остановка, а не молчаливый счёт по
старой постановке.  Этот файл заменяет cc.py.
"""
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vtp  # noqa: E402

FLAG = {a.split("=")[0]: a.split("=", 1)[1]
        for a in sys.argv[1:] if a.startswith("--") and "=" in a}
SINK = [float(x) for x in FLAG["--sink"].split(",")] if "--sink" in FLAG else None
MESH = float(FLAG.get("--mesh", 1))
URANS = float(FLAG["--urans"]) if "--urans" in FLAG else None
if SINK and URANS:
    sys.exit("cs.py: --sink и --urans вместе не предусмотрены")
# Подвижная сетка (coupled.py --подвижная, 28.10) пока только для гладкого
# купола и установившегося счёта: у купола со щелями другая поверхность, а в
# URANS продолжение считается в секундах, а не в итерациях.
if ("--подвижная" in " ".join(sys.argv) or "--morph" in " ".join(sys.argv)) and (URANS or "mks3" in FLAG.get("--patch", "")):
    sys.exit("cs.py: --подвижная пока не сочетается с --urans и --patch=mks3")

src = (HERE / "coupled.py").read_text(encoding="utf-8")


def sub(old, new, count=1):
    global src
    if src.count(old) != count:
        sys.exit(f"cs.py: {src.count(old)} совпадений вместо {count}: {old[:70]!r}")
    src = src.replace(old, new)


def line(key, new):
    """Заменить единственную строку, в которой есть key."""
    global src
    rows = src.split("\n")
    hit = [i for i, r in enumerate(rows) if key in r]
    if len(hit) != 1:
        sys.exit(f"cs.py: строк с {key!r}: {len(hit)}")
    rows[hit[0]] = new
    src = "\n".join(rows)


# ---- кластер: пути, ядра, запуск без WSL ------------------------------------
sub("import json\n", "import json\nimport os\nfrom shlex import quote\n")
line('CASE = Path(r"D:', 'CASE = Path("case").resolve()')
line('OUT = Path(r"D:', 'OUT = Path("out").resolve()')
line("D0, LINES, V, RHO, NU =", "D0, LINES, V, RHO, NU = 19.51, 19.51, " + FLAG.get("--V", "9.0") + ", 1.225, " + FLAG.get("--nu", "1.46e-5"))
line("GORES, RINGS = 16, 8", "GORES, RINGS = 96, 12")
line("CORES = 4", 'CORES = int(os.environ.get("SLURM_NTASKS", 4))')
line("WHERE = ", "WHERE = quote(str(CASE))")
line('FIELDS = Path(r"D:', 'FIELDS = Path("fields").resolve()')
line('KEEP = "/mnt', "KEEP = str(FIELDS)")
# Три сценария: проход с перестройкой, только сетка, проход на подвижной сетке.
sub("source /usr/lib/openfoam/openfoam2412/etc/bashrc\n", "", count=3)
sub("pkill -f simpleFoam; pkill -f mpirun; sleep 2\n", "", count=2)
line("coupled_run.sh\")", '    path = Path("coupled_run.sh").resolve()')
rows = src.split("\n")
i = [k for k, r in enumerate(rows) if "export PASS={step:02d}" in r]
assert len(i) == 1 and '"wsl"' in rows[i[0] - 1], "cs.py: вызов wsl не найден"
rows[i[0] - 1:i[0] + 3] = ['        ["bash", str(path)], env=dict(os.environ, PASS=f"{step:02d}"),']
src = "\n".join(rows)

# Сечение y = 0 совпадает с плоскостью граней сетки (блоки по 4 м от -80), и
# cuttingPlane режет ячейки с обеих сторон: на картинке сетки появляются
# несуществующие «кривые».  Сдвиг на 1 мм -- доля самой мелкой ячейки.
sub("type cuttingPlane;  point (0 0 0);  normal (0 1 0);",
    "type cuttingPlane;  point (0 0.001 0);  normal (0 1 0);")

MODE = []
MODE.append("V = " + FLAG.get("--V", "9.0") + " м/с")
MODE.append("nu = " + FLAG.get("--nu", "1.46e-5") + " м2/с")

if "--shape-tol" in FLAG:
    try:
        shape_tol = float(FLAG["--shape-tol"])
    except ValueError:
        sys.exit("cs.py: --shape-tol должен быть положительным числом")
    if not np.isfinite(shape_tol) or shape_tol <= 0:
        sys.exit("cs.py: --shape-tol должен быть конечным и положительным")
    line("SHIFT_TOL = ", f"SHIFT_TOL = {shape_tol!r}")
    sub("        settled = left < SHIFT_TOL\n", "        settled = steady >= 2\n")
    MODE.append(f"Порог невязки формы: {shape_tol:g} от D0")

if "--all-passes" in sys.argv:
    sub("    if steady >= 2:\n", "    if steady >= 2 and step >= PASSES:\n")
    MODE.append("Ранняя остановка по допуску формы отключена")

# ---- густота сетки ----------------------------------------------------------
if MESH != 1:
    n = [round(c * MESH) for c in (40, 40, 64)]
    sub("(40 40 64)", f"({n[0]} {n[1]} {n[2]})")
    MODE.append(f"сетка: шаг / {MESH:g} по каждому направлению, фон {n[0]}x{n[1]}x{n[2]}")

# ---- нестационарный счёт ----------------------------------------------------
if URANS:
    T = URANS
    sub("application simpleFoam;  startFrom startTime;  startTime 0;\n"
        "stopAt endTime;  endTime {ITERS};  deltaT 1;",
        "application pimpleFoam;  startFrom startTime;  startTime 0;\n"
        f"stopAt endTime;  endTime {T:g};  deltaT 0.002;\n"
        "adjustTimeStep yes;  maxCo 1;  maxDeltaT 0.05;")
    sub("writeControl timeStep;  writeInterval {max(ITERS // 8, 1)};",
        f"writeControl adjustable;  writeInterval {T / 8:g};")
    sub("writeControl timeStep;  writeInterval 25;",
        "writeControl timeStep;  writeInterval 1;")
    sub("fields (p);", "fields (pMean);")
    sub("    surfacePressure\n    {{",
        "    average\n    {{\n        type fieldAverage;  libs (fieldFunctionObjects);\n"
        f"        timeStart {T / 2:g};  writeControl writeTime;\n"
        "        fields ( p {{ mean on; prime2Mean off; base time; }} );\n    }}\n"
        "    surfacePressure\n    {{")
    sub("ddtSchemes { default steadyState; }", "ddtSchemes { default backward; }")
    sub("bounded Gauss", "Gauss", count=3)
    sub("    p { solver GAMG; tolerance 1e-6; relTol 0.01; smoother GaussSeidel; }",
        "    p { solver GAMG; tolerance 1e-6; relTol 0.01; smoother GaussSeidel; }\n"
        "    pFinal { solver GAMG; tolerance 1e-6; relTol 0; smoother GaussSeidel; }")
    sub('    "(U|k|omega)" {',
        '    "(U|k|omega)Final" { solver smoothSolver; smoother symGaussSeidel;'
        " tolerance 1e-8; relTol 0; }\n"
        '    "(U|k|omega)" {')
    sub("SIMPLE { nNonOrthogonalCorrectors 0; consistent yes; }\n"
        'relaxationFactors { equations { ".*" 0.9; } }',
        "PIMPLE { nOuterCorrectors 1; nCorrectors 2; nNonOrthogonalCorrectors 0; }\n"
        'relaxationFactors { equations { ".*" 1; } }')
    sub("simpleFoam -parallel", "pimpleFoam -parallel", count=2)
    # Перепад на куполе берём из осреднённого давления.
    _arrays = vtp.arrays

    def mean_arrays(path):
        got = _arrays(path)
        if "pMean" in got:
            got["p"] = got["pMean"]
        return got
    vtp.arrays = mean_arrays
    MODE.append(f"НЕСТАЦИОНАРНО: pimpleFoam {T:g} с, Co <= 1, "
                f"среднее давление за {T / 2:g}...{T:g} с")

if "--robust" in sys.argv and not URANS:
    sub('SIMPLE { nNonOrthogonalCorrectors 0; consistent yes; }\nrelaxationFactors { equations { ".*" 0.9; } }',
        'SIMPLE { nNonOrthogonalCorrectors 1; consistent no; }\nrelaxationFactors { fields { p 0.3; } equations { ".*" 0.7; } }')
    MODE.append("robust SIMPLE: p 0.3, eq 0.7, nonOrth 1")
if "--skew" in FLAG:
    sub("  nSmoothScale 4; errorReduction 0.75; }",
        "  nSmoothScale 4; errorReduction 0.75; maxBoundarySkewness " + FLAG["--skew"] + "; maxInternalSkewness " + FLAG["--skew"] + "; }")
    MODE.append("snappy skewness max " + FLAG["--skew"])
if "--pcg" in sys.argv and not URANS:
    sub("    p { solver GAMG; tolerance 1e-6; relTol 0.01; smoother GaussSeidel; }",
        "    p { solver PCG; preconditioner DIC; tolerance 1e-6; relTol 0.01; }")
    MODE.append("p solver PCG DIC")

# ---- сток на ткани ----------------------------------------------------------
SINK_CODE = r"""
fabricSink
{
    type            vectorCodedSource;
    selectionMode   all;
    fields          (U);
    name            fabricSink;
    codeInclude
    #{
        #include "OFstream.H"
    #};
    codeCorrect
    #{
    #};
    codeConstrain
    #{
    #};
    codeAddSup
    #{
        const fvMesh& m = mesh_;
        const label zoneID = m.faceZones().findZoneID("canopyFaces");
        if (zoneID < 0) return;
        const faceZone& fz = m.faceZones()[zoneID];
        const surfaceVectorField& Sf = m.Sf();
        const surfaceVectorField& Cf = m.Cf();
        const labelUList& own = m.owner();
        const labelUList& nei = m.neighbour();
        const boolList& flip = fz.flipMap();
        const bool byOwner = SINK_OWNER;
        const volVectorField& U = m.lookupObject<volVectorField>("U");
        vectorField& src = eqn.source();
        scalarField& dg = eqn.diag();
        const scalar Dc = SINK_D;
        const scalar Cw = SINK_CW;
        const label nInt = m.nInternalFaces();

        // on write steps: face centre, area vector, jump dp/rho per face
        OFstream* os = nullptr;
        if (m.time().writeTime())
        {
            const fileName dir = m.time().rootPath()/m.time().globalCaseName()
                /"postProcessing"/"surfacePressure"/m.time().timeName();
            mkDir(dir);
            os = new OFstream(dir/("fabric" + Foam::name(Pstream::myProcNo()) + ".dat"));
        }
        // Fabric normal, smoothed over neighbouring zone faces (once per run).
        // The snapped zone faces are stair-stepped, nearly axis-aligned; the
        // normal of a single face is not the normal of the fabric, and flow
        // "tangential" to a step is partly flow THROUGH the fabric, which the
        // sink would never resist.  FlowVision's parachute modifier uses the
        // shell element normal for the same reason.  Area is projected onto
        // the smoothed normal, so the steps do not overcount the fabric.
        static vectorField nsm;
        if (SMOOTH && nsm.size() != fz.size())
        {
            nsm.setSize(fz.size(), Zero);
            forAll(fz, i)
            {
                const label fi = fz[i];
                if (fi >= nInt) continue;
                const scalar r2 = sqr(2.5*Foam::sqrt(mag(Sf[fi])));
                vector acc(Zero);
                forAll(fz, j)
                {
                    const label fj = fz[j];
                    if (fj >= nInt) continue;
                    if (magSqr(Cf[fj] - Cf[fi]) > r2) continue;
                    acc += (flip[j] ? -1.0 : 1.0)*Sf[fj];
                }
                nsm[i] = acc/(mag(acc) + VSMALL);
            }
        }
        vector force(Zero);
        label skipped = 0;
        forAll(fz, i)
        {
            const label facei = fz[i];
            if (facei >= nInt) { ++skipped; continue; }
            const vector S0 = Sf[facei];
            const scalar a0 = mag(S0);
            if (a0 < SMALL) continue;
            const vector ns = SMOOTH ? nsm[i] : (flip[i] ? -1.0 : 1.0)*S0/a0;
            const scalar a = SMOOTH ? a0*mag(S0/a0 & ns) : a0;
            if (a < SMALL) continue;
            const vector S = a*ns;
            // Zone normal = surface normal (outward, away from the confluence):
            // Sf when not flipped, -Sf when flipped.  The sink cell is always the
            // one BEHIND the zone normal (inside the canopy): a continuous layer.
            // Taking the owner instead (byOwner) gives a checkerboard on a curved
            // surface and the air leaks through it -- on the canopy that lost
            // half the drag (28.09, sb_a00: 5248 N against 10180 N for the wall).
            const scalar sgn = 1.0;   // S is already the oriented fabric area vector
            const vector n = S/a;
            const label c = byOwner ? own[facei] : (flip[i] ? nei[facei] : own[facei]);
            const scalar Un = (U[c] & n);
            const scalar k = a*(Dc + Cw*mag(Un));
            dg[c] -= k;
            src[c] -= k*(U[c] - Un*n);
            force += k*Un*n;
            if (os)
            {
                const vector& x = Cf[facei];
                *os << x.x() << ' ' << x.y() << ' ' << x.z() << ' '
                    << sgn*S.x() << ' ' << sgn*S.y() << ' ' << sgn*S.z() << ' '
                    << k*Un/a << nl;
            }
        }
        delete os;
        reduce(force, sumOp<vector>());
        reduce(skipped, sumOp<label>());
        if (m.time().timeIndex() % 50 == 0)
        {
            Info<< "fabricSink: time " << m.time().timeName()
                << " force(kinematic) " << force
                << " skippedFaces " << skipped << endl;
        }
    #};
}
"""


def extra_dicts(write, head, cores):
    """Словари стока: fvOptions и разбиение, не рвущее ткань."""
    code = (SINK_CODE.replace("SINK_D", repr(SINK[0])).replace("SINK_CW", repr(SINK[1]))
            .replace("SINK_OWNER", "true" if "--sink-owner" in sys.argv else "false")
            .replace("SMOOTH", "false" if "--sink-raw" in sys.argv else "true"))
    write("system/fvOptions", head("dictionary", "fvOptions", "system") + code)
    write("system/decomposeParDict", head("dictionary", "decomposeParDict", "system")
          + f"\nnumberOfSubdomains {cores};\nmethod scotch;\n"
          "constraints { fabric { type preserveFaceZones; zones (canopyFaces); } }\n")


def sink_dcp(folder, kin=vtp.KIN, master=None, slave=None):
    """Перепад по граням ткани -- ровно тот, что снял сток: грань, площадь,
    нормаль и dp/rho = (D + CW*|Un|)*Un.  Сила грани dcp*S*q*n."""
    parts = [np.loadtxt(f, ndmin=2) for f in sorted(Path(folder).glob("fabric*.dat"))
             if f.stat().st_size]
    rows = np.vstack(parts)
    area = np.linalg.norm(rows[:, 3:6], axis=1)
    return rows[:, :3], area, rows[:, 3:6] / area[:, None], rows[:, 6] / kin


if SINK:
    sub("faceZone canopyFaces; faceType baffle;", "faceZone canopyFaces; faceType internal;")
    sub('patches ( ".*canopy.*" );', "patches ( farfield );")
    src, hits = re.subn(r"    surfacePressure\n.*?\n    cut\n", "    cut\n", src, flags=re.S)
    if hits != 1:
        sys.exit("cs.py: блок surfacePressure не найден")
    sub('    (CASE / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)',
        '    (CASE / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)\n'
        "    EXTRA(write, head, CORES)")
    vtp.delta_cp = sink_dcp
    MODE.append(f"ТКАНЬ -- СТОК в одну ячейку: D = {SINK[0]:g} м/с, Cw = {SINK[1]:g}"
                + ("  (глухой контроль)" if SINK[0] >= 1000 and SINK[1] == 0 else ""))

# ---- проницаемая ткань: стенка с «вдувом-отсосом» (код граничного условия -- fabricU.txt)
FABRIC = [float(x) for x in FLAG["--fabric"].split(",")] if "--fabric" in FLAG else None
RELAX = float(FLAG.get("--fabric-relax", 0.2))
FABRIC_TAN = float(FLAG.get("--fabric-tan", 0.0))
if not 0.0 <= FABRIC_TAN <= 1.0:
    sys.exit("cs.py: --fabric-tan должен быть от 0 до 1")


def fabric_u():
    code = (HERE / "fabricU.txt").read_text(encoding="utf-8")
    code = code.replace("FAB_D", repr(FABRIC[0])).replace("FAB_CW", repr(FABRIC[1]))
    code = code.replace("FAB_RELAX", repr(RELAX))
    code = code.replace("FAB_TAN", repr(FABRIC_TAN))
    out = []
    for patch, name in (("canopy", "fabricIn"), ("canopy_slave", "fabricOut")):
        out.append("    " + patch + "\n    {\n        type codedFixedValue;\n"
                   + "        value uniform (0 0 0);\n        name " + name + ";\n"
                   + "        code\n        #{\n" + code + "\n        #};\n    }")
    return "\n".join(out)


def extra_fabric(write, head, cores):
    write("system/decomposeParDict", head("dictionary", "decomposeParDict", "system")
          + "\nnumberOfSubdomains " + str(cores) + ";\nmethod scotch;\n"
          + "constraints { baffles { type preserveBaffles; } }\n")


if FABRIC:
    sub('    ".*canopy.*" {{ type noSlip; }}', "{FABRIC_U}")
    sub('    (CASE / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)',
        '    (CASE / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)\n'
        + "    EXTRA(write, head, CORES)")
    MODE.append("ТКАНЬ -- СТЕНКА С ВДУВОМ-ОТСОСОМ: D = " + str(FABRIC[0])
                + " м/с, Cw = " + str(FABRIC[1]) + ", недорелаксация " + str(RELAX))
    if FABRIC_TAN:
        MODE.append(f"Ткань: доля касательной скорости на выдуве {FABRIC_TAN:g}")

if FLAG.get("--load", "raw") not in {"raw", "rings"}:
    sys.exit("cs.py: --load принимает raw или rings")
if FLAG.get("--load") == "rings":
    if SINK or FLAG.get("--patch") or FLAG.get("--перенос") == "точный":
        sys.exit("cs.py: --load=rings требует круглый купол и обычный перенос нагрузки")
    sub("import vtp\n", "import vtp\nfrom ringload import ring_load\n")
    sub("    relaxed = False\n",
        '    ring_output = OUT / f"{NAME}_ringload_{step:02d}.npz"\n'
        "    structural_load = ring_load(mesh, folder, Q, KIN, ring_output)\n"
        "    relaxed = False\n")
    if "applied = attitude_reaction(mesh, load)" in src:
        sub("    applied = attitude_reaction(mesh, load) if HOLD is not None and HOLDER else load\n",
            "    applied = (attitude_reaction(mesh, structural_load)\n"
            "               if HOLD is not None and HOLDER else structural_load)\n")
    else:
        sub("result = inflate(mesh, node_force=load,",
            "result = inflate(mesh, node_force=structural_load,")
    MODE.append("Нагрузка оболочки: кольцевое осреднение перепада давления")

sub('say(f"диагонали: ', 'for _m in MODE:\n    say(_m)\nsay(f"диагонали: ')

# --patch=имя[,имя] -- дополнительные правки из coupled/имя.py (например, mks3)
for _name in filter(None, FLAG.get("--patch", "").split(",")):
    exec(compile((HERE / f"{_name}.py").read_text(encoding="utf-8"), _name + ".py", "exec"))

sys.argv[0] = str(HERE / "coupled.py")
g = {"__name__": "__main__", "__file__": sys.argv[0], "MODE": MODE,
     "EXTRA": extra_fabric if FABRIC else extra_dicts,
     "FABRIC_U": fabric_u() if FABRIC else ""}
exec(compile(src, sys.argv[0], "exec"), g)

# ---- нестационарный: разброс сил по времени ----------------------------------
if URANS:
    say, mesh = g["say"], g["mesh"]
    runs = sorted((Path("fields") / g["NAME"]).glob("pass*/postProcessing/forces/0"))
    files = {p.name: p for p in runs[-1].glob("*.dat")} if runs else {}
    if "force.dat" not in files or "moment.dat" not in files:
        say("  !! нет force.dat / moment.dat -- разброс сил не посчитан")
        sys.exit(0)

    def table(path):
        rows = [[float(x) for x in s.replace("(", " ").replace(")", " ").split()]
                for s in path.read_text().splitlines() if s.strip() and s[0] != "#"]
        a = np.array(rows)
        return a[:, 0], (a[:, 4:7] if a.shape[1] >= 10 else a[:, 1:4])

    t, force = table(files["force.dat"])
    _, moment = table(files["moment.dat"])
    anchor = mesh.position[mesh.confluence[0]]
    moment = moment - np.cross(anchor, force)
    qs = g["Q"] * g["S0"]
    late = t >= URANS / 2
    say(f"\n  силы по времени (давление), {late.sum()} отсчётов за "
        f"{t[late][0]:.1f}...{t[-1]:.1f} с:")
    for name, val in (("c_x", force @ g["ONSET"] / qs), ("c_y", force @ g["SIDE"] / qs),
                      ("m_z", moment @ g["PIVOT"] / (qs * g["D0"]))):
        v = val[late]
        say(f"    {name}  среднее {v.mean():+.4f}  разброс {v.std():.4f}  "
            f"от {v.min():+.4f} до {v.max():+.4f}")
