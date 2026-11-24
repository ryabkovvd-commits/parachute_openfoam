"""Связанный расчёт купола: форма и обтекание, один самодостаточный файл.

    python coupled.py ИМЯ ХОД СТРОП [ИТЕРАЦИЙ] [ПРОХОДОВ] [ЗАТРАВКА.npy]

ПОЧЕМУ ЗДЕСЬ ВСЁ В ОДНОМ ФАЙЛЕ.  Прежние драйверы подставляли постановку из
coupled_final.py через exec, и это дважды выходило боком: сперва пропадали имена
(NameError уже в разгар счёта), потом временную папку вычистило вместе со всеми
скриптами. Постановка восстановлена из самих расчётных папок -- то есть из
словарей, которыми посчитаны имеющиеся результаты, -- и вписана сюда целиком.
Файл лежит на D:, а не во временной папке.

МЕТОДИКА, коротко, с указанием чем каждое обосновано:

  купол на воле        поворот не подавляется: опыт показал, что он сам
                       доворачивается к равновесию за ~13 проходов, вчетверо
                       дешевле развёртки по одиннадцати углам
  2000 итераций        на 250-й момент бывает ОБРАТНОГО знака, нуль он
                       переходит около 590-й
  узловые силы         перенос по граням: 0.000 % по силе против 2.8 % и
                       1840 % по моменту у осреднения по поясам
  недорелаксация       множитель по Эйткену, подбирается сам по двум последним
                       невязкам; приём из связок через preCICE
  остановка            наклон меняется меньше 0.5° два прохода подряд; 0.5° --
                       ЗАМЕРЕННЫЙ разброс метода от перестройки сетки
  k-омега SST          стационарный SIMPLEC, недорелаксация 0.9

ЧТО ПРОВЕРЕНО ПО ОПЫТУ:
  форма    при отношении строп NASA даёт Sp/S0 = 0.473 против замеренных 0.480;
           профиль совпал с профилем Хайнриха (табл. B-1) до 0.1 %
  сила     C_D0 = 0.606 против 0.680 у жёсткой продувочной модели Хайнриха
           той же формы и наполненности -- 11 %, обычная точность RANS на
           тупом теле с массивным отрывом
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from parafoil.inflation import flat_circular, inflate
from parafoil.inflation.exchange import export_surface
from parafoil.inflation.mesh import panel_area_and_normal
from parafoil.inflation.pressure import uniform_pressure

import vtp
from run_guard import atomic_json, begin_run, check_cfd, geometry_quality, mark_status, source_hashes

CASE = Path(r"D:\cfd\run\queue")
OUT = Path(r"D:\cfd\queue")
D0, LINES, V, RHO, NU = 19.51, 15.5, 9.0, 1.225, 1.46e-5
Q, KIN = 0.5 * RHO * V**2, 0.5 * V**2
S0 = np.pi * D0**2 / 4
GORES, RINGS = 16, 8
AZIMUTH = -146.3
CORES = 4
#: Потолок шагов релаксации формы.  Было 40 000 x 12 = 480 000, и этого НЕ
#: хватает: замер на 40 градусах показал, что до полного равновесия нужно
#: 1 220 000 шагов.  Обрезалось при этом в основном медленное вращение вокруг
#: коуша, которое в режиме удержания привеса всё равно снимается -- поэтому
#: развёртка по углам от обрезки не пострадала, -- но при новом флаге
#: `relaxed` такой случай честно помечается несошедшимся.
#:
#: Потолок -- страховка, а не работа: релаксация останавливается сама, как
#: только выполнены оба критерия, и высокий потолок ничего не стоит, пока до
#: него не дошло.
CHUNK, CHUNKS = 40_000, 100
ALPHA = 0.0
#: Перепад для первичного наполнения, если затравки нет.
PRELOAD_DCP = 1.6

FLAGS = {a.split("=")[0]: a.split("=", 1)[1] if "=" in a else ""
         for a in sys.argv[1:] if a.startswith("--")}
WORDS = [a for a in sys.argv[1:] if not a.startswith("--")]

NAME = WORDS[0]
STROKE = float(WORDS[1])
PULLED = int(WORDS[2])
ITERS = int(WORDS[3]) if len(WORDS) > 3 else 2000
PASSES = int(WORDS[4]) if len(WORDS) > 4 else 4
SEED = Path(WORDS[5]) if len(WORDS) > 5 else None
FIXED_SHAPE = "--fixed-shape" in FLAGS
if FIXED_SHAPE and (PASSES!=1 or SEED is None or not SEED.is_file() or PULLED):
    raise ValueError('--fixed-shape requires one CFD pass, an existing input shape and no line pulling')

#: Продувка под заданным углом атаки: --угол=35.  Привес держим, форму нет.
#:
#: Это постановка Носарева (ЦАГИ, вып. 1735, с. 9): купол держат за коуши строп
#: и за СКОЛЬЗЯЩИЙ полюс, чтобы крепление «не ограничивало форму купола», а
#: угол задают поворотом кабины весов.  Так форма остаётся за стержневой
#: моделью, а искать привес расчёту не нужно -- и уходит самое дорогое: в
#: свободном счёте купол доворачивается по 0.9° за проход.
HOLD = float(FLAGS["--угол"]) if "--угол" in FLAGS else None
ATTITUDE = FLAGS.get('--attitude','legacy')
if ATTITUDE not in {'legacy','inner'} or (ATTITUDE=='inner' and (HOLD is None or FIXED_SHAPE)):
    raise ValueError('--attitude=inner requires a deformable held-angle case')
if ATTITUDE=='inner' and ('--holder' in FLAGS or '--держатель' in FLAGS):
    raise ValueError('Do not combine inner kinematic constraint with holder reaction')
CX_TOL = float(FLAGS.get('--cx-tol',.003))
TURBULENCE_LENGTH = float(FLAGS.get('--turbulence-length',1.24))
TURBULENCE_INTENSITY = float(FLAGS.get('--turbulence-intensity',.01))
if not np.isfinite([D0,LINES,V,RHO,NU,CX_TOL,TURBULENCE_LENGTH,TURBULENCE_INTENSITY]).all() or min(D0,LINES,V,RHO,NU,CX_TOL,TURBULENCE_LENGTH,TURBULENCE_INTENSITY)<=0:
    raise ValueError('Physical parameters and tolerances must be finite and positive')
if HOLD is not None:
    # Наклон кладём в плоскость xz: тогда поперечная сила идёт по +x, а ось
    # момента -- ровно +y, вокруг которой и поворачиваем.
    AZIMUTH = 0.0

#: Число полотнищ и длина строп -- параметры, а не константы.
#:
#: Число полотнищ у Носарева -- величина первого порядка, а не разрешение:
#: между стропами ткань выпучивается дугами, и чем строп меньше, тем мидель
#: больше.  Его формула (7) даёт c_x0(16)/c_x0(64) = 1.18.  Наша модель до сих
#: пор считалась на 16 полотнищах, то есть как купол с 16 стропами, тогда как у
#: G-12 их 64.  Длина нужна затем же: перевести расчёт в конфигурацию фиг. 32
#: (n = 24, l = 1.0) и сравнивать без единой поправки.
GORES = int(FLAGS.get("--полотнищ") or GORES)
LINES = float(FLAGS.get("--стропы") or LINES)
RINGS = int(FLAGS.get("--колец") or RINGS)
#: Стропа не на каждом узле, а через этот шаг: иначе между двумя стропами у
#: ткани нет ни одного узла и она не может выгнуться дугой.  Замер: без этого
#: силуэт даёт 0.4282 против 0.4001 у многоугольника по точкам крепления, то
#: есть вклад выпуклостей ровно нулевой.  Сошедшееся значение -- 4 узла на
#: полотнище и 12 колец.
STRIDE = int(FLAGS.get("--шаг") or 1)
#: Доля сдвиговой жёсткости к растяжной, для диагоналей панели.  Ноль -- как
#: было, без диагоналей вовсе.
SHEAR = float(FLAGS["--сдвиг"]) if "--сдвиг" in FLAGS else None
#: Своя расчётная папка: развёртка по углам не должна лезть в ту, где идёт
#: очередь свободных случаев -- controlDict читается на ходу (runTimeModifiable).
CASE = Path(FLAGS["--папка"]) if "--папка" in FLAGS else CASE
#: Узлы купола идут первыми, коуш последним.
CANOPY = (RINGS + 1) * GORES
#: Подвижная сетка: --подвижная[=N].  Первый проход -- snappy как обычно, дальше
#: сетка прошлого прохода ДЕФОРМИРУЕТСЯ под новую форму (moveMesh) и счёт
#: продолжается с её полей N итераций (по умолчанию 500: на n24_a20 m_z после
#: деформации встал за 100-200 итераций, дальше 0.0201 до 3000-й).
#:
#: ЗАМЕР 28.10 (n24_a20, проход 1 -> 2): скачок m_z между проходами
#: 0.0198 -> 0.0232 на 94 % дала перестройка сетки и лишь на 6 % изменение
#: формы (на деформированной сетке 0.0200).  c_x: форма -0.004, сетка -0.002.
#:
#: Зачем.  Перестройка сетки на каждом проходе смешивает изменение формы с
#: изменением сетки: на установившейся форме момент от одной перестройки гуляет
#: на ±2000 Н·м (отсюда TILT_TOL = 0.5°).  И каждый проход шёл с нуля, потому
#: что перенос поля на другую сетку вносит ошибку интерполяции.  У
#: деформированной сетки топология та же -- поле прошлого прохода ложится на
#: неё без интерполяции.  Как переносится форма -- tools/meshmorph.py.
#:
#: Если после деформации есть вывернутые ячейки или перекос больше MAX_SKEW,
#: проход откатывается на обычную перестройку.
#: Латинские синонимы -- для кластера, где ключи набираются руками, а
#: кириллица в них уже подводила (буквы-двойники).
for _ru, _en in (("--подвижная", "--morph"), ("--невязка-нагрузки", "--load-tol")):
    if _en in FLAGS:
        FLAGS[_ru] = FLAGS[_en]
if "--exact-load" in FLAGS:
    FLAGS["--перенос"] = "точный"
MORPH = int(FLAGS["--подвижная"] or 500) if "--подвижная" in FLAGS else None
#: --невязка-нагрузки=0.02: случай сошёлся, только если и узловые силы между
#: проходами изменились меньше этой доли (по норме).  По умолчанию невязка
#: нагрузки только пишется в журнал.
LOAD_TOL = float(FLAGS["--невязка-нагрузки"]) if "--невязка-нагрузки" in FLAGS else None
MAX_SKEW = 8.0
#: --след=L (латиницей --wake=L): цилиндр сгущения уровня 3 (вдвое крупнее
#: сферы) от центра купола вниз по потоку на L*D0, радиус 0.42*D0.  Только в
#: режиме --угол (там есть сфера).  См. dynmesh/ДРУГИЕ_ПАКЕТЫ.md, п. 3.
if "--wake" in FLAGS:
    FLAGS["--след"] = FLAGS["--wake"]
WAKE = float(FLAGS["--след"]) if FLAGS.get("--след") else 0.0
if MORPH:
    import meshmorph

#: Наклон встал, когда за два прохода подряд он двигается меньше этого.
#: 0.5° -- замеренный разброс: на s091_p4_b четыре полных прохода по
#: установившейся форме дали Δнакл -0.11, +0.34, -0.02, -0.47 при неподвижном
#: наклоне.  Источник -- перестройка сетки каждый проход (480 673 / 480 407 /
#: 480 554 / 481 275 ячеек), момент от неё гуляет на ±2000 Н·м, что при наклоне
#: 5365 Н·м/град и даёт 0.4°.
TILT_TOL = 0.50
#: И момент должен быть мал.  Наклон замирает РАНЬШЕ нагрузки, и правило
#: только по наклону останавливало расчёт с невыбранным моментом: s180_p8 встал
#: при 23 825 Н·м, то есть в четырёх с половиной градусах от равновесия.
#: Что такое «мал», показал симметричный случай: там момент вышел 2.9 Н·м.
#: Ниже двух тысяч требовать нельзя -- столько даёт перестройка сетки от
#: прохода к проходу.
MOMENT_TOL = 2000.0
#: В продувке под заданным углом ни наклон, ни момент критерием быть не могут:
#: наклон задан, а момент мы как раз и меряем -- он и не должен обращаться в
#: ноль нигде, кроме балансировки.  Останавливаемся, когда встала ФОРМА.
#: 3e-3 от D0 -- это 5.9 см среднего хода узла, вчетверо меньше ячейки у
#: поверхности (0.14...0.25 м), то есть сетка такого сдвига уже не различает.
SHIFT_TOL = 3e-3
OMEGA_START, OMEGA_MIN, OMEGA_MAX = 0.5, 0.10, 1.0

OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / f"{NAME}.txt"
SHAPE = OUT / f"{NAME}_shape.npy"
FRAMES = OUT / "кадры"
STORY = OUT / f"{NAME}.json"
DONE = OUT / f"{NAME}.done"
K_IN = 1.5 * (TURBULENCE_INTENSITY * V) ** 2
OMEGA_IN = K_IN**0.5 / (0.09**0.25 * TURBULENCE_LENGTH)
_rad, _a = np.radians(AZIMUTH), np.radians(ALPHA)
TILT = np.array([np.cos(_rad), np.sin(_rad), 0.0])
PIVOT = np.array([-np.sin(_rad), np.cos(_rad), 0.0])
UP = np.array([0.0, 0.0, 1.0])
ONSET = np.sin(_a) * TILT + np.cos(_a) * UP
SIDE = np.cos(_a) * TILT - np.sin(_a) * UP


def say(text=""):
    print(text, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def tilt_of(pos):
    axis = pos[:CANOPY].mean(axis=0)-pos[CANOPY]
    norm=float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm<=1e-12:
        raise ValueError('Undefined canopy axis')
    if HOLD is not None:
        return float(np.degrees(np.arctan2(axis@TILT,axis[2])))
    return float(np.degrees(np.arccos(np.clip(axis[2]/norm,-1,1))))


def sane(mesh):
    try:
        geometry_quality(mesh)
        return True
    except ValueError:
        return False


def aitken_factor(residual, last_residual, last_omega):
    """Множитель недорелаксации связки, подбирается сам.

    w_k = -w_{k-1} * <r_{k-1}, r_k - r_{k-1}> / |r_k - r_{k-1}|^2

    Это метод секущих по невязке связки: невязка считается приблизительно
    линейной по форме, и её нуль ищется по двум последним проходам.  Нужен
    из-за сильной присоединённой массы: купол лёгкий, а воздух, который он
    двигает, тяжёлый, и связка без релаксации раскачивается.
    """
    if last_residual is None:
        return OMEGA_START
    delta = residual - last_residual
    denom = float((delta * delta).sum())
    if denom < 1e-30:
        return last_omega
    raw = -last_omega * float((last_residual * delta).sum()) / denom
    return float(np.clip(raw, OMEGA_MIN, OMEGA_MAX))


def set_attitude(pos, degrees, confluence):
    """Развернуть сборку вокруг коуша так, чтобы ось купола села на заданный угол.

    Это поворот КАК ЦЕЛОГО: расстояния между узлами не меняются ни на сколько,
    значит форма остаётся ровно той, что насчитала стержневая модель, а
    меняется только привес.  Поворот минимальный (формула Родрига по двум
    векторам), поэтому сборку вокруг собственной оси не закручивает.
    """
    anchor = pos[confluence]
    rel = pos - anchor
    axis = rel[:CANOPY].mean(axis=0)
    norm=np.linalg.norm(axis)
    if not np.isfinite(norm) or norm<=1e-12:
        raise ValueError('Undefined attitude axis')
    axis = axis / norm
    tilt = np.radians(degrees)
    want = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
    cross = np.cross(axis, want)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(axis @ want,-1,1))
    if sine < 1e-12:
        if cosine>0:
            return pos.copy()
        perpendicular=np.cross(axis,np.eye(3)[np.argmin(np.abs(axis))])
        perpendicular/=np.linalg.norm(perpendicular)
        return anchor+rel@(2*np.outer(perpendicular,perpendicular)-np.eye(3)).T
    skew = np.array([[0.0, -cross[2], cross[1]],
                     [cross[2], 0.0, -cross[0]],
                     [-cross[1], cross[0], 0.0]])
    rot = np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)
    return anchor + rel @ rot.T


#: --держатель (латиницей --holder), только с --угол: реакция стенда, который
#: держит угол (у Носарева -- коуш и скользящий полюс, вып. 1735, с. 9).
#: Без неё под нагрузкой с моментом равновесия при заданном угле нет:
#: релаксация поворачивает купол к балансировке (на 40° -- 1.2 млн шагов), и
#: форма находит равновесие под нагрузкой, повёрнутой вместе с куполом, а
#: потом купол поворачивают обратно set_attitude.  С реакцией форма в
#: равновесии ровно при заданном угле.  Замер 28.10 (dynmesh/hold_check.py,
#: n24_a30 проход 2, момент 9831 Н·м): без реакции 300 тыс. шагов не
#: сошлось, ось уползла на 0.94°, узлы на 37 см; с реакцией -- 59 тыс.
#: шагов, 6 мин, ось на месте, узлы на 1.9 см.
HOLDER = "--держатель" in FLAGS or "--holder" in FLAGS


def attitude_reaction(mesh, load):
    """Нагрузка плюс реакция держателя угла: dF_i = m_i (eps x r_i),
    eps = -I^-1 T, T -- момент нагрузки около коуша, I -- тензор инерции узлов
    около коуша.  Момент dF около коуша равен -T точно, равнодействующая уходит
    в закреплённый коуш; деформации поле ускорений жёсткого тела почти не даёт.
    Та же функция -- dynmesh/structdyn.py."""
    anchor = mesh.position[mesh.confluence[0]]
    movable = np.isfinite(mesh.mass)
    r = mesh.position[movable] - anchor
    m = mesh.mass[movable]
    torque = np.cross(r, load[movable]).sum(0)
    inertia = (m[:, None, None] * ((r * r).sum(1)[:, None, None] * np.eye(3)
                                   - r[:, :, None] * r[:, None, :])).sum(0)
    eps = -np.linalg.solve(inertia, torque)
    out = load.copy()
    out[movable] += m[:, None] * np.cross(eps, r)
    return out


def refine_ball(pos):
    """Сфера измельчения вокруг купола -- вместо цилиндра вдоль оси.

    Зачем сфера.  Прежний цилиндр прибит к оси области (радиус 10 от x=0), а
    при наклоне 47° центр купола уходит на x ~ 13 м, то есть след вываливается
    из измельчения.  Измельчение следа само по себе стоит около 11 % лобовой
    силы, так что развёртка по углам на таком цилиндре мерила бы не физику, а
    съезжание области.  Сфера изотропна: купол внутри неё можно поворачивать
    как угодно, обработка остаётся той же.  Так же измельчает и ТЕСИС.

    Радиус 11 м подобран по цене: объём 5575 м3 против 4932 м3 у цилиндра, то
    есть +13 % к измельчённой зоне.  Накрывает купол (радиус ~6.2 м), около
    11 м следа за ним и верхние две трети строп -- конус строп работает как
    плохо проницаемый экран и снижает давление внутри купола (Носарев, с. 28).
    """
    return pos[:CANOPY].mean(axis=0), 11.0


# ---- разбор снятых полей ---------------------------------------------------
# Чтение vtp и перенос на узлы -- в vtp.py, общем для всех инструментов.

#: --перенос=точный: сила грани делится между узлами с весами, которыми точка
#: сглаженной поверхности выражается через узлы (meshmorph.consistent_load).
#: Сохраняет и силу, и момент тождественно.  Замер 28.10 на n24_a25/30/40:
#: прежний перенос ошибается в m_z на 0.00002...0.00007, то есть по
#: интегралам он хорош, но узловые силы расходятся с точными на 14 %, а форма
#: зависит именно от них.
EXACT_LOAD = FLAGS.get("--перенос") == "точный"
if EXACT_LOAD:
    import meshmorph


def nodal_load(mesh, folder):
    """Узловые силы при скорости и плотности этого случая."""
    if not EXACT_LOAD:
        return vtp.nodal_load(mesh, folder, q=Q, kin=KIN)
    centre, area, unit, dcp = vtp.delta_cp(folder, KIN)
    face_force = (dcp * area * Q)[:, None] * unit
    load, _ = meshmorph.consistent_load(mesh, centre, face_force)
    load[mesh.confluence[0]] = 0.0
    return load, face_force.sum(axis=0), dcp


# ---- постановка обтекания --------------------------------------------------
# Всё ниже восстановлено из словарей расчётной папки D:\cfd\run\queue, то есть
# из того, чем посчитаны имеющиеся результаты, а не по памяти.

def head(cls, obj, loc=None):
    place = f'\n    location    "{loc}";' if loc else ""
    return (f"FoamFile\n{{\n    version 2.0;\n    format ascii;\n"
            f"    class {cls};{place}\n    object {obj};\n}}\n")


def write(rel, text):
    path = CASE / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8").replace(b"\r\n", b"\n"))


def static_dicts(ball=None):
    """Словари расчёта.  ball -- (центр, радиус) сферы измельчения; без него
    остаётся прежний цилиндр вдоль оси, чтобы уже посчитанные свободные случаи
    не разъехались с новыми."""
    if ball is None:
        shape = ("    wake { type searchableCylinder; point1 (0 0 12.0);\n"
                 "           point2 (0 0 27.7); radius 10.0; }")
    else:
        centre, radius = ball
        shape = (f"    wake {{ type searchableSphere;\n"
                 f"           centre ({centre[0]:.4f} {centre[1]:.4f} "
                 f"{centre[2]:.4f}); radius {radius:.4f}; }}")
        if WAKE:
            # След -- вдоль ПОТОКА (+z), а не вдоль оси купола: поток у нас
            # всегда по +z, так что цилиндр не надо поворачивать с углом.
            # LAVA, AERO Suite разрешают след на несколько диаметров (AMR);
            # As'ad: нестационарный след появляется только на мелкой сетке.
            tail = centre[2] + WAKE * D0
            shape += (f"\n    wakeTube {{ type searchableCylinder;\n"
                      f"           point1 ({centre[0]:.4f} {centre[1]:.4f} {centre[2]:.4f});"
                      f" point2 ({centre[0]:.4f} {centre[1]:.4f} {tail:.4f});"
                      f" radius {0.42 * D0:.4f}; }}")
    write("system/blockMeshDict", head("dictionary", "blockMeshDict") + """
scale 1;
vertices ( (-80 -80 -48) (80 -80 -48) (80 80 -48) (-80 80 -48)
           (-80 -80 208) (80 -80 208) (80 80 208) (-80 80 208) );
blocks ( hex (0 1 2 3 4 5 6 7) (40 40 64) simpleGrading (1 1 1) );
edges ();
boundary ( farfield { type patch; faces (
    (0 3 2 1) (4 5 6 7) (0 1 5 4) (2 3 7 6) (1 2 6 5) (0 4 7 3) ); } );
""")
    write("system/snappyHexMeshDict",
          head("dictionary", "snappyHexMeshDict") + f"""
castellatedMesh true;  snap true;  addLayers false;
geometry
{{
    canopy.stl {{ type triSurfaceMesh; name canopy; }}
    nearField {{ type searchableBox; min (-26 -26 -10); max (26 26 75); }}
{shape}
}}
castellatedMeshControls""" + """
{
    maxLocalCells 6000000;  maxGlobalCells 12000000;
    minRefinementCells 10;  maxLoadUnbalance 0.10;  nCellsBetweenLevels 3;
    features ();
    refinementSurfaces
    { canopy { level (3 4); faceZone canopyFaces; faceType baffle; } }
    resolveFeatureAngle 30;
    refinementRegions
    {
        nearField { mode inside; levels ((1e15 1)); }
        wake { mode inside; levels ((1e15 4)); }
WAKE_TUBE    }
    locationInMesh (0 0 -36);
    allowFreeStandingZoneFaces true;
}
snapControls
{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
  nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false;
  multiRegionFeatureSnap false; }
addLayersControls { relativeSizes true; layers {} expansionRatio 1.2;
  finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60;
  nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3;
  nSmoothThickness 10; maxFaceThicknessRatio 0.5;
  maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90;
  nBufferCellsNoExtrude 0; nLayerIter 50; }
meshQualityControls { #includeEtc "caseDicts/meshQualityDict"
  nSmoothScale 4; errorReduction 0.75; }
writeFlags ( scalarLevels );  mergeTolerance 1e-6;
""".replace("WAKE_TUBE", "        wakeTube { mode inside; levels ((1e15 3)); }\n"
            if WAKE and ball is not None else ""))
    write("constant/transportProperties",
          head("dictionary", "transportProperties", "constant")
          + f"\ntransportModel Newtonian;\nnu {NU};\n")
    write("constant/turbulenceProperties",
          head("dictionary", "turbulenceProperties", "constant")
          + "\nsimulationType RAS;\nRAS { RASModel kOmegaSST; turbulence on;"
            " printCoeffs on; }\n")
    write("system/fvSchemes", head("dictionary", "fvSchemes", "system") + """
ddtSchemes { default steadyState; }
gradSchemes { default cellLimited Gauss linear 1; }
divSchemes { default none; div(phi,U) bounded Gauss linearUpwind grad(U);
  div(phi,k) bounded Gauss limitedLinear 1;
  div(phi,omega) bounded Gauss limitedLinear 1;
  div((nuEff*dev2(T(grad(U))))) Gauss linear; }
laplacianSchemes { default Gauss linear limited corrected 0.33; }
interpolationSchemes { default linear; }
snGradSchemes { default limited corrected 0.33; }
wallDist { method meshWave; }
""")
    write("system/fvSolution", head("dictionary", "fvSolution", "system") + """
solvers
{
    p { solver GAMG; tolerance 1e-6; relTol 0.01; smoother GaussSeidel; }
    "(U|k|omega)" { solver smoothSolver; smoother symGaussSeidel;
                    tolerance 1e-8; relTol 0.1; }
}
SIMPLE { nNonOrthogonalCorrectors 0; consistent yes; }
relaxationFactors { equations { ".*" 0.9; } }
""")
    write("system/controlDict", head("dictionary", "controlDict", "system")
          + f"""
application simpleFoam;  startFrom startTime;  startTime 0;
stopAt endTime;  endTime {ITERS};  deltaT 1;
// Промежуточные шаги сохраняем ВСЕ, а не только итог: сошедшийся ответ ничего
// не говорит о том, как он получался.  На этом уже дважды теряли время --
// расчёт, который не сходился, а полз по 0.9° за проход, и сгущение следа,
// съехавшее с наклонённого купола.  Плюс поле давления -- единственный способ
// сверить наш перепад с эпюрами Носарева (фиг. 19-22).
writeControl timeStep;  writeInterval {max(ITERS // 8, 1)};  purgeWrite 0;
writeFormat binary;  writePrecision 8;  writeCompression off;
timeFormat general;  timePrecision 6;  runTimeModifiable true;
functions
{{
    forces
    {{
        type forces;  libs (forces);
        writeControl timeStep;  writeInterval 25;
        patches ( ".*canopy.*" );  rho rhoInf;  rhoInf {RHO};
        CofR (0 0 0);
    }}
    surfacePressure
    {{
        // vtk, а не raw: нужны площади и нормали граней.
        type surfaces;  libs (sampling);
        writeControl onEnd;  surfaceFormat vtk;  fields (p);
        interpolate false;
        surfaces
        {{
            master {{ type patch; patches ( canopy );       }}
            slave  {{ type patch; patches ( canopy_slave ); }}
        }}
    }}
    cut
    {{
        // Сечение через ось купола в плоскости наклона (y = 0) со скоростью и
        // давлением.  Геометрия у vtp своя, поэтому такой срез читается и
        // рисуется без сетки -- в отличие от объёмных полей.  Несколько
        // мегабайт против трёхсот, и это ровно то, из чего делается кино.
        type surfaces;  libs (sampling);
        writeControl writeTime;  surfaceFormat vtk;
        fields (p U);  interpolate true;
        surfaces
        {{
            plane
            {{
                type cuttingPlane;  point (0 0 0);  normal (0 1 0);
                interpolate true;
            }}
        }}
    }}
    streamlines
    {{
        // Затравка -- отрезок поперёк потока НИЖЕ купола: поток идёт по +z,
        // коуш стоит в нуле, купол на 14...19.5 м.  По x берём с запасом до
        // 22 м, потому что при наклоне 45° купол уезжает туда почти целиком.
        type streamLine;  libs (fieldFunctionObjects);
        writeControl writeTime;  setFormat vtk;
        direction forward;  fields (p U);  U U;
        lifeTime 12000;  nSubCycle 5;  cloud tracks;
        seedSampleSet
        {{
            type uniform;  axis x;
            start (-12 0 -5);  end (22 0 -5);  nPoints 60;
        }}
    }}
}}
""")
    write("system/decomposeParDict",
          head("dictionary", "decomposeParDict", "system")
          + f"\nnumberOfSubdomains {CORES};\nmethod scotch;\n")
    (CASE / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)


def write_fields(vec):
    """Поля начального приближения.  farfield -- свободный поток на всех
    гранях области: купол стоит в её середине, и заранее неизвестно, где
    втекает, а где вытекает."""
    write("0/U", head("volVectorField", "U", "0") + f"""
dimensions [0 1 -1 0 0 0 0];
internalField uniform {vec};
boundaryField
{{
    farfield {{ type freestreamVelocity; freestreamValue uniform {vec}; }}
    ".*canopy.*" {{ type noSlip; }}
}}
""")
    write("0/p", head("volScalarField", "p", "0") + """
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    farfield { type freestreamPressure; freestreamValue uniform 0; }
    ".*canopy.*" { type zeroGradient; }
}
""")
    write("0/k", head("volScalarField", "k", "0") + f"""
dimensions [0 2 -2 0 0 0 0];
internalField uniform {K_IN:.6g};
boundaryField
{{
    farfield {{ type freestream; freestreamValue uniform {K_IN:.6g}; }}
    ".*canopy.*" {{ type kqRWallFunction; value uniform {K_IN:.6g}; }}
}}
""")
    write("0/omega", head("volScalarField", "omega", "0") + f"""
dimensions [0 0 -1 0 0 0 0];
internalField uniform {OMEGA_IN:.6g};
boundaryField
{{
    farfield {{ type freestream; freestreamValue uniform {OMEGA_IN:.6g}; }}
    ".*canopy.*" {{ type omegaWallFunction; value uniform {OMEGA_IN:.6g}; }}
}}
""")
    write("0/nut", head("volScalarField", "nut", "0") + """
dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;
boundaryField
{
    farfield { type calculated; value uniform 0; }
    ".*canopy.*" { type nutUSpaldingWallFunction; value uniform 0; }
}
""")


WHERE = "/mnt/d/" + str(CASE).replace("D:\\", "").replace("\\", "/")
#: Куда складывать сохранённые поля -- наружу от расчётной папки, потому что
#: она чистится в начале каждого прохода.
#: Имена латиницей: путь уходит в bash внутри WSL, и кириллица там -- лишний
#: риск на ровном месте.
FIELDS = Path(r"D:\cfd\fields")
KEEP = "/mnt/d/" + str(FIELDS).replace("D:\\", "").replace("\\", "/")
RUN = f"""
source /usr/lib/openfoam/openfoam2412/etc/bashrc
KEEP_ROOT="{KEEP}/{NAME}/pass$PASS"
cd "{WHERE}" || exit 17
# После прохода на подвижной сетке controlDict правлен -- возвращаем исходный.
[ -f system/controlDict.base ] && cp system/controlDict.base system/controlDict
rm -rf 0.* [1-9]* processor* postProcessing constant/polyMesh dynamicCode
blockMesh                               > log.blockMesh 2>&1     || exit 11
snappyHexMesh -overwrite                > log.snappyHexMesh 2>&1 || exit 12
# Отрезанные области выкидываем ДО разбиения.
#
# ЗАЧЕМ.  На 60 градусах купол-баффл при прилипании замуровал РОВНО ОДНУ
# ячейку из 514 812, и она осталась без единой грани с потоком.  Давление в
# такой ячейке ничем не определено: строка матрицы вырождена, на грубом
# уровне GAMG диагональ обнуляется, DICPreconditioner делит на ноль, ловушка
# FPE валит решатель на первом же шаге по давлению.  Семь часов счёта из-за
# одной ячейки, и snappyHexMesh при этом пишет "Finished meshing without any
# errors" -- связность он не проверяет.
#
# largestOnly оставляет только главную область.  Если сетка и так односвязна,
# команда ничего не меняет, так что ставить её можно всегда.
splitMeshRegions -largestOnly -overwrite > log.split 2>&1        || exit 15
decomposePar -force                     > log.decomposePar 2>&1  || exit 13
mpirun -np {CORES} simpleFoam -parallel > log.simpleFoam 2>&1    || exit 14
reconstructPar                          > log.reconstruct 2>&1 || exit 18
# Поля складываем НАРУЖУ: расчётная папка чистится в начале следующего прохода,
# и без этого промежуточные шаги жили бы ровно до него.
KEEP="$KEEP_ROOT"
mkdir -p "$KEEP"
for step in [0-9]*; do
    if [ -d "$step" ] && [ "$step" != "0" ]; then cp -r "$step" "$KEEP/" || exit 19; fi
done
# Сетку -- обязательно: без неё поля просто столбцы чисел без геометрии.
# Стоит 65 МБ на проход при 323 МБ полей, то есть пятую часть сверх.
cp -r constant/polyMesh "$KEEP/" || exit 19
# Всю postProcessing целиком, а не по угаданным подпутям: линии тока лежат в
# sets/streamlines, а не в streamlines, и прежняя копия молча промахивалась.
cp -r postProcessing "$KEEP/" || exit 19
echo KEPT=$(du -sm "$KEEP" 2>/dev/null | cut -f1)
echo CELLS=$(grep -oP 'cells:\\s*\\K[0-9]+' log.snappyHexMesh | tail -1)
"""
#: Проход на подвижной сетке: деформация прошлой сетки + продолжение счёта.
#: 0/pointDisplacement к этому моменту уже записан (meshmorph.write_displacement).
MORPH_RUN = f"""
source /usr/lib/openfoam/openfoam2412/etc/bashrc
KEEP_ROOT="{KEEP}/{NAME}/pass$PASS"
cd "{WHERE}" || exit 17
LAST=$(ls -d [1-9]* 2>/dev/null | sort -n | tail -1)
[ -n "$LAST" ] && [ -f constant/polyMesh/points ] || exit 16
for d in [1-9]*; do [ "$d" != "$LAST" ] && rm -rf "$d"; done
rm -rf processor* postProcessing dynamicCode "$LAST/phi" "$LAST/uniform"
[ -f system/controlDict.base ] || cp system/controlDict system/controlDict.base
cat > constant/dynamicMeshDict <<'EOF'
FoamFile {{ version 2.0; format ascii; class dictionary; object dynamicMeshDict; }}
dynamicFvMesh   dynamicMotionSolverFvMesh;
motionSolverLibs (fvMotionSolvers);
motionSolver    displacementLaplacian;
diffusivity     quadratic inverseDistance (canopy canopy_slave);
EOF
foamDictionary system/fvSolution -entry solvers/cellDisplacement \\
    -set "{{ solver GAMG; smoother GaussSeidel; tolerance 1e-9; relTol 0; }}" > /dev/null 2>&1
cp system/controlDict.base system/controlDict
# moveMesh берёт смещения из начального времени -- 0/pointDisplacement.
foamDictionary system/controlDict -entry startTime -set 0 > /dev/null
foamDictionary system/controlDict -entry endTime -set 1 > /dev/null
foamDictionary system/controlDict -entry writeInterval -set 1 > /dev/null
foamDictionary system/controlDict -entry functions -remove > /dev/null
moveMesh                                > log.moveMesh 2>&1      || exit 16
checkMesh -time 1                       > log.checkMesh.morph 2>&1
grep -qE 'negative|Error in face pyramids|incorrectly oriented' log.checkMesh.morph && exit 16
SKEW=$(grep -oP 'Max skewness = \\K[0-9.]+' log.checkMesh.morph | head -1)
awk -v s="${{SKEW:-0}}" 'BEGIN {{ exit !(s < {MAX_SKEW}) }}'      || exit 16
echo SKEW=$SKEW
cp 1/polyMesh/points constant/polyMesh/points
rm -rf 1 0/pointDisplacement constant/dynamicMeshDict
cp system/controlDict.base system/controlDict
foamDictionary system/controlDict -entry startTime -set $LAST > /dev/null
foamDictionary system/controlDict -entry endTime -set $((LAST + {MORPH or 0})) > /dev/null
foamDictionary system/controlDict -entry writeInterval -set {max((MORPH or 4) // 4, 1)} > /dev/null
decomposePar -force -latestTime         > log.decomposePar 2>&1  || exit 13
mpirun -np {CORES} simpleFoam -parallel > log.simpleFoam 2>&1    || exit 14
reconstructPar -latestTime              > log.reconstruct 2>&1 || exit 18
KEEP="$KEEP_ROOT"
mkdir -p "$KEEP"
for step in [0-9]*; do
    if [ -d "$step" ] && [ "$step" != "0" ] && [ "$step" != "$LAST" ]; then cp -r "$step" "$KEEP/" || exit 19; fi
done
cp -r constant/polyMesh "$KEEP/" || exit 19
cp -r postProcessing "$KEEP/" || exit 19
echo KEPT=$(du -sm "$KEEP" 2>/dev/null | cut -f1)
echo CELLS=$(grep -oP 'cells:\\s*\\K[0-9]+' log.checkMesh.morph | head -1)
"""
WHY = {11: "blockMesh", 12: "snappyHexMesh", 13: "decomposePar",
       14: "simpleFoam", 15: "splitMeshRegions", 16: "деформация сетки",
       18: "reconstructPar",19: "архивирование"}


def run_cfd(step, script=RUN):
    path = Path(r"C:\Users\vladr\AppData\Local\Temp\coupled_run.sh")
    path.write_bytes(script.encode("utf-8").replace(b"\r\n", b"\n"))
    done = subprocess.run(
        ["wsl", "-e", "bash", "-lc",
         f"export PASS={step:02d}; "
         "tr -d '\\r' < /mnt/c/Users/vladr/AppData/Local/Temp/coupled_run.sh"
         " > /tmp/coupled_run.sh; bash /tmp/coupled_run.sh"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    cells, kept = 0, 0
    for line in (done.stdout or "").splitlines():
        if line.startswith("CELLS="):
            cells = int(line.split("=")[1] or 0)
        elif line.startswith("KEPT="):
            kept = int(line.split("=")[1] or 0)
        elif line.startswith("SKEW="):
            say(f"      перекос после деформации {line.split('=')[1]}")
    if kept:
        say(f"      сохранено полей: {kept:,} МБ")
    root = CASE / "postProcessing" / "surfacePressure"
    folder = None
    if root.exists():
        steps = sorted([d for d in root.iterdir()
                        if d.name.replace(".", "").isdigit()],
                       key=lambda d: float(d.name))
        folder = steps[-1] if steps else None
    return done.returncode, cells, folder


def why_failed(code):
    return f"упал {WHY.get(code, 'расчёт')} (код {code})"


# ---- случай ----------------------------------------------------------------
say(f"\n{'=' * 96}")
say(f"=== {NAME}: ход {STROKE} м, затянуто {PULLED} из {GORES}, "
    f"{ITERS} итераций, {time.strftime('%d.%m %H:%M:%S')} ===")
say(f"узлов по окружности {GORES}, шаг строп {STRIDE} "
    f"(строп {GORES // STRIDE}), колец {RINGS}, "
    f"стропы {LINES} м (l = {LINES / D0:.3f})")
say(f"диагонали: {'нет' if SHEAR == 0 else f'доля сдвига {SHEAR}'}"
    f"; недорелаксация по Эйткену")
if MORPH:
    say(f"подвижная сетка: со второго прохода деформация сетки и +{MORPH} "
        f"итераций с полей прошлого прохода; перекос больше {MAX_SKEW} -- перестройка")
if HOLD is None:
    say(f"купол на воле, остановка при Δнаклона < {TILT_TOL}° "
        f"и моменте < {MOMENT_TOL:,.0f} Н·м два прохода подряд")
else:
    say(f"привес ДЕРЖИМ на {HOLD}°, форма свободна (стенд Носарева, с. 9); "
        f"остановка при сдвиге формы < {SHIFT_TOL:.0e} два прохода подряд")

mesh = flat_circular(D0, LINES, gores=GORES, rings=RINGS,
                     line_stride=STRIDE, shear_ratio=SHEAR)

history = json.loads(STORY.read_text(encoding="utf-8")) if STORY.exists() else []
_config=dict(D0=D0,lines=LINES,V=V,rho=RHO,nu=NU,gores=GORES,rings=RINGS,stride=STRIDE,
             shear=SHEAR,hold=HOLD,attitude=ATTITUDE,stroke=STROKE,pulled=PULLED,
             turbulence_length=TURBULENCE_LENGTH,turbulence_intensity=TURBULENCE_INTENSITY,
             flags={k:v for k,v in FLAGS.items() if k not in {'--shape-tol','--cx-tol','--load-tol','--невязка-нагрузки','--all-passes'}})
begin_run(OUT,NAME,_config,history,SHAPE,sources=source_hashes(Path(__file__).parent,FLAGS.get('--patch','')))
_run_active = True
if SHAPE.exists():
    mesh.position = np.load(SHAPE)
    say(f"продолжаю с прохода {len(history) + 1}")
elif SEED is not None and SEED.exists():
    mesh.position = np.load(SEED)
    say(f"затравка: {SEED.name}")
else:
    old = Path(r"D:\cfd\final") / f"seed_s{int(STROKE * 100):03d}_p{PULLED}.npy"
    if old.exists():
        mesh.position = np.load(old)
        say(f"затравка: {old}")
    else:
        # Надуваем СИММЕТРИЧНЫЙ купол, затяжку прикладываем ПОСЛЕ.
        #
        # Иначе беда: равномерное давление на открытой поверхности потенциала
        # не имеет, восстанавливающего момента нет, и затянутый купол при
        # наполнении разгоняется без предела.  На ходе 0.50 м это дало наклон
        # 131.65° на старте -- купол лёг почти ребром, dCp упал до 0.076.
        # Симметричное наполнение устойчиво: наклон выходит ровно 0.00°.
        say(f"затравки нет -- надуваю СИММЕТРИЧНЫЙ купол перепадом "
            f"{PRELOAD_DCP}, затяжку прикладываю после")
        inflate(mesh, uniform_pressure(PRELOAD_DCP, dynamic_pressure=Q),
                damping=0.98, tolerance=2e-3, drift_tolerance=1e-8,
                max_steps=3_000_000)
        say(f"  наполнен: наклон {tilt_of(mesh.position):.2f}°, "
            f"Dp/D0 = {mesh.projected_diameter() / D0:.4f}")

# Затяжка прикладывается к готовой форме, а не к плоскому раскрою.
if PULLED:
    mesh.rest_length[mesh.line_rods[:PULLED]] += STROKE

if HOLD is not None and not FIXED_SHAPE:
    mesh.position = set_attitude(mesh.position, HOLD, mesh.confluence[0])

say(f"старт: наклон {tilt_of(mesh.position):.2f}°, "
    f"Dp/D0 = {mesh.projected_diameter() / D0:.4f}")
geometry_quality(mesh)
say(f'Удержание угла: {ATTITUDE}; k={K_IN:g}, omega={OMEGA_IN:g}, Lтурб={TURBULENCE_LENGTH:g} м')
static_dicts(refine_ball(mesh.position) if HOLD is not None else None)

if "--только-сетка" in FLAGS:
    # Восстановить сетку под уже посчитанные поля.  Сетку начали сохранять
    # не сразу, и у первых случаев в архиве лежат одни поля -- без геометрии
    # они бесполезны.  Сетка однозначно задаётся формой, а формы сохранены,
    # так что её можно построить заново; совпадение проверяется по числу
    # ячеек, записанному в журнале того расчёта.
    say("режим ТОЛЬКО СЕТКА: строю и выхожу, решатель не запускаю")
    # Сфера сгущения ставится по ЗАТРАВКЕ (static_dicts вызван выше, до
    # цикла проходов, и внутри случая она не двигается), а поверхность -- по
    # форме нужного прохода.  Формы это разные, и подменять надо только
    # вторую: иначе сфера встанет не туда, где стояла в оригинале, и сетка
    # выйдет похожей, но не той -- на n24_a20 это дало 28 лишних ячеек из
    # полумиллиона, чего достаточно, чтобы поля не легли.
    if "--форма" in FLAGS:
        mesh.position = np.load(FLAGS["--форма"])
        say(f"  поверхность беру из {Path(FLAGS['--форма']).name}, "
            f"сферу оставляю по затравке")
    export_surface(mesh, CASE / "constant" / "triSurface" / "canopy.stl",
                   subdivide=12)
    write_fields(f"({V * ONSET[0]:.6f} {V * ONSET[1]:.6f} {V * ONSET[2]:.6f})")
    path = Path(r"C:\Users\vladr\AppData\Local\Temp\mesh_only.sh")
    path.write_bytes(f"""
source /usr/lib/openfoam/openfoam2412/etc/bashrc
cd "{WHERE}" || exit 17
rm -rf 0.* [1-9]* processor* postProcessing constant/polyMesh dynamicCode
blockMesh                > log.blockMesh 2>&1     || exit 11
snappyHexMesh -overwrite > log.snappyHexMesh 2>&1 || exit 12
echo CELLS=$(grep -oP 'cells:\\s*\\K[0-9]+' log.snappyHexMesh | tail -1)
""".encode("utf-8").replace(b"\r\n", b"\n"))
    done = subprocess.run(
        ["wsl", "-e", "bash", "-lc",
         "tr -d '\\r' < /mnt/c/Users/vladr/AppData/Local/Temp/mesh_only.sh"
         " > /tmp/mesh_only.sh; bash /tmp/mesh_only.sh"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in (done.stdout or "").splitlines():
        if line.startswith("CELLS="):
            say(f"  ячеек получилось: {int(line.split('=')[1] or 0):,}")
    say(f"  код {done.returncode}")
    sys.exit(0 if done.returncode == 0 else 1)

say(f"\n  {'пр':>2} {'ячеек':>9} {'dCp':>6} {'Dp/D0':>7} {'наклон':>7} "
    f"{'Δнакл':>7} {'ω':>5} {'невязка':>9} {'лобовая':>9} {'поперечн':>9} "
    f"{'момент':>10} {'c_x':>6} {'c_y':>7} {'m_z':>8} {'мин':>6}")

before = history[-1]["tilt"] if history else tilt_of(mesh.position)
previous = mesh.position.copy()
steady, last_residual, omega = 0, None, OMEGA_START
last_load = None
#: Форма, под которую сейчас стоит сетка в CASE.  Неизвестна до первого
#: прохода и после продолжения с прерванного случая -- тогда сетка строится.
mesh_shape = None
for step in range(len(history) + 1, PASSES + 1):
    started = time.time()
    export_surface(mesh, CASE / "constant" / "triSurface" / "canopy.stl",
                   subdivide=12)
    write_fields(f"({V * ONSET[0]:.6f} {V * ONSET[1]:.6f} {V * ONSET[2]:.6f})")
    how, iters = "snappy", ITERS
    if MORPH and mesh_shape is not None:
        say(f"      подвижная сетка: деформирую сетку прошлого прохода, "
            f"+{MORPH} итераций с её полей")
        meshmorph.write_displacement(CASE, mesh, mesh_shape, mesh.position, say=say)
        code, cells, folder = run_cfd(step, MORPH_RUN)
        how, iters = "morph", MORPH
        if code == 16:
            say("      сетка после деформации негодна -- строю заново")
            how, iters = "snappy", ITERS
            code, cells, folder = run_cfd(step)
    else:
        code, cells, folder = run_cfd(step)
    mesh_shape = mesh.position.copy() if not code else None
    if code or folder is None:
        say(f"  {step:>2}  {why_failed(code)}")
        raise RuntimeError(f'Incomplete CFD pass {step}; no result accepted (code {code})')
    cfd_validation=check_cfd(CASE,FIELDS/NAME/f'pass{step:02d}',folder.name,Q,S0,D0)

    # Preserve the exact structural input of this completed CFD pass. The
    # relaxed frame written below belongs to the NEXT pass instead.
    input_archive = FIELDS / NAME / f"pass{step:02d}"
    input_archive.mkdir(parents=True, exist_ok=True)
    np.save(input_archive / "input_shape.npy", mesh_shape)

    load, total, dcp = nodal_load(mesh, folder)
    # Невязка НАГРУЗКИ между проходами, а не только формы: момент сходится
    # позже силы и формы (на 250-й итерации бывает обратного знака), а
    # Тримарки прямо требует критерия по давлению -- «иначе случаются очень
    # плохие вещи».  При перестройке сетки сюда входит и её шум, при
    # подвижной -- только отклик на форму.
    dload = (float(np.linalg.norm(load - last_load) / np.linalg.norm(load))
             if last_load is not None else None)
    last_load = load.copy()
    if dload is not None:
        say(f"      невязка нагрузки между проходами {dload * 100:.2f} %")
    anchor = mesh.position[mesh.confluence[0]]
    torque = np.cross(mesh.position - anchor, load).sum(axis=0)
    #: Поперечник ТОЙ формы, которую видел решатель.  Силы принадлежат ей, а
    #: не той, что выйдет из релаксации ниже, и при большом отклике это две
    #: очень разные формы: на 90 градусах решатель видел наполненный купол с
    #: 0.4179, а релаксация увела форму к 0.2951.  Без этого поля строка итога
    #: молча смешивала обе, и c_x наполненного купола читался как c_x
    #: сложенного.
    ratio_seen = float(mesh.projected_diameter() / D0)

    relaxed = FIXED_SHAPE
    applied = attitude_reaction(mesh, load) if HOLD is not None and HOLDER else load
    for _ in range(0 if FIXED_SHAPE else CHUNKS):
        result = inflate(mesh, node_force=applied, damping=0.98, tolerance=2e-3,
                         drift_tolerance=1e-8, max_steps=CHUNK,
                         attitude_deg=HOLD if ATTITUDE=='inner' else None)
        if result.converged:
            relaxed = True
            break
    if not relaxed:
        # Форма не дошла до равновесия за отведённые шаги, и дальше в связку
        # уходит недорелаксированное состояние.  Замер показал, что при
        # наклоне 40 градусов этого не хватает: за 400 тысяч шагов форма не
        # сходится, а до равновесия ей ещё далеко.
        say(f"      !! форма не сошлась за {CHUNKS * CHUNK:,} шагов")
    if not sane(mesh):
        raise ValueError(f'Pass {step}: invalid or collapsed structural geometry')

    inner_reaction = (result.constraint_force.sum(0).tolist()
                      if not FIXED_SHAPE and result.constraint_force is not None else None)

    settled = mesh.position.copy()
    # Привес снимаем ДО невязки, иначе Эйткен подбирал бы множитель по
    # смеси поворота и деформации, а поворот мы всё равно уберём.
    if HOLD is not None and not FIXED_SHAPE:
        settled = set_attitude(settled, HOLD, mesh.confluence[0])
    residual = settled - previous
    omega = 0.0 if FIXED_SHAPE else aitken_factor(residual, last_residual, omega)
    mesh.position = previous + omega * residual
    if HOLD is not None and not FIXED_SHAPE:
        # Смесь двух форм с одинаковым привесом сама привес почти держит,
        # но линейная смесь поворотов точной не бывает -- поправляем.
        mesh.position = set_attitude(mesh.position, HOLD, mesh.confluence[0])
    last_residual = residual

    now = tilt_of(mesh.position)
    # СУДИМ ПО НЕВЯЗКЕ, а не по сделанному шагу.  Шаг -- это omega*residual,
    # и по нему связка объявляла себя сошедшейся в 1/omega раз рано: при
    # omega = 0.2 форма ещё уезжала на 5*SHIFT_TOL, а случай писался готовым.
    # gap -- на сколько разошлись форма от прошлого прохода и форма, которую
    # запросила нынешняя нагрузка; moved -- сколько мы на самом деле прошли.
    gap = float(np.linalg.norm(residual, axis=1).mean() / D0)
    moved = float(np.linalg.norm(mesh.position - previous, axis=1).mean() / D0)
    row = {
        "pass": step, "cells": cells, "iters": iters, "mesh": how,
        "dcp": float(dcp.mean()), "dload": dload,
        "omega": omega, "tilt": now, "dtilt": now - before,
        "gap": gap, "relaxed": relaxed, "ratio_seen": ratio_seen,
        "shift": moved, "ratio": float(mesh.projected_diameter() / D0),
        "drag": float(total @ ONSET), "side": float(total @ SIDE),
        "moment": float(torque @ PIVOT), "minutes": (time.time() - started) / 60,
        "gores": GORES, "lines": LINES, "hold": HOLD,
        "D0": D0, "S0": S0, "V": V, "rho": RHO, "nu": NU,
        "fixed_shape": FIXED_SHAPE, "fabric_coefficients": FLAGS.get('--fabric'),
        "mesh_source": FLAGS.get('--mesh-from'),
        "canopy_model": ("fabric" if "--fabric" in FLAGS else
                         "sink" if "--sink" in FLAGS else "wall"),
        "stride": STRIDE, "rings": RINGS, "ropes": GORES // STRIDE,
        "shear": SHEAR,
        "attitude_mode": ATTITUDE,"shape_tolerance": SHIFT_TOL,
        "turbulence_length_m": TURBULENCE_LENGTH,"turbulence_intensity": TURBULENCE_INTENSITY,
        "inner_residual": None if FIXED_SHAPE else float(result.residual),
        "inner_constraint_resultant_N": inner_reaction,
        "status_scope": "outer coupling criteria, not full CFD/grid convergence",
        **cfd_validation,
    }
    # Безразмерные -- в тех же осях и с той же опорной площадью, что у
    # Носарева: c = сила/(q*F_п), m_z = момент/(q*F_п*D_п), момент около коуша.
    row["cx"] = row["drag"] / (Q * S0)
    row["cx_change"] = abs(row["cx"]-history[-1]["cx"]) if history else None
    row["cx_tol"] = CX_TOL
    row["cy"] = row["side"] / (Q * S0)
    row["mz"] = row["moment"] / (Q * S0 * D0)
    history.append(row)
    say(f"  {step:>2} {cells:>9,} {row['dcp']:6.3f} {row['ratio']:7.4f} "
        f"{now:7.2f} {row['dtilt']:+7.2f} {omega:5.2f} {gap:9.2e} "
        f"{row['drag']:9.1f} {row['side']:9.1f} {row['moment']:10.1f} "
        f"{row['cx']:6.3f} {row['cy']:+7.3f} {row['mz']:+8.4f} "
        f"{row['minutes']:6.1f}")

    np.save(SHAPE, mesh.position)
    # Форма КАЖДОГО прохода, а не только последняя: файл 3.6 КБ, а без них
    # потом нечем показать, как купол двигался -- на этом уже споткнулись.
    FRAMES.mkdir(parents=True, exist_ok=True)
    np.save(FRAMES / f"{NAME}_проход{step:02d}.npy", mesh.position)
    atomic_json(STORY,history)
    if HOLD is None:
        quiet = (abs(row["dtilt"]) < TILT_TOL
                 and abs(row["moment"]) < MOMENT_TOL)
    else:
        # Привес задан, момент мы меряем -- судить можно только по форме.
        quiet = gap < SHIFT_TOL
    if LOAD_TOL is not None:
        quiet = quiet and dload is not None and dload < LOAD_TOL
    # Недорелаксированная форма не считается установившейся ни в одном режиме:
    # внутренний счёт оборвали на потолке, и равновесия конструкция не нашла.
    quiet = quiet and relaxed
    if HOLD is not None:
        quiet = quiet and row['cx_change'] is not None and row['cx_change']<CX_TOL
    steady = steady + 1 if quiet else 0
    before, previous = now, mesh.position.copy()
    if steady >= 2:
        if HOLD is None:
            say(f"\n  встало: два прохода подряд наклон меньше {TILT_TOL}° "
                f"И момент меньше {MOMENT_TOL:,.0f} Н·м")
        else:
            say(f"\n  форма встала: два прохода подряд сдвиг меньше "
                f"{SHIFT_TOL:.0e} от D0")
        break

#: Сошёлся ли случай на самом деле.  Раньше .done писался по последней строке
#: истории безотносительно того, чем всё кончилось, и аварийный случай выглядел
#: сошедшимся: на 45° третий проход упал, а итог записался как готовый.
settled = False
if history:
    row = history[-1]
    # У старых историй поля gap нет -- там судили по затухшему шагу; для них
    # откатываемся на shift, чтобы не падать, но новые пишутся уже по невязке.
    left = row.get("gap", row["shift"])
    if HOLD is None:
        settled = (abs(row["dtilt"]) < TILT_TOL
                   and abs(row["moment"]) < MOMENT_TOL)
    else:
        settled = steady >= 2
    settled = settled and row.get("relaxed", True)

if history:
    last = history[-1]
    say(f"\n{'=' * 96}")
    say(f"{NAME}: наклон {last['tilt']:.2f}°, Dp/D0 = {last['ratio']:.4f}, "
        f"полотнищ {GORES}, l = {LINES / D0:.3f}")
    seen = last.get("ratio_seen")
    if seen is not None and abs(seen - last["ratio"]) > 0.01:
        say(f"  ВНИМАНИЕ: коэффициенты ниже принадлежат форме, которую видел "
            f"решатель (Dp/D0 = {seen:.4f}),")
        say(f"  а не той, что вышла из релаксации (Dp/D0 = "
            f"{last['ratio']:.4f}).  Нужен ещё проход.")
    say(f"  лобовая {last['drag']:,.1f} Н  (c_x = {last['cx']:.3f})")
    say(f"  поперечная {last['side']:,.1f} Н  (c_y = {last['cy']:+.3f}), "
        f"качество {last['side'] / last['drag']:.4f}")
    say(f"  момент {last['moment']:,.1f} Н·м  (m_z = {last['mz']:+.4f})")
    say(f"  проходов {len(history)}, "
        f"всего {sum(r['minutes'] for r in history) / 60:.1f} ч")
    last["settled"] = settled
    if FIXED_SHAPE:
        last['settled'] = None
        mark_status(OUT,NAME,'fixed',last)
        say('  Фиксированная форма: один CFD-проход завершён; сходимость CFD проверяется отдельно')
    elif settled:
        mark_status(OUT,NAME,'done',last)
    else:
        # Не сошёлся -- .done не пишем вовсе, иначе очередь и сборка отчёта
        # примут это за результат.  Кладём рядом .НЕСОШЁЛСЯ, чтобы было видно,
        # что случай считался и чем кончился.
        mark_status(OUT,NAME,'НЕСОШЁЛСЯ',last)
        # Причин две, и они разные: связка не устоялась ИЛИ форма не дошла до
        # равновесия внутри прохода.  Раньше сообщение всегда винило сдвиг, и
        # на n24_a40v это ввело в заблуждение: сдвиг там был 2.34e-03 ПРИ
        # пороге 3e-03, то есть в норме, а не сошлась именно релаксация.
        why = []
        if HOLD is not None and (last['cx_change'] is None or last['cx_change']>=CX_TOL):
            why.append(f"Cx change {last['cx_change']}; tolerance {CX_TOL}")
        if HOLD is not None and left >= SHIFT_TOL:
            why.append(f"сдвиг формы {left:.2e} при пороге {SHIFT_TOL:.0e}")
        if HOLD is None and not (abs(last["dtilt"]) < TILT_TOL
                                 and abs(last["moment"]) < MOMENT_TOL):
            why.append(f"наклон {last['dtilt']:+.2f}° и момент "
                       f"{last['moment']:,.0f} Н·м")
        if not last.get("relaxed", True):
            why.append(f"форма не дошла до равновесия за {CHUNKS * CHUNK:,} "
                       f"шагов релаксации")
        say("  !! НЕ СОШЁЛСЯ: " + "; ".join(why or ["причина не определена"]))
