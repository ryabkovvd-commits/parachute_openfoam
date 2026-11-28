"""cs.py patch for displacement control of numbered suspension lines."""
# Apply after nos6,cords. Numbering follows the undeformed material skirt, from +X.
import math

selection = FLAG.get("--line-ids", "")
ids = tuple(int(item) for item in selection.split(",")) if selection else ()
fraction = float(FLAG.get("--take-up", "0"))
if not math.isfinite(fraction) or not 0 <= fraction < 1:
    raise ValueError("--take-up must be a finite fraction of nominal length in [0, 1)")
if len(ids) != len(set(ids)) or any(index < 1 for index in ids):
    raise ValueError("--line-ids requires unique one-based line numbers")
if bool(ids) != bool(fraction):
    raise ValueError("Specify both a nonempty line group and positive take-up, or neither")
sub("history = json.loads(STORY", f"""
controlled_ids = {ids!r}
if STROKE or PULLED or (FIXED_SHAPE and controlled_ids):
    raise ValueError("Sector control requires zero legacy stroke/count and deformable controlled shapes")
if controlled_ids and max(controlled_ids) > len(mesh.line_rods):
    raise ValueError("A selected suspension line does not exist")
history = json.loads(STORY""")
sub("    mesh.rest_length[mesh.line_rods[:PULLED]] += STROKE",
    f"""    mesh.rest_length[mesh.line_rods[:PULLED]] += STROKE
if controlled_ids:
    controlled_rods = mesh.line_rods[np.asarray(controlled_ids, dtype=int) - 1]
    mesh.rest_length[controlled_rods] -= LINES * {fraction!r}
    say(f"Controlled lines {{controlled_ids}}, take-up {{LINES * {fraction!r}:.6f}} m")""")

sub("    history.append(row)", """    row["aero_force_xyz_N"] = total.tolist()
    row["aero_moment_xyz_Nm"] = torque.tolist()
    row["controlled_line_ids"] = list(controlled_ids)
    if not FIXED_SHAPE:
        row["settled_line_tensions_N"] = result.tension[mesh.line_rods].tolist()
    history.append(row)""")

