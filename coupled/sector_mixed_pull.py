"""Apply an independent shortening to each selected line after sector_pull."""

if FLAG.get("--line-ids") or float(FLAG.get("--take-up", "0")):
    raise ValueError("Use either --line-take-up-bp or the uniform sector input")
selection = FLAG.get("--line-take-up-bp", "")
pairs = [tuple(map(int, item.split(":"))) for item in selection.split(",")] if selection else []
line_ids = tuple(index for index, _ in pairs)
stroke_bp = tuple(stroke for _, stroke in pairs)
if len(line_ids) != len(set(line_ids)) or any(index < 1 for index in line_ids):
    raise ValueError("--line-take-up-bp requires unique one-based line IDs")
if any(not 0 < stroke < 10000 for stroke in stroke_bp):
    raise ValueError("Line shortening must lie between 0 and 10000 basis points")

line("controlled_ids = ()", f"controlled_ids = {line_ids!r}\ncontrolled_take_up_bp = {stroke_bp!r}")
sub('    mesh.rest_length[controlled_rods] -= LINES * 0.0\n'
    '    say(f"Controlled lines {controlled_ids}, take-up {LINES * 0.0:.6f} m")',
    '    mesh.rest_length[controlled_rods] -= LINES * np.asarray(controlled_take_up_bp) / 10000\n'
    '    say(f"Line shortening in basis points: {list(zip(controlled_ids, controlled_take_up_bp))}")')
sub('    row["controlled_line_ids"] = list(controlled_ids)',
    '    row["controlled_line_ids"] = list(controlled_ids)\n'
    '    row["controlled_take_up_bp"] = list(controlled_take_up_bp)')
