"""Turn PDF tables into one self-contained line per row.

Plain text extraction flattens a table into loose cells, so the model cannot tell which
number belongs to which row. Here every row keeps its column names:

    Offence: Punishment for murder | IPC section: 302 | BNS section: 103
"""


def _clean(cell) -> str:
    return " ".join(str(cell or "").split())


def rows_to_text(rows) -> str:
    """rows: list of rows (list of cell strings) with the header row first."""
    if not rows or len(rows) < 2:
        return ""
    header = [_clean(c) for c in rows[0]]
    lines = []
    for row in rows[1:]:
        cells = [_clean(c) for c in row]
        if not any(cells):
            continue
        parts = [f"{h}: {v}" if h and v else (v or "") for h, v in zip(header, cells)]
        line = " | ".join(p for p in parts if p)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _overlaps(block, bbox) -> bool:
    return not (block[2] <= bbox[0] or block[0] >= bbox[2] or block[3] <= bbox[1] or block[1] >= bbox[3])


def assemble_page_text(blocks, tables) -> str:
    """blocks: PyMuPDF page.get_text('blocks') tuples; tables: list of (bbox, rows).
    Text blocks that sit inside a table are dropped and replaced by the row lines,
    and everything is put back in reading order (top to bottom)."""
    items = []
    for b in blocks:
        if b[6] == 0 and not any(_overlaps(b, bbox) for bbox, _ in tables):
            text = b[4].strip()
            if text:
                items.append((b[1], text))
    for bbox, rows in tables:
        text = rows_to_text(rows)
        if text:
            items.append((bbox[1], text))
    return "\n".join(text for _, text in sorted(items, key=lambda item: item[0]))