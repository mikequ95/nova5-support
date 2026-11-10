# SPDX-License-Identifier: MulanPSL-2.0
"""Generate a vector A4 chessboard. Print the PDF at 100% / actual size.

The 9 x 6 dimensions refer to INTERNAL CORNERS, so there are 10 x 7 squares.
After printing, measure multiple squares; enter the measured square size in
the calibration config. The TOP LEFT label is a human orientation reference,
not a machine-readable fiducial or an automatically detected origin marker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


def make_board(output: Path, columns: int = 9, rows: int = 6,
               square_mm: float = 25.0) -> Path:
    if columns < 2 or rows < 2 or square_mm <= 0:
        raise ValueError("Inner-corner counts must be >= 2 and square size > 0.")
    width, height = landscape(A4)
    board_width = (columns + 1) * square_mm * mm
    board_height = (rows + 1) * square_mm * mm
    left, bottom = (width - board_width) / 2, (height - board_height) / 2
    if min(left, bottom) < 10 * mm:
        raise ValueError("Board must leave at least 10 mm of white margin on A4.")

    output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output), pagesize=(width, height), pageCompression=1)
    c.setTitle(f"Chessboard {columns} x {rows} inner corners - {square_mm:g} mm")
    c.setAuthor("Eye-in-hand RGB calibration project")
    c.setSubject("Print at 100 percent actual size; disable fit-to-page scaling")
    c.setFillColorRGB(1, 1, 1)
    c.rect(0, 0, width, height, fill=1, stroke=0)
    c.setFillColorRGB(0, 0, 0)
    for row in range(rows + 1):
        for col in range(columns + 1):
            if (row + col) % 2 == 0:
                # Rows are counted from the top; top-left square is black.
                x = left + col * square_mm * mm
                y = bottom + (rows - row) * square_mm * mm
                c.rect(x, y, square_mm * mm, square_mm * mm, fill=1, stroke=0)

    top = bottom + board_height
    c.setFont("Helvetica", 7)
    c.drawString(left, top + 4 * mm, "TOP LEFT (human reference)")
    c.drawRightString(
        left + board_width, top + 4 * mm,
        f"INNER {columns} x {rows} | SQUARE {square_mm:g} mm | PRINT 100% / ACTUAL SIZE",
    )
    # Keep the ruler and text in the white margin, never touching the board.
    ruler_y = bottom - 6 * mm
    ruler_x = (width - 100 * mm) / 2
    c.setLineWidth(0.4)
    c.line(ruler_x, ruler_y, ruler_x + 100 * mm, ruler_y)
    for delta in (0, 25, 50, 75, 100):
        x = ruler_x + delta * mm
        c.line(x, ruler_y - 1.1 * mm, x, ruler_y + 1.1 * mm)
    c.setFont("Helvetica", 6.5)
    c.drawCentredString(width / 2, ruler_y - 4 * mm,
                       "100 mm - verify with a ruler after printing; disable page scaling")
    c.showPage()
    c.save()

    metadata = {
        "inner_corners": [columns, rows],
        "squares": [columns + 1, rows + 1],
        "square_size_mm": square_mm,
        "page_size_mm": [297, 210],
        "board_size_mm": [(columns + 1) * square_mm, (rows + 1) * square_mm],
        "white_margin_mm": {"left": left / mm, "top": bottom / mm},
        "top_left_square": "black",
        "origin_label": "human reference only; not automatically detected",
        "print_instruction": "A4 landscape, actual size / 100%, no fit-to-page",
        "measurement_instruction": "Measure printed squares and use actual square size in calibration config.",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("boards/checkerboard_A4_9x6_25mm.pdf"))
    parser.add_argument("--columns", type=int, default=9, help="Number of internal corners across")
    parser.add_argument("--rows", type=int, default=6, help="Number of internal corners down")
    parser.add_argument("--square-mm", type=float, default=25.0)
    args = parser.parse_args()
    print(make_board(args.output, args.columns, args.rows, args.square_mm))


if __name__ == "__main__":
    main()
