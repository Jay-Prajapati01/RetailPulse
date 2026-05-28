from __future__ import annotations

from io import BytesIO
from typing import Iterable

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def build_pdf_report(title: str, summary_lines: Iterable[str], tables: list[tuple[str, pd.DataFrame]] | None = None) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=letter, title=title)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]

    for line in summary_lines:
        story.append(Paragraph(line, styles["BodyText"]))
        story.append(Spacer(1, 6))

    if tables:
        for heading, frame in tables:
            story.append(Spacer(1, 12))
            story.append(Paragraph(heading, styles["Heading2"]))
            if frame.empty:
                story.append(Paragraph("No rows available.", styles["BodyText"]))
                continue
            preview = frame.head(10).copy()
            table_data = [list(preview.columns)] + preview.astype(str).values.tolist()
            table = Table(table_data, repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#94a3b8")),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 7),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                    ]
                )
            )
            story.append(table)

    document.build(story)
    return buffer.getvalue()