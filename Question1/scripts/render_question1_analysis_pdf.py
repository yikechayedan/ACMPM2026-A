"""Render the Question 1 markdown report to PDF with a Chinese font."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


REPORT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "xgboost_ensemble"
INPUT_MD = REPORT_DIR / "question1_analysis.md"
OUTPUT_PDF = REPORT_DIR / "question1_analysis.pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT_BOLD_PATH = Path(r"C:\Windows\Fonts\simsunb.ttf")


def register_fonts() -> tuple[str, str]:
    regular = "NotoSansSC"
    bold = "NotoSansSCBold"
    pdfmetrics.registerFont(TTFont(regular, str(FONT_PATH)))
    pdfmetrics.registerFont(TTFont(bold, str(FONT_BOLD_PATH)))
    return regular, bold


def strip_inline_code(text: str) -> str:
    return text.replace("`", "")


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    table_lines = []
    index = start
    while index < len(lines) and lines[index].strip().startswith("|"):
        table_lines.append(lines[index].strip())
        index += 1

    rows = []
    for line in table_lines:
        cells = [strip_inline_code(cell.strip()) for cell in line.strip("|").split("|")]
        if cells and all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append(cells)
    return rows, index


def build_table(rows: list[list[str]], font_name: str, bold_font: str) -> Table:
    table = Table(rows, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTNAME", (0, 0), (-1, 0), bold_font),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAEAEA")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#999999")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def render_pdf() -> None:
    font_name, bold_font = register_fonts()
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ChineseTitle",
            fontName=bold_font,
            fontSize=18,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ChineseHeading",
            fontName=bold_font,
            fontSize=13,
            leading=18,
            spaceBefore=12,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ChineseBody",
            fontName=font_name,
            fontSize=10,
            leading=16,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ChineseCode",
            fontName=font_name,
            fontSize=8.5,
            leading=13,
            leftIndent=10,
            textColor=colors.HexColor("#333333"),
            backColor=colors.HexColor("#F5F5F5"),
            spaceAfter=8,
        )
    )

    lines = INPUT_MD.read_text(encoding="utf-8").splitlines()
    story = []
    index = 0
    while index < len(lines):
        raw = lines[index]
        line = raw.strip()
        if not line:
            index += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(strip_inline_code(line[2:]), styles["ChineseTitle"]))
        elif line.startswith("## "):
            story.append(Paragraph(strip_inline_code(line[3:]), styles["ChineseHeading"]))
        elif line.startswith("|"):
            rows, index = parse_table(lines, index)
            story.append(build_table(rows, font_name, bold_font))
            story.append(Spacer(1, 0.25 * cm))
            continue
        elif line.startswith("`") and line.endswith("`"):
            story.append(Paragraph(strip_inline_code(line), styles["ChineseCode"]))
        elif line[:2].replace(".", "").isdigit() and ". " in line[:4]:
            items = []
            while index < len(lines):
                item_line = lines[index].strip()
                if not (item_line[:2].replace(".", "").isdigit() and ". " in item_line[:4]):
                    break
                item_text = item_line.split(". ", 1)[1]
                items.append(ListItem(Paragraph(strip_inline_code(item_text), styles["ChineseBody"])))
                index += 1
            story.append(ListFlowable(items, bulletType="1", start="1", leftIndent=18))
            continue
        else:
            story.append(Paragraph(strip_inline_code(line), styles["ChineseBody"]))
        index += 1

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="问题一模型分析报告",
    )
    doc.build(story)


if __name__ == "__main__":
    render_pdf()
