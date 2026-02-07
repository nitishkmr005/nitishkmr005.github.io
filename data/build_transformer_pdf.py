from __future__ import annotations

from pathlib import Path
import hashlib
import html
import re
import subprocess
import tempfile

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Preformatted,
    Table,
    TableStyle,
    PageBreak,
    Image,
)
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from cairosvg import svg2png

ARTICLE_PATH = Path("content/posts/transformer-internals-what-changed-since-2017.md")
OUTPUT_PATH = Path("data/transformer-internals-what-changed-since-2017.pdf")
STATIC_ROOT = Path("static")
IMAGE_CACHE = Path("data/transformer-pdf-images")
MMDC_PATH = Path("node_modules/.bin/mmdc")

PALETTE = {
    "ink": colors.HexColor("#1C1C1C"),
    "muted": colors.HexColor("#4A4A4A"),
    "paper": colors.HexColor("#F6F1E7"),
    "panel": colors.HexColor("#FFFDF8"),
    "accent": colors.HexColor("#D76B38"),
    "teal": colors.HexColor("#1E5D5A"),
    "line": colors.HexColor("#E2D7C9"),
    "code": colors.HexColor("#F2EEE7"),
    "table": colors.HexColor("#F8F4ED"),
}


def strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            return parts[2].lstrip()
    return text


def inline_md(text: str) -> str:
    safe = html.escape(text)
    safe = re.sub(r"`([^`]+)`", r"<font face='Courier'>\1</font>", safe)
    safe = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", safe)
    safe = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", safe)
    return safe


def parse_table(table_lines: list[str]) -> list[list[str]]:
    rows = []
    for line in table_lines:
        parts = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(re.match(r"^:?-{2,}:?$", cell) for cell in parts):
            continue
        rows.append(parts)
    return rows


def parse_markdown_lines(text: str):
    lines = text.splitlines()
    in_code = False
    code_lang = ""
    code_lines = []
    table_lines = []
    bullets = []

    def flush_table():
        nonlocal table_lines
        if table_lines:
            yield ("table", parse_table(table_lines))
            table_lines = []

    def flush_bullets():
        nonlocal bullets
        if bullets:
            yield ("bullets", bullets)
            bullets = []

    for line in lines:
        if line.strip().startswith("```"):
            if in_code:
                content = "\n".join(code_lines)
                kind = "mermaid" if code_lang == "mermaid" else "code"
                yield (kind, content)
                code_lines = []
                code_lang = ""
                in_code = False
            else:
                in_code = True
                code_lang = line.strip().lstrip("```").strip().lower()
            continue

        if in_code:
            code_lines.append(line)
            continue

        if line.strip().startswith("|") and "|" in line:
            table_lines.append(line)
            continue

        if table_lines:
            yield from flush_table()

        list_match = re.match(r"^[-*]\s+(.*)$", line)
        if list_match:
            bullets.append(list_match.group(1))
            continue

        if bullets:
            yield from flush_bullets()

        if not line.strip():
            yield ("spacer", "")
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading_match:
            level = len(heading_match.group(1))
            yield (f"h{level}", heading_match.group(2))
            continue

        image_match = re.match(r"^!\[(.*?)\]\((.*?)\)", line)
        if image_match:
            alt = image_match.group(1) or "Figure"
            url = image_match.group(2)
            yield ("image", (alt, url))
            continue

        quote_match = re.match(r"^>\s?(.*)$", line)
        if quote_match:
            yield ("quote", quote_match.group(1))
            continue

        yield ("para", line)

    if code_lines:
        kind = "mermaid" if code_lang == "mermaid" else "code"
        yield (kind, "\n".join(code_lines))
    if table_lines:
        yield from flush_table()
    if bullets:
        yield from flush_bullets()


def make_banner(text: str, styles) -> Table:
    banner = Table([[Paragraph(inline_md(text), styles["SectionBanner"]) ]], colWidths=[6.9 * inch])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALETTE["accent"]),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return banner


def resolve_image_path(url: str) -> Path | None:
    cleaned = url.lstrip("/")
    candidate = STATIC_ROOT / cleaned
    if candidate.exists():
        return candidate
    return None


def rasterize_svg(svg_path: Path) -> Path:
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)
    output_path = IMAGE_CACHE / (svg_path.stem + ".png")
    if not output_path.exists():
        svg2png(url=str(svg_path), write_to=str(output_path), output_width=1600)
    return output_path


def make_image_flowable(alt: str, url: str, styles) -> list:
    local_path = resolve_image_path(url)
    if not local_path:
        return [Paragraph(f"Image placeholder: {inline_md(alt)}", styles["ImageCaption"])]

    if local_path.suffix.lower() == ".svg":
        local_path = rasterize_svg(local_path)

    img = ImageReader(str(local_path))
    width_px, height_px = img.getSize()
    max_width = 6.9 * inch
    max_height = 4.4 * inch
    scale = min(max_width / width_px, max_height / height_px)
    render_w = width_px * scale
    render_h = height_px * scale

    flow = [Image(str(local_path), width=render_w, height=render_h)]
    flow.append(Paragraph(inline_md(alt), styles["ImageCaption"]))
    flow.append(Spacer(1, 6))
    return flow


def make_code_block(code: str, styles) -> Table:
    block = Preformatted(code, styles["CodeBlock"])
    table = Table([[block]], colWidths=[6.9 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALETTE["code"]),
        ("BOX", (0, 0), (-1, -1), 0.8, PALETTE["line"]),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def render_mermaid(mermaid_text: str) -> Path | None:
    if not MMDC_PATH.exists():
        return None
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(mermaid_text.encode("utf-8")).hexdigest()[:12]
    out_path = IMAGE_CACHE / f"mermaid-{digest}.png"
    if out_path.exists():
        return out_path

    with tempfile.NamedTemporaryFile(suffix=".mmd", delete=False) as temp_file:
        temp_file.write(mermaid_text.encode("utf-8"))
        temp_path = Path(temp_file.name)

    try:
        subprocess.run(
            [str(MMDC_PATH), "-i", str(temp_path), "-o", str(out_path), "-b", "transparent", "-w", "1600"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    finally:
        temp_path.unlink(missing_ok=True)

    return out_path if out_path.exists() else None


def make_mermaid_flowable(mermaid_text: str, styles) -> list:
    rendered = render_mermaid(mermaid_text)
    if not rendered:
        return [Paragraph("Mermaid diagram (see article).", styles["ImageCaption"]), Spacer(1, 6)]

    img = ImageReader(str(rendered))
    width_px, height_px = img.getSize()
    max_width = 6.9 * inch
    max_height = 4.4 * inch
    scale = min(max_width / width_px, max_height / height_px)
    render_w = width_px * scale
    render_h = height_px * scale

    flow = [Image(str(rendered), width=render_w, height=render_h)]
    flow.append(Spacer(1, 6))
    return flow


def make_quote(text: str, styles) -> Table:
    box = Table([[Paragraph(inline_md(text), styles["Quote"]) ]], colWidths=[6.9 * inch])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4FBF8")),
        ("BOX", (0, 0), (-1, -1), 1, PALETTE["line"]),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return box


def make_table(table_data: list[list[str]], styles) -> Table:
    if not table_data:
        return Table([[]])
    wrapped = [[Paragraph(inline_md(cell), styles["TableCell"]) for cell in row] for row in table_data]
    table = Table(wrapped, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PALETTE["teal"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, PALETTE["line"]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def build_pdf():
    text = ARTICLE_PATH.read_text(encoding="utf-8")
    text = strip_frontmatter(text)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TitleCover",
        parent=styles["Title"],
        fontName="Times-Roman",
        fontSize=28,
        leading=32,
        textColor=PALETTE["ink"],
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="SubtitleCover",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=PALETTE["muted"],
    ))
    styles.add(ParagraphStyle(
        name="SectionBanner",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=14,
    ))
    styles.add(ParagraphStyle(
        name="Heading2Custom",
        parent=styles["Heading2"],
        fontName="Times-Roman",
        fontSize=16,
        leading=20,
        textColor=PALETTE["ink"],
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="Heading3Custom",
        parent=styles["Heading3"],
        fontName="Times-Roman",
        fontSize=13,
        leading=17,
        textColor=PALETTE["ink"],
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        textColor=PALETTE["muted"],
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="BulletCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        leftIndent=12,
        bulletIndent=0,
        spaceAfter=4,
        textColor=PALETTE["muted"],
    ))
    styles.add(ParagraphStyle(
        name="CodeBlock",
        parent=styles["BodyText"],
        fontName="Courier",
        fontSize=8.5,
        leading=11,
        leftIndent=6,
        rightIndent=6,
        spaceAfter=6,
        textColor=PALETTE["ink"],
    ))
    styles.add(ParagraphStyle(
        name="ImageCaption",
        parent=styles["BodyText"],
        fontName="Helvetica-Oblique",
        fontSize=9.5,
        leading=12,
        textColor=PALETTE["muted"],
        alignment=1,
    ))
    styles.add(ParagraphStyle(
        name="Quote",
        parent=styles["BodyText"],
        fontName="Helvetica-Oblique",
        fontSize=10.5,
        leading=14,
        textColor=PALETTE["ink"],
    ))
    styles.add(ParagraphStyle(
        name="TableCell",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=12,
        textColor=PALETTE["ink"],
    ))

    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
        title="Transformer Internals: What Actually Changed Since 2017",
        author="",
    )

    story = []

    story.append(Paragraph("Transformer Internals: What Actually Changed Since 2017", styles["TitleCover"]))
    story.append(Paragraph("Position embeddings, normalization, sparse attention, and why decoder-only won.", styles["SubtitleCover"]))
    story.append(Spacer(1, 12))

    stats = Table(
        [[
            Paragraph("<b>2017 baseline</b><br/>65M params, 512 tokens", styles["BodyCustom"]),
            Paragraph("<b>Modern scale</b><br/>100B+ params, 128K tokens", styles["BodyCustom"]),
            Paragraph("<b>Core shift</b><br/>Position + norm + attention", styles["BodyCustom"]),
        ]],
        colWidths=[2.2 * inch, 2.2 * inch, 2.2 * inch],
    )
    stats.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALETTE["panel"]),
        ("BOX", (0, 0), (-1, -1), 1, PALETTE["line"]),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, PALETTE["line"]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(stats)
    story.append(PageBreak())

    for kind, content in parse_markdown_lines(text):
        if kind == "spacer":
            story.append(Spacer(1, 6))
        elif kind == "h1":
            story.append(Paragraph(inline_md(content), styles["Heading2Custom"]))
        elif kind == "h2":
            story.append(Spacer(1, 8))
            story.append(make_banner(content, styles))
            story.append(Spacer(1, 6))
        elif kind == "h3":
            story.append(Paragraph(inline_md(content), styles["Heading3Custom"]))
        elif kind.startswith("h"):
            story.append(Paragraph(inline_md(content), styles["Heading3Custom"]))
        elif kind == "image":
            alt, url = content
            story.extend(make_image_flowable(alt, url, styles))
        elif kind == "quote":
            story.append(make_quote(content, styles))
            story.append(Spacer(1, 6))
        elif kind == "bullets":
            for item in content:
                story.append(Paragraph(inline_md(item), styles["BulletCustom"], bulletText="-"))
        elif kind == "mermaid":
            story.extend(make_mermaid_flowable(content, styles))
        elif kind == "code":
            story.append(make_code_block(content, styles))
        elif kind == "table":
            story.append(make_table(content, styles))
            story.append(Spacer(1, 6))
        else:
            story.append(Paragraph(inline_md(content), styles["BodyCustom"]))

    doc.build(story)


if __name__ == "__main__":
    build_pdf()
