"""Render REPORT.md to REPORT.pdf (A4) with the two figures embedded. Handles the Markdown subset used in REPORT.md."""
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
                                ListFlowable, ListItem, Preformatted, KeepTogether)

FIGS = {"5.": ("fig_main.png", "Figure 1. Accuracy (left) and attack success rate (right) vs number of poisoned passages, n = 46."),
        "6.": ("fig_temporal.png", "Figure 2. Accuracy on the current answer under Attack A (left) and Attack B (right); coinciding lines hide each other.")}

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["Normal"], fontName="Times-Roman", fontSize=10, leading=13.5, spaceAfter=5)
cell = ParagraphStyle("c", parent=body, fontSize=8.3, leading=10, spaceAfter=0)
cellb = ParagraphStyle("cb", parent=cell, fontName="Times-Bold")
h1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Times-Bold", fontSize=16, leading=20, spaceAfter=6)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Times-Bold", fontSize=12.5, spaceBefore=10, spaceAfter=4)
h3 = ParagraphStyle("h3", parent=h2, fontSize=11, spaceBefore=6)
item = ParagraphStyle("item", parent=body, leftIndent=16, bulletIndent=4, spaceAfter=3)
cap = ParagraphStyle("cap", parent=body, fontSize=9, leading=11, alignment=1, textColor=colors.HexColor("#333333"))
code = ParagraphStyle("code", parent=ss["Code"], fontSize=8, leading=10)

def inline(s):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2" color="blue">\1</link>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", s)
    s = re.sub(r"`([^`]+)`", r'<font face="Courier" size="8.5">\1</font>', s)
    return s.replace(">=", "&gt;=").replace("<=", "&lt;=") if False else s

def table(rows, width):
    data = [[Paragraph(inline(c), cellb if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[width / len(rows[0])] * len(rows[0]), repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return t

def figure(path, caption, width):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    return KeepTogether([Spacer(1, 4), Image(path, width=width, height=width * ih / iw), Paragraph(caption, cap), Spacer(1, 6)])

def build(md="REPORT.md", out="REPORT.pdf"):
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="RobustRAG: Generation Against Temporal Conflicts, Adversarial Poisoning and Hallucination", author="Akash Tiwari, Vaibhav Soni")
    W = A4[0] - 4 * cm
    lines = open(md, encoding="utf8").read().splitlines()
    story, i, section, pending_fig = [], 0, None, None
    def flush_fig():
        nonlocal pending_fig
        if pending_fig: story.append(figure(*pending_fig, W)); pending_fig = None
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("# "):
            story.append(Paragraph(inline(ln[2:]), h1)); i += 1
        elif ln.startswith("### "):
            story.append(Paragraph(inline(ln[4:]), h3)); i += 1
        elif ln.startswith("## "):
            flush_fig()
            title = ln[3:]; section = title.split()[0]
            story.append(Paragraph(inline(title), h2)); i += 1
            if section in FIGS: pending_fig = FIGS[section]
        elif ln.startswith("```"):
            j = i + 1; buf = []
            while not lines[j].startswith("```"): buf.append(lines[j]); j += 1
            story.append(Preformatted("\n".join(buf), code)); i = j + 1
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"-+", c) for c in cells): rows.append(cells)
                i += 1
            prev = story.pop() if story and isinstance(story[-1], Paragraph) and len(story[-1].text) < 40 else None
            story.append(KeepTogether(([prev] if prev else []) + [table(rows, W), Spacer(1, 6)]))   # keep short label + table on one page
        elif re.match(r"^(\d+\.|-) ", ln):
            # one Paragraph per item (not a ListFlowable), so a list can break across pages
            # and a heading is only kept together with the first item
            ordered, n = ln[0].isdigit(), 0
            while i < len(lines) and re.match(r"^(\d+\.|-) ", lines[i]):
                txt = re.sub(r"^(\d+\.|-) ", "", lines[i]); i += 1
                while i < len(lines) and lines[i].startswith("   ") and lines[i].strip():
                    txt += " " + lines[i].strip(); i += 1
                n += 1
                story.append(Paragraph(inline(txt), item, bulletText=f"{n}." if ordered else "•"))
        elif ln.strip():
            txt = ln
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||```|\d+\. |- )", lines[i]):
                txt += " " + lines[i]; i += 1
            story.append(Paragraph(inline(txt), body))
            # place each figure right after the first paragraph that names it
            if pending_fig and pending_fig[0] in txt: flush_fig()
        else:
            i += 1
    flush_fig()
    merged, k = [], 0                      # keep every section heading on the same page as what follows it
    while k < len(story):
        f = story[k]
        is_label = isinstance(f, Paragraph) and len(f.text) < 80 and f.text.rstrip().rstrip("</b>").endswith(":")
        if isinstance(f, Paragraph) and (f.style in (h2, h3) or is_label) and k + 1 < len(story):
            nxt = story[k + 1]
            inner = list(nxt._content) if isinstance(nxt, KeepTogether) else [nxt]   # flatten: nested KeepTogether misplaces breaks
            merged.append(KeepTogether([f] + inner)); k += 2
        else:
            merged.append(f); k += 1
    story = merged
    doc.build(story, onLaterPages=lambda c, d: (c.setFont("Times-Roman", 8), c.drawRightString(A4[0] - 2 * cm, 1 * cm, f"{d.page}")))

if __name__ == "__main__":
    build()
