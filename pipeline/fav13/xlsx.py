#!/usr/bin/env python3
"""Minimal multi-sheet .xlsx writer -- no third-party dependencies.

An xlsx file is a zip of OOXML parts.  We emit the smallest valid set:
content types, relationships, a workbook, one worksheet per sheet, and a
style table carrying a bold header font and a few number formats.
Strings are written inline, so no shared-string table is needed.
"""

import zipfile


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _col(i):
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


CT = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
{sheets}
</Types>"""

RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""

STYLES = """<?xml version="1.0" encoding="UTF-8"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<numFmts count="2">
<numFmt numFmtId="164" formatCode="#,##0"/>
<numFmt numFmtId="165" formatCode="0.0000"/>
</numFmts>
<fonts count="3">
<font><sz val="11"/><name val="Calibri"/></font>
<font><b/><sz val="11"/><name val="Calibri"/><color rgb="FFFFFFFF"/></font>
<font><i/><sz val="10"/><name val="Calibri"/><color rgb="FF666666"/></font>
</fonts>
<fills count="3">
<fill><patternFill patternType="none"/></fill>
<fill><patternFill patternType="gray125"/></fill>
<fill><patternFill patternType="solid"><fgColor rgb="FF1E5B4F"/><bgColor indexed="64"/></patternFill></fill>
</fills>
<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="5">
<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>
<xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
<xf numFmtId="165" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
<xf numFmtId="0" fontId="2" fillId="0" borderId="0" xfId="0" applyFont="1"/>
</cellXfs>
</styleSheet>"""

S_PLAIN, S_HEAD, S_INT, S_DEC, S_NOTE = 0, 1, 2, 3, 4


def _sheet_xml(rows, widths=None):
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">']
    if widths:
        out.append("<cols>")
        for i, w in enumerate(widths):
            out.append(f'<col min="{i+1}" max="{i+1}" width="{w}" customWidth="1"/>')
        out.append("</cols>")
    out.append("<sheetData>")
    for ri, row in enumerate(rows, 1):
        out.append(f'<row r="{ri}">')
        for ci, cell in enumerate(row):
            ref = f"{_col(ci)}{ri}"
            if isinstance(cell, tuple):
                val, style = cell
            else:
                val, style = cell, (S_HEAD if ri == 1 else S_PLAIN)
            if val is None or val == "":
                continue
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                if style in (S_PLAIN, S_HEAD):
                    style = S_INT if isinstance(val, int) else S_DEC
                out.append(f'<c r="{ref}" s="{style}"><v>{val}</v></c>')
            else:
                out.append(f'<c r="{ref}" s="{style}" t="inlineStr">'
                           f'<is><t xml:space="preserve">{_esc(val)}</t></is></c>')
        out.append("</row>")
    out.append("</sheetData></worksheet>")
    return "".join(out)


def write(path, sheets):
    """sheets: list of (name, rows, widths)."""
    n = len(sheets)
    ct = CT.format(sheets="\n".join(
        f'<Override PartName="/xl/worksheets/sheet{i+1}.xml" '
        f'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(n)))
    wb = ['<?xml version="1.0" encoding="UTF-8"?>',
          '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
          "<sheets>"]
    for i, (name, _, _) in enumerate(sheets):
        wb.append(f'<sheet name="{_esc(name)[:31]}" sheetId="{i+1}" r:id="rId{i+1}"/>')
    wb.append("</sheets></workbook>")

    rel = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
    for i in range(n):
        rel.append(f'<Relationship Id="rId{i+1}" Type="http://schemas.openxmlformats.org/'
                   f'officeDocument/2006/relationships/worksheet" '
                   f'Target="worksheets/sheet{i+1}.xml"/>')
    rel.append(f'<Relationship Id="rId{n+1}" Type="http://schemas.openxmlformats.org/'
               f'officeDocument/2006/relationships/styles" Target="styles.xml"/>')
    rel.append("</Relationships>")

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", RELS)
        z.writestr("xl/workbook.xml", "".join(wb))
        z.writestr("xl/_rels/workbook.xml.rels", "".join(rel))
        z.writestr("xl/styles.xml", STYLES)
        for i, (_, rows, widths) in enumerate(sheets):
            z.writestr(f"xl/worksheets/sheet{i+1}.xml", _sheet_xml(rows, widths))
    return path
