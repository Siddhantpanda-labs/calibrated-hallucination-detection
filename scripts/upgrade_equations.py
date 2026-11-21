import docx
from docx.oxml import parse_xml
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
import shutil
import os

m_ns = "http://schemas.openxmlformats.org/officeDocument/2006/math"
w_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# OMML definitions for the 9 equations
eq_omml = {
    1: f'''<m:oMathPara xmlns:m="{m_ns}"><m:oMath>
  <m:r><m:rPr><m:nor/></m:rPr><m:t>P</m:t></m:r>
  <m:d>
    <m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr>
    <m:e>
      <m:sSub><m:e><m:r><m:t>y</m:t></m:r></m:e><m:sub><m:r><m:t>t</m:t></m:r></m:sub></m:sSub>
      <m:r><m:t> | </m:t></m:r>
      <m:r><m:t>q</m:t></m:r>
      <m:r><m:t>, </m:t></m:r>
      <m:sSub><m:e><m:r><m:t>y</m:t></m:r></m:e><m:sub><m:r><m:t>&lt;t</m:t></m:r></m:sub></m:sSub>
    </m:e>
  </m:d>
  <m:r><m:t> = </m:t></m:r>
  <m:f>
    <m:num>
      <m:r><m:rPr><m:nor/></m:rPr><m:t>exp</m:t></m:r>
      <m:d>
        <m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr>
        <m:e>
          <m:sSub>
            <m:e><m:r><m:t>z</m:t></m:r></m:e>
            <m:sub>
              <m:r><m:t>t, </m:t></m:r>
              <m:sSub><m:e><m:r><m:t>y</m:t></m:r></m:e><m:sub><m:r><m:t>t</m:t></m:r></m:sub></m:sSub>
            </m:sub>
          </m:sSub>
        </m:e>
      </m:d>
    </m:num>
    <m:den>
      <m:nary>
        <m:naryPr>
          <m:chr m:val="∑"/>
          <m:limLoc m:val="undOvr"/>
          <m:supHide m:val="1"/>
        </m:naryPr>
        <m:sub>
          <m:r><m:t>v ∈ </m:t></m:r>
          <m:r><m:rPr><m:scr m:val="script"/></m:rPr><m:t>V</m:t></m:r>
        </m:sub>
        <m:sup/>
        <m:e>
          <m:r><m:rPr><m:nor/></m:rPr><m:t>exp</m:t></m:r>
          <m:d>
            <m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr>
            <m:e>
              <m:sSub><m:e><m:r><m:t>z</m:t></m:r></m:e><m:sub><m:r><m:t>t, v</m:t></m:r></m:sub></m:sSub>
            </m:e>
          </m:d>
        </m:e>
      </m:nary>
    </m:den>
  </m:f>
</m:oMath></m:oMathPara>''',

    2: f'''<m:oMathPara xmlns:m="{m_ns}"><m:oMath>
  <m:r><m:rPr><m:nor/></m:rPr><m:t>mean_logprob</m:t></m:r>
  <m:r><m:t> = </m:t></m:r>
  <m:f>
    <m:num><m:r><m:t>1</m:t></m:r></m:num>
    <m:den><m:r><m:t>T</m:t></m:r></m:den>
  </m:f>
  <m:nary>
    <m:naryPr>
      <m:chr m:val="∑"/>
      <m:limLoc m:val="undOvr"/>
      <m:subHide m:val="0"/>
      <m:supHide m:val="0"/>
    </m:naryPr>
    <m:sub><m:r><m:t>t = 1</m:t></m:r></m:sub>
    <m:sup><m:r><m:t>T</m:t></m:r></m:sup>
    <m:e>
      <m:r><m:rPr><m:nor/></m:rPr><m:t>log </m:t></m:r>
      <m:r><m:rPr><m:nor/></m:rPr><m:t>P</m:t></m:r>
      <m:d>
        <m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr>
        <m:e>
          <m:sSub><m:e><m:r><m:t>y</m:t></m:r></m:e><m:sub><m:r><m:t>t</m:t></m:r></m:sub></m:sSub>
          <m:r><m:t> | </m:t></m:r>
          <m:r><m:t>q</m:t></m:r>
          <m:r><m:t>, </m:t></m:r>
          <m:sSub><m:e><m:r><m:t>y</m:t></m:r></m:e><m:sub><m:r><m:t>&lt;t</m:t></m:r></m:sub></m:sSub>
        </m:e>
      </m:d>
    </m:e>
  </m:nary>
</m:oMath></m:oMathPara>''',

    3: f'''<m:oMathPara xmlns:m="{m_ns}"><m:oMath>
  <m:r><m:rPr><m:scr m:val="script"/></m:rPr><m:t>H</m:t></m:r>
  <m:d>
    <m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr>
    <m:e><m:sSub><m:e><m:r><m:t>z</m:t></m:r></m:e><m:sub><m:r><m:t>t</m:t></m:r></m:sub></m:sSub></m:e>
  </m:d>
  <m:r><m:t> = - </m:t></m:r>
  <m:nary>
    <m:naryPr>
      <m:chr m:val="∑"/>
      <m:limLoc m:val="undOvr"/>
      <m:supHide m:val="1"/>
    </m:naryPr>
    <m:sub>
      <m:r><m:t>v ∈ </m:t></m:r>
      <m:r><m:rPr><m:scr m:val="script"/></m:rPr><m:t>V</m:t></m:r>
    </m:sub>
    <m:sup/>
    <m:e>
      <m:r><m:rPr><m:nor/></m:rPr><m:t>P</m:t></m:r>
      <m:d>
        <m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr>
        <m:e><m:r><m:t>v | q, </m:t></m:r><m:sSub><m:e><m:r><m:t>y</m:t></m:r></m:e><m:sub><m:r><m:t>&lt;t</m:t></m:r></m:sub></m:sSub></m:e>
      </m:d>
      <m:r><m:rPr><m:nor/></m:rPr><m:t> log </m:t></m:r>
      <m:r><m:rPr><m:nor/></m:rPr><m:t>P</m:t></m:r>
      <m:d>
        <m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr>
        <m:e><m:r><m:t>v | q, </m:t></m:r><m:sSub><m:e><m:r><m:t>y</m:t></m:r></m:e><m:sub><m:r><m:t>&lt;t</m:t></m:r></m:sub></m:sSub></m:e>
      </m:d>
    </m:e>
  </m:nary>
</m:oMath></m:oMathPara>''',

    4: f'''<m:oMathPara xmlns:m="{m_ns}"><m:oMath>
  <m:r><m:rPr><m:nor/></m:rPr><m:t>agreement_rate</m:t></m:r>
  <m:r><m:t> = </m:t></m:r>
  <m:f>
    <m:num><m:r><m:t>1</m:t></m:r></m:num>
    <m:den><m:r><m:t>N</m:t></m:r></m:den>
  </m:f>
  <m:nary>
    <m:naryPr>
      <m:chr m:val="∑"/>
      <m:limLoc m:val="undOvr"/>
      <m:subHide m:val="0"/>
      <m:supHide m:val="0"/>
    </m:naryPr>
    <m:sub><m:r><m:t>i = 1</m:t></m:r></m:sub>
    <m:sup><m:r><m:t>N</m:t></m:r></m:sup>
    <m:e>
      <m:r><m:rPr><m:scr m:val="script"/></m:rPr><m:t>M</m:t></m:r>
      <m:d>
        <m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr>
        <m:e>
          <m:sSub><m:e><m:r><m:t>s</m:t></m:r></m:e><m:sub><m:r><m:t>i</m:t></m:r></m:sub></m:sSub>
          <m:r><m:t>, </m:t></m:r>
          <m:sSub><m:e><m:r><m:t>y</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:nor/></m:rPr><m:t>greedy</m:t></m:r></m:sub></m:sSub>
        </m:e>
      </m:d>
    </m:e>
  </m:nary>
</m:oMath></m:oMathPara>''',

    5: f'''<m:oMathPara xmlns:m="{m_ns}"><m:oMath>
  <m:r><m:rPr><m:nor/></m:rPr><m:t>P</m:t></m:r>
  <m:d>
    <m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr>
    <m:e><m:r><m:t>u ⇒ v</m:t></m:r></m:e>
  </m:d>
  <m:r><m:t> ≥ γ  ∧  </m:t></m:r>
  <m:r><m:rPr><m:nor/></m:rPr><m:t>P</m:t></m:r>
  <m:d>
    <m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr>
    <m:e><m:r><m:t>v ⇒ u</m:t></m:r></m:e>
  </m:d>
  <m:r><m:t> ≥ γ</m:t></m:r>
</m:oMath></m:oMathPara>''',

    6: f'''<m:oMathPara xmlns:m="{m_ns}"><m:oMath>
  <m:r><m:rPr><m:nor/></m:rPr><m:t>SE</m:t></m:r>
  <m:d>
    <m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr>
    <m:e><m:r><m:t>S</m:t></m:r></m:e>
  </m:d>
  <m:r><m:t> = - </m:t></m:r>
  <m:nary>
    <m:naryPr>
      <m:chr m:val="∑"/>
      <m:limLoc m:val="undOvr"/>
      <m:subHide m:val="0"/>
      <m:supHide m:val="0"/>
    </m:naryPr>
    <m:sub><m:r><m:t>k = 1</m:t></m:r></m:sub>
    <m:sup><m:r><m:t>K</m:t></m:r></m:sup>
    <m:e>
      <m:sSub><m:e><m:r><m:t>p</m:t></m:r></m:e><m:sub><m:r><m:t>k</m:t></m:r></m:sub></m:sSub>
      <m:r><m:rPr><m:nor/></m:rPr><m:t> log </m:t></m:r>
      <m:sSub><m:e><m:r><m:t>p</m:t></m:r></m:e><m:sub><m:r><m:t>k</m:t></m:r></m:sub></m:sSub>
    </m:e>
  </m:nary>
</m:oMath></m:oMathPara>''',

    7: f'''<m:oMathPara xmlns:m="{m_ns}"><m:oMath>
  <m:sSub>
    <m:e><m:r><m:rPr><m:nor/></m:rPr><m:t>P</m:t></m:r></m:e>
    <m:sub><m:r><m:rPr><m:nor/></m:rPr><m:t>probe</m:t></m:r></m:sub>
  </m:sSub>
  <m:d>
    <m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr>
    <m:e>
      <m:r><m:t>y = 1 | </m:t></m:r>
      <m:sSub>
        <m:e><m:r><m:rPr><m:sty m:val="b"/></m:rPr><m:t>x</m:t></m:r></m:e>
        <m:sub><m:r><m:t>l</m:t></m:r></m:sub>
      </m:sSub>
    </m:e>
  </m:d>
  <m:r><m:t> = σ</m:t></m:r>
  <m:d>
    <m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr>
    <m:e>
      <m:sSubSup>
        <m:e><m:r><m:rPr><m:sty m:val="b"/></m:rPr><m:t>w</m:t></m:r></m:e>
        <m:sub><m:r><m:t>l</m:t></m:r></m:sub>
        <m:sup><m:r><m:t>T</m:t></m:r></m:sup>
      </m:sSubSup>
      <m:sSub>
        <m:e><m:r><m:rPr><m:sty m:val="b"/></m:rPr><m:t>x</m:t></m:r></m:e>
        <m:sub><m:r><m:t>l</m:t></m:r></m:sub>
      </m:sSub>
      <m:r><m:t> + </m:t></m:r>
      <m:sSub>
        <m:e><m:r><m:t>b</m:t></m:r></m:e>
        <m:sub><m:r><m:t>l</m:t></m:r></m:sub>
      </m:sSub>
    </m:e>
  </m:d>
  <m:r><m:t> = </m:t></m:r>
  <m:f>
    <m:num><m:r><m:t>1</m:t></m:r></m:num>
    <m:den>
      <m:r><m:t>1 + </m:t></m:r>
      <m:r><m:rPr><m:nor/></m:rPr><m:t>exp</m:t></m:r>
      <m:d>
        <m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr>
        <m:e>
          <m:r><m:t>-(</m:t></m:r>
          <m:sSubSup>
            <m:e><m:r><m:rPr><m:sty m:val="b"/></m:rPr><m:t>w</m:t></m:r></m:e>
            <m:sub><m:r><m:t>l</m:t></m:r></m:sub>
            <m:sup><m:r><m:t>T</m:t></m:r></m:sup>
          </m:sSubSup>
          <m:sSub>
            <m:e><m:r><m:rPr><m:sty m:val="b"/></m:rPr><m:t>x</m:t></m:r></m:e>
            <m:sub><m:r><m:t>l</m:t></m:r></m:sub>
          </m:sSub>
          <m:r><m:t> + </m:t></m:r>
          <m:sSub>
            <m:e><m:r><m:t>b</m:t></m:r></m:e>
            <m:sub><m:r><m:t>l</m:t></m:r></m:sub>
          </m:sSub>
          <m:r><m:t>)</m:t></m:r>
        </m:e>
      </m:d>
    </m:den>
  </m:f>
</m:oMath></m:oMathPara>''',

    8: f'''<m:oMathPara xmlns:m="{m_ns}"><m:oMath>
  <m:r><m:rPr><m:nor/></m:rPr><m:t>P</m:t></m:r>
  <m:d>
    <m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr>
    <m:e>
      <m:r><m:t>y = 1 | </m:t></m:r>
      <m:acc>
        <m:accPr><m:chr m:val="^"/></m:accPr>
        <m:e><m:r><m:t>p</m:t></m:r></m:e>
      </m:acc>
      <m:r><m:t> = p</m:t></m:r>
    </m:e>
  </m:d>
  <m:r><m:t> = p,   ∀p ∈ [0, 1]</m:t></m:r>
</m:oMath></m:oMathPara>''',

    9: f'''<m:oMathPara xmlns:m="{m_ns}"><m:oMath>
  <m:r><m:rPr><m:nor/></m:rPr><m:t>ECE</m:t></m:r>
  <m:r><m:t> = </m:t></m:r>
  <m:nary>
    <m:naryPr>
      <m:chr m:val="∑"/>
      <m:limLoc m:val="undOvr"/>
      <m:subHide m:val="0"/>
      <m:supHide m:val="0"/>
    </m:naryPr>
    <m:sub><m:r><m:t>m = 1</m:t></m:r></m:sub>
    <m:sup><m:r><m:t>M</m:t></m:r></m:sup>
    <m:e>
      <m:f>
        <m:num>
          <m:d>
            <m:dPr><m:begChr m:val="|"/><m:endChr m:val="|"/></m:dPr>
            <m:e>
              <m:sSub><m:e><m:r><m:t>B</m:t></m:r></m:e><m:sub><m:r><m:t>m</m:t></m:r></m:sub></m:sSub>
            </m:e>
          </m:d>
        </m:num>
        <m:den><m:r><m:t>N</m:t></m:r></m:den>
      </m:f>
      <m:d>
        <m:dPr><m:begChr m:val="|"/><m:endChr m:val="|"/></m:dPr>
        <m:e>
          <m:r><m:rPr><m:nor/></m:rPr><m:t>acc</m:t></m:r>
          <m:d>
            <m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr>
            <m:e><m:sSub><m:e><m:r><m:t>B</m:t></m:r></m:e><m:sub><m:r><m:t>m</m:t></m:r></m:sub></m:sSub></m:e>
          </m:d>
          <m:r><m:t> - </m:t></m:r>
          <m:r><m:rPr><m:nor/></m:rPr><m:t>conf</m:t></m:r>
          <m:d>
            <m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr>
            <m:e><m:sSub><m:e><m:r><m:t>B</m:t></m:r></m:e><m:sub><m:r><m:t>m</m:t></m:r></m:sub></m:sSub></m:e>
          </m:d>
        </m:e>
      </m:d>
    </m:e>
  </m:nary>
</m:oMath></m:oMathPara>'''
}

def create_equation_table(doc, eq_num, omath_xml_str):
    """Creates a 1-row 2-col borderless table with OMML equation centered and (eq_num) flush right."""
    tbl = doc.add_table(rows=1, cols=2)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False

    # Remove all borders
    tblBorders = parse_xml(r'''
        <w:tblBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
            <w:top w:val="none" w:sz="0" w:space="0" w:color="auto"/>
            <w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>
            <w:bottom w:val="none" w:sz="0" w:space="0" w:color="auto"/>
            <w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>
            <w:insideH w:val="none" w:sz="0" w:space="0" w:color="auto"/>
            <w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto"/>
        </w:tblBorders>
    ''')
    tbl._tbl.tblPr.append(tblBorders)

    cell_eq = tbl.cell(0, 0)
    cell_num = tbl.cell(0, 1)

    # Column widths: 2.92 in for math, 0.46 in for eq number (total ~3.38 in)
    cell_eq.width = Inches(2.92)
    cell_num.width = Inches(0.46)
    cell_num.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    # Equation cell
    p_eq = cell_eq.paragraphs[0]
    p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_eq.paragraph_format.space_before = Pt(3)
    p_eq.paragraph_format.space_after = Pt(3)
    p_eq.paragraph_format.line_spacing = 1.0
    p_eq._element.append(parse_xml(omath_xml_str))

    # Number cell
    p_num = cell_num.paragraphs[0]
    p_num.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_num.paragraph_format.space_before = Pt(3)
    p_num.paragraph_format.space_after = Pt(3)
    p_num.paragraph_format.line_spacing = 1.0
    r_num = p_num.add_run(f"({eq_num})")
    r_num.font.name = "Times New Roman"
    r_num.font.size = Pt(10)
    r_num.font.italic = False

    return tbl

def process_document(input_path, output_path):
    print(f"Loading document from {input_path}...")
    doc = docx.Document(input_path)

    # Verify front page preservation
    front_page_text = [doc.paragraphs[i].text for i in range(min(15, len(doc.paragraphs)))]
    print(f"Preserving front page ({len(front_page_text)} paragraphs checked)...")
    assert any("PROJECT-BASED ASSESSMENT" in t for t in front_page_text), "Front page header not found!"

    # Also clean up inline formulas in prose paragraphs (replacing plaintext tokens with clean mathematical typography)
    inline_replacements = [
        ("where gamma = 0.5 is the entailment", "where γ = 0.5 is the entailment"),
        ("threshold tau = 0.5 for token overlap", "threshold τ = 0.5 for token overlap"),
        ("threshold tau = 0.50", "threshold τ = 0.50"),
        ("tau in [0.1, 0.9]", "τ ∈ [0.1, 0.9]"),
        ("tau in [0, 1]", "τ ∈ [0, 1]"),
        ("Let y in {0, 1} denote", "Let y ∈ {0, 1} denote"),
        ("and let p_hat in [0, 1]", "and let p̂ ∈ [0, 1]"),
        ("conf(B_m) = (1/|B_m|) sum_{i in B_m} p_hat_i", "conf(B_m) = (1/|B_m|) ∑_{i ∈ B_m} p̂_i"),
        ("acc(B_m) = (1/|B_m|) sum_{i in B_m} y_i", "acc(B_m) = (1/|B_m|) ∑_{i ∈ B_m} y_i"),
        ("activation vector x_l in R^d", "activation vector x_l ∈ ℝᵈ"),
        ("depths l in {4, 8, 12, 16, 20, 24, -1}", "depths l ∈ {4, 8, 12, 16, 20, 24, -1}")
    ]

    for p in doc.paragraphs:
        for old_txt, new_txt in inline_replacements:
            if old_txt in p.text:
                for run in p.runs:
                    if old_txt in run.text:
                        run.text = run.text.replace(old_txt, new_txt)

    # Locate equation paragraphs (only in body paragraphs > 18)
    replaced_count = 0
    paragraphs_to_replace = []

    for idx, p in enumerate(doc.paragraphs):
        if idx <= 18:
            continue
        for eq_num in range(1, 10):
            if f"({eq_num})" in p.text and len(p.text) < 130 and any(k in p.text for k in ["=", ">=", "P(", "ECE", "SE(", "logprob", "H(z", "probe"]):
                paragraphs_to_replace.append((p, eq_num, idx, p.text.strip()))
                break

    print(f"Found {len(paragraphs_to_replace)} equation paragraphs to replace with OMML tables:")
    for p, eq_num, idx, txt in paragraphs_to_replace:
        print(f"  Eq {eq_num} at P{idx}: {txt[:60]}...")

    assert len(paragraphs_to_replace) == 9, f"Expected 9 equations, found {len(paragraphs_to_replace)}"

    # Replace each paragraph in-place
    for p, eq_num, idx, _ in paragraphs_to_replace:
        tbl = create_equation_table(doc, eq_num, eq_omml[eq_num])
        p._element.addprevious(tbl._tbl)
        p._element.getparent().remove(p._element)
        replaced_count += 1

    print(f"Successfully replaced {replaced_count} equations with rendered OMML tables!")
    
    # Save output to output_path
    doc.save(output_path)
    print(f"Saved calibrated report to {output_path} successfully!")

    # Attempt to copy to original if possible
    if output_path != input_path:
        try:
            shutil.copyfile(output_path, input_path)
            print(f"Successfully synced update back to {input_path}!")
        except Exception as e:
            print(f"Note: Could not overwrite {input_path} directly (file is currently open in Word).")
            print(f"The updated file is saved as: {output_path}")

if __name__ == "__main__":
    src = "results/Calibrated_Hallucination_Detection_IEEE_Report.docx"
    target = "results/Calibrated_Hallucination_Detection_IEEE_Report_Updated.docx"
    process_document(src, target)
