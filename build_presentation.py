"""Builds the AUDA project presentation (AUDA_Project_Presentation.pptx).
Regenerates charts, then assembles all slides including detailed preprocessing steps.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# ---------------- charts ----------------
CSV = "auda_benchmark_3/multiseed_results.csv"
d = pd.read_csv(CSV)

fig, ax = plt.subplots(figsize=(10, 5.2)); dd = d.sort_values("svm_mean"); x = np.arange(len(dd))
ax.bar(x - 0.2, dd["rf_mean"], 0.4, label="RandomForest", color="#27AE60", alpha=.85)
ax.bar(x + 0.2, dd["svm_mean"], 0.4, label="SVM", color="#8E44AD", alpha=.85)
ax.axhline(0, color="k", lw=.8)
ax.set_xticks(x); ax.set_xticklabels([n.replace(".csv", "") for n in dd["dataset"]], rotation=90, fontsize=6)
ax.set_ylabel("Mean delta AUC (AUDA - fair baseline)")
ax.set_title("AUDA vs Fair Baseline - per-dataset (5-seed mean). Most ~0 = parity")
ax.legend(); plt.tight_layout(); plt.savefig("_c_perdataset.png", dpi=140); plt.close()

fig, ax = plt.subplots(figsize=(6.5, 4.2)); means = [d["rf_mean"].mean(), d["svm_mean"].mean()]
ax.bar(["RandomForest\np=0.29", "SVM\np=0.94"], means, color=["#27AE60", "#8E44AD"], alpha=.85, width=.5)
ax.axhline(0, color="k", lw=.8); ax.set_ylim(-0.02, 0.02)
ax.set_ylabel("Mean delta AUC (AUDA - fair baseline)")
ax.set_title("Final Result: Statistical Parity (both ~0, not significant)")
for i, m in enumerate(means):
    ax.text(i, m, f"{m:+.4f}", ha="center", va="bottom" if m >= 0 else "top", fontweight="bold")
plt.tight_layout(); plt.savefig("_c_parity.png", dpi=140); plt.close()

fig, ax = plt.subplots(figsize=(8.5, 4.4))
stages = ["Single-seed\n(weak base)", "Multi-seed\n(weak base)", "+ saturation\noff", "+ dedup", "+ fair\nbaseline", "+ bugs\nfixed"]
svm = [0.051, 0.030, 0.116, 0.091, -0.037, 0.003]; sig = ["?", "no", "yes*", "yes*", "no", "no"]
colors = ["#95a5a6", "#e67e22", "#e74c3c", "#e74c3c", "#3498db", "#27AE60"]
bars = ax.bar(stages, svm, color=colors, alpha=.88); ax.axhline(0, color="k", lw=.8)
ax.set_ylabel("SVM mean delta AUC")
ax.set_title("The Journey: SVM 'win' shrinks to parity as artifacts are removed")
for bar, v, s in zip(bars, svm, sig):
    ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:+.3f}\n({s})", ha="center",
            va="bottom" if v >= 0 else "top", fontsize=8)
plt.tight_layout(); plt.savefig("_c_journey.png", dpi=140); plt.close()

# ---------------- theme ----------------
NAVY = RGBColor(0x1B, 0x26, 0x3B); BLUE = RGBColor(0x2E, 0x6D, 0xB8)
GREEN = RGBColor(0x27, 0xAE, 0x60); PURPLE = RGBColor(0x8E, 0x44, 0xAD)
RED = RGBColor(0xC0, 0x39, 0x2B); GREY = RGBColor(0x55, 0x5F, 0x6B)
LGREY = RGBColor(0xEC, 0xF0, 0xF3); WHITE = RGBColor(0xFF, 0xFF, 0xFF)
AMBER = RGBColor(0xE6, 0x7E, 0x22); SKY = RGBColor(0x6E, 0xC6, 0xF0)

prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]; SW, SH = prs.slide_width, prs.slide_height


def slide():
    return prs.slides.add_slide(BLANK)


def bg(s, color=WHITE):
    s.background.fill.solid(); s.background.fill.fore_color.rgb = color


def box(s, l, t, w, h):
    return s.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))


def title_bar(s, title, sub=None):
    bar = s.shapes.add_shape(1, 0, 0, SW, Inches(1.15))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
    tb = box(s, 0.55, 0.16, 12.2, 0.95); tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; r = p.add_run(); r.text = title
    r.font.size = Pt(30); r.font.bold = True; r.font.color.rgb = WHITE; r.font.name = "Calibri"
    if sub:
        p2 = tf.add_paragraph(); r2 = p2.add_run(); r2.text = sub
        r2.font.size = Pt(14); r2.font.italic = True; r2.font.color.rgb = RGBColor(0xB8, 0xC6, 0xD8)


def bullets(s, items, l=0.7, t=1.5, w=12.0, h=5.4, size=18, gap=8):
    tb = box(s, l, t, w, h); tf = tb.text_frame; tf.word_wrap = True; first = True
    for it in items:
        if isinstance(it, str):
            txt, lvl, col, bold = it, 0, NAVY, False
        elif isinstance(it[0], int):
            lvl = it[0]; txt = it[1]; col = it[2] if len(it) > 2 else NAVY; bold = it[3] if len(it) > 3 else False
        else:
            txt = it[0]; lvl = it[1] if len(it) > 1 else 0; col = it[2] if len(it) > 2 else NAVY; bold = it[3] if len(it) > 3 else False
        p = tf.paragraphs[0] if first else tf.add_paragraph(); first = False
        p.level = lvl; p.space_after = Pt(gap); r = p.add_run()
        r.text = ("•  " if lvl == 0 else "–  ") + txt
        r.font.size = Pt(size - (2 if lvl else 0)); r.font.color.rgb = col; r.font.bold = bold; r.font.name = "Calibri"
    return tb


def table(s, data, l, t, w, h, header_fill=NAVY, fontsize=13, col_w=None):
    rows, cols = len(data), len(data[0])
    gt = s.shapes.add_table(rows, cols, Inches(l), Inches(t), Inches(w), Inches(h)).table
    if col_w:
        for i, cw in enumerate(col_w):
            gt.columns[i].width = Inches(cw)
    for ri, row in enumerate(data):
        for ci, val in enumerate(row):
            c = gt.cell(ri, ci)
            c.margin_left = Inches(0.06); c.margin_right = Inches(0.06)
            c.margin_top = Inches(0.02); c.margin_bottom = Inches(0.02)
            tf = c.text_frame; tf.word_wrap = True; p = tf.paragraphs[0]; r = p.add_run(); r.text = str(val)
            r.font.size = Pt(fontsize); r.font.name = "Calibri"
            if ri == 0:
                r.font.bold = True; r.font.color.rgb = WHITE; c.fill.solid(); c.fill.fore_color.rgb = header_fill
            else:
                r.font.color.rgb = NAVY; c.fill.solid(); c.fill.fore_color.rgb = (WHITE if ri % 2 else LGREY)
    return gt


def img(s, path, l, t, w=None, h=None):
    if os.path.exists(path):
        s.shapes.add_picture(path, Inches(l), Inches(t), Inches(w) if w else None, Inches(h) if h else None)


def note(s, text, t=6.9, color=GREY, size=12):
    tb = box(s, 0.7, t, 12.0, 0.5); tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.italic = True; r.font.color.rgb = color


def centered(s, l, t, w, h, runs):
    tb = box(s, l, t, w, h); tf = tb.text_frame; tf.word_wrap = True; first = True
    for text, size, color, bold, italic in runs:
        p = tf.paragraphs[0] if first else tf.add_paragraph(); first = False
        p.alignment = PP_ALIGN.CENTER; r = p.add_run(); r.text = text
        r.font.size = Pt(size); r.font.color.rgb = color; r.font.bold = bold; r.font.italic = italic; r.font.name = "Calibri"


def rbox(s, l, t, w, h, lines, fill, fsize=14, fcolor=WHITE):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill; sh.line.color.rgb = fill
    tf = sh.text_frame; tf.word_wrap = True
    if isinstance(lines, str):
        lines = [lines]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER; r = p.add_run(); r.text = ln
        r.font.size = Pt(fsize if i == 0 else fsize - 3)
        r.font.bold = (i == 0); r.font.color.rgb = fcolor; r.font.name = "Calibri"
    return sh


def arrow(s, l, t, w, h, color):
    a = s.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(l), Inches(t), Inches(w), Inches(h))
    a.fill.solid(); a.fill.fore_color.rgb = color; a.line.fill.background()
    return a


# ===== 1. TITLE =====
s = slide(); bg(s, NAVY)
centered(s, 1.0, 2.1, 11.3, 3.0, [
    ("AUDA — Automated EDA & Preprocessing", 44, WHITE, True, False),
    ("Can automated preprocessing beat manual EDA?", 24, SKY, False, False),
    ("A rigorous benchmark — and an honest answer", 16, RGBColor(0xB8, 0xC6, 0xD8), False, True)])
note(s, "Big Data Final Project   •   32 datasets   •   RandomForest + SVM   •   5 seeds", t=5.5, color=RGBColor(0x8A, 0x9B, 0xB5), size=14)

# ===== 2. PROBLEM =====
s = slide(); bg(s); title_bar(s, "The Problem & Goal")
bullets(s, [
    "EDA + preprocessing is manual, slow, and depends on analyst skill.",
    "AUDA automates it: profile data, handle missing values, detect anomalies, transform features — in one call.",
    ("Research question:", 0, BLUE, True),
    (1, "Can AUDA's automated preprocessing match or beat a human analyst?"),
    (1, "Measured objectively: train ML models on AUDA- vs manually-preprocessed data, compare ROC-AUC."),
    ("This talk: what AUDA does (in detail), how we tested it rigorously, the bugs we found, and the honest result.", 0, GREEN, True)], size=18, gap=11)

# ===== 3. OVERVIEW =====
s = slide(); bg(s); title_bar(s, "What AUDA Does — Overview", "4 understanding stages + 1 transforming stage")
table(s, [
    ["Stage", "Purpose", "Changes data?"],
    ["1. Profile", "Detect column types + numeric sub-types", "No (understand)"],
    ["2. Missing values", "Impute or drop, per column", "YES"],
    ["3. Anomalies", "Detect outliers (IQR / Z / Isolation Forest)", "No (report)"],
    ["4. Recommend", "Feature-engineering suggestions (rules + optional LLM)", "No (report)"],
    ["Feature Transformer", "Encode, cap outliers, log1p, scale, select features", "YES — fed to model"],
], 0.7, 1.5, 12.0, 3.3, fontsize=14, col_w=[2.9, 6.6, 2.5])
note(s, "Only Stage 2 and the Feature Transformer change the data the model sees. The next slides detail each step.")

# ===== 3a. STAGE 1 PROFILING =====
s = slide(); bg(s); title_bar(s, "Stage 1 — Profiling: Understand Every Column", "profiler.py")
bullets(s, [
    ("Type-detection waterfall (first match wins):", 0, BLUE, True),
    (1, "Empty column → text   |   True/False → boolean"),
    (1, "Real dates, or text where ≥80% parse as dates → datetime"),
    (1, "Numbers: only {0,1} → boolean ;  ≤2 values → categorical ;  else → numeric"),
    (1, "Text: ≤20 unique → categorical ;  avg length >50 chars → text"),
    ("Numeric sub-type (makes later steps smart):", 0, BLUE, True),
    (1, "id = unique & sequential (PassengerId)  •  ordinal = integer ≤10 values (rating 1-5)"),
    (1, "count = non-negative integers  •  continuous = floats / high-cardinality (income)"),
], size=17, gap=8)
note(s, "Why it matters: knowing a column is an ID or a 1-5 rating prevents harmful transforms later (e.g. don't log-scale a rating).")

# ===== 3b. STAGE 2 MISSING =====
s = slide(); bg(s); title_bar(s, "Stage 2 — Missing Values: One Decision per Column", "missing_handler.py  (this step CHANGES the data)")
table(s, [
    ["Condition", "Action"],
    ["No missing values", "Do nothing"],
    ["≥ 60% missing", "Drop the whole column"],
    ["< 5% missing", "Drop those few rows"],
    ["Numeric (else)", "KNN imputation (5 nearest rows) if ≤10,000 rows, otherwise median"],
    ["Categorical / boolean (else)", "Fill with the mode (most common value)"],
    ["Datetime (else)", "Linear interpolation"],
    ["Text (else)", "Fill with constant \"Unknown\""],
], 0.7, 1.5, 12.0, 3.7, fontsize=13.5, col_w=[4.2, 7.8])
note(s, "Checked top-to-bottom. KNN looks at the 5 most similar rows to estimate a missing number — smarter than a flat mean.")

# ===== 3c. STAGE 3 ANOMALY =====
s = slide(); bg(s); title_bar(s, "Stage 3 — Anomaly Detection (report only)", "anomaly_detector.py  •  does NOT change the data")
table(s, [
    ["Method", "Flags a value / row when…"],
    ["IQR fence", "outside [Q1 − 1.5·IQR ,  Q3 + 1.5·IQR]"],
    ["Z-score", "more than 3 standard deviations from the mean (|z| > 3)"],
    ["Isolation Forest", "an ML model can 'isolate' the whole row easily across all numeric columns (needs ≥10 rows)"],
], 0.7, 1.7, 12.0, 2.4, fontsize=14, col_w=[3.0, 9.0])
bullets(s, [("This stage only reports counts — outlier *handling* happens in the transformer (Step 6).", 0, GREY, True)], t=4.5, size=15)

# ===== 3d. STAGE 4 TRANSFORMER =====
s = slide(); bg(s); title_bar(s, "Stage 4 — Feature Transformer: The Real Preprocessing", "feature_transformer.py  •  8 ordered steps fed to the model")
table(s, [
    ["#", "Step", "Detail"],
    ["1", "Drop zero-variance", "column has only one unique value"],
    ["2", "Drop near-constant categoricals", "one value covers > 95% of rows"],
    ["3", "Drop pure-ID columns", "sequential IDs (+ all-unique integers on small data)"],
    ["4", "Frequency-encode high-cardinality", "> 20 unique → replace value with how often it appears"],
    ["5", "One-hot encode (SVM / linear only)", "low-cardinality categories → avoids fake ordering"],
    ["6", "Cap outliers (IQR, k=5)", "mild clip; skipped for ordinal / count / ID"],
    ["7", "log1p skewed continuous", "skew>1, all ≥0, wide range; skipped for ordinal/count/ID"],
    ["8", "Scale + feature budget", "RobustScaler (SVM) / StandardScaler (linear) / none (tree); keep top-K by SelectKBest"],
], 0.5, 1.4, 12.4, 4.6, fontsize=12, col_w=[0.5, 4.3, 7.6])
note(s, "Steps 6 & 7 use the Stage-1 sub-types to avoid damaging ratings / counts / IDs.")

# ===== 3e. ONE TRANSFORMER, TWO SETTINGS (flow diagram) =====
s = slide(); bg(s); title_bar(s, "One Transformer, Two Settings — RF vs SVM",
                              "The Feature Transformer runs for BOTH models — only the settings differ")
rbox(s, 4.55, 1.45, 4.2, 0.8, ["Clean data", "(after Stages 1–2)"], BLUE, 16)
arrow(s, 3.4, 2.45, 0.45, 0.7, GREY)
arrow(s, 9.5, 2.45, 0.45, 0.7, GREY)
rbox(s, 0.7, 3.35, 5.6, 1.35,
     ["FeatureTransformer — TREE settings", "encode • cap outliers • log1p",
      "NO scaling · NO one-hot · NO feature cap"], GREEN, 14)
rbox(s, 7.05, 3.35, 5.6, 1.35,
     ["FeatureTransformer — KERNEL settings", "encode • cap outliers • log1p",
      "+ scaling · + one-hot · + feature cap"], PURPLE, 14)
arrow(s, 3.4, 4.85, 0.45, 0.6, GREY)
arrow(s, 9.5, 4.85, 0.45, 0.6, GREY)
rbox(s, 1.8, 5.55, 3.4, 0.8, "RandomForest", GREEN, 16)
rbox(s, 8.1, 5.55, 3.4, 0.8, "SVM", PURPLE, 16)
note(s, "Same component, applied twice. RF skips scaling / one-hot / feature-cap (trees don't need them); SVM uses all of them.")

# ===== 4. BASELINE vs AUDA =====
s = slide(); bg(s); title_bar(s, "Manual EDA (Baseline) vs AUDA")
table(s, [
    ["Step", "Weak baseline", "Fair baseline", "AUDA"],
    ["Missing numbers", "mean", "median", "KNN"],
    ["Encode categories", "ordinal (all)", "one-hot + frequency", "one-hot + frequency"],
    ["Scaling (for SVM)", "none", "StandardScaler", "RobustScaler"],
    ["Outliers / skew", "none", "none", "IQR cap + log1p"],
    ["Drop ID / junk cols", "no", "no", "yes"],
    ["Knows column meaning", "no", "no", "yes (sub-types)"],
], 0.6, 1.45, 12.2, 3.5, fontsize=13, col_w=[2.7, 2.6, 3.6, 3.3])
note(s, "A FAIR baseline already does the two things that matter most — proper encoding + scaling. That is what AUDA must beat.")

# ===== 5. EXPERIMENT =====
s = slide(); bg(s); title_bar(s, "The Experiment — How We Measure \"Better\"")
bullets(s, [
    "For each dataset: 80/20 train-test split, stratified.",
    "Preprocess two ways: Baseline  vs  AUDA.",
    "Train BOTH RandomForest (tree) and SVM (kernel); compare ROC-AUC.",
    ("Delta = AUDA AUC − Baseline AUC.   Positive = AUDA better.", 0, BLUE, True),
    ("Rigor added during this project:", 0, GREEN, True),
    (1, "5 random seeds per dataset → mean ± std (not one lucky split)"),
    (1, "Wilcoxon signed-rank test → is the difference statistically real?"),
    (1, "Fair baseline + duplicate-dataset removal")], size=18, gap=9)

# ===== 6. ORIGINAL CLAIM =====
s = slide(); bg(s); title_bar(s, "The Original Claim — and the Red Flags", "\"AUDA wins, 88% no-loss rate\"")
bullets(s, [
    "The first benchmark reported AUDA beating the baseline on most datasets.",
    ("But the result rested on four hidden problems:", 0, RED, True),
    (1, "Single train/test split — one lucky draw, no significance test"),
    (1, "Rigged baseline — ordinal-encoded everything & never scaled (sabotages SVM)"),
    (1, "Duplicate datasets — same files counted twice, inflating results"),
    (1, "Software bugs — scrambled labels & dropped predictive columns"),
    ("\"If a result can't survive scrutiny, it isn't a result.\"", 0, GREY, True)], size=18, gap=9)

# ===== 7. RIGOR 1 =====
s = slide(); bg(s); title_bar(s, "Building Rigor #1 — Multi-Seed + Significance")
bullets(s, [
    "One split = flipping a coin once. Change the seed, get a different score.",
    (1, "Example (haberman, SVM): seed 42 → +0.10, seed 0 → +0.07. Same data!", GREY),
    ("Fix: run 5 seeds, report mean ± std → exposes the noise.", 0, BLUE, True),
    ("Wilcoxon signed-rank test across datasets → a p-value:", 0, BLUE, True),
    (1, "p < 0.05 = real effect ;  p > 0.05 = indistinguishable from baseline."),
    "Result: AUDA's win-rates dropped once luck was averaged out — single-seed numbers were optimistic."], size=18, gap=11)

# ===== 8. RIGOR 2 + JOURNEY =====
s = slide(); bg(s); title_bar(s, "Building Rigor #2 — A Fair Baseline")
bullets(s, [
    "The weak baseline never scaled features — fatal for SVM.",
    "AUDA's big SVM 'wins' were just AUDA fixing that mistake.",
    ("Fair baseline scales + one-hot encodes, like a real analyst.", 0, BLUE, True)], l=0.7, t=1.3, w=12, h=1.6, size=17, gap=6)
img(s, "_c_journey.png", 1.7, 2.95, w=10.0)
note(s, "As each artifact is removed, the SVM 'advantage' collapses from +0.05 to ~0.", t=6.95)

# ===== 9. DATA INTEGRITY =====
s = slide(); bg(s); title_bar(s, "Data Integrity — The Datasets Were Corrupted")
bullets(s, [
    ("Hash check revealed 4 DUPLICATE pairs masquerading as different datasets:", 0, RED, True),
    (1, "breast_cancer_wide = qsar_bio ;  arcene = colon_cancer"),
    (1, "ionosphere = musk ;  sonar = banknote"),
    "Worse — filenames lied: all four 'wide' files were breast-cancer data; 'sonar' actually held banknote data.",
    ("Cause: the dataset downloader saved wrong / duplicate files.", 0, GREY, False),
    ("Fix: removed duplicates (39 → 35); benchmark now de-dups by content hash.", 0, GREEN, True)], size=18, gap=10)

# ===== 10. BUGS =====
s = slide(); bg(s); title_bar(s, "Three Real Bugs Found & Fixed")
table(s, [
    ["Bug", "Effect", "Example fix"],
    ["Label misalignment after row-drop", "Training labels scrambled when AUDA dropped rows", "flight_delay RF  −0.35 → 0.00"],
    ["Missing target became a junk class", "Models trained to predict \"missing\"", "weatheraus RF  −0.13 → −0.01"],
    ["Continuous floats dropped as 'IDs' (library)", "Deleted predictive features on small data", "vertebral RF  −0.056 → 0.00"],
], 0.7, 1.55, 12.0, 3.2, fontsize=14, col_w=[3.8, 4.6, 3.6])
note(s, "Two were measurement bugs (made AUDA look worse); the third was a genuine AUDA library bug — now improved.")

# ===== 11. JOURNEY TABLE =====
s = slide(); bg(s); title_bar(s, "The Results Journey — From False Claim to Truth")
table(s, [
    ["Stage", "SVM mean Δ", "Significant?", "Reality"],
    ["Original (single seed, weak baseline)", "+0.051", "— (no test)", "anecdote"],
    ["Multi-seed, weak baseline", "+0.030", "No (p=0.10)", "noise"],
    ["+ saturation off, deduplicated", "+0.091", "Yes (p=0.02)", "inflated by weak baseline"],
    ["+ FAIR baseline", "−0.037", "No (p=0.17)", "win evaporates"],
    ["+ all bugs fixed (FINAL)", "+0.003", "No (p=0.94)", "PARITY"],
], 0.55, 1.5, 12.3, 3.5, fontsize=13, col_w=[4.6, 2.3, 2.4, 3.0])
note(s, "Each step removed an artifact. The honest endpoint: AUDA neither beats nor harms a competent baseline.")

# ===== 12. FINAL RESULT =====
s = slide(); bg(s); title_bar(s, "Final Result — Statistical Parity")
img(s, "_c_parity.png", 0.6, 1.5, w=6.3)
bullets(s, [
    ("RandomForest:  −0.001   (p = 0.29)", 0, GREEN, True),
    ("SVM:  +0.003   (p = 0.94)", 0, PURPLE, True),
    ("Both ≈ 0, neither significant.", 0, NAVY, True),
    "Across 32 datasets, 5 seeds.",
    "AUDA matches competent manual preprocessing — no significant difference."], l=7.2, t=1.9, w=5.6, h=4.5, size=17, gap=13)

# ===== 13. PER-DATASET =====
s = slide(); bg(s); title_bar(s, "Per-Dataset View — Mostly Parity")
img(s, "_c_perdataset.png", 1.1, 1.45, w=11.1)
note(s, "Most datasets sit at ~0 (parity). A few small wins/losses roughly cancel — no systematic advantage either way.")

# ===== 14. CONCLUSION =====
s = slide(); bg(s, NAVY)
centered(s, 1.0, 1.3, 11.3, 1.0, [("Honest Conclusion", 34, WHITE, True, False)])
centered(s, 1.4, 2.7, 10.5, 3.0, [
    ("\"AUDA matches the performance of competent manual preprocessing — no statistically "
     "significant difference on tree or kernel models across 32 datasets — while requiring "
     "zero human effort.\"", 22, RGBColor(0xDC, 0xE6, 0xF2), False, True),
    (" ", 10, WHITE, False, False),
    ("AUDA does not beat a skilled analyst — but it equals one, automatically.", 18, SKY, True, False)])

# ===== 15. LESSONS =====
s = slide(); bg(s); title_bar(s, "Methodology Lessons")
bullets(s, [
    "Never trust a single train/test split — always multi-seed + a significance test.",
    "Your baseline must be fair — beating a deliberately weak baseline proves nothing.",
    "Validate your data — duplicates and mislabeled files silently corrupt results.",
    ("A neutral result, honestly obtained, beats an impressive result that can't survive scrutiny.", 0, GREEN, True),
    "Bugs can masquerade as 'findings' — the −0.35 'AUDA is destructive' was a label bug."], size=19, gap=15)

# ===== 16. FUTURE WORK =====
s = slide(); bg(s); title_bar(s, "Future Work — How AUDA Could Actually Win")
table(s, [
    ["Lever", "Idea", "Effort / payoff"],
    ["Feature creation", "Datetime decomposition, CV target encoding, interactions — features the baseline can't make", "High / only real upside"],
    ["Validation-driven preprocessing", "Try several preprocessing variants per dataset, keep best by CV (AutoML-style)", "High / highest ceiling"],
    ["Apply LLM suggestions", "Actually execute the LLM's feature ideas (currently report-only)", "Medium / uncertain"],
], 0.6, 1.55, 12.2, 3.0, fontsize=13, col_w=[3.2, 6.4, 2.6])
note(s, "Parity is already a complete, defensible result. These are paths to push AUDA from parity to advantage.")

# ===== 17. SUMMARY =====
s = slide(); bg(s, NAVY)
centered(s, 1.0, 0.7, 11.3, 0.9, [("Summary", 34, WHITE, True, False)])
rows = [("Built AUDA: automated EDA + preprocessing pipeline.", GREEN),
        ("Tested rigorously: 32 datasets, RF + SVM, 5 seeds, significance test, fair baseline.", SKY),
        ("Found & fixed 3 real bugs and corrupted datasets along the way.", AMBER),
        ("Result: AUDA achieves PARITY with competent manual preprocessing — automatically.", WHITE),
        ("A credible, defensible thesis — and a clean path forward.", SKY)]
tb = box(s, 1.4, 2.0, 10.6, 4.4); tf = tb.text_frame; tf.word_wrap = True; first = True
for txt, col in rows:
    p = tf.paragraphs[0] if first else tf.add_paragraph(); first = False
    p.space_after = Pt(16); r = p.add_run(); r.text = "•  " + txt
    r.font.size = Pt(19); r.font.color.rgb = col; r.font.bold = (col == WHITE)
note(s, "Thank you — questions?", t=6.7, color=RGBColor(0x8A, 0x9B, 0xB5), size=16)

out = "AUDA_Project_Presentation.pptx"
try:
    prs.save(out)
except PermissionError:
    out = "AUDA_Project_Presentation_DETAILED.pptx"
    prs.save(out)
    print("NOTE: original file was open/locked — saved to new file instead.")
for f in ("_c_perdataset.png", "_c_parity.png", "_c_journey.png"):
    if os.path.exists(f):
        os.remove(f)
print("Saved", out, "with", len(prs.slides._sldIdLst), "slides")
