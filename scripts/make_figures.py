"""Draw the report's four SVG figures into paper/figures/ from paper/numbers.json.

Usage:
    .venv/bin/python scripts/make_figures.py
"""
import json, pathlib
from xml.sax.saxutils import escape

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, GREEN, SLATE = "#2a78d6", "#eb6834", "#1baf7a", "#6b7280"
FONT = "Inter, system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

MODEL_ORDER = ["qwen0.5b", "qwen1.5b", "bert", "mpnet", "clip", "qwen3b"]
MODEL_LABEL = {
    "qwen0.5b": "Qwen2.5-0.5B",
    "qwen1.5b": "Qwen2.5-1.5B",
    "bert": "BERT-base",
    "mpnet": "all-mpnet-base-v2",
    "clip": "CLIP ViT-B/32 text",
    "qwen3b": "Qwen2.5-3B (bf16)",
}


class Svg:
    def __init__(self, w, h, title, desc):
        self.w, self.h, self.parts = w, h, []
        self.title, self.desc = title, desc

    def add(self, s):
        self.parts.append(s)

    def rect(self, x, y, w, h, fill, rx=0, stroke=None, sw=1):
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{st}/>')

    def line(self, x1, y1, x2, y2, stroke=GRID, sw=1, dash=None, extra=""):
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"{ds}{extra}/>')

    def circle(self, cx, cy, r, fill, ring=False, stroke=None, sw=1.8):
        if ring:
            self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r + 2:.1f}" fill="{SURFACE}"/>')
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}"{st}/>')

    def polyline(self, pts, stroke, sw=2.2, dash=None):
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.add(f'<polyline points="{coords}" fill="none" stroke="{stroke}" stroke-width="{sw}"{ds}/>')

    def text(self, x, y, s, size=12, fill=INK, anchor="start", weight="400"):
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{escape(str(s))}</text>'
        )

    def save(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" '
            f'font-family="{FONT}" role="img" aria-labelledby="t d">'
            f'<title id="t">{escape(self.title)}</title>'
            f'<desc id="d">{escape(self.desc)}</desc>'
            f'<rect width="{self.w}" height="{self.h}" fill="{SURFACE}"/>'
        )
        (OUT / name).write_text(head + "".join(self.parts) + "</svg>\n")


def fig1(N):
    s = Svg(
        760,
        320,
        "How the replication study is organized",
        "Three phases on a single Apple M4 Mac: Phase 1 audits tests, CSVs and metric implementations; Phase 2 extracts and scores 6 open text models against DINOv2-B on WIT-1024; Phase 3 tests good views against bad views in finite worlds and small LeJEPA runs.",
    )
    s.text(24, 28, "Figure 1. How the replication is organized", 15, INK, weight="600")
    s.text(24, 46, "Frozen analysis plan before the first scored run; all compute on one Apple M4 Mac (16 GB memory).", 12, INK2)
    s.add(
        f'<defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{MUTED}"/></marker></defs>'
    )

    cards = [
        (
            24,
            BLUE,
            "Phase 1: Released repo and metrics",
            [
                "Pinned galilai-group/prh-proof commit",
                "135 of 139 upstream tests pass on CPU",
                "Rebuild headline numbers from CSVs",
                "Cross-check NumPy metrics vs PyTorch",
                "Test 0.95-clip under orthogonal rotation",
            ],
        ),
        (
            268,
            ORANGE,
            "Phase 2: WIT-1024 baselines",
            [
                "1,024 Wikipedia image-caption pairs",
                "Frozen DINOv2-B (12 blocks, 518 px)",
                "6 open text models up to Qwen2.5-3B",
                "Mutual 10-NN and unbiased linear CKA",
                "Sensitivity: unclipped and final-layer",
            ],
        ),
        (
            512,
            GREEN,
            "Phase 3: Good views vs bad views",
            [
                "Exact spectral check on 20 finite worlds",
                "Sample-density sweep (1 to 64 per state)",
                "3-seed BERT-T LeJEPA on COCO captions",
                "51,200 scenes per run, same seed & lr",
                "Scored on WIT-1024 (mKNN and CKA)",
            ],
        ),
    ]
    for x, col, title, lines in cards:
        s.rect(x, 68, 224, 152, "#ffffff", rx=6, stroke=col, sw=2)
        s.rect(x, 68, 6, 152, col)
        s.text(x + 16, 90, title, 12.2, INK, weight="600")
        for idx, line in enumerate(lines):
            s.text(x + 16, 112 + idx * 20, line, 11.3, INK2)

    s.line(248, 144, 268, 144, MUTED, 1.6, extra=' marker-end="url(#arr)"')
    s.line(492, 144, 512, 144, MUTED, 1.6, extra=' marker-end="url(#arr)"')

    s.rect(24, 240, 712, 58, "#f1f0ec", rx=6, stroke=GRID)
    s.text(38, 262, "Audit trail: results/phase1.json, results/phase2_wit1024.json, results/phase3_views.json -> paper/numbers.json", 11.8, INK, weight="600")
    s.text(38, 282, "Every number in the report and figures is rebuilt by scripts/report_numbers.py; claims are indexed in paper/claims.md.", 11.3, INK2)
    s.save("fig1-study-design.svg")


def fig2(N):
    rows = N["phase2"]["rows"]
    s = Svg(
        760,
        350,
        "WIT-1024 mutual 10-NN alignment with DINOv2-B",
        "Comparison of our independent WIT-1024 mutual 10-NN scores (with 95% bootstrap intervals) against the numbers reported by Balestriero and LeCun (2026) across 6 open text encoders.",
    )
    s.text(24, 28, "Figure 2. WIT-1024 mutual 10-NN with DINOv2-B: our runs against the paper's table", 15, INK, weight="600")
    s.text(24, 46, "All 6 open encoders match the paper within 0.0012 (pre-registered reproduction threshold: 0.010).", 12, INK2)

    # Legend
    s.circle(34, 68, 4.5, BLUE)
    s.line(24, 68, 44, 68, BLUE, 2)
    s.text(52, 72, "Our measurement (95% bootstrap CI)", 11.5, INK2)
    s.circle(290, 68, 5.0, "none", stroke=ORANGE, sw=2)
    s.text(302, 72, "Paper's CSV value", 11.5, INK2)
    s.line(432, 68, 454, 68, GREEN, 1.8, dash="5,4")
    s.text(462, 72, "Paper's 109M good-view JEPA (.184, second-hand)", 11.5, INK2)

    x0, x1 = 190, 670
    y_top, y_bot = 96, 296
    max_v = 0.20

    def x_of(v):
        return x0 + (x1 - x0) * (v / max_v)

    for tick in (0.00, 0.05, 0.10, 0.15, 0.20):
        xt = x_of(tick)
        s.line(xt, y_top, xt, y_bot, GRID)
        s.text(xt, y_bot + 18, f"{tick:.2f}", 11, MUTED, "middle")

    # Shuffled floor (~0.009) and paper's 109M JEPA (.1837)
    x_shuf = x_of(0.0092)
    s.line(x_shuf, y_top, x_shuf, y_bot, MUTED, 1.3, dash="3,3")
    s.text(x_shuf + 4, y_top + 11, "chance (.009)", 10.5, MUTED)

    x_jepa = x_of(N["phase1"]["headline_numbers"]["wit1024_dinov2_b_headline"]["jepa_109m_mean"])
    s.line(x_jepa, y_top, x_jepa, y_bot, GREEN, 1.8, dash="5,4")
    s.text(x_jepa - 5, y_top + 11, "JEPA-109M (.184)", 10.5, GREEN, "end")

    row_h = (y_bot - y_top - 20) / len(MODEL_ORDER)
    for idx, key in enumerate(MODEL_ORDER):
        r = rows[key]
        ym = y_top + 28 + idx * row_h
        s.text(x0 - 12, ym + 4, MODEL_LABEL[key], 12, INK, "end", weight="500")
        lo, hi = r["mutual_knn_ci95_bootstrap"]
        s.line(x_of(lo), ym, x_of(hi), ym, BLUE, 2.4)
        s.line(x_of(lo), ym - 4, x_of(lo), ym + 4, BLUE, 1.6)
        s.line(x_of(hi), ym - 4, x_of(hi), ym + 4, BLUE, 1.6)
        s.circle(x_of(r["paper_reported_mknn"]), ym, 5.8, "none", stroke=ORANGE, sw=2.0)
        s.circle(x_of(r["mutual_knn"]), ym, 3.8, BLUE)
        s.text(x1 + 10, ym + 4, f"{r['mutual_knn']:.3f} (paper {r['paper_reported_mknn']:.3f})", 11, INK2)

    s.text((x0 + x1) / 2, 334, "Mutual 10-NN overlap with DINOv2-B on WIT-1024", 11.5, INK2, "middle")
    s.save("fig2-wit1024-baselines.svg")


def fig3(N):
    rows = N["phase2"]["rows"]
    rot = N["phase1"]["metrics_crosscheck"]["rotation_invariance_check_n1024_d64"]
    s = Svg(
        760,
        376,
        "Sensitivity of WIT-1024 alignment to 0.95-quantile coordinate clipping and layer selection",
        "Grouped bars comparing standard PRH mutual 10-NN (best layer with 0.95-quantile clipping), final-layer mutual 10-NN, and unclipped best-layer mutual 10-NN across 6 open text encoders.",
    )
    s.text(24, 28, "Figure 3. Coordinate clipping and layer choice strongly move LLM scores on WIT-1024", 15, INK, weight="600")
    s.text(24, 46, "Removing the 0.95-quantile elementwise clip cuts Qwen2.5-3B by 35% (.126 to .082), while mpnet barely moves (.118 to .116).", 11.8, INK2)

    # Legend
    leg = [
        (BLUE, "Standard PRH (best layer, 0.95 clip)"),
        (GREEN, "Final layer only (0.95 clip)"),
        (ORANGE, "Best layer, no coordinate clip"),
    ]
    lx = 24
    for col, lab in leg:
        s.rect(lx, 60, 13, 10, col, rx=2)
        s.text(lx + 18, 69, lab, 11.3, INK2)
        lx += 238

    x0, x1 = 188, 520
    y_top, y_bot = 88, 336
    max_v = 0.145

    def w_of(v):
        return (x1 - x0) * (v / max_v)

    for tick in (0.00, 0.04, 0.08, 0.12):
        xt = x0 + w_of(tick)
        s.line(xt, y_top, xt, y_bot, GRID)
        s.text(xt, y_bot + 16, f"{tick:.2f}", 11, MUTED, "middle")

    group_h = (y_bot - y_top) / len(MODEL_ORDER)
    for idx, key in enumerate(MODEL_ORDER):
        r = rows[key]
        yg = y_top + idx * group_h + 6
        s.text(x0 - 10, yg + 17, MODEL_LABEL[key], 11.8, INK, "end", weight="500")
        vals = [
            (r["mutual_knn"], BLUE),
            (r["last_layer_mutual_knn"], GREEN),
            (r["unclipped_max_mutual_knn"], ORANGE),
        ]
        for b_idx, (val, col) in enumerate(vals):
            by = yg + b_idx * 9.5
            bw = w_of(val)
            s.rect(x0, by, bw, 7.5, col, rx=1.5)
            s.text(x0 + bw + 5, by + 7, f"{val:.3f}", 9.8, INK2)

    # Callout box on the right explaining the two-sided nature of the 0.95 clip
    s.rect(556, 94, 184, 232, "#ffffff", rx=6, stroke=GRID)
    s.text(568, 116, "Two sides of prh_prep", 12.2, INK, weight="600")
    note_lines = [
        "1. Helps causal LLMs:",
        "   Autoregressive hidden",
        "   states develop extreme",
        "   outlier coordinates.",
        "   Clipping at the 0.95",
        "   quantile restores cosine",
        "   neighbours (+.044 on 3B).",
        "",
        "2. Breaks rotation symmetry:",
        "   On N=1024, d=64 Gaussian",
        "   features and an exact",
        "   orthogonal rotation Q,",
        f"   raw mKNN is {rot['raw_mutual_knn']:.3f}, but",
        f"   clipped PRH drops to {rot['prh_score_mknn_after_quantile_clip']:.3f}.",
    ]
    for i, ln in enumerate(note_lines):
        w = "600" if ln.startswith(("1.", "2.")) else "400"
        s.text(568, 136 + i * 13.5, ln, 10.6, INK if w == "600" else INK2, weight=w)

    s.save("fig3-metric-sensitivity.svg")


def fig4(N):
    p3 = N["phase3"]
    curves = p3["finite_world"]["theorem4_density_curves"]
    summ = p3["neural_jepa"]["summary"]

    s = Svg(
        760,
        360,
        "Good views against bad views in finite worlds and at laptop neural scale",
        "Left panel: exact spectral alignment in finite worlds across sample density (1 to 64 samples per state). Right panel: 3-seed BERT-T LeJEPA trained for 400 steps (51,200 COCO scenes) scored on WIT-1024 mutual 10-NN and unbiased CKA.",
    )
    s.text(24, 28, "Figure 4. Good views vs bad views: exact finite worlds and laptop-scale LeJEPA", 15, INK, weight="600")
    s.text(24, 46, "Exact spectral theory separates cleanly; at 51,200 scenes on a laptop, CKA separates on all 3 seeds while 10-NN stays near floor.", 11.5, INK2)

    # Left panel: finite-world density curves
    lx0, lx1, ly0, ly1 = 66, 356, 92, 294
    s.text(lx0, 76, "A. Finite-world spectral alignment (prh.theory)", 12.2, INK, weight="600")
    densities = [1, 2, 4, 8, 16, 32, 64]

    def lx_of(idx):
        return lx0 + (lx1 - lx0) * idx / (len(densities) - 1)

    def ly_of(val):
        return ly1 - (ly1 - ly0) * val

    for yv in (0.0, 0.25, 0.50, 0.75, 1.0):
        yy = ly_of(yv)
        s.line(lx0, yy, lx1, yy, GRID)
        s.text(lx0 - 8, yy + 4, f"{yv:.2f}", 10.5, MUTED, "end")

    for idx, d in enumerate(densities):
        xx = lx_of(idx)
        s.line(xx, ly0, xx, ly1, GRID)
        s.text(xx, ly1 + 16, str(d), 10.5, MUTED, "middle")
    s.text((lx0 + lx1) / 2, ly1 + 34, "Samples per world state (log2 spacing)", 11, INK2, "middle")

    series = [
        ("good_literal_gap050", GREEN, None),
        ("bad_r002_literal_gap050", BLUE, None),
        ("bad_r002_mixed_gap000", ORANGE, "5,3"),
    ]
    for key, col, dash in series:
        pts = [(lx_of(i), ly_of(curves[key][str(d)]["mean"])) for i, d in enumerate(densities)]
        s.polyline(pts, col, 2.2, dash=dash)
        for px, py in pts:
            s.circle(px, py, 3.0, col)

    s.text(lx0 + 6, ly_of(1.0) - 6, "Good views (1.00 at n>=1)", 10.3, GREEN, weight="600")
    s.text(lx0 + 38, ly_of(0.68), "Bad r=0.02, separated", 10.2, BLUE)
    s.text(lx0 + 96, ly_of(0.52), "Bad r=0.02, mixed (caps at .89)", 10.2, ORANGE)

    # Right panel: 3-seed BERT-T LeJEPA at 51,200 scenes
    rx0, rx1, ry0, ry1 = 434, 728, 92, 294
    s.text(rx0, 76, "B. Laptop BERT-T LeJEPA (51.2k scenes, 3 seeds)", 12.2, INK, weight="600")
    max_r = 0.085

    def ry_of(val):
        return ry1 - (ry1 - ry0) * (val / max_r)

    for yv in (0.00, 0.02, 0.04, 0.06, 0.08):
        yy = ry_of(yv)
        s.line(rx0, yy, rx1, yy, GRID)
        s.text(rx0 - 8, yy + 4, f"{yv:.2f}", 10.5, MUTED, "end")

    # Left half of Panel B: mutual 10-NN; Right half: unbiased CKA
    groups = [
        ("Mutual 10-NN", rx0 + 68, summ["init_mknn_seeds"], summ["bad_mknn_seeds"], summ["good_mknn_seeds"]),
        ("Unbiased CKA", rx0 + 216, [r["checkpoints"]["0"]["unbiased_cka"] for r in p3["neural_jepa"]["runs"] if r["arm"] == "good"], summ["bad_cka_seeds"], summ["good_cka_seeds"]),
    ]
    s.line(rx0 + 144, ry0, rx0 + 144, ry1, GRID, 1.2, dash="3,3")

    for g_title, cx, init_s, bad_s, good_s in groups:
        s.text(cx, ry1 + 18, g_title, 11.3, INK, "middle", weight="600")
        cols = [
            (cx - 42, "Init", init_s, SLATE),
            (cx, "Bad", bad_s, ORANGE),
            (cx + 42, "Good", good_s, GREEN),
        ]
        for bx, blab, vals, col in cols:
            mean_v = sum(vals) / len(vals)
            s.rect(bx - 13, ry_of(mean_v), 26, ry1 - ry_of(mean_v), col, rx=2)
            for si, sv in enumerate(vals):
                s.circle(bx + (si - 1) * 5.5, ry_of(sv), 2.6, "#ffffff", stroke=INK, sw=1.1)
            s.text(bx, ry_of(max(vals)) - 7, f"{mean_v:.3f}", 9.8, INK2, "middle")
            s.text(bx, ry1 + 33, blab, 10.2, MUTED, "middle")

    s.save("fig4-good-vs-bad-views.svg")


def main():
    N = json.loads((ROOT / "paper" / "numbers.json").read_text())
    fig1(N)
    fig2(N)
    fig3(N)
    fig4(N)
    print("Wrote 4 SVG figures to", OUT)


if __name__ == "__main__":
    main()
