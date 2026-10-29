"""Shared publication-figure styling.

Fonts, type sizes and colors for every figure panel; colors come from
code/config/palette.json (Okabe-Ito, colorblind-safe).

Usage:
    from figure_style import apply_publication_style, get_feature_category_colors, get_color
    apply_publication_style()                 # set rcParams once, before plotting
    get_color("model_pathways.rf_test")       # -> hex color
    get_neutral("rule")                       # unencoded ink for rules, grids, annotations

Run `python tools/figure_style.py` for a palette self-test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

# ---- Module constants ----
_THIS_DIR = Path(__file__).resolve().parent
_PALETTE_PATH = _THIS_DIR.parent / "config" / "palette.json"

# Sans-serif fallback chain; DejaVu Sans always ships with matplotlib.
_DEFAULT_FONT_STACK = ["Arial", "Helvetica", "Helvetica Neue", "DejaVu Sans"]

# Cache for the parsed palette
_palette_cache: Optional[Dict] = None


# ---- Palette loading ----
def load_palette(path: Optional[str] = None, force_reload: bool = False) -> Dict:
    """Load the color palette JSON (default code/config/palette.json); cached unless force_reload."""
    global _palette_cache

    if _palette_cache is not None and not force_reload and path is None:
        return _palette_cache

    palette_path = Path(path) if path is not None else _PALETTE_PATH
    if not palette_path.exists():
        raise FileNotFoundError(f"Palette file not found: {palette_path}")

    with open(palette_path, "r") as fh:
        palette = json.load(fh)

    if path is None:
        _palette_cache = palette
    return palette


# ---- matplotlib styling ----
def apply_publication_style(font_family: Optional[str] = None) -> None:
    """Apply the publication style to matplotlib rcParams (fonts, type sizes, spines, PDF font embedding).

    font_family, if given, is prepended to the Arial -> Helvetica -> DejaVu Sans fallback stack.
    """
    import matplotlib as mpl

    stack = ([font_family] + _DEFAULT_FONT_STACK) if font_family else list(_DEFAULT_FONT_STACK)
    neutrals = get_neutrals()

    mpl.rcParams.update(
        {
            # Fonts
            "font.family": "sans-serif",
            "font.sans-serif": stack,
            # mathtext must use the same font family
            "mathtext.fontset": "custom",
            "mathtext.rm": stack[0],
            "mathtext.it": f"{stack[0]}:italic",
            "mathtext.bf": f"{stack[0]}:bold",
            "mathtext.sf": stack[0],
            "mathtext.default": "regular",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.titleweight": "bold",
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "legend.title_fontsize": 7,
            "figure.titlesize": 9,
            "figure.titleweight": "bold",
            # Spines / axes (neutral ink from the palette)
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "axes.edgecolor": neutrals["ink"],
            "axes.labelcolor": neutrals["ink"],
            "axes.grid": False,
            "grid.color": neutrals["grid"],
            "grid.linewidth": 0.4,
            "text.color": neutrals["ink"],
            "xtick.color": neutrals["ink"],
            "ytick.color": neutrals["ink"],
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            # Lines / markers
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            # hairline hatching
            "hatch.linewidth": 0.4,
            # Legend
            "legend.frameon": False,
            # Output: vector-safe, fonts embedded (no Type-3)
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            # zero save padding keeps the saved MediaBox at the authored width
            "savefig.pad_inches": 0.0,
            "savefig.dpi": 300,
            "figure.dpi": 150,
        }
    )


# ---- Color accessors ----
def get_color(dotted_path: str, palette: Optional[Dict] = None) -> str:
    """Look up a color by dotted key path, e.g. "model_pathways.rf_test"; raise KeyError if unresolved."""
    pal = palette if palette is not None else load_palette()
    node: Union[Dict, str] = pal
    for key in dotted_path.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"Palette path not found: {dotted_path!r} (missing {key!r})")
        node = node[key]
    if not isinstance(node, str):
        raise KeyError(f"Palette path {dotted_path!r} does not resolve to a color string")
    return node


def get_palette(name: str, palette: Optional[Dict] = None) -> Union[List, Dict]:
    """Return a named palette group (e.g. ``"qualitative"``, ``"feature_groups"``)."""
    pal = palette if palette is not None else load_palette()
    if name not in pal:
        raise KeyError(f"Palette group not found: {name!r}")
    return pal[name]


def get_feature_category_colors(palette: Optional[Dict] = None) -> Dict[str, str]:
    """Return the feature-group color map {demographic, sample_type, technical}."""
    pal = palette if palette is not None else load_palette()
    return dict(pal["feature_groups"])


def categorize_feature(
    feature: str, default: Optional[str] = None, palette: Optional[Dict] = None
) -> Optional[str]:
    """Categorize a raw feature name as demographic / sample_type / technical (else `default`)."""
    pal = palette if palette is not None else load_palette()
    for group, members in pal["feature_group_members"].items():
        if feature in members:
            return group
    return default


def get_feature_color(
    feature: str, default: Optional[str] = None, palette: Optional[Dict] = None
) -> Optional[str]:
    """Return the feature-group color for a raw feature name (via :func:`categorize_feature`)."""
    pal = palette if palette is not None else load_palette()
    group = categorize_feature(feature, palette=pal)
    if group is None:
        return default
    return pal["feature_groups"][group]


def get_categorical_colors(feature: str, palette: Optional[Dict] = None) -> Dict[str, str]:
    """Return the value->color map for a categorical feature, or ``{}`` if undefined."""
    pal = palette if palette is not None else load_palette()
    return dict(pal.get("categorical_values", {}).get(feature, {}))


def get_neutrals(palette: Optional[Dict] = None) -> Dict[str, str]:
    """Return the neutral ink ramp (ink, annotation, rule, muted, missing, inactive, grid, band, on_fill)."""
    pal = palette if palette is not None else load_palette()
    return dict(pal["neutrals"])


def get_neutral(name: str, palette: Optional[Dict] = None) -> str:
    """Return one neutral by name (see :func:`get_neutrals`)."""
    pal = palette if palette is not None else load_palette()
    neutrals = pal["neutrals"]
    if name not in neutrals:
        raise KeyError(f"Unknown neutral {name!r}; available: {sorted(neutrals)}")
    return neutrals[name]


def get_graphviz_theme(palette: Optional[Dict] = None) -> Dict[str, str]:
    """Return the graphviz node/edge theme for the STROBE flow diagram."""
    pal = palette if palette is not None else load_palette()
    return dict(pal["graphviz_theme"])


# ---- Contrast / grayscale legibility ----
def relative_luminance(hex_color: str) -> float:
    """WCAG 2.x relative luminance of a hex color, in [0, 1]."""
    s = hex_color.lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        raise ValueError(f"Not a hex color: {hex_color!r}")
    channels = []
    for i in (0, 2, 4):
        c = int(s[i : i + 2], 16) / 255.0
        channels.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(hex_a: str, hex_b: str) -> float:
    """WCAG contrast ratio between two colors (1.0 = identical in grayscale, 21.0 = max)."""
    la, lb = relative_luminance(hex_a), relative_luminance(hex_b)
    lighter, darker = max(la, lb), min(la, lb)
    return (lighter + 0.05) / (darker + 0.05)


#: Pairs that must be distinguishable within one figure: (label, path_a, path_b).
SEMANTIC_PAIRS = [
    ("model: RF vs LR (test)", "model_pathways.rf_test", "model_pathways.lr_test"),
    ("model: RF vs LR (train)", "model_pathways.rf_train", "model_pathways.lr_train"),
    ("lit status: previously reported vs unresolved", "hit_status.confirmed", "hit_status.unresolved"),
    ("direction: up vs down", "direction.positive", "direction.negative"),
    ("feature: demographic vs sample type", "feature_groups.demographic", "feature_groups.sample_type"),
    ("feature: sample type vs technical", "feature_groups.sample_type", "feature_groups.technical"),
    ("feature: demographic vs technical", "feature_groups.demographic", "feature_groups.technical"),
    ("outcome: actionable vs not", "outcome.actionable", "outcome.non_actionable"),
    ("neutral: hit vs not-significant", "neutrals.muted", "neutrals.inactive"),
]

#: Grayscale contrast below which a pair needs a redundant channel (shape, hatch, label).
MIN_PAIR_CONTRAST = 1.5


def check_semantic_contrast(palette: Optional[Dict] = None) -> List[Dict[str, object]]:
    """Measure grayscale contrast for each pair in SEMANTIC_PAIRS (reports, does not enforce)."""
    pal = palette if palette is not None else load_palette()
    out = []
    for label, path_a, path_b in SEMANTIC_PAIRS:
        ca, cb = get_color(path_a, pal), get_color(path_b, pal)
        ratio = contrast_ratio(ca, cb)
        out.append(
            {
                "label": label,
                "colors": (ca, cb),
                "luminance": (relative_luminance(ca) * 100, relative_luminance(cb) * 100),
                "contrast": ratio,
                "passes": ratio >= MIN_PAIR_CONTRAST,
            }
        )
    return out


# ---- Panel geometry ----
MM_PER_INCH = 25.4

#: Standard on-page slot widths (mm); must match the assemble_figure*.py defaults.
SLOT_FULL_MM = 183.0
SLOT_HALF_MM = 88.25
SLOT_COLUMN_MM = 86.0

#: Vertical pitch per data row for content-driven panels (fits a 6 pt label).
ROW_MM = 4.0

#: Fixed vertical overhead outside the data rows (ticks, axis title, legend).
HEADER_MM = 16.0


def panel_size_in(
    width_mm: float,
    height_mm: Optional[float] = None,
    n_rows: Optional[int] = None,
    row_mm: float = ROW_MM,
    header_mm: float = HEADER_MM,
    min_height_mm: float = 22.0,
) -> tuple:
    """Return a matplotlib figsize (inches) for a panel authored at its print width.

    Pass height_mm for a fixed height, or n_rows for a content-driven height
    (header_mm + n_rows * row_mm), so more data gives a taller panel rather than smaller type.
    """
    if (height_mm is None) == (n_rows is None):
        raise ValueError("panel_size_in: pass exactly one of height_mm or n_rows")
    if n_rows is not None:
        height_mm = header_mm + n_rows * row_mm
    height_mm = max(float(height_mm), min_height_mm)
    return (width_mm / MM_PER_INCH, height_mm / MM_PER_INCH)


# ---- Plot helpers ----
def add_panel_label(
    ax,
    letter: str,
    x: float = -0.08,
    y: float = 1.02,
    fontsize: int = 8,
    fontweight: str = "bold",
    **kwargs,
):
    """Draw a panel label (A/B/C...) in axes-fraction coordinates; for figure-assembly scripts."""
    return ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
        va="bottom",
        ha="right",
        **kwargs,
    )


# ---- Self-test ----
if __name__ == "__main__":
    print("Balder figure-style palette self-test")
    print("=" * 60)
    pal = load_palette()

    meta = pal.get("metadata", {})
    print(f"version       : {meta.get('version')}")
    print(f"palette family: {meta.get('palette_family')}")
    print(f"source file   : {_PALETTE_PATH}")
    print()

    print("Feature groups (3-way split):")
    cat_colors = get_feature_category_colors(pal)
    for group, color in cat_colors.items():
        members = pal["feature_group_members"].get(group, [])
        print(f"  {group:12s} {color}  ({len(members)} features)")
    print()

    print("Member -> group categorization:")
    n_features = 0
    for group, members in pal["feature_group_members"].items():
        for feat in members:
            resolved = categorize_feature(feat, palette=pal)
            assert resolved == group, f"categorize_feature({feat!r}) = {resolved!r} != {group!r}"
            n_features += 1
    print(f"  OK - all {n_features} features categorize to their declared group")
    print()

    print("Named color lookups:")
    for path in ("model_pathways.rf_test", "hit_status.unresolved", "direction.negative"):
        print(f"  {path:24s} -> {get_color(path, pal)}")
    print()

    print("Categorical value maps:")
    for feat in ("SEX", "ANCESTRY", "SAMPLE_TYPE_DETAILED", "ETHNICITY"):
        vals = get_categorical_colors(feat, pal)
        print(f"  {feat:22s} {len(vals)} values")
    print()

    print("Neutral ink ramp (unencoded vocabulary):")
    usage = pal.get("neutrals_usage", {})
    for name, color in get_neutrals(pal).items():
        lum = relative_luminance(color) * 100
        first_sentence = usage.get(name, "").split(".")[0]
        print(f"  {name:11s} {color}  L={lum:5.1f}  {first_sentence}")
    assert set(get_neutrals(pal)) >= {
        "ink",
        "annotation",
        "rule",
        "muted",
        "missing",
        "inactive",
        "grid",
        "band",
        "on_fill",
    }, "neutrals block is missing a documented key"
    # the legacy aliases must stay equal to their entries in the neutrals ramp
    assert pal["neutral"] == get_neutral("muted", pal), "neutral != neutrals.muted"
    assert pal["missing_indicator"] == get_neutral("missing", pal), (
        "missing_indicator != neutrals.missing"
    )
    print("  OK - ramp complete; legacy `neutral` / `missing_indicator` aliases agree")
    print()

    print("Grayscale contrast of semantic pairs (WCAG; >= 1.5 survives mono print):")
    results = check_semantic_contrast(pal)
    for rec in results:
        ca, cb = rec["colors"]
        la, lb = rec["luminance"]
        flag = "ok " if rec["passes"] else "LOW"
        print(
            f"  [{flag}] {rec['label']:38s} {ca} / {cb}  "
            f"L={la:4.1f}/{lb:4.1f}  {rec['contrast']:.2f}:1"
        )
    n_low = sum(1 for r in results if not r["passes"])
    if n_low:
        print(
            f"\n  NOTE: {n_low} of {len(results)} pairs fall below {MIN_PAIR_CONTRAST}:1 and are "
            f"indistinguishable\n        in grayscale. That is permitted ONLY where the figure also "
            f"encodes the\n        distinction by shape, position or a direct label (style guide: "
            f"'don't encode\n        a distinction by color alone'). Okabe-Ito separates hue, not "
            f"luminance --\n        this is not a defect to fix inside the palette."
        )
    print()

    print("Graphviz theme (Figure 2 STROBE):")
    gv = get_graphviz_theme(pal)
    print(f"  {len(gv)} keys; font={gv['font']!r} / {gv['font_bold']!r}")
    for key in ("font", "font_bold"):
        assert gv[key].startswith("Arial"), f"graphviz_theme.{key} is off the font spec: {gv[key]!r}"
    print()

    # Validate every declared hex looks like a hex color.
    def _walk(node):
        if isinstance(node, dict):
            for v in node.values():
                yield from _walk(v)
        elif isinstance(node, list):
            for v in node:
                yield from _walk(v)
        elif isinstance(node, str):
            yield node

    bad = [
        s
        for s in _walk(
            {k: v for k, v in pal.items() if k not in ("metadata", "sequential", "sequential_alt")}
        )
        if s.startswith("#") and len(s) not in (4, 7)
    ]
    assert not bad, f"Malformed hex colors: {bad}"
    print("OK - all hex colors well-formed")
    print("Self-test passed.")
