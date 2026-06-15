"""
Shared publication-figure styling for Balder visualizations.

Single Python entry point for consistent fonts, type sizes, and colors across
every figure panel. Colors come from ``code/config/palette.json`` (the single
source of truth, also read by ``code/tools/figure_theme.R``). The palette is the
colorblind-safe Okabe-Ito family; see ``figures/style/FIGURE_STYLE_GUIDE.md`` for
the documented spec.

Typical usage in a panel script::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code" / "tools"))
    from figure_style import apply_publication_style, get_feature_category_colors

    apply_publication_style()                     # set rcParams once, before plotting
    colors = get_feature_category_colors()        # {'demographic','sample_type','technical'}

Color lookups::

    from figure_style import get_color, get_palette, get_categorical_colors
    get_color("model_pathways.rf_test")           # -> '#0072B2'
    get_palette("qualitative")                    # -> ['#0072B2', '#E69F00', ...]
    get_categorical_colors("SEX")                 # -> {'Male': '#0072B2', 'Female': '#CC79A7'}

Panel labels (used by figure-assembly scripts, not individual panels)::

    from figure_style import add_panel_label
    add_panel_label(ax, "A")

Run ``python tools/figure_style.py`` for a self-test that validates the palette.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

# --------------------------------------------------------------------------- #
# Module constants
# --------------------------------------------------------------------------- #
_THIS_DIR = Path(__file__).resolve().parent
_PALETTE_PATH = _THIS_DIR.parent / "config" / "palette.json"

# Sans-serif fallback chain. Arial/Helvetica are the publication defaults; DejaVu
# Sans always ships with matplotlib so rendering never hard-fails when the
# preferred families are absent (matplotlib silently falls back down the list).
_DEFAULT_FONT_STACK = ["Arial", "Helvetica", "Helvetica Neue", "DejaVu Sans"]

# Cache for the parsed palette (mirrors feature_config.load_feature_config).
_palette_cache: Optional[Dict] = None


# --------------------------------------------------------------------------- #
# Palette loading
# --------------------------------------------------------------------------- #
def load_palette(path: Optional[str] = None, force_reload: bool = False) -> Dict:
    """
    Load the figure color palette from JSON.

    Parameters
    ----------
    path : str, optional
        Path to a palette JSON file. Defaults to ``code/config/palette.json``.
    force_reload : bool, default False
        Re-read from disk even if a cached palette is available.

    Returns
    -------
    dict
        Parsed palette mapping.

    Raises
    ------
    FileNotFoundError
        If the palette file does not exist.
    """
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


# --------------------------------------------------------------------------- #
# matplotlib styling
# --------------------------------------------------------------------------- #
def apply_publication_style(font_family: Optional[str] = None) -> None:
    """
    Apply the Balder publication style to matplotlib's global ``rcParams``.

    Sets font family + the type-size scale, hides top/right spines, and forces
    vector-PDF font embedding (``pdf.fonttype=42`` — no Type-3 fonts). Call once
    near the top of a plotting routine, before creating the figure. Individual
    scripts may still override any rcParam per-call.

    Type sizes are the on-page publication sizes (assume journal-column panel
    dimensions); panels rendered at large dev figsizes will look small-texted,
    which is expected — final dimensions are set during the panel sweep.

    Parameters
    ----------
    font_family : str, optional
        Preferred sans-serif family. Prepended to the default fallback stack
        (Arial -> Helvetica -> DejaVu Sans).
    """
    import matplotlib as mpl

    stack = ([font_family] + _DEFAULT_FONT_STACK) if font_family else list(_DEFAULT_FONT_STACK)

    mpl.rcParams.update(
        {
            # Fonts
            "font.family": "sans-serif",
            "font.sans-serif": stack,
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
            # Spines / axes
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "axes.grid": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            # Lines / markers
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            # Legend
            "legend.frameon": False,
            # Output: vector-safe, fonts embedded (no Type-3)
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "figure.dpi": 150,
        }
    )


# --------------------------------------------------------------------------- #
# Color accessors
# --------------------------------------------------------------------------- #
def get_color(dotted_path: str, palette: Optional[Dict] = None) -> str:
    """
    Look up a single color by a dotted key path.

    Parameters
    ----------
    dotted_path : str
        e.g. ``"model_pathways.rf_test"`` or ``"hit_status.novel"``.
    palette : dict, optional
        A pre-loaded palette; loaded from disk if omitted.

    Returns
    -------
    str
        Hex color string.

    Raises
    ------
    KeyError
        If any segment of the path is missing or does not resolve to a string.
    """
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
    """
    Return the 3-way feature-group color map.

    Drop-in replacement for the per-script ``CATEGORY_COLORS`` dict:
    ``{'demographic': hex, 'sample_type': hex, 'technical': hex}``.
    """
    pal = palette if palette is not None else load_palette()
    return dict(pal["feature_groups"])


def categorize_feature(
    feature: str, default: Optional[str] = None, palette: Optional[Dict] = None
) -> Optional[str]:
    """
    Categorize a raw feature name into ``demographic`` / ``sample_type`` /
    ``technical`` using ``feature_group_members`` in the palette.

    Parameters
    ----------
    feature : str
        Raw feature name (e.g. ``"SAMPLE_TYPE_DETAILED"``).
    default : str, optional
        Returned when the feature is not a member of any group.
    palette : dict, optional
        Pre-loaded palette.

    Returns
    -------
    str or None
        The group name, or ``default`` if unmatched.
    """
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


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def add_panel_label(
    ax,
    letter: str,
    x: float = -0.08,
    y: float = 1.02,
    fontsize: int = 8,
    fontweight: str = "bold",
    **kwargs,
):
    """
    Draw a consistent panel label (A/B/C...) in axes-fraction coordinates.

    Intended for figure-assembly scripts that compose multiple panels — not for
    individual panel scripts (a single panel script's output is one labeled panel
    of an assembled figure).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    letter : str
        Panel letter, e.g. ``"A"``.
    x, y : float
        Position in axes-fraction coordinates (default top-left, just outside).
    fontsize, fontweight :
        Text styling. Remaining kwargs pass through to ``ax.text``.
    """
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


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
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
    for path in ("model_pathways.rf_test", "hit_status.novel", "direction.negative"):
        print(f"  {path:24s} -> {get_color(path, pal)}")
    print()

    print("Categorical value maps:")
    for feat in ("SEX", "ANCESTRY", "SAMPLE_TYPE_DETAILED", "ETHNICITY"):
        vals = get_categorical_colors(feat, pal)
        print(f"  {feat:22s} {len(vals)} values")
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
