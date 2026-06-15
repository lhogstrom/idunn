# figure_theme.R — Balder publication figure theme for R / ggplot2.
#
# R-side counterpart to code/tools/figure_style.py. Both read the SAME single
# source of truth, code/config/palette.json, so Python (matplotlib) and R
# (ggplot2) figures share one colorblind-safe Okabe-Ito palette, font, and type
# scale. See figures/style/FIGURE_STYLE_GUIDE.md for the documented spec.
#
# Status: lightweight skeleton. Not yet wired into any R script — it exists so
# the later Fig 3A (circular institution heatmap) and Fig 5B (age patterns) R
# refactors inherit parity from day one.
#
# Usage:
#   source("code/tools/figure_theme.R")
#   library(ggplot2)
#   pal <- balder_palette("qualitative")          # character vector of hex
#   ggplot(df, aes(x, y, color = grp)) + geom_point() +
#     scale_color_balder() + theme_balder()
#
# Sourcing this file only DEFINES functions (no jsonlite/ggplot2 needed at source
# time); those packages are required lazily when the functions are called.

# --------------------------------------------------------------------------- #
# Path resolution + palette loading (lazy, cached)
# --------------------------------------------------------------------------- #

# Resolve this script's directory AT SOURCE TIME — whether run via Rscript
# (--file=) or source()'d (ofile in a parent frame). Captured now (in a local()
# that executes during sourcing) because `ofile` is only on the call stack while
# source() runs, not later when the accessor functions are actually called.
.balder_script_dir <- local({
  this_file <- NULL
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    this_file <- sub("^--file=", "", file_arg[1])
  } else {
    for (i in rev(seq_len(sys.nframe()))) {
      of <- tryCatch(get("ofile", envir = sys.frame(i), inherits = FALSE),
                     error = function(e) NULL)
      if (!is.null(of)) { this_file <- of; break }
    }
  }
  if (!is.null(this_file)) dirname(normalizePath(this_file)) else NA_character_
})

# Default palette path: code/config/palette.json (this file lives in code/tools).
# Try the script-relative location first, then common working-dir candidates so
# the function works regardless of cwd. Returns the first that exists.
.balder_default_palette_path <- function() {
  candidates <- character(0)
  if (!is.na(.balder_script_dir)) {
    candidates <- c(candidates,
                    file.path(.balder_script_dir, "..", "config", "palette.json"))
  }
  candidates <- c(
    candidates,
    file.path(getwd(), "code", "config", "palette.json"),  # cwd = repo root
    file.path(getwd(), "config", "palette.json"),          # cwd = code/
    file.path(getwd(), "..", "config", "palette.json")     # cwd = code/tools or code/analysis/*
  )
  for (cand in candidates) {
    if (file.exists(cand)) return(normalizePath(cand))
  }
  normalizePath(candidates[1], mustWork = FALSE)  # for the error message
}

# Cache so repeated calls don't re-read disk.
.balder_cache <- new.env(parent = emptyenv())

#' Load the Balder color palette from JSON.
#'
#' @param path Path to palette.json; defaults to code/config/palette.json.
#' @param force_reload Re-read from disk even if cached.
#' @return Named list parsed from the palette JSON.
load_balder_palette <- function(path = NULL, force_reload = FALSE) {
  if (is.null(path) && !force_reload && !is.null(.balder_cache$palette)) {
    return(.balder_cache$palette)
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("load_balder_palette() requires the 'jsonlite' package.")
  }
  palette_path <- if (is.null(path)) .balder_default_palette_path() else path
  if (!file.exists(palette_path)) {
    stop(sprintf("Palette file not found: %s", palette_path))
  }
  palette <- jsonlite::fromJSON(palette_path, simplifyVector = TRUE)
  if (is.null(path)) .balder_cache$palette <- palette
  palette
}

#' Return a named palette group (e.g. "qualitative", "feature_groups").
#'
#' @param name Top-level key in palette.json.
#' @param palette Optional pre-loaded palette.
#' @return The value at that key (character vector or named list).
balder_palette <- function(name = "qualitative", palette = NULL) {
  pal <- if (is.null(palette)) load_balder_palette() else palette
  if (!name %in% names(pal)) {
    stop(sprintf("Palette group not found: %s", name))
  }
  pal[[name]]
}

#' Look up a single color by dotted path, e.g. "model_pathways.rf_test".
balder_color <- function(dotted_path, palette = NULL) {
  pal <- if (is.null(palette)) load_balder_palette() else palette
  node <- pal
  for (key in strsplit(dotted_path, ".", fixed = TRUE)[[1]]) {
    if (!is.list(node) || is.null(node[[key]])) {
      stop(sprintf("Palette path not found: %s (missing %s)", dotted_path, key))
    }
    node <- node[[key]]
  }
  if (!is.character(node) || length(node) != 1) {
    stop(sprintf("Palette path %s does not resolve to a single color", dotted_path))
  }
  node
}

# --------------------------------------------------------------------------- #
# ggplot2 theme + scales (require ggplot2 when called)
# --------------------------------------------------------------------------- #

#' Balder publication ggplot2 theme.
#'
#' Type sizes match the on-page publication scale (axis title ~7pt, ticks ~6pt);
#' top/right spines removed; minimal grid. Mirrors apply_publication_style() in
#' figure_style.py.
#'
#' @param base_size Base font size in points.
#' @param base_family Font family (Arial/Helvetica default).
theme_balder <- function(base_size = 7, base_family = "Arial") {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("theme_balder() requires the 'ggplot2' package.")
  }
  ggplot2::theme_classic(base_size = base_size, base_family = base_family) +
    ggplot2::theme(
      plot.title       = ggplot2::element_text(size = base_size + 2, face = "bold"),
      axis.title       = ggplot2::element_text(size = base_size),
      axis.text        = ggplot2::element_text(size = base_size - 1),
      legend.title     = ggplot2::element_text(size = base_size),
      legend.text      = ggplot2::element_text(size = base_size - 1),
      legend.key       = ggplot2::element_blank(),
      strip.background = ggplot2::element_blank(),
      strip.text       = ggplot2::element_text(size = base_size, face = "bold")
    )
}

#' Discrete color scale from the Okabe-Ito qualitative palette.
scale_color_balder <- function(name = "qualitative", ...) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("scale_color_balder() requires the 'ggplot2' package.")
  }
  ggplot2::scale_color_manual(values = unlist(balder_palette(name)), ...)
}

#' Discrete fill scale from the Okabe-Ito qualitative palette.
scale_fill_balder <- function(name = "qualitative", ...) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("scale_fill_balder() requires the 'ggplot2' package.")
  }
  ggplot2::scale_fill_manual(values = unlist(balder_palette(name)), ...)
}
