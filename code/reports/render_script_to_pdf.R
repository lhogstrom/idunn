#install.packages(c("highlight","tinytex"))
#tinytex::install_tinytex()

library(highlight)
library(tinytex)

# Produce a standalone LaTeX file with colors
baseDir <- "/Users/larsonhogstrom/Documents/balder_analysis/balder/code/"
inFile <- paste0(baseDir,"model_oncokb_level1_hits.r")
outFile <- paste0(baseDir,"model_oncokb_level1_hits.tex")

highlight::highlight(inFile,
          renderer = highlight::renderer_latex(doc             = TRUE,
                                               showLineNumbers = TRUE,   # turn on line numbers
                                               numberLines     = TRUE,   # actually emit numbered lines
                                               lineNumberStep  = 1),       # show every line number),
          output   = outFile)

# Compile to PDF
tinytex::latexmk(outFile)  # creates myscript_highlight.pdf
