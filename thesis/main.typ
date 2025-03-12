// vim.lsp.get_clients()[1].request('workspace/executeCommand', { command = 'tinymist.pinMain', arguments = {vim.api.nvim_buf_get_name(0)} })
#import "lib.typ": *
#show: thm-rules

#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#set page(numbering: "1")
#set figure(placement: auto)

// LaTeX look -------------------------------------------------
#set page(margin: 1.5in)
//#set par(leading: 0.55em, spacing: 0.55em, first-line-indent: 1.8em, justify: true)
#set par(leading: 0.55em, spacing: 1.2em, first-line-indent: 0em, justify: true)
#set text(font: "New Computer Modern")
#show raw: set text(font: "New Computer Modern Mono")
#show heading: set block(above: 1.4em, below: 1em)
// ------------------------------------------------------------

#outline(title: "Table of contents")
#pagebreak()

#include "ExplainableAI.typ"
#include "k-means.typ"
#include "methods.typ"

#pagebreak()
#bibliography("bibliography.bib", style: "springer-vancouver")
