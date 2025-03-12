// vim.lsp.get_clients()[1].request('workspace/executeCommand', { command = 'tinymist.pinMain', arguments = {vim.api.nvim_buf_get_name(0)} })
#import "lib.typ": *
#import "./front_page.typ": front_page
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

#front_page(
  title: "Counterfactuals for Centroid-based clustering algorithms",
  authors: (
    "Jacob Pedersen",
    "Jeppe Allerslev",
    "Kristian Mortensen" 
  ),
  numbers: (
    "202008056",
    "202007296",
    "202007843"
  ),
  date: datetime.today().display()
)

#pagebreak()

#heading(numbering: none, "Project Description", outlined: false)
This project aims to explore the field of explainable AI (XAI). Specifically, we will conduct an exploratory analysis of the current state-of-the-art methods for producing counterfactuals for centroid-based clustering algorithms. For supervised methods, the concept of a counterfactual is more clearly defined as a minimal change that would change the classification of a given point. For clustering, the concept is less intuitive due to the lack of a ground truth, but may be defined as a minimal change that would change the assigned cluster for a given point. Additionally, the clustering might change in response to a counterfactual, moving the cluster boundaries. This might impact validity and similarity of the counterfactual. We aim to also study this phenomenon.

While counterfactuals for classification is a well-studied area, there is less literature available for clustering methods. We will conduct initial research by gathering existing literature. We will then conduct experiments and analyse existing methods. Furthermore, we aim to develop novel methods and evaluate them against existing methods.

#pagebreak()

#heading(numbering: none, "Abstract", outlined: false)
#lorem(80)

#pagebreak()

#outline(title: "Table of contents")

#heading(numbering: none, outlined: false, "Preliminary TOC")
+ Introduction (3-5 pages)
+ Preliminaries (8-10 pages)
  - XAI introduction
  - K-means introduction
  - Other centroid-based algorithms?
+ Related Work (\~15 pages)
  - Supervised counterfactual methods
  - Unsupervised counterfactual methods
+ Novel methods (15-20 pages)
  - Accounting for cluster changes when transforming counterfactuals
  - Our proposed methods (presentation, pseudocode, analysis etc.)
+ Experiments (5-8 pages)
  - Datasets & preprocessing
  - Evaluation metrics
  - Baseline method
+ Results (5-8 pages)
  - Presentation of results (tables, figures, digits examples)
+ Discussion (3-5 pages)
+ Conclusion (2-4 pages)
 
 Total: 56-75 pages

#pagebreak()

#include "chapters/ExplainableAI.typ"
#include "chapters/k-means.typ"
#include "chapters/methods.typ"
#include "chapters/novel_methods.typ"

#pagebreak()
#bibliography("bibliography.bib", style: "springer-vancouver")
