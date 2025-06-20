// vim.lsp.get_clients()[1].request('workspace/executeCommand', { command = 'tinymist.pinMain', arguments = {vim.api.nvim_buf_get_name(0)} })
#import "lib.typ": *
#import "./front_page.typ": front_page
#show: thm-rules


#set heading(numbering: "1.1")

#set math.equation(numbering: "(1,1)")
#show: equate.with(sub-numbering: false, number-mode: "label")
#set figure(placement: auto)

#show figure: fig => {
  // let fig-width = measure(fig.body).width
  show figure.caption: c => box(width: 92.5%)[
    #let fig-num = numbering(fig.numbering, ..c.counter.at(fig.location()))
    *#c.supplement #fig-num#c.separator*#text()[#c.body]
  ]
  fig
}


#let appendix(body) = {
  set heading(numbering: "A.1.", supplement: [Appendix])
  counter(heading).update(0)
  body
}


// LaTeX look -------------------------------------------------
#set page(margin: 1.5in)
//#set par(leading: 0.55em, spacing: 0.55em, first-line-indent: 1.8em, justify: true)
#set par(leading: 0.55em, spacing: 1.2em, first-line-indent: 0em, justify: true)
#set text(size: 11pt, font: "New Computer Modern")
#show raw: set text(font: "New Computer Modern Mono")
#show heading: set block(above: 1.4em, below: 1em)
// ------------------------------------------------------------
// Front Page
#let authors = (
    "Jacob Pedersen",
    "Jeppe Allerslev",
    "Kristian Mortensen" 
)
#let numbers = (
    "202008056",
    "202007296",
    "202007843"
)
#let title = [Developing and Evaluating Counterfactual Explanation Methods for \ $k$-means Clustering]
#set document(author: authors, title: title)

#align(horizon+center)[
  #block(text(weight: 700, 1.9em, title))
  #v(4em)
  #image("assets/logo.png", width: 25%)
  #v(4em)
  #grid(
    columns: authors.len(),
    gutter: 1em,
    ..authors.map(author => align(center, strong(author))),
    ..numbers.map(number => align(center, number)),
  )
  #block(text(weight: 300, 1.25em, [Department of Computer Science \ Aarhus University]))
  #v(5em)
  #block(text(weight: 500, 1.5em, [Master's Thesis]))
  #v(1em, weak: true)
  #text(1.2em, [June 2025])

]
// ------------------------------------------------------------

#pagebreak()
- *Advisor*: Ira Assent - ira\@cs.au.dk
- *Co-advisors*: 
  - Pernille Matthews - matthews\@cs.au.dk
  - Tommaso Amico - tomam\@cs.au.dk

#v(2em, weak: true)

*Code*: #link("https://github.com/jepp3183/Thesis2025")[#underline[GitHub Repository]]

#v(2em, weak: true)

*Contributions:* Each group member has contributed equally to all parts of the following thesis.

#v(2em, weak: true)

*Generative AI Statement*\
There has been minimal use of generative AI (GAI) throughout this thesis. Google Gemini has been used for rephrasing a minor amount of sentences, and Copilot has been used to assist in creating the code needed to generate plots.
  
#pagebreak()


#heading(numbering: none, "Project Description", outlined: false)
This project aims to explore the field of explainable artificial intelligence (XAI). Specifically, we will conduct an exploratory analysis of the current state of the art methods for producing counterfactuals for the $k$-means clustering task. For supervised methods, the concept of a counterfactual is clearly defined as a minimal modification that would change the classification of a given point. For clustering, the concept is less intuitive due to the lack of a ground truth, but may be defined as a minimal modification that would change the assigned cluster for a given point. Furthermore, a counterfactual may shift cluster boundaries, potentially affecting the counterfactual's own validity and similarity. Our research will also investigate this phenomenon.

While counterfactuals for classification is a well-studied area, there is less literature available for clustering methods. We will conduct initial research by gathering existing literature. We will then conduct experiments and analyze existing methods. Furthermore, we aim to develop novel methods for counterfactual generation and evaluate them against the current state of the art.

#pagebreak()

#heading(numbering: none, "Abstract", outlined: false)
Explainable artificial intelligence (XAI) is a field within computer science which aims to solve the black box problem, where decisions taken by machine learning models are able to be explained by neither experts nor the parties using them. One such method for explaining decisions are counterfactuals, which tells you what is required in order to move a point from one class to another. In this thesis, we specifically aim to explore the field of counterfactuals for $k$-means clustering, for which little research exists. We present 4 novel methods for generating counterfactuals: GradCF, CFAE, NeCS, and TGCF. GradCF employs gradient descent combined with a custom loss function. CFAE is an iterative algorithm which approaches the cluster boundary. NeCS is a greedy pick algorithm which uses $k$-nearest neighbors to construct counterfactuals, and TGCF is a decision tree algorithm which uses thresholds to approximate the cluster boundaries. 

We compare our contributions against 2 baseline methods along with a state of the art method for clustering counterfactuals. We conduct experiments on 4 different datasets for a random selection of problem instances. Furthermore, we identify strengths and weaknesses of the novel methods, and discuss how some of them may be improved. Through these experiments, we show that the novel methods can compete with state of the art methods, and in some cases outperform them.

#pagebreak()
#outline(title: "Table of contents", target: heading.where(numbering: "1.1"))

#pagebreak()
#outline(target: heading.where(supplement: [Appendix]), title: [Appendix])


#pagebreak()
#set page(numbering: "1")
#counter(page).update(1)
#include "chapters/Introduction.typ"
#include "chapters/ExplainableAI.typ"
#include "chapters/k-means.typ"
#include "chapters/methods.typ"
#include "chapters/k_mean_counterfactuals.typ"
#include "chapters/novel_methods.typ"
#include "chapters/dice_mod.typ"
#include "chapters/cf_heuristic_methods.typ"
#include "chapters/cf_tree_based_methods.typ"
#include "chapters/experiments.typ"
#include "chapters/future_work.typ"
#include "chapters/conclusion.typ"

#pagebreak()
#bibliography("bibliography.bib", style: "american-psychological-association")
//#bibliography("bibliography.bib", style: "springer-vancouver")
#pagebreak()
#show: appendix
#heading("Appendix", numbering: none, outlined: false)
#include "chapters/appendix.typ"
