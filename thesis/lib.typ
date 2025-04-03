#import "@preview/dashy-todo:0.0.2": todo
#import "@preview/lemmify:0.1.8": *
#import "@preview/lovelace:0.3.0": *

#let (
  theorem, lemma, corollary,
  remark, proposition, example,
  proof, rules: thm-rules
) = default-theorems(
  "thm-group",
  lang: "en",
  thm-numbering: thm-numbering-linear
)

