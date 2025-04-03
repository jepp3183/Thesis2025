= Novel Tree-Based Counterfactual Methods
Another venue of design for generating counterfactuals for unsupervised tasks could be implemented by exploiting information contained in decision tree approximations of the clustering task. This idea of utilizing decision trees for explanations in a post-hoc manner comes from methods such as @cravenExtractingTreeStructuredRepresentations1995, who highlight that decision trees functions as intuitive explanation models, as they reveal decision logic implemented by the black box model. This enables decision trees to be used as an appropriate surrogate model for our black box. These surrogate decision trees enable an analyzer to move along the tree to reason about the clustering or classification of a certain data instance. This in turn enables an analyst to reason about possible counterfactual changes for an alternative outcome.

== Our novel model (name) / alternative title
To solve the task of generating valid and intuitive counterfactuals, we have made a novel method based on this class of tree-based explanation models. Our proposed method is model-agnostic in that it works with any clustering model and that any kind of surrogate decision tree model can be employed. Our method furthermore reveals intuitive counterfactuals changes as a usual decision tree partitions the data based on simple if-else rules. Though one thing to keep in mind when using these kinds of models, is that decision trees can have a fidelity/complexity trade-off as we often have to employ deeper trees to generate a more faithful surrogate model.




