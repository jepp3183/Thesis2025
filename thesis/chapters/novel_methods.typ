#import "../lib.typ": *

= Novel Methods

== Our contributions #todo[maybe different name]

Generating counterfactuals aimed at centroid based clustering, is a task for which very few methods exists. Currently, to the best of our knowledge only @spagnolCounterfactualExplanationsClustering2024 and @vardakasCounterfactualExplanationsKmeans2025 presents solutions which are specifically designed for the centroid clustering objective. One could also apply other counterfactual methods to the task, but since such methods were not designed with this objective in mind, the resulting counterfactuals are not guarantied to produce ideal samples. Therefore, in this paper we aim to propose 3 methods which all aim to find good counterfactuals for clusters, these are CFAE in @cfae, NeCS in @Necs and PC-Clust in XXX#todo[add reference to PC-clust]. Due to the varied nature of these methods we will cover the specific contributions in their respective sections.

In order to determine how well these methods perform, we have implemented 2 baseline methods which is a Prototype based approach inspired by @vanlooverenInterpretableCounterfactualExplanations2021 covered in #todo[maybe describe actual method(prototypes doesn't exist anymore)] and a decision tree approach covered in @tree_methods. We also compare our methods with @spagnolCounterfactualExplanationsClustering2024 which is state of the art method for generating counterfactuals in clustering models.

In order to evaluate our methods we have defined the following properties which we can use to decide whether our methods succeeds in different aspect of the counterfactual generation for centroid clustering problem.

- *Model Agnostic:* In this paper we mainly argue about the K-means clustering objective since it's a well known clustering objective applicable to many problems. Our methods will however not be solely usable on k-means cluster's, as many other centroid based clustering problems exists.
- *Validity:* Our methods should produce mostly valid counterfactuals. Valid in this case means that they belong to the correct cluster. What this means is that regardless of the distributions of points and clustering centers our methods should be able to produce valid counterfactuals, even if the clustering is structured in such a way which makes it harder to do so.
- *Actionable:* For a counterfactual to be actionable is is required that an end-user can act upon a counterfactual. This requirement is hard to quantify as whether an end-user can act upon a counterfactual is dependent on the end-user. What we can do is design metrics which measures how actionable a counterfactual is and use such metrics to evaluate the counterfactuals produced by our methods. The obvious candidates for actionability metrics are similarity which measures the size of the applied change and sparsity which measures how many features in the counterfactual differs from the instance. Another aspect of making actionable counterfactuals are immutable features. This is because an end-user might not be able to affect some features, one example could be the gender or the age of the suspect. Therefore we aim to where allow for users to pick immutable features in our methods. Yet another metric which is applicable in the case where multiple counterfactuals are generated is diversity. This metric measures how diverse the generated counterfactuals in order to avoid near duplicate counterfactuals, and to reward counterfactuals with different approaches.
- *Time:* In order to produce effective counterfactual the running time any such method should be manageable. Therefore we aim to create methods which can compete with state of the art algorithms when it comes the running time.

When looking at algorithms such as @romashovBayConModelagnosticBayesian2022 it is clear that methods which presents a large number of counterfactuals at once, is clearly a valid approach to counterfactual generation. This can however be confusing for the end-user who can be overwhelmed by such solutions. In this paper we look at another approach where we limit the amount of counterfactuals found to a constant amount for each method, which can then be modified to suit the end-user. 

== K-means counterfactuals
#todo[this entire thing is kinda suspect]
One property of the k-means objective is that adding points to a dataset will inevitably change the clustering to some degree. This is not guarantied to change the clustering label of any points but at the very least the centers are malleable. This is because centers are calculated based on all the points in the cluster. Hence, moving a point to another cluster, as is the case with counterfactuals, would move the clustering boundary for both clusters involved. An example of a clustering boundary changing after a number of counterfactual changes can be seen on @fig:decisionBoundaryChange.

#figure(
  image("../assets/DecisionBoundaryChangesCF.png"),
  caption: [On each plot except the first, the arrow shows how a point was changed due to a counterfactual change, with the red ball being the resulting point. The background colors visualize the clustering boundary after the counterfactual change]
)<fig:decisionBoundaryChange>

As can be seen on @fig:decisionBoundaryChange, moving points into a new cluster can sometimes have significant effect on the clustering boundary. 

This change can be beneficial considering that we want counterfactuals to be as close to the original instance as possible. To see why this is, consider the following lemmas.

#lemma[
  Let $C$ an arbitrary clustering. Let $C_"origin" in C$ be the origin cluster, $C_"target" in C$ be the target cluster with centers $c_"orgin"$ and $c_"target"$. Let $x in C_"origin"$ be any point in the origin cluster and let $hat(x) in C_"target"$ be a counterfactual into the target cluster. If we replace $x$ in the graph with $hat(x)$ then the resulting target cluster are denoted as $C'_"target"$ with center $c'_"target"$. Then it holds that $d(c'_"target",hat(x)) <= d(c_"target",hat(x))$. 
]<lemma_cf_dis>

#proof[
The only difference between $c_"target"$ and $c'_"target"$ is the addition of $hat(x)$ to the clustering. Therefore $c'_"target"$ can be written the following way:
$ c'_"target" &= 1/(|c_"target"| + 1) (hat(x) + sum_(x in C_"target") x) \
&= 1/(|c_"target"| + 1) hat(x) + 1/(|c_"target"| + 1) sum_(x in C_"target") x $<proof:lemmaCloser>
This implies that either $hat(x)=c_"target"$ in which case $d(hat(x),c'_"target")=d(hat(x),c_"target")=0$ since adding a point at the center of clustering does not change it, or that $hat(x)!=c_"target"$ in which case $d(c'_"target",hat(x)) < d(c_"target",hat(x))$ holds due to the first term in @proof:lemmaCloser will always result in center close to $hat(x)$ than before. Since this covers both cases the proof is complete.
]#todo(position: right)[TODO: only with 2 centers]

The idea behind why @lemma_cf_dis allows us to make better counterfactuals is that since the target cluster-center is closer to the counterfactual, the clustering boundary also changes. This observation implies that the counterfactual will be further from the clustering boundary than expected when creating the counterfactual, which allows us to move it closer to the original point without invalidating the counterfactual. 

The only caveat to this observation is that the origin cluster center also moves as a consequence of removing a point. This will in some cases move the clustering boundary even closer to the point from which counterfactuals were created, but could also lead to the opposite result. If 
$d(x,c_("target"))-d(c_"target",c_"origin")>0$ is true then the clustering boundary moves farther away from the target cluster center since it implies the point lies between the two clusters. If not true it is unclear whether the clustering boundary improves or not. This process is visualized by @fig:circlePlot. #todo(position: right)[TODO: I think it is possible to say something about the behavior in the "bad" case]

#figure(
  image("../assets/circlePlot.png"),
  caption: [Both of the circles are abstract representations of clusters. If the point we want to explain lies in the red half-circle then the origin center would move away from the target center as a consequence of transforming the point into a counterfactual, hence why $d(x,c_("target"))-d(c_"target",c_"origin")>0$ implies the clustering boundary moves away from the target center. In the other case, when the point exists in the green half circle, the origin center would move towards the target center. In this case, the clustering boundary only moves closer to the origin center if the target center move further towards the counterfactual than the origin center does.]
)<fig:circlePlot>

=== Validity of changing cluster centers
Previously we argued that cluster centers change when transforming points, due to cluster centers being the mean of all points in their respective clusters. However there is another viewpoint to consider, maybe the clustering resulting from executing the K-means++ algorithm should instead be viewed as the final result, rather than a starting point for counterfactual changes. The logic behind this standpoint extends to general unsupervised learning, as running such models on the new data, usually do not transfigure the model while using it. 

Therefore, we should consider whether procedurally changing the cluster centers is a valid process, or the centers should be viewed as static. The correct interpretation depends heavily on the problem being solved. If the application requires that one first finds the labels and thereafter find counterfactuals for a static system, then finding new centers could invalidate previous counterfactuals. Another common way of using clustering is to work on some training data and then use that for future references, in such a process future counterfactuals changes could invalidate the clustering according to the original purpose of the model. 

It is however also true that if counterfactual changes affects the dataset iteratively, then changing the clustering would be the only way to keep the clustering correct. This could be especially valid in dynamic systems, where counterfactuals changes should be interpreted as an immediate change to the dataset.
