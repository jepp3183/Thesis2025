#import "../lib.typ": *

= $k$-means counterfactuals <sec:kmeans_cluster_change>

In this section, we consider the concept of non-static clusters as a result of counterfactual changes, and the consequences of such an interpretation. 

A crucial property of $k$-means clustering is that modifying the dataset changes the clustering. This is not guaranteed to change the cluster label of any points, but at the very least the centers are subject to change, since they are calculated as the mean of all points in the cluster. Hence, moving a point to another cluster, as is the case with counterfactuals, would move the cluster boundary for both clusters involved. An example of cluster boundaries changing after a number of counterfactual changes can be seen in @fig:decisionBoundaryChange. As more points are moved due to counterfactual changes, the cluster boundaries shift, causing a number of point reassignments to a different cluster.

#figure(
  image("../assets/DecisionBoundaryChangesCF.png"),
  caption: [The arrow shows how a point is moved due to a counterfactual change i.e. removing $x$ from the dataset and adding $x'$. The red circle is the applied counterfactual $x'$. The background colors visualize the cluster boundaries after each counterfactual change.]
)<fig:decisionBoundaryChange>

Thus, there are two possible viewpoints to consider. The first viewpoint states that the clustering should be seen as static, where counterfactuals are not added to do data, and therefore do not change the existing centers. The second viewpoint applies counterfactuals changes to the clustering by removing $x$ and adding $x'$ to the dataset, and then updating the centers. 

The static viewpoint follows the convention of general machine learning, where predicting on new data does not modify the underlying model. In this case, applying counterfactuals would slowly remove the original structure of the clustering, which is not desirable in many applications. Additionally, applying new counterfactuals may also invalidate previous counterfactuals.

However, the dynamic viewpoint where the cluster centers change based on applying the counterfactuals, could be useful in certain applications. If a model only uses counterfactuals for instances already in the clustered data, and the counterfactuals are carried out in the real world, applying the counterfactual and updating the centers would be preferable.

Whether a clustering should change over time or be static depends on the specific use case. In the following section, we analyze the dynamic viewpoint where clusterings are updated due to counterfactual changes. Additionally, in @sec:invalidation, we conduct experiments to see how often cluster changes affect the generated counterfactuals.

== Analysis of counterfactual changes<sec:cluster_change_theory>
As can be seen in @fig:decisionBoundaryChange, moving points to a new cluster can have a significant effect on the cluster boundaries. This change can be beneficial, but also disadvantageous. To understand why this is the case, consider that if the target cluster center $c_"target"$ moves towards the generated counterfactual, then the cluster boundary also moves away from the counterfactual and towards the source cluster center $c_"source"$. However, if $c_"source"$ moves away from the counterfactual, then the effect is similar to that of the target cluster. But in the case where $c_"source"$ moves towards the counterfactual, it then depends on how much each of the centers move to decide whether the change is beneficial or disadvantageous, as we will discuss later in this section.

As previously argued, if $c_"target"$ moves towards the counterfactual it can improve plausibility, since the counterfactual will lie further inside the target cluster. As we will argue, such a movement is guaranteed to always happen. To see why, consider the following lemma.

#lemma[
  Let $C$ be an arbitrary clustering. Let $C_"source" in C$ be the source cluster, and $C_"target" in C$ be the target cluster with respective centers $c_"source"$ and $c_"target"$. Let $x in C_"source"$ be any point in the source cluster and let $x' in C_"target"$ be a counterfactual into the target cluster. If we replace $x$ in the data with $x'$, then the resulting target cluster is denoted as $C'_"target"$ with center $c'_"target"$. With this, it holds that $d(c'_"target",x') <= d(c_"target",x')$, where $d$ is the Euclidian distance. 
]<lemma_cf_dis>

#proof[
The only difference between $c_"target"$ and $c'_"target"$ is the addition of $x'$ to the clustering. Therefore $c'_"target"$ can be written the following way:
$ 
c'_"target" &= 1/(|c_"target"| + 1) (x' + sum_(x in C_"target") x) \
&= 1/(|c_"target"| + 1) x' + 1/(|c_"target"| + 1) sum_(x in C_"target") x. #<proof:lemmaCloser>
$
This implies that either $x'=c_"target"$ in which case $d(x',c'_"target")=d(x',c_"target")=0$ since adding a point at the center of a clustering does not change it, or that $x'!=c_"target"$ in which case $d(c'_"target",x') < d(c_"target",x')$ holds due to the first term in @proof:lemmaCloser always resulting in a center closer to $x'$ than before.
]

#figure(
  placement: none,
  image("../assets/circlePlot.png", width: 65%),
  caption: [Visual representation of the different scenarios in which the cluster boundary moves based on a counterfactual change.]
)<fig:circlePlot>

@lemma_cf_dis can be used to improve a generated counterfactual. Since $c_"target"$ is closer to the counterfactual after applying it, the cluster boundary moves. This observation implies that $x'$ will be further from the cluster boundary than when originally creating it. This allows us to potentially move $x'$ even closer to the instance $x$ without crossing back over the cluster boundary.


The only caveat to this observation is that $c_"source"$ also moves as a consequence of removing a point from $C_"source"$. This can either move the cluster boundary even further from $c_"target"$, but it may also move it closer. In order to understand these 2 scenarios, consider the circles in @fig:circlePlot which are abstract representations of clusters. If $x$ lies in the green area, then $c_"source"$ would move away from $c_"target"$ as a consequence of applying the counterfactual, changing the cluster boundary in the process.

In the other case, if $x$ lies in the red area, the source center would move towards the target center when applying the counterfactual. In this case, the cluster boundary only moves closer to $c_"source"$ if $c_"target"$ moves further towards $x'$ than $c_"source"$ does.

Thus, in the case where $x$ lies in the red area, we know that $c_"source"$ moves closer to $x'$. In some situations, this change may invalidate the counterfactual. Specifically, this can only happen when a counterfactual is generated in the space between the target and source centers. In this case, if the cluster boundary moves towards the target center as a result of both centers moving, then the updated boundary can potentially cross over the counterfactual and invalidate it. To visualize this process, consider @fig:invalid_cf_change where a counterfactual is invalidated due to cluster boundary shifts after applying it.

#figure(
  image("../assets/invalid_cf_change.png"),
  caption: [Example of a counterfactual becoming invalid due to the cluster centers changing after moving the instance.]
)<fig:invalid_cf_change>

From @fig:invalid_cf_change we can reason that for a counterfactual to be invalidated, the instance needs to be in a spot such that $c_"source"$ moves further towards the counterfactual than $c_"target"$ does. Even when this is the case, the movement of the updated cluster boundary also needs to be greater than the margin between $x'$ and the original cluster boundary. From this, it would seem that situations where this happens in larger datasets are rare, since the cluster centers change very little when moving a single point. However, if a large number of such counterfactuals are applied e.g. in a real world deployment, the changes may start invalidating previously generated counterfactuals.

In @fig:correction_example, the same counterfactual is visualized in both plots. In the left plot, the counterfactual is invalid since it lies on the wrong side of the cluster boundary. However, if we apply the counterfactual change and recalculate the centers, we get the clustering in the right plot. Thus, the change in the cluster centers causes the initially invalid counterfactual to become valid. We call this phenomenon correction. 

#figure(
  image("../assets/correction_example.png"),
  caption: [Example counterfactual which gets corrected after cluster\ reassignments.]
)<fig:correction_example>

In contrast to invalidation, which is primarily caused by $c_"source"$ moving towards the counterfactual, corrections are harder to predict as they require $c_"source"$ (or whatever other cluster $x'$ is assigned to) to change in a way which causes $c_"target"$ to get closer to $x'$. For this reason, we speculate that corrections are rarer than invalidations.


/*
$
c_t ' = (n dot c_t + x')/(n + 1), c_o ' = (n dot c_o - x')/(n - 1)  \
 d(x',c_t ') < d(x',c_o ') \
 sum_(i in "dim") (x_i ' - c'_t_i)^2 < sum_(i in "dim") (x_i ' - c'_o_i)^2 \
 sum_(i in "dim") (x_i ' - (n dot c_t_i + x'_i)/(n+1))^2 < sum_(i in "dim") (x_i ' - (n dot c_o - x')/(n - 1))^2 \
 sum_(i in "dim") x'_i^2 + sum_(i in "dim") ((n dot c_t_i + x'_i)/(n+1))^2 + 2(sum_(i in "dim") x'_i^2)(sum_(i in "dim") ((n dot c_t_i + x'_i)/(n+1))^2) < \ 
 sum_(i in "dim") x'_i^2 + sum_(i in "dim") ((n dot c_o_i - x'_i)/(n-1))^2 + 2(sum_(i in "dim") x'_i^2)(sum_(i in "dim") ((n dot c_o_i - x'_i)/(n-1))^2) \
 sum_(i in "dim") ((n dot c_t_i + x'_i)/(n+1))^2 + 2(sum_(i in "dim") x'_i^2)(sum_(i in "dim") ((n dot c_t_i + x'_i)/(n+1))^2) < \
 sum_(i in "dim") ((n dot c_o_i - x'_i)/(n-1))^2 + 2(sum_(i in "dim") x'_i^2)(sum_(i in "dim") ((n dot c_o_i - x'_i)/(n-1))^2) 
$*/


