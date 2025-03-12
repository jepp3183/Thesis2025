// https://typst.app/docs/reference/scripting/
#import "lib.typ": *

= Introduction to K-means

Centroid clustering is an objective concerned with finding groups of data based on optimizing a set of cluster centers. In such a setting any point is assigned to it's closest cluster based on some distance metric. As our objective is to find counterfactuals for such methods we will now look into the most well known method for finding such clusters, that being the k-means objective. K-means was originally created by Lloyd@lloydLeastSquaresQuantization1982 and then generalized for clustering by MacQueen@macqueenMethodsClassificationAnalysis1967. The method described in these papers are what's commonly known as Lloyds's method, later this was further improved upon with better initializing of the centers in Arthur@arthurKmeansAdvantagesCareful who introduced the K-means++ algorithm.

In k-means we start with a set of n points $X={x_0, x_1, dots , x_n}$ and k-centers $C={c_0, c_1, dots , c_k}$, in order to solve this problem we want to minimize:
$ theta = sum_(x in X) min_(c in C) ||x - c||^2 $
Less formally we want to minimize the distance from any point to it's closest center. The process of finding such centers is however difficult as the problem is NP-hard. Lloyd's method is a way to find good clusters without solving the NP-hard problem. While Lloyd's method has been shown to produce good clusters, is offers no approximation guaranties or useful running time bounds. It is however clear that the method always terminates, which we will argue shortly.

The following is the Lloyds method for solving k-means. @arthurKmeansAdvantagesCareful

1. Initially we choose k centers at random $C_0 = {c_0, c_1, dots, c_k}$ which will be our initial centers. 
2. Then for each point in our data we assign it to one of the k clusters. 
3. The cluster centers are then re-calculated by taking the mean of each point in each cluster, $c_i = 1/(|c_i|) sum_(x in c_i) x$.
4. Repeat 2,3 until C stops changing.

The idea behind why this method terminates is that in each iteration the objective  function $theta$ decreases, and since there is an upper limit on the amount of possible clusters, the method must at some point either stop improving or find the optimal clustering. The intuition as to why the method always improves in each iteration comes from the following equation from linear algebra @arthurKmeansAdvantagesCareful:
$ sum_(x in C_p) ||x-z||^2 - sum_(x in C_p) ||x-c_p||^2 = |C_p| dot ||c_p-z||^2 $ <lin_loyd>
Where $C_p$ is a set of points, $c_p$ is the center of mass in $C_p$ and $z$ is any point in the metric space.

Combing Lloyds method with @lin_loyd allows us to argue why the method improves in each iteration. If we assume we just completed iteration $j$ in Lloyds method then we can think of $c_(i,j)$ being center $i$ after iteration $j$, then we define $z$ from @lin_loyd as $c_(i,j-1)$ and $c_p$ from @lin_loyd to be $c_(i,j)$. If $c_(i,j)=c_(i,j-1)$ then $C_p$ has not changed and we terminate. Otherwise if $c_(i,j)$ and $c_(i,j-1)$ are not identical we consider the cost of each cluster center respectively, $sum_(x in C_i) ||x-c_(i,j)||^2$ and $sum_(x in C_i) ||x-c_(i,j-1)||^2$. From this it is clear that the first term is smaller since it represents the current mean of the points in the cluster. Looking at @lin_loyd it becomes clear that $|C_i| dot ||c_(i,j)-c_(i,j-1)||^2$ represents the decrease in costs of from iteration $j-1$ to $j$ which is always greater or equal to 0.

== K-means++
Even though K-means has properties which supports its effectiveness as a general clustering method, the lack of guaranties can in some cases make it hard to produce a useful clustering. Therefore, in 2006 the K-means++ algorithm was invented which improves upon the method while also relating the resulting clusters to the optimum solution @arthurKmeansAdvantagesCareful. 

The k-means++ algorithm uses the function $D(x)$ which denotes the distance to the closest center from $x$. The following method is the k-means++ method:
1. Pick $c_0$ at random from X, this is the first cluster center
2. Pick new center randomly from X weighted by $(D(c_i)^2)/(sum_(x in X) D(x)^2)$
3. Repeat step 2 until k centers are found
4. Perform the normal k-means algorithm, starting with the found centers

The intuition behind the $D^2$ weighting in step 2 is that increasing the distance to the closest center while decreasing the squared distance to all other points makes it likely that the method picks diverse centers. The resulting clustering with potential $Phi$ which denotes the cost of clustering relates to the optimal clustering in the following way:
$ E[Phi] <= 8 (ln k + 2)Phi_(text("OPT")) $
Hence we can expect the error of the K-means++ clustering to be at most $O(log k)$, which is a significant improvement over the original k-means clustering algorithm. @arthurKmeansAdvantagesCareful

== K-means counterfactuals
One property of the k-means objective is that adding points to a dataset will inevitably change the clustering to some degree. This is not guarantied to change the clustering label of any points but at the very least the centers are malleable. This is because centers are calculated based on all the points in the cluster. Hence, moving a point to another cluster, as is the case with counterfactuals, would move the clustering boundary for both clusters involved. An example of a clustering boundary changing after a number of counterfactual changes can be seen on @fig:decisionBoundaryChange.

#figure(
  image("assets/DecisionBoundaryChangesCF.png"),
  caption: [On each plot except the first, the arrow shows how a point was changed due to a counterfactual change, with the red ball being the resulting point. The background colors visualize the clustering boundary after the counterfactual change]
)<fig:decisionBoundaryChange>

As can be seen on the figure, moving points into a new cluster can sometimes have significant effect on the clustering boundary. 

This change can be beneficial considering that we want counterfactuals to be as close to the original instance as possible. To see why this is, consider the following lemmas.

#lemma[
  Let $C$ an arbitrary clustering. Let $C_"origin" in C$ be the origin cluster, $C_"target" in C$ be the target cluster with centers $c_"orgin"$ and $c_"target"$. Let $x in C_"origin"$ be any point in the origin cluster and let $hat(x) = delta x$ where $hat(x) in C_"target"$ be a counterfactual into the target cluster. If we replace $x$ in the graph with $hat(x)$ then the resulting target cluster are denoted as $C'_"target"$ with center $c'_"target"$. Then it holds that $d(c'_"target",hat(x)) <= d(c_"target",hat(x))$. 
]<lemma_cf_dis>

#proof[
The only difference between $c_"target"$ and $c'_"target"$ is the addition of $hat(x)$ to the clustering. Therefore $c'_"target"$ can be written the following way:
$ c'_"target" &= 1/(|c_"target"| + 1) (hat(x) + sum_(x in C_"target") x) \
&= 1/(|c_"target"| + 1) hat(x) + 1/(|c_"target"| + 1) sum_(x in C_"target") x $<proof:lemmaCloser>
This implies that either $hat(x)=c_"target"$ in which case $d(hat(x),c'_"target")=d(hat(x),c_"target")=0$ or that $hat(x)!=c_"target"$ in which case $d(c'_"target",hat(x)) < d(c_"target",hat(x))$ holds due to the first term in @proof:lemmaCloser will always result in center close to $hat(x)$ than before. Since this covers both cases the proof is complete.
]

The idea behind why @lemma_cf_dis allows us to make better counterfactuals is that since the target cluster-center is closer to the counterfactual, the clustering boundary also changes. This observation implies that the counterfactual will be further from the clustering boundary than expected when creating the counterfactual, which allows us to move it closer to the original point without invalidating the counterfactual. 

The only caveat to this observation is that the origin cluster center also moves as a consequence of removing a point. This will in some cases move the clustering boundary even closer to the point from which counterfactuals were created, but could also lead to the opposite result. If 
$d(x,c_("target"))-d(c_"target",c_"origin")>0$ is true then the clustering boundary moves closes to point since it implies the point lies between the two clusters. If not true it is unclear whether the clustering boundary improves or not. This process is visualized by @fig:circlePlot. #todo(position: right)[TODO: I think it is possible to say something about the behavior in the "bad" case]

#figure(
  image("assets/circlePlot.png"),
  caption: [If the point we want to explain lies in the red half-circle then the origin center would move towards the right as a consequence of transforming the point into a counterfactual, hence why $d(x,c_("target"))-d(c_"target",c_"origin")>0$ implies the clustering boundary moves away from the target center. In the other case, when the point exists in the green half circle, the origin center would move towards the target center. In this case, the clustering boundary only moves closer to the origin center if the target center move further towards the counterfactual than the origin center does.]
)<fig:circlePlot>

=== Validity of changing cluster centers
Previously we argued that cluster centers change when transforming points, due to cluster centers being the mean of all points in their respective clusters. However there is another viewpoint to consider, maybe the clustering resulting from executing the K-means++ algorithm should instead be viewed as the final result, rather than a starting point for counterfactual changes. The logic behind this standpoint extends to general unsupervised learning, as running such models on the new data, usually do not transfigure the model while using it. 

Therefore, we should consider whether procedurally changing the cluster centers is a valid process, or the centers should be viewed as static. The correct interpretation depends heavily on the problem being solved. If the application requires that one first finds the labels and thereafter find counterfactuals for a static system, then finding new centers could invalidate previous counterfactuals. Another common way of using clustering is to work on some training data and then use that for future references, in such a process future counterfactuals changes could invalidate the clustering according to the original purpose of the model. 

It is however also true that if counterfactual changes affects the dataset iteratively, then changing the clustering would be the only way to keep the clustering correct. This could be especially valid in dynamic systems, where counterfactuals changes should be interpreted as an immediate change to the dataset.
