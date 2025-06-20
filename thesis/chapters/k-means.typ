#import "../lib.typ": *

= $k$-means clustering<sec:kmeans_clustering>
Centroid-based clustering is concerned with finding groups represented by a set of cluster centers. In such a setting, any point is assigned to its closest cluster center based on some distance metric. Since the goal of this thesis is to find counterfactuals for $k$-means clusterings @macqueenMethodsClassificationAnalysis1967, we first outline the objective and cluster assignment functions, and then present the most common algorithm for finding such clusterings.

In $k$-means clustering, given a set of $n$ points $X={x_1, dots , x_n}$ and parameter $k$, the goal is to find $k$ cluster centers $C={c_0,dots,c_(k-1)}$ minimizing the objective function defined as
$ 
theta = sum_(x in X) min_(c in C) ||x - c||^2. 
$<eq:kmeans_obj>

The clusters $C_0,...,C_(k-1)$ associated with each center is calculated based on the following cluster assignment function, which assigns a point to the cluster with the closest center:

$
cal(C)(x, {c_0,dots,c_(k-1)}) = argmin_(i=0,...,k-1)  ||x-c_i||.
$<eq:cluster_assigment>

Less formally, to obtain an optimal $k$-means clustering, the sum of squared distances from any point to its closest center must be minimized. However, the process of finding the optimal centers is an NP-hard problem. Lloyd's algorithm @lloydLeastSquaresQuantization1982 is a way to approximate the optimal clustering without solving the NP-hard problem, by iteratively updating the centers and reassigning points to clusters. While Lloyd's algorithm has been shown to produce good clusters, it offers no approximation guarantee or useful running time bounds. However, it is clear that the method always terminates, which we argue below. The algorithm is given in @alg:lloyd. It may be initialized with any arbitrary selection of $k$ centers, and proceeds in iterations until the objective value as defined in @eq:kmeans_obj no longer improves. In each iteration, it assigns all points to its closest center, and then recalculates the centers based on the new clusters. This proceeds until the centers no longer change, or equivalently, $theta$ stops improving.

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  pseudocode-list(booktabs: true, title: 
    [*Lloyds-Algorithm*$(X, k, c^0_0,..,c^(0)_(k-1))$]
  )[
+ $t=1$
+ *do*
  + #comment("Assignment step")
  + *for* $i=0$ *to* $k-1$
    +  $C_i^t = {x in X | cal(C)(x, {c_0^(t-1),dots,c^(t-1)_(k-1)}) = i}$ #h(5em)
  
  + #comment("Center calculation")
  + *for* $i=0$ *to* $k-1$
    + $c_i^t = 1/(|C_i^(t)|) sum_(x in C_i^(t)) x$
  + $t = t + 1$
+ *while* $theta^t != theta^(t-1)$
+ *return* $c^t_0, ..., c^t_(k-1)$
  ],
  caption: [Lloyd's algorithm, as defined in @arthurKmeansAdvantagesCareful2007. We define $theta^t$ as the objective value in iteration $t$.],
)<alg:lloyd>

This algorithm ensures termination due to the fact that $theta$ improves in each iteration. Since there is an upper limit on the amount of possible clusters, the method must at some point stop improving, having found a local optimum.

The fact that $theta$ decreases in each iteration is due to the following statement @arthurKmeansAdvantagesCareful2007:
$ 
sum_(x in C_p) ||x-z||^2 - sum_(x in C_p) ||x-c_p||^2 = |C_p| dot ||c_p-z||^2, 
$ <lin_loyd>
where $C_p$ is a set of points, $c_p$ is the center of $C_p$ and $z$ is any point in the metric space. 

Combining Lloyd's algorithm with @lin_loyd allows us to argue why the algorithm improves in each iteration. After completion of iteration $t$ in Lloyd's algorithm, we can define $z$ from @lin_loyd as $c_i^(t-1)$ and $c_p$ from @lin_loyd to be $c_i^t$.

$ 
sum_(x in C_i^t) ||x-c_i^(t-1)||^2 - sum_(x in C_i^t) ||x-c_i^t||^2 = |C_i^t| dot ||c_i^t-c_i^(t-1)||^2 
$<lin_loyd_mod>

Now, by considering the cost of each cluster separately, the first term in @lin_loyd_mod is the cost of $C_i^t$ when using $c_i^(t-1)$ as a center, while the second term is the cost using the newly calculated center. The difference between these costs is then $|C_i^t| dot ||c_i^t-c_i^(t-1)||^2$. Thus, if $c_i^(t) != c_i^(t-1)$, the cost of $C_i^t$ has decreased, and if $c_i^(t) = c_i^(t-1)$, the cost has stayed the same, terminating the algorithm. Since this holds for all disjoint clusters #box[$C_0^t, dots, C_(k-1)^t$], we get $theta^t <= theta^(t-1)$. 

== $k$-means++
Although Lloyd's algorithm has properties which support its effectiveness as a general clustering method, the lack of theoretical guarantees is a disadvantage. Therefore, in 2007, the $k$-means++ algorithm was formulated, which improves upon the method by intelligently selecting the initial centers, while also relating the resulting clusters to the optimal solution @arthurKmeansAdvantagesCareful2007, thereby yielding an approximation ratio. The optimal solution for the $k$-means problem is the set of centers which achieves the lowest objective value $theta$. We define $theta_"OPT"$ as the objective value induced by the optimal clustering.

The $k$-means++ algorithm, given in @alg:kmeans_plusplus, uses the function #box[$D(x)=min_(c in C) ||x - c||$], which denotes the distance to the closest center from $x$. The algorithm picks the first center uniformly at random, and then proceeds to pick the remaining $k-1$ centers weighted proportionately to $D(x)^2$, before finally running Lloyd's algorithm with the chosen centers. The intuition behind the $D^2$ weighting in line 3 is that increasing the distance to the closest center makes it likely that the method picks diverse centers. 

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  pseudocode-list(booktabs: true, title: 
    [$k$-*means++*$(X, k)$]
  )[
+ $c_0 ~ X$ #comment("First center is sampled uniformly at random")
+ *for* $i=1$ *to* $k-1$
  + $c_i = z$ sampled proportionate to $(D(z)^2)/(sum_(x in X) D(x)^2)$
+ *return* Lloyds-Algorithm$(X,k,c_0,dots,c_(k-1))$
  ],
  caption: [$k$-means++ algorithm, as defined in @arthurKmeansAdvantagesCareful2007.],
)<alg:kmeans_plusplus>



The resulting clustering with objective value $theta$ relates to the optimal clustering in the following way:
$ 
E[theta] <= 8 (ln k + 2)theta_(text("OPT")).
$
Hence, we can expect the error of the $k$-means++ clustering to be bounded by $O(log k)$, which is a significant improvement over the original Lloyd's algorithm, which has no approximation guarantee @arthurKmeansAdvantagesCareful2007. Note that this approximation ratio is based entirely on the initial centers calculated in @alg:kmeans_plusplus. However, as argued in the previous section, Lloyd's algorithm can only improve the clustering, thereby keeping the approximation guarantee. 

For the entirety of this thesis, we will be calculating $k$-means clusterings using the $k$-means++ algorithm as implemented by Scikit-learn#footnote[https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html].
