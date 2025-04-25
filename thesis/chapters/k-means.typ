#import "../lib.typ": *

= Introduction to k-means

Centroid based clustering is concerned with finding groups of represented by a set of cluster centers. In such a setting any point is assigned to its closest cluster center based on some distance metric. As our objective is to find counterfactuals for such methods we will now look into the most well known method for finding such clusters, that being k-means. k-means was originally created in @lloydLeastSquaresQuantization1982 and then generalized for clustering in @macqueenMethodsClassificationAnalysis1967. 

In k-means, given a set of n points $X={x_1, dots , x_n}$ and parameter k, we want to find $k$ clusters $C_0,dots,C_(k-1)$ with associated centers $C={c_0,dots,c_(k-1)}$ minimizing the following objective:
$ theta = sum_(x in X) min_(c in C) ||x - c||^2 $
Less formally we want to minimize the distance from any point to its closest center. The process of finding such centers is however difficult as the problem is NP-hard. Lloyd's method is a way to find good clusters without solving the NP-hard problem. While Lloyd's method has been shown to produce good clusters, it offers no approximation guaranty or useful running time bounds. It is however clear that the method always terminates, which we will argue shortly. @arthurKmeansAdvantagesCareful

The following is the Lloyd's method for solving k-means. @arthurKmeansAdvantagesCareful

1. Initially we choose parameter k and centers $C^0 = {c_0^0, dots, c_(k-1)^0}$ randomly which will be our initial centers. 
2. Then for each point in our data we assign it to the closest cluster center. 
3. The cluster centers are then re-calculated by taking the mean of all points in each cluster, $c_i^t = 1/(|c_i^(t-1)|) sum_(x in c_i^(t-1)) x$.
4. Repeat 2,3 until $theta$ convergences.

The idea behind why this method terminates is that in each iteration the objective  function $theta$ decreases, and since there is an upper limit on the amount of possible clusters, the method must at some point either stop improving or find the optimal clustering. The intuition as to why the method always improves in each iteration comes from the following equation @arthurKmeansAdvantagesCareful:
$ sum_(x in C_p) ||x-z||^2 - sum_(x in C_p) ||x-c_p||^2 = |C_p| dot ||c_p-z||^2 $ <lin_loyd>
Where $C_p$ is a set of points, $c_p$ is the center of $C_p$ and $z$ is any point in the metric space. 

Combining Lloyds method with @lin_loyd allows us to argue why the method improves in each iteration. If we assume iteration $j$ was just completed in Lloyds method, we can define $z$ from @lin_loyd as $c_i^(j-1)$ and $c_p$ from @lin_loyd to be $c_i^j$:

$ sum_(x in C_i^j) ||x-c_i^(j-1)||^2 - sum_(x in C_i^j) ||x-c_i^j||^2 = |C_i^j| dot ||c_i^j-c_i^(j-1)||^2 $<lin_loyd_mod>


If $c_i^j=c_i^(j-1)$ then $C_p$ has not changed and we terminate. Otherwise, if $c_i^(j)$ and $c_i^(j-1)$ are not identical we consider the cost of each cluster respectively. Clearly, the first term is greater since it represents the cost of the cluster in iteration $j-1$. Looking at @lin_loyd_mod it becomes clear that $|C_i^j| dot ||c_i^j-c_i^(j-1)||^2$ represents the decrease in cost from iteration $j-1$ to $j$ which is always greater or equal to 0.

== k-means++
Even though k-means has properties which supports its effectiveness as a general clustering method, the lack of guaranty can in some cases make it hard to produce a useful clustering. Therefore, in 2006 the k-means++ algorithm was invented which improves upon the method while also relating the resulting clusters to the optimum solution @arthurKmeansAdvantagesCareful. 

The k-means++ algorithm uses the function $D(x)=min_(c in C) ||x - c||$ which denotes the distance to the closest center from $x$. The following method is the k-means++ method: @arthurKmeansAdvantagesCareful
1. Pick $c_0$ at random from X, this is the first cluster center
2. Pick new center randomly from X where each candidate point $z$ is weighted by $(D(z)^2)/(sum_(x in X) D(x)^2)$
3. Repeat step 2 until k centers are found
4. Perform the normal k-means algorithm, starting with the found centers

The intuition behind the $D^2$ weighting in step 2 is that increasing the distance to the closest center makes it likely that the method picks diverse centers. The resulting clustering with objective value $theta$ which denotes the cost of clustering relates to the optimal clustering in the following way:
$ E[theta] <= 8 (ln k + 2)theta_(text("OPT")) $
Hence we can expect the error of the k-means++ clustering to be at most $O(log k)$, which is a significant improvement over the original k-means clustering algorithm. @arthurKmeansAdvantagesCareful
