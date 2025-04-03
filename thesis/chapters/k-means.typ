#import "../lib.typ": *

= Introduction to K-means

Centroid clustering is an objective concerned with finding groups of data based on optimizing a set of cluster centers. In such a setting any point is assigned to it's closest cluster center based on some distance metric. As our objective is to find counterfactuals for such methods we will now look into the most well known method for finding such clusters, that being the k-means objective. K-means was originally created in @lloydLeastSquaresQuantization1982 and then generalized for clustering in @macqueenMethodsClassificationAnalysis1967. The method described is commonly known as Lloyds's method, later this was further improved upon with better initializing of the centers in  @arthurKmeansAdvantagesCareful who introduced the K-means++ algorithm.

In k-means we start with a set of n points $X={x_0, x_1, dots , x_n}$ and k-centers $C={c_0, c_1, dots , c_k}$, in order to solve this problem we want to minimize:
$ theta = sum_(x in X) min_(c in C) ||x - c||^2 $
Less formally we want to minimize the distance from any point to it's closest center. The process of finding such centers is however difficult as the problem is NP-hard. Lloyd's method is a way to find good clusters without solving the NP-hard problem. While Lloyd's method has been shown to produce good clusters, it offers no approximation guaranties or useful running time bounds. It is however clear that the method always terminates, which we will argue shortly. @arthurKmeansAdvantagesCareful

The following is the Lloyd's method for solving k-means. @arthurKmeansAdvantagesCareful

1. Initially we choose k centers at random $C_0 = {c_0, c_1, dots, c_k}$ which will be our initial centers. 
2. Then for each point in our data we assign it to the closest cluster center. 
3. The cluster centers are then re-calculated by taking the mean of each point in each cluster, $c_i = 1/(|c_i|) sum_(x in c_i) x$.
4. Repeat 2,3 until C stops changing.

The idea behind why this method terminates is that in each iteration the objective  function $theta$ decreases, and since there is an upper limit on the amount of possible clusters, the method must at some point either stop improving or find the optimal clustering. The intuition as to why the method always improves in each iteration comes from the following equation from linear algebra @arthurKmeansAdvantagesCareful:
$ sum_(x in C_p) ||x-z||^2 - sum_(x in C_p) ||x-c(C_p)||^2 = |C_p| dot ||c(C_p)-z||^2 $ <lin_loyd>
Where $C_p$ is a set of points, $c(C_p)$ is the center of mass in $C_p$ and $z$ is any point in the metric space.

Combining Lloyds method with @lin_loyd allows us to argue why the method improves in each iteration. If we assume we just completed iteration $j$ in Lloyds method then we can think of $c_(i,j)$ being center $i$ after iteration $j$, then we define $z$ from @lin_loyd as $c_(i,j-1)$ and $c(C_p)$ from @lin_loyd to be $c_(i,j)$. If $c_(i,j)=c_(i,j-1)$ then $C_p$ has not changed and we terminate. Otherwise, if $c_(i,j)$ and $c_(i,j-1)$ are not identical we consider the cost of each cluster center respectively, $sum_(x in C_i) ||x-c_(i,j)||^2$ and $sum_(x in C_i) ||x-c_(i,j-1)||^2$. From this it is clear that the first term is smaller since it represents the current mean of the points in the cluster. Looking at @lin_loyd it becomes clear that $|C_i| dot ||c_(i,j)-c_(i,j-1)||^2$ represents the decrease in costs of from iteration $j-1$ to $j$ which is always greater or equal to 0.

== K-means++
Even though K-means has properties which supports its effectiveness as a general clustering method, the lack of guaranties can in some cases make it hard to produce a useful clustering. Therefore, in 2006 the K-means++ algorithm was invented which improves upon the method while also relating the resulting clusters to the optimum solution @arthurKmeansAdvantagesCareful. 

The k-means++ algorithm uses the function $D(x)$ which denotes the distance to the closest center from $x$. The following method is the k-means++ method: @arthurKmeansAdvantagesCareful
1. Pick $c_0$ at random from X, this is the first cluster center
2. Pick new center randomly from X weighted by $(D(c_i)^2)/(sum_(x in X) D(x)^2)$
3. Repeat step 2 until k centers are found
4. Perform the normal k-means algorithm, starting with the found centers

The intuition behind the $D^2$ weighting in step 2 is that increasing the distance to the closest center while decreasing the squared distance to all other points makes it likely that the method picks diverse centers. The resulting clustering with potential $Phi$ which denotes the cost of clustering relates to the optimal clustering in the following way:
$ E[Phi] <= 8 (ln k + 2)Phi_(text("OPT")) $
Hence we can expect the error of the K-means++ clustering to be at most $O(log k)$, which is a significant improvement over the original k-means clustering algorithm. @arthurKmeansAdvantagesCareful
