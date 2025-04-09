#import "../lib.typ": *

= Dice modification TO-DO: change name (PC-Clust?)


Hello everyone and welcome to another counterfactual method.
#todo(position: right)[Come up with a better introduction]

The idea behind this approach is similar to how BayCon and its clustering-based adaption works @romashovBayConModelagnosticBayesian2022 @spagnolCounterfactualExplanationsClustering2024. We present an objective-function comprised of two terms which can be efficiently optimized through gradient descent or a randomized stepping algorithm, where the latter is capable of providing multiple counterfactuals. The first term ensures that the generated counterfactual is close to the instance being explained. The second term ensures both validity and plausibility by guiding the search towards the target cluster. The overall objective function is the product of these terms.

The first term is defined using Gower's similarity @gowerGeneralCoefficientSimilarity1971. This is a similarity measure between two data points, which supports both categorical and continuous features, although we only implement it for continuous features since we do not treat categorical data in this thesis. For two continuous data points $x_i,x_j in RR^d$, the Gower's similarity is defined as follows:
$
S(x_i, x_j) &= 1/d sum_(k=1)^d s_(i j k) \ #v(1em)
s_(i j k) &= 1 - (| x_i^k - x_j^k |)/R_k
$
Where $R_k$ is the range of the $k$-th feature in the given dataset. Thus, when #box($x_i = x_j$), we get $S_(i j) = 1$ and when the difference between them is maximal, we get $S_(i j) = 0$, since the range of the features is based on the given sample. We use this similarity measure to define the first term of the objective function, which measures the similarity of a counterfactual $x'$ to the original instance. See @fig:pcclust_terms for an illustration in 2 dimensions. 

The choice of this metric is based on the fact that it gives an easily interpretable result scaled between 0 and 1. As we show next, the second term of the objective function is also scaled to this range, making them easily comparable. It is also very easy to compute, since the only computationally significant step of computing the feature ranges must only be done a single time. 

#figure(
  image("../assets/pcclust_gain_terms.png"),
  caption: [The two terms of the PC-Clust objective function. The left term is the Gower's similarity between the counterfactual and the original instance, and the right term is based on the distance to the target cluster center.]
)<fig:pcclust_terms>

The second term is loosely based on the sigmoid activation function often used in machine learning applications. The traditional sigmoid activation function is defined as
$
sigma(d) &= 1/(1 + e^(-d))
$
and gives a result between 0 and 1, where a higher input value gives a result closer to 1. The idea is to use this function to guide the search closer towards the target cluster. This can be done by inverting and offsetting the function and letting the input be the distance from $x'$ to the target cluster center $c_t$. Thus, as a counterfactual gets closer to the target, the score increases. We also add a constant multiplier to make the function steeper around the offset. Implementing these modifications gives the function
$
sigma'(d) = 1/(1 + e^(lambda_2 dot (d - lambda_1)))
$

Care must be taken to set the offset $lambda_1$. The goal is to both ensure validity by letting $sigma'$ be maximized only inside the target cluster, while also ensuring plausibility by letting the maximum be close to other points in the cluster. To achieve the first goal, one can $lambda_1$ equal to the distance to the closest cluster boundary i.e. 
$
lambda_1 = 1/2 min_(0 <= s <= k-1) d(c_s, c_t)
$

For the second goal, one can set it equal to the maximum distance from the target cluster center to a point in the target cluster.
$
lambda_1=max_(y in C_t) d(y, c_t)
$
However, as illustrated in figure *INSERT FIGURE HERE*, situations can arise where both of these may cause problems with validity and/or plausibility. To solve this, we set $lambda_1$ to be the minimum of the two definitions.

The hyperparameter $lambda_2$ controls the steepness of the function. In our implementation, this is set to 50 in order to ensure validity. 


However, this function suffers from some major deficiencies that make it unusable without some modifications. First, since we seek to perform gradient descent starting 
