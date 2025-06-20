#import "../lib.typ": *

== Gradient-based counterfactual search (GradCF) <sec:gradcf>

In this section, we present a simple and efficient method for generating a single counterfactual based around an objective function. The objective function is comprised of two terms which can be efficiently optimized through gradient descent, and is defined as

$ 
f(x') = G(x, x') dot S(d(x', c_"target")). 
$

The first term, $G(x, x')$, ensures that the generated counterfactual is close to the instance being explained. The second term, $S(d(x', c_"target"))$, takes as input the Euclidian distance between the counterfactual and target cluster. This term ensures both validity and plausibility by guiding the search towards the target cluster.

$G(x, x')$ is defined using Gower's similarity @gowerGeneralCoefficientSimilarity1971. This is a similarity measure which supports both categorical and continuous features. We implement it for continuous features, since we do not treat categorical data in this thesis. For two continuous data points $x_i,x_j in RR^m$, Gower's similarity is defined as
$
G(x_i, x_j) &= 1/m sum_(k=1)^m s_(i j k) \ #v(1em)
s_(i j k) &= 1 - (| x_i^k - x_j^k |)/R_k,
$
where $R_k$ is the range of the $k$-th feature in the given dataset. Thus, when #box($x_i = x_j$), we get $G(x_i, x_j) = 1$ and when the difference between them is maximal, we get $G(x_i, x_j) = 0$, since the range of the features is based on the input data. We use this similarity measure to define the first term of the objective function, which measures the similarity of a counterfactual $x'$ to the original instance $x$.

The choice of this metric is based on the fact that it gives an easily interpretable result scaled between 0 and 1. As we show next, the second term of the objective function is also scaled to this range, making them easily comparable. It is also easy to compute, since the computationally significant step of computing the feature ranges must only be done once. 

The second term is loosely based on the sigmoid activation function often used in machine learning applications. The traditional sigmoid activation function is defined as
$
sigma(d) &= 1/(1 + e^(-d)) .
$
It returns a result between 0 and 1, where a higher input value gives a result closer to 1. The idea is to use this function to guide the search closer towards the target cluster. This can be done by inverting and offsetting the function and letting the input be the Euclidian distance from $x'$ to the target cluster center $c_"target"$. Thus, as a counterfactual gets closer to the target, the score increases. Finally, we also add a constant multiplier to make the function steeper around the offset. Implementing these modifications gives the function
$
sigma'(d) = 1/(1 + e^(lambda_2 dot (d - lambda_1))),
$<eq:almost_sigmoid_hinge>

where the parameter $lambda_1$ controls the offset, and $lambda_2$ controls how steep the function is around the offset.

Care must be taken to set the offset $lambda_1$. The goal is to both ensure validity by letting $sigma'$ be maximized only inside the target cluster, while also ensuring plausibility by letting the maximum be close to other points in the cluster. To achieve the first goal, one can define $lambda_1$ as the distance to the cluster boundary closest to $c_"target"$ i.e. 
$
lambda_1^"boundary" = 1/2 min_(0 <= s <= k-1) d(c_s, c_"target").
$<eq:lambda_1_1>

For the second goal, one can define it as the maximum distance from $c_"target"$ to a point in the target cluster:
$
lambda_1^"max-dist"=max_(y in C_"target") d(y, c_"target").
$<eq:lambda_1_2>

#figure(
  placement: none,
  grid(columns: 2,
  image("../assets/sigmoid_bad_boundary.png", width: 100%),
  image("../assets/sigmoid_bad_boundary_max.png", width:100%)
  ),
  caption: [Problematic definitions of $lambda_1$. On the left, @eq:lambda_1_1 is used, causing poor plausibility. On the right, @eq:lambda_1_2 is used, breaking validity. $lambda_2$ is set to 50, causing a steep increase around the offset.]
)<fig:sigmoid_sucks>

However, as illustrated in @fig:sigmoid_sucks, situations can easily arise where both of these may cause problems with validity and/or plausibility. When using @eq:lambda_1_1, the offset may be very far from the remaining points in the cluster, yielding counterfactuals with poor plausibility, since the objective function converges far from the other points. However, when using @eq:lambda_1_2, the offset may lie inside other clusters, yielding invalid counterfactuals. To solve this, we set $lambda_1$ as the minimum of the two. 

Regarding the second parameter $lambda_2$, recall that it controls how steep the sigmoid function is around the offset $lambda_1$. Setting $lambda_2$ to a low value will result in the sigmoid function not being very steep, meaning the gradient descent process will move further beyond the offset before converging. On the other hand, a large $lambda_2$ will result in a steep increase around the offset, meaning the process will converge quickly after crossing the boundary defined by $lambda_1$. Thus, $lambda_2$ is able to control how far into the target cluster the counterfactual is generated, thereby increasing plausibility with higher values of $lambda_2$. However, since $lambda_1$ is designed to serve as a proxy for the cluster boundary, it makes intuitive sense to let $lambda_2$ be large, and instead control plausibility by moving the offset $lambda_1$ further inside the cluster. 

To accomplish this, we introduce a user-defined hyperparameter $epsilon in [0,1]$, which defines how close the generated counterfactual will be to $c_"target"$. This parameter is much more intuitive for controlling plausibility, and can be determined by users through experimentation, which we expand on in @sec:gradcf_comparison. This gives the final definition of $lambda_1$ in @eq:lambda_1_final.
$
lambda_1=(1-epsilon)min(lambda_1^"boundary", lambda_1^"max-dist")
$<eq:lambda_1_final>
Setting $epsilon=0$ yields a counterfactual close to the validity-boundary, and setting $epsilon=1$ yields one close to the cluster center. An example of this for a synthetic 2D dataset can be seen in @fig:gradcf_eps.

#figure(
  image("../assets/gradcf_eps.png", width:75%),
  caption: [
    The effect of varying values of $epsilon$ on the generated counterfactual. A higher $epsilon$ yields counterfactuals close to the target center.
  ]
)<fig:gradcf_eps>

With the definition in @eq:almost_sigmoid_hinge, we can almost define the second term of the objective function $S(d(x', c_"target"))$. However, since the function will be optimized using gradient descent, it is important to consider the well-known problem of saturating gradients that often arises when using the sigmoid function. This problem comes from the fact that the gradient of $sigma$ converges quickly towards 0 outside the main activation region around the offset. This leads to a very low overall gradient of the objective function, which then leads to the gradient descent process converging very slowly, making little progress each step.

To avoid this problem, we define $S(d)$ as a continuous piecewise function based on the offset $lambda_1$:

$
S(d) = cases(
  sigma'(d) "if" d<lambda_1,
  rho(d) " otherwise,"
)
$
where
$
rho(d) = (d(x,c_"target") - d)/(2(d(x,c_"target") - lambda_1)).
$
This allows the gradient of $S(d)$ to be significant throughout the gradient descent process. Thus, the final objective function to be maximized is defined as
$
f(x') = G(x, x') dot S(d(x', c_"target")).
$<eq:pcclust_obj>

The method used for optimizing this objective function uses gradient descent on $-f(x')$. During development, we implemented the naive version of gradient descent, but discovered various issues regarding how to define the step-size to work for both low- and high-dimensional data. Instead, we opted to use the ADAM optimizer @kingmaAdamMethodStochastic2017 implemented in the PyTorch library #footnote[https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html]. This method uses an adaptive learning rate with momentum to find the optimum faster and more reliably, as can be seen in @fig:adam_run. 

#figure(
  image("../assets/ADAM_run.png", width:60%),
  caption: [Example optimization path using ADAM. The method finds an approximate optimum efficiently.]
)<fig:adam_run>

Of course, this method is quite basic, and therefore lacks certain properties. Firstly, it is only capable of generating a single counterfactual for each instance and target pair. As mentioned previously, generating multiple counterfactuals can be beneficial for developers in order to increase understanding of the underlying structure of the clustering. This is especially the case if the returned counterfactuals are meaningfully different, like the supervised DiCE method presented in @sec:dice tries to accomplish. A second potential deficiency of this method is the fact that gradient descent inherently seeks to change every single feature of the given instance $x$, harming sparsity. However, to improve this aspect, we can use the post-processing method also used in the DiCE paper @mothilalExplainingMachineLearning2020 to greedily reset features of $x'$ to their original value from $x$, as long as the change does not break validity and does not significantly affect the counterfactual. In our testing, this improves sparsity for the generated counterfactual, although the plausibility score also decreases noticeably.