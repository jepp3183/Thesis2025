#import "../lib.typ": *

== Novel heuristic-based counterfactual methods
Clearly, in order to maximize the similarity of a counterfactual, the optimal way is to find the closest point on the cluster boundary towards the target cluster. However, it is not obvious whether such a point would be a high quality counterfactual, given that we aim to generate both sparse and plausible counterfactuals by not focusing solely on similarity. By using heuristic methods, we can approach the decision boundary in steps without sacrificing sparsity and plausibility.

Counterfactual Ascent (CFAE) in @cfae is a method that aims to achieve high similarity by approaching dense regions while changing as few features as possible. Neighbor Counterfactual search (NeCS) in @Necs also aims to achieve high sparsity, and does so by swapping features with points in the target cluster.

  === Counterfactual Ascent (CFAE) <cfae>
In this section, we introduce two methods for finding counterfactuals using a sampling based iterative strategy. The first method is Counterfactual Descent (CFDE), which aims to approach high density areas in the target cluster while minimizing the distance to the instance. Since this method has some core issues regarding plausibility and diversity, Counterfactual Ascent (CFAE) was developed as an improvement to CFDE. As the issues with CFDE heavily contributed to the creation of CFAE, we first describe CFDE.

CFDE uses sampling in order to find a highly similar counterfactual. By using a different sample in each step, the counterfactual is iteratively improved by approaching the cluster boundary. Initially, the counterfactual is set to the target cluster center $x'_0=c_"target"$. $x'$ is then recalculated in each iteration $i$ as
$ x'_i = x'_(i-1) + lambda_1  (y_i e_(u') - x'_(i-1) e_(u')), $
where
$ u' = arg &min_(u in F)  &d (x,x'_(i-1) + lambda_1(y_i e_u - x'_(i-1) e_u)). $

In this equation, $d$ is the Euclidean distance, $F$ is the set of features in the dataset, $y_i$ is a random sample chosen uniformly from cluster $C_"target"$ in iteration $i$, $lambda_1$ is the learning rate, and $e_u$ is the $u$'th basis vector.

Less formally, in each iteration $i$ we start by picking a random sample $y_i$, and pick the best feature $u' in F$ based on the minimization term. This term measures how much closer the counterfactual $x'_(i-1)$ would get to $x$, if we moved $x'_(i-1)$'s feature $u$ towards $y_i$ weighted by the learning rate. The feature change that decreases the distance to $x$ the most is then applied to the counterfactual. Due to the randomness of this approach, we can run it multiple times to produce multiple counterfactuals.

In some iterations, no such feature can decrease the distance to $x$, or the change might move the counterfactual into another cluster, thereby invalidating it. When this happens we make no change to $x'_i$. We call this occurrence a miss. After a certain amount of misses, the algorithm stops and the final iteration's counterfactual is used. Thus, the algorithm will always find a valid counterfactual since it starts in the target cluster and moves towards the instance $x$.

Given that CFDE's starting point is the center of the target cluster, every feature in the resulting counterfactual is likely to be different from $x$, resulting a sparsity of 0. Therefore, we instead pick the starting point by greedily resetting features from $c_"target"$ to the original features from $x$, starting with the features which bring the starting point closest to $x$, but still keeps the starting point in the target cluster. An example execution of CFDE with this modification can be seen in @cfde_example. The purple line represents the path of the counterfactual. This path starts out with one feature equal to that of the instance as a result of the greedy selection. After the greedy selection, the counterfactual slowly approaches the cluster boundary until it can no longer improve.

#figure(
  image("../assets/cfde_example.png", width: 60%),
  caption: [CFDE execution on a synthetic 2D dataset.]
)<cfde_example>

CFDE does however have some deficiences, the primary one being diversity. Due to the greedy sparsity pick, running the algorithm multiple times results in almost identical counterfactuals despite the random sampling. Additionally, whether the method produces sparse counterfactuals or not depends entirely on if it is possible to pick enough features in the greedy pick step. Lastly, there is no proper stopping condition, which means the method is not guaranteed to end up close to any points in $c_"target"$, hence limiting plausibility.

In order to fix these issues, we suggest Counterfactual Ascent (CFAE) as another approach to the problem. While similar to CFDE, several changes have been made in order to solve the core issues outlined above. Like before, the goal is to find a counterfactual $x'$ similar to $x$ by using an iterative heuristic sampling process. In CFAE, we initially let $x'_0 = x$, meaning it starts from the instance instead of a point in the target cluster. 

Additionally, the algorithm starts by sampling a random point $z$ in the target cluster, which the algorithm will approach in each iteration. Then, by running the algorithm multiple times, one can obtain multiple different counterfactuals, since a new $z$ is sampled each time. Finally, we account for immutable features by using the user-defined set $F_"mutable"$, which contains the set of mutable features in the dataset.


Next, $x'_i$ is recalculated in each iteration as
$ x'_i = x'_(i-1) + lambda_1  (y_i e_(u') - x'_(i-1) e_(u')), $<cfae_it>
where
$ 
u' = arg min_(u in F_"mutable") P(u,x,x'_(i-1)) dot &d(z,x'_(i-1) + lambda_1(y_i e_u - x'_(i-1) e_u))
$
and $y_i$ is a point in the target cluster chosen uniformly at random in step $i$. $P(u,x,x'_(i-1))$ is a penalty which encourages sparsity by punishing changing features not changed previously. This penalty function is defined as

$ 
P(u,x,x') = cases(1 &"if" x_u != x'_u, lambda_2^(psi) &"if" x_u = x'_u",") 
$
where $psi$ is the number of features changed between $x$ and $x'$:
$
psi = sum_(l in F)   bb(1)_(x_l != x'_l)
$

and $x_u$ indicates the value of the $u$'th feature of $x$. $lambda_2$ is the penalty parameter, which must be larger than $1$. Thus, if feature $u$ has already been changed previously, then there is no penalty. On the other hand, if feature $u$ has not been changed previously, then the penalty is $lambda_2 ^("#changed features")$. This has the effect of making it exponentially more expensive to change any feature not previously altered.

In order to get an intuitive understanding of CFAE and @cfae_it we present a high-level explanation along with the pseudocode in @alg:cfae. Initially, $x'_0$ is equal to the instance $x$ which is the starting point of the algorithm. In each iteration, a random point $y_i$ is sampled from the target cluster. We pick the feature of $y_i$ which, when applied to the counterfactual, will move the counterfactual closest to $z$. This evaluation is weighted by the penalty, such that picking features which we already changed in previous iterations is favored. This process is repeated until the target cluster is reached.

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Pseudocode for CFAE.  "target" is the target cluster label, $x$ is the instance for which we want to generate a counterfactual, and ${C_0,C_1, dots, C_(k-1)}$ is the clustering. The cluster assignment rule $cal(C)$ is defined in @eq:cluster_assigment.],
  pseudocode-list(booktabs: true, title: [*CFAE*($x$, target, ${C_0,C_1,dots,C_(k-1)}$, {$lambda_1$, $lambda_2$}, $F_("Mutable") subset.eq F$)])[
  + $x' = x$
  + $z ~ C_"target"$ #comment("Picked uniformly at random")
  + *while* $cal(C)(x',{c_0,dots,c_(k-1)}) != "target"$
    + $y ~ C_"target"$ #comment("Picked uniformly at random")
    + best = $x'$
    + $"best_score" = infinity$
    + *for* $u$ *in* $F_("Mutable")$
      + $x^"temp" = x' $
      + $x^"temp"_u = x^"temp"_u + lambda_1 dot (y_u - x'_u)$
      + $"score" = P(u,x,x^"temp") dot d(x^"temp",z)$
      + *if* $"score" < "best_score"$
        + $"best" = x^"temp"$
        + $"best_score" = "score"$
    + $x' = "best"$
  + *return* $x'$
])<alg:cfae>



In @fig:cfae_example a single run of the algorithm is presented. By starting at the instance, CFAE slowly approaches the cluster boundary in order to create a valid counterfactual. At first, only a single feature is changed in each iteration due to the penalty. This continues until the improvement gained when changing the second feature outweighs the increased penalty. Finally, when the counterfactual reaches the target cluster, the algorithm terminates.

The ideal way of using CFAE is running it multiple times and then picking the best counterfactual amongst them. This is because the target $z$ is randomly sampled and therefore not guaranteed to produce a high quality counterfactual in every execution. If one would like to generate a single counterfactual, setting $z=c_"target"$ is a way to generate a single counterfactual, which is not affected by the inconsistent performance of picking a random $z$. One can also choose to simply return all found counterfactuals, which might be preferable in certain use-cases.

CFAE uses the sampled point $z$ for the general direction and then approaches dense regions by sampling $y_i$ for the actual change in each iteration. However, in certain cases it may happen that an iteration leads to no change in the counterfactual. This occurs when moving towards $y_i$ does not decrease the distance from $x'$ to $z$ for any feature value. We argue that this does not impede CFAE in terminating, as there will always be data points which allow $x'$ to improve. Thus, in each iteration, $x'$ either gets closer to $z$ or nothing is changed, making it highly unlikely that CFAE does not find a valid counterfactual if the number of iterations are unbounded. 

#figure(
  image("../assets/cfae_example.png", width: 60%),
  caption: [CFAE execution on a synthetic 2D dataset.]
)<fig:cfae_example>

If $F_"mutable" != F$ and a significant number of features are immutable, then it becomes less likely for CFAE to terminate. Specifically, this may happen if the features available to CFAE are insufficient for reaching the cluster boundary. Therefore, we can reach a state where CFAE has changed all the mutable features in $x'$ as much as possible, without generating a valid counterfactual. Of course, this same issue will exist for all possible algorithms which allow immutable features. To avoid CFAE running indefinitely, the number of iterations should therefore be bounded in practice.

Thus, the CFAE algorithm seeks to improve some of the negative aspects from CFDE. Primarily, it seeks to improve diversity of the counterfactuals by randomly sampling a point $z in C_"target"$ to move towards at the start of the algorithm. Additionally, CFAE seeks to improve plausibility by approaching the cluster boundary from the source cluster, thereby allowing it to terminate whenever it reaches $c_t$. In contrast, CFDE only stops when there are no possible improving iterations left, which may lead to counterfactuals that are outliers in the target cluster.


=== Neighbor Counterfactual Search (NeCS) <Necs>
Neighbor Counterfactual Search (NeCS) is a method which aims to generate sparse counterfactuals, where as few features as possible are changed from the original instance $x$. This is done by directly copying feature values from points in the target cluster, contrary to CFAE which uses the learning rate to gradually change features. NeCS is able to generate multiple counterfactuals for a given instance, depending on the input parameter $k$. It works by finding the $k$-nearest neighbors of the instance $x$ from points in the target cluster $C_"target"$. For each of these, it generates a single counterfactual $x'$ by iteratively replacing features of $x$ with the corresponding feature from the neighbor until it reaches the target cluster.

We want to maximize the following objective function, where $x_j$ is one of the #box[$k$-nearest neighbors]:
$ 
sum_(x_j in "k-nn"(x,C_"target")) max_(U subset.eq F, |U|<=b) g(U, x_j),
$
where
$ 
g(U, x_j) =   f(x^'_(j,U)) (d(x,x_j) - d(x_j,x'_(j,U))).
$<NeCS_obj> 
$U$ is a subset of the complete set of features $F$. It denotes the features that have been replaced in $x'_(j,U)$ by features from the neighbor $x_j$. Thus, the $i$'th feature of $x'_(j,U)$ is defined as $x_(j,i)$ if $i in U$, and $x_i$ otherwise. Finally, to restrict the optimization, the budget $b$ limits the amount of features that can be replaced, i.e. #box[$|U|<= b$].

Thus, in order to maximize the objective for any of the $k$-nearest neighbors, we need to find the best subset of the neighbor's features that will replace $x$'s features to form the counterfactual. However, given a small budget $b$, it might not be possible to generate a valid counterfactual. A valid counterfactual can only be guaranteed if the budget allows replacing every feature i.e. $b=|F|$.

The function $f$ from @NeCS_obj is defined as
$
f(x'_(j,U)) = (<x'_(j,U) - x,V>)/(||V||^2), 
$<func_f>
where $V$ is the vector going from $c_"source"$ to $c_"target"$, and $x'_(j,U) - x$  denotes the changes made  to the instance $x$. Thus, the value $f(x)$ is a scalar projection of the counterfactual changes onto the vector $V$, normalized to the range $[0,1]$ when the projected vector lies between $c_"source"$ and $c_"target"$. A value close to 1 indicates that the counterfactual is closer to $c_"target"$ than $c_"source"$. Note that the value of $f$ will be negative when the vector $x'_(j,U) - x$ points away from $c_"target"$, and will be larger than 1 if $x'_(j,U)$ moves beyond $c_"target"$.

// The objective function from @NeCS_obj measures the score of the counterfactuals generated without considering whether the generated counterfactuals are valid. Therefore, we introduce a stopping condition which ensures that the algorithm terminates whenever an iterative step produces a counterfactual in the target cluster. #todo[Remove this paragraph? or move down?]

The objective function $g$ therefore consists of two terms. The first term, $f(x'_(j,U))$, increases as the counterfactual $x'_(j,U)$ moves toward the target center, while the second term increases with proximity to the neighbor $x_j$.

Note that adding a feature $u$ to $U$ has a constant impact on $f$ independent of the set $U$. Similarly, if adding a feature $u$ to $U$ increases the second term of the objective more than adding another feature $v$, the same relation holds independent of the set $U$. With these observations, a simple initial algorithm for maximizing the objective follows: for each feature $u in F$, calculate the initial objective increase $g({u}, x_j)$. Then, simply pick the top $b$ features, stopping prematurely when $x'_(j,U)$ becomes valid.

However, an issue with this approach becomes apparent when considering the replacement of certain features. When measuring the objective score increase for a feature $u$, if $f(x'_(j,{u}))$ is negative, meaning the change vector $x'_(j,{u}) - x$ points away from $c_"target"$, the updated score $g({u}, x_j)$ will decrease, even though the counterfactual has moved closer to $x_j$. However, as the following example illustrates, this property of feature $u$ might be rectified later, when other features have been added to $U$.

Consider the following example:
$
x &= (0,0,5), #h(10pt) x_j &&= (10,8,10)\
c_"source" &= (0,0,5), #h(10pt) c_"target" &&= (10,8,4)\ \ \
V &= c_"target" - c_"source" &&= (10,8,-1) #label("eq:necs_ex")
$
$
f(x'_(j, {2})) &approx -0.03,#h(2em)  &&g({2}, x_j) &&approx -0.028\
f(x'_(j, {1})) &approx 0.39,#h(2em) &&g({1}, x_j) &&approx 0.997\
f(x'_(j, {1, 2})) &approx 0.36,#h(2em) &&g({1,2}, x_j) &&approx 1.341.
$
Here, choosing feature $2$ in the first iteration would decrease the objective. However, after choosing feature 1, adding feature 2 will instead increase the objective significantly due to the counterfactual moving closer to $x_j$. Thus, the simple algorithm above is not sufficient.

Instead, consider a greedy algorithm, which expands the set $U$ by adding the feature which increases $g$ the most. With this approach, instead of only calculating the objective increases in the beginning, a greedy approach recalculates the objective increase for each feature $u in.not U$ in each iteration. Thus, this approach does not suffer from the deficiency above. 

#heading([Computational hardness of the objective function], level: 4, outlined: false, numbering: none)
Given the properties of the optimization problem above, we speculate that picking the optimal $b$ features is computationally hard. Although we do not argue this formally, we present an intuitive argument. We consider each neighbor $x_j$ separately, as the overall objective is simply a sum of the objective scores for each $x_j in "k-nn"(x, C_"target")$. 

To solve the optimization problem for a neighbor $x_j$, an algorithm must select a set $U subset.eq F$ with $|U|<= b$, such that $g(U, x_j)$ is maximized. Considering the example in @eq:necs_ex, we define $F^"safe"_U = {u in F: f(x'_(j,U union {u}))>=0}$ as the subset of features which do not exhibit the deficiencies presented in the example i.e. the features which do not decrease the objective function when added to the set $U$. As the example demonstrated, $F^"safe"_U$ is dependent on $U$, and may change as more features are added to $U$.

Thus, in a given iteration, the combination of previously picked features $U$ will potentially impact the set of possible next choices $F^"safe"_U$. If a feature $u_1$ leads to the largest increase of $g$ in a given iteration, choosing it might prove to be non-optimal in the long term, as selecting another feature $u_2$ may imply better future iterations. In a situation where only 2 picks remain based on the budget $b$, picking $u_2$ instead of $u_1$ might enable a bigger improvement in the objective score in the last iteration. 

If the situation above holds, the problem generalizes to any amount of interacting choices, implying that the problem is computationally hard. A greedy algorithm is therefore not guaranteed to produce an optimal solution.

#heading([Approximation Algorithm for the NeCS Problem], level: 4, outlined: false, numbering: none)
Based on the discussion above, it is infeasible to compute an optimal solution to the NeCS problem. We therefore propose the greedy approximation algorithm given in @alg:necs. In this section, we first prove an approximation bound for this algorithm. In the following proofs, we assume a static $F^"safe"$, which may be computed initially as 
$ 
F^"safe" = {u in F: f(x'_(j,{u}))>=0}. 
$ <eq:necs_fsafe>
However, the final algorithm does allow for dynamically adding items to $F^"safe"$ by recomputing the set in each iteration. Additionally, we consider the counterfactual found for each neighbor $x_j$ independently. 


We want to show that the objective function
$
g(U, x_j) = f(x'_(j,U)) dot (d(x,x_j) - d(x_j,x'_(j,U)))
$<eq:g>
from @NeCS_obj has the three properties required for yielding an approximation guarantee for the greedy algorithm as covered in @nemhauserBestAlgorithmsApproximating1978. These properties are: non-negativity, non-decreasing, and submodularity. For a function to be submodular it is required that, for a set $S subset.eq T$, including a new feature in $S$ would increase the objective at least as much as including it in $T$. Thus, as we include more features, we get diminishing returns for the objective function.
We prove these properties in the following 3 lemmas.

#lemma[
  $g$ is non-negative, meaning $forall S subset.eq F^"safe": g(S,x_j) >= 0.$
]<NeCS_non_nega>
#proof[
  Looking at the definition in @eq:g, it is clear that for $g$ to be non-negative it must hold that $forall S subset.eq F^"safe": d(x,x_j)>=d(x'_(j,S), x_j)$. Since every replacement performed on $x$ can only move it closer to $x_j$, this clearly holds. $f$ produces a non-negative result since $S subset.eq F^"safe"$, and does therefore not interfere with the non-negativity property.
]

#lemma[
  $ g$ is non-decreasing, meaning $ forall S  subset.eq F^"safe": g(S,x_j) <= g(S union {u},x_j) #h(15pt) "for all" u in F^"safe" \\ S. $
]<NeCS_non_decreasing>
#proof[
  By a similar reasoning as the proof of @NeCS_non_nega, any sequence of replacements performed on $x$ can only move it closer to $x_j$. Since $d(x, x_j)$ is constant throughout the algorithm, $d(x,x_j) - d(x_j,x'_(j,S))$ cannot decrease when adding a feature to $S$.

  Since $F^"safe"$ is exactly the set of features which do not move $x'_(j,S)$ further from $c_"target"$, the scalar projection of $x'_(j,S)$ onto $V$ is also non-decreasing.
]



#lemma[
  $g$ is submodular, meaning that $ forall S subset.eq T subset.eq F^"safe": g(S union {u}, x_j) - g(S,x_j) >= g(T union {u},x_j) - g(T,x_j),  #h(15pt) $
  for all $u in F^"safe" \\ T.  $
]<NeCS_sub>
#proof[
#let c(content) = {text([(#content)])}
$
  g(S union {u}, x_j) - g(S,x_j) &>= g(T union {u}, x_j) - g(T, x_j) \
  
  &arrow.t.b.double #c[Expand @NeCS_obj]\
    
  f(x'_(j,S union {u})) d(x,x_j) -  f(x'_(j,S union {u}))d(x'_(j,S union {u}),x_j) &-  f(x'_(j,S)) d(x,x_j) + f(x'_(j,S))d(x'_(j,S),x_j) \
  &>= \ 
  f(x'_(j,T union {u})) d(x,x_j) -  f(x'_(j,T union {u}))d(x'_(j,T union {u}),x_j) &-  f(x'_(j,T)) d(x,x_j) + f(x'_(j,T))d(x'_(j,T),x_j) \
  &arrow.t.b.double\
  (f(x'_(j,S union {u})) - f(x'_(j,T union {u}))) d(x,x_j) -  f(x'_(j,S union {u}))d(x'_(j,S union {u}),x_j) &+  ( f(x'_(j,T))-f(x'_(j,S))) d(x,x_j) + f(x'_(j,S))d(x'_(j,S),x_j) \
  &>= \ 
  -  f(x'_(j,T union {u}))d(x'_(j,T union {u}),x_j) &+ f(x'_(j,T))d(x'_(j,T),x_j) \
  
  &arrow.t.b.double #c[Expand @func_f]\
  
  ((<x'_(j, S union {u})-x, V> - <x'_(j,T union {u})-x, V>)/(|V|^2)) &d(x,x_j) -  f(x'_(j,S union {u}))d(x'_(j,S union {u}),x_j) \ + ((<x'_(j,T)-x,V> - <x'_(j,S)-x,V>)/(|V|^2)) &d(x,x_j) + f(x'_(j,S))d(x'_(j,S),x_j) \
  &>=   \ 
  -  f(x'_(j,T union {u}))d(x'_(j,T union {u}),x_j) &+ f(x'_(j,T))d(x'_(j,T),x_j) \
  &arrow.t.b.double#<dot_product_deletion> \
  
  f(x'_(j,S))d(x'_(j,S),x_j) &- f(x'_(j,S union {u}))d(x'_(j,S union {u}),x_j)  \
  &>= #<last_proof_line> \ 
  f(x'_(j,T))d(x'_(j,T),x_j) &- f(x'_(j,T union {u}))d(x'_(j,T union {u}),x_j). \
$
#set math.equation(number-align: end+horizon)

The implication in #ref(<dot_product_deletion>, supplement: "line") follows from the fact that 
$ <x'_(j,T)-x,V> - <x'_(j,S)-x,V> = <x'_(j,T\\S)-x,V> $ 
and 
$ <x'_(j,S union {u})-x, V> - <x'_(j,T union {u})-x, V> = -<x'_(j,T\\S)-x,V>. $

In #ref(<last_proof_line>, supplement: "line"), it follows from the same reasoning that
$
f(x'_(j,S)) - f(x'_(j,S union {u})) = f(x'_(j,T)) - f(x'_(j,T union {u})).
$
Thus, the inequality in #ref(<last_proof_line>, supplement: "line") holds if
$
d(x'_(j,S),x_j) - d(x'_(j,S union {u}),x_j) >= d(x'_(j,T),x_j) - d(x'_(j,T union {u}),x_j).
$ 

To show this, we expand the Euclidian distance function:

$ 
sqrt(sum_(i in F^"safe"\\S union {u}) (x^('i)_(j,F^"safe"\\S union {u}) - x^i_j)^2 + (x^('u)_(j,{u}) -x_j^(u))^2) - 
sqrt(sum_(i in F^"safe"\\S union {u}) (x^('i)_(j,F^"safe"\\S union {u}) - x^i_j)^2) \ 
>= \
sqrt(sum_(i in F^"safe"\\T union {u}) (x^('i)_(j,F^"safe"\\T union {u}) - x^i_j)^2 + (x^('u)_(j,{u}) -x_j^(u))^2) - 
sqrt(sum_(i in F^"safe"\\T union {u}) (x^('i)_(j,F^"safe"\\T union {u}) - x^i_j)^2 ).
$

With this, since 
$ 
forall m,n,k in RR^+: n >= m ==> sqrt(n + k) - sqrt(n) <= sqrt(m + k) - sqrt(m)
$
and 
$
S subset.eq T ==> |F^"safe"\\S| >= |F^"safe"\\T|
$
the inequality holds.
]

The three lemmas above, combined with the result from @nemhauserBestAlgorithmsApproximating1978, lead to an approximation guarantee for the greedy algorithm given in @alg:necs. The algorithm uses marginal gain to pick the best feature replacement in each iteration, and repeats the process for all nearest neighbors of the instance $x$ in the target cluster $C_"target"$, yielding $k$ counterfactuals, one for each neighbor. In our case, marginal gain is defined as
$ 
g(S union {u}, x_j) - g(S,x_j)
$
for all $u in F^"safe" \\ T.  $<NeCS_mar_gain>


#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Pseudocode for NeCS.  The cluster assignment rule $cal(C)$ is defined in @eq:cluster_assigment.],
  pseudocode-list(booktabs: true, title: [*NeCS*($x$, target, ${C_0,C_2,dots,C_(k-1)}$, $k$)])[
    // #set text(font: "Roboto Mono")
  + cfs = ${}$\
  + *for* n *in* *$k$-NearestNeighbors*($x,C_("target")$):
    + $x' = x$
    + $A={}$
    + *while* $cal(C)(x',{c_0,dots,c_(k-1)}) != "target"$
      + $u = arg max_(f in F^"safe" \\ A) g(A union {u}, n) - g(A,n)$
      + $A = A union {u}$
      + $x'_u  = n_u$
    + cfs = cfs $ union {x'}$
  + *return* cfs
])<alg:necs>
// #set text(font: "New Computer Modern")

In @NeCS_obj we use a budget $b$ to decide on the maximum amount of features that are allowed to change. In @alg:necs, we instead introduce a stopping condition, according to which the algorithm terminates as soon the target cluster is reached. This is done to improve sparsity and similarity by generating counterfactuals with few changed features and a lower distance to the original instance $x$. However, the budget is used to relate the stopping condition to an approximation ratio. We can calculate this approximation ratio for @alg:necs since $g$ adheres to all 3 properties proven above. It can be calculated using the following formula from @nemhauserBestAlgorithmsApproximating1978:
$ [ 1-((K-q)/(K))((K-q-1)/(K-q))^(K-q) ]dot "optimum result," $<nemhauser_eq>
where $q$ is the upper limit on the number of brute-force iterations, meaning that the first $q$ picks of the algorithm are brute-forced instead of picked greedily. In our case, we therefore simply have $q=1$. $K$ is the number of greedy steps taken by the algorithm. The approximation ratio from @nemhauser_eq then compares the greedy algorithm's objective score with the optimal objective score from performing $K$ optimal picks.

As this value will be different depending on the dataset, we pick an arbitrarily large value ($K=10^6$). Picking $g(S)$ to be the solution from @alg:necs and $g(S_"opt")$ to be the optimal solution, the approximation ratio becomes:

$ g(S,x_j) >= 0.63 dot g(S_"opt",x_j). $<NeCS_aprox>

One could consider a version of @alg:necs which instead of using the stopping condition uses the budget for exactly $b$ greedy steps. This algorithm would then adhere to the approximation ratio from @nemhauserBestAlgorithmsApproximating1978 with #box[$K=b$]. Our algorithm is a special case of this variant, where instead of setting $b$ beforehand, we stop making greedy picks after entering the target cluster. In the general case, if we wanted to find $b$ such that a counterfactual stopped after crossing the clustering boundary, this could be done with trial and error. Our variation achieves the same result by terminating itself after crossing the cluster boundary.

Regarding the time complexity of @alg:necs there are two main parts. First, a $k$-nearest neighbors procedure is used exactly once, which has time complexity $O(n|F|)$ where $n$ is the number of points and $|F|$ is the number of features. The greedy part is where a counterfactual for each neighbor is generated, which has complexity $O(k|F|^2)$ where $k$ is the number of neighbors. This puts the overall time complexity at $O(n|F| + k|F|^2)$.

Finally, recall that Lemmas#ref(<NeCS_non_nega>, supplement:""),#ref(<NeCS_non_decreasing>, supplement:""), and#ref(<NeCS_sub>, supplement:"") were proven by letting $F^"safe"$ be statically defined as in @eq:necs_fsafe throughout the algorithm. However, in our implementation, we are able to recompute the set $F^"safe"$ in each iteration, thereby accounting for the possibility of adding previously unsafe features to the set of swapped features $U$.