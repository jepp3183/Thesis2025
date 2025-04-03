#import "../lib.typ": *

== Novel Heuristic based Counterfactual methods
In centroid based methods, the clustering boundary is a concept used to define the hyperplane, where a point on each side of the boundary would be assigned different labels. Hence, it is clear that for any point a valid counterfactual candidate could be the closest point which lies just beyond the boundary of the target cluster, since such a point would have very high similarity. It is however not obvious whether such a point would be a feasible counterfactual, given that we aim to sparse and feasible counterfactuals. By using heuristic methods, we can approach the decision boundary in steps without sacrificing sparsity and feasibility.

Counterfactual Ascent @cfae is a method that aims to achieve a high feasibility by approaching dense regions while changing as few features as possible. Neighbor Counterfactual search @Necs aims to achieve a high sparsity by approaching points in the target cluster near the decision boundary by changing as few features as possible.

=== CFAE: Counterfactual Ascent <cfae>
Before defining CFAE it is worth describing it's predecessor, that being Counterfactual Descent(CFDE). CFDE aims to use sampling in order to find a feasible counterfactual $hat(x)$ for the instance $x$. This process iteratively updates a counterfactual until the counterfactual no longer improves. Initially, the counterfactual is set to the target cluster center $hat(x)_0=c(C_t)$. Then in order to improve $hat(x)$ the following step is executed:
$ hat(x)_i = hat(x)_(i-1) + lambda_1 arg &min_(y_i e_u - hat(x)_(i-1) e_u, u in F)\  &d (x,hat(x)_(i-1) + (y_i e_u - hat(x)_(i-1) e_u)lambda_1) $
Where $d(*,*)$ is a distance metric, $F$ is the set of features in the dataset, $y_i$ is a random sample chosen uniformly from cluster $C_t$ in iteration $i$, $lambda_1$ is the learning rate and $e_i$ is $i$-basis vector. 

Less formally, in iteration $i$ we start by picking a random sample $y_i$, and then we run through each feature in the dataset in the minimization term. The distance function's first input is the instance we are generating a counterfactual for. The next term is current counterfactual $hat(x)_(i-1)$ plus it's difference to the sample $y_i$ according to feature $u$ multiplied by the learning rate. Taking this into account, the distance metric measures how much closer the counterfactual $hat(x)_(i-1)$ would get to $x$ if we moved $hat(x)_(i-1)$'s feature u towards $y_i$'s feature u weighted by the learning rate. The feature change which decrease the distance to the instance $x$ the most is then applied to counterfactual.

In each iteration CFDE looks at the sample and greedily chooses which feature it should make more similar to the sample in order to get closer to $x$. The only exception to this is when no feature would decrease the distance to $x$ or if the change would move the counterfactual into another cluster, in which case we make no change to $hat(x)_i$, we call this a miss. After a certain amount of misses the algorithm stops and the final iteration's counterfactual is used. 

As CFDE's starting point is the center of the target cluster, every feature in the resulting counterfactuals is likely to be different from the instance $x$. Therefore, we initially greedily find the features which can made identically to the instance's features without changing the label. An example of this process can be seen on @cfde_example.

#figure(
  image("../assets/cfde_example.png"),
  caption: [Example CFDE execution run on the make_blobs randomly generated dataset method by @ScikitlearnMachineLearning]
)<cfde_example>

CFDE does however have some major drawbacks. The first being it's sparsity. Since we start at the target center, we rely on resetting features such that they are identical to $x$, and in some cases this might not be possible for immutable features. The sparsity problem can to some extent be improved by introducing a penalty for changing to many features in the iterative step, but even that relies on the pre-processing step to change enough features to be identical to the instance. There is also the troublesome flaw that it can be hard to generate diverse counterfactuals, as the preprocessing step can sometimes leave  very little room for the iterative step to generate diverse counterfactuals if the method is run multiple times. 

In order to fix these issues, we introduce a new method, that being Counterfactual Ascent (CFAE). While similar to CFDE there have been several changes made in order to solve the core issues of CFDE. Just like before we aim to find a counterfactual $hat(x)$ similar to $x$ by using an iterative process using sampling. In CFAE we set $hat(x)_0 = x$. First, we take a look at the iterative step:
$ hat(x)_i = hat(x)_(i-1) + lambda_1 arg &min_(y_i e_u - hat(x)_(i-1) e_u, u in F)\  P(u,x,hat(x)_(i-1)) &d(psi,hat(x)_(i-1) + (y_i e_u - hat(x)_(i-1) e_u)lambda_1) $<cfae_it>
Where $psi$ is a point in the target cluster chosen uniformly at random in each step. $P(f,a,b)$ is a penalty which punish changing a feature not changed previously, it does so the following way:
$ P(f,a,b) = cases(1 &"if" a.f != b.f, lambda_2^(sum_(u in F) cases(1 "if" a.f != b.f, 0 "else")) &"if" a.f = b.f) $
Where $a.f$ indicates the values stored at row $f$ in vector $a$. $lambda_2$ is the penalty hyper-parameter. What this means is that if feature $f$ have already been changed previously, then there is no penalty. On the other hand, if $f$ has not been changed previously, then the penalty is equal to $lambda_2 ^("amount of features changed total")$. This has the effect of making it exponentially more expensive to change any feature not previously changed, and hence improving the sparsity of the method.

In order to get an intuitive understanding of CFAE and @cfae_it we now give an informal explanation. Initially, $hat(x)_0$ is equal to the instance $x$ which is the starting point of the method. Then we pick the target, which is chosen amongst the points in the target cluster at random. This random selection exists such that running the method multiple times produces diverse counterfactuals. Then in each iteration we sample $y$ which is a random point in the target cluster. By running through $y$'s features, we find out which feature applied to $hat(x)_i$ weighted by the learning rate gets us the closest to $psi$. This evaluation is weighted by the penalty, such that picking features which we already changed in previous iterations are favored. This iterative process is repeated until the target cluster is reached, after which we terminate the process.

The ideal way of using CFAE is running it multiple times and then picking the best counterfactual amongst them, this is because the target metric is random and hence not guarantied to produce good results in every run-through.

CFAE aims to fix the issues with CFDE by approaching the target cluster iteratively instead of approaching the origin cluster, this distinction makes it easier to limit the amount of features changed but also allows us to go towards different point in the target cluster and thereby produce diverse counterfactuals. A visualization of CFAE can be seen on @cfae_example. 

#figure(
  image("../assets/cfae_example.png"),
  caption: [Example CFAE execution run on the make_blobs randomly generated dataset method by @ScikitlearnMachineLearning]
)<cfae_example>

The implementation details are covered in @cfae_alg.
#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Pseudo-code for CFAE, predictor is the method used to label new points, target and origin are both labels, $x$ is the instance for which we want to generate a counterfactual and ${C_0,C_1, dots, C_j}$ is the clustering],
  pseudocode-list(booktabs: true, title: [Counterfactual-Ascent($x$, target, origin, ${C_0,C_1,dots,C_j}$, predictor, {$lambda_1$, $lambda_2$}, $F_("Mutable") subset.eq F$)])[
  + $hat(x) = x$
  + $psi = z in C_t$ \#Picked uniformly at random
  + *while* predictor($hat(x)$) = origin *or* predictor($hat(x)$) != target
    + $y in C_t$ \#Picked uniformly at random
    + $"best" = hat(x)$
    + *for* $u in F_("Mutable")$
      + $p = P(u,hat(x),x, lambda_2)$
      + step = $y[u] - hat(x)[u]$
      + $hat(x)' = hat(x) $
      + $hat(x)'[u] = hat(x)'[u] + lambda_1 dot "step"$
      + *if* $p dot d(hat(x)',psi) < d("best",psi)$
        + $"best" = hat(x)'$
    + $hat(x) = "best"$
  + *return* $hat(x)$
])<cfae_alg>

Looking at @cfae_alg it not clear whether it always terminates, however this is indeed the case as can be seen in @cfae_terminates.

#lemma[Given a clustering ${C_1,C_2,dots,C_j}$, target label $c_t$ and origin label $c_o$. @cfae_alg always return a counterfactual $hat(x)$ for instance $x$ where $"predictor"(hat(x))=c_t$ when $F_"mutable"=F$.
]<cfae_terminates>

#proof[
  Looking at line 3 in @cfae_alg the algorithm only terminates when counterfactual $hat(x)$ belongs to cluster $C_t$. In any of these iterations we have our counterfactual $hat(x)_i$ and sample $y_i$. Since $y_i$ belongs to cluster $C_t$, and hence lies on the other side of clustering boundary respective to $hat(x)_i$ what we want to show is that $d(hat(x)_(i+1),y_i)<d(hat(x)_i,y_i)$, since this implies that we have indeed moved closer to the clustering boundary. If $hat(x)_i=y_i$ then we would already have terminated before entering the current iteration since $y_i$ belongs to $C_t$. If $exists u in F, hat(x)_i.u!=y_i.u$ then it is trivial that changing the respective feature for $hat(x)_i$ such that it is closer to $y_i$ makes the distance shorter. Hence, in each iteration, $hat(x)$ moves closer to the clustering boundary and therefore will at some point cross it due to the fact that not all point can lie exactly on the decision boundary.
]<proof_cfae_terminates>

As can be seen in @cfae_terminates the algorithm is not guarantied to terminate when $F_"mutable" != F$. This is because of the argument from @proof_cfae_terminates which stated that if $hat(x)_i = y_i$ the algorithm would have terminated before entering the loop on line 3 in @cfae_alg. If some features are not mutable then this does not hold since we can reach a state where $forall u in F_"mutable", hat(x)_i.u = y_i.u and exists u in F, hat(x)_i.u != y_i.u$. 

=== NeCS: Neighbor Counterfactual Search <Necs>
Neighbor counterfactual Search (NeCS) is a method which aims to generate sparse counterfactuals. If we want to generate $k$ counterfactuals for instance $x$, then we define that to be the k-NeCS method going forward. k-NeCS works by finding the k-nearest neighbors#todo[Source?] for the instance $x$ relative to points in the target cluster $C_t$. These neighbors are then used to generate counterfactuals. The objective function we ant maximize is the following:
$ "maximize"_(F subset.eq A, |F|<=b) epsilon = sum_(x_j in "k-nn"(x,C_t)) max_(u subset F) f(hat(x)_(u,j)) (d(x,x_j) - d(x_j,hat(x)_(u,j))) $<NeCS_obj> 
Where $hat(x)_(u,j).i = cases(x_j.i "if" i in u, x.i "else")$ and $j$ is the associate neighbor. $f(dot)$ is the label predicted by the model, meaning that in order to maximize the function for any of the k-nearest neighbors, you would need to find the best subset of features in the neighbor which would replace $x$'s features in the counterfactual. The set $A$ is  the set off all features such that the budget restricts the maximization in order to avoid the trivial solution $hat(x)=x_j$. 

The model function from @NeCS_obj is defined as:
$ f(bold(n)) = (<bold(n),bold(V)>)/(|V|) $

As can be seen in @NeCS_obj the objective is to find the subset of features such that the resulting counterfactual has score $f(hat(x)_(u,j)) > 0.5$ which implies that is lies in the target cluster. What this problem essentially boils down to for each neighbor is finding the subset of features $u in F$ such that $ceil(f(hat(x)_(u,j))-0.5) = 1$ and $d(x_j,hat(x)_(u,j))$ is minimized. Since the objective function does not require that the produced counterfactuals lie in the correct cluster, it is up to a stopping condition to maintain validity. Metrics such a sparsity and similarity are considered in the stopping condition.

It is however impossible to directly solve this problem due to it being NP-Hard, we show this by reducing it to the Knapsack #todo[source?] problem. The Knapsack problem is concerned with picking a subset of items, all of which have a weight and a score. The goal is to pick the subset with the greatest score with total weight under some threshold.

The NeCS algorithm start with the instance and goes towards the neighbor. The items in the knapsack problem are features, such that putting a feature in the back implies setting $n.u = x.u$. We define the weight of each feature to 1 and the weight threshold to be the budget $b$. Each item's value would then be the resulting counterfactuals objective function score after applying that item's feature. If we have some item $f$ representing feature $f$, then the score would be $f(hat(x)_(u union {f},j)) (d(x,x_j) - d(x_j,hat(x)_(u union {f},j))$.

This reduction does however not function like the classical definition of the knapsack problem, since the scores of each feature can change dependently of previous items in the bag. This is due to the fact that when a feature $u$ is added to the back, the point $n$ changes, which would invalidate previously calculated distances. The weights never change, so we only consider the version of knapsack where the scores are dependent on previous insertions into the bag. This problem is studied in @gawiejnowiczKnapsackProblemsPositiondependent2023 where they designate our problem as a P2 variation of the knapsack problem. Just like knapsack, the P2 variation is also weakly NP-hard. @gawiejnowiczKnapsackProblemsPositiondependent2023

With this, it is clear that an approximation algorithm is required. We start by redefining our objective. We consider the counterfactual found for each neighbor independently. 

$ beta(x) = max_(u subset F) f(hat(x)_(u)) (d(x,x_j) - d(x_j,hat(x)_(u))) $<NeCS_ind_obj>

Now, since @NeCS_ind_obj has the three properties required for us to make a greedy algorithm with an approximation guarantee as covered in @nemhauserBestAlgorithmsApproximating1978. These properties are non-negative, non-decreasing and submodular. We will now prove these three properties.

#lemma[
  $beta(dot)$ is non-negative.
]<NeCS_non_nega>
#proof[
  Looking at @NeCS_ind_obj is becomes clear that for $beta$ to be non-negative it requires that $forall u in F, d(x,x_j)>=d(x_j,hat(x)_u)$. Since every action taken on $hat(x)_0$ and future iterations can only move it closer to $x$ it is obvious that $d(x,x_j)>=d(x,hat(x)_u)$ is always true. $f(dot)$ produces a result between $0$ and $1$ and does therefore not interfere with the non-negative property.
]

#lemma[
  $beta(dot)$ is non-decreasing, meaning that $beta(S) <= beta(T), forall S subset.eq T$.
]<NeCS_non_decreasing>
#proof[
  For the instance $x$, $S$ is then a subset of features where $hat(x)_S = cases(x.i "if" i in S, hat(x).i "else")$ such tht $S subset.eq F$. Then adding extra features to $S$ such that $T = S union {u subset.eq F}$ would then always move $hat(x)_T$ closer towards $x$ than $hat(x)_S$. Since it holds that $d(hat(x)_T,c_"target")<= d(x,c_"target")$ it also holds that $f(hat(x)_T)>=f(hat(x)_S)$.
]

#lemma[
  $beta(dot)$ is submodular, meaning that $beta(S union {i_k}) - beta(S) >= beta(T union {i_k}) - beta(T), forall S subset.eq T $
]<NeCS_sub>
#proof[
  $ beta(S union {i_k}) - beta(S) &= max_(u subset S union {i_k}) f(hat(x)_(u)) (d(x,x_j) -d(x_j,hat(x)_(u))) - \ &max_(u subset S) f(hat(x)_(u)) (d(x,x_j) -d(x_j,hat(x)_(u))) \ 
  &>= max_(u subset T union {i_k}) f(  hat(x)_(u)) (d(x,x_j) -d(x_j,hat(x)_(u))) - \ &max_(u subset S) f(hat(x)_(u)) (d(x,x_j) -d(x_j,hat(x)_(u))) space space (1)\
  &>= max_(u subset T union {i_k}) f(  hat(x)_(u)) (d(x,x_j) -d(x_j,hat(x)_(u))) - \ &max_(u subset T) f(hat(x)_(u)) (d(x,x_j) -d(x_j,hat(x)_(u))) space space (2) \ 
  &= beta(T union {i_k}) - beta(T) $
  Steps $(1),(2)$ both were possible due to the non-decreasing property @NeCS_non_decreasing.]

With this we can design a method that uses marginal gain in order to pick the best counterfactual in each iteration. Marginal gain is defined as follows:
$ rho_beta (S, i_k) = beta(S union {i_k}) - beta(S) $<NeCS_mar_gain>
Now if we design an algorithm that greedily picks features according to @NeCS_mar_gain we would then be able to use @nemhauserBestAlgorithmsApproximating1978 to find the approximation ratio for said algorithm. 

In @NeCS_obj we used as budget to decide on the maximum amount of features that were allowed to change. In @NecS_alg we do however not restrict this. The reason for this is the stopping condition, where the algorithm terminates as soon the clustering boundary is reached. This is because the sparsity and similarity metrics prefer counterfactuals which have few changed features and exists close to the boundary, both of which are enhanced by such a stopping condition.

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Pseudo-code for NeCS],
  pseudocode-list(booktabs: true, title: [NeCS($x$, target, origin, ${C_0,C_1,dots,C_j}$, predictor)])[
  + $k_1,k_2,dots,k_k=$ *K-NearestNeighbors*($x,C_t$)
  + cfs = ${}$\
  + *for* k *in* [$k_1,k_2,dots,k_k$]:
    + $"cf" = x$
    + $F'={}$
    + *while* predictor(cf)$!=$ target
      + $u = arg max_(f in F and f in.not F') rho_beta (F', f, k)$
      + $F' = F' union {u}$
      + $"cf".u  = k.u$
    + cfs = cfs $ union {"cf"}$
  + *return* cfs
])<NecS_alg>

With this we can calculate our approximation ratio for @NecS_alg since $beta$ adheres to all 3 properties, non-negative @NeCS_non_nega, non-decreasing @NeCS_non_decreasing and submodularity @NeCS_sub. By using the process in @nemhauserBestAlgorithmsApproximating1978 we can use these properties to find our approximation ration. The equation has the following form:
$ [ 1-((K-q)/(K))((K-q-1)/(K-q))^(K-q) ]dot "optimum result" $<nemhauser_eq>
Where $q$ is the upper limit on the amount of iteration we are willing to brute-force our picks. Meaning that the first $q$ picks in the method are brute-forced instead of using greedy selection, for our purpose we have $q=1$. $K$ is the maximum amount of greedy picks necessary to execute the algorithm. As this value will be different depending on the dataset, we pick an arbitrarily large value ($K=10^6$). Then of pick $beta(S)$ to be our solution and $beta(S_"opt")$ to be the optimal solution, then our approximation ratio becomes. 

$ beta(S) >= 0.63 dot beta(S_"opt") $<NeCS_aprox>
Since our algorithm is restricted by a budget this approximation ratio is worse than @NeCS_aprox. In order to find the approximation ratio for any solution one must count the amount of steps @NecS_alg takes before terminating, and then using this value as $K$ in @nemhauser_eq. As an example if @NecS_alg terminates in 15 steps, then the approximation ratio becomes $beta(S_15) >= 0.67 dot beta(S_"opt")$ 

Looking at the time complexity of @NecS_alg there are two parts. Firstly, the K-nearest neighbors method is used, which has time complexity $O(n f)$ where $n$ is the amount of points and $f$ is the amount of features. The next part is where the counterfactuals for each neighbor is generated, which has complexity $O(k f^2)$ where k is the amount of neighbors. This puts the overall time complexity at $O(n f + k f^2)$.

