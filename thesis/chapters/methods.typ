#import "../lib.typ": *
= Supervised counterfactuals<sec:supervised_methods>
In this section, we outline two well-known existing methods for finding counterfactuals in a supervised setting. Since there exists very few methods for generating counterfactuals in an unsupervised setting, we use the supervised methods along with a classifier trained on the cluster labels in order to establish two baseline methods to compare other methods against. For one of these baseline methods, we use DiCE @mothilalExplainingMachineLearning2020, which is able to generate a set of highly diverse counterfactuals. For the second baseline we use BayCon @romashovBayConModelagnosticBayesian2022, which uses Bayesian optimization and also serves as the main building block for one of the few existing methods for the unsupervised setting presented in @spagnolCounterfactualExplanationsClustering2024.

//== Guided By Prototypes<prototypes>
//A state-of-the-art counterfactual method for supervised learning is presented in @vanlooverenInterpretableCounterfactualExplanations2021, in this paper they propose a fast model agnostic method for finding interpretable counterfactual explanations using class prototypes. In this paper the counterfactuals are generated to go from a specific class instance to any other class. 

// They present two approaches for finding such prototypes, and suggest objectives that can be used in conjunction to speed up the counterfactual search process. The proposed method can either be run on a black box model, where we will only need access to a prediction model or be given the model architecture to take advantage of automatic differentiation when performing the counterfactual search process. They found that using the architecture for automatic differentiation introduced a drastic reduction in the computation time of counterfactuals, as the method using a prediction model would have to evaluate each feature change using a prediction to approximate the gradient numerically.

// They furthermore introduce four desirable properties for their counterfactuals, which align with our evaluation metrics, which are as follows: Similarity, Sparsity, Plausibility and Running time. The first method for finding class prototypes they present uses an autoencoder trained on the training set, and defines each class prototype to be equal to its class' mean encoding of the points corresponding to that class. To find the label for each datapoint in the training set we can simply query our black box model for predictions. The generated prototypes are defined in the latent space and the distance measures between our explained instance or counterfactual and our prototypes are therefore also measured in the latent space. 

// The second method for finding class prototypes utilizes k-d trees@bentleyMultidimensionalBinarySearch1975. After labeling the training set by calling the predictive model, each class of label $i$ can be represented as a k-d tree built with instances with label $i$. A k-d tree is a special case of binary space partitioning trees used for storing k-dimensional data. k-d trees can be constructed in different ways, but a canonical way to construct them is to loop through all k-dimensions and in each step split the data on the median, the points will then be inserted into their subtree and recursively split until all points are stored in leaf nodes, this procedure will generate a balanced tree for storing the points, as in each iteration the data is split on the median. To find a prototype for an explained instance $x_0$, they run through each k-d tree with a different label than $x_0$, find the k-nearest item in the tree using Euclidean distance, and finally out of all classes they choose the closest of these items to be the class prototype. 

// To generate a counterfactual using Guided By Prototypes, they apply a fast iterative shrinkage-thresholding optimization algorithm called FISTA@amirbeckFastIterativeShrinkageThresholding. Where the counterfactual space is restricted to be in the data manifold. The FISTA algorithm iteratively updates the change $delta$, to the explained instance using momentum for $N$ optimization steps, and optimizes to find the best result for the objective function with a different label from the initial instance.


== DiCE <sec:dice>
In @mothilalExplainingMachineLearning2020 DiCE is introduced as a method that focuses on diversity and similarity metrics in order to produce actionable counterfactuals. Although DiCE is strictly concerned with generating counterfactuals in a supervised setting, the methods and procedures remain relevant for clustering. The paper is concerned with a variety of issues that affect counterfactual generation, one being that features are not independent of each other. Imagine we have 2 features, income and marriage status. While these features might seem unrelated they are in fact not, as being married is often related to age and the amount of years the individual has been working. Another aspect is the existence of immutable features. For example, a counterfactual might suggest changing your race or gender which is not actionable. DiCE allows the user to mark these features as immutable, such that the generated counterfactuals do not modify them.

The objective of DiCE is to generate a set of counterfactuals $Z = {x'_1, x'_2, dots , x'_n }$ for a machine learning model $f$ on an instance $x$. In order to generate counterfactuals, DiCE includes a diversity and similarity metric into their generation method. However, certain features might not be mutable in some range, meaning that a feature cannot go under or over a limit set by the user. DiCE allows for such constraints as well.

The diversity metric is used to generate diverse counterfactuals, such that all combinations of feature changes are evenly distributed. DiCE defines the diversity of a set of counterfactuals in the following way:
$ #text([dpp_diversity]) = det(K), $
where $K_(i,j) = 1 / (1 + d(x'_i, x'_j))$ and $d(a,b)$ denotes the distance between $a$ and $b$ according to some distance metric. This process is known as a determinantal point processes (DPP) @kuleszaDeterminantalPointProcesses2012. Given the marginal kernel $K$, which measures similarities between counterfactuals, $det(K)$ approximates the probability that the set of counterfactuals $Z$ was picked amongst the set of all possible counterfactuals $Z'$, i.e.
$ 
P(Z | Z subset Z') approx det(K). 
$
For some instance $x$, one can imagine $Z'$ as all the possible counterfactuals that exist for the instance. Then, picturing $Z'$ as a box with its content evenly distributed, the probability of picking a subset of evenly distributed counterfactuals from the box is higher than picking a subset of similar counterfactuals. This probability is what dpp_diversity approximates. Therefore, the dpp_diversity of a set of counterfactuals is higher on more diverse sets @kuleszaDeterminantalPointProcesses2012.

Similarity is evaluated through a proximity measure, since a counterfactual that is closer to the instance $x$ will be easier for the end-user to act on. The proximity measure used to achieve similarity is defined as the negative distance between the counterfactual and the corresponding instance $x$:
$ 
"Proximity" = -1/n sum^n_(i=1) d(x'_i,x). 
$

Sparsity is the measure of how few features are changed in a counterfactual compared to the instance. This is an important property, since needing to change fewer features in order to achieve a counterfactual reduces its complexity. DiCE does not include Sparsity in the loss function, but instead uses a post-hoc method. In essence, their method for enhancing sparsity is to greedily pick features in the counterfactual that differs from the instance $x$ and change them back @mothilalExplainingMachineLearning2020.

With all this is mind, DiCE defines their loss function as
$ 
C(x) = arg min_(c_1,dots,c_k) 1/k sum^k_(i=1) "yloss"(f(c_i),y) + lambda_1/k sum^k_(i=1) d(c_i,x)\ - lambda_2 "dpp_diversity"(c_1, dots , c_k), 
$
where the first term is the loss which minimizes the difference between the target label $y$ and the actual counterfactual label $f(c_i)$. The second term is the similarity metric and the last term is the diversity metric. In order to implement this objective function, they use gradient descent, followed by the post-hoc step mentioned above in order to enhance the sparsity of the produced counterfactuals @mothilalExplainingMachineLearning2020. 

Lastly, DiCE uses a filtering step in order to ensure all counterfactuals are feasible. In this context, a counterfactual is feasible if it does not change a feature in such a way as to make it impossible to achieve. Some features might be directly related to each other in a way such that changing one is not feasible without changing the other. These relations are the reason why the filtering step exists, as without filtering, counterfactuals where only one part of any such relation is changed are infeasible. This relation can be impossible to detect without domain-specific knowledge. Before generation of a counterfactual, a user can input pairs of features which are related to each other. In this way, any counterfactual which alters one but not the other would be infeasible. An example of such a relation is education and age, since a counterfactual that suggests achieving a higher education must also account for the increased age, as these are of course highly correlated @mothilalExplainingMachineLearning2020.

== BayCon<sec:baycon>
In @romashovBayConModelagnosticBayesian2022 they introduce BayCon, a model-agnostic method for generating counterfactuals in supervised tasks, which is based on Bayesian Optimization. In this thesis, BayCon is used as a baseline method by training a classifier on the cluster labels. Additionally, it has also been adapted to the clustering task in  @spagnolCounterfactualExplanationsClustering2024, which presents one of the few unsupervised counterfactual generation methods in the field.

BayCon makes use of an optimization objective constructed using three terms that implement desirable properties for a counterfactual. The objective function for a candidate counterfactual $x'$ and an instance $x$ is defined as

$
F(x',x) = S_x dot S_y dot S_f,
$<eq:baycon_obj>
  
where $S_x$ is the similarity in feature space calculated using Gower distance, $d_("Gower")$ @gowerGeneralCoefficientSimilarity1971. Gower distance is a metric used for mixed feature spaces, as it supports computation for categorical attributes. The term is calculated as
$
S_x (x',x) = 1 - d_("Gower")(x',x).
$

The second term $S_y$ in @eq:baycon_obj measures the similarity in the output space, which we call validity. This property ensures that the counterfactual is predicted by the black-box model to be the requested target class. BayCon uses a hard scoring metric for this term, with 1 for correct labeling and 0 otherwise. The final term $S_f$ is the proportion of equal features between $x'$ and $x$ features; a term that guarantees sparsity for a found counterfactual. This term is calculated as

$
S_f (x',x)= ("# of equal feature values between" x' "and" x )/("# of features").
$

// Could write about prior and posterior probabilities related to bayes' theorem
To help with the counterfactual search, BayCon utilizes Bayesian Optimization. Bayesian Optimization is useful for efficiently optimizing an objective function where each evaluation is time-consuming, making methods like gradient descent impractical. It works by intelligently sampling points on which to evaluate the function, in order to obtain as much information about the function's structure as possible.

#figure(
  image("../assets/bayesian optimization.png"),
  caption: [Bayesian optimization example, where the acquisition function scores highly in unexplored promising areas of the objective function.]
)<fig:bayesian_optimization>

In @fig:bayesian_optimization, a regression problem is visualized where we want to find an approximate maximum of the true function. On this graph, the shaded areas represent the uncertainty of the model, as the objective function has not been evaluated in these intervals. Exploring new candidate data points in these intervals can then help better fit our regression to the task at hand. New candidate points are sampled based on an acquisition function, shown in the bottom of @fig:bayesian_optimization. Since this example is for a maximization task, the acquisition function helps find new candidate data points to evaluate in order to achieve a higher objective value. High values in the acquisition function can be seen as candidate points with a high probability of becoming a new global maximum. 

In BayCon, they utilize Bayesian Optimization in conjunction with a surrogate model trained on the data as a means to sample the most promising candidate counterfactuals. This idea enables a guided search in the candidate counterfactual space, by finding regions with promising counterfactuals. To estimate the objective function in BayCon they mention that this is typically done using a Gaussian Process (GP), but as GPs are quite computationally expensive they instead employ a Random Forest ensemble regression model. Such a model still allows one to get the mean and variance around the output of the surrogate model by simply getting the mean prediction for each tree in the forest, and calculating the variance of them. With these values, the acquisition function finds the highest expected improvement of the current best objective value for each candidate counterfactual. These improvements are then scaled by the variance, where a higher uncertainty is preferred for two candidates with the same mean.

To find a counterfactual for an instance, they apply an iterative algorithm. At each step, a set of candidate counterfactuals are generated. In order to obtain a higher sparsity, only a subset of features are selected for modification. To do this, they set the probability of changing $n$ features to be twice as large as changing $n+1$ features. After a set of features have been chosen, candidate counterfactuals are generated by randomly changing each feature value from this set, by sampling a normal distribution around the initial instance. 

After the initial counterfactuals have been generated and their objective scores have been calculated, they use this information to train the Random Forest surrogate model. This surrogate model is used in conjunction with the acquisition function to guide future counterfactual search by estimating the objective function. In each iteration after this, the algorithm explores the current best counterfactuals and looks for better candidates in their neighborhoods. To do this, it prunes counterfactuals with worse objective scores, leaving only the counterfactuals that score higher in the objective function. After no more improvements can be made or the maximum number of iterations have been run, the algorithm terminates and outputs all found counterfactuals with an objective score larger than some predefined threshold.

= Unsupervised counterfactual explanation #box[methods]<sec:unsupervised_counterfactual_methods>

In this section, we outline two existing methods capable of generating counterfactuals for clustering. To the best of our knowledge, these are the only methods that have been released in this specific field. The first method is a simple adaption of BayCon, which changes the objective function to handle clusterings instead of classification. The second method defines an objective function that can be optimized directly, without iterative methods.

== BayCon adaption <sec:baycon_soft>
In @spagnolCounterfactualExplanationsClustering2024 they suggest adapting BayCon @romashovBayConModelagnosticBayesian2022 to generate counterfactuals for the clustering task. This is done by modifying the calculation of the similarity in the output space ($S_y$), which is used in the objective function. They adapt it to an unsupervised setting based on the cluster structure, instead of being based on a black-box classification model. Additionally, they modify it to be a soft score between 0 and 1 instead of a binary score, in order to guide the search more effectively.

Concretely, they introduce 2 model-specific soft scoring methods, and 1 model-agnostic soft scoring method:
- The first model-specific scoring method is used for hierarchical density-based clustering algorithms like HDBSCAN. We will not further explain this method, as this thesis is primarily concerned with $k$-means clusterings.
- The second model-specific method works for centroid-based clusterings and is based on distances, making it more suitable for our purpose.
- The final method is model-agnostic, and is based on training a specific classifier to predict cluster membership probabilities, and then using these probabilities as soft scores.
We expand on the second and third method below, and will be implementing them for comparison in the experiments in @sec:experiments.

In the centroid-based soft scoring method, they employ a measure based on the distance between the candidate counterfactual and the target cluster center. To calculate the score for a given counterfactual candidate they use the formula

$
S_y (x', c_"target") = 1 - (d(x',c_"target") - min_"target")/(max_"target" - min_"target"),
$<eq:soft_scoring_kmeans>

where $x'$ is the candidate counterfactual, $c_"target"$ is the target cluster center, and $d$ is the Euclidian distance. $min_"target"$ and $max_"target"$ are the minimum and maximum distances from the target cluster center $c_"target"$ to any point in the target cluster. Values above $1$ or below $0$ are clipped to $1$ and $0$, respectively. Thus, @eq:soft_scoring_kmeans will give a score in the interval $[0,1]$, which indicates whether a candidate counterfactual is approaching the center of the target cluster.

To calculate a soft score for the model-agnostic method, they split the procedure into two parts. Initially, they extract representative points for each cluster using Prototypes and Criticisms @kimExamplesAreNot2016, which is a method similar to coresets that finds a subset of representatives for a cluster. These representatives are picked such that solving a problem on the representatives approximates a solution on the full dataset. Secondly, they train a classifier using the representatives in order to predict a membership probability for the target cluster. This probability is the soft score $S_y$.

Expanding on the first step, the algorithm finds a subset of the data called the prototypes. It then finds points poorly represented by these prototypes and generates an additional set called the criticisms. These two sets work in conjunction to represent the data. 

To find the prototypes and criticisms, they use the Maximum Mean Discrepancy (MMD) @grettonKernelTwoSampleTest2012, which measures the discrepancy between two different distributions. In this case, we aim to minimize the discrepancy between the actual dataset and the prototypes. The specific approach that they use in the paper is called MMD-critic, which uses a procedure to systematically find both prototypes and criticisms. In each step, the algorithm aims to minimize an MMD objective whilst ensuring that the chosen prototypes resemble the data distribution. Criticism selection functions as a second step after finding the prototypes, where criticisms are chosen as points not represented well by the prototypes. This is done by selecting points where the distribution of the prototypes and the distribution of the data deviates the most.

For the implementation in this thesis, we use a python library implementing MMD-critic @maxMaxidlMMDcriticGitHub2020, where we use a $1:4$ ratio between criticisms and prototypes as suggested in the paper. The total amount of prototypes and criticisms generated for each cluster is set to $20%$ of the size of that cluster. 

As the second step of the method, the representative points are given to a Self-Training Classifier#footnote[https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.SelfTrainingClassifier.html] (STC). This type of model allows a supervised classifier to function as a semi-supervised classifier, which uses a small portion of labeled data to ground its predictions or to gain domain background knowledge. This allows the model to perform well when predicting cluster labels for unseen data.

To use this model, a base estimator is needed for the STC, where they opted to use an Extra Trees Classifier#footnote[https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html], which is a tree-based ensemble model. This model is similar to a Random Forest, but with some desirable properties like faster training and lower sensitivity to hyperparameters @spagnolCounterfactualExplanationsClustering2024. Finally, when a point is predicted using the STC, it will output a vector of probabilities, designating the likelihood of belonging to each cluster. The probability of a candidate counterfactual belonging to the target cluster is then used as the soft score $S_y$ in the objective function. The remainder of the algorithm is exactly as in the original BayCon algorithm, outlined in @sec:baycon.

== CFClust<sec:cfclust>
The CFClust algorithm, published in @vardakasCounterfactualExplanationsKmeans2025, takes an alternative approach to generating counterfactuals for the clustering task, specifically $k$-means and Gaussian clustering, although we focus on their approach to $k$-means given the subject of this thesis. The method they present accounts for immutable features and tries to create plausible counterfactuals, meaning they should not be an outlier in the cluster they belong to.

The main property of the CFClust method is that it generates counterfactuals using a non-iterative method for $k$-means clusters using Euclidian distance. Therefore, the method is very efficient, but does not generate multiple counterfactuals for a single instance like the DiCE- or BayCon-based methods discussed previously. They introduce a general definition of counterfactuals for model-based clusterings that encapsulates both $k$-means and Gaussian clusterings by noting that $k$-means can be considered a special case of Gaussian clustering.

The general definition they present is based around each cluster $C_0,...,C_(k-1)$ having an associated probability density function $p_0,...,p_(k-1)$ as well as a prior probability $pi_0,...,pi_(k-1)$. With these, the cluster assignment rule states that a point $x$ is assigned to the cluster $l$ which maximizes $pi_l p_l (x)$. With these definitions, a Gaussian clustering simply means that $p_j (x) = N(x; mu_j, S_j)$ i.e. the density function is normally distributed with mean $mu_j$ and covariance matrix $S_j$. $k$-means clustering is simply a special case of this with $pi_0 = pi_1 = ... = pi_(k-1)$ and $S_i = I "for" i=0,...,k-1$.

With this model for clustering, they define a preference density $r(y|x)$ expressing the preference for $y$ to become a counterfactual for $x$. Importantly, this preference density should be an unimodal distribution. Using the preference density, their general definition of a counterfactual $x'$ for an instance $x in C_j$ is as follows:
$
x' = arg max_y r(y|x) \
"s.t." \
C_l eq.not C_j "where " l = arg max_(m = 0..k-1) pi_m p_m (x').
$

To generate counterfactuals they use the preference function 
$ r(y|x) = exp(-d(x,y)), $
where $d$ is the squared Euclidian distance, meaning the distance should be minimized in order to find a counterfactual according to the problem formulation above. To account for plausibility, they introduce a user-defined parameter $epsilon$, where $epsilon = 0$ means the counterfactual will be found directly on the boundary between the instance and target clusters. Together, these parameters and the clustering form the constraint set $C S_epsilon$, which is the set of points where the counterfactual can be located. For example, if $epsilon=0$, then the constraint set is simply the boundary. To account for immutable features, they introduce an indicator vector $M$ where $M_i = 1$ means feature $i$ is actionable, and $M_i=0$ means it is immutable. 

To solve the optimization problem of maximizing the preference function, they introduce an analytical solution for $k$-means, and a method that requires solving a non-linear equation system for the more general Gaussian clustering case. We will focus primarily on their solution to the $k$-means problem. 

For $k$-means, they define the constraint set as follows, given source and target centers $c_"source"$ and $c_"target"$:
$
C S_epsilon = {x': |x' - c_"source"|^2 = |x' - c_"target"|^2 + epsilon |c_"target" - c_"source"|^2} ,#h(5pt) epsilon >= 0.
$
The constraints can be written as an equation of the form
$
x'^T v = c,
$
where
$
v &= c_"source" - c_"target" \
c &= (|c_"source"|^2 - |c_"target"|^2 - epsilon|c_"target" - c_"source"|^2) / 2.
$
Finally, by splitting the vectors $x'$ and $v$ into sub-vectors comprised of their actionable and immutable features, respectively, one can write
$
x'_A^T v_A + x_I^T v_I = c
$
since $x'_I = x_I$. We are left with the following optimization problem to solve:
$
min_(x'_A) #h(5pt) |x'_A - x_A|^2 \
"s.t." \
x'_A^T v_A = c - x_I^T v_I,
$
which is a quadratic minimization with a linear constraint that can be solved analytically. In @fig:cfclust they present example counterfactuals found using this method. By varying the immutable features and the $epsilon$ parameter, one gets different counterfactuals, with $epsilon=1$ yielding a counterfactual near the target center, and $epsilon=0$ yielding one directly on the boundary. 

#figure(
  image("../assets/CFClust.png", width: 70%),
  caption: [
    Example counterfactuals generated by CFClust for a $k$-means clustering @vardakasCounterfactualExplanationsKmeans2025.
  ]
)<fig:cfclust>

As the authors of the method are still refining the implementation of these concepts, we have not been able to test the method in this thesis. However, in their own experiments they found promising results while comparing to a baseline of using DiCE @mothilalExplainingMachineLearning2020 and GuidedByPrototypes @vanlooverenInterpretableCounterfactualExplanations2021 on a logistic regression classifier with a decision hyperplane corresponding to the boundary between the source and target cluster. The CFClust method generates slightly better counterfactuals than the baselines, with the gap between them increasing on higher dimensional datasets. The main advantage of the CFClust method is the computation time, which they found to be below 0.001 seconds for both the Iris, Wine, and Pendigits datasets @iris_53 @wine_109 @Alpaydin1998penbased. 