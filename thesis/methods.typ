
= Supervised Counterfactual explanation methods
To create a baseline for our survey of current state-of-the-art counterfactual methods in the field of unsupervised learning, we will train a supervised surrogate model on the labels assigned by our black-box clustering method. This will enable us to use counterfactual methods for supervised models as a baseline reference.

== Guided By Prototypes
A state-of-the-art counterfactual method for supervised learning is presented in @vanlooverenInterpretableCounterfactualExplanations2021, in this paper they propose a fast model agnostic method for finding interpretable counterfactual explanations using class prototypes. In this paper the counterfactuals are generated to go from a specific class instance to any other class. 

They present two approaches for finding such prototypes, and suggest objectives that can be used in conjunction to speed up the counterfactual search process. The proposed method can either be run on a black box model, where we will only need access to a prediction model or be given the model architecture to take advantage of automatic differentiation when performing the counterfactual search process. They found that using the architecture for automatic differentiation introduced a drastic reduction in the computation of counterfactuals, as the method using a prediction model would have to evaluate each feature change using a prediction to approximate the gradient numerically.

They furthermore introduce four desirable properties for their counterfactuals, which align with our evaluation metrics, which are as follows: Similarity, Sparsity, Plausibility and Running time. The first method for finding class prototypes they present uses an autoencoder trained on the training set, and defines each class prototype to be equal to its class's mean encoding of the points corresponding to that class. To find the label for each datapoint in the training set we can simply query our black box model for predictions. The generated prototypes are defined in the latent space and the distance measures between our explained instance or counterfactual and our prototypes are therefore also measured in the latent space. 

The second method for finding class prototypes utilizes k-d trees@bentleyMultidimensionalBinarySearch1975. After labeling the training set by calling the predictive model, each class of label $i$ can be represented as a k-d tree built with instances with label $i$. A k-d tree is a special case of binary space partitioning trees used for storing k-dimensional data. k-d trees can be constructed in different ways, but a canonical way to construct them is to loop through all k-dimensions and in each step split the data on the median, the points will then be inserted into their subtree and recursively split until all points are stored in leaf nodes, this procedure will generate a balanced tree for storing the points. To find a prototype for an explained instance $x_0$, they run through each k-d tree with a different label than $x_0$, find the k-nearest item in the tree using Euclidean distance, and finally out of all classes they choose the closest of these items to be the class prototype. 

== Dice
In @mothilalExplainingMachineLearning2020 a counterfactual generation method is introduced that focuses on the diversity and feasibility metrics in order to produce actionable counterfactuals. The paper is concerned with a variety of issues that affect other counterfactual generation method, one being that features rarely exists in a vacuum. let's say we have 2 features, income and marriage status. While these features may seem unrelated they are in fact not, as being married often are related to age and in conjunction the amount of years the individual has been working. Another aspect is the existence of immutable features. A counterfactual might suggest changing either your race or gender which is not just impossible, but also hint at bias in the training data.

In order to fix these issues DICE aims to produce diverse and feasible counterfactuals. In order to generate a set of counterfactuals ${c_1, c_2, dots , c_k }$ for some machine learning model $f$ on some instance $x$, in order to do this DICE includes a diversity and feasibility metrics into their generation method. Some features might however might be mutable in some range, meaning that some feature cannot go under or over a limit set by the user. DICE allows for such constraint as well.  @mothilalExplainingMachineLearning2020

The diversity metric is used to generate diverse counterfactuals such that each counterfactual combination and size of feature changes are spread out. This allows the user to pick the counterfactual that suits them the most. DICE defines diversity the following way:
$ "dpp_diversity" = det(K) $
Where $K_(i,j) = 1 / (1 + d(c_i, c_j))$. Feasibility is evaluated through a proximity measure, since a counterfactual that is closer to the instance $x$ will be easier for the end-user to execute on. The proximity measure used to achieve feasibility is defined as the negative vector distance between the counterfactuals features and the responding features for the instance $x$:
$ "Proximity" = -1/k sum^k_(i=1) d(c_i,x) $
Sparsity is also a useful metric when evaluating actionable counterfactuals, since the amount of features that an user need to change heavily impact how easy it is to achieve. DICE does not include this metric in loss function but instead use a post-hoc method. In essence their method for enhancing sparsity is to greedily pick features in counterfactual which a different from the instance $x$ and change them back until no longer possible. @mothilalExplainingMachineLearning2020

With all this is mind DICE define their loss function as follows:
$ C(x) = arg min_(c_1,dots,c_k) 1/k sum^k_(i=1) "yloss"(f(c_i),y) + lambda_1/k sum^k_(i=1) d(c_i,x) - lambda_2 "dpp_diversity"(c_1, dots , c_k) $
Where the first term is the loss which minimizes the difference between the target label ($y$) and the actual counterfactual label $f(c_i)$. The second term is the feasibility metric and the last term is the diversity metric. In order to implement this objective function they use gradient descent, followed by the above-mentioned post-hoc step in order to enhance the sparsity of the produced counterfactuals. @mothilalExplainingMachineLearning2020

Lastly DICE uses a filtering step in order ensure all counterfactuals are feasible. This filtering step is necessary due to the issue of some features working in tandem with each other. This relation can be impossible to detect algorithmically hence why domain-specific knowledge is required. Before generation the counterfactuals a user can input pairs of features which a related to each other, in this way any counterfactual which alters one but not the other would be infeasible. These infeasible counterfactuals are then filtered(removed) such that only feasible counterfactual remains. @mothilalExplainingMachineLearning2020

== Baycon<sec:baycon>
In @romashovBayConModelagnosticBayesian2022 they introduce a model-agnostic method for generating counterfactuals in supervised tasks called Baycon, which is based on bayesian optimization. In this paper Baycon will serve a dual purpose, first we will talk about Baycon's adaptation to the clustering task as introduced in @spagnolCounterfactualExplanationsClustering2024, and we will secondly use it as another baseline reference by training a supervised surrogate model on our clustering task.

$
F(overline(c),overline(x)) = S_x * S_y * S_f
$<eq:baycon_obj>

Baycon makes use of an optimization objective constructed using three terms that implement desirable properties for a counterfactual. The objective function can be seen on @eq:baycon_obj, for a counterfactual candidate $overline(c)$ and an explained instance $overline(x)$. In @eq:baycon_obj, $S_x$ is the "similarity in feature space" calculated using Gower distance. Gower distance is a metric used for mixed feature spaces, as it supports computation for categorical attributes. The term is calculated as:

$
S_x(overline(c),overline(x)) = 1 - d_("Gower")
$

The second term $S_y$ ensures "similarity in the output space", which we call validity, a property that ensures that the counterfactual is predicted to be the requested class. Baycon uses a hard scoring metric for this term, with 1 for correct labeling and 0 otherwise. The final term $S_f$ called "Proportion of tweaked features" is a term that guarantees sparsity for a found counterfactual. The term is calculated as:

$
S_f(x',x^*)= ("# of equal features between" x' "and" x^* )/("Overall # of features")
$

// Could write about prior and posterior probabilities related to bayes' theorem
To help with the counterfactual search, Baycon utilizes Bayesian Optimization. Bayesian Optimization is a useful tool when sampling or experimenting is expensive and you therefore want to be selective with the samples that you try out. It could fx be that for every experiment or sample you would have to make some kind of computation or prediction. On @fig:bayesian_optimization a regression problem is visualized, where we want fit function to the observed points. On this graph the greyed out areas represent the uncertainty in the model, as we do not have any points in these intervals to fit our model to. 

#figure(
  image("assets/bayesian optimization.png"),
  caption: [Bayesian optimization example]
)<fig:bayesian_optimization>

Bayesian Optimization can be used to e.g. maximize or minimize a task, where the sampling of a new point can be based on what is called the acquisition function. For the example on @fig:bayesian_optimization, we are working with a maximization task, and the acquisition function, shows which point we should choose to achieve the highest probability of a better result. In Baycon they utilize Bayesian Optimization in conjunction with a surrogate model trained on the data as a means to sample the most promising candidate counterfactuals. This technique enables a guided search in the candidate counterfactual space, by finding regions with good counterfactuals. To estimate the objective function in Baycon they mention that this is typically done using a Gaussian Process(GP), but as this GPs are quite computationally expensive they instead employ an ensemble regression model called Random Forest.







= Unsupervised Counterfactual explanation methods <sec:unsupervised_counterfactual_methods>



== Baycon utilizing soft scoring
In @spagnolCounterfactualExplanationsClustering2024 they suggest adapting Baycon@romashovBayConModelagnosticBayesian2022 to generating counterfactuals for the clustering task, by adapting the objective function to utilizing soft scoring in its calculation of the "similarity in the output space"($S_y$), which is suitable for clustering models. 

In @spagnolCounterfactualExplanationsClustering2024 they introduce 2 model-specific soft scores: one for hierarchical density-based clustering like HDBSCAN, this method uses probabilities for cluster membership, which is provided by the method. We will omit this soft scoring method, as we will not be surveying density-based counterfactual methods. The second model-specific method is used for k-means, which is a centroid-based clustering method, suitable for the topic of this paper. They additionally introduce one model-agnostic soft scoring method to be used together with any clustering method.

In the centroid-based soft scoring they employ a measure based on the pairwise between candidate counterfactuals and cluster centroids or summaries. To calculate the score for a given counterfactual candidate they use the following formula:

$
S_y (C F(x^*), C_t) = 1 - (d(C F(x^*),C_t) - min_t)/(max_t - min_t)
$<eq:soft_scoring_kmeans>

where $C_t$ is the target cluster, that we want our counterfactual to be assigned to. $min_t$ and $max_t$ are the minimum and maximum distances from that target cluster to any point in the corresponding cluster. To evaluate @eq:soft_scoring_kmeans we will also need a distance metric, where for non-categorical datasets we chose to employ euclidean distance, but usually one could use the same metric as used in the black box model. For values above $1$ or below $0$, they clip them to $1$ and $0$ respectively. @eq:soft_scoring_kmeans will give a $S_y$ in the interval $[0,1]$, which indicates whether our candidate counterfactual is approaching the summary or centroid of the target cluster.

To calculate the soft scoring for the model-agnostic method, they split the procedure up into two parts. Initially they extract representative points for each cluster using a method called "prototypes and criticisms"@kimExamplesAreNot, which is a method that is similar to coresets. To find these prototypes and criticisms we will have to use Maximum Mean Discrepancy (MMD), which measures the discrepancy between two different distributions, in this case we aim to minimize the discrepancy between the actual dataset and our prototypes. The specific approach that they use in the paper is called MMD-critic, this algorithm uses a procedure to systematically find both prototypes and criticisms. In each step the algorithm aims to minimize an MMD objective whilst ensuring that the chosen prototypes resemble the data manifold. Criticism selection function as a second step, that picks criticisms to help scrutinize prototypes that diverge from other data instances. 

To use this approach, we made use of the following python library implementing MMD-critic@maxMaxidlMMDcriticGitHub2020, where we used a $1:4$ ratio between criticisms and prototypes as suggested in the paper, furthermore they fed $20%$ of the found representative points to self-training classifier used in the second part of the method.



