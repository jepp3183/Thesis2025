#import "../lib.typ": *

#pagebreak()
= Experiments<sec:experiments>
In this section, we conduct a range of experiments for each of the methods described throughout this thesis. We investigate performance based on a wide set of metrics defined in @sec:metrics, and compare the novel methods from @sec:novel_methods against both the existing methods for unsupervised counterfactuals presented in @vardakasCounterfactualExplanationsKmeans2025, and baseline generators based on the supervised methods BayCon @romashovBayConModelagnosticBayesian2022 and DiCE @mothilalExplainingMachineLearning2020. To find the optimal parameters and variation for each of the novel methods, we also present small-scale experiments in @sec:results comparing the variations against each other, and then continue with only the relevant variations for further experimentation.

== Experimental setup
We first present the parameters for the experiments, including datasets in @sec:datasets, the methodology in @sec:methodology, and the metrics we are measuring in @sec:metrics.

=== Datasets<sec:datasets>
#figure(
  placement: none,
  table(
    columns: 7,
    table.header([*Dataset*], [*Size*], [*\#Features*], [*\#Classes*], [*H*], [*C*], [*Source*]),
    "Synthetic", [250],[2], [3], [1.0], [1.0],[Scikit-learn#footnote[https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html]], 
    "Iris", [150], [4], [3], [0.53], [0.66],[@iris_53], 
    "Wine", [178], [13],  [3], [0.87], [0.88],[@wine_109],
    "Breast Cancer", [569], [30], [2], [0.55], [0.57], [@williamwolbergBreastCancerWisconsin1993],
  ),
  caption: [Summary of the different datasets on which the experiments are conducted. *H* is the homogeneity score for the clustering, and *C* is the completeness score.]
)<tab:datasets>

The experiments are conducted on multiple different datasets in order to investigate performance for a variation of dimensions and sizes. A summary of the datasets can be seen in @tab:datasets. For each dataset, we report both the number of clusters found using the $k$-means algorithm described in @sec:kmeans_clustering, as well as the homogeneity- and completeness-scores as defined in @rosenbergVMeasureConditionalEntropyBased2007 (abbreviated as *H* and *C* in @tab:datasets, respectively). The homogeneity score is higher if the clusters contain only data points from a single class in the dataset, whereas the completeness score is higher if all data points for a given class are members of the same cluster. Both are in the range $[0,1]$. For example, if a clustering contains $|X|$ clusters (i.e. one cluster for each data point) it would obtain a perfect homogeneity score, but a very poor completeness score. On the other hand, if the clustering contains only a single cluster, it would get a perfect completeness score, but a very poor homogeneity score.


=== Methodology<sec:methodology>
In order to evaluate each of the methods presented in this thesis, we test them on various different instance-target pairs. We pick 100 instances and targets in each dataset uniformly at random, and run each method on each of these pairs. Before running any test, the data is normalized to 0-mean and unit variance. This is done in order to ensure that features with naturally higher measurements do not influence metrics like _dissimilarity_ disproportionally. For each instance-target pair in each dataset, the metrics in @sec:metrics are evaluated. In order to compare how each method performs on the different datasets with varying dimensionality, we keep the results from each dataset separate.

To summarize and present the results, we calculate the arithmetic mean of each metric along with the standard deviation. These are calculated over each counterfactual found for each instance. For metrics not involving validity of the counterfactuals, we filter out the invalid counterfactuals before computing the results. If we did not filter the invalid counterfactuals, many metrics would be trivially easy to maximize. For example, dissimilarity would be easily minimized by counterfactuals which lie exactly at the instance.

In order to obtain a baseline level of performance to compare each method against, we adapt well-known supervised counterfactual generation methods to an unsupervised setting. This is done by training a classifier on the cluster labels, and using the supervised method to generate counterfactuals using this classifier as a black-box model. We do this using 2 different supervised methods, where the underlying classifier trained on the clustering is a Random Forest Classifier from Scikit-learn#footnote[https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html] trained with the default parameters, using 100 estimators. The first of the baseline methods uses BayCon @romashovBayConModelagnosticBayesian2022 (described in @sec:baycon) in order to generate the counterfactuals, while the second uses DiCE @mothilalExplainingMachineLearning2020 (described in @sec:dice).

=== Metrics<sec:metrics>
To evaluate the performance of each algorithm, we formalize some desirable properties for counterfactuals. These properties include the metrics from @sec:novel_methods, which had significant influence concerning the creation of the novel methods introduced in this thesis. The _invalidation_ and _correction_ metrics are used to measure the concepts introduced in @sec:kmeans_cluster_change.

- *Dissimilarity*. A counterfactual $x'$ should be similar to the instance $x$, i.e. given a distance function $d$, we want to minimize $d(x,x')$. We are using the Euclidian distance directly as the _dissimilarity_ metric, meaning a lower score is better for this metric. Other possible similarity metrics exist, but since we are using $k$-means clustering with only continuous features, Euclidian distance is the most appropriate metric.
- *Sparsity*. Keeping with the idea that the difference between $x$ and $x'$ should be small, the _sparsity_ of a counterfactual $x'$ is the fraction of unchanged features from $x$ to $x'$. Recall that sparsity is a highly-desired quality of a counterfactual explanation, since a high sparsity makes the changes from $x$ to $x'$ much easier to grasp for users and developers.
- *Plausibility*. Given a reference population $A$, a data point is plausible subject to $A$ if its characteristics are similar to points in $A$. What this means for our task is that $x'$ should be an inlier in the target cluster $C_"target"$. To quantify this concept, we use the Local Outlier Factor (LOF) @breunigLOFIdentifyingDensitybased2000, which is a measure of a point's outlierness compared to its nearest neighbors in $C_"target"$. LOF is measured by comparing the local reachability density of $x'$ to its $k$-nearest neighbors' densities. For a data point to be an inlier it should be located in a comparably dense region as other inliers in this data population. Conversely, if a point's local reachability density is lower than its $k$-nearest neighbors, it is seen as having higher outlierness. A LOF value of $1$ indicates that a point has a similar density to its neighbors, while a value lower than $1$ indicates higher density than its neighbors. An LOF value higher than 1 indicates that it has lower density than its neighbors and is more of an outlier. We define plausibility as being the negative LOF, which means that a higher value will signify a more plausible, or inlying, point. 
- *Validity*. A counterfactual $x'$ for an instance $x$ is valid if it would get assigned to the target cluster. This is a binary measure for each counterfactual. To obtain a score, we report the number of valid counterfactuals generated as a proportion of the total amount for each method.
- *Diversity*. A set of counterfactuals $x'_1,x'_2,dots,x'_n$ for an instance $x$ are diverse if they explore different approaches for counterfactual change e.g. by being more spread out in the target cluster. To calculate this metric, we follow the approach of the DiCE paper @mothilalExplainingMachineLearning2020 based on average pairwise distances. Thus, diversity is defined as
$
  1/n^2 sum^(n)_(i=1)sum^(n)_(j=1) d(x'_i,x'_j).
$
- *Invalidation*. In @sec:kmeans_cluster_change, we discuss how moving an instance according to some counterfactual can potentially change the clustering, such that a valid counterfactual becomes invalid. This is measured by taking all the valid counterfactual individually, moving the instance according to the counterfactual and then recalculating the centers according to Lloyd's algorithm in @lloydLeastSquaresQuantization1982. The fraction of valid counterfactuals which become invalid because of this recalculation is called the _invalidation_.
- *Correction*. _Correction_ is concerned with the same core issue as _invalidation_, but measures the opposite phenomenon. For _correction_, we measure how many invalid counterfactuals become valid after the cluster change. The fraction of invalid counterfactuals which become valid after the cluster change is the _correction_.
- *Runtime*. _Runtime_ measures the amount of time it takes to generate a counterfactual or a set of counterfactuals for an instance $x$. This is measured as the time it takes to run the algorithm one time, where we are not taking into account how many counterfactuals are actually generated for a single instance. The _runtime_ is given in seconds.
- *%Explained*. To evaluate how often the algorithm succeeds in generating at least one valid counterfactual for a given instance $x$. We measure the _%Explained_ metric as the proportion of instances for which the algorithm succeeds in generating at least one valid counterfactual, divided by the total amount of instances that we run the algorithm for, which is 100.
- *\#Valid*. Some counterfactual methods generate multiple counterfactuals for a given instance $x$. This metric measures how many of the generated counterfactuals for a single instance are valid when evaluating cluster membership.

== Results<sec:results>
First, we investigate the methods outlined in @sec:novel_methods in order to find the optimal parameters and the best variation of each algorithm. Finally, we compare these against each other together with the baseline methods and the BayCon method with soft scores outlined in @sec:baycon_soft.

=== GradCF <sec:gradcf_comparison>
The GradCF method outlined in @sec:gradcf is a relatively simple method with few parameters one can tweak. Recall that the method is based around an objective function which is optimized using ADAM. The objective function is defined as:
$
f(x') = G(x, x') dot S(d(x', c_"target")),
$
where the first term is the Gower's similarity, which ensures similarity between the instance $x$ and the counterfactual $x'$. The second term is based loosely on a sigmoid function, which is close to 1 when inside the target cluster. The goal of this term is to guide the search towards the target cluster.

In this section, we investigate the effect of modifying the $epsilon$ parameter, which controls how close the generated counterfactual will be to the target cluster center $c_"target"$. We also investigate how well the sparsity improvement method works for the generated counterfactual. Recall that this method greedily resets features of $x'$ back to $x$ as presented in the DiCE paper @mothilalExplainingMachineLearning2020. We set $lambda_2=50$ to get a steep sigmoid function as discussed in @sec:gradcf. 

#let epss = (0.0,0.1,0.25,0.5,0.75,1)
#let gradcf_data = csv("../assets/gradcf_data.csv", row-type: array)
#let gradcf_data_sparsity = csv("../assets/gradcf_data_sparsity.csv")

#figure(
  block(
    metric-table(
      csv("../assets/breast_cancer_complete.csv", row-type: dictionary), 
      ("GradCF (eps=0)","GradCF (eps=0.1)","GradCF (eps=0.25)","GradCF (eps=0.5)","GradCF (eps=0.75)","GradCF (eps=1)"), 
      ("Dissimilarity", "Sparsity", "Plausibility", "Validity", "Runtime", "%Explained",),
      header: (table.cell([*GradCF* (Without sparsity improvement)], colspan: 7), "",) + epss.map(e => [$epsilon=#e$]),
      include_std: false
    )
  ),
  caption: [Counterfactual metrics for varying values of $epsilon$. The standard deviation has been omitted for brevity, but is nearly identical for all metrics.]
)<fig:gradcf_data>

The results for varying values of $epsilon$ can be seen in @fig:gradcf_data. The results are obtained for the Breast Cancer dataset by following the methodology outlined in @sec:methodology, and we report the metrics as defined in @sec:metrics. Firstly, the method consistently finds a counterfactual for all 100 randomly chosen instances in the Breast Cancer dataset, and the runtime is relatively low at 0.6 seconds on average. Looking at both the _dissimilarity_ and _plausibility_ of the counterfactuals, one can clearly see the effect $epsilon$ has. As $epsilon$ gets closer to 1, the _dissimilarity_, measured as the distance between $x$ and $x'$, grows. As this happens, the _plausibility_ also increases from $-1.01$ to $-0.98$, indicating that the counterfactuals are changing from being slight outliers to being inliers as they get closer to $c_"target"$. 

#figure(
  block(
    metric-table(
      csv("../assets/breast_cancer_complete.csv", row-type: dictionary), 
      ("GradCF (eps=0) (sparsity fixed)", "GradCF (eps=0.1) (sparsity fixed)", "GradCF (eps=0.25) (sparsity fixed)", "GradCF (eps=0.5) (sparsity fixed)", "GradCF (eps=0.75) (sparsity fixed)", "GradCF (eps=1) (sparsity fixed)"), 
      ("Dissimilarity", "Sparsity", "Plausibility", "Validity", "Runtime", "%Explained",),
      header: (table.cell([*GradCF* (With sparsity improvement)], colspan: 7), "",) + epss.map(e => [$epsilon=#e$]),
      include_std: false
    )
  ),
  caption: [Counterfactual metrics for the sparsity improvement method applied to all counterfactuals from @fig:gradcf_data.]
)<fig:gradcf_data_sparsity>

However, as noted in @sec:gradcf, the _sparsity_ of the counterfactuals are all 0 due to the gradient descent routine used in the algorithm, which changes all features when moving in the best direction for optimizing the objective function. @fig:gradcf_data_sparsity shows the metrics for the same counterfactuals as @fig:gradcf_data, but with the sparsity improvement method from the DiCE paper applied to them @mothilalExplainingMachineLearning2020. Here, we mostly observe the same characteristics, except for a few metrics. Most notably, the _sparsity_ of the counterfactuals has improved significantly, although the effect decreases with higher $epsilon$, which is likely due to the fact that the improvement routine only resets feature values to the original value if they do not deviate significantly from $x$. Secondly, we also note a slight decrease in _dissimilarity_, which is expected since resetting features of $x'$ back to the values from $x$ can only decrease the Euclidian distance $d(x, x')$. The only performance decrease as an effect of the sparsity improvement is a slightly worse _plausibility_ score, although the degradation is not significant.

As we conduct further experiments in the following sections, we set $epsilon=0$ when running the GradCF algorithm. This means that the algorithm will find a counterfactual closer to the boundary of the target cluster, but still relatively plausible. We also enable the sparsity improvement method, as the increase in _sparsity_ is significant, incurring only a modest decrease in the _plausibility_ score.

=== Heuristic methods <sec:cfae_comparison>
In the following section, we investigate the performance differences between the CFDE and CFAE methods as introduced in @cfae. Recall that CFAE was developed in order to improve some of the deficiencies from CFDE. Therefore, we investigate whether CFAE shows improved performance compared to CFDE. Finally, we also conduct a small experiment regarding the parameter $k$ for the NeCS algorithm as defined in @Necs. This parameter decides the amount of nearest neighbors used for counterfactual generation.

Both CFAE and CFDE are run with step size $lambda_1 = 0.05$, and misses before termination is set to 20 with the maximum amount of iterations set to 1000. CFAE's penalty is set to $lambda_2 = 1.001$. The parameters are selected after manual testing, and are static in all following experiments. Fine-tuning these parameters can increase the performance, but the selected parameters have demonstrated robust performance across all of the datasets. For CFAE and CFDE, we produce 10 counterfactuals for each instance. 

#figure(
  image("../assets/cfae_vs_cfde/cfae_vs_cfde_image.png"),
  caption: [Counterfactuals generated using CFAE and CFDE on the Wine dataset, and visualized using PCA @pearsonLIIILinesPlanes1901.]
)<fig:cfae_vs_cfde>

Recall that the purpose of CFAE and CFDE is to generate highly similar counterfactuals using random sampling. In @fig:cfae_vs_cfde, the counterfactuals generated for an instance is visualized. It is clear that all counterfactuals generated by CFDE are concentrated in approximately the same region. Contrary to this, CFAE produces counterfactuals spanning the cluster boundary. Looking at @tab:cfae_vs_cfde_full, this is also clearly evident when running the algorithms on the datasets. We observe a very low _diversity_ score for CFDE, while CFAE obtains a much higher score.

We believe the poor _diversity_ score for CFDE is caused by the greedy sparsity picks, where CFDE picks the starting point by resetting as many features as possible between $c_"target"$ and $x$. Since these picks are deterministic, the starting point will be the same for all 10 executions of the algorithm. Clearly, the random sampling when moving towards $x$ is not enough to ensure diversity, which is why all counterfactuals end up in the same region. CFAE does not use the same method to pick a starting point, since it always starts at $x$, and moves towards a random target in $C_"target"$, which enables the counterfactuals to approach the cluster boundary in different directions. 

CFAE vastly outperforms CFDE on _plausibility_ on all datasets as seen in @tab:cfae_vs_cfde_full, which we speculate is due to CFDE creating points which are similar to outliers due to its lack of a stopping condition other than a max iteration count or maximum number of misses. Since CFAE starts in the source cluster, it is able to stop once it has reached the target cluster, thereby ensuring plausibility.

Looking at _sparsity_ in @tab:cfae_vs_cfde_full, CFAE's and CFDE's performances are similar, with CFAE outperforming CFDE in higher dimensions. CFAE achieves better _sparsity_ on the Breast Cancer dataset, almost equal performance on Wine, and worse on Iris. CFAE also performs worse w.r.t. _dissimilarity_ on the Breast Cancer dataset than CFDE, but scores better on the remaining datasets. 

#figure(
  metric-table(
    csv("../assets/cfae_vs_cfde/cfae_full2.csv", row-type: dictionary),
    ("cfde1", "cfae1","cfde2","cfae2","cfde3","cfae3"),
    ("Dissimilarity","Sparsity", "Plausibility", "Validity", "Diversity"),
    header: (
      table.cell("", colspan: 1, rowspan:2),table.cell([*Iris*], colspan:2),table.cell([*Wine*], colspan:2),table.cell([*Breast Cancer*], colspan:2),
      [*CFDE*],[*CFAE*],[*CFDE*],[*CFAE*],[*CFDE*],[*CFAE*]
    )
  ),
  caption: [CFAE compared to CFDE across the Iris @iris_53, Wine @wine_109 and Breast Cancer @williamwolbergBreastCancerWisconsin1993 datasets.],
  // placement: none
)<tab:cfae_vs_cfde_full>

Based on these observations, we find CFAE to be superior to CFDE, and therefore exclude CFDE from further experiments.

#figure(
  metric-table(
    csv("../assets/new_data/necs_amount.csv", row-type: dictionary),
    ("necs5","necs10","necs20","necs50"),
    ("Dissimilarity","Sparsity", "Plausibility", "Diversity", "Runtime", "#Valid"),
    header: (
      table.cell([*Breast Cancer (NeCS)*], colspan: 5),
      table.cell("", rowspan: 1), 
      [*$k=5$*],[*$k=10$*],[*$k=20$*],[*$k=50$*]
    ),
    include_std: false
  ),
  caption: [Counterfactual metrics for varying values of $k$ for NeCS. The standard deviation has been omitted for brevity, but is nearly identical for all metrics.]
) <tab:necs_amount>


To investigate the final heuristic method introduced in this thesis, we conduct a small experiment on NeCS by varying its parameter $k$. The amount of counterfactuals generated by NeCS can significantly impact the results as can be seen in @tab:necs_amount. _Dissimilarity_ increases as the neighbors chosen will be further from $x$ as $k$ increases. The _sparsity_ increases as feature replacements become more impactful on neighbors further from $x$. _Diversity_ significantly increases with more counterfactuals, as the neighbors picked cover more of the target cluster. 

We speculate that these effects are caused by NeCS generating a counterfactual for each $k$-nearest neighbor. Thus, with increasing values of $k$, each added neighbor will lie further from the instance $x$. Since NeCS uses only complete feature replacements, each of these will have an increasing impact on the 3 metrics above. For example, _sparsity_ likely increases due to the feature replacements moving the counterfactual across the cluster boundary in fewer iterations. 

We set $k=10$ for easier comparison with CFAE in future experiments and to avoid increased _dissimilarity_. 

=== TGCF<sec:tree_comparison>
In @sec:tgcf we introduced Tree-Guided Counterfactuals (TGCF), a novel tree-based counterfactual method. TGCF enables a developer to pick any decision tree algorithm for creating the tree, and in this thesis we are using the Decision Tree Classifier (DTC)#footnote[https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html] from Scikit-learn and the Iterative Mistake Minimization (IMM) algorithm @dasguptaExplainableKMeansKMedians2020 as surrogate models. In this section, we compare the performance of TGCF when using these two algorithms.

Additionally, we introduced plausibility improvement, a post-processing step for improving validity and plausibility by moving the counterfactual further inside the target cluster. As a further addition to the algorithm, we introduced a fidelity improvement step for increasing the fidelity of generated decision trees by encapsulating the clusters more tightly. We also compare the effects of these improvements based on the experiments.

#figure(
  metric-table(
    csv("../assets/imm_synthetic_and_breast.csv", row-type: dictionary),
    ("TGCF IMM","TGCF IMM'","2TGCF IMM","2TGCF IMM'"),
    ("Dissimilarity","Sparsity", "Plausibility", "Validity", "Runtime"),
    header: (
      table.cell("", rowspan:2), table.cell([*Synthetic*], colspan:2), table.cell([*Breast Cancer*], colspan: 2),
      [*TGCF IMM*],[*TGCF IMM'*],[*TGCF IMM*],[*TGCF IMM'*]
    )
  ),
  caption: [TGCF IMM results for both datasets. _Diversity_ metrics are omitted, since IMM only produces a single counterfactual. (\') signifies usage of plausibility improvement.]
) <tab:tgcf_imm>

We run the tests using both the 2D synthetic dataset with 3 clusters and the Breast Cancer dataset with 2 clusters and 30 features. For the fidelity improvement, we set the hyperparameters $phi=0.95$ and $tau=0.67$, which corresponds to recursively splitting leaves if $95%$ of the data in that leaf, makes up less than two thirds of a feature span represented by this leaf. We also set $lambda=0.5$. The results can be seen in @tab:tgcf_imm and @tab:tgcf_synthetic_vs_breast. We chose these datasets for brevity as we find the same results replicated in the other datasets. In the results, we denote the algorithms with the counterfactual plausibility improvement step using the (\') symbol, and the ones using fidelity improvement are marked using 'Fidelity'.


It is clear from @tab:tgcf_imm and @tab:tgcf_synthetic_vs_breast that the methods using a Decision Tree Classifier (DTC) and the fidelity improvement step greatly outperform the other configurations w.r.t. the _validity_ metric. This performance improvement over the other configurations is even clearer in higher dimensions as can be seen in @tab:tgcf_synthetic_vs_breast. The same observation can be made with the _%Explained_ metric, where TGCF using DTC and fidelity improvement outperforms the other methods.

#figure(
  grid(rows: 2,row-gutter: 15pt,
  metric-table(
    csv("../assets/full_tables/blobs_all.csv", row-type: dictionary), 
    ("TGCF_DTC","TGCF_DTC'","TGCF_DTC_Fidelity","TGCF_DTC_Fidelity'"),
    ("Dissimilarity","Sparsity", "Plausibility", "Validity", "Diversity", "Runtime", "%Explained", "#Valid"),
    header: (table.cell([Synthetic Dataset], colspan: 5),"", [*TGCF DTC*],[*TGCF DTC'*],[*TGCF DTC Fidelity*],[*TGCF DTC Fidelity'*], )
  ),
  metric-table(
    csv("../assets/full_tables/breast_cancer_all.csv", row-type: dictionary), 
    ("TGCF_DTC","TGCF_DTC'","TGCF_DTC_Fidelity","TGCF_DTC_Fidelity'"),
    ("Dissimilarity","Sparsity", "Plausibility", "Validity", "Diversity", "Runtime", "%Explained", "#Valid"),
    header: (table.cell([Breast Cancer Dataset], colspan: 5),"", [*TGCF DTC*],[*TGCF DTC'*],[*TGCF DTC Fidelity*],[*TGCF DTC Fidelity'*])
  ),
  ),
  caption: [Summary of results for TGCF using DTC run on the synthetic and Breast Cancer datasets. In these tables "Fidelity" means that we are applying the fidelity improvement on the DTC surrogate model, and (\') signifies usage of plausibility improvement.]
)<tab:tgcf_synthetic_vs_breast>

It is also important to note that the overall performance of TGCF tends to decrease in higher dimensional data. This happens as the depth of the surrogate tree is correlated to the dimensionality of the data for DTC, in turn inducing further counterfactual changes. This fact also explains why IMM doesn't score well for more complex datasets, as it can make at most $k-1$ thresholds for a dataset with k clusters as explained in @sec:tgcf. An observation that stands out in @tab:tgcf_imm is that IMM\' on the synthetic dataset achieves a very high _validity_ score. We attribute this to the synthetic dataset only having 2 dimensions with 3 clusters. This makes it easier for the IMM algorithm to create a surrogate model that is true to the clustering. Combining this fact with the plausibility improvement step enables it to score well for this specific dataset.

// From @tab:tgcf_synthetic_vs_breast it might seem that the configurations without fidelity improvement outperform those with fidelity improvement on the _dissimilarity_ metric. We attribute this result to the fact that they achieve a very low _validity_, and are therefore most likely only able to explain instances that are already very close to the decision boundary. An instance close to the decision boundary to the target cluster would only have to make a minimal change to change cluster membership. 


// Another positive observation from the two TGCF DTC Fidelity methods is that they generate a higher number of valid counterfactuals for a given instance. This is especially clear in higher dimensions, where many new leaves could be generated to improve the fidelity. As a consequence of the new leaves, the amount of counterfactuals also increases. This can cause many of the newly generated counterfactuals to be redundant, as leaves may tend to be somewhat similar as a consequence of the fidelity and the plausibility improvements. 

Between the two TGCF DTC Fidelity methods we also observe a _dissimilarity_/_validity_ trade-off, which is likely due to the fact that the plausibility improvement step moves the counterfactuals closer to the target center, increasing $d(x, x')$ while also correcting some invalid counterfactuals. For lower dimensions like the synthetic dataset in @tab:tgcf_synthetic_vs_breast, this step also improves the _plausibility_, although the effect is ambiguous in higher dimensions. 

A final observation to make from these results is that all of the proposed configurations apart from IMM perform poorly w.r.t. the _sparsity_ metric. We attribute this result to the fact that the Decision Tree Classifier does not take sparsity into consideration when creating the tree. Instead, it prioritizes accuracy, causing a deep tree which splits on most features. For this metric, the IMM configuration scores much better, given the fact that it can make at most $k-1$ thresholds on the dataset.

Having compared the two surrogate tree models used for TGCF and the effect of the plausibility and fidelity improvements, we finally investigate the effect of modifying the $lambda$ hyperparameter. We performed the experiments using the TGCF DTC Fidelity' configuration for varying values of $lambda$, which can be seen in @tab:tgcf_lambda. We observe a clear increase in _dissimilarity_ for increasing values of $lambda$, as the counterfactual will be moved further towards the target cluster center. The same observation holds for _validity_, since a counterfactual closer to the target cluster center more likely belongs to the target cluster. For all other metrics, we found minimal variation in the results.


#figure(
  metric-table(
    csv("../assets/tgcf_epsi.csv", row-type: dictionary),
    ("tgcf0","tgcf25","tgcf50","tgcf75", "tgcf100"),
    ("Dissimilarity","Sparsity", "Plausibility", "Validity", "Diversity", "Runtime", "%Explained", "#Valid"),
    header: (
      table.cell([*Breast Cancer (TGCF DTC Fidelity')*], colspan: 6),
      table.cell("", rowspan: 1), 
      [*$lambda=0.00$*],[*$lambda=0.25$*],[*$lambda=0.50$*],[*$lambda=0.75$*],[*$lambda=1.00$*]
    ),
    include_std: false
  ),
  caption: [Counterfactual metrics for varying values of $lambda$ for TGCF DTC Fidelity'. The standard deviation has been omitted for brevity, but is nearly identical for all metrics.]
) <tab:tgcf_lambda>

Based on these results, we find that TGCF using DTC as surrogate model with both the fidelity and plausibility improvements (TGCF DTC Fidelity') is the best performing variation. For the fidelity improvement, we set $phi=0.95$ and $tau=0.67$. We also set $lambda=0.5$ for the plausibility improvement, as we find it to have the most desirable trade-off between _dissimilarity_ and _validity_. For further experiments, we will be using this configuration of TGCF, as it scores highly on both the _validity_ and _plausibility_ metrics, with the only drawback being the _dissimilarity_ metric.

=== Invalidation and correction<sec:invalidation>
Throughout the experiments, very few methods ever generated a counterfactual which became invalidated or corrected due to cluster changes as described in @sec:kmeans_cluster_change. 

//The only concrete examples of invalidations are in the Breast cancer dataset where CFAE scored $0.00 plus.minus 0.05$ and NeCS scored $0.00 plus.minus 0.03$, and in the Iris dataset where CFAE got an Invalidation of $0.01 plus.minus 0.07$. We never saw any counterfactual be corrected due to cluster changes in the experiments#todo("Why?"). 

For the novel methods, the only occurrences of invalidations are in the Breast Cancer and Iris datasets. On the Breast Cancer dataset, CFAE had 5 invalidations while NeCS had 1. On the Iris dataset, CFAE had 6 invalidations. However, noting that these methods both produced a total of approximately 1000 valid counterfactuals on each dataset, the number of invalidations remains relatively low. For the baseline methods, the results are similar, with only the BayCon variations experiencing any invalidations. BayCon (Model-Agnostic) only led to a single invalidation, while BayCon ($k$-means) experienced 3, both on the Iris dataset. Again, since both methods each produced approximately 8000 valid counterfactuals, this number of invalidations is relatively low.

We believe the reason only CFAE and NeCS experienced any invalidations is because they tend to generate counterfactuals located directly on the cluster boundary, or very close to it. For example, GradCF is very resistant against invalidation and correction, since the objective function prevents it from generating counterfactuals too close to the boundary. 

Additionally, we speculate that the reason invalidations are so rare, is because a very specific set of conditions are needed (see @sec:kmeans_cluster_change). Not only does the instance need to be in a position to make $c_"source"$ move towards the counterfactual, but it must also more further than $c_"target"$ moves towards $x'$. Even in the case where such conditions are met, the cluster boundary change needs to be significant enough to warrant cluster reassignments, which is also unlikely.

The only method which experienced any corrections of an invalid counterfactual was the BayCon ($k$-means) algorithm on the Wine dataset, where 2 counterfactuals were corrected. Similarly to invalidations, we attribute the rarity of this phenomenon to a very specific set of conditions being needed.

// Logically, whenever a counterfactual is moved, but does not cross the cluster boundary of $C_"target"$, the target center should not change as no points have been added to it. The cluster center that is affected, is whichever cluster the invalid counterfactual were assigned to. This causes that cluster center to change. We believe that corrections can only happen if the center movement caused by the invalid counterfactual, then causes further point reassignments, starting a chain reaction of cluster movements which in some cases could potentially make the counterfactual valid. It is clear that such a scenario is rare, which evident from our results.

Therefore, since invalidations and corrections are so rare, we omit the _invalidation_ and _correction_ metrics from the results in other sections, as their impact on the evaluation are negligible.

=== Comparison of novel methods
In this section, we compare the novel methods presented in this thesis. We use the parameters and configurations chosen in @sec:results, with the goal of outlining some of the strengths and weaknesses of the proposed methods.

#figure(
  image("../assets/violin_breast_sparsity.png"),
  caption: [Violin plot of the _sparsity_ distributions of the novel methods for the Breast Cancer dataset. (\*) means that DiCE sparsity post-processing is applied.]
)<fig:violin_breast_sparsity>

#figure(
  table(
    columns: 5,
    ..("", [*Synthetic*],[*Iris*],[*Wine*],[*Breast Cancer*]),
    ..csv("../assets/table_tgcf_sparsity.csv").flatten()
  ),
  caption: [_Sparsity_ values of TGCF before and after using post-processing sparsity improvement method from DiCE @mothilalExplainingMachineLearning2020. (\*) means that sparsity post-processing is applied.]
)<tab:tgcf_sparsity_improvement>

In @fig:violin_breast_sparsity we present a violin plot of the _sparsity_ distribution of the novel methods for the Breast Cancer dataset. Based on this illustration, GradCF, CFAE, and NeCS perform similarly in the _sparsity_ metric, although CFAE performs noticeably better. The biggest outlier is TGCF, which attains a _sparsity_ score of 0. 

As detailed in @sec:tgcf, the goal of using decision trees as surrogate models for the clustering was to improve sparsity, based on the idea that such trees present simple if-else rules for counterfactual change. However, as can be seen in @fig:violin_breast_sparsity, TGCF performs very poorly on the _sparsity_ metric. This is likely due to the underlying DTC model heavily prioritizing accuracy above tree simplicity, resulting in a tree which splits on all features. Furthermore, the fidelity improvement step also further splits the leaves, potentially worsening the problem.

To improve the low _sparsity_, we experiment further by applying the post-processing method from DiCE @mothilalExplainingMachineLearning2020 which greedily resets changed features in $x'$ back to values from $x$. The same post-processing method is also used in the GradCF algorithm presented in @sec:gradcf. The results can be seen in @tab:tgcf_sparsity_improvement and @fig:violin_breast_sparsity, and a table containing all metrics can be seen in @app:full_tgcf_sparsity. With the application of this post-processing step, the _sparsity_ metric is greatly improved and is now on par with the remaining methods. The impact on the other metrics for TGCF is negligible for the Breast Cancer dataset. The only significant downside is a notable increase in the _runtime_ due to the extra processing step. The increase in mean _runtime_ went from around 0.05 seconds to 0.45 seconds, but is still much faster than the baselines and BayCon variations.

#figure(
  image("../assets/plausibility_dataset_comparison.png"),
  caption: [Boxplots of _plausibility_ distributions of our novel methods for all dataset.]
)<fig:boxplot_breast_plausibility>


In @fig:boxplot_breast_plausibility we visualize the _plausibility_ of counterfactuals created by the novel methods on the different datasets. It is clear that GradCF outperforms the other novel methods on this metric in all datasets with more dimensions than 2. Generally, the _plausibility_ of GradCF is densely distributed around -1, which, as explained in @sec:metrics, means that the generated counterfactuals are inliers in the target cluster. Furthermore, we observe that CFAE generally scores worst on this metric with one of the largest deviations in its _plausibility_, a trend that we observe on all datasets apart from Wine, where it scores more comparable to the other methods.

We presume that the result on the synthetic dataset deviates because of the perfect homogeneity- and completeness-scores of the synthetic dataset as presented in @tab:datasets. These scores signify well defined and clear clusters, making it easier for most methods to generate counterfactuals. For example, the performance of TGCF should increase, as it becomes easier to threshold and separate clusters when they are clearly defined.

#figure(
  image("../assets/similarity_dataset_comparrison.png"),
  caption: [Boxplots for _dissimilarity_ distribution on all datasets comparing the novel methods. Generally, the _dissimilarity_ will increase in higher dimensions, as further features add distance between the points.]
)<fig:sim_datasets>

In @fig:sim_datasets we compare the novel methods based on their _dissimilarity_ scores. For the synthetic dataset, CFAE yields the most similar counterfactuals, followed by GradCF, NeCS and then TGCF. This pattern persists on the Iris and Wine datasets. Finally, in the Breast Cancer dataset, the data follows the original pattern except CFAE and TGCF have slightly worse _dissimilarity_ scores, being outperformed by GradCF and NeCS. Although the methods' _dissimilarity_ scores are very close, it is still clear that their performance in relation to each other stays roughly the same in higher dimensions. Finally, we note that the number of outliers in the Breast Cancer dataset is significantly increased compared to the remaining datasets. This happens as the counterfactual explanation complexity increases significantly with increasing dimensionality, causing much more varied performance compared to the low-dimensional datasets.


=== Visualization of counterfactuals
The 4 novel methods contributed in this thesis produce counterfactuals using vastly different approaches. In order to show the differences without referencing performance, we visualize the counterfactuals generated for a single instance in the Wine dataset @wine_109, as can be seen in @fig:cf_vias_comp. The dataset has been embedded in two dimensions using PCA @pearsonLIIILinesPlanes1901, although the different characteristics of each method are still apparent.

#figure(
  image("../assets/giant cf plot.png"),
  caption: [Counterfactuals generated for the Wine dataset, visualized using PCA @pearsonLIIILinesPlanes1901.]
)<fig:cf_vias_comp>

The GradCF method creates a single plausible counterfactual. This can be seen as the counterfactual does not lie directly on the cluster boundary, but is instead placed slightly further into the cluster. However, since GradCF optimizes a single highly specialized objective function, it can only ever generate a single counterfactual for a given set of parameters. In a situation where a user is not satisfied with this single counterfactual, they have no practical recourse for obtaining alternate ones. Only generating a single counterfactual can be seen as a major drawback as one might prefer being presented with multiple diverse counterfactuals to suggest different approaches for counterfactual changes. 

CFAE creates a number of counterfactuals close to the cluster boundary, spread out in order to achieve higher diversity. Although they are more diverse than CFDE counterfactuals as covered in @sec:cfae_comparison, they are still grouped relatively close together. NeCS produces counterfactuals which are relatively spread out, which is caused by the $k$-nearest neighbor approach. This aligns with our quantitative results, as the NeCS method scores higher _diversity_ than others on all datasets.

TGCF uses the thresholds of a trained tree-based classifier to place the counterfactuals. The validity issues with TGCF are clearly visible, with most counterfactuals still belonging to the source cluster. However, based on the experiment data, it still achieves a _validity_ score of 56% across every target-instance pair for Wine.


=== General comparison <sec:general_comp>
#figure(
  grid(rows: 2,row-gutter: 15pt,
  metric-table(
    csv("../assets/full_tables/breast_cancer_all.csv", row-type: dictionary),
    ("Baseline (DiCE)", "Baseline (BayCon)", "Baycon - Model Agnostic","Baycon - KMeans"),
    ("Dissimilarity","Sparsity", "Plausibility", "Validity", "Diversity", "Runtime", "%Explained", "#Valid"),
    header: (
      table.cell([*Breast Cancer (Baseline methods)*], colspan: 5),"",
      [*Baseline (DiCE)*],
      [*Baseline (BayCon)*], 
      [*BayCon (Model-\ Agnostic)*], 
      [*BayCon \ ($k$-means)*]
    )
  ),
  metric-table(
    csv("../assets/full_tables/breast_cancer_all.csv", row-type: dictionary),
    ("GradCF (eps=0) (sparsity fixed)", "CFAE", "NeighborSearch","TGCF_DTC_Fidelity'"),
    ("Dissimilarity","Sparsity", "Plausibility", "Validity", "Diversity", "Runtime","%Explained", "#Valid"),
    header: (
      table.cell([*Breast Cancer (Novel methods)*], colspan: 5),"",
      [*GradCF*],[*CFAE*], [*NeCS*], [*TGCF*]
      ),
      fill_width: true
    )
  ),
  caption: [Summary of results for the tests run on the Breast Cancer dataset.]
)<tab:breast_cancer_results>

#figure(
  grid(rows: 2,row-gutter: 15pt,
    metric-table(
    csv("../assets/full_tables/wine_all.csv", row-type: dictionary),
    ("Baseline (DiCE)", "Baseline (BayCon)", "Baycon - Model Agnostic","Baycon - KMeans"),
    ("Dissimilarity","Sparsity", "Plausibility", "Validity", "Diversity", "Runtime", "%Explained", "#Valid"),
    header: (
      table.cell([*Wine (Baseline methods)*], colspan: 5),"",
      [*Baseline (DiCE)*],
      [*Baseline (BayCon)*], 
      [*BayCon (Model-\ Agnostic)*], 
      [*BayCon \ ($k$-means)*]
    )
  ),
  metric-table(
    csv("../assets/full_tables/wine_all.csv", row-type: dictionary),
    ("GradCF (eps=0) (sparsity fixed)", "CFAE", "NeighborSearch","TGCF_DTC_Fidelity'"),
    ("Dissimilarity","Sparsity", "Plausibility", "Validity", "Diversity", "Runtime", "%Explained", "#Valid"),
    header: (
      table.cell([*Wine (Novel methods)*], colspan: 5),"",
      [*GradCF*],[*CFAE*], [*NeCS*], [*TGCF*]
      ),
      fill_width: true
    )
  ),
  caption: [Summary of results for the tests run on the Wine dataset.]
)<tab:wine_results>

#figure(
  grid(rows: 2,row-gutter: 15pt,
    metric-table(
    csv("../assets/full_tables/blobs_all.csv", row-type: dictionary),
    ("Baseline (DiCE)", "Baseline (BayCon)", "Baycon - Model Agnostic","Baycon - KMeans"),
    ("Dissimilarity","Sparsity", "Plausibility", "Validity", "Diversity", "Runtime", "%Explained", "#Valid"),
    header: (
      table.cell([*Synthetic (Baseline methods)*], colspan: 5),"",
      [*Baseline (DiCE)*],
      [*Baseline (BayCon)*], 
      [*BayCon (Model-\ Agnostic)*], 
      [*BayCon \ ($k$-means)*]
    )
  ),
  metric-table(
    csv("../assets/full_tables/blobs_all.csv", row-type: dictionary),
    ("GradCF (eps=0) (sparsity fixed)", "CFAE", "NeighborSearch","TGCF_DTC_Fidelity'"),
    ("Dissimilarity","Sparsity", "Plausibility", "Validity", "Diversity", "Runtime", "%Explained", "#Valid"),
    header: (
      table.cell([*Synthetic (Novel methods)*], colspan: 5),"",
      [*GradCF*],[*CFAE*], [*NeCS*], [*TGCF*]
      ), 
      fill_width: true
    )
  ),
  caption: [Summary of results for the tests run on the synthetic dataset.]
)<tab:blobs_results>

In this section, we compare the algorithm variations selected in @sec:results with the unsupervised adaptations of DiCE and BayCon, as well as the unsupervised BayCon variations presented in @sec:baycon_soft. These will subsequently be referred to as the baseline methods. Although the experiments were run on each of the 4 datasets from @tab:datasets, we present only the relevant results for the Breast Cancer, Wine, and synthetic datasets. A complete overview of the results can be seen in @app:dataframe_all. In @tab:breast_cancer_results, the results for each method run on the Breast Cancer dataset are presented. The results for the Wine dataset can be seen in @tab:wine_results, and the results for the synthetic dataset are presented in @tab:blobs_results.

When comparing the baseline methods with the novel methods for the Breast Cancer dataset, one notable difference is the _%Explained_ metric. The baseline methods are unable to produce a valid counterfactual for a significant amount of instances. GradCF and NeCS are able to generate a valid counterfactual for all instances, while CFAE misses a single instance. However, TGCF explains only 46% of instances, which is roughly on par with the baseline methods from @tab:breast_cancer_results. We attribute this _%Explained_ score to the issues presented in @sec:tree_comparison.

While it seems the baseline methods vastly outperform the novel methods in the _dissimilarity_ metric, this is mainly due to the fact that invalid counterfactuals are not considered when evaluating this metric, as explained in @sec:metrics. Thus, _dissimilarity_ is not a good metric for comparison between methods when the _%Explained_ or _validity_ metric is very low. The low _dissimilarity_ score is likely obtained from the easy cases, where the instance $x$ was very close to the cluster boundary in the first place. This same logic can also be applied to _sparsity_, _plausibility_, and _diversity_. Including invalid counterfactuals in these metrics would make the results meaningless, as a trivial method could just focus on creating counterfactuals that maximize the metrics instead of focusing on valid, usable counterfactuals. However, failing to create counterfactuals for more than 17% of the instances is a major flaw, since being unable to create any counterfactuals for certain instances makes the method unreliable. Especially since the 17% of instances are likely the less trivial cases, where counterfactuals would be more useful for gaining insight in the underlying clusters.

The Wine dataset is where the performance of the baseline methods is most similar to that of the novel methods. All the baseline methods have an almost perfect _%Explained_ score, except for Baseline (DiCE), which is slightly lower. These results might be achieved because of the high homogeneity- and completeness-scores for the Wine dataset as presented in @tab:datasets, which is evidence of clearly defined clusters in the data. Apart from NeCS, all novel methods outperform the baseline methods on _plausibility_, but for _dissimilarity_ the scores are not decisive enough to declare which is better. All baseline methods score highly on the _sparsity_ metric, with NeCS being the only novel method which achieves a comparable score.

The synthetic dataset shows similar results to the Breast Cancer dataset, where the baseline methods are unable to generate counterfactuals for most instances. The exception to this is the Baseline (DiCE) variant, which explains $100%$ of the data. This variant does however attain a worse similarity than the novel methods. For the remaining metrics, Baseline (DiCE) gets outperformed by at least one of the novel methods.

The novel methods achieve a comparable _diversity_ score amongst themselves. The only slight outlier is NeCS, which generally outperforms the other novel methods. The baseline methods are comparable to the novel methods, except for DiCE, which generally outperforms all other methods in this regard. As DiCE is the only method presented in this thesis which directly optimizes for diversity, this is an expected result.

The BayCon-based methods exhibit some fundamental differences from the remaining methods, resulting from their approach of generating a large amount of counterfactuals for each instance. This approach greatly affects their _validity_, which is generally very low. This also results in them usually producing more valid counterfactuals than the remaining methods as evident in the _\#Valid_ metric. Whether more counterfactuals are beneficial is heavily dependent on the use-case, and may be a detriment as it can easily overwhelm users by presenting too many options. 

Additionally, the baseline methods are quite slow, taking much longer to compute their counterfactuals than all novel methods. GradCF is the slowest of the novel methods, but even GradCF is often 10 times faster than the baselines in explaining an instance $x$. For the remaining novel methods, this effect is even more pronounced, as they are over 100 times faster than the baseline methods.

Overall, the baseline methods are competitive in datasets like Wine, but on other datasets their performances were inconsistent. On the other hand, the novel methods introduced in this thesis seem to function well on all datasets, except for TGCF on the Breast Cancer dataset. This consistency shows that the novel methods are robust, and that it's unlikely for a specific dataset to render one of them unusable.