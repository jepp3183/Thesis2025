#import "../lib.typ": *

= TGCF Decision Tree Classifier tree structure example <app:tt_tree_example>

#figure(
  image("../assets/tt_dtc_low_validity_tree.png", width: 100%),
  caption: [Generated decision tree for TGCF using Decision Tree Classifier as surrogate model.],
  placement: none
)<fig:dtc_low_validity_tree>

#pagebreak()

= Sparsity improvement of TGCF <app:full_tgcf_sparsity>

#figure(
  metric-table(
    csv("../assets/tgcf_sparsity_fix.csv", row-type: dictionary),
    ("TGCF","TGCF *","2TGCF","2TGCF *"),
    ("Dissimilarity","Sparsity", "Plausibility", "Validity", "Diversity", "Runtime", "%Explained", "#Valid"),
    header: (
      table.cell("",colspan:1, rowspan:2),table.cell([*Synthetic*], colspan:2),table.cell([*Breast Cancer*], colspan:2),
      [*TGCF*],[*TGCF \**],[*TGCF*],[*TGCF \**]
    )
  ),
  caption: [TGCF with and without sparsity fix.],
  placement: none,
)


#pagebreak()
= Test results
<app:dataframe_all>
In this appendix all of the tables from the result section can be seen.

== Breast Cancer dataset 

#figure(
  metric-table(
    csv("../assets/full_tables/breast_cancer_all.csv", row-type: dictionary),
    ("Baseline (DiCE)", "Baseline (BayCon)", "Baycon - Model Agnostic","Baycon - KMeans"),
    ("Dissimilarity","Sparsity", "Plausibility", "Validity", "Diversity", "Runtime", "%Explained", "#Valid"),
    header: (
      table.cell([*Breast Cancer (Baseline methods)*], colspan: 5),"",
      [*Baseline (DiCE)*],
      [*Baseline (BayCon)*], 
      [*BayCon (Model-\ Agnostic)*], 
      [*BayCon\ ($k$-means)*]
    ),
    fill_width: true
  ),
  placement: none
)

#figure(
  metric-table(
    csv("../assets/full_tables/breast_cancer_all.csv", row-type: dictionary),
    ("GradCF (eps=0) (sparsity fixed)", "CFAE", "NeighborSearch","TGCF_DTC_Fidelity'"),
    ("Dissimilarity","Sparsity", "Plausibility", "Validity", "Diversity", "Runtime","%Explained", "#Valid"),
    header: (
      table.cell([*Breast Cancer (Novel methods)*], colspan: 5),"",
      [*GradCF*],[*CFAE*], [*NeCS*], [*TGCF*]
      ),
    fill_width: true
    ),
  placement: none
)

== Synthetic dataset
#let all_baseline_blobs = csv("../assets/blobs_baselines.csv").flatten()
#figure(
  metric-table(
    csv("../assets/full_tables/blobs_all.csv", row-type: dictionary),
    ("Baseline (DiCE)", "Baseline (BayCon)", "Baycon - Model Agnostic","Baycon - KMeans"),
    ("Dissimilarity","Sparsity", "Plausibility", "Validity", "Diversity", "Runtime", "%Explained", "#Valid"),
    header: (
      table.cell([*Synthetic (Baseline methods)*], colspan: 5),"",
      [*Baseline (DiCE)*],
      [*Baseline (BayCon)*], 
      [*BayCon (Model-\ Agnostic)*], 
      [*BayCon\ ($k$-means)*]
    ),
    fill_width: true
  ),
  placement: none
)

#let all_novels_blobs = csv("../assets/blobs_novels.csv").flatten()
#figure(
  metric-table(
    csv("../assets/full_tables/blobs_all.csv", row-type: dictionary),
    ("GradCF (eps=0) (sparsity fixed)", "CFAE", "NeighborSearch","TGCF_DTC_Fidelity'"),
    ("Dissimilarity","Sparsity", "Plausibility", "Validity", "Diversity", "Runtime", "%Explained", "#Valid"),
    header: (
      table.cell([*Synthetic (Novel methods)*], colspan: 5),"",
      [*GradCF*],[*CFAE*], [*NeCS*], [*TGCF*]
      ),
    fill_width: true
    ),
  placement: none
)

== Wine dataset
#let all_baseline_wine = csv("../assets/wine_baselines.csv").flatten()
#figure(
  metric-table(
    csv("../assets/full_tables/wine_all.csv", row-type: dictionary),
    ("Baseline (DiCE)", "Baseline (BayCon)", "Baycon - Model Agnostic","Baycon - KMeans"),
    ("Dissimilarity","Sparsity", "Plausibility", "Validity", "Diversity", "Runtime", "%Explained", "#Valid"),
    header: (
      table.cell([*Wine (Baseline methods)*], colspan: 5),"",
      [*Baseline (DiCE)*],
      [*Baseline (BayCon)*], 
      [*BayCon (Model-\ Agnostic)*], 
      [*BayCon\ ($k$-means)*]
    ),
    fill_width: true
  ),
  placement: none
)

#let all_novels_wine = csv("../assets/wine_novels.csv").flatten()
#figure(
  metric-table(
    csv("../assets/full_tables/wine_all.csv", row-type: dictionary),
    ("GradCF (eps=0) (sparsity fixed)", "CFAE", "NeighborSearch","TGCF_DTC_Fidelity'"),
    ("Dissimilarity","Sparsity", "Plausibility", "Validity", "Diversity", "Runtime", "%Explained", "#Valid"),
    header: (
      table.cell([*Wine (Novel methods)*], colspan: 5),"",
      [*GradCF*],[*CFAE*], [*NeCS*], [*TGCF*]
      ),
    fill_width: true
    ),
  placement: none
)

== Iris dataset
#let all_baseline_iris = csv("../assets/iris_baselines.csv").flatten()
#figure(
  metric-table(
    csv("../assets/full_tables/iris_all.csv", row-type: dictionary),
    ("Baseline (DiCE)", "Baseline (BayCon)", "Baycon - Model Agnostic","Baycon - KMeans"),
    ("Dissimilarity","Sparsity", "Plausibility", "Validity", "Diversity", "Runtime", "%Explained", "#Valid"),
    header: (
      table.cell([*Iris (Baseline methods)*], colspan: 5),"",
      [*Baseline (DiCE)*],
      [*Baseline (BayCon)*], 
      [*BayCon (Model-\ Agnostic)*], 
      [*BayCon\ ($k$-means)*]
    ),
    fill_width: true
  ),
  placement: none
)

#let all_novels_iris = csv("../assets/iris_novels.csv").flatten()
#figure(
  metric-table(
    csv("../assets/full_tables/iris_all.csv", row-type: dictionary),
    ("GradCF (eps=0) (sparsity fixed)", "CFAE", "NeighborSearch","TGCF_DTC_Fidelity'"),
    ("Dissimilarity","Sparsity", "Plausibility", "Validity", "Diversity", "Runtime", "%Explained", "#Valid"),
    header: (
      table.cell([*Iris (Novel methods)*], colspan: 5),"",
      [*GradCF*],[*CFAE*], [*NeCS*], [*TGCF*]
      ),
    fill_width: true
    ),
  placement: none
)

