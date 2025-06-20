#import "@preview/dashy-todo:0.0.2": todo
#import "@preview/lemmify:0.1.8": *
#import "@preview/lovelace:0.3.0": *
#import "@preview/equate:0.3.1": equate

#let (
  theorem, lemma, corollary,
  remark, proposition, example,
  proof, rules: thm-rules
) = default-theorems(
  "thm-group",
  lang: "en",
  thm-numbering: thm-numbering-linear,
  max-reset-level: 0,
)


#let baseline_headers = ("", [*Baseline (DiCE)*], [*Baseline (BayCon)*], [*BayCon (Model-\ Agnostic)*], [*BayCon ($k$-means)*])
#let novel_headers = ("", [*GradCF*], [*CFAE*], [*NeCS*], [*TGCF*])

#let argmin = math.op("arg min", limits: true)
#let comment(content) = [*\##content*]

#let metric-table(
  data, 
  methods, 
  metrics, 
  header: none, 
  include_std: true,
  fill_width: false,
) = {
  let row-dict = (:)
  for row in data {
    let metric_name = row.at("")
    let _ = row.remove("")
    if metrics.contains(metric_name) {
      row-dict.insert(metric_name, row)
    }
  }

  //filter columns by chosen methods
  let filtered_rows = (:)
  for (k, v) in row-dict {
    
    let f(pair) = {
      let (m, _) = pair
      methods.contains(m)
    }
    
    let f = v.pairs().filter(f)
    for (method, value) in f {

      let dict = filtered_rows.at(k, default: (:))
      dict.insert(str(method), value)
      
      filtered_rows.insert(k, dict)
    }
  }

  // Format data for table
  let header_data = ()
  if header == none {
    header_data.push("")
    for m in methods {header_data.push([*#m*])}  
  } else {
    for h in header {header_data.push(h)}
  }

  let table_data = ()
  for metric in metrics {
    table_data.push(metric)
    let dict = filtered_rows.at(metric)
    for method in methods {
      let v = dict.at(method)
      let split = v.split()
      if split.len() == 3 {let _ = split.remove(1)}

      let avg = 999999999999999 //placeholder for redefining later...
      let stdenv = avg
      if split.len() == 2 {(avg, stdenv) = split} 
      else {avg = split.at(0)}

      if include_std and split.len() == 2 {
        if avg != "N/A" {
          let adecs = 2
          let sdecs = 2

          avg = decimal(avg)
          stdenv = decimal(stdenv)

          if avg > 99 {adecs = 1}
          if stdenv > 99 {sdecs = 1}
          
          avg = calc.round(avg,digits: adecs)
          stdenv = calc.round(stdenv, digits: sdecs)
        }
        table_data.push([$#avg plus.minus #stdenv$])
      } else {
        table_data.push([$#avg$])
      }
      
    }
  }

  let cols = methods.len() + 1
  if fill_width {
    cols = range(methods.len() + 1).map(c => 1fr)
  }
  
  table(
    columns: cols,
    ..header_data,
    ..table_data
  )
}

// #let data = csv("breast_cancer_complete.csv", row-type: dictionary)
// #let cols = ("Baycon - KMeans", "GradCF (eps=0)", "TGCF_DTC'")
// #let rows = ("Similarity", "% Explained", "Validity")
// #metric-table(data, cols, rows)