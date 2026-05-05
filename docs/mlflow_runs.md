# MLflow experiment: credit-default-risk

Runs (from `mlflow.search_runs`):

| run                    | algorithm              |   maxDepth |   maxIter |   regParam |   elasticNetParam |   test_auc |   test_pr_auc |   test_ks |   cv_best_auc |
|:-----------------------|:-----------------------|-----------:|----------:|-----------:|------------------:|-----------:|--------------:|----------:|--------------:|
| gradient_boosted_trees | gradient_boosted_trees |          5 |        40 |            |                   |     0.7831 |        0.5542 |    0.4258 |         0.78  |
| logistic_regression    | logistic_regression    |            |           |       0.01 |                 0 |     0.7556 |        0.5069 |    0.4068 |         0.756 |

Model registry:

| model | version | aliases | test_auc tag | run |
|---|---:|---|---:|---|
| credit_default_gbt | 1 | champion | 0.7831 | f4196b1f |
