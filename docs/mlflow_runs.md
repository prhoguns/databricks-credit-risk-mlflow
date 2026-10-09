# MLflow experiment: credit-default-risk

Runs (from `mlflow.search_runs`):

| run                    | algorithm              |   maxDepth |   maxIter |   regParam |   elasticNetParam |   test_auc |   test_pr_auc |   test_ks |   cv_best_auc |
|:-----------------------|:-----------------------|-----------:|----------:|-----------:|------------------:|-----------:|--------------:|----------:|--------------:|
| gradient_boosted_trees | gradient_boosted_trees |          3 |        80 |            |                   |     0.7885 |        0.5613 |    0.4428 |        0.7815 |
| logistic_regression    | logistic_regression    |            |           |       0.01 |                 0 |     0.7519 |        0.508  |    0.4052 |        0.7529 |

Model registry:

| model | version | aliases | test_auc tag | run |
|---|---:|---|---:|---|
| credit_default_gbt | 1 | champion | 0.7885 | 24dcee95 |
