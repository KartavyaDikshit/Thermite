# Thermite vs Scikit-Learn Benchmark Results

| Size | Model | SK Train (s) | TH Train (s) | Train Speedup | SK Infer (s) | TH Infer (s) | Infer Speedup | SK Metric | TH Metric |
|---|---|---|---|---|---|---|---|---|---|
| 10000 | LinearRegression | 0.002 | 0.001 | 1.38x | 0.000 | 0.000 | 0.34x | 1.0000 | 1.0000 |
| 10000 | LogisticRegression | 0.003 | 0.003 | 0.81x | 0.000 | 0.001 | 0.19x | 0.8892 | 0.8747 |
| 10000 | RandomForestClassifier | 0.565 | 0.306 | 1.85x | 0.017 | 0.004 | 4.29x | 0.9504 | 0.9493 |
| 10000 | GradientBoostingRegressor | 2.347 | 0.942 | 2.49x | 0.008 | 0.009 | 0.87x | 0.9595 | 0.9595 |
| 10000 | HistGradientBoostingClassifier | 0.212 | 0.340 | 0.62x | 0.004 | 0.006 | 0.64x | 0.9548 | 0.9231 |
| 10000 | KMeans | 0.053 | 0.004 | 12.64x | 0.000 | 0.000 | 1.35x | 0.0000 | 0.0000 |
| 10000 | MiniBatchKMeans | 0.011 | 0.034 | 0.32x | 0.000 | 0.000 | 1.80x | 0.0000 | 0.0000 |
| 10000 | pairwise_distances | 0.255 | 0.826 | 0.31x | 0.000 | 0.000 | 0.00x | 0.0000 | 0.0000 |
| 100000 | LinearRegression | 0.015 | 0.015 | 0.98x | 0.001 | 0.007 | 0.09x | 1.0000 | 1.0000 |
| 100000 | LogisticRegression | 0.012 | 0.056 | 0.21x | 0.001 | 0.008 | 0.11x | 0.8680 | 0.8613 |
| 100000 | RandomForestClassifier | 7.139 | 10.304 | 0.69x | 0.158 | 0.040 | 3.90x | 0.8800 | 0.8795 |
| 100000 | GradientBoostingRegressor | 27.915 | 16.359 | 1.71x | 0.070 | 0.091 | 0.78x | 0.9286 | 0.9286 |
| 100000 | HistGradientBoostingClassifier | 0.311 | 3.697 | 0.08x | 0.027 | 0.076 | 0.35x | 0.8770 | 0.8698 |
| 100000 | KMeans | 0.016 | 0.007 | 2.30x | 0.001 | 0.002 | 0.70x | 0.0000 | 0.0000 |
| 100000 | MiniBatchKMeans | 0.013 | 0.338 | 0.04x | 0.002 | 0.002 | 0.91x | 0.0000 | 0.0000 |
| 10000 | pairwise_distances | 0.291 | 0.792 | 0.37x | 0.000 | 0.000 | 0.00x | 0.0000 | 0.0000 |
