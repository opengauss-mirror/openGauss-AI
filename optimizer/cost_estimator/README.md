# Learning-based-cost-estimator

Source code of Feature Encoding and Model Training for cardinality and cost estimation based on openGauss Execution Plans.
The tree-structured model can generate representations for sub-plans which could be used by other query processing tasks like Join Reordering,
Materialized View and even Index Selection.

## Setup

Modify paths in test files.

```python
# Relational Tables
data_dir = '/home/sunji/cost_estimation/test_files_open_source/imdb_data_csv'
# String Embedding Files
word_vec_file = '/home/sunji/cost_estimation/test_files_open_source/wordvectors_updated.kv'
# MinMax Value in All Attributes
minmax_file = '/home/sunji/cost_estimation/test_files_open_source/min_max_vals.json'
# Temporary Directory
tmp_dir = '/home/sunji/cost_estimation/test_files_open_source/'
```

## Unit Test

Run the following script in `src/gausskernel/dbmind/kernel` directory.

```bash
export PYTHONPATH=$PYTHONPATH:cost_estimator/
python -m unittest cost_estimator.test.test_feature_extraction.TestFeatureExtraction
python -m unittest cost_estimator.test.test_feature_encoding.TestFeatureEncoding
python -m unittest cost_estimator.test.test_training.TestTraining
```

## Test Data

We offer the test data including sampled Execution Plans from openGauss,
IMDB datasets, Statistics of Minimum and Maximum number of each columns.

[click here for files](https://cloud.tsinghua.edu.cn/d/3d1966e2e81040f99607/)  

The pre-trained dictionary for string keywords is too large. Due to the file size limitation of Tsinghua cloud, we split it into two files `wordvectors_updated.kv.vectors.npy_aa` and `wordvectors_updated.kv.vectors.npy_ab`, you need to merge two files by following command:

```bash
cat wordvectors_updated.kv.vectors.npy_aa wordvectors_updated.kv.vectors.npy_ab > wordvectors_updated.kv
```
