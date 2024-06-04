## Error for 1_Baseline

cannot import name 'SequenceNotStr' from 'pandas._typing' (/home/giovanni/.local/lib/python3.10/site-packages/pandas/_typing.py)
Traceback (most recent call last):
  File "/home/giovanni/.local/lib/python3.10/site-packages/supervised/base_automl.py", line 1178, in _fit
    trained = self.train_model(params)
  File "/home/giovanni/.local/lib/python3.10/site-packages/supervised/base_automl.py", line 387, in train_model
    self.keep_model(mf, model_subpath)
  File "/home/giovanni/.local/lib/python3.10/site-packages/supervised/base_automl.py", line 300, in keep_model
    self.select_and_save_best()
  File "/home/giovanni/.local/lib/python3.10/site-packages/supervised/base_automl.py", line 1364, in select_and_save_best
    ldb.to_csv(os.path.join(self._results_path, "leaderboard.csv"), index=False)
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/util/_decorators.py", line 211, in wrapper
    raise TypeError(msg)
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/core/generic.py", line 3720, in to_csv
    str or None
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/util/_decorators.py", line 211, in wrapper
    raise TypeError(msg)
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/io/formats/format.py", line 1162, in to_csv
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/io/formats/csvs.py", line 24, in <module>
    from pandas._typing import SequenceNotStr
ImportError: cannot import name 'SequenceNotStr' from 'pandas._typing' (/home/giovanni/.local/lib/python3.10/site-packages/pandas/_typing.py)


Please set a GitHub issue with above error message at: https://github.com/mljar/mljar-supervised/issues/new

## Error for 2_DecisionTree

cannot import name 'SequenceNotStr' from 'pandas._typing' (/home/giovanni/.local/lib/python3.10/site-packages/pandas/_typing.py)
Traceback (most recent call last):
  File "/home/giovanni/.local/lib/python3.10/site-packages/supervised/base_automl.py", line 1178, in _fit
    trained = self.train_model(params)
  File "/home/giovanni/.local/lib/python3.10/site-packages/supervised/base_automl.py", line 387, in train_model
    self.keep_model(mf, model_subpath)
  File "/home/giovanni/.local/lib/python3.10/site-packages/supervised/base_automl.py", line 300, in keep_model
    self.select_and_save_best()
  File "/home/giovanni/.local/lib/python3.10/site-packages/supervised/base_automl.py", line 1364, in select_and_save_best
    ldb.to_csv(os.path.join(self._results_path, "leaderboard.csv"), index=False)
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/util/_decorators.py", line 211, in wrapper
    raise TypeError(msg)
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/core/generic.py", line 3720, in to_csv
    str or None
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/util/_decorators.py", line 211, in wrapper
    raise TypeError(msg)
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/io/formats/format.py", line 1162, in to_csv
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/io/formats/csvs.py", line 24, in <module>
    from pandas._typing import SequenceNotStr
ImportError: cannot import name 'SequenceNotStr' from 'pandas._typing' (/home/giovanni/.local/lib/python3.10/site-packages/pandas/_typing.py)


Please set a GitHub issue with above error message at: https://github.com/mljar/mljar-supervised/issues/new

## Error for 3_Default_Xgboost

cannot import name 'SequenceNotStr' from 'pandas._typing' (/home/giovanni/.local/lib/python3.10/site-packages/pandas/_typing.py)
Traceback (most recent call last):
  File "/home/giovanni/.local/lib/python3.10/site-packages/supervised/base_automl.py", line 1178, in _fit
    trained = self.train_model(params)
  File "/home/giovanni/.local/lib/python3.10/site-packages/supervised/base_automl.py", line 384, in train_model
    mf.train(results_path, model_subpath)
  File "/home/giovanni/.local/lib/python3.10/site-packages/supervised/model_framework.py", line 249, in train
    learner.fit(
  File "/home/giovanni/.local/lib/python3.10/site-packages/supervised/algorithms/xgboost.py", line 235, in fit
    result.to_csv(log_to_file, index=False, header=False)
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/util/_decorators.py", line 211, in wrapper
    raise TypeError(msg)
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/core/generic.py", line 3720, in to_csv
    str or None
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/util/_decorators.py", line 211, in wrapper
    raise TypeError(msg)
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/io/formats/format.py", line 1162, in to_csv
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/io/formats/csvs.py", line 24, in <module>
    from pandas._typing import SequenceNotStr
ImportError: cannot import name 'SequenceNotStr' from 'pandas._typing' (/home/giovanni/.local/lib/python3.10/site-packages/pandas/_typing.py)


Please set a GitHub issue with above error message at: https://github.com/mljar/mljar-supervised/issues/new

## Error for 4_Default_NeuralNetwork

cannot import name 'SequenceNotStr' from 'pandas._typing' (/home/giovanni/.local/lib/python3.10/site-packages/pandas/_typing.py)
Traceback (most recent call last):
  File "/home/giovanni/.local/lib/python3.10/site-packages/supervised/base_automl.py", line 1178, in _fit
    trained = self.train_model(params)
  File "/home/giovanni/.local/lib/python3.10/site-packages/supervised/base_automl.py", line 384, in train_model
    mf.train(results_path, model_subpath)
  File "/home/giovanni/.local/lib/python3.10/site-packages/supervised/model_framework.py", line 249, in train
    learner.fit(
  File "/home/giovanni/.local/lib/python3.10/site-packages/supervised/algorithms/nn.py", line 60, in fit
    result.to_csv(log_to_file, index=False, header=False)
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/util/_decorators.py", line 211, in wrapper
    raise TypeError(msg)
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/core/generic.py", line 3720, in to_csv
    str or None
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/util/_decorators.py", line 211, in wrapper
    raise TypeError(msg)
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/io/formats/format.py", line 1162, in to_csv
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/io/formats/csvs.py", line 24, in <module>
    from pandas._typing import SequenceNotStr
ImportError: cannot import name 'SequenceNotStr' from 'pandas._typing' (/home/giovanni/.local/lib/python3.10/site-packages/pandas/_typing.py)


Please set a GitHub issue with above error message at: https://github.com/mljar/mljar-supervised/issues/new

## Error for 5_Default_RandomForest

cannot import name 'SequenceNotStr' from 'pandas._typing' (/home/giovanni/.local/lib/python3.10/site-packages/pandas/_typing.py)
Traceback (most recent call last):
  File "/home/giovanni/.local/lib/python3.10/site-packages/supervised/base_automl.py", line 1178, in _fit
    trained = self.train_model(params)
  File "/home/giovanni/.local/lib/python3.10/site-packages/supervised/base_automl.py", line 384, in train_model
    mf.train(results_path, model_subpath)
  File "/home/giovanni/.local/lib/python3.10/site-packages/supervised/model_framework.py", line 249, in train
    learner.fit(
  File "/home/giovanni/.local/lib/python3.10/site-packages/supervised/algorithms/sklearn.py", line 178, in fit
    df_result.to_csv(log_to_file, index=False, header=False)
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/util/_decorators.py", line 211, in wrapper
    raise TypeError(msg)
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/core/generic.py", line 3720, in to_csv
    str or None
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/util/_decorators.py", line 211, in wrapper
    raise TypeError(msg)
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/io/formats/format.py", line 1162, in to_csv
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/io/formats/csvs.py", line 24, in <module>
    from pandas._typing import SequenceNotStr
ImportError: cannot import name 'SequenceNotStr' from 'pandas._typing' (/home/giovanni/.local/lib/python3.10/site-packages/pandas/_typing.py)


Please set a GitHub issue with above error message at: https://github.com/mljar/mljar-supervised/issues/new

## Error for Ensemble

cannot import name 'SequenceNotStr' from 'pandas._typing' (/home/giovanni/.local/lib/python3.10/site-packages/pandas/_typing.py)
Traceback (most recent call last):
  File "/home/giovanni/.local/lib/python3.10/site-packages/supervised/base_automl.py", line 1174, in _fit
    trained = self.ensemble_step(
  File "/home/giovanni/.local/lib/python3.10/site-packages/supervised/base_automl.py", line 422, in ensemble_step
    self.keep_model(self.ensemble, ensemble_subpath)
  File "/home/giovanni/.local/lib/python3.10/site-packages/supervised/base_automl.py", line 300, in keep_model
    self.select_and_save_best()
  File "/home/giovanni/.local/lib/python3.10/site-packages/supervised/base_automl.py", line 1364, in select_and_save_best
    ldb.to_csv(os.path.join(self._results_path, "leaderboard.csv"), index=False)
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/util/_decorators.py", line 211, in wrapper
    raise TypeError(msg)
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/core/generic.py", line 3720, in to_csv
    str or None
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/util/_decorators.py", line 211, in wrapper
    raise TypeError(msg)
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/io/formats/format.py", line 1162, in to_csv
  File "/home/giovanni/.local/lib/python3.10/site-packages/pandas/io/formats/csvs.py", line 24, in <module>
    from pandas._typing import SequenceNotStr
ImportError: cannot import name 'SequenceNotStr' from 'pandas._typing' (/home/giovanni/.local/lib/python3.10/site-packages/pandas/_typing.py)


Please set a GitHub issue with above error message at: https://github.com/mljar/mljar-supervised/issues/new

