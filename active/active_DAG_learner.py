import numpy as np
import pandas as pd
from modAL.models import ActiveLearner
from typing import Callable, Optional, Union, Tuple
from DAG_estimator import DAGEstimator
from sklearn.utils.validation import check_array
from modAL.utils.data import data_vstack, modALinput
import warnings
from DAG_acquisition_functions import uniform


class ActiveDAGLearner(ActiveLearner):

    def __init__(self,
                 estimator: DAGEstimator,
                 query_strategy: Callable = uniform,
                 X_training: Optional[modALinput] = None,
                 bootstrap_init: bool = False,
                 on_transformed: bool = False,
                 pretrain_epochs = 16,
                 **fit_kwargs
                 ) -> None:
        
        y_training = None
        super().__init__(estimator, query_strategy,
                         X_training, y_training, bootstrap_init, on_transformed, max_epochs=pretrain_epochs, **fit_kwargs)
        # self._fit_to_known(max_epochs=64)
        
    def query(self, *query_args, **query_kwargs):

        query_result = self.query_strategy(self.estimator.get_vars(), self.estimator.sample_models(), *query_args, **query_kwargs)
        return query_result

    def _add_training_data(self, X: modALinput) -> None:

        X = check_array(X, accept_sparse=True, accept_large_sparse=True, dtype=None,
            force_all_finite=self.force_all_finite, ensure_2d=False, allow_nd=True, input_name="X",)


        if self.X_training is None:
            self.X_training = X
        else:
            try:
                if(type(self.X_training) == np.ndarray):
                    self.X_training = data_vstack((self.X_training, X))
                elif(type(self.X_training) == pd.DataFrame):
                    self.X_training = pd.concat([self.X_training, pd.DataFrame(X)], axis=0)
            except ValueError:
                raise ValueError('the dimensions of the new training data must'
                                 'agree with the training data provided so far')

    def _fit_to_known(self, bootstrap: bool = False, **fit_kwargs) :
        if not bootstrap:
            self.estimator.fit(self.X_training, **fit_kwargs)
        else:
            n_instances = self.X_training.shape[0]
            bootstrap_idx = np.random.choice(range(n_instances), n_instances, replace=True)
            self.estimator.fit(self.X_training[bootstrap_idx], **fit_kwargs)  

    def _fit_on_new(self, X: modALinput, bootstrap: bool = False, **fit_kwargs):

        X = check_array(X, accept_sparse=True, accept_large_sparse=True, dtype=None,
        force_all_finite=self.force_all_finite, ensure_2d=False, allow_nd=True, input_name="X",)

        if not bootstrap:
            self.estimator.fit(X, **fit_kwargs)
        else:
            bootstrap_idx = np.random.choice(range(X.shape[0]), X.shape[0], replace=True)
            self.estimator.fit(X[bootstrap_idx])

    def teach(self, X, only_new=False, bootstrap=False, **fit_kwargs):
        self._add_training_data(X)
        if not only_new:
            self._fit_to_known(bootstrap=bootstrap, **fit_kwargs)
        else:
            self._fit_on_new(X, bootstrap=bootstrap, **fit_kwargs)

    def score(self, X):
        self.estimator.score(X)

    def sample_models(self):
        self.estimator.sample_models()

    def get_vars(self):
        self.estimator.get_vars()
