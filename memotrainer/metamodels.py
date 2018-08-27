"""
This module provides a prototypical interface that allows the user to train approximation models based on given
training datasets.
"""

import numpy as np
import copy


from sklearn.preprocessing.data import StandardScaler
from sklearn import neighbors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, MultiTaskLassoCV, RidgeCV
from sklearn.linear_model.coordinate_descent import MultiTaskElasticNetCV
from sklearn.kernel_ridge import KernelRidge
from sklearn import tree
from sklearn.svm import SVR
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from memotrainer import scores
import pandas


class MetaModel(object):
    """
    This class serves as a superclass for all approximation models and provides a common interface to be
    used by the Trainer and outside of this module. It manages a chain of preprocessing steps and provides the methods
    :meth:`fit` and :meth:`predict` to fit the concrete model to the given training data and to predict output values
    given the respective inputs.

    :param kwargs: arbitrary keyword arguments that will be passed to *this.model* when it is fitted to the training
           data.

    """

    train_score = {
        'r2_score': {
            'name': 'r2_score', 'function': make_scorer(
                r2_score, greater_is_better=True)},
        'avg': {
            'name': 'mean_absolute_error',
            'function': make_scorer(
                mean_absolute_error, greater_is_better=False)},
        'hae': {
            'name': 'harmonic_ average_error',
            'function': make_scorer(
                scores.harmonic_averages_error, greater_is_better=False)},
        'mse': {
            'name': 'mean_squared_error',
            'function': make_scorer(
                mean_squared_error, greater_is_better=False)}}

    def __init__(self, input_names=None, response_names=None, preprocessors=None, trainer_score='r2_score', **kwargs):
        """

        :param kwargs: arbitrary keyword arguments that will be passed to *this.regression_pipeline* when it is fitted
               to the training data.
        """

        if input_names is None:
            input_names = []

        if response_names is None:
            response_names = []

        if preprocessors is None:
            preprocessors = []

        self.kwargs = kwargs

        self.regression_pipeline = None
        """
        An sklearn Pipeline. See http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline
        """
        self.processing_steps = preprocessors
        """
        A list of dataset transformations that will be added to the pipeline before the model will be fitted to the
        data.
        """

        self.input_names = input_names
        """
        A list of input names in the same order as in the training dataset. The trainer class sets this value
        according to the dataset used for the training.
        """

        self.response_names = response_names
        """
        A list of response names in the same order as in the training dataset. The trainer class sets this value
        according to the dataset used for the training.

        """

        self.score = self.train_score[trainer_score]
    
    def fit(self, x_train, y_train):
        """
        Fits the model to the data such that it reproduces the expected output data.

        Raises an exception because this method must be implemented by concrete MetaModel implementations.

        :param x_train: array-like of shape [n_samples,n_features]

            training data

        :param y_train: array-like of shape [n_samples, n_targets]

            target values

        """
        raise Exception('Not implemented')

    def predict(self, X):
        """
        Uses *self.regression_pipeline* to predict an output value for input vector *X*.

        :param X: array-like, shape = (n_samples, n_features)

            Input data

        :return: array, shape = (n_samples, n_outputs)

            Returns predicted values.

        """
        val = self.regression_pipeline.predict(X)
        return pandas.DataFrame(val, columns=self.response_names)

    # def _set_input_and_output_names(self, x_train, y_train):
    #     # assume input_data is a DataFrame and extract column names
    #     self.input_names = list(x_train.columns.values)
    #     self.response_names = list(y_train.columns.values)
    
    def _update_pipeline_and_fit(self, x_train, y_train, steps):
        """
        Constructs a pipeline, fits it to the input and output data and


        :param x_train: array-like, shape = (n_samples, n_features)

            input data

        :param y_train: array-like, shape = (n_samples, n_outputs)

            output data

        :param steps: list of data transformations

            data transformations to add to the model pipeline.

        """
        # work on a copy of the pipeline, so that it can be reused
        processing_steps = copy.copy(self.processing_steps)
        for step in steps: 
            processing_steps.append(step)
        pipeline = make_pipeline(*processing_steps)
        
        # perform preprocessing and create metamodel
        self.regression_pipeline = pipeline.fit(x_train, y_train)
        if hasattr(pipeline._final_estimator, 'best_params_'):
            print('best params: ', pipeline._final_estimator.best_params_)

    def create_model_parameter_dict(self, key_value):
        return {arg.key: arg.value for arg in key_value}

    def __repr__(self):
        return '%s [%s]' % (self.__class__.__name__, str(self.__dict__))
        #return 'MetaModel [model=%s, inputs=%s, responses=%s]' % (
        #    self.regression_pipeline, self.input_names, self.response_names)
    
    
class OLSModel(MetaModel):
    """
    Fits a linear model to the data using ordinary least squares method.
    See http://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares for a more detailed explanation
    of this method.

    :param kwargs:

        keyword arguments that will be passed to the constructor of the LinearRegression model.
        See http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression
        for more information about.rst available arguments.

    """

    def __init__(self, **kwargs):
        """

        :param kwargs:

            keyword arguments that will be passed to the constructor of the LinearRegression model.
            See: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression
            for more information about.rst available arguments.
        """
        MetaModel.__init__(self, **kwargs)
        
    def fit(self, x_train, y_train):
        """
        Fits the model to the data such that it reproduces the expected output data.

        :param x_train: array-like of shape [n_samples,n_features]

            training data

        :param y_train: array-like of shape [n_samples, n_targets]

            target values

        """
        ols = LinearRegression(**self.kwargs)
        self._update_pipeline_and_fit(x_train, y_train, [ols])
        
        
class Lasso(MetaModel):
    """
    Fits a linear model to the data using a multitask lasso implementation with built-in cross-validation.
    See http://scikit-learn.org/stable/modules/linear_model.html#lasso for
    a general explanation of the lasso method.

    :param kwargs: keyword arguments that will be passed on to the constructor of the lasso model. See
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLassoCV.html#sklearn.linear_model.MultiTaskLassoCV
        for more information about.rst available arguments.

    """
    def __init__(self, **kwargs):
        MetaModel.__init__(self, **kwargs)
        
    def fit(self, x_train, y_train):
        """
        Fits the model to the data such that it reproduces the expected output data.

        :param x_train: array-like of shape [n_samples,n_features]

            training data

        :param y_train: array-like of shape [n_samples, n_targets]

            target values

        """
        lasso = MultiTaskLassoCV(**self.kwargs)
        self._update_pipeline_and_fit(x_train, y_train, [lasso])


class RidgeRegressionModel(MetaModel):
    """


    """

    def __init__(self, **kwargs):
        MetaModel.__init__(self, **kwargs)

    def fit(self, x_train, y_train):
        """
        Fits the model to the data such that it reproduces the expected output data.
        cv = None, to use the efficient Leave-One-Out cross-validation

        :param x_train: array-like of shape [n_samples,n_features]

            training data

        :param y_train: array-like of shape [n_samples, n_targets]

            target values

        """
        regr = RidgeCV(**self.kwargs)
        self._update_pipeline_and_fit(x_train, y_train, [regr])


class ElasticNetModel(MetaModel):
    """
    Fits a linear model to the data using a multitask *elastic net* implementation with built-in cross-validation.
    See
    http://scikit-learn.org/stable/modules/linear_model.html#elastic-net
    for a general explanation of the elastic net method.


    :param kwargs: keyword arguments that will be passed on to the constructor of the lasso model. See
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNetCV.html#sklearn.linear_model.MultiTaskElasticNetCV
        for more information about.rst available arguments.
    """
    def __init__(self, **kwargs):
        MetaModel.__init__(self, **kwargs)
        
    def fit(self, x_train, y_train):
        """
        Fits the model to the data such that it reproduces the expected output data.

        :param x_train: array-like of shape [n_samples,n_features]

            training data

        :param y_train: array-like of shape [n_samples, n_targets]

            target values

        """
        elastic_net = MultiTaskElasticNetCV(**self.kwargs)
        self._update_pipeline_and_fit(x_train, y_train, [elastic_net])
        
         
class Kriging(MetaModel):
    """
    Fits a gaussian process to the data while optionally using GridSearchCV for an exhaustive search over specified
    parameter values.
    See
    http://scikit-learn.org/stable/modules/gaussian_process.html
    for a general explanation of gaussian processes and regression with gaussian processes.


    :param kwargs: keyword arguments that will be passed on to the constructor of the gaussian process. See
        http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcess.html#sklearn.gaussian_process.GaussianProcess
        for more information about.rst available arguments.

    """
    def __init__(self, **kwargs):
        MetaModel.__init__(self, **kwargs)
    
    def fit(self, x_train, y_train):
        """
        Fits the model to the data such that it reproduces the expected output data.

        :param x_train: array-like of shape [n_samples,n_features]

            training data

        :param y_train: array-like of shape [n_samples, n_targets]

            target values

        """

        """
        sklearn.gaussian_process.GaussianProcessRegressor(
            kernel=None,
            alpha=1e-10,
            optimizer='fmin_l_bfgs_b',
            n_restarts_optimizer=0,
            normalize_y=False,
            copy_X_train=True,
            random_state=None)[source]
        """

        #clf = GaussianProcess(theta0=0.1, thetaL=.001, thetaU=1., nugget=0.01, **self.kwargs)
        clf = GaussianProcessRegressor(alpha=0.01, n_restarts_optimizer=100, random_state=42)
        self._update_pipeline_and_fit(x_train, y_train, [clf])


class KNeighborsModel(MetaModel):
    """
    TODO: Rename to KNeighborsRegressor

    Uses nearest neighbors regression to represent a function that maps input values to output values
    while optionally using GridSearchCV for an exhaustive search over specified parameter values.
    See
    http://scikit-learn.org/stable/modules/neighbors.html#regression
    for a general explanation of nearest neighbors regression.


    :param kwargs: keyword arguments that will be passed on to the constructor of the nearset neighbors regressor. See
        http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor
        for more information about.rst available arguments.

        Additional keyword arguments:

        :param_grid: If the parameter *param_grid* is present in the keyword arguments it will be used to set up an
            exhaustive grid search for the best estimator among all combinations of hyperparameters such as the number
            of neighbors *n_neighbours* to consider and the way how the neighbors are weighed *weights*.
            See http://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search for an example of such
            a *param_grid*.

    """
    def __init__(self, **kwargs):
        MetaModel.__init__(self, **kwargs)
        
    def fit(self, x_train, y_train):
        """
        Fits the model to the data such that it reproduces the expected output data.

        :param x_train: array-like of shape [n_samples,n_features]

            training data

        :param y_train: array-like of shape [n_samples, n_targets]

            target values

        """
        if 'param_grid'in self.kwargs:
            param_grid =self.kwargs['param_grid']
        else:
            param_grid = {
                'n_neighbors': sp_randint(1, 15)
            }

        clf = RandomizedSearchCV(
            neighbors.KNeighborsRegressor(),
            param_distributions=param_grid,
            n_iter=5)

        self._update_pipeline_and_fit(x_train, y_train, [clf])


class ArtificialNeuralNetwork(MetaModel):
    def __init__(self, **kwargs):
        MetaModel.__init__(self, **kwargs)

    def fit(self, x_train, y_train):
        self.processing_steps = [StandardScaler()]

        ann = MLPRegressor()

        params = {'hidden_layer_sizes': sp_randint(20, 150),
                  'alpha': sp_uniform(0, 100),
                  'max_iter': sp_randint(100, 2000),
                  'solver': ['lbfgs'],
                  'activation': ['relu'] #'identity', 'logistic', 'tanh', 'relu'
        }
        if 'hidden_layer_sizes' in self.kwargs:
            self.kwargs['hidden_layer_sizes'] = self.parsefunction(self.kwargs['hidden_layer_sizes'])

        params.update(self.kwargs)
        clf = RandomizedSearchCV(
            estimator=ann,
            param_distributions=params,
            n_iter=10,
            scoring=self.score['function'])
        print('random search ann')
        #print(params)

        self._update_pipeline_and_fit(x_train, y_train, [clf])

    def parsefunction(self, string_tuple):
        array = []

        for string in string_tuple:
            tuple = self.parse_tuple(string)
            if tuple is None:
                tuple = int(string)
            array.append(tuple)
        return array

    def parse_tuple(self, string):
        try:
            s = eval(string)
            if type(s) == tuple:
                return s
            return
        except:
            return


class SupportVectorRegression(MetaModel):
    def __init__(self, **kwargs):
        MetaModel.__init__(self, **kwargs)

    def fit(self, x_train, y_train):
        self.processing_steps = [StandardScaler()]
        svr = SVR(kernel='rbf', gamma=0.1)

        '''http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf'''
        # C = [2**i for i in np.arange(start=-5, stop=16, step=2)]
        # gamma = [2**i for i in np.arange(start=-15, stop=4, step=2)]


        '''https://stats.stackexchange.com/questions/43943/which-search-range-for-determining-svm-optimal-c-and-gamma-parameters'''
        C = [2 ** i for i in [-3,-2, -1, 0, 1, 2, 3, 4, 5]]
        gamma = [2 ** i for i in [-5, -4, -3, -2, -1, 0, 1, 2, 3]]

        params = {"C": sp_uniform(0.125, 32),
                  "gamma": sp_uniform(0.03125, 8)}
        params.update(self.kwargs)

        reg = RandomizedSearchCV(
            estimator=svr,
            param_distributions=params,
            n_iter=10,
            scoring=self.score['function'])
            # clf = RandomizedSearchCV(estimator=svr, param_distributions=params, scoring=self.score['function'], n_iter=20)


        clf = MultiOutputRegressor(reg)
        print('random search svr')

     #   y_train = np.ravel(y_train)
        self._update_pipeline_and_fit(x_train, y_train, [clf])

class DecisionTreeRegression(MetaModel):
    def __init__(self, **kwargs):
        MetaModel.__init__(self, **kwargs)

    def fit(self, x_train, y_train):
        if 'param_grid' in self.kwargs:
            raise Exception('not implemented')
        else:
            clf = tree.DecisionTreeRegressor()
        self._update_pipeline_and_fit(x_train, y_train, [clf])


class KernelRidgeRegression(MetaModel):
    def __init__(self, **kwargs):
        MetaModel.__init__(self, **kwargs)

    def fit(self, x_train, y_train):
        if 'param_grid' in self.kwargs:
            raise Exception('not implemented')
        else:
            clf = KernelRidge(alpha=1.0)
        self._update_pipeline_and_fit(x_train, y_train, [clf])


class KernelRidgeRegressionCV(MetaModel):
    def __init__(self, **kwargs):
        MetaModel.__init__(self, **kwargs)

    def fit(self, x_train, y_train):
        # Fit KernelRidge with parameter selection based on 5-fold cross validation
        #if 'kernel' in self.kwargs:
        #    print(self.kwargs['kernel'])

        if all([np.isscalar(param) for param in self.kwargs.values()]):
            # print('all scalar')
            clf = KernelRidge(**self.kwargs)
        else:
            # print('not all scalar')
            param_grid = {}
            kwarg_params = {}
            for param_name, param_value in self.kwargs.items():
                if np.isscalar(param_value):
                    kwarg_params[param_name] = param_value
                else:
                    param_grid[param_name] = param_value

            clf = GridSearchCV(KernelRidge(**kwarg_params), param_grid=param_grid)

        from sklearn.preprocessing import PolynomialFeatures
        self._update_pipeline_and_fit(x_train, y_train, [clf])
        if len(param_grid) > 0:
            # TODO: the following causes errors, when other methotds then predict are called on the regression pipeline
            self.regression_pipeline = clf.best_estimator_
        #print(self.regression_pipeline)


# # Helper functions - distance
# PAIRWISE_KERNEL_FUNCTIONS = {
#     # If updating this dictionary, update the doc in both distance_metrics()
#     # and also in pairwise_distances()!
#     'additive_chi2': additive_chi2_kernel,
#     'chi2': chi2_kernel,
#     'linear': linear_kernel,
#     'polynomial': polynomial_kernel,
#     'poly': polynomial_kernel,
#     'rbf': rbf_kernel,
#     'laplacian': laplacian_kernel,
#     'sigmoid': sigmoid_kernel,
#     'cosine': cosine_similarity, }


# class RBFModel(MetaModel):
#     """
#     Fits an :class:`RBFInterpolator` to the data.
#
#     :param \**kwargs: keyword arguments that will be passed on to the constructor of the nearset neighbors regressor. See
#         http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.interpolate.Rbf.html
#         for more information about.rst available arguments.
#
#     """
#
#     def __init__(self, **kwargs):
#         MetaModel.__init__(self, **kwargs)
#
#     def fit(self, x_train, y_train):
#         """
#         Fits the model to the data such that it reproduces the expected output data.
#
#         :param x_train: array-like of shape [n_samples,n_features]
#
#             training data
#
#         :param y_train: array-like of shape [n_samples, n_targets]
#
#             target values
#
#         """
#         clf = RBFInterpolator()
#         self._update_pipeline_and_fit(x_train, y_train, [clf])
#
#
# class RBFInterpolator():
#     """
#     Uses rbf interpolation to represent a function that maps input values to output values. Manages one regression
#     pipeline for each output. Uses each regression pipeline separately during calls to :meth:`fit` and :meth:`predict`
#     and combines their results as a multitask regressor might do.
#     See
#     http://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html#using-radial-basis-functions-for-smoothing-interpolation
#     for a general explanation of rbf interpolation.
#
#
#     :param \**kwargs: keyword arguments that will be passed on to the constructor of the nearset neighbors regressor. See
#         http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.interpolate.Rbf.html
#         for more information about.rst available arguments.
#
#
#     http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.interpolate.Rbf.html
#     """
#
#     def __init__(self):
#         self.regression_pipelines = []
#
#     def fit(self, x_train, y_train):
#         """
#         For each output separately: Fits a model to the data such that it reproduces the expected output data.
#
#         :param x_train: array-like of shape [n_samples,n_features]
#
#             training data
#
#         :param y_train: array-like of shape [n_samples, n_targets]
#
#             target values
#         """
#
#         num_outputs = y_train.shape[1]
#         for y in range(num_outputs):
#             args = []
#             for c in range(x_train.shape[1]):
#                 args.append(x_train[:,c])
#             args.append(y_train[:,y])
#             self.regression_pipelines.append(Rbf(*args))
#             pass
#         pass
#
#     def predict(self, x):
#         x = np.array(x)
#         data = []
#         for row in range(x.shape[0]):
#             result = []
#             for model in self.regression_pipelines:
#                 args = []
#                 for c in range(x.shape[1]):
#                     args.append(float(x[row,c]))
#                 result.append(model(*args))
#             data.append(result)
#         return np.array(data)


class Score(object):
    def __init__(self, type, value):
        self.type = type
        self.value = value


class MetaModelClassFinder(object):
    def __init__(self):
        # TODO: this solution is not optimal if new metamodels must be loaded as well
        self.metamodel_classes = {clazz.__name__: clazz for clazz in MetaModel.__subclasses__()}


if __name__ == '__main__':
    pass