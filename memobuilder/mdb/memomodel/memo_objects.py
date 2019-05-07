"""This module contains memo objects for the yaml configuration."""
import collections
import enum

import numpy
import pandas
import yaml
from yaml import Loader

from memobuilder.mdb import h5db


class SimConfig(h5db.H5DBObject, yaml.YAMLObject):
    yaml_loader = Loader
    yaml_tag = u'!SimConfig'
    arguments = h5db.ObjectList


class VirtualState(h5db.H5DBObject, yaml.YAMLObject):
    yaml_loader = Loader
    yaml_tag = u'!VirtualState'
    name = h5db.Scalar
    update_attribute = h5db.Scalar
    init_attribute = h5db.Scalar


class ModelStructure(h5db.H5DBObject, yaml.YAMLObject):
    yaml_loader = Loader
    yaml_tag = u'!ModelStructure'
    simulator_parameters = h5db.List
    model_parameters = h5db.List
    model_inputs = h5db.List
    model_outputs = h5db.List
    virtual_states = h5db.ObjectList


class SamplerConfig(h5db.H5DBObject, yaml.YAMLObject):
    yaml_loader = Loader
    yaml_tag = '!SamplerConfig'
    name = h5db.Scalar
    strategy = h5db.Object
    sim_config = h5db.Object
    model_structure = h5db.Object
    parameter_variations = h5db.ObjectList


class ParameterVariation(h5db.H5DBObject, yaml.YAMLObject):
    yaml_loader = Loader
    yaml_tag = '!ParameterVariation'
    parameter_name = h5db.Scalar
    variation_mode = h5db.Scalar
    variation_arguments = h5db.ObjectList


class ParameterVariationMode(enum.Enum):
    CONSTANT = 'constant'
    # supported by LHS sampling strategy
    RANGE_OF_REAL_NUMBERS = 'range_of_real_numbers'
    # not supported by a sampling strategy yet
    RANGE_OF_INTEGERS = 'range_of_integers'
    # supported by FullFactorial sampling strategy
    NUMERICAL_LEVELS = 'numerical_levels'
    # supported by FullFactorial sampling strategy
    NOMINAL_LEVELS = 'nominal_levels'


class StrategyConfig(h5db.H5DBObject, yaml.YAMLObject):
    yaml_loader = Loader
    yaml_tag = u'!StrategyConfig'
    name = h5db.Scalar
    arguments = h5db.ObjectList


class KeyValuePair(h5db.H5DBObject, yaml.YAMLObject):
    yaml_loader = Loader
    yaml_tag = u'!KeyValuePair'
    key = h5db.Scalar
    value = h5db.Scalar


class KeyListPair(h5db.H5DBObject, yaml.YAMLObject):
    yaml_loader = Loader
    yaml_tag = u'!KeyListPair'
    key = h5db.Scalar
    value = h5db.List


class KeyObjectPair(h5db.H5DBObject, yaml.YAMLObject):
    yaml_loader = Loader
    yaml_tag = u'!KeyObjectPair'
    key = h5db.Scalar
    value = h5db.Object


class ApproximationFunctionConfig(h5db.H5DBObject, yaml.YAMLObject):
    yaml_loader = Loader
    yaml_tag = '!ApproximationFunctionConfig'
    inputs = h5db.List
    outputs = h5db.List
    model_type = h5db.Scalar
    model_arguments = h5db.ObjectList
    trainer_options = h5db.ObjectList
    trainer_score = h5db.List


class SurrogateModelConfig(h5db.H5DBObject, yaml.YAMLObject):
    yaml_loader = Loader
    yaml_tag = '!SurrogateModelConfig'
    name = h5db.Scalar
    approximation_functions = h5db.ObjectList
    sampler_configuration = h5db.Object


class DatasetOwnership(h5db.H5DBObject):
    dataset = h5db.Object
    owner = h5db.Object

    def __init__(self, **kwargs):
        self.dataset = None
        self.owner = None
        h5db.H5DBObject.__init__(self, **kwargs)


class InputResponseDataset(h5db.H5DBObject):
    inputs = h5db.DataFrame
    responses = h5db.DataFrame

    def __init__(self, input_cols=[], response_cols=[]):
        """
        :param input_cols: List of input column names
        :param response_cols: List of response column names
        """
        # initialize empty dataframes
        self.inputs = pandas.DataFrame(columns=input_cols,
                                       dtype=numpy.float64)
        self.responses = pandas.DataFrame(columns=response_cols,
                                          dtype=numpy.float64)
        h5db.H5DBObject.__init__(self)

    def update(self, sample, response):
        next_idx = self.inputs.shape[0]
        # update inputs
        self.inputs.loc[next_idx, :] = sample
        # update responses
        stripped_response = {key: values[0]
                             for key, values in response.items()}
        self.responses.loc[next_idx, :] = stripped_response

    def select(self, selected_inputs=[], selected_responses=[]):
        result = InputResponseDataset()

        result.inputs = self.inputs[selected_inputs]
        result.responses = self.responses[selected_responses]
        return result

    def __len__(self):
        return self.inputs.shape[0]

    def __repr__(self):
        return str(self.__dict__)


class TrainingResult(h5db.H5DBObject):
    train_data = h5db.Object
    test_data = h5db.Object
    metamodel = h5db.Blob
    score_r2 = h5db.Scalar
    score_mae = h5db.Scalar
    score_hae = h5db.Scalar
    score_mse = h5db.Scalar

    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.metamodel = None
        # score_r2 = h5db.Scalar
        # score_avg = h5db.Scalar
        # score_hae = h5db.Scalar
        # score_mse = h5db.Scalar
        h5db.H5DBObject.__init__(self)


class SurrogateModelTrainingResult(h5db.H5DBObject):
    training_results = h5db.ObjectList
    surrogate_model_name = h5db.Scalar

    def __init__(self):
        self.training_results = None
        self.surrogate_model_name = None
        h5db.H5DBObject.__init__(self)


class SurrogateModel(h5db.H5DBObject):
    name = h5db.Scalar
    metamodels = h5db.Blob
    model_structure = h5db.Object

    def __init__(self):
        self.surrogate_model_name = None
        self.metamodel = None
        self.model_structure = None
        h5db.H5DBObject.__init__(self)


class SimulationModelDescription(h5db.H5DBObject):
    model_structure = h5db.Object
    regression_model = h5db.Object

    def __init__(self, *args, **kwargs):
        self.model_structure = None
        self.regression_model = None
        h5db.H5DBObject.__init__(self, *args, **kwargs)


class GenericModelDescription():
    sklearn_estimator = h5db.Blob

    def __init__(self, *args, **kwargs):
        self.sklearn_estimator = None
        h5db.H5DBObject.__init__(self, *args, **kwargs)


class OLSModelDescription(h5db.H5DBObject):
    """TODO: Refactor this to LinearModelDescription"""
    intercept = h5db.Vector
    coefs = h5db.Matrix

    def __init__(self, *args, **kwargs):
        self.intercept = None
        self.coefs = None
        h5db.H5DBObject.__init__(self, *args, **kwargs)


class KernelRidgeRegressionModelDescription(h5db.H5DBObject):
    kernel = h5db.Scalar
    gamma = h5db.Scalar
    degree = h5db.Scalar
    coef0 = h5db.Scalar
    X_fit = h5db.Matrix
    dual_coef = h5db.Matrix

    def __init__(self, *args, **kwargs):
        self.intercept = None
        self.coefs = None
        h5db.H5DBObject.__init__(self, *args, **kwargs)


class MeMoSimDB(h5db.H5DB):
    def __init__(self, h5filename):
        mapped_classes = [SimulationModelDescription, OLSModelDescription,
                          GenericModelDescription,
                          KernelRidgeRegressionModelDescription,
                          ModelStructure, VirtualState]
        h5db.H5DB.__init__(self, h5filename, mapped_classes)


class MeMoDB(h5db.H5DB):
    def __init__(self, h5filename):
        mapped_classes = [SimConfig, ModelStructure, VirtualState,
                          SamplerConfig, ParameterVariation,
                          StrategyConfig, KeyValuePair, SurrogateModelConfig,
                          ApproximationFunctionConfig, InputResponseDataset,
                          TrainingResult, SurrogateModelTrainingResult,
                          DatasetOwnership, SurrogateModel]
        h5db.H5DB.__init__(self, h5filename, mapped_classes)
