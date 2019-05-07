import yaml

from memobuilder.mdb import h5db
from memobuilder.mdb.memomodel.memo_objects import (
    ApproximationFunctionConfig, DatasetOwnership, GenericModelDescription,
    InputResponseDataset, KernelRidgeRegressionModelDescription, KeyObjectPair,
    KeyValuePair, MeMoDB, MeMoSimDB, ModelStructure, OLSModelDescription,
    ParameterVariation, ParameterVariationMode, SamplerConfig, SimConfig,
    SimulationModelDescription, StrategyConfig, SurrogateModel,
    SurrogateModelConfig, SurrogateModelTrainingResult, TrainingResult,
    VirtualState)

h5db.core.register_h5db_yaml_objects()
