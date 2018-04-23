import random
import copy

import memomodel

from h5db.core import H5AccessMode


def export_model_to_hdf(target_file, model_structure, metamodel):

    linear_models = [
        # auskommentiert, damit alle mit dem genersischen export behandelt werden
        #'OLSModel',
        #'LassoModel',
        #'ElasticNetModel',
        #'RidgeRegressionModel',
    ]

    krr_models = [
        # auskommentiert, damit alle mit dem genersischen export behandelt werden
        #'KernelRidgeRegressionCV',
    ]

    model_type = metamodel.__class__.__name__
    if model_type in linear_models:
        raise Exception('umgebaut, damit alle mit dem genersischen export behandelt werden')
        pipeline = metamodel.regression_pipeline
        name, estimator = pipeline.steps[-1]
        export_linear_model_to_hdf(target_file, model_structure, estimator)
    elif model_type in krr_models:
        raise Exception('umgebaut, damit alle mit dem genersischen export behandelt werden')
        pipeline = metamodel.regression_pipeline
        estimator = pipeline
        export_kernel_ridge_regression_model_to_hdf(target_file, model_structure, estimator)
    else:
        pipeline = metamodel.regression_pipeline
        estimator = pipeline
        export_generic_sklearn_model_to_hdf(target_file, model_structure, estimator)
        print('WARNING: using generic export morde for models of type "%s" '%(str(type(metamodel))))


def export_generic_sklearn_model_to_hdf(target_file, model_structure, estimator):
    model_structure = _copy_model_structure(model_structure)

    generic_model = memomodel.GenericModelDescription(sklearn_estimator=estimator)

    simmodel_description = memomodel.SimulationModelDescription(
        model_structure=model_structure,
        regression_model=generic_model)

    db = memomodel.MeMoSimDB(target_file)
    db.open(access_mode=H5AccessMode.WRITE_TRUNCATE_ON_EXIST)
    db.save_object(simmodel_description)
    db.close()


def export_linear_model_to_hdf(target_file, model_structure, estimator):
    model_structure = _copy_model_structure(model_structure)

    ols_model = memomodel.OLSModelDescription(
        intercept=estimator.intercept_,
        coefs=estimator.coef_)

    simmodel_description = memomodel.SimulationModelDescription(
        model_structure=model_structure,
        regression_model=ols_model)

    db = memomodel.MeMoSimDB(target_file)
    db.open(access_mode=H5AccessMode.WRITE_TRUNCATE_ON_EXIST)
    db.save_object(simmodel_description)
    db.close()


def export_kernel_ridge_regression_model_to_hdf(target_file, model_structure, estimator):
    model_structure = _copy_model_structure(model_structure)

    krr_model = memomodel.KernelRidgeRegressionModelDescription(
        kernel=estimator.kernel,
        gamma=estimator.gamma,
        degree=estimator.degree,
        coef0=estimator.coef0,
        X_fit=estimator.X_fit_,
        dual_coef=estimator.dual_coef_
    )

    simmodel_description = memomodel.SimulationModelDescription(
        model_structure=model_structure,
        regression_model=krr_model)

    db = memomodel.MeMoSimDB(target_file)
    db.open(access_mode=H5AccessMode.WRITE_TRUNCATE_ON_EXIST)
    db.save_object(simmodel_description)
    db.close()


def _copy_model_structure(model_structure):
    model_structure = copy.copy(model_structure)
    model_structure.ID = None
    for vstate in model_structure.virtual_states:
        vstate.ID = None
    return model_structure
