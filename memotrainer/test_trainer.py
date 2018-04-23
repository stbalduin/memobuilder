import numpy as np
import pandas as pd
import sklearn.datasets
import memotrainer.metamodels as metamodels

from sklearn.preprocessing.data import MinMaxScaler, StandardScaler, PolynomialFeatures, Normalizer
from memomodel.memo_objects import InputResponseDataset
from memotrainer.trainer import Trainer


def load_boston():
    boston = sklearn.datasets.load_boston()
    dataset = InputResponseDataset()
    dataset.inputs = pd.DataFrame(boston.data, columns=boston.feature_names)
    dataset.responses = pd.DataFrame(boston.target, columns=['MEDV'])
    return dataset


def create_simple_dataset():
    dataset = InputResponseDataset(input_cols=['IN1'], response_cols=['OUT1'])
    for num in np.arange(0, 100., 1.0):
        dataset.update({'IN1': num}, {'OUT1': [num ** 2]})
    return dataset


def test_kriging_with_boston_dataset():

    # load boston dataset
    dataset = load_boston()

    # create metamodel
    input_names = list(dataset.inputs)
    response_names = list(dataset.responses)
    metamodel = metamodels.KrigingModel(preprocessors=[], input_names=input_names, response_names=response_names)

    # create trainer and fit metamodel to the dataset
    result = Trainer().fit(metamodel, dataset)

    print('score:', result.score)

    # score: 0.787001297651
    assert result.score > 0.78
    assert result.score < 0.79


def test_kriging_with_simple_dataset():

    # construct a simple datset
    dataset = create_simple_dataset()

    # create metamodel
    input_names = list(dataset.inputs)
    response_names = list(dataset.responses)
    metamodel = metamodels.KrigingModel(preprocessors=[], input_names=input_names, response_names=response_names)

    # create trainer and fit metamodel to the dataset
    result = Trainer().fit(metamodel, dataset)

    print('score:', result.score)

    #score: 0.999950619377
    assert result.score > 0.99994
    assert result.score < 0.99996


def test_ols_with_simple_dataset():

    # construct a simple datset
    dataset = create_simple_dataset()

    # create metamodel
    input_names = list(dataset.inputs)
    response_names = list(dataset.responses)
    metamodel = metamodels.OLSModel(preprocessors=[], input_names=input_names, response_names=response_names)

    # create trainer and fit metamodel to the dataset
    result = Trainer().fit(metamodel, dataset)

    print('score:', result.score)

    # score: 0.946672107012
    assert result.score > 0.94
    assert result.score < 0.95


def test_ols_with_boston_dataset():

    # load boston dataset
    dataset = load_boston()

    # create metamodel
    input_names = list(dataset.inputs)
    response_names = list(dataset.responses)
    metamodel = metamodels.OLSModel(preprocessors=[PolynomialFeatures(degree=2)], input_names=input_names, response_names=response_names)

    # create trainer and fit metamodel to the dataset
    result = Trainer().fit(metamodel, dataset)

    print('score:', result.score)

    # score: 0.682539990982
    assert result.score > 0.68
    assert result.score < 0.69


def test_lasso_with_simple_dataset():

    # construct a simple datset
    dataset = create_simple_dataset()

    # create metamodel
    input_names = list(dataset.inputs)
    response_names = list(dataset.responses)
    metamodel = metamodels.LassoModel(preprocessors=[], input_names=input_names, response_names=response_names, cv=5)

    # create trainer and fit metamodel to the dataset
    result = Trainer().fit(metamodel, dataset)

    print('score:', result.score)

    # score: 0.946672107012
    assert result.score > 0.94
    assert result.score < 0.95


def test_lasso_with_boston_dataset():

    # load boston dataset
    dataset = load_boston()

    # create metamodel
    input_names = list(dataset.inputs)
    response_names = list(dataset.responses)
    metamodel = metamodels.LassoModel(preprocessors=[], input_names=input_names,
                                    response_names=response_names, cv=5)

    # create trainer and fit metamodel to the dataset
    result = Trainer().fit(metamodel, dataset)

    print('score:', result.score)

    # score: 0.787001297651
    assert result.score > 0.5
    assert result.score < 0.51

def test_elasticnet_with_simple_dataset():

    # construct a simple datset
    dataset = create_simple_dataset()

    # create metamodel
    input_names = list(dataset.inputs)
    response_names = list(dataset.responses)
    metamodel = metamodels.ElasticNetModel(preprocessors=[], input_names=input_names, response_names=response_names, cv=5)

    # create trainer and fit metamodel to the dataset
    result = Trainer().fit(metamodel, dataset)

    print('score:', result.score)

    # score: 0.939611874972
    assert result.score > 0.93
    assert result.score < 0.94


def test_elasticnet_with_boston_dataset():

    # load boston dataset
    dataset = load_boston()

    # create metamodel
    input_names = list(dataset.inputs)
    response_names = list(dataset.responses)
    metamodel = metamodels.ElasticNetModel(preprocessors=[], input_names=input_names,
                                    response_names=response_names, cv=5)

    # create trainer and fit metamodel to the dataset
    result = Trainer().fit(metamodel, dataset)

    print('score:', result.score)

    # score: 0.477073081177
    assert result.score > 0.47
    assert result.score < 0.48

def test_kneighbors_with_simple_dataset():

    # construct a simple datset
    dataset = create_simple_dataset()

    # create metamodel
    input_names = list(dataset.inputs)
    response_names = list(dataset.responses)
    metamodel = metamodels.KNeighborsModel(preprocessors=[], input_names=input_names, response_names=response_names)

    # create trainer and fit metamodel to the dataset
    result = Trainer().fit(metamodel, dataset)

    print('score:', result.score)

    # score: 0.998574286629
    assert result.score > 0.99
    assert result.score < 0.999


def test_kneighbors_with_boston_dataset():

    # load boston dataset
    dataset = load_boston()

    # create metamodel
    input_names = list(dataset.inputs)
    response_names = list(dataset.responses)
    metamodel = metamodels.KNeighborsModel(preprocessors=[], input_names=input_names, response_names=response_names)

    # create trainer and fit metamodel to the dataset
    result = Trainer().fit(metamodel, dataset)

    print('score:', result.score)

    # score: 0.364547876566
    assert result.score > 0.36
    assert result.score < 0.37

def test_decisiontree_with_simple_dataset():

    # construct a simple datset
    dataset = create_simple_dataset()

    # create metamodel
    input_names = list(dataset.inputs)
    response_names = list(dataset.responses)
    metamodel = metamodels.DecisionTreeRegression(preprocessors=[], input_names=input_names, response_names=response_names)

    # create trainer and fit metamodel to the dataset
    result = Trainer().fit(metamodel, dataset)

    print('score:', result.score)

    # score: 0.998615575778
    assert result.score > 0.998
    assert result.score < 0.999


def test_decisiontree_with_boston_dataset():

    # load boston dataset
    dataset = load_boston()

    # create metamodel
    input_names = list(dataset.inputs)
    response_names = list(dataset.responses)
    metamodel = metamodels.DecisionTreeRegression(preprocessors=[], input_names=input_names,
                                                  response_names=response_names)

    # create trainer and fit metamodel to the dataset
    result = Trainer().fit(metamodel, dataset)

    print('score:', result.score)

    # score: 0.594024623862
    assert result.score > 0.5
    assert result.score < 0.65


def test_kernelridgeregression_with_simple_dataset():

    # construct a simple datset
    dataset = create_simple_dataset()

    # create metamodel
    input_names = list(dataset.inputs)
    response_names = list(dataset.responses)
    metamodel = metamodels.KernelRidgeRegression(preprocessors=[], input_names=input_names,
                                                  response_names=response_names)
    #metamodel = metamodels.DecisionTreeRegression(preprocessors=[], input_names=input_names, response_names=response_names)

    # create trainer and fit metamodel to the dataset
    result = Trainer().fit(metamodel, dataset)

    print('score:', result.score)

    # 0.881282222886
    assert result.score > 0.88
    assert result.score < 0.89


def test_kernelridgeregression_with_boston_dataset():

    # load boston dataset
    dataset = load_boston()

    # create metamodel
    input_names = list(dataset.inputs)
    response_names = list(dataset.responses)
    metamodel = metamodels.KernelRidgeRegression(preprocessors=[], input_names=input_names,
                                                  response_names=response_names)

    # create trainer and fit metamodel to the dataset
    result = Trainer().fit(metamodel, dataset)

    print('score:', result.score)

    # score: 0.556351440256
    assert result.score > 0.55
    assert result.score < 0.56


if __name__ == '__main__':
    test_kriging_with_simple_dataset()
    test_kriging_with_boston_dataset()

    test_ols_with_simple_dataset()
    test_ols_with_boston_dataset()

    test_lasso_with_simple_dataset()
    test_lasso_with_boston_dataset()

    test_elasticnet_with_simple_dataset()
    test_elasticnet_with_boston_dataset()

    # kneighbors
    # decisiontree