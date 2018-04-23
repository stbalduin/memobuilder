==========================
``memotrainer.metamodels``
==========================

.. automodule:: memotrainer.metamodels


Inheritance diagram
===================

.. inheritance-diagram:: memotrainer.metamodels
    :parts: 1


API reference
=============

.. autoclass:: Trainer
    :members:

    .. uml::

        class Trainer{
            + test_ratio: float
            + x_train: DataFrame
            + x_test: DataFrame
            + y_train: DataFrame
            + y_test: DataFrame
            + input_names: str[1..*]
            + output_names: str[1..*]
            + input_transform: data transformation
            + output_transform: data transformation
            + <<create>> Trainer(input_data: DataFrame, output_data: DataFrame, input_names: str[1..*], output_names: str[1..*], test_ratio: float)
            + fit(approximation_model: ApproximationMddel)
            + mean_squared_error(approximation_model: ApproximationMddel)
            + mean_absolute_error(approximation_model: ApproximationMddel)
            + explained_variance_score(approximation_model: ApproximationMddel)
            + r2_score(approximation_model: ApproximationMddel)
            + max_absolute_componentwise_error(approximation_model: ApproximationMddel)
        }

.. autoclass:: OutputTransformer
    :members:

    .. uml::

        class OutputTransformer{
            + approximation_model: MetaModel
            + transform: data transformation
            + <<create>> OutputTransformer(approximation_model: MetaModel, transform: data transformation)
            + predict(X: array-like)
        }

.. autoclass:: MetaModel
    :members:

    .. uml::

        class MetaModel{
            + regression_pipeline: Pipeline
            + kwargs: dictionary
            + input_names: str[1..*]
            + output_names: str[1..*]
            + processing_steps: data transformation [0..*]
            + fit(X: array-like, Y: array-like)
            + predict(X: array-like)
        }

.. autoclass:: OLSModel
    :members:

    .. uml::

        class OLSModel{
            + fit(X: array-like, Y: array-like)
            + predict(X: array-like)
        }

.. autoclass:: LassoModel
    :members:

    .. uml::

        class LassoModel{
            + fit(X: array-like, Y: array-like)
            + predict(X: array-like)
        }

.. autoclass:: ElasticNetModel
    :members:

    .. uml::

        class ElasticNetModel{
            + fit(X: array-like, Y: array-like)
            + predict(X: array-like)
        }

.. autoclass:: KrigingModel
    :members:

    .. uml::

        class KrigingModel{
            + fit(X: array-like, Y: array-like)
            + predict(X: array-like)
        }

.. autoclass:: KNeighborsModel
    :members:

    .. uml::

        class KNeighborsModel{
            + fit(X: array-like, Y: array-like)
            + predict(X: array-like)
        }

.. autoclass:: RBFModel
    :members:

    .. uml::

        class RBFModel{
            + fit(X: array-like, Y: array-like)
            + predict(X: array-like)
        }

.. autoclass:: RBFInterpolator
    :members:

    .. uml::

        class RBFInterpolator{
            + fit(X: array-like, Y: array-like)
            + predict(X: array-like)
        }