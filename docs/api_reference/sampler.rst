========================
``memosampler.sampler``
========================

.. automodule:: memosampler.sampler


Inheritance diagram
===================

.. inheritance-diagram:: memosampler.sampler
    :parts: 1


API reference
=============

.. autoclass:: SamplingModel
    :members:
    :show-inheritance:

    .. uml::

        class SamplingModel{
           + simulator_params: ModelElement[0..*]
           + model_params: ModelElement[0..*]
           + model_inputs: ModelElement[0..*]
           + model_outputs: String[0..*]
           + <<create>> SamplingModel()
           + add_simulator_param(name: String, variability: Variability)
           + add_model_param(name: String, variability: Variability)
           + add_model_input(name: String, variability: Variability)
           + add_model_output(name: String)
           + add_model_state(name: String, variability: Variability)
           + get_static_model_elements(): ModelElement[0..*]
           + get_variable_model_elements(): ModelElement[0..*]
           + get_outputs(): String[0..*]
           + is_simulator_param(name: String): boolean
           + is_model_param(name: String):: boolean
           + is_model_input(name: String: boolean
           + is_model_output(name: String): boolean
        }



.. autoclass:: ModelElement
    :members:

    .. uml::

        class ModelElement{
           + name: str
           + variability: Variability
           + <<create>> ModelElement(name: str, variability: Variability)
        }


.. autoclass:: Variability
    :members:

    .. uml::

        abstract class Variability{
            + is_constant()
        }


.. autoclass:: Constant
    :members:

    .. uml::

        class Constant{
            + value: object
            + <<create>> Constant(value: object)
        }



.. autoclass:: RangeOfIntegers
    :members:

    .. uml::

        class RangeOfIntegers{
            + min:int
            + max:int
            + <<create>> RangeOfIntegers(min:int , max:int)
        }


.. autoclass:: RangeOfRealNumbers
    :members:

    .. uml::

        class RangeOfRealNumbers{
            + min:float
            + max:float
            + <<create>> RangeOfRealNumbers(min:float , max:float)
        }


.. autoclass:: NumericalLevels
    :members:

    .. uml::

        class NumericalLevels{
            + levels: float[2..*]
            + <<create>> NumericalLevels(levels: float[2..*])
        }


.. autoclass:: NonNumericalLevels
    :members:

    .. uml::

        class NonNumericalLevels{
            + levels: object[2..*]
            + <<create>> NumericalLevels(levels: object[2..*])
        }

        
.. autoclass:: SamplingStrategy
    :members:

    .. uml::

        abstract class SamplingStrategy{
            + sampling_model: SamplingModel
            + sampling_scenario: SamplingScenario
            + <<create>> SamplingStrategy(sampling_model: SamplingModel, sampling_scenario: SamplingScenario)
            + {abstract} run(**kwargs):
        }


.. autoclass:: LHS
    :members:

    .. uml::

        class LHS{
            + sampling_model: SamplingModel
            + sampling_scenario: SamplingScenario
            + <<create>> LHS(sampling_model: SamplingModel, sampling_scenario: SamplingScenario)
            + run(num_samples: int, **kwargs):
        }


Examples
========