========================
``memosampler.scenario``
========================

.. automodule:: memosampler.scenario

Inheritance diagram
===================

.. inheritance-diagram:: memosampler.scenario
    :parts: 1



API reference
=============

.. autoclass:: SamplingScenario
    :members:

    .. uml::

        class SamplingScenario {
            + {static} EXPERIMENT_SIM_NAME: str
            + {static} IN_SIM: str
            + {static} OUT_SIM:str
            + {static} EXPERIMENT_SIM_CONFIG: dict<str,dict<str,str>>
            + sim_name: str
            + sim_config: dict<str,dict<str,str>>
            + static_sim_params: dict<str,object>
            + static_model_params: dict<str,object>
            + world: mosaik.World
            + data: dict<str,object>
            + <<create>> SamplingScenario(sim_config: dict<str,str>, sampling_model: SamplingModel)
            + prepare_create_arguments()
            + prepare_init_arguments()
            + simulate()
        }

Examples
========


.. code-block:: python

    import memosampler.util
    from memosampler.scenario import SamplingScenario
    from memosampler.simulators import TestSimulator

    if __name__ == '__main__':

        # simulator configuration for mosaik
        sim_config = TestSimulator.SIM_CONFIG

        # create a sampling model with sim_parameter *step_size* and
        # model parameter *b* held constant.
        sampling_model = memosampler.util.create_testsim_sampling_model()

        # create the experiment:
        experiment = SamplingScenario(sim_config, sampling_model)

        # parameter *a* and input *in* are allowed to vary so their values
        # must be specified before we can run the
        # experiment
        model_params = {'a' : 1}
        model_inputs = {'in': 9}
        sim_params = {}  # empty because all simulator parameters are constant
                         # (and already specified in the sampling model)

        # run the experiment:
        data = experiment.simulate(sim_params, model_params, model_inputs)