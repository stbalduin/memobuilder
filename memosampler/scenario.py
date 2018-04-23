"""
This module facilitates the sampling process by encapsulating the setup of 
a mosaik sampling scenario. The class :class:`SamplingScenario` is responsible
for the creation of a mosaik world and populating it with simulators. The class
:class:`SourceSimulator` represents simulators which provide input data at simulation time and
the :class:`SinkSimulator` represents a storage for output data that is also generated
at simulation time.     


Overview
========

TODO
"""

import copy
import pandas

import mosaik

from mosaik.scenario import ModelMock

from memosampler.simulators import Collector
from memosampler.simulators import InputProvider


STEP_SIZE = 1
VERBOSE = True


class SamplingScenario(object):
    EXPERIMENT_SIM_NAME = 'SIM'
    """ Name for the simulator under investigation in the mosaik scenario"""

    IN_SIM = 'INPUT_SIM'
    """
    Name for the simulator that provides inputs for the simulator under
    investigation in the mosaik scenario.
    """

    OUT_SIM = 'COLLECTOR_SIM'
    """
    Name for the simulator that collects outputs of the simulator under
    investigation in the mosaik scenario.
    """

    EXPERIMENT_SIM_CONFIG = {
        OUT_SIM: Collector.SIM_CONFIG,  # simulator that gathers outputs
        IN_SIM: InputProvider.SIM_CONFIG,  # simulators that provide constant inputs
        EXPERIMENT_SIM_NAME: None,
    }
    """ Simulator configuration.py for the execution of an experiment in the
    mosaik environment. A copy of this sim_config will be updated when the
    configuration.py of the simulator under investigation is provided."""

    def __init__(self, sim_config, model_structure):
        self.sim_config = sim_config
        self.model_structure = model_structure

        # add sim config to experiment
        self.sim_config = copy.copy(SamplingScenario.EXPERIMENT_SIM_CONFIG)  # copy initial sim_config
        self.sim_config[SamplingScenario.EXPERIMENT_SIM_NAME] = sim_config  # add sim_config

        self.sim_name = SamplingScenario.EXPERIMENT_SIM_NAME  # name of the simulator

        self.input_adapter = MosaikInputSimulatorAdapter()
        self.collector_adapter = MosaikOutputCollectorAdapter()
        self.model_adapter = MosaikSimulatorAdapter()

    def simulate(self, parameter_assignments, duration=STEP_SIZE):
        print('SIMULATE with parameters %s' % (parameter_assignments))

        # compose scenario
        world = mosaik.World(self.sim_config)

        # create input simulator
        self.input_adapter.init(world, self.model_structure)
        self.input_adapter.mosaik_init(parameter_assignments)
        self.input_adapter.mosaik_create(parameter_assignments)

        # create a data collector
        self.collector_adapter.init(world, self.model_structure)
        self.collector_adapter.mosaik_init(parameter_assignments)
        self.collector_adapter.mosaik_create(parameter_assignments)

        # create the model under investigation

        self.model_adapter.init(world, self.model_structure)
        self.model_adapter.mosaik_init(parameter_assignments)
        self.model_adapter.mosaik_create(parameter_assignments)

        # connect inputs
        self.connect_inputs(world, self.input_adapter, self.model_adapter)

        # connect outputs
        self.connect_outputs(world, self.collector_adapter, self.model_adapter)

        # simulate
        world.run(until=duration)

        # collect data
        model_entity = self.model_adapter.model_entity
        output_data = self.collector_adapter.get_data()
        return output_data[model_entity.full_id]

    def connect_inputs(self, world, input_simulator, model_simulator):
        model_entity = model_simulator.model_entity
        for input_name, input_sim in input_simulator.input_entities.items():
            world.connect(input_sim, model_entity, ('value', input_name))

    def connect_outputs(self, world, collector, model_simulator):
        collector_entity = collector.collector_entity
        model_entity = model_simulator.model_entity
        model_outputs = self.model_structure.model_outputs
        for output_name in model_outputs:
            world.connect(model_entity, collector_entity, output_name)


class MosaikInputSimulatorAdapter(object):
    def __init__(self):
        self.world = None
        self.model_structure = None
        self.simulator = None
        self.input_entities = {}

    def init(self, world, model_structure):
        self.world = world
        self.model_structure = model_structure

    def mosaik_init(self, parameter_assignments):
        self.simulator = self.world.start(SamplingScenario.IN_SIM, step_size=STEP_SIZE)

    def mosaik_create(self, parameter_assignments):
        inputs = {attr_name: parameter_assignments[attr_name] for attr_name in self.model_structure.model_inputs}
        input_values = self.preprocess_input_values(inputs, parameter_assignments, self.model_structure)
        for attr_name, attr_val in input_values.items():
            params = {'value': attr_val}
            sim = self.simulator.InputProvider(**params)
            self.input_entities[attr_name] = sim

    def preprocess_input_values(self, input_values, parameters, model_structure):
        return input_values


class MosaikOutputCollectorAdapter(object):
    def __init__(self):
        self.world = None
        self.model_structure = None
        self.simulator = None
        self.collector_entity = None

    def init(self, world, model_structure):
        self.world = world
        self.model_structure = model_structure

    def mosaik_init(self, parameter_assignments):
        self.simulator = self.world.start(SamplingScenario.OUT_SIM, step_size=STEP_SIZE)

    def mosaik_create(self, parameter_assignments):
        self.collector_entity = self.simulator.Monitor()

    def get_data(self):
        return self.simulator.get_monitored_data()

    def postprocess_output_values(self, output_values, parameters, model_structure):
        raise Exception('not yet implemented')


class MosaikSimulatorAdapter(object):
    def __init__(self):
        self.world = None
        self.model_structure = None
        self.simulator = None
        self.model_entity = None

    def init(self, world, model_structure):
        self.world = world
        self.model_structure = model_structure

    def mosaik_init(self, parameter_assignments):
        sim_params = {param: parameter_assignments[param] for param in self.model_structure.simulator_parameters}
        self.__fix_step_size_datatype(sim_params)
        sim_param_values = self.preprocess_simulator_parameters(sim_params, parameter_assignments, self.model_structure)
        self.simulator = self.world.start(SamplingScenario.EXPERIMENT_SIM_NAME, **sim_param_values)

    def mosaik_create(self, parameter_assignments):
        model_params = {param: parameter_assignments[param] for param in self.model_structure.model_parameters}
        model_param_values = self.preprocess_model_parameters(model_params, parameter_assignments, self.model_structure)

        # find the 'ModelMock' object, which allows access to the sim
        mocks = [m for a, m in self.simulator.__dict__.items() if type(m) == ModelMock]
        if len(mocks) > 1:
            raise Exception("Too many matching ModelMocks")
        mock = mocks[0]
        self.model_entity = mock(**model_param_values)
        #model_factory = getattr(self.simulator, modelname)
        #self.model_entity =

    def preprocess_simulator_parameters(self, sim_params, parameters, model_structure):
        return sim_params

    def preprocess_model_parameters(self, model_params, parameters, model_structure):
        return model_params

    def __fix_step_size_datatype(self, sim_param_assignments):
        # TODO: mosaik does not accept 'numpy.int64' as type of the next_step value, which is returned by simulators after each step
        # TODO: UGLY solution here is to just convert type of step_size parameters, if present:
        if 'step_size' in sim_param_assignments:
            sim_param_assignments['step_size'] = int(sim_param_assignments['step_size'])


if __name__ == '__main__':
    pass
