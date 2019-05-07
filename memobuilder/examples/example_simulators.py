# TODO: all examples are deprecated


"""
Example of how to use InputProvider and Collector simulators in a mosaik scenario.
"""

import mosaik

from memosampler.simulators import InputProvider
from memosampler.simulators import Collector
from memosampler.simulators import TestSimulator

if __name__ == '__main__':

    sim_config = {
        'InputProvider': InputProvider.SIM_CONFIG,
        'Collector': Collector.SIM_CONFIG,
        'TestSim': TestSimulator.SIM_CONFIG,
    }
    step_size = 60
    duration = step_size * 10

    world = mosaik.World(sim_config)

    # create a constant input provider
    input_provider_sim = world.start('InputProvider', step_size=step_size)
    params = {'value': -7}
    input_provider = input_provider_sim.InputProvider(**params)

    # create a monitor
    collector_sim = world.start('Collector', step_size=step_size)
    monitor = collector_sim.Monitor()

    # create a test simulator
    test_sim = world.start('TestSim', step_size=step_size)
    test_model = test_sim.TestModel(a=1, b=2)

    # connect entities:
    world.connect(input_provider, test_model, ('value', 'in'))
    world.connect(test_model, monitor, 'out')
    world.connect(input_provider, monitor, ('value', 'in') )

    # run the simulation :
    world.run(until=duration)

    # print monitored data
    print(collector_sim.get_monitored_data())