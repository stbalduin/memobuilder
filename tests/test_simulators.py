import mosaik
import memosampler

from memosampler.simulators import SeriesInputProvider, Collector, InputProvider

def test_constant_inputsim_and_collector():
    # prepare mosaik sim config
    sim_config = {
        'InputSim': InputProvider.SIM_CONFIG,
        'CollectorSim': Collector.SIM_CONFIG
    }
    sim_step_size = 300
    sim_steps = 10

    # create world
    world = mosaik.World(sim_config)

    # input simulator(s)
    input_values = {'a': 5., 'b': 3.}
    map_of_input_entities = create_constant_input_providers(world, input_values, sim_step_size)
    full_ids = {k:entity.full_id for k,entity in map_of_input_entities.items()}

    # data collector
    collector_sim = world.start('CollectorSim', step_size=sim_step_size)
    monitor = collector_sim.Monitor()

    # connect entities
    for attr, input_entity in map_of_input_entities.items():
        world.connect(input_entity, monitor, ('value', attr))

    # Run simulation
    world.run(until=sim_step_size * sim_steps)
    monitored_data = collector_sim.get_monitored_data()
    print(monitored_data)

    for key, id in full_ids.items():
        data = monitored_data[id][key]
        # assert that length of the collected data is as expected
        assert len(data) == sim_steps
        # assert that the value is constant and the one given to the input provider
        assert all([val==input_values[key] for val in data])

def create_constant_input_providers(world, map_of_input_series, step_size):
    data_sim = world.start('InputSim', step_size=step_size)
    entities = {}
    for attr, value in map_of_input_series.items():
        entities[attr] = data_sim.InputProvider(value=value)
    return entities


def test_series_inputsim_and_collector():
    # prepare mosaik sim config
    sim_config = {
        'InputSim': SeriesInputProvider.SIM_CONFIG,
        'CollectorSim': Collector.SIM_CONFIG
    }
    sim_step_size = 300
    sim_steps = 10

    # create world
    world = mosaik.World(sim_config)

    # input simulator(s)
    input_values = {'a': list(range(10)), 'b': list(range(10, 30, 2))}
    map_of_input_entities = create_series_input_providers(world, input_values, sim_step_size)
    full_ids = {k:entity.full_id for k,entity in map_of_input_entities.items()}

    # data collector
    collector_sim = world.start('CollectorSim', step_size=sim_step_size)
    monitor = collector_sim.Monitor()

    # connect entities
    for attr, input_entity in map_of_input_entities.items():
        world.connect(input_entity, monitor, ('value', attr))

    # Run simulation
    world.run(until=sim_step_size * sim_steps)
    monitored_data = collector_sim.get_monitored_data()

    for key, id in full_ids.items():
        data = monitored_data[id][key]
        # assert that length of the collected data is as expected
        assert len(data) == sim_steps
        # assert that the value is constant and the one given to the input provider
        assert all(returned==expected for returned,expected in zip(data,input_values[key]))


def create_series_input_providers(world, map_of_input_series, step_size):
    data_sim = world.start('InputSim', step_size=step_size)
    entities = {}
    for attr, series in map_of_input_series.items():
        entities[attr] = data_sim.SeriesInputProvider(series=series)
    return entities


if __name__ == '__main__':
    test_series_inputsim_and_collector()