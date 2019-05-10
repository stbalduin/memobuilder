"""
This module contains some simulators that are used by other components
of MeMoSampler.


Overview
========

..  uml::

    class memosampler.simulators.InputProvider {
    }

    class memosampler.simulators.Collector {
    }

    class memosampler.simulators.TestSimulator {

    }

* InputProvider is a simulator which provides inputs to the simulator
  under investigation at simulation time. The input value may be of an
  arbitrary complex data type that the simulator under investigation is
  able to process. If simulated over several steps of time, the
  provided value does not change - InputProvider was designed for
  single step experiments.

* Collector simply collects the values of all incoming connections. The
  simulator has an extra function :meth:`get_monitored_data` which
  returns a list of DataFrames at runtime. One for each connected entity.
  If simulated over several steps of time the collector always collects
  all connected values. Thus the collector simulator has been used in
  single step experiments and in scenarios built for the comparison of
  simulator and surrogate simulator.

* TestSimulator is a simulator used in tests and during development of
  InputProvider and Collector simulators.

"""
import collections
import copy
import mosaik_api


class InputProvider(mosaik_api.Simulator):
    """A simulator that can be used to provide constant input values for other
    simulators.

    """
    # Sim config for this simulator for inprocess use.
    SIM_CONFIG = {'python': 'memobuilder.msampler.simulators:InputProvider'}

    # EID prefix for all model instances.
    EID_PREFIX = 'Input_'
    # Mosaik meta data for this simulator.
    META = {
        'models': {
            'InputProvider': {
                'public': True,
                'params': ['value'],
                'attrs': ['value'],
            }
        }
    }

    def __init__(self):
        # Create a copy of global meta_data data
        meta = copy.copy(InputProvider.META)
        # meta_data data
        super().__init__(meta)
        # prefix for eids
        self.eid_prefix = InputProvider.EID_PREFIX
        # step size of the simulation
        self.step_size = None
        # maps eid to value
        self.data = dict()

    def init(self, sid, step_size):
        """
        see :meth:`mosaik_api.Simulator.init()`

        :param sid: str
            id of this simulator

        :param step_size: int
            number of seconds between two simulation steps

        :return:
            meta data describing the simulator
        """
        self.step_size = step_size
        return self.meta

    def create(self, num, model, value):
        """
        see :meth:`mosaik_api.Simulator.create()`

        :param num: int, he number of model instances to create.

        :param model: str, needs to be a public entry in the
        simulator’s meta['models'].

        :param value: object an object which will be returned after
        each call to the method :meth:`get_data` of the new model
        instance.

        :return: Return a (nested) list of dictionaries describing the
        created model instances (entities).

        """
        entities = []
        next_eid = len(self.data)

        for i in range(next_eid, next_eid + num):
            eid = '%s%d' % (self.eid_prefix, i)
            # No model needs to be created. just store the value in a
            # dict which maps eids to the given values
            self.data[eid] = value
            entities.append({'eid': eid, 'type': model})
        return entities

    def step(self, time, inputs):
        """
        see :meth:`mosaik_api.Simulator.step()`

        :param time: int, seconds since simulation start.

        :param inputs: a dict of dicts mapping entity IDs to attributes
        and dicts of values (each simulator has do decide on its own
        how to reduce the values (e.g., as its sum, average or maximum).
        This simulator ignores inputs.

        :return: int, time of the next simulation step (also in seconds
        since simulation start)

        :raises: Exception, if *inputs* is not empty.

        """
        # noting to do: we only want to provide constant values.
        if inputs:
            raise Exception("There should be no inputs.")
        return time + self.step_size

    def get_data(self, outputs):
        """
        see :meth:`mosaik_api.Simulator.get_data()`

        :param outputs: outputs is a dict mapping entity IDs to lists
        of attribute names whose values are requested.

        :return: The return value needs to be a dict of dicts mapping
        entity IDs and attribute names to their values.

        """
        new_data = {}
        for eid, value in self.data.items():
            new_data[eid] = {'value': value}
        return new_data


class SeriesInputProvider(mosaik_api.Simulator):
    """A simulator that can be used to provide constant input values
    for other simulators.

    """
    # Sim config for this simulator for inprocess use.
    SIM_CONFIG = {'python': 'memobuilder.msampler.simulators:'
                            'SeriesInputProvider'}
    # EID prefix for all model instances.
    EID_PREFIX = 'SeriesInput_'
    # Mosaik meta data for this simulator.
    META = {
        'models': {
            'SeriesInputProvider': {
                'public': True,
                'params': ['series'],
                'attrs': ['value'],
            }
        }
    }

    def __init__(self):
        # create a copy of global meta_data data
        meta = copy.copy(SeriesInputProvider.META)
        super().__init__(meta)  # meta_data data
        self.eid_prefix = SeriesInputProvider.EID_PREFIX  # prefix for eids
        self.step_size = None  # step size of the simulation
        self.series = dict()  # maps eid to remaining series data

    def init(self, sid, step_size):
        """
        see :meth:`mosaik_api.Simulator.init()`

        :param sid: str
            id of this simulator

        :param step_size: int
            number of seconds between two simulation steps

        :return:
            meta data describing the simulator
        """
        self.step_size = step_size
        return self.meta

    def create(self, num, model, series):
        """
        see :meth:`mosaik_api.Simulator.create()`

        :param num: int, the number of model instances to create.

        :param model: str, needs to be a public entry in the
        simulator’s meta['models'].

        :param value: object, an object which will be returned after
        each call to the method :meth:`get_data` of the new model instance.

        :return: Return a (nested) list of dictionaries describing the
        created model instances (entities).
        """
        entities = []
        next_eid = len(self.series)

        for i in range(next_eid, next_eid + num):
            eid = '%s%d' % (self.eid_prefix, i)
            # No model needs to be created. just store the value in a
            # dict which maps eids to the given values
            self.series[eid] = copy.copy(series)
            entities.append({'eid': eid, 'type': model})
        return entities

    def step(self, time, inputs):
        """
        see :meth:`mosaik_api.Simulator.step()`

        :param time: int, seconds since simulation start.

        :param inputs: a dict of dicts mapping entity IDs to attributes
        and dicts of values (each simulator has do decide on its own
        how to reduce the values (e.g., as its sum, average or maximum).
        This simulator ignores inputs.

        :return: int, time of the next simulation step (also in seconds
        since simulation start)

        :raises: Exception ,if *inputs* is not empty.

        """
        # Noting to do: we only want to provide constant values.
        if inputs:
            raise Exception("There should be no inputs.")
        return time + self.step_size

    def get_data(self, outputs):
        """
        see :meth:`mosaik_api.Simulator.get_data()`

        :param outputs: outputs is a dict mapping entity IDs to lists
        of attribute names whose values are requested.

        :return: The return value needs to be a dict of dicts mapping
        entity IDs and attribute names to their values.

        """
        data = {}
        for eid in self.series.keys():
            value = self.series[eid].pop(0)
            data[eid] = {'value': value}
        return data


class Collector(mosaik_api.Simulator):
    """
    This simulator provides a data collector which accepts any given
    inputs from other models and collects their values. The
    *extra_method* :meth:`get_monitored_data` provides access to the
    monitored data.

    """
    # Sim config for this simulator for inprocess use.
    SIM_CONFIG = {'python': 'memobuilder.msampler.simulators:Collector'}
    # EID prefix for all model instances.
    EID_PREFIX = 'Sink_'
    # Mosaik meta data for this simulator.
    META = {
        'models': {
            'Monitor': {
                'public': True,
                'params': ['container'],
                'attrs': [],  # will be filled during init step
                'any_inputs': True,
            }
        },
        'extra_methods': ['get_monitored_data']
    }

    def __init__(self):
        # create a copy of global meta_data data
        meta = copy.copy(Collector.META)
        # meta_data data
        super().__init__(meta)
        self.data = collections.defaultdict(lambda:
                                            collections.defaultdict(list))
        self.eid = None
        # step size of the simulation
        self.step_size = None

    def init(self, sid, step_size):
        """
        see :meth:`mosaik_api.Simulator.init()`

        :param sid: str, id of this simulator

        :param step_size: int, number of seconds between two simulation steps

        :return: meta data describing the simulator

        """
        self.step_size = step_size
        return self.meta

    def create(self, num, model, container=dict()):
        """
        see :meth:`mosaik_api.Simulator.create()`

        :param num: int, the number of model instances to create.

        :param model: str, needs to be a public entry in the
        simulator’s meta['models'].

        :param container: dict, an empty dictionary which will be
        filled with data during simulation.

        :return: Return a (nested) list of dictionaries describing
        the created model instances (entities).
        """
        if num > 1 or self.eid is not None:
            raise RuntimeError('Can only create one instance of Monitor.')
        self.eid = 'Monitor'
        return [{'eid': self.eid, 'type': model}]

    def step(self, time, inputs):
        """
        see :meth:`mosaik_api.Simulator.step()`

        :param time: int, seconds since simulation start.

        :param inputs: a dict of dicts mapping entity IDs to attributes
        and dicts of values (each simulator has do decide on its own
        how to reduce the values (e.g., as its sum, average or maximum).

        :return: int, time of the next simulation step (also in seconds
        since simulation start)

        """
        if self.eid in inputs:
            data = inputs[self.eid]
            for attr, values in data.items():
                for src, value in values.items():
                    self.data[src][attr].append(value)
        return time + self.step_size

    def get_data(self, outputs):
        """
        see :meth:`mosaik_api.Simulator.get_data()`

        :param outputs: outputs is a dict mapping entity IDs to lists
        of attribute names whose values are requested.

        :raises: Exception because this simulator is not supposed to be
        connected as input for other simulators.
        """
        # Nothing to do. get data is not supposed to be called.
        raise Exception('Unsupported Operation')

    def get_monitored_data(self):
        """

        :return: a dictionary which maps the full entity ids of source
        simulators to DataFrames containing the monitored data from
        that source.
        """
        # source_ids = list(self.data.keys())
        # output_data = {}
        # for source_id in source_ids:
        #    output_data[source_id] = pandas.DataFrame(self.data[source_id])
        # return output_data
        return self.data


class TestSimulator(mosaik_api.Simulator):
    """
    The test simulator has one input named *in*, one output named *out* and
    two parameters *a* and *b*. In this example *b* is considered to be a
    constant value. *a* and *in* may be varied.
    """

    SIM_CONFIG = {'python': 'memobuilder.msampler.simulators:TestSimulator'}
    """
    Sim config for the test simulator.
    """

    STEP_SIZE = 60 * 15  # 15 minutes
    """
    Step size of the test simulator.
    """

    META = {
        'models': {
            'TestModel': {
                'public': True,
                'params': ['a', 'b'],
                'attrs': ['in', 'out', 'a', 'b'],
            }
        },
        'extra_methods': []
    }
    """
    Mosaik meta data for this simulator.
    """

    EID_PREFIX = 'TestSimulator_'
    """EID prefix for all model instances. """

    def __init__(self):
        self.meta = TestSimulator.META
        super().__init__(self.meta)
        self.models = dict()  # contains the model instances
        self.sid = None
        self.eid_prefix = TestSimulator.EID_PREFIX
        self.step_size = None
        self.data = None
        self.expression = '(%f) * (%f)**(%f)'
        self.a = None
        self.b = None

    def init(self, sid, step_size):
        """
        see :meth:`mosaik_api.Simulator.init()`

        :param sid: str
            id of this simulator

        :param step_size: int
            number of seconds between two simulation steps

        :return:
            meta data describing the simulator
        """
        self.sid = sid  # simulator id
        self.step_size = step_size
        return self.meta

    def create(self, num, model, a, b):
        """
        see :meth:`mosaik_api.Simulator.create()`

        :param num: int
            the number of model instances to create.

        :param model: str
            needs to be a public entry in the simulator’s meta['models'].

        :param a: float
           value of model parameter *a*.

        :param b: float
            value of model parameter *b*.

        :return:
            Return a (nested) list of dictionaries describing the created
            model instances (entities).
        """
        self.a = a
        self.b = b
        entities = []
        next_eid = len(self.models)
        for i in range(next_eid, next_eid + num):
            eid = '%s%d' % (self.eid_prefix, i)
            entities.append({'eid': eid, 'type': model})
        return entities

    def step(self, time, inputs):
        """
        see :meth:`mosaik_api.Simulator.step()`

        :param time: int
            seconds since simulation start.

        :param inputs:  a dict of dicts mapping entity IDs to
        attributes and dicts of values (each simulator has do decide on
        its own how to reduce the values (e.g., as its sum, average or
        maximum).

        :return: int time of the next simulation step (also in seconds
        since simulation start)
        """
        output_data = {}
        for eid, data in inputs.items():
            for attr, attr_data in data.items():
                in_value = list(attr_data.values())[0]
                expression = self.expression % (self.a, in_value, self.b)
                # print(expression)
                # out_value = eval(expression) # eval ist extrem ungenau!
                out_value = self.a * in_value ** self.b
                output_data[eid] = {
                    'out': out_value
                }
        self.data = output_data
        return time + self.step_size
        pass

    def get_data(self, outputs):
        """
        see :meth:`mosaik_api.Simulator.get_data()`

        :param outputs: outputs is a dict mapping entity IDs to lists
        of attribute names whose values are requested.

        :return: The return value needs to be a dict of dicts mapping
        entity IDs and attribute names to their values.

        """
        return self.data
