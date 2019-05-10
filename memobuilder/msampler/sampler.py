import collections
from memobuilder.mdb import memomodel


class ParameterVariationModel(object):

    def __init__(self):
        self.design_constants = collections.OrderedDict()
        self.design_variables = collections.OrderedDict()

    def __repr__(self):
        return 'ParameterVariationModel [%s]' % (str(self.__dict__))


class Sampler(object):

    def __init__(self):
        # fields are usually initialized be create_sampler factory method.
        self.strategy = None
        self.model_structure = None
        self.parameter_variation_model = None
        self.sampling_scenario = None
        self.sampler_configuration = None

    def run(self):
        """This is the main loop of the sampling process, where the
        sampling scenario is invoked repeatedly with varying inputs. It
        is up to the sampling strategy in use to decide when the
        sampling process is finished and how to draw input samples.
        """
        # prepare a dictionary with constant parameters for later reuse
        dconstants = self.parameter_variation_model.design_constants
        constants = {}
        for attr_name, variability in dconstants.items():
            constants[attr_name] = variability.value

        # initialize an empty sampling result object
        # (column names needed here).
        input_names = list(
            self.parameter_variation_model.design_variables.keys()
        )
        response_names = list(self.model_structure.model_outputs)
        sampling_result = memomodel.InputResponseDataset(
            input_cols=input_names, response_cols=response_names
        )

        current_assignments = {}
        while not self.strategy.is_done(sampling_result):
            # draw new sample location
            current_sample = self.strategy.get_next_sample()

            # populate the current parameter assignments with the
            # constant parameters and the current sample
            current_assignments.update(constants)
            current_assignments.update(current_sample)

            # invoke a simulation with these inputs
            response = self.sampling_scenario.simulate(current_assignments)

            # udpate the sampling result
            sampling_result.update(current_sample, response)
            # print(current_sample, response)
            current_assignments.clear()
        return sampling_result
