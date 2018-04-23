import collections

import memosampler
import memomodel


def create_sampler(sampler_configuration):
    """
    Factory method that create a new sampler object based on the information from a memomodel.SamplerConfig object.

    :param sampler_configuration: a fully initialized memomodel.SamplerConfig object with simConfig, modelStructure,
        strategy and parameter variations
    :return: a Sampler object, ready to be executed.
    """

    # create sampling model
    param_variation_model = create_parameter_variation_model(sampler_configuration.parameter_variations)

    # create sampling_strategy
    strategy = create_sampling_strategy(param_variation_model, sampler_configuration.strategy)

    # create a sampling scenario
    scenario = create_sampling_scenario(sampler_configuration.sim_config, sampler_configuration.model_structure)

    # compose the sampler
    sampler = memosampler.Sampler()
    sampler.strategy = strategy
    sampler.model_structure = sampler_configuration.model_structure
    sampler.parameter_variation_model = param_variation_model
    sampler.sampling_scenario = scenario
    sampler.sampler_configuration = sampler_configuration

    return sampler


def create_parameter_variation_model(parameter_variations):
    """
    Factory method that create a memosampler.sampler.ParameterVariationModel from a list of memomodel.ParameterVariation
      objects.

    Interprets ParameterVariation objects from the configuration and maps them to corresponding Variabilty objects,
    that the sampler understands. Constants and design variables will be provided in two separated dictionaries, which
    map the parameter or attribute name to the variability object.

    :param parameter_variations: List of memomodel.ParameterVariation objects
    :return: a ParameterVariationModel
    """

    design_constants = collections.OrderedDict()
    design_variables = collections.OrderedDict()
    for param_var in parameter_variations:
        args = {arg.key: arg.value for arg in param_var.variation_arguments}
        if param_var.variation_mode == memomodel.ParameterVariationMode.CONSTANT.value:
            design_constants[param_var.parameter_name] = memosampler.Constant(**args)
        elif param_var.variation_mode == memomodel.ParameterVariationMode.RANGE_OF_REAL_NUMBERS.value:
            design_variables[param_var.parameter_name] = memosampler.RangeOfRealNumbers(**args)
        elif param_var.variation_mode == memomodel.ParameterVariationMode.RANGE_OF_INTEGERS.value:
            design_variables[param_var.parameter_name] = memosampler.RangeOfIntegers(**args)
        elif param_var.variation_mode == memomodel.ParameterVariationMode.NUMERICAL_LEVELS.value:
            design_variables[param_var.parameter_name] = memosampler.NumericalLevels(**args)
        else:
            raise Exception('not yet implemented')
        pass

    parameter_variation_model = memosampler.ParameterVariationModel()
    parameter_variation_model.design_constants = design_constants
    parameter_variation_model.design_variables = design_variables
    return parameter_variation_model


class RecursiveSubClassFinder(object):

    @staticmethod
    def find_subclasses(superclazz):
        subclass_map = {clazz.__name__: clazz for clazz in superclazz.__subclasses__()}
        subclasses = list(subclass_map.values())
        for subclass in subclasses:
            subclass_map.update(RecursiveSubClassFinder().find_subclasses(subclass))
        return subclass_map


def create_sampling_strategy(param_variation_model, sampling_strategy_configuration):
    """
    A factory method that creates a memosampler.strategies.SamplingStrategy object from a
    memosampler.sampler.ParameterVariationModel and a memomodel.SamplingStrategy configuration object.

    :param param_variation_model: a ParameterVariationModel object, that contains variabilites for all constant and
        varying parameters
    :param sampling_strategy_configuration: a memomodel.SamplingStrategy configuration object.
    :return: A memosampler.strategies.SamplingStrategy object, ready to be used within a sampler. the concrete class
        of the object depends on type information in the configuration.
    """
    # build a map of all known SamplingStrategy subclasses
    #strategies_by_name = {clazz.__name__: clazz for clazz in memosampler.SamplingStrategy.__subclasses__()}
    strategies_by_name = RecursiveSubClassFinder.find_subclasses(memosampler.SamplingStrategy)


    # choose a class by name
    strategy_name = sampling_strategy_configuration.name
    strategy_class = strategies_by_name[strategy_name]

    # convert the argument list to a dictionary
    kwargs = {arg.key: arg.value for arg in sampling_strategy_configuration.arguments}

    # create the new SamplingStrategy object with keyword arguments
    instance = strategy_class(param_variation_model, **kwargs)
    return instance


def create_sampling_scenario(sim_config, model_structure):
    """
    A factory method that creates a memosampler.scenario.SamplingScenario from a memomodel.SimConfig and a
    memomodel.ModelStructure configuration object.

    :param sim_config: A memomodel.SimConfig configuration object
    :param model_structure: A memomodel.ModelStructure configuration object
    :return: a memosampler.scenario.SamplingScenario object, ready for experimentation
    """
    # convert sim_config arguments to a dictionary:
    config_dict = {arg.key: arg.value for arg in sim_config.arguments}
    # create the experiment:
    return memosampler.SamplingScenario(config_dict, model_structure)
