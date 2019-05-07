from h5db import H5DB

from memomodel import SimConfig, VirtualState, ModelStructure, SamplerConfig, ParameterVariation, \
    StrategyConfig, KeyValuePair, KeyObjectPair, ApproximationFunctionConfig, SurrogateModel


def load_and_print_objects(db, clazz):
    results = db.load_objects(clazz)
    for res in results:
        print(res)


def battery_sim():
    db = H5DB('batterysim.h5', [SimConfig, ModelStructure, VirtualState, SamplerConfig, ParameterVariation,
                                StrategyConfig, KeyValuePair, SurrogateModel, ApproximationFunctionConfig])
    db.open()

    # sim config
    sim_config = SimConfig()
    sim_config.arguments = [
        KeyValuePair(key='python', value='pysimmods.batterysim.battery_mosaik:BatterySimulator'),
        KeyValuePair(key='cwd', value='.')]


    # model structure
    model_structure = ModelStructure()
    model_structure.simulator_parameters = ['step_size']
    model_structure.model_parameters = ['capacity', 'P_el_max', 'P_el_min', 'eta_pc', 'SoC']
    model_structure.model_inputs = ['P_el_set']
    model_structure.model_outputs = ['P_el', 'SoC']

    soc = VirtualState(name='SoC', init_attribute='SoC', update_attribute='SoC')
    model_structure.virtual_states = [soc]

    # sampling strategy
    num_samples = KeyValuePair(key='num_samples', value=1000)
    strategy = StrategyConfig(name='lhs')
    strategy.arguments = [num_samples]

    # parameter variations
    step_size_variation = ParameterVariation(parameter_name='step_size', variation_mode='constant')
    step_size_variation.variation_arguments = [KeyValuePair(key='value', value=60)]

    capacity_variation = ParameterVariation(parameter_name='capacity', variation_mode='constant')
    capacity_variation.variation_arguments = [KeyValuePair(key='value', value=5.0)]

    p_el_max_variation = ParameterVariation(parameter_name='P_el_max', variation_mode='constant')
    p_el_max_variation.variation_arguments = [KeyValuePair(key='value', value=1000.0)]

    p_el_min_variation = ParameterVariation(parameter_name='P_el_min', variation_mode='constant')
    p_el_min_variation.variation_arguments = [KeyValuePair(key='value', value=-1000.0)]

    eta_pc_variation = ParameterVariation(parameter_name='eta_pc', variation_mode='constant')
    eta_pc_variation.variation_arguments = [KeyValuePair(key='value', value=[-2.109566, 0.403556, 97.110770])]

    p_el_set_variation = ParameterVariation(parameter_name='P_el_set', variation_mode='range_of_real_numbers')
    p_el_set_variation.variation_arguments = [KeyValuePair(key='min', value=-1000), KeyValuePair(key='max', value=1000)]

    soc_variation = ParameterVariation(parameter_name='SoC', variation_mode='range_of_real_numbers')
    soc_variation.variation_arguments = [KeyValuePair(key='min', value=0.0), KeyValuePair(key='max', value=1.0)]

    parameter_variations = [step_size_variation, capacity_variation, p_el_max_variation, p_el_min_variation,
                            eta_pc_variation, p_el_set_variation, soc_variation]

    # sampler configuration
    sampler_config = SamplerConfig(name='LHS_N_100')
    sampler_config.sim_config = sim_config
    sampler_config.strategy = strategy
    sampler_config.model_structure = model_structure
    sampler_config.parameter_variations = parameter_variations
    # db.save_object(sampler_config)


    # trainer configuration
    approx = ApproximationFunctionConfig()
    approx.inputs = ['P_el_set', 'SoC']
    approx.outputs = ['P_el', 'SoC']
    approx.model_type = 'Kriging'
    approx.model_arguments = []
    approx.trainer_options = []

    sm = SurrogateModel()
    sm.name = 'mySurrogateModel'
    sm.approximation_functions = [approx]
    sm.sampler_configuration = sampler_config
    db.save_object(sm)

    sm2 = SurrogateModel()
    sm2.name = 'mySurrogateModel'
    sm2.approximation_functions = [approx]
    sm2.sampler_configuration = sampler_config
    db.save_object(sm2)

    load_and_print_objects(db, SurrogateModel)
    db.close()


if __name__ == '__main__':
    battery_sim()
