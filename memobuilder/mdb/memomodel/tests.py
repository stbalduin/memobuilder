import pandas
import yaml

from membuilder.mdb.memomodel import (ApproximationFunctionConfig,
                                      InputResponseDataset, KeyObjectPair,
                                      KeyValuePair, ModelStructure,
                                      ParameterVariation, SamplerConfig,
                                      SimConfig, StrategyConfig,
                                      SurrogateModel, VirtualState)
from memobuilder.mdb import h5db
from memobuilder.mdb.h5db import H5DB


def load_and_print_objects(db, clazz):
    results = db.load_objects(clazz)
    for res in results:
        print(res)


def test_save_sim_config():
    db = H5DB('sim_configs.h5', [SimConfig, KeyValuePair])
    db.open()
    obj = SimConfig()
    obj.arguments = [
        KeyValuePair(key='python', value='234'),
        KeyValuePair(key='cwd', value='.')]
    db.save_object(obj)
    load_and_print_objects(db, SimConfig)
    #print(yaml.dump(obj))

    db.close()


def test_save_yaml_sim_config():
    # !SimConfig
    # arguments:
    #   - !KeyValuePair {key: python, value: '234'}
    #   - !KeyValuePair {key: cwd, value: .}

    print('>>> test_save_yaml_sim_config')
    yaml_config = yaml.load(
    """
    !SimConfig
    arguments:
      - !KeyValuePair {key: python, value: '234'}
      - !KeyValuePair {key: cwd, value: .}
    """)

    db = H5DB('yaml_sim_configs.h5', [SimConfig, KeyValuePair])
    db.open()
    db.save_object(yaml_config)
    load_and_print_objects(db, SimConfig)
    db.close()
    print('<<< test_save_yaml_sim_config')


def test_save_model_structure():
    print('>>> test_save_model_structure')

    db = H5DB('modelstructure_test.h5', [ModelStructure, VirtualState])
    db.open()

    model_structure = ModelStructure()
    model_structure.simulator_parameters = ['a', 'b']
    model_structure.model_parameters = ['c', 'd']
    model_structure.model_inputs = ['e', 'f']
    model_structure.model_outputs = ['g', 'h']

    soc = VirtualState()
    soc.name = 'SoC'
    soc.init_attribute = 'SoC'
    soc.update_attribute = 'SoC'
    model_structure.virtual_states = [soc]

    db.save_object(model_structure)
    load_and_print_objects(db, ModelStructure)
    db.close()

    #print(yaml.dump(model_structure))
    print('<<< test_save_model_structure')


def test_save_yaml_model_structure():
    print('>>> test_save_yaml_model_structure')
    # !ModelStructure
    # model_inputs: [e, f]
    # model_outputs: [g, h]
    # model_parameters: [c, d]
    # simulator_parameters: [a, b]
    # virtual_states:
    # - !VirtualState
    # {ID: VirtualState_0, init_attribute: SoC, name: SoC, update_attribute: SoC}

    yaml_config = yaml.load(
    """
    !ModelStructure
    model_inputs: [e, f]
    model_outputs: [g, h]
    model_parameters: [c, d]
    simulator_parameters: [a, b]
    virtual_states:
      - !VirtualState {init_attribute: SoC, name: SoC, update_attribute: SoC}
    """)
    #print(yaml_config)

    db = H5DB('yaml_modelstructure_test.h5', [ModelStructure, VirtualState])
    db.open()
    db.save_object(yaml_config)
    load_and_print_objects(db, ModelStructure)
    db.close()
    print('<<< test_save_yaml_model_structure')


def test_save_key_value_pair():
    db = H5DB('keyvaluepair_test.h5', [KeyValuePair])
    db.open()

    kvp = KeyValuePair()
    kvp.key = '123'
    kvp.value = 123

    db.save_object(kvp)
    load_and_print_objects(db, KeyValuePair)
    db.close()


def test_save_strategy_config():
    print('>>> test_save_strategy_config')
    db = H5DB('strategy_test.h5', [StrategyConfig, KeyValuePair])
    db.open()

    parameters = []
    for i in range(5):
        parameters.append(KeyValuePair())
        parameters[i].key = 'param_%d' % (i)
        parameters[i].value = i

    strategy = StrategyConfig()
    strategy.name = "examples strategie"
    strategy.arguments = parameters

    db.save_object(strategy)
    load_and_print_objects(db, StrategyConfig)
    db.close()

    #print(yaml.dump(strategy))
    print('<<< test_save_strategy_config')


def test_save_yaml_strategy_config():
    print('>>> test_save_yaml_strategy_config')
    # !StrategyConfig
    # ID: StrategyConfig_1
    # arguments:
    #     - !KeyValuePair {ID: KeyValuePair_5, key: param_0, value: 0}
    #     - !KeyValuePair {ID: KeyValuePair_6, key: param_1, value: 1}
    #     - !KeyValuePair {ID: KeyValuePair_7, key: param_2, value: 2}
    #     - !KeyValuePair {ID: KeyValuePair_8, key: param_3, value: 3}
    #     - !KeyValuePair {ID: KeyValuePair_9, key: param_4, value: 4}
    # name: examples strategie

    yaml_config = yaml.load(
    """
    !StrategyConfig
        arguments:
            - !KeyValuePair {key: param_0, value: 0}
            - !KeyValuePair {key: param_1, value: 1}
            - !KeyValuePair {key: param_2, value: 2}
            - !KeyValuePair {key: param_3, value: 3}
            - !KeyValuePair {key: param_4, value: 4}
        name: examples strategie
    """)

    db = H5DB('yaml_strategy_test.h5', [StrategyConfig, KeyValuePair])
    db.open()
    db.save_object(yaml_config)
    load_and_print_objects(db, StrategyConfig)
    db.close()
    print('<<< test_save_yaml_strategy_config')


def test_save_parameter_variation():
    print('>>> test_save_parameter_variation')
    db = H5DB('parameter_variation_test.h5', [ParameterVariation, KeyValuePair])
    db.open()

    step_size = KeyValuePair(key='value', value=60)
    #step_size.key = 'value'
    #step_size.value = 60

    eta_pc =  KeyValuePair()
    eta_pc.key = 'value'
    eta_pc.value = [-2.109566, 0.403556, 97.110770]

    parameter_variation =  ParameterVariation()
    parameter_variation.parameter_name = "param_name"
    parameter_variation.variation_mode = "constant"
    parameter_variation.variation_arguments = [step_size]
    db.save_object(parameter_variation)

    parameter_variation = ParameterVariation()
    parameter_variation.parameter_name = "param_name"
    parameter_variation.variation_mode = "constant"
    parameter_variation.variation_arguments = [eta_pc]

    db.save_object(parameter_variation)
    load_and_print_objects(db, ParameterVariation)
    db.close()

    #print(yaml.dump(parameter_variation))
    print('<<< test_save_parameter_variation')


def test_save_yaml_parameter_variation():#
    print('>>> test_save_yaml_parameter_variation')
    # !ParameterVariation
    #     ID: ParameterVariation_1
    #     parameter_name: param_name
    #     variation_arguments:
    #         - !KeyValuePair
    #             ID: KeyValuePair_1
    #             key: value
    #             value: [-2.109566, 0.403556, 97.11077]
    #     variation_mode: constant

    yaml_config = yaml.load(
    """
    - !ParameterVariation
        parameter_name: step_size
        variation_arguments:
            - !KeyValuePair
                key: value
                value: 60
        variation_mode: constant

    - !ParameterVariation
        parameter_name: eta_pc
        variation_arguments:
            - !KeyValuePair
                key: value
                value: [-2.109566, 0.403556, 97.11077]
        variation_mode: constant
    """)

    db = H5DB('yaml_parameter_variation_test.h5', [ParameterVariation, KeyValuePair])
    db.open()

    for parameter_variation in yaml_config:
        db.save_object(parameter_variation)
    load_and_print_objects(db, ParameterVariation)
    db.close()
    print('<<< test_save_yaml_parameter_variation')


def test_save_sampling_result():
    print('>>> test_save_sampling_result')
    db = H5DB('test_save_sampling_result.h5', [InputResponseDataset])
    db.open(access_mode=h5db.H5AccessMode.WRITE_TRUNCATE_ON_EXIST)

    sampling_result = InputResponseDataset()

    import numpy
    input_data = {
        'col1' : list(numpy.arange(0,1,0.1)),
        'col2': list(range(33,43))
    }

    response_data = {
        'col1': [i**2 for i in input_data['col1']],
        'col2': [i**2 for i in input_data['col2']],
    }


    sampling_result.inputs = pandas.DataFrame(input_data)
    sampling_result.responses = pandas.DataFrame(response_data)


    print(len(sampling_result))
    print(sampling_result.inputs.dtypes)

    db.save_object(sampling_result)

    print('##')
    for sr in db.load_objects(InputResponseDataset):
        print('==')
        print(sr)

    db.close()
    print('<<< test_save_sampling_result')
    pass


def battery_sim():
    print('>>> battery_sim')
    db = H5DB('batterysimtest.h5', [SimConfig, ModelStructure, VirtualState, SamplerConfig, ParameterVariation,
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
    #db.save_object(sampler_config)


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

    #self.inputs = List
    #self.outputs = List
    #self.model_type = Scalar
    #self.model_arguments = ObjectList
    #self.trainer_options = ObjectList

    db.close()

    #print(yaml.dump(sm))
    print('>>> battery_sim')


def yaml_battery_sim():
    print('<<< yaml_battery_sim')
    # !SurrogateModel
    # ID: SurrogateModel_0
    # approximation_functions:
    # - !ApproximationFunctionConfig
    #   ID: ApproximationFunction_0
    #   inputs: [P_el_set, SoC]
    #   model_arguments: []
    #   model_type: Kriging
    #   outputs: [P_el, SoC]
    #   trainer_options: []
    # name: mySurrogateModel
    # sampler_configuration: !SamplerConfig
    #   ID: SamplerConfig_0
    #   model_structure: !ModelStructure
    #     ID: ModelStructure_0
    #     model_inputs: [P_el_set]
    #     model_outputs: [P_el, SoC]
    #     model_parameters: [capacity, P_el_max, P_el_min, eta_pc, SoC]
    #     simulator_parameters: [step_size]
    #     virtual_states:
    #     - !VirtualState {ID: VirtualState_0, init_attribute: SoC, name: SoC, update_attribute: SoC}
    #   name: LHS_N_100
    #   parameter_variations:
    #   - !ParameterVariation
    #     ID: ParameterVariation_0
    #     parameter_name: step_size
    #     variation_arguments:
    #     - !KeyValuePair {ID: KeyValuePair_2, key: value, value: 60}
    #     variation_mode: constant
    #   - !ParameterVariation
    #     ID: ParameterVariation_1
    #     parameter_name: capacity
    #     variation_arguments:
    #     - !KeyValuePair {ID: KeyValuePair_3, key: value, value: 5.0}
    #     variation_mode: constant
    #   - !ParameterVariation
    #     ID: ParameterVariation_2
    #     parameter_name: P_el_max
    #     variation_arguments:
    #     - !KeyValuePair {ID: KeyValuePair_4, key: value, value: 1000.0}
    #     variation_mode: constant
    #   - !ParameterVariation
    #     ID: ParameterVariation_3
    #     parameter_name: P_el_min
    #     variation_arguments:
    #     - !KeyValuePair {ID: KeyValuePair_5, key: value, value: -1000.0}
    #     variation_mode: constant
    #   - !ParameterVariation
    #     ID: ParameterVariation_4
    #     parameter_name: eta_pc
    #     variation_arguments:
    #     - !KeyValuePair
    #       ID: KeyValuePair_6
    #       key: value
    #       value: [-2.109566, 0.403556, 97.11077]
    #     variation_mode: constant
    #   - !ParameterVariation
    #     ID: ParameterVariation_5
    #     parameter_name: P_el_set
    #     variation_arguments:
    #     - !KeyValuePair {ID: KeyValuePair_7, key: min, value: -1000}
    #     - !KeyValuePair {ID: KeyValuePair_8, key: max, value: 1000}
    #     variation_mode: range_of_real_numbers
    #   - !ParameterVariation
    #     ID: ParameterVariation_6
    #     parameter_name: SoC
    #     variation_arguments:
    #     - !KeyValuePair {ID: KeyValuePair_9, key: min, value: 0.0}
    #     - !KeyValuePair {ID: KeyValuePair_10, key: max, value: 1.0}
    #     variation_mode: range_of_real_numbers
    #   sim_config: !SimConfig
    #     ID: SimConfig_0
    #     arguments:
    #     - !KeyValuePair {ID: KeyValuePair_0, key: python, value: 'pysimmods.batterysim.battery_mosaik:BatterySimulator'}
    #     - !KeyValuePair {ID: KeyValuePair_1, key: cwd, value: .}
    #   strategy: !StrategyConfig
    #     ID: StrategyConfig_0
    #     arguments:
    #     - !KeyValuePair {ID: KeyValuePair_11, key: num_samples, value: 1000}
    #     name: lhs
    yaml_config = yaml.load(
    """
    !SurrogateModel
    approximation_functions:
    - !ApproximationFunctionConfig
      inputs: [P_el_set, SoC]
      model_arguments: []
      model_type: Kriging
      outputs: [P_el, SoC]
      trainer_options: []
    name: mySurrogateModel
    sampler_configuration: !SamplerConfig
      model_structure: !ModelStructure
        model_inputs: [P_el_set]
        model_outputs: [P_el, SoC]
        model_parameters: [capacity, P_el_max, P_el_min, eta_pc, SoC]
        simulator_parameters: [step_size]
        virtual_states:
        - !VirtualState {init_attribute: SoC, name: SoC, update_attribute: SoC}
      name: LHS_N_100
      parameter_variations:
      - !ParameterVariation
        parameter_name: step_size
        variation_arguments:
        - !KeyValuePair {key: value, value: 60}
        variation_mode: constant
      - !ParameterVariation
        parameter_name: capacity
        variation_arguments:
        - !KeyValuePair {key: value, value: 5.0}
        variation_mode: constant
      - !ParameterVariation
        parameter_name: P_el_max
        variation_arguments:
        - !KeyValuePair {key: value, value: 1000.0}
        variation_mode: constant
      - !ParameterVariation
        parameter_name: P_el_min
        variation_arguments:
        - !KeyValuePair {key: value, value: -1000.0}
        variation_mode: constant
      - !ParameterVariation
        parameter_name: eta_pc
        variation_arguments:
        - !KeyValuePair
          key: value
          value: [-2.109566, 0.403556, 97.11077]
        variation_mode: constant
      - !ParameterVariation
        parameter_name: P_el_set
        variation_arguments:
        - !KeyValuePair {key: min, value: -1000}
        - !KeyValuePair {key: max, value: 1000}
        variation_mode: range_of_real_numbers
      - !ParameterVariation
        parameter_name: SoC
        variation_arguments:
        - !KeyValuePair {key: min, value: 0.0}
        - !KeyValuePair {key: max, value: 1.0}
        variation_mode: range_of_real_numbers
      sim_config: !SimConfig
        arguments:
        - !KeyValuePair {key: python, value: 'pysimmods.batterysim.battery_mosaik:BatterySimulator'}
        - !KeyValuePair {key: cwd, value: .}
      strategy: !StrategyConfig
        arguments:
        - !KeyValuePair {key: num_samples, value: 1000}
        name: lhs
    """)
    #print(yaml_config)

    db = H5DB('yaml_batterysimtest.h5', [SimConfig, ModelStructure, VirtualState, SamplerConfig, ParameterVariation,
                                         StrategyConfig, KeyValuePair, SurrogateModel, ApproximationFunctionConfig])
    db.open()

    # # sim config
    # sim_config = SimConfig()
    # sim_config.arguments = [
    #     KeyValuePair(key='python', value='pysimmods.batterysim.battery_mosaik:BatterySimulator'),
    #     KeyValuePair(key='cwd', value='.')]
    #
    # # model structure
    # model_structure = ModelStructure()
    # model_structure.simulator_parameters = ['step_size']
    # model_structure.model_parameters = ['capacity', 'P_el_max', 'P_el_min', 'eta_pc', 'SoC']
    # model_structure.model_inputs = ['P_el_set']
    # model_structure.model_outputs = ['P_el', 'SoC']
    #
    # soc = VirtualState(name='SoC', init_attribute='SoC', update_attribute='SoC')
    # model_structure.virtual_states = [soc]
    #
    # # sampling strategy
    # num_samples = KeyValuePair(key='num_samples', value=1000)
    # strategy = StrategyConfig(name='lhs')
    # strategy.arguments = [num_samples]
    #
    #
    # # parameter variations
    # step_size_variation = ParameterVariation(parameter_name='step_size', variation_mode='constant')
    # step_size_variation.variation_arguments = [KeyValuePair(key='value', value=60)]
    #
    # capacity_variation = ParameterVariation(parameter_name='capacity', variation_mode='constant')
    # capacity_variation.variation_arguments = [KeyValuePair(key='value', value=5.0)]
    #
    # p_el_max_variation = ParameterVariation(parameter_name='P_el_max', variation_mode='constant')
    # p_el_max_variation.variation_arguments = [KeyValuePair(key='value', value=1000.0)]
    #
    # p_el_min_variation = ParameterVariation(parameter_name='P_el_min', variation_mode='constant')
    # p_el_min_variation.variation_arguments = [KeyValuePair(key='value', value=-1000.0)]
    #
    # eta_pc_variation = ParameterVariation(parameter_name='eta_pc', variation_mode='constant')
    # eta_pc_variation.variation_arguments = [KeyValuePair(key='value', value=[-2.109566, 0.403556, 97.110770])]
    #
    # p_el_set_variation = ParameterVariation(parameter_name='P_el_set', variation_mode='range_of_real_numbers')
    # p_el_set_variation.variation_arguments = [KeyValuePair(key='min', value=-1000), KeyValuePair(key='max', value=1000)]
    #
    # soc_variation = ParameterVariation(parameter_name='SoC', variation_mode='range_of_real_numbers')
    # soc_variation.variation_arguments = [KeyValuePair(key='min', value=0.0), KeyValuePair(key='max', value=1.0)]
    #
    # parameter_variations = [step_size_variation, capacity_variation, p_el_max_variation, p_el_min_variation,
    #                         eta_pc_variation, p_el_set_variation, soc_variation]
    #
    # # sampler configuration
    # sampler_config = SamplerConfig(name='LHS_N_100')
    # sampler_config.sim_config = sim_config
    # sampler_config.strategy = strategy
    # sampler_config.model_structure = model_structure
    # sampler_config.parameter_variations = parameter_variations
    # #db.save_object(sampler_config)
    #
    #
    # # trainer configuration
    # approx = ApproximationFunctionConfig()
    # approx.inputs = ['P_el_set', 'SoC']
    # approx.outputs = ['P_el', 'SoC']
    # approx.model_type = 'Kriging'
    # approx.model_arguments = []
    # approx.trainer_options = []
    #
    # sm = SurrogateModel()
    # sm.name = 'mySurrogateModel'
    # sm.approximation_functions = [approx]
    # sm.sampler_configuration = sampler_config
    # db.save_object(sm)
    #
    # sm2 = SurrogateModel()
    # sm2.name = 'mySurrogateModel'
    # sm2.approximation_functions = [approx]
    # sm2.sampler_configuration = sampler_config
    # db.save_object(sm2)
    #
    # load_and_print_objects(db, SurrogateModel)

    #self.inputs = List
    #self.outputs = List
    #self.model_type = Scalar
    #self.model_arguments = ObjectList
    #self.trainer_options = ObjectList

    db.save_object(yaml_config)
    load_and_print_objects(db, SurrogateModel)
    db.close()

    #print(yaml.dump(sm))
    print('>>> yaml_battery_sim')


if __name__=='__main__':
    #test_save_sim_config()
    #test_save_yaml_sim_config()

    #test_save_model_structure()
    #test_save_yaml_model_structure()

    #test_save_key_value_pair()

    #test_save_strategy_config()
    #test_save_yaml_strategy_config()

    #test_save_parameter_variation()
    #test_save_yaml_parameter_variation()

    test_save_sampling_result()

    #battery_sim()
    #yaml_battery_sim()
