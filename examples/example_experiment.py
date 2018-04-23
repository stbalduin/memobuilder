# TODO: all examples are deprecated

"""
An example of how to use the class :class:`memosampler.scenario.SamplingScenario`
"""

from memosampler.scenario import SamplingScenario 

import memosampler.simulators
import memosampler.util as testsim
import memosampler

if __name__ == '__main__':

    from memosampler.sampler import Variability

    # initial setup
    sim_config = memosampler.simulators.TestSimulator.SIM_CONFIG
    out_attr_names = {'out'}
    static_vars = {'b' : 2}
    step_size = 60 * 15 
    
    sampling_plan = testsim.create_testsim_sampling_model()
    
    for i in range(3):
        print()
        print('run #%d:' %(i))
        duration = step_size # duration of the simulation
        experiment = SamplingScenario(sim_config, sampling_plan)
        
        sim_params = {}
        model_params = {'a' : -1*i}
        inputs = {'in':-3.0+i}
        data = experiment.simulate(sim_params, model_params, inputs)
        print(data)
    
    