# TODO: all examples are deprecated

"""
This is an example of how to use the memosim sampler. First a sampling plan 
for a simple test simulator is set up and then the resulting input samples and
responses are plotted. 

TODO: zu einem automatisierbaren Test erweitern?
"""
import numpy as np 

from memosampler.sampler import Sampler, SamplingModel, Constant, LHS,\
    RangeOfRealNumbers
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import memosampler.simulators
from memosampler.scenario import SamplingScenario
import memosampler

PLOT = True  # whether or not to show a plot of the input data
NUM_SAMPLES = 100  # number of sampling points 

def main(): 
    
    np.random.seed(42)
    
    # mosaik configuration.py
    #sim_config = {'python': 'memosampler.util:TestSimulator'}
    sim_config = memosampler.simulators.TestSimulator.SIM_CONFIG
    step_size = memosampler.simulators.TestSimulator.STEP_SIZE # 60 * 15

    # 
    # The test simulator has one input named *in*, one output named *out* and
    # two parameters *a* and *b*. In this example *b* is considered to be a  
    # constant value. *a* and *in* may be varied.
    # 
   
    # Create an empty sampling plan 
    sampling_model = SamplingModel()
    
    # add constant parameters to the sampling plan configuration.py
    sampling_model.add_model_param('b', Constant(2))
    
    # add internal state to the sampling plan: 
    sampling_model.add_model_param('a', RangeOfRealNumbers(-1.0, 1.0))
    
    # add inputs to the sampling plan 
    sampling_model.add_model_input('in', RangeOfRealNumbers(10, 50))
    
    # add output variable names
    out_attr_names = ['out']
    for name in out_attr_names: 
        sampling_model.add_model_output(name)

    #sim_params = {'step_size':step_size}
    #sim_duration = step_size
    sampling_model.add_simulator_param('step_size', Constant(step_size))
    
        
    # retrieve ordered lists of input and output names:
    x_names = [me.name for me in sampling_model.get_variable_model_elements()]
    y_names = sampling_model.get_outputs()
    
    
    scenario = SamplingScenario(sim_config, sampling_model)

    sampling_strategy = LHS(num_samples=NUM_SAMPLES, criterion='corr')
    sampler = Sampler(sampling_model, scenario, sampling_strategy)
    x, y = sampler.run()

    
    print('x:', x)
    print('y:', y)
    
    if PLOT:
        # plot the input data points
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x['a'], x['in'], 'o')
        ax.set_xlabel(x_names[0])
        ax.set_ylabel(x_names[1])
        ax.set_title('sampling points')
        plt.grid()
        plt.show()
        
        # plot responses
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.scatter(x['a'], x['in'], y['out'], c='b', marker='o')
        ax2.set_xlabel(x_names[0])
        ax2.set_ylabel(x_names[1])
        ax2.set_zlabel(y_names[0])
        plt.show()

if __name__ == "__main__":
    main()
