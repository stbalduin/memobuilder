# TODO: all examples are deprecated

"""
Example of how to fit a model to a given dataset.
"""


from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing.data import StandardScaler, MinMaxScaler

from memosampler.sampler import Sampler, SamplingModel, RangeOfRealNumbers, Constant, LHS
import memosampler.util
from memosampler.scenario import SamplingScenario
from memotrainer.metamodels import Trainer, KrigingModel


if __name__=="__main__":
    import numpy as np 
    np.random.seed(42)
    
    # draw samples 
    num_samples = 100
    #step_size = testsim.STEP_SIZE
    sim_config = memosampler.util.TEST_SIM_CONFIG
     
    sampling_plan = memosampler.util.create_testsim_sampling_model()
    
    
    scenario = SamplingScenario(sim_config, sampling_plan)
    sampling_strategy = LHS(num_samples=num_samples, criterion='corr')
    sampler = Sampler(sampling_plan, scenario, sampling_strategy)
    x, y = sampler.run()
    
    print(x)
    print(y)    

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        x, y, test_size=0.4, random_state=0)
    
    
    param_grid = [{
            'theta0': [0.1**n for n in range(5)],
            'nugget': [0.1**n for n in range(5)], 
            }]
    metamodel = KrigingModel(param_grid=param_grid)
    
    metamodel.processing_steps.append(
        PolynomialFeatures(degree=2, interaction_only=False)
    )
        
    test_ratio = 0.2
    trainer = Trainer(x,y, test_ratio=test_ratio)
    trainer.input_transform = MinMaxScaler()
    trainer.output_transform = MinMaxScaler()
        
    trainer.fit(metamodel)
    
    print(metamodel)
    mean_squared_error = trainer.mean_squared_error(metamodel)
    print('mean_squared_error', mean_squared_error)
    mean_absolute_error = trainer.mean_absolute_error(metamodel)
    print('mean_absolute_error', mean_absolute_error)
    explained_variance_score = trainer.explained_variance_score(metamodel)
    print('explained_variance_score', explained_variance_score)
    r2_score = trainer.r2_score(metamodel)
    print('r2_score', r2_score)
    max_absolute_componentwise_error = trainer.max_absolute_componentwise_error(metamodel)
    print('max_absolute_componentwise_error', max_absolute_componentwise_error)
    

