import math
import yaml
import memosampler


def test_create_and_run_lhs_sampler(yaml_file ='tests/data/test_lhs_sampler.yaml'):
    # load yaml configuration
    config_objects = yaml.load(open(yaml_file))

    # select sampler configuration objects
    sampler_configurations = config_objects['sampler_configuration']

    # create sampler
    sampler = memosampler.create_sampler(sampler_configurations[0])

    sampling_result = sampler.run()
    inputs = sampling_result.inputs['in']
    responses = sampling_result.responses['out']

    assert len(sampling_result) == 100

    a = 2.0
    b = 3.0
    for i in range(len(sampling_result)):
        x = inputs[i]
        y = responses[i]
        expected_value = a*x**b
        assert math.fabs(expected_value - y) < 1E-9
