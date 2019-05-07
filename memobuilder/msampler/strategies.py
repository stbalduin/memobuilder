import pyDOE as doe
import numpy as np
from memobuilder import msampler as memosampler


class SamplingStrategy(object):
    def __init__(self, param_variation_model):
        self.param_variation_model = param_variation_model

    def is_done(self, sampling_result):
        raise Exception('This method is supposed to be implemented '
                        'by subclasses')

    def get_next_sample(self):
        raise Exception('This method is supposed to be implemented by '
                        'subclasses')

    def __repr__(self):
        return '%s [%s]' % (self.__class__.__name__, str(self.__dict__))


class FixedDesignBasedSamplingStrategy(SamplingStrategy):

    def __init__(self, param_variation_model, **kwargs):
        SamplingStrategy.__init__(self, param_variation_model)
        self.kwargs = kwargs
        self.design = None
        self.index = -1
        self.init()

    def is_done(self, input_response_dataset):
        return len(input_response_dataset) >= len(self.design)

    def get_next_sample(self):
        self.index += 1
        experiment = self.design[self.index]
        variable_assignments = {}
        for (param_name, variability), value in zip(
            self.param_variation_model.design_variables.items(), experiment
        ):
            variable_assignments[param_name] = variability.denormalize(value)
        return variable_assignments


class FullFactorial(FixedDesignBasedSamplingStrategy):

    def init(self):
        num_levels = []
        dvar = self.param_variation_model.design_variables
        for attr_name, variability in dvar.items():
            if not isinstance(variability, (memosampler.NumericalLevels,
                                            memosampler.NonNumericalLevels)):
                raise ValueError('FullFactorial can only handle '
                                 '"NumericalLevels"  or "NonNumericalLevels" '
                                 'types of variability')
            num_levels.append(len(variability.levels))
        self.design = doe.fullfact(num_levels)


class LHS(FixedDesignBasedSamplingStrategy):

    def __init__(self, param_variation_model, num_samples, **kwargs):
        self.num_samples = num_samples
        FixedDesignBasedSamplingStrategy.__init__(
            self, param_variation_model, **kwargs
        )

    def init(self):
        num_vars = len(self.param_variation_model.design_variables)
        self.design = doe.lhs(num_vars, self.num_samples, **self.kwargs)


class MonteCarlo(FixedDesignBasedSamplingStrategy):
    def __init__(self, param_variation_model,  num_samples, **kwargs):
        self.num_samples = num_samples
        FixedDesignBasedSamplingStrategy.__init__(
            self, param_variation_model, **kwargs
        )

    def init(self):
        num_vars = len(self.param_variation_model.design_variables)
        self.design = np.random.rand(self.num_samples, num_vars)


class HaltonSeq(FixedDesignBasedSamplingStrategy):
    def __init__(self, param_variation_model,  num_samples, **kwargs):
        self.num_samples = num_samples
        FixedDesignBasedSamplingStrategy.__init__(
            self, param_variation_model, **kwargs
        )

    def init(self):
        num_vars = len(self.param_variation_model.design_variables)
        self.design = memosampler.halton(num_vars, self.num_samples)


class SobolSeq(FixedDesignBasedSamplingStrategy):
    def __init__(self, param_variation_model, num_samples, **kwargs):
        self.num_samples = num_samples
        FixedDesignBasedSamplingStrategy.__init__(
            self, param_variation_model, **kwargs
        )

    def init(self):
        num_vars = len(self.param_variation_model.design_variables)
        self.design = memosampler.sobol(num_vars, self.num_samples)
