import numpy as np
import math
import sklearn.metrics as metrics
from sklearn.utils import validation
# from scipy.stats.stats import pearsonr


def r2_score(approximation_model, test_data):
    """
    Computes the r² score of the *approximation_model* applied to the previously configured test
    data in *self.y_pred*.

    :param approximation_model: :class:`MetaModel`
    :return: float r² value
    """
    if test_data.responses.shape[1] > 1:
        scorer = metrics.make_scorer(metrics.r2_score, multioutput='uniform_average')
    else:
        scorer = metrics.make_scorer(metrics.r2_score)
    score_value = scorer(approximation_model, test_data.inputs, test_data.responses)
    return score_value


def avg_score(approximation_model, test_data):
    """
    computes the average score of the *approximation_model*
    :param approximation_model: in :class:'MetaModel'
    :param test_data:
    :return:
    """
    if test_data.responses.shape[1] > 1:
        scorer = metrics.make_scorer(metrics.mean_absolute_error, multioutput='uniform_average')
    else:
        scorer = metrics.make_scorer(metrics.mean_absolute_error)
    score_value = scorer(approximation_model, test_data.inputs, test_data.responses)
    return score_value


def mse_score(approximation_model, test_data):
    if test_data.responses.shape[1] > 1:
        scorer = metrics.make_scorer(metrics.mean_squared_error, multioutput='uniform_average')
    else:
        scorer = metrics.make_scorer(metrics.mean_squared_error)
    score_value = scorer(approximation_model, test_data.inputs, test_data.responses)
    return score_value


def hae_score(approximation_model, test_data):
    if test_data.responses.shape[1] > 1:
        scorer = metrics.make_scorer(harmonic_averages_error, multioutput='uniform_average')
    else:
        scorer = metrics.make_scorer(harmonic_averages_error)
    score_value = scorer(approximation_model, test_data.inputs, test_data.responses)
    return score_value


def harmonic_averages_error(y_true, y_pred, multioutput='uniform_average'):
    """computes the harmonic average as a optimistic error function"""
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput)

    if multioutput == 'uniform_average':
        num_outputs = y_true.shape[1]
        multioutput = np.zeros(shape=num_outputs)

        for i in range(num_outputs):
            multioutput[i] = __hae_fct(y_true[i], y_pred[i])
        return np.average(multioutput)
    else:
        return __hae_fct(y_true, y_pred)


def __hae_fct(y_true, y_pred):
    n = len(y_pred)
    denominator = 0
    for index in range(n):
        denominator += __denominator_of_hae(y_true[index], y_pred[index])

    # print(n/denominator)
    scorer = n / denominator
    return scorer


def __denominator_of_hae(y_true, y_pred):
    dif = y_true - y_pred
    if dif < 1e-150:
        dif = 1e-10# e-100
    # print("error was zero")
    try:
        result = 1 / math.sqrt((dif) ** 2)
    except ZeroDivisionError:
        print('error was zero')
    return result

#Copy pasted from sklearn.metrics.regression.py :D ##wird nochmal aufgeräumt #TODO
def _check_reg_targets(y_true, y_pred, multioutput):
    """Check that y_true and y_pred belong to the same regression task

    Parameters
    ----------
    y_true : array-like,

    y_pred : array-like,

    multioutput : array-like or string in ['raw_values', uniform_average',
        'variance_weighted'] or None
        None is accepted due to backward compatibility of r2_score().

    Returns
    -------
    type_true : one of {'continuous', continuous-multioutput'}
        The type of the true target data, as output by
        'utils.multiclass.type_of_target'

    y_true : array-like of shape = (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape = (n_samples, n_outputs)
        Estimated target values.

    multioutput : array-like of shape = (n_outputs) or string in ['raw_values',
        uniform_average', 'variance_weighted'] or None
        Custom output weights if ``multioutput`` is array-like or
        just the corresponding argument if ``multioutput`` is a
        correct keyword.

    """
    validation.check_consistent_length(y_true, y_pred)
    y_true = validation.check_array(y_true, ensure_2d=False)
    y_pred = validation.check_array(y_pred, ensure_2d=False)

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError("y_true and y_pred have different number of output "
                         "({0}!={1})".format(y_true.shape[1], y_pred.shape[1]))

    n_outputs = y_true.shape[1]
    multioutput_options = (None, 'raw_values', 'uniform_average',
                           'variance_weighted')
    if multioutput not in multioutput_options:
        multioutput = validation.check_array(multioutput, ensure_2d=False)
        if n_outputs == 1:
            raise ValueError("Custom weights are useful only in "
                             "multi-output cases.")
        elif n_outputs != len(multioutput):
            raise ValueError(("There must be equally many custom weights "
                              "(%d) as outputs (%d).") %
                             (len(multioutput), n_outputs))
    y_type = 'continuous' if n_outputs == 1 else 'continuous-multioutput'

    return y_type, y_true, y_pred, multioutput