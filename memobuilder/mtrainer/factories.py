"""This module contains factory methods of the memo trainer."""
import memotrainer


def create_surrogate_model_trainer(surrogate_model_configuration):
    surrogate_model_trainer = memotrainer.SurrogateModelTrainer()
    surrogate_model_trainer.name = surrogate_model_configuration.name
    approx_functions = surrogate_model_configuration.approximation_functions

    for approximation_config in approx_functions:
        model_type = approximation_config.model_type
        model_arguments = approximation_config.model_arguments
        input_names = approximation_config.inputs
        response_names = approximation_config.outputs

        for trainer_score in approximation_config.trainer_score:
            metamodel = create_metamodel(
                model_type, model_arguments, input_names,
                response_names, trainer_score)
            surrogate_model_trainer.metamodels.append(metamodel)

    return surrogate_model_trainer


def create_metamodel(model_type, model_arguments, inputs, responses,
                     trainer_score):
    metamodel_classes = memotrainer.MetaModelClassFinder().metamodel_classes
    if model_type not in metamodel_classes:
        raise Exception('Unknown Metamodel type: %s' % (model_type))

    # convert model arguments to dictionary
    kwargs = {arg.key: arg.value for arg in model_arguments}

    # lookup a fitting metamodel class
    class_ = metamodel_classes[model_type]

    # create an instance of that class with keyword arguments
    instance = class_(input_names=inputs, response_names=responses,
                      trainer_score=trainer_score, **kwargs)

    return instance
