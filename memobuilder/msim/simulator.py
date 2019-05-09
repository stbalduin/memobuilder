"""
This module comprises the class :class:`SurrogateSimulationModel` and
its building blocks. The :class:`SurrogateSimulationModel` is a generic
simulation model, that has a simple model of its own state and a set of
metamodels that are used as transfer functions to compute output and
new state values during each simulation step.

"""
import copy


class MetaModelSimulator(object):
    """
    This class is supposed to be an abstract super class for simulation
    meta model implementations. A simulation meta model must provide a
    function :meth:`step` that maps a state dictionary to approximate
    responses.

    """
    def __init__(self):
        """
        :raises: Exception, because this class is not supposed to be
        instantiated.

        """
        raise Exception('this class is not supposed to be instantiated.')

    def step(self, state):
        """
        Computes new values for one or more output variables.

        :param state: dict<str, object>,
            a dictionary that maps property names to values.

        :return: dict<str, object>, returns a dictionary that maps
        output names to their predicted values.

        :raises: Exception, because this class is not supposed to be
        instantiated.

        """
        raise Exception('this class is not supposed to be instantiated.')


class SimpleMetaModelSimulator(MetaModelSimulator):
    """
    This MetaModelSimulator relies on a
    :class:`memotrainer.metamodels.MetaModel` to compute output values.

    :param meta_model: :class:`memotrainer.metamodels.MetaModel`,
    a previously trained meta_model.

    """

    def __init__(self, meta_model):
        """

        :param meta_model: :class:`memotrainer.metamodels.MetaModel`
            a previously trained meta_model.
        """
        self.meta_model = meta_model

    def step(self, state):
        """
        Computes new values for one or more output variables.

        :param state: dict<str, object>, a dictionary that maps
        property names to values.

        :return: dict<str, object>, returns a dictionary that maps
        output names to their predicted values.

        """
        input_names = self.meta_model.input_names
        response_names = self.meta_model.response_names
        x = [state[var] for var in input_names]
        y = self.meta_model.predict([x])
        responses = {name: y[name][0] for name in response_names}
        return responses


class CombinedMetaModelSimulator(MetaModelSimulator):
    def __init__(self, metamodels):
        self.metamodel_simulators = [SimpleMetaModelSimulator(m)
                                     for m in metamodels]

    def step(self, state):
        """
        Computes new values for one or more output variables.

        :param state: dict<str, object>,
            a dictionary that maps property names to values.

        :return: dict<str, object>, returns a dictionary that maps
        output names to their predicted values.
        """
        # compute output for each metamodel
        responses = {}
        for m in self.metamodel_simulators:
            m_out = m.step(state)
            responses.update(m_out)
        return responses


class SurrogateModelSimulator(object):

    def __init__(self, model_structure_description, metamodel):
        self.state = ModelState(model_structure_description)
        if type(metamodel) in [list, tuple]:
            self.metamodel_simulator = CombinedMetaModelSimulator(metamodel)
        else:
            self.metamodel_simulator = SimpleMetaModelSimulator(metamodel)
        self.state.set_mode('init')

    def init(self, **init_values):
        # create a new initial state
        for attribute, init_val in init_values.items():
            self.state.set_initial_value(attribute, init_val)
        self.state.set_mode('pre-step')

    def step(self):
        # Necessary; otherwise virtual states are not updated
        if not self.state.mode == 'pre-step':
            self.state.set_mode('pre-step')
        # make sure all inputs have been updated by the user
        self.state.set_mode('step')

        # compute output for each metamodel
        new_state = self.metamodel_simulator.step(self.state)

        # update state
        self.state.set_mode('post-step')
        for attribute, new_val in new_state.items():
            self.state.update_output_value(attribute, new_val)
        self.state.set_mode('idle')

    def __getitem__(self, attr_name):
        return self.state[attr_name]

    def __setitem__(self, attr_name, value):
        # make sure the state is in 'updating inputs mode'
        self.state.set_mode('pre-step')
        self.state[attr_name] = value


class ModelState(object):

    def __init__(self, model_structure):
        self.model_structure = model_structure
        self.static_attributes = {
            name: None for name in model_structure.model_parameters
        }
        self.input_attributes = {
            name: None for name in model_structure.model_inputs
        }
        self.output_attributes = {
            name: None for name in model_structure.model_outputs
        }
        self.internal_attributes = {
            mapping.name: None for mapping in model_structure.virtual_states
        }
        self.internal_states = [VirtualStateSimulator(m)
                                for m in model_structure.virtual_states]

        self.mode = None
        self.modes = {
            'init': InitMode(self),
            'pre-step': PreStepMode(self),
            'step': StepMode(self),
            'post-step': PostStepMode(self),
            'idle': IdleMode(self),
        }

    def set_initial_value(self, attribute, value):
        self.modes[self.mode].set_initial_value(attribute, value)

    def update_output_value(self, attribute, value):
        self.modes[self.mode].update_output_value(attribute, value)

    def update_internal(self, attribute, value):
        if attribute in self.internal_attributes:
            self.internal_attributes[attribute] = value

    def __getitem__(self, attr_name):
        if attr_name in self.static_attributes:
            if attr_name in self.internal_attributes:
                return self.internal_attributes[attr_name]
            else:
                return self.static_attributes[attr_name]
        elif attr_name in self.output_attributes:
            return self.output_attributes[attr_name]
        elif attr_name in self.input_attributes:
            return self.input_attributes[attr_name]
        elif attr_name in self.internal_attributes:
            return self.internal_attributes[attr_name]
        raise ValueError('Unknown attribute: %s' % (attr_name))

    def __setitem__(self, attr_name, value):
        self.modes[self.mode].set_input_value(attr_name, value)

    def set_mode(self, mode):
        if self.mode == mode:
            return
        if self.mode is not None:
            self.modes[self.mode].deactivate()
        self.mode = mode
        self.modes[mode].activate()

    def __repr__(self):
        repr = {'mode': self.mode,
                'static': self.static_attributes,
                'inputs': self.input_attributes,
                'outputs': self.output_attributes,
                'internals': self.internal_attributes,
                }
        return str(repr)


class Mode():
    def __init__(self, model_state):
        self.model_state = model_state

    def activate(self):
        raise Exception('Illegal state')

    def deactivate(self):
        raise Exception('Illegal state')

    def set_initial_value(self, attribute, value):
        # see InitMode
        raise Exception('Illegal state')

    def set_input_value(self, attribute, value):
        # see PreStepMode
        raise Exception('Illegal state')

    def update_output_value(self, attribute, value):
        # see PostStepMode
        raise Exception('Illegal state')


class InitMode(Mode):
    def __init__(self, model_state):
        self.model_state = model_state

    def activate(self):
        pass

    def deactivate(self):
        # do not proceed if some static attributes are unitialized:
        for attribute, value in self.model_state.static_attributes.items():
            if value is None:
                raise Exception('Parameter %s has not been initialized.'
                                % (attribute))

    def set_initial_value(self, attribute, value):
        # accept only inital values for static attributes
        if attribute not in self.model_state.static_attributes:
            raise Exception('Unknown parameter: %s' % (attribute))
        self.model_state.static_attributes[attribute] = value


class PreStepMode(Mode):

    def __init__(self, model_state):
        self.model_state = model_state

    def activate(self):
        # clear previous values of input variables:
        for attribute in self.model_state.input_attributes.keys():
            self.model_state.input_attributes[attribute] = None

        # clear previous values of internal state variables
        for attribute in self.model_state.internal_attributes.keys():
            self.model_state.internal_attributes[attribute] = None

    def set_input_value(self, attribute, value):
        # accept only values for known input attributes
        if attribute not in self.model_state.input_attributes:
            raise Exception('Unknown parameter: %s' % (attribute))
        self.model_state.input_attributes[attribute] = value

    def deactivate(self):
        # do not proceed if some input attributes were not updated
        for attribute in self.model_state.input_attributes.keys():
            if self.model_state.input_attributes[attribute] is None:
                raise Exception('output %s is none' % (attribute))

        # update values of internal states
        for mapping in self.model_state.internal_states:
            mapping.execute(self.model_state)


class StepMode(Mode):
    def __init__(self, model_state):
        self.model_state = model_state

    def activate(self):
        pass

    def deactivate(self):
        pass


class PostStepMode(Mode):
    def __init__(self, model_state):
        self.model_state = model_state

    def activate(self):
        # clear values of output variables:
        for attribute in self.model_state.output_attributes.keys():
            self.model_state.output_attributes[attribute] = None

    def update_output_value(self, attribute, value):
        # accept only values for known output attributes
        if attribute not in self.model_state.output_attributes:
            raise Exception('Unknown parameter: %s' % (attribute))
        self.model_state.output_attributes[attribute] = value

    def deactivate(self):
        # do not proceed if some output attributes were not updated
        for attribute in self.model_state.output_attributes.keys():
            if self.model_state.output_attributes[attribute] is None:
                raise Exception('output %s is none' % (attribute))
        # all outputs have been updated - switch to idle mode


class IdleMode(Mode):
    def __init__(self, model_state):
        self.model_state = model_state

    def activate(self):
        pass

    def deactivate(self):
        pass


class VirtualStateSimulator():

    def __init__(self, vs_descriptor):
        self.name = vs_descriptor.name
        self.init_src = vs_descriptor.init_attribute
        self.update_src = vs_descriptor.update_attribute
        self.has_run_before = False

    def execute(self, model_state):
        if not self.has_run_before:
            # init from init values
            val = model_state.static_attributes[self.init_src]
            model_state.update_internal(self.name, val)
            self.has_run_before = True
        else:
            # update from previous output
            val = model_state.output_attributes[self.update_src]
            model_state.update_internal(self.name, val)


class StupidMetaModelSimulator(MetaModelSimulator):
    def __init__(self, input_names, output_names):
        self.input_names = input_names
        self.output_names = output_names

    def step(self, model_state):
        p_set = model_state['P_el_set']
        internal_soc = model_state['soc']
        return {'P_el': p_set+1, 'SoC': internal_soc+1}


if __name__ == '__main__':
    pass
