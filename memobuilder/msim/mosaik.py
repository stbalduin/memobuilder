"""
A mosaik simulator for :class:`SurrogateSimulationModels
<.SurrogateSimulationModel>`.

"""
import mosaik_api

from memobuilder.mdb import h5db, memomodel
from memobuilder.msim.simulator import (MetaModelSimulator,
                                        SurrogateModelSimulator)


class MosaikMeMoSimulator(mosaik_api.Simulator):
    """
    A mosaik simulator for :class:`SurrogateSimulationModels
    <.SurrogateSimulationModel>`.


    """
    # Sim config for this simulator for inprocess use.
    SIM_CONFIG = {'python': 'memobuilder.msim.mosaik:MosaikMeMoSimulator'}

    def __init__(self):
        super().__init__({})
        # prefix for eids
        self.eid_prefix = 'memo_'
        # maps EIDs to model
        self.entities = dict()
        self.sid = None
        # step size of the simulation
        self.step_size = None

    def init(self, sid, step_size, surrogate_model_file, surrogate_name):
        self.sid = sid
        self.step_size = step_size

        # load surrogate model objects
        surrogate = self.load_surrogate(surrogate_model_file, surrogate_name)
        self.model_structure = surrogate.model_structure
        self.metamodels = surrogate.metamodels
        self.model_name = surrogate_name

        # create meta data dynamically:
        param_names = self.model_structure.model_parameters
        attr_names = self.model_structure.model_outputs + \
            self.model_structure.model_inputs
        self.meta['models'] = {
            self.model_name: {
                'public': True,
                'params': param_names,  # parameters for model creation
                'attrs': attr_names  # attributes available within
            }
        }
        self.meta['extra_methods'] = [
            'get_output_attributes'
        ]
        return self.meta

    def open_db(self, surrogate_model_file):
        db = memomodel.MeMoDB(surrogate_model_file)
        db.open(access_mode=h5db.H5AccessMode.READ_EXISTING_FILE)
        return db

    def load_surrogate(self, surrogate_model_file, surrogate_name):
        # load surrogate moddels
        db = self.open_db(surrogate_model_file)
        surrogates = db.load_objects(memomodel.SurrogateModel)
        db.close()

        # find the requested model
        filtered_surrogates = [s for s in surrogates
                               if s.name == surrogate_name]
        if len(filtered_surrogates) != 1:
            raise Exception('Unexpected number of matching surrogate models:',
                            len(filtered_surrogates))
        return filtered_surrogates[0]

    def get_output_attributes(self):
        return self.model_structure.model_outputs

    def create(self, num, model, **init_vals):
        entities = []
        next_eid = len(self.entities)
        for i in range(next_eid, next_eid + num):
            eid = '%s%d' % (self.eid_prefix, i)
            entity = SurrogateModelSimulator(self.model_structure,
                                             self.metamodels)
            entity.init(**init_vals)
            self.entities[eid] = entity
            entities.append({'eid': eid, 'type': model})
        return entities

    def step(self, time, inputs):
        """
        see :meth:`mosaik_api.Simulator.step()`

        :param time: int, seconds since simulation start.

        :param inputs: a dict of dicts mapping entity IDs to attributes
        and dicts of values (each simulator has do decide on its own
        how to reduce the values (e.g., as its sum, average or maximum).
        This simulator ignores inputs.

        :return: int, time of the next simulation step (also in seconds
        since simulation start)

        """
        for eid, inputdata in inputs.items():
            model = self.entities[eid]
            for attr, attr_data in inputdata.items():
                model[attr] = sum(attr_data.values())
        for eid, model in self.entities.items():
            model.step()
        return (time + self.step_size)

    def get_data(self, outputs):
        """
        see :meth:`mosaik_api.Simulator.get_data()`

        :param outputs: outputs is a dict mapping entity IDs to lists
        of attribute names whose values are requested.

        :return: The return value needs to be a dict of dicts mapping
        entity IDs and attribute names to their values.
        """
        data = {}
        for eid, attrs in outputs.items():
            data[eid] = {}
            for attr in attrs:
                data[eid][attr] = self.entities[eid][attr]
        return data


if __name__ == '__main__':
    pass
