import yaml
import h5db
import memomodel
import memosampler
import memotrainer


class MeMoBuilderModel():
    def __init__(self):
        self.db = None # the hdf5 data file
        self.configuration_objects = None # all configuration objects. loaded on startup from hdf5 file
        self.sampler_objects = {} # maps sampler_configuration IDs to sampler objects
        self.sampling_results = {} # maps sampler_configuration IDs to sampling result objects
        self.trainers = {} # maps trainer_configuration IDs to trainer objects
        self.training_results = {} # maps surrogate model name to training result objects

    def open_db(self, h5_file, access_mode=h5db.H5AccessMode.WRITE_TRUNCATE_ON_EXIST):
        self.db = memomodel.MeMoDB(h5_file)
        self.db.open(access_mode=access_mode)

    def close_db(self):
        self.db.close()

    def import_configuration_from_yaml(self, yaml_file):
        configuration_objects = yaml.load(open(yaml_file))
        self.save_configuration_to_hdf(configuration_objects)

    def load_configuration_from_yaml(self,  yaml_file):
        self.configuration_objects = yaml.load(open(yaml_file))
        return self.configuration_objects

    def save_configuration_to_hdf(self, configuration_objects):
       for configuration_segment in configuration_objects.keys():
           for object in configuration_objects[configuration_segment]:
               self.db.save_object(object)

    def load_configuration_from_db(self):
        self.configuration_objects = {
            # load SimConfig objects
            'simulator_configuration': self.db.load_objects(memomodel.SimConfig),
            # load ModelStructure objects
            'model_structure_configuration': self.db.load_objects(memomodel.ModelStructure),
            # load ParameterVariation objects
            'parameter_variation': self.db.load_objects(memomodel.ParameterVariation),
            # load StrategyConfig objects
            'sampler_strategy': self.db.load_objects(memomodel.StrategyConfig),
            # load SamplerConfig objects
            'sampler_configuration': self.db.load_objects(memomodel.SamplerConfig),
            # load SurrogateModelConfig objects
            'surrogate_model_configuration': self.db.load_objects(memomodel.SurrogateModelConfig)
        }

    def get_sampler_configurations(self):
        return self.configuration_objects['sampler_configuration']

    def create_sampler(self, sampler_configuration):
        id = sampler_configuration.ID
        self.sampler_objects[id] = memosampler.create_sampler(sampler_configuration)
        return id

    def apply_sampler_custom_adapters(self, custom_adapters, sid):
        sampler = self.sampler_objects[sid]
        scenario = sampler.sampling_scenario

        scenario_adapters = ['input_adapter', 'collector_adapter', 'model_adapter']
        for adapter_name in scenario_adapters:
            if adapter_name in custom_adapters:
                setattr(scenario, adapter_name, custom_adapters[adapter_name])

    def run_sampler(self, sid):
        # run the sampler
        sampler = self.sampler_objects[sid]
        sampling_result = sampler.run()

        # attach the sampler_configuration to the result
        sampling_result.sampler_configuration = sampler.sampler_configuration

        # save the sampling result and update the model
        self.db.save_object(sampling_result)
        self.db.save_object(memomodel.DatasetOwnership(owner=sampler.sampler_configuration, dataset=sampling_result))

        self.update_sampling_results()

    def update_sampling_results(self):
        self.sampling_results.clear()

        # load dataset ownership data and put the information into a dictionary
        dataset_ownerships = self.db.load_objects(memomodel.DatasetOwnership)
        owner_by_dataset = {ownership.dataset.ID: ownership.owner for ownership in dataset_ownerships}

        # load all existing datasets
        datasets = self.db.load_objects(memomodel.InputResponseDataset)
        for dataset in datasets:
            if not dataset.ID in owner_by_dataset:
                continue
            owner_id = owner_by_dataset[dataset.ID].ID
            self.sampling_results[owner_id] = dataset

    def get_surrogate_model_configurations(self):
        return self.configuration_objects['surrogate_model_configuration']

    def create_trainer(self, trainer_configuration):
        trainer = memotrainer.create_surrogate_model_trainer(trainer_configuration)
        tid = trainer_configuration.ID
        self.trainers[tid] = trainer

    def run_trainer(self, tid, sid):
        # load the dataset

        sampling_result = self.sampling_results[sid]

        # run the sampler
        trainer = self.trainers[tid]
        result = trainer.fit(sampling_result)

        # save the sampling result and update the model
        self.db.save_object(result)

    def save_surrogate_model(self, surrogate_name):
        surrogate_training_result = self.training_results[surrogate_name]
        metamodels = [result.metamodel for result in surrogate_training_result.training_results]
        model_structure = self.find_model_structure_by_surrogate_name(surrogate_name)
        model = memomodel.SurrogateModel()
        model.name = surrogate_name
        model.metamodels = metamodels
        model.model_structure = model_structure
        self.db.save_object(model)

    def find_model_structure_by_surrogate_name(self, surrogate_name):
        configs = self.db.load_objects(memomodel.SurrogateModelConfig)
        config = [config for config in configs if config.name == surrogate_name][0]
        return config.sampler_configuration.model_structure


    def get_training_results(self):
        return self.training_results

    def update_training_results(self):
        self.training_results.clear()
        results = self.db.load_objects(memomodel.SurrogateModelTrainingResult)
        for result in results:
            self.training_results[result.surrogate_model_name] = result