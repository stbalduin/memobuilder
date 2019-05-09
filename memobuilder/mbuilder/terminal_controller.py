"""This module contains the TerminalController."""
from memobuilder.mbuilder.memobuilder_model import MeMoBuilderModel
from memobuilder.mbuilder.terminal_view import TerminalView
from memobuilder.mdb import h5db


class TerminalController(object):

    def __init__(self, yaml_file, h5_file, run_sampler, run_trainer):
        # the yaml configuration file
        self.yaml_file = yaml_file
        # the hdf data file
        self.h5_file = h5_file
        # switches the sampler component on or off
        self.run_sampler_option = run_sampler
        # switchtes the trainer componen on or off
        self.run_trainer_option = run_trainer

        self.model = MeMoBuilderModel()
        self.view = TerminalView()

    def run(self, custom_adapters=None):
        if custom_adapters is None:
            custom_adapters = {}

        # initialize the model (establish db connection.)
        self.open_db()

        # import yaml model to hdf5 db
        self.import_configuration()

        # load configuration from hdf5 db
        self.load_configuration()

        # load new and previously existing sampling results
        self.update_sampling_results()

        # load previously existing training results
        self.model.update_training_results()

        # run the sampler
        if self.run_sampler_option:
            self.run_sampler(custom_adapters=custom_adapters)
            # update sampling results
            self.update_sampling_results()
        else:
            # announce that sampling was skipped by the user
            self.view.announce_sampling_skipped()

        # run the trainer
        if self.run_trainer_option:
            self.run_trainer()
            # update training results
            self.model.update_training_results()
        else:
            self.view.announce_training_skipped()

        # present results
        results = self.model.get_training_results()
        self.view.present_results(results)

        # for surrogate_name in results.keys():
        #     self.model.save_surrogate_model(surrogate_name)

        # close the database
        self.model.close_db()
        return results

    def open_db(self):
        """
        Creates and opens a new hdf5 db iff a yaml config file is present.
        Else opens an existing hdf5 db.
        """
        yaml_file = self.yaml_file.strip()
        if yaml_file:
            # open and truncate existing files
            self.view.announce_database_truncated(True)
            self.model.open_db(
                self.h5_file,
                access_mode=h5db.H5AccessMode.WRITE_TRUNCATE_ON_EXIST
            )
        else:
            # open, but keep existing content
            self.view.announce_database_truncated(False)
            self.model.open_db(
                self.h5_file,
                access_mode=h5db.H5AccessMode.READ_WRITE_EXISTING_FILE
            )

    def import_configuration(self):
        """
        Loads configuration from yaml config iff such a config file is present.
        Else nothing happens.
        """
        """
        :return:
        """
        yaml_file = self.yaml_file.strip()
        if yaml_file:
            self.view.announce_configuration_import()
            self.model.import_configuration_from_yaml(self.yaml_file)

    def run_sampler(self, custom_adapters=None):
        if custom_adapters is None:
            custom_adapters = {}

        # load sampler configurations
        sampler_configurations = self.model.get_sampler_configurations()
        if len(sampler_configurations) == 0:
            # announce that no sampler was found
            self.view.announce_sampling_error('no sampler found')
        for sampler_conf in sampler_configurations:
            sid = sampler_conf.ID
            if sid in self.model.sampling_results:
                # announce that this sampling configuration was run before
                self.view.announce_sampler_status(
                    sid, 'skipped (data is already present)'
                )
                continue
            self.view.announce_sampler_status(sid, 'starting ...')
            self.model.create_sampler(sampler_conf)
            self.model.apply_sampler_custom_adapters(custom_adapters, sid)
            self.model.run_sampler(sid)
            self.view.announce_sampler_status(sid, 'finished ...')

    def load_configuration(self):
        # always load configuration from HDF5, such that each object has
        # an ID
        self.view.announce_configuration_source('HDF5')
        self.model.load_configuration_from_db()

    def update_sampling_results(self):
        # update the model's state, such that it is aware of all stored
        # sampling result objects
        self.model.update_sampling_results()

    def update_training_results(self):
        # update the model's state, such that it is aware of all stored
        # training result objects
        self.model.update_training_results()

    def run_trainer(self):
        trainer_configurations = self.model.\
            get_surrogate_model_configurations()
        if len(trainer_configurations) == 0:
            # announce that no trainer configuration was found
            self.view.announce_training_error('No surrogate model training '
                                              'configuration found')
        for trainer_configuration in trainer_configurations:
            tid = trainer_configuration.ID
            name = trainer_configuration.name
            if self.trainer_result_exists(name):
                self.view.announce_trainer_status(
                    tid, 'Skipped (data is already present)'
                )
                continue
            # create a trainer object from the surrogate model configuration
            self.view.announce_trainer_status(name, 'starting ...')
            self.model.create_trainer(trainer_configuration)
            # run the trainer
            sid = trainer_configuration.sampler_configuration.ID
            self.model.run_trainer(tid, sid)
            self.view.announce_trainer_status(name, 'finished ...')

        pass

    def trainer_result_exists(self, name):
        return name in self.model.training_results
