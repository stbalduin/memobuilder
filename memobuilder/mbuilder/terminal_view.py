"""This module contains the TerminalView."""
import sys


class TerminalView():
    def __init__(self):
        pass

    def announce_database_truncated(self, truncated):
        if truncated:
            print('Truncating HDF5 database ...')
        else:
            print('Reusing HDF5 database ...')

    def announce_configuration_import(self):
        print('Importing configuration from YAML into HDF5 database ...')

    def announce_configuration_source(self, source):
        print('Loading configuration from: {} ...'.format(source))

    def announce_sampling_skipped(self):
        print('Running samplers was skipped by user')

    def announce_training_skipped(self):
        print('Running trainers was skipped by user')

    def announce_sampler_status(self, sid, status):
        print('Current status of sampler {} is: {}'.format(sid, status))

    def announce_trainer_status(self, tid, status):
        print('Current status of trainer {} is: {}'.format(tid, status))

    def announce_sampling_error(self, msg):
        print(msg, file=sys.stderr)

    def announce_training_error(self, msg):
        print(msg, file=sys.stderr)

    def present_results(self, results):

        print('\n')
        for surrogate_name, surrogate_result in results.items():
            print('%s:' % (surrogate_name))

            for result in surrogate_result.training_results:
                func = '%s --> %s' % (result.metamodel.input_names,
                                      result.metamodel.response_names)
                score_r2 = 'r2_score=%f' % result.score_r2
                score_mae = 'mae_score=%f' % result.score_mae
                score_hae = 'hae_score=%f' % result.score_hae
                score_mse = 'mse_score=%f' % result.score_mse
                mtype = '(%s)' % result.metamodel.__class__.__name__

                print('  -', func, ':', score_r2, score_hae,
                      score_mae, score_mse, mtype)
