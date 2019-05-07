import sys
import argparse

from memobuilder.mbuilder.terminal_controller import TerminalController


class MeMoBuilder(object):

    @staticmethod
    def from_cli_args(arguments):
        """
        Creates a MeMoBuilder from the given argument strings. The
        caller must provide an array of string arguments like this:

        usage: memo_builder.py [-h] [--yaml [YAML_FILE]] [--sampler |
                                                          --no-sampler]
                       [--trainer | --no-trainer]
                       H5File
        memo_builder.py: error: the following arguments are required: H5File

        :param arguments: string array of command line arguments
        :return: an MeMoBuilder instance
        """
        parsed_args = MeMoBuilder.parse_args(arguments)
        return MeMoBuilder(parsed_args)

    @staticmethod
    def parse_args(arguments):
        """
        This function uses the argparse module to convert the given
        argument string to objects and assigns them as attributes
        to a namespace.

        :param arguments: argument strings
        :return: a new Namespace populated with the parsed objects
        """
        parser = argparse.ArgumentParser()

        parser.add_argument(
            'H5File',
            type=str,
            help='HDF5 file that will be used as storage for configuration, '
                 'sampled data and trained models. '
                 'Will be created when a configuration file is imported.'
        )

        parser.add_argument(
            '--yaml',
            dest='yaml_file',
            default='',
            action='store',
            nargs='?',
            type=str,
            help='YAML configuration file. If present a new HDF5 file '
                 'will be created and the configuration '
                 'will be imported. Overwrites existing HDF5 datastores.'
        )

        sampling_group = parser.add_mutually_exclusive_group(required=False)
        sampling_group.add_argument(
            '--sampler', dest='sampler', action='store_true')
        sampling_group.add_argument(
            '--no-sampler', dest='sampler', action='store_false')
        parser.set_defaults(sampler=True)

        trainer_group = parser.add_mutually_exclusive_group(required=False)
        trainer_group.add_argument(
            '--trainer', dest='trainer', action='store_true')
        trainer_group.add_argument(
            '--no-trainer', dest='trainer', action='store_false')
        parser.set_defaults(trainer=True)

        return parser.parse_args(arguments)

    def __init__(self, parsed_args):
        self.controller = TerminalController(
            parsed_args.yaml_file, parsed_args.H5File,
            parsed_args.sampler, parsed_args.trainer
        )

    def run(self, custom_adapters=None):
        """
        This function executes the process of building a SurrogateModel.
        It starts with the sampling of data points and then trains models
        to fit this data. The configuration must first be imported from a
        yaml file. If no --yaml parameter is given, the program assumes
        that the configuration data was already imported, and loads if
        from the hdf datastore.

        Running the sampler involves invoking a mosaik simulator, with
        all necessary parameters. Mosaik allows it's users freely to
        specify how to pass parameter values to their models. Thus, it
        may happen, that the parameters must be wrapped in dictionaries
        or in specialized objects in order to initialize a simulator or
        to create an instance of a model. Because of this MeMoBuilder
        allows the user to implement prepare_init_arguments() and
        prepare_create_arguments() functions. These functions may be
        passed in the function_overrides dictionary. By default, the
        parameter values would be passed as keyword arguemnts to mosaik.

        :param custom_adapters: a dictionary containing functions, that
            will be attached to the SamplingScenario in use. Only the keys
            'prepare_init_arguments' and 'prepare_create_arguments' will be
            processed.
        :return:
        """
        if custom_adapters is None:
            custom_adapters = {}
        self.controller.run(custom_adapters=custom_adapters)


if __name__ == '__main__':
    # TODO: in einen testcase verlegen
    # args = sys.argv[1:]
    # args = ['test.h5', '--yaml', 'test.yaml', '--no-sampler', '--no-trainer']
    # args = ['test.h5', '--sampler', '--trainer']

    args = ['test.h5', '--yaml', 'test.yaml', '--sampler', '--trainer']
    # args = ['test.h5', '--sampler', '--trainer']

    memo_builder = MeMoBuilder.from_cli_args(args)
    memo_builder.run()
