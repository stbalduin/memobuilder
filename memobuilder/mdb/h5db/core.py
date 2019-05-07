"""This module contains core functionalities of memodb."""
import pickle
from enum import Enum

import numpy as np
import pandas
import yaml
from h5py import File
from h5py import Reference as h5Reference
from h5py import special_dtype

# DEFAULT TYPES
UNINITIALIZED_SCALAR = 'UninitializedScalar'
REF_DTYPE = special_dtype(ref=h5Reference)


class Scalar():
    @staticmethod
    def save(parent_group, attr_name, value):
        if value == Scalar:
            # values has not been initialized
            value = UNINITIALIZED_SCALAR
        if value is None:
            value = np.nan
        parent_group[attr_name] = value

    @staticmethod
    def read(parent_group, attr_name):
        return parent_group[attr_name][()]


class Vector():
    @staticmethod
    def save(parent_group, attr_name, value):
        parent_group[attr_name] = value

    @staticmethod
    def read(parent_group, attr_name):
        return parent_group[attr_name][()]


class Matrix():
    @staticmethod
    def save(parent_group, attr_name, values):
        dataset = parent_group.create_dataset(attr_name, data=values)

    @staticmethod
    def read(parent_group, attr_name):
        return parent_group[attr_name][...]


class List():
    @staticmethod
    def save(parent_group, attr_name, value):
        dataset = parent_group.create_dataset(
            attr_name, data=np.array(value, dtype=np.string_))

    @staticmethod
    def read(parent_group, attr_name):
        return [item.decode() for item in parent_group[attr_name][()]]


class DataFrame():

    @staticmethod
    def save(parent_group, attr_name, dataframe):
        dataset = parent_group.create_dataset(attr_name, data=dataframe.values)
        dataset.attrs['index'] = np.array(dataframe.index.tolist(), dtype='S')
        dataset.attrs['columns'] = np.array(
            dataframe.columns.tolist(), dtype='S')

    @staticmethod
    def read(parent_group, attr_name):
        dataset = parent_group[attr_name]
        index = DataFrame.make_index(dataset.attrs['index'])
        columns = DataFrame.make_index(dataset.attrs['columns'])
        return pandas.DataFrame(data=dataset[...], index=index,
                                columns=columns)

    @staticmethod
    def make_index(raw):
        index = raw.astype('U')
        if index.ndim > 1:
            return pandas.MultiIndex.from_tuples(index.tolist())
        else:
            return pandas.Index(index)


class Object:

    @staticmethod
    def save(parent_group, attr_name, object):
        # save the object in the group of its own class
        ref = Object._h5db.save_object(object)

        # save a object reference in the target group
        ref_dataset = parent_group.create_dataset(
            attr_name, data=np.array(ref, dtype=REF_DTYPE))

    @staticmethod
    def read(parent_group, attr_name):
        # read reference dataset
        ref_dataset = parent_group[attr_name][()]
        # resolve reference
        return Object._resolve_reference(ref_dataset)

    @staticmethod
    def _resolve_reference(reference):
        h5obj = Object._h5db.resolve_ref(reference)
        fullpath = h5obj.name
        clazz_name, ID = fullpath[1:].split('/')
        clazz = Object._h5db.resolve_class_name(clazz_name)
        obj = Object._h5db.load_object(clazz, ID)
        return obj


class ObjectList:
    @staticmethod
    def save(parent_group, attr_name, objects):
        # save each object
        refs = []
        for object in objects:
            ref = Object._h5db.save_object(object)
            refs.append(ref)
        # save list of references
        ref_dataset = parent_group.create_dataset(
            attr_name, data=np.array(refs, dtype=REF_DTYPE))

    @staticmethod
    def read(parent_group, attr_name):
        # read reference dataset
        ref_dataset = parent_group[attr_name][()]
        # resolve all references
        objects = [Object._resolve_reference(ref) for ref in ref_dataset]
        return objects


class Reference():
    @staticmethod
    def save(parent_group, attr_name, value, type):
        parent_group[attr_name] = value
        parent_group[attr_name].attrs['type'] = type

    @staticmethod
    def read(parent_group, attr_name):
        return parent_group[attr_name][()]

    @staticmethod
    def type(parent_group, attr_name):
        return parent_group[attr_name].attrs['type']


class Blob():

    @staticmethod
    def save(parent_group, attr_name, object):
        pickled_obj = pickle.dumps(object)
        dataset = parent_group.create_dataset(
            attr_name, data=np.void(pickled_obj))
        #     dataset.attrs['memo_dataset_type'] = DatasetType.metamodel.value

    @staticmethod
    def read(parent_group, attr_name):
        pickled_obj = parent_group[attr_name][()]
        result = pickle.loads(pickled_obj)
        return result


class H5DBMapper():
    def __init__(self):
        self.group_name = None
        self.attributes = []
        self.attribute_types = []

    def __repr__(self):
        return str(self.__dict__)


class H5DBObject():

    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.ID = None

    def __repr__(self):
        return '%s [%s]' % (self.__class__.__name__, str(self.__dict__))


class H5AccessMode(Enum):
    """
    r 	Readonly, file must exist
    r+ 	Read/write, file must exist
    w 	Create file, truncate if exists
    w- or x 	Create file, fail if exists
    a 	Read/write if exists, create otherwise (default)
    """
    READ_EXISTING_FILE = 'r'
    READ_WRITE_EXISTING_FILE = 'r+'
    WRITE_TRUNCATE_ON_EXIST = 'w'
    WRITE_FAIL_ON_EXIST = 'x'
    DEFAULT = 'a'


class H5DB():
    DEFAULT_TYPES = [Scalar, Vector, List, Matrix,
                     DataFrame, Object, ObjectList, Blob]

    def __init__(self, h5filename, mapped_classes):
        self._h5backend = None
        self._h5filename = h5filename

        # construct class mappers
        for class_ in mapped_classes:
            H5DB._create_class_mapper(class_)

        self.mappers = {class_: class_.mapper for class_ in mapped_classes}
        self.name_to_class = {
            class_.__name__: class_ for class_ in mapped_classes
        }

    def open(self, access_mode=H5AccessMode.DEFAULT):
        self._h5backend = File(self._h5filename, access_mode.value)
        self._init_object_mapper()
        self._init_top_level_groups()

    @staticmethod
    def _create_class_mapper(class_):

        # initialize mapper only once for each class
        if not hasattr(class_, 'mapper'):
            mapper = H5DBMapper()
            mapper.group_name = class_.__name__
            for attr_name, attr_val in class_.__dict__.items():
                if attr_val in H5DB.DEFAULT_TYPES:
                    mapper.attributes.append(attr_name)
                    mapper.attribute_types.append(attr_val)
            class_.mapper = mapper

    def _init_object_mapper(self):
        Object._h5db = self

    def _init_top_level_groups(self):
        for _, mapper in self.mappers.items():
            if mapper.group_name not in self._h5backend:
                self._h5backend.create_group(mapper.group_name)

    def close(self):
        self._h5backend.close()

    def save_object(self, obj):
        # looking up a suitable mapper
        if obj.__class__ not in self.mappers:
            raise Exception('Unkown class encountered: %s' %
                            (obj.__class__.__name__))
        mapper = self.mappers[obj.__class__]

        if obj.ID is None:
            obj.ID = '%s_%d' % (mapper.group_name, len(
                self._h5backend[mapper.group_name]))

            # create a group for the object [--> obj must have an ID]
            target_group = self._h5backend[mapper.group_name].create_group(
                obj.ID)
            # save attributes of the object
            for attr_name, attr_type in zip(mapper.attributes,
                                            mapper.attribute_types):
                attr_type.save(target_group, attr_name,
                               getattr(obj, attr_name))
            ref = target_group.ref
        else:
            ref = self._h5backend[mapper.group_name][obj.ID].ref
        return ref

    def delete_object(self, class_, ID):
        # looking up a suitable mapper
        mapper = self.mappers[class_]
        del self._h5backend[mapper.group_name][ID]

    def update_object(self, obj):
        # first delete the old object,
        self.delete_object(obj.__class__, obj.ID)
        # then save the new object
        self.save_object(obj)

    def load_object(self, class_, ID):
        # looking up a suitable mapper
        mapper = self.mappers[class_]
        # find group with the object id
        parent_group = self._h5backend[mapper.group_name][ID]

        # create result object
        obj = class_()
        obj.ID = ID

        # populate properties of the result
        for attr_name, attr_type in zip(mapper.attributes,
                                        mapper.attribute_types):
            setattr(obj, attr_name, attr_type.read(parent_group, attr_name))
        return obj

    def load_objects(self, class_):
        # looking up a suitable mapper
        mapper = self.mappers[class_]
        # find group with the object id
        parent_group = self._h5backend[mapper.group_name]

        objects = []
        for ID in parent_group:
            objects.append(self.load_object(class_, ID))
        return objects

    def resolve_ref(self, h5ref):
        return self._h5backend[h5ref]

    def resolve_class_name(self, class_name):
        return self.name_to_class[class_name]


class H5DBYAMLIntegrator():

    def __init__(self):
        self.registered_classes = []
        self.tag_to_class_map = {}

    def find_all_h5db_yaml_classes(self):
        # lookup all subclasses of H5DBObject that are also
        # subclasses of YAMLObject
        yaml_subclasses = yaml.YAMLObject.__subclasses__()
        h5db_subclasses = H5DBObject.__subclasses__()
        h5db_yaml_classes = set(yaml_subclasses).intersection(h5db_subclasses)
        return h5db_yaml_classes

    def register_h5db_yaml_constructor(self, clazz):
        # register the constructor
        yaml.add_constructor(clazz.yaml_tag, self.yaml_memo_object_constructor)
        self.registered_classes.append(clazz)
        self.tag_to_class_map[clazz.yaml_tag] = clazz

    def yaml_memo_object_constructor(self, loader, node):
        """
        PyYAML uses this function in order to construct all memo objects.
        Important thing here is that the init function
        of h5db.H5DBObject is invoked.

        :param loader: YAML Loader
        :param node: YAML node
        :return:
        """
        clazz = self.tag_to_class_map[node.tag]
        instance = clazz.__new__(clazz)
        yield instance
        state = loader.construct_mapping(node, deep=True)
        H5DBObject.__init__(instance)
        instance.__init__(**state)


def register_h5db_yaml_objects():
    h5db_yaml = H5DBYAMLIntegrator()
    h5db_yaml_classes = h5db_yaml.find_all_h5db_yaml_classes()
    for clazz in h5db_yaml_classes:
        h5db_yaml.register_h5db_yaml_constructor(clazz)
