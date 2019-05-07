import h5db
import yaml

### EXAMPLE MODEL

class Person(h5db.H5DBObject, yaml.YAMLObject):
    yaml_tag = '!Person'
    name = h5db.Scalar
    age = h5db.Scalar
    c = h5db.Vector
    d = h5db.List
    address = h5db.Object


class Address(h5db.H5DBObject, yaml.YAMLObject):
    yaml_tag = '!Address'
    street = h5db.Scalar
    zip_code = h5db.Scalar
    city = h5db.Scalar

h5db.core.register_h5db_yaml_objects()

### EXAMPLE CODE

db = h5db.H5DB('yaml-persons.h5', [Person, Address])
db.open()

yaml_persons = yaml.load(
"""
- !Person
  name: Simone Musterfrau
  age: '54'
  address: !Address {city: Musterstadt, street: "Musterstra\xDFe 42", zip_code: 26133}
  c: [1, 2.0]
  d: [d, e, f]
- !Person
  name: Max Mustermann
  age: '34'
  address: !Address {city: Oldenburg, street: "Musterstra\xDFe 4711", zip_code: 26125}
  c: [1, 2.0]
  d: [d, e, f]
""")

for person in yaml_persons:
    db.save_object(person)

for person in db.load_objects(Person):
    print(person)

db.close()


