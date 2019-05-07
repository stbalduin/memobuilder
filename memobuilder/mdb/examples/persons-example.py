import h5db
import yaml

### EXAMPLE MODEL

class Person(h5db.H5DBObject):
    name = h5db.Scalar
    age = h5db.Scalar
    c = h5db.Vector
    d = h5db.List
    address = h5db.Object


class Address(h5db.H5DBObject):
    street = h5db.Scalar
    zip_code = h5db.Scalar
    city = h5db.Scalar

### EXAMPLE CODE

db = h5db.H5DB('persons.h5', [Person, Address])
db.open()


#import sys
#sys.exit()

### save an address
address = Address()
address.street = 'Musterstraße 4711'
address.zip_code = 26125
address.city = 'Oldenburg'

#db.save_object(address)
#a = db.load_object(Address, 'Address 1')
#print(a)

### save a person
person = Person()
person.name = 'Max Mustermann'
person.age = '34'
person.c = [1.0, 2.0]
person.d = ['a', 'b', 'c']

person.address = address
db.save_object(person)


person = Person()
person.name = 'Simone Musterfrau'
person.age = '54'
person.c = [1, 2.0]
person.d = ['d', 'e', 'f']

address = Address()
address.street = 'Musterstraße 42'
address.zip_code = 26133
address.city = 'Musterstadt'

person.address = address
db.save_object(person)

#save_object(person)
#db.save_object(person)
#db.delete_object(Person, 'Person_01')
#db.update_object(person)
#db.load_object(Person, 'Person_0')


#print(db.load_object(Person, 'Person_0'))
for person in db.load_objects(Person):
    print(person)

db.close()


