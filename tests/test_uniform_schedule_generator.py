from memoutil.schedules import UniformScheduleGenerator


def test_uniform_schedule_generator_with_maximum_slot_number():
    num_schedules = 10 # total number of schedules to generate
    resolution = 900 # simulation step size. time difference of two schedule items in seconds
    duration = 86400 # total duration of each target schedule: One day
    num_slots = 96 # number of slots for the target schedule: 96 slots for one day means: slots of 900s duration
    min = -1000.
    max = 1000.
    schedules = UniformScheduleGenerator.generate_schedules(num_schedules, resolution, duration, num_slots, min, max)

    assert len(schedules) == 10
    for schedule in schedules:
        #print(schedule)
        assert len(schedule) == 96
        assert all([v > min for v in schedule])
        assert all([v < max for v in schedule])


def test_uniform_schedule_generator_with_slots_of_1Hour():
    num_schedules = 10 # total number of schedules to generate
    resolution = 900 # simulation step size. time difference of two schedule items in seconds
    duration = 86400 # total duration of each target schedule: One day
    num_slots = 24 # number of slots for the target schedule: 24 slots for one day means: slots of one hour duration
    min = -1000.
    max = 1000.
    schedules = UniformScheduleGenerator.generate_schedules(num_schedules, resolution, duration, num_slots, min, max)

    assert len(schedules) == 10
    for schedule in schedules:
        #print(schedule)
        assert len(schedule) == 96
        assert all([v > min for v in schedule])
        assert all([v < max for v in schedule])


def test_uniform_schedule_generator_with_slots_of_4Hour():
    num_schedules = 10 # total number of schedules to generate
    resolution = 900 # simulation step size. time difference of two schedule items in seconds
    duration = 86400 # total duration of each target schedule: One day
    num_slots = 6 # number of slots for the target schedule: 6 slots for one day means: slots of 4 hour duration
    min = -1000.
    max = 1000.
    schedules = UniformScheduleGenerator.generate_schedules(num_schedules, resolution, duration, num_slots, min, max)

    assert len(schedules) == 10
    for schedule in schedules:
        # print(schedule)
        assert len(schedule) == 96
        assert all([v > min for v in schedule])
        assert all([v < max for v in schedule])



if __name__ == '__main__':
    #test_lhs_schedule_generator_with_slots_of_1Hour()
    test_lhs_schedule_generator_with_slots_of_4Hour()