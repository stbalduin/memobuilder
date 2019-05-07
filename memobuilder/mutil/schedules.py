import numpy
import pyDOE as doe
import random


class UniformScheduleGenerator():

    @staticmethod
    def generate_schedules(num_schedules, resolution, duration, num_slots,
                           min, max):
        if num_schedules == 0:
            return []
        # compute slot duration and number of datapoints per slot:
        slot_duration = duration / num_slots
        datapoints_per_slot = slot_duration / resolution

        schedules = []
        for i in range(num_schedules):
            # draw random values between min and max
            raw = [random.uniform(min, max) for i in range(num_slots)]
            # reapeat each value if slots of the target schedule have
            # several values
            schedule = numpy.repeat(raw, datapoints_per_slot)
            # convert numpy array to simple list
            schedule = list(schedule)
            schedules.append(schedule)
        return schedules


class LHSScheduleGenerator():

    @staticmethod
    def generate_schedules(num_schedules, resolution, duration, num_slots,
                           min, max):
        """
        This function may be used to generate test schedules. The
        duration of each schedule and the number of equally sized
        slots of the schedule may be specified by the user. Within
        a slot the value of a schedule is constant and has a randomly
        chosen value between *min* and *max*. For the construction of
        schedules a latin-hypercube sampling approach is used, which
        comprises following steps:

        * Latin hypercube sampling is used to create a plan for
          *num_schedules* experiments and *num_slots* factors.
        * Each row of this plan is then translated into a device schedule
          by
            * denormalizing its values
            * by repeating each value *datapoints_per_slot* times

        :param num_schedules: int, number of schedules to generate.

        :param resolution: int, step size of the controlled device
        in seconds (e.g. simulator step_size).

        :param duration: int, total duration of each target schedule in
        seconds.

        :param num_slots: int, number of equally sized time slots for
        each target schedule.

        :param min: float, minimal allowed schedule value

        :param max: float, maximal allowed schedule value

        :return: schedule_description[*num_schedules*],
        a list of json encoded schedules to be interpreted by DataSeriesSim
        (https://ecode.offis.de/simulators/DataSeriesSim)

        """
        if num_schedules == 0:
            return []

        # compute slot duration and number of datapoints per slot:
        slot_duration = duration / num_slots
        datapoints_per_slot = slot_duration / resolution

        # create a latin hypercube design:
        plan = doe.lhs(num_slots, samples=num_schedules)

        # translate each row of the sampled plan into a schedule by
        # denormalizing it and by repeating each value
        # *datapoints_per_slot* times:
        test_schedules = []
        for i in range(num_schedules):
            schedule = plan[i]
            schedule = min + (max - min) * schedule
            schedule = numpy.repeat(schedule, datapoints_per_slot)
            schedule = list(schedule)
            test_schedules.append(schedule)
        return test_schedules
