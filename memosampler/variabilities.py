class Variability():
    """This class serves as a superclass for all classes which specify the
    variability of variables. It should not be invoked."""

    def __init__(self):
        """
        :raises:
            Raises an exception becuase this class is not supposed to be instantiated.
        """
        raise Exception('Variability should not be invoked.')

    def is_constant(self):
        """

        :return:
            *True* iff this variability is constant
        """
        return isinstance(self, Constant)


    def denormalize(self, value):
        """

        :param value:
        :return:
        """
        raise Exception('Variability should not be invoked.')


    def __repr__(self):
        """

        :return: str
            a readable representation of this object for debugging purposes
        """
        return '%s %s' % (self.__class__.__name__, self.__dict__)


class Constant(Variability):
    """Represents a constant value. Variables with constant variability
    will not be varied during the sampling process. The user may provide any
    object as *value*."""

    def __init__(self, value):
        """

        :param value: object
            A valueobject that will be passed to the simulator during the sampling process.
        """
        self.value = value

    def denormalize(self, value):
        return value


class RangeOfIntegers(Variability):
    """
    This class should be used to represent the variability of integer parameters that may be varied between a *min* and
    a *max* value.
    """

    def __init__(self, min, max):
        """

        :param min: int
            the minimal allowed value that may be used during the sampling process.

        :param max: int
            the maximal allowed value that may be used during the sampling process.

        """
        self.min = min
        self.max = max

    def denormalize(self, value):
        raise Exception('not yet implemented')


class RangeOfRealNumbers(Variability):
    """
    This class should be used to represent the variability of real-value parameters that may be varied between a *min*
    and a *max* value.
    """

    def __init__(self, min, max):
        """

        :param min: float
            the minimal allowed value that may be used during the sampling process.

        :param max: float
            the maximal allowed value that may be used during the sampling process.
        """
        self.min = min
        self.max = max

    def denormalize(self, value):
        return self.min + (self.max - self.min) * value


class NumericalLevels(Variability):
    """
    This class should be used to represent the variability of parameters that may take on only *numbers* in the
    specified list of allowed *levels*.
    """

    def __init__(self, levels):
        self.levels = levels

    def denormalize(self, value):
        return self.levels[int(value)]


class NonNumericalLevels(Variability):
    """
    This class should be used to represent the variability of parameters that may take on only *objects* in the
    specified list of allowed *levels*.
    """

    def __init__(self, levels):
        """

        :param levels: object[1..*]
            a list of objects which may be used by the sampler during the sampling process.
        """
        self.levels = levels

    def denormalize(self, value):
        raise Exception('not yet implemented')