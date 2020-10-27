from collections.abc import Sequence

import numpy as np

__all__ = ['Multiple',
           'Nil']


class NilType(type):

    def __repr__(self):
        return self.__name__

    def __str__(self):
        return self.__name__


class Nil(object, metaclass=NilType):
    pass


class Multiple:

    def __new__(cls, modules):
        if isinstance(modules, Sequence) or isinstance(modules, np.ndarray):
            return super(Multiple, cls).__new__(cls)
        return modules

    def __init__(self, modules):
        if not isinstance(modules, np.ndarray):
            self.modules = np.array(modules, dtype=object)
        else:
            self.modules = modules

    def __len__(self):
        return len(self.modules)

    def __iter__(self):
        return iter(self.modules)

    def __getitem__(self, item):
        return Multiple(self.modules[item])

    def __getattr__(self, item):
        return Multiple(tuple(getattr(module, item) for module in self.modules))

    def __call__(self, *args, **kwargs):
        grouped_args = tuple(tuple(arg[i] for arg in args if arg[i] is not Nil)
                             for i in range(len(self.modules)))
        grouped_kwargs = tuple({key: vals[i] for key, vals in kwargs.items() if vals[i] is not Nil}
                               for i in range(len(self.modules)))
        outputs = tuple(module(*args, **kwargs) for module, args, kwargs
                        in zip(self.modules, grouped_args, grouped_kwargs))
        return Multiple(outputs)

    def __repr__(self):
        rep = f"{self.__class__.__name__}(\n\t"
        rep += repr(self.modules).replace('\n', '\n\t')
        rep += "\n)"
        return rep

    def t(self):
        self.modules = self.modules.T
        return self

    def sizes(self):
        return self.modules.shape

    @classmethod
    def from_generator(cls, generator):
        return cls(list(generator))
