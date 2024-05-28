import numpy as np


class AgentArray:
    def __init__(
        self,
        input_array=None,
        n=None,
        max_n=None,
        extra_dims=None,
        dtype=None,
        fill_value=None,
    ):
        if input_array is None and dtype is None:
            raise ValueError("Either input_array or dtype must be given")
        elif input_array is not None and dtype is not None:
            raise ValueError("Only one of input_array or dtype can be given")

        if input_array is not None:
            assert (
                extra_dims is None
            ), "extra_dims cannot be given if input_array is given"
            if n is None and max_n is None:
                raise ValueError("Either n or max_n must be given")
            elif n is not None and max_n is not None:
                raise ValueError("Only one of n or max_n can be given")
            if max_n:
                if input_array.ndim == 1:
                    shape = max_n
                else:
                    shape = (max_n, *input_array.shape[1:])
                if isinstance(shape, tuple):
                    if shape[0] >  input_array.shape[0]:
                        self._data = np.empty_like(input_array, shape = shape)
                    else:
                        self._data = np.empty_like(input_array)
                else: 
                    if shape >  input_array.shape[0]:
                        self._data = np.empty_like(input_array, shape = shape)
                    else:
                        self._data = np.empty_like(input_array)
                n = input_array.shape[0]
                self._n = n
                self._data[:n] = input_array
            elif n:
                self._data = input_array
                self._n = n
        else:
            assert dtype is not None
            assert n is not None
            assert max_n is not None
            if extra_dims is None:
                shape = max_n
            else:
                shape = (max_n,) + extra_dims
            if fill_value is not None:
                self._data = np.full(shape, fill_value, dtype=dtype)
            else:
                self._data = np.empty(shape, dtype=dtype)
            self._n = n

    @property
    def data(self):
        return self._data[: self.n]

    @data.setter
    def data(self, value):
        self._data[: self.n] = value

    @property
    def max_n(self):
        return self._data.shape[0]

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        if value > self.max_n:
            raise ValueError("n cannot exceed max_n")
        self._n = value

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __array__(self, dtype=None):
        return np.asarray(self._data, dtype=dtype)

    def __array_interface__(self):
        return self._data.__array_interface__()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        modified_inputs = tuple(
            input_.data if isinstance(input_, AgentArray) else input_
            for input_ in inputs
        )
        result = self._data.__array_ufunc__(ufunc, method, *modified_inputs, **kwargs)
        if method == "reduce":
            return result
        elif not isinstance(inputs[0], AgentArray):
            return result
        else:
            return self.__class__(result, max_n=self._data.shape[0])

    def __array_function__(self, func, types, args, kwargs):
        # Explicitly call __array_function__ of the underlying NumPy array
        modified_args = tuple(
            arg.data if isinstance(arg, AgentArray) else arg for arg in args
        )
        modified_types = tuple(
            type(arg.data) if isinstance(arg, AgentArray) else type(arg) for arg in args
        )
        return self._data.__array_function__(
            func, modified_types, modified_args, kwargs
        )

    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __repr__(self):
        return "AgentArray(" + self.data.__str__() + ")"

    def __str__(self):
        return self.data.__str__()

    def __len__(self):
        return self._n

    def __getattr__(self, name):
        if name in ("_data", "data", "_n", "n"):
            return super().__getattr__(name)
        else:
            return getattr(self.data, name)

    def __setattr__(self, name, value):
        if name in ("_data", "data", "_n", "n"):
            super().__setattr__(name, value)
        else:
            setattr(self.data, name, value)

    def __getstate__(self):
        return self.data.__getstate__()

    def __setstate__(self, state):
        self.data.__setstate__(state)

    def __sizeof__(self):
        return self.data.__sizeof__()

    def __add__(self, other):
        if isinstance(other, AgentArray):
            other = other._data[: other._n]
        return self.__class__(self.data.__add__(other), max_n=self._data.shape[0])

    def _perform_operation(self, other, operation: str, inplace: bool = False):
        if isinstance(other, AgentArray):
            other = other._data[: other._n]
        result = getattr(self.data, operation)(other)
        if inplace:
            self.data = result
            return self
        else:
            return self.__class__(result, max_n=self._data.shape[0])

    def __add__(self, other):
        return self._perform_operation(other, "__add__")

    def __radd__(self, other):
        return self._perform_operation(other, "__radd__")

    def __iadd__(self, other):
        return self._perform_operation(other, "__add__", inplace=True)

    def __sub__(self, other):
        return self._perform_operation(other, "__sub__")

    def __rsub__(self, other):
        return self._perform_operation(other, "__rsub__")

    def __isub__(self, other):
        return self._perform_operation(other, "__sub__", inplace=True)

    def __mul__(self, other):
        return self._perform_operation(other, "__mul__")

    def __rmul__(self, other):
        return self._perform_operation(other, "__rmul__")

    def __imul__(self, other):
        return self._perform_operation(other, "__mul__", inplace=True)

    def __truediv__(self, other):
        return self._perform_operation(other, "__truediv__")

    def __rtruediv__(self, other):
        return self._perform_operation(other, "__rtruediv__")

    def __itruediv__(self, other):
        return self._perform_operation(other, "__truediv__", inplace=True)

    def __floordiv__(self, other):
        return self._perform_operation(other, "__floordiv__")

    def __rfloordiv__(self, other):
        return self._perform_operation(other, "__rfloordiv__")

    def __ifloordiv__(self, other):
        return self._perform_operation(other, "__floordiv__", inplace=True)

    def __mod__(self, other):
        return self._perform_operation(other, "__mod__")

    def __rmod__(self, other):
        return self._perform_operation(other, "__rmod__")

    def __imod__(self, other):
        return self._perform_operation(other, "__mod__", inplace=True)

    def __pow__(self, other):
        return self._perform_operation(other, "__pow__")

    def __rpow__(self, other):
        return self._perform_operation(other, "__rpow__")

    def __ipow__(self, other):
        return self._perform_operation(other, "__pow__", inplace=True)

    def _compare(self, value: object, operation: str) -> bool:
        if isinstance(value, AgentArray):
            return self.__class__(
                getattr(self.data, operation)(value.data), max_n=self._data.shape[0]
            )
        return getattr(self.data, operation)(value)

    def __eq__(self, value: object) -> bool:
        return self._compare(value, "__eq__")

    def __ne__(self, value: object) -> bool:
        return self._compare(value, "__ne__")

    def __gt__(self, value: object) -> bool:
        return self._compare(value, "__gt__")

    def __ge__(self, value: object) -> bool:
        return self._compare(value, "__ge__")

    def __lt__(self, value: object) -> bool:
        return self._compare(value, "__lt__")

    def __le__(self, value: object) -> bool:
        return self._compare(value, "__le__")

    def __and__(self, other):
        return self._perform_operation(other, "__and__")

    def __or__(self, other):
        return self._perform_operation(other, "__or__")
