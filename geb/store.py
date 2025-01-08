from pathlib import Path
from operator import attrgetter

import numpy as np


class StoreArray:
    __slots__ = ["_data", "_n", "_extra_dims_names"]

    def __init__(
        self,
        input_array=None,
        n=None,
        max_n=None,
        extra_dims=None,
        extra_dims_names=[],
        dtype=None,
        fill_value=None,
    ):
        self.extra_dims_names = np.array(extra_dims_names, dtype=str)

        if input_array is None and dtype is None:
            raise ValueError("Either input_array or dtype must be given")
        elif input_array is not None and dtype is not None:
            raise ValueError("Only one of input_array or dtype can be given")

        if input_array is not None:
            assert (
                extra_dims is None
            ), "extra_dims cannot be given if input_array is given"
            assert n is None, "n cannot be given if input_array is given"
            # assert dtype is not object
            assert input_array.dtype != object, "dtype cannot be object"
            n = input_array.shape[0]
            if max_n:
                if input_array.ndim == 1:
                    shape = max_n
                else:
                    shape = (max_n, *input_array.shape[1:])
                self._data = np.empty_like(input_array, shape=shape)
                n = input_array.shape[0]
                self._n = n
                self._data[:n] = input_array
            else:
                self._data = input_array
                self._n = n
        else:
            assert dtype is not None
            assert dtype is not object
            assert n is not None
            assert max_n is not None
            if extra_dims is None:
                shape = max_n
            else:
                shape = (max_n,) + extra_dims
                assert self.extra_dims_names is not None
                assert len(extra_dims) == len(self.extra_dims_names)

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

    @property
    def extra_dims_names(self):
        return self._extra_dims_names

    @extra_dims_names.setter
    def extra_dims_names(self, value):
        self._extra_dims_names = value

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
            input_.data if isinstance(input_, StoreArray) else input_
            for input_ in inputs
        )
        result = self._data.__array_ufunc__(ufunc, method, *modified_inputs, **kwargs)
        if method == "reduce":
            return result
        elif not isinstance(inputs[0], StoreArray):
            return result
        else:
            return self.__class__(result, max_n=self._data.shape[0])

    def __array_function__(self, func, types, args, kwargs):
        # Explicitly call __array_function__ of the underlying NumPy array
        modified_args = tuple(
            arg.data if isinstance(arg, StoreArray) else arg for arg in args
        )
        modified_types = tuple(
            type(arg.data) if isinstance(arg, StoreArray) else type(arg) for arg in args
        )
        return self._data.__array_function__(
            func, modified_types, modified_args, kwargs
        )

    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __repr__(self):
        return "StoreArray(" + self.data.__str__() + ")"

    def __str__(self):
        return self.data.__str__()

    def __len__(self):
        return self._n

    def __getattr__(self, name):
        if name in (
            "_data",
            "data",
            "_n",
            "n",
            "_extra_dims_names",
            "extra_dims_names",
        ):
            return super().__getattr__(name)
        else:
            return getattr(self.data, name)

    def __setattr__(self, name, value):
        if name in (
            "_data",
            "data",
            "_n",
            "n",
            "_extra_dims_names",
            "extra_dims_names",
        ):
            super().__setattr__(name, value)
        else:
            setattr(self.data, name, value)

    def __getstate__(self):
        return self.data.__getstate__()

    def __setstate__(self, state):
        self.data.__setstate__(state)

    def __sizeof__(self):
        return self.data.__sizeof__()

    def _perform_operation(self, other, operation: str, inplace: bool = False):
        if isinstance(other, StoreArray):
            other = other._data[: other._n]
        fn = getattr(self.data, operation)
        if other is None:
            args = ()
        else:
            args = (other,)
        result = fn(*args)
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
        if isinstance(value, StoreArray):
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

    def __neg__(self):
        return self._perform_operation(None, "__neg__")

    def __pos__(self):
        return self._perform_operation(None, "__pos__")

    def __invert__(self):
        return self._perform_operation(None, "__invert__")

    def save(self, path):
        np.savez_compressed(
            path.with_suffix(".storearray.npz"),
            **{slot: getattr(self, slot) for slot in self.__slots__},
        )

    @classmethod
    def load(cls, path):
        assert path.suffixes == [".storearray", ".npz"]
        with np.load(path) as data:
            obj = cls.__new__(cls)
            for slot in cls.__slots__:
                setattr(obj, slot, data[slot])
            return obj


class Bucket:
    def __init__(self):
        pass

    def __setattr__(self, name, value):
        assert isinstance(value, (StoreArray, int, float, np.ndarray))
        super().__setattr__(name, value)

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        for name, value in self.__dict__.items():
            if isinstance(value, StoreArray):
                value.save(path / name)
            elif isinstance(value, np.ndarray):
                np.savez_compressed(
                    (path / name).with_suffix(".array.npz"), value=value
                )
            else:
                np.save((path / name).with_suffix(".npy"), value)

    def load(self, path):
        for filename in path.iterdir():
            if filename.suffixes == [".storearray", ".npz"]:
                setattr(
                    self,
                    filename.name.removesuffix("".join(filename.suffixes)),
                    StoreArray.load(filename),
                )
            elif filename.suffixes == [".array", ".npz"]:
                setattr(
                    self,
                    filename.name.removesuffix("".join(filename.suffixes)),
                    np.load(filename)["value"],
                )
            else:
                setattr(self, filename.stem, np.load(filename).item())

        return self


class Store:
    def __init__(self, model):
        self.model = model
        self.buckets = {}

    def get_name(self, cls):
        return cls.__class__.__module__.replace("geb.", "")

    def create_bucket(self, cls):
        name = self.get_name(cls)
        assert name not in self.buckets
        bucket = Bucket()
        self.buckets[name] = bucket
        return bucket

    def get_bucket(self, cls):
        name = self.get_name(cls)
        return self.buckets[name]

    def save(self):
        for name, bucket in self.buckets.items():
            bucket.save(self.path / name)

    def load(self):
        for bucket_folder in self.path.iterdir():
            print("remove this")
            if bucket_folder.stem == "grid":
                continue
            bucket = Bucket().load(bucket_folder)

            self.buckets[bucket_folder.name] = bucket

            attrgetter(bucket_folder.name)(self.model).bucket = bucket

    @property
    def path(self):
        return Path(self.model.initial_conditions_folder)
