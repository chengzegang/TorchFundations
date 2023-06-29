from abc import ABCMeta
from typing import (
    Any,
    Callable,
    Dict,
    Tuple,
    Type,
)

from dataclasses import dataclass
import yaml
import inspect


class Available(ABCMeta):
    __available_registry__: Dict[Tuple[str, str], Type] = {}
    task: str = ""
    model: Type = object

    def __new__(
        cls,
        name: str,
        bases: Tuple[Type, ...],
        attrs: Dict[str, Any],
        task: str,
        model: Type,
    ):
        if name != "Available":
            bases = (cls, *bases)
            attrs["task"] = task
            attrs["model"] = model
            attrs["__annotations__"] = {}
            attrs["__annotations__"].update(model.__annotations__.copy())
            cls.__available_registry__[(task, model.__name__)] = model

        mcls = super().__new__(cls, name, bases, attrs)
        return dataclass(mcls)

    def save(self, path: str) -> None:
        config = self.__dict__.copy()
        yaml.dump(config, open(path, "w+"), yaml.Dumper)

    @classmethod
    def load(cls, path: str) -> Any:
        config = yaml.load(open(path, "r"), yaml.FullLoader)
        mcls: Type = cls.__available_registry__[(config["task"], config["model"])]
        arg_names = inspect.getfullargspec(mcls).args
        found_args = {}
        for key in config.keys():
            if key in arg_names:
                found_args[key] = config[key]
        target_obj = mcls(**found_args)
        return target_obj

    @property
    def keys(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(self.__available_registry__.keys())

    def __getitem__(self, key: Tuple[str, str]) -> Type:
        return self.__available_registry__[key]

    def __contains__(self, key: str) -> bool:
        return key in self.__available_registry__

    def __len__(self) -> int:
        return len(self.__available_registry__)
