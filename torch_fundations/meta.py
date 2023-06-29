from abc import ABCMeta
from typing import (
    Any,
    Dict,
    Tuple,
    Type,
)

from dataclasses import dataclass
import yaml


class Available(ABCMeta):
    __available_registry__: Dict[str, Type] = {}

    def __new__(
        cls,
        name: str,
        bases: Tuple[Type, ...],
        attrs: Dict[str, Any],
        target: Type,
    ):
        if name != "Configurable":
            bases = (Available, *bases)
            attrs["__target__"] = target
            attrs["__annotations__"].update(target.__annotations__.copy())

        mcls = super().__new__(cls, name, bases, attrs)
        return dataclass(mcls)

    def save(self, path: str) -> None:
        config = self.__dict__.copy()
        config["class"] = self.__class__.__name__
        yaml.dump(config, open(path, "w+"), yaml.Dumper)

    @classmethod
    def load(cls, path: str) -> Any:
        config = yaml.load(open(path, "r"), yaml.Loader)
        mcls: Type = cls.__available_registry__[config["class"]]
        config_obj: Available = mcls(**config)
        target_obj_cls = getattr(config_obj, "__target__")
        target_obj = target_obj_cls(**config_obj.__dict__)
        return target_obj

    @property
    def keys(self) -> Tuple[str, ...]:
        return tuple(self.__available_registry__.keys())

    def __getitem__(self, key: str) -> Type:
        return self.__available_registry__[key]

    def __contains__(self, key: str) -> bool:
        return key in self.__available_registry__

    def __len__(self) -> int:
        return len(self.__available_registry__)
