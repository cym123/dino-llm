from typing import Callable, Generic, Iterable, List, TypeVar


T = TypeVar("T")



class Registry(Generic[T]):

    def __init__(self, type: str):
        self._registry = {}
        self._type = type

 
    def register(self, name: str) -> Callable[[T], None]:
        if name in self._registry:
            raise KeyError(f"{self._type} '{name}' 已经被注册过了！")

        def decorator(item: T) -> None:
            self._registry[name] = item

        return decorator


    def __getitem__(self, name: str) -> T:
        if name not in self._registry:
            raise KeyError(f"不支持的{self._type}类型：{name}")

        return self._registry[name]


    def supported_names(self) -> List[str]:
        return list(self._registry.keys())


    def assert_supported(self, names: str | Iterable[str]) -> None:
        if isinstance(names, str):
            names = [names]

        for name in names:
            if name not in self._registry:
                from argparse import ArgumentTypeError
                raise ArgumentTypeError(
                    f"不支持的{self._type}:{name}.\n"
                    f"支持的选项:{self.supported_names()}"
                )