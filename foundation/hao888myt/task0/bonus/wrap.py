from collections.abc import Callable
from typing import Any


def my_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        print("函数执行前")
        result = func(*args, **kwargs)
        print("函数执行后")
        return result

    return wrapper


@my_decorator
def say_hello(name: str) -> None:
    print(f"Hello, {name}!")


say_hello("张三")
