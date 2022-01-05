"""Utility functions."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import import_module
from itertools import groupby, zip_longest
from numbers import Integral
from typing import Any, cast, Literal, Optional, overload, Union

from polanyi.data import atomic_numbers, atomic_symbols


@dataclass
class Import:
    """Class for handling optional dependency imports."""

    module: str
    item: Optional[str] = None
    alias: Optional[str] = None


def requires_dependency(  # noqa: C901
    imports: list[Import], _globals: dict
) -> Callable[..., Callable]:
    """Decorator factory to control optional dependencies.

    Args:
        imports: Imports
        _globals: Global symbol table from calling module.

    Returns:
        decorator: Either 'noop_decorator' that returns the original function or
            'error_decorator' that raises an ImportError and lists absent dependencies.
    """

    def noop_decorator(function: Callable[..., Any]) -> Callable[..., Any]:
        """Returns function unchanged."""
        return function

    def error_decorator(function: Callable[..., Any]) -> Callable[..., Any]:
        """Raises error."""

        def error(*args, **kwargs) -> ImportError:
            error_msg = "Install extra requirements to use this function:"
            for e in import_errors:
                error_msg += f" {e.name}"
            raise ImportError(error_msg)

        return error

    import_errors = []
    for imp in imports:
        # Import module
        try:
            module = import_module(imp.module)

            # Try to import item as attribute
            if imp.item is not None:
                try:
                    item = getattr(module, imp.item)
                except AttributeError:
                    item = import_module(f"{imp.module}.{imp.item}")
                name = imp.item
            else:
                item = module
                name = imp.module

            # Convert item name to alias
            if imp.alias is not None:
                name = imp.alias

            _globals[name] = item
        except ImportError as import_error:
            import_errors.append(import_error)

    return error_decorator if len(import_errors) > 0 else noop_decorator


def all_equal(iterable: Iterable) -> bool:
    """Returns True if all the elements are equal to each other."""
    g = groupby(iterable)
    try:
        next(g)
    except StopIteration as e:
        raise ValueError("Empty iterable.") from e
    try:
        next(g)
        return False
    except StopIteration:
        return True


def validate_atom_order(
    elements: Iterable[Union[Iterable[int], Iterable[str]]]
) -> bool:
    """Check whether atom types and length of elements is consistent.

    Note that the elements need to be given with consistent representation (str or int).

    Args:
        elements: An iterable of iterables of elements (atomic symbols or numbers)

    Returns:
        True if all elements match, False otherwise
    """
    return all(all_equal(i) for i in zip_longest(*elements))


@overload
def convert_elements(
    elements: Union[Iterable[int], Iterable[str]], output: Literal["numbers"]
) -> list[int]:
    ...


@overload
def convert_elements(
    elements: Union[Iterable[int], Iterable[str]], output: Literal["symbols"]
) -> list[str]:
    ...


def convert_elements(
    elements: Union[Iterable[int], Iterable[str]], output: str = "numbers"
) -> Union[list[int], list[str]]:
    """Converts elements to atomic symbols or numbers.

    Args:
        elements: Elements as atomic symbols or numbers
        output: Output format: 'numbers' (default) or 'symbols'.

    Returns:
        elements: Converted elements

    Raises:
        TypeError: When input type not supported
        ValueError: When output not supported
    """
    if output not in ["numbers", "symbols"]:
        raise ValueError(f"ouput={output} not supported. Use 'numbers' or 'symbols'")

    if all(isinstance(element, str) for element in elements):
        elements = cast("list[str]", elements)
        if output == "numbers":
            elements = [atomic_numbers[element.capitalize()] for element in elements]
        return elements
    elif all(isinstance(element, Integral) for element in elements):
        elements = cast("list[int]", elements)
        if output == "symbols":
            elements = [atomic_symbols[element] for element in elements]
        return elements
    else:
        raise TypeError("elements must be all integers or all strings.")
