"""Utility functions."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import import_module
from itertools import groupby, zip_longest
from numbers import Integral
from typing import Any, cast, Literal, overload
import subprocess
import re
from packaging.version import Version

from polanyi.data import atomic_numbers, atomic_symbols


def is_min_xtb_version(min_version: str) -> bool:
    """Check if the used version of xtb is greater than or equal to a given version.

    Args:
        min_version: Minimum version of xtb required. Either:
            - 'X.Y.Z' for version number
            - 'bleed' for bleeding edge version

    Returns:
        True if the used xtb version is at least the min_version, False otherwise.
    """
    try:
        out = subprocess.run(
            ["xtb", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout
    except Exception:
        return False

    # Search for version and commit hash in xtb version output
    match = re.search(r"version\s*([\d\.]+)(?:\s*\((\w+)\))?", out)
    version, commit = match.groups()
    version = Version(version)

    # For bleeding edge version -> TODO: remove when xtb >6.7.1 is released
    # Currently correspond to "6.7.1" and with commit hash not "edcfbbe" (otherwise it is the official 6.7.1 release)
    if min_version == "bleed":
        if version >= Version("6.7.1"):
            if commit != "edcfbbe":
                return True
        return False

    elif version >= Version(min_version):
        return True

    return False


@dataclass
class Import:
    """Class for handling optional dependency imports."""

    module: str
    item: str | None = None
    alias: str | None = None


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
    elements: Iterable[Iterable[int] | Iterable[str]],
) -> bool:
    """Check whether atom types and length of elements is consistent.

    Note that the elements need to be given with consistent representation (str or int).

    Args:
        elements: An iterable of iterables of elements (atomic symbols or numbers)

    Returns:
        True if all elements match, False otherwise
    """
    return all(all_equal(i) for i in zip_longest(*elements))


# Disable black formatting for the overloaded functions otherwise it conflicts with flake8
# fmt: off

@overload
def convert_elements(
    elements: Iterable[int] | Iterable[str], output: Literal["numbers"]
) -> list[int]:
    ...


@overload
def convert_elements(
    elements: Iterable[int] | Iterable[str], output: Literal["symbols"]
) -> list[str]:
    ...

# fmt: on


def convert_elements(
    elements: Iterable[int] | Iterable[str], output: str = "numbers"
) -> list[int] | list[str]:
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
