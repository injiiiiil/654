# DO NOT EDIT! This file was generated by jschema_to_python version 0.0.1.dev29,
# with extension for dataclasses and type annotation.

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class Suppression(object):
    """A suppression that is relevant to a result."""

    kind: Any = dataclasses.field(metadata={"schema_property_name": "kind"})
    guid: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "guid"}
    )
    justification: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "justification"}
    )
    location: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "location"}
    )
    properties: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "properties"}
    )
    state: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "state"}
    )


# flake8: noqa
