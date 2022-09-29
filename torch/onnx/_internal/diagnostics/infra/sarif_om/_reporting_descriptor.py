# DO NOT EDIT! This file was generated by jschema_to_python version 0.0.1.dev29,
# with extension for dataclasses and type annotation.

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class ReportingDescriptor(object):
    """Metadata that describes a specific report produced by the tool, as part of the analysis it provides or its runtime reporting."""

    id: Any = dataclasses.field(metadata={"schema_property_name": "id"})
    default_configuration: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "defaultConfiguration"}
    )
    deprecated_guids: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "deprecatedGuids"}
    )
    deprecated_ids: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "deprecatedIds"}
    )
    deprecated_names: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "deprecatedNames"}
    )
    full_description: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "fullDescription"}
    )
    guid: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "guid"}
    )
    help: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "help"}
    )
    help_uri: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "helpUri"}
    )
    message_strings: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "messageStrings"}
    )
    name: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "name"}
    )
    properties: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "properties"}
    )
    relationships: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "relationships"}
    )
    short_description: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "shortDescription"}
    )


# flake8: noqa
