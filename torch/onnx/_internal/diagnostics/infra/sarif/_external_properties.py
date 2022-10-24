# DO NOT EDIT! This file was generated by jschema_to_python version 0.0.1.dev29,
# with extension for dataclasses and type annotation.

from __future__ import annotations

import dataclasses
from typing import List, Optional

from typing_extensions import Literal

from torch.onnx._internal.diagnostics.infra.sarif import (
    _address,
    _artifact,
    _conversion,
    _graph,
    _invocation,
    _logical_location,
    _property_bag,
    _result,
    _thread_flow_location,
    _tool_component,
    _web_request,
    _web_response,
)


@dataclasses.dataclass
class ExternalProperties(object):
    """The top-level element of an external property file."""

    addresses: Optional[List[_address.Address]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "addresses"}
    )
    artifacts: Optional[List[_artifact.Artifact]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "artifacts"}
    )
    conversion: Optional[_conversion.Conversion] = dataclasses.field(
        default=None, metadata={"schema_property_name": "conversion"}
    )
    driver: Optional[_tool_component.ToolComponent] = dataclasses.field(
        default=None, metadata={"schema_property_name": "driver"}
    )
    extensions: Optional[List[_tool_component.ToolComponent]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "extensions"}
    )
    externalized_properties: Optional[_property_bag.PropertyBag] = dataclasses.field(
        default=None, metadata={"schema_property_name": "externalizedProperties"}
    )
    graphs: Optional[List[_graph.Graph]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "graphs"}
    )
    guid: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "guid"}
    )
    invocations: Optional[List[_invocation.Invocation]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "invocations"}
    )
    logical_locations: Optional[
        List[_logical_location.LogicalLocation]
    ] = dataclasses.field(
        default=None, metadata={"schema_property_name": "logicalLocations"}
    )
    policies: Optional[List[_tool_component.ToolComponent]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "policies"}
    )
    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(
        default=None, metadata={"schema_property_name": "properties"}
    )
    results: Optional[List[_result.Result]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "results"}
    )
    run_guid: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "runGuid"}
    )
    schema: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "schema"}
    )
    taxonomies: Optional[List[_tool_component.ToolComponent]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "taxonomies"}
    )
    thread_flow_locations: Optional[
        List[_thread_flow_location.ThreadFlowLocation]
    ] = dataclasses.field(
        default=None, metadata={"schema_property_name": "threadFlowLocations"}
    )
    translations: Optional[List[_tool_component.ToolComponent]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "translations"}
    )
    version: Optional[Literal["2.1.0"]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "version"}
    )
    web_requests: Optional[List[_web_request.WebRequest]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "webRequests"}
    )
    web_responses: Optional[List[_web_response.WebResponse]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "webResponses"}
    )


# flake8: noqa
