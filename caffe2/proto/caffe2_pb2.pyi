"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor = ...

global___DeviceTypeProto = DeviceTypeProto
class _DeviceTypeProto(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[DeviceTypeProto.V], builtins.type):
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor = ...
    PROTO_CPU = DeviceTypeProto.V(0)
    PROTO_CUDA = DeviceTypeProto.V(1)
    PROTO_MKLDNN = DeviceTypeProto.V(2)
    PROTO_OPENGL = DeviceTypeProto.V(3)
    PROTO_OPENCL = DeviceTypeProto.V(4)
    PROTO_IDEEP = DeviceTypeProto.V(5)
    PROTO_HIP = DeviceTypeProto.V(6)
    PROTO_FPGA = DeviceTypeProto.V(7)
    PROTO_MSNPU = DeviceTypeProto.V(8)
    PROTO_XLA = DeviceTypeProto.V(9)
    PROTO_COMPILE_TIME_MAX_DEVICE_TYPES = DeviceTypeProto.V(10)
class DeviceTypeProto(metaclass=_DeviceTypeProto):
    V = typing.NewType('V', builtins.int)
PROTO_CPU = DeviceTypeProto.V(0)
PROTO_CUDA = DeviceTypeProto.V(1)
PROTO_MKLDNN = DeviceTypeProto.V(2)
PROTO_OPENGL = DeviceTypeProto.V(3)
PROTO_OPENCL = DeviceTypeProto.V(4)
PROTO_IDEEP = DeviceTypeProto.V(5)
PROTO_HIP = DeviceTypeProto.V(6)
PROTO_FPGA = DeviceTypeProto.V(7)
PROTO_MSNPU = DeviceTypeProto.V(8)
PROTO_XLA = DeviceTypeProto.V(9)
PROTO_COMPILE_TIME_MAX_DEVICE_TYPES = DeviceTypeProto.V(10)

class TensorProto(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    class _DataType(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[DataType.V], builtins.type):
        DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor = ...
        UNDEFINED = TensorProto.DataType.V(0)
        FLOAT = TensorProto.DataType.V(1)
        INT32 = TensorProto.DataType.V(2)
        BYTE = TensorProto.DataType.V(3)
        STRING = TensorProto.DataType.V(4)
        BOOL = TensorProto.DataType.V(5)
        UINT8 = TensorProto.DataType.V(6)
        INT8 = TensorProto.DataType.V(7)
        UINT16 = TensorProto.DataType.V(8)
        INT16 = TensorProto.DataType.V(9)
        INT64 = TensorProto.DataType.V(10)
        FLOAT16 = TensorProto.DataType.V(12)
        DOUBLE = TensorProto.DataType.V(13)
        ZERO_COLLISION_HASH = TensorProto.DataType.V(14)
        REBATCHING_BUFFER = TensorProto.DataType.V(15)
    class DataType(metaclass=_DataType):
        V = typing.NewType('V', builtins.int)
    UNDEFINED = TensorProto.DataType.V(0)
    FLOAT = TensorProto.DataType.V(1)
    INT32 = TensorProto.DataType.V(2)
    BYTE = TensorProto.DataType.V(3)
    STRING = TensorProto.DataType.V(4)
    BOOL = TensorProto.DataType.V(5)
    UINT8 = TensorProto.DataType.V(6)
    INT8 = TensorProto.DataType.V(7)
    UINT16 = TensorProto.DataType.V(8)
    INT16 = TensorProto.DataType.V(9)
    INT64 = TensorProto.DataType.V(10)
    FLOAT16 = TensorProto.DataType.V(12)
    DOUBLE = TensorProto.DataType.V(13)
    ZERO_COLLISION_HASH = TensorProto.DataType.V(14)
    REBATCHING_BUFFER = TensorProto.DataType.V(15)

    class Segment(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
        BEGIN_FIELD_NUMBER: builtins.int
        END_FIELD_NUMBER: builtins.int
        begin: builtins.int = ...
        end: builtins.int = ...

        def __init__(self,
            *,
            begin : typing.Optional[builtins.int] = ...,
            end : typing.Optional[builtins.int] = ...,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal[u"begin",b"begin",u"end",b"end"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal[u"begin",b"begin",u"end",b"end"]) -> None: ...

    DIMS_FIELD_NUMBER: builtins.int
    DATA_TYPE_FIELD_NUMBER: builtins.int
    FLOAT_DATA_FIELD_NUMBER: builtins.int
    INT32_DATA_FIELD_NUMBER: builtins.int
    BYTE_DATA_FIELD_NUMBER: builtins.int
    STRING_DATA_FIELD_NUMBER: builtins.int
    DOUBLE_DATA_FIELD_NUMBER: builtins.int
    INT64_DATA_FIELD_NUMBER: builtins.int
    RAW_DATA_FIELD_NUMBER: builtins.int
    NAME_FIELD_NUMBER: builtins.int
    DEVICE_DETAIL_FIELD_NUMBER: builtins.int
    SEGMENT_FIELD_NUMBER: builtins.int
    dims: google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int] = ...
    data_type: global___TensorProto.DataType.V = ...
    float_data: google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float] = ...
    int32_data: google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int] = ...
    byte_data: builtins.bytes = ...
    string_data: google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.bytes] = ...
    double_data: google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float] = ...
    int64_data: google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int] = ...
    raw_data: builtins.bytes = ...
    name: typing.Text = ...

    @property
    def device_detail(self) -> global___DeviceOption: ...

    @property
    def segment(self) -> global___TensorProto.Segment: ...

    def __init__(self,
        *,
        dims : typing.Optional[typing.Iterable[builtins.int]] = ...,
        data_type : typing.Optional[global___TensorProto.DataType.V] = ...,
        float_data : typing.Optional[typing.Iterable[builtins.float]] = ...,
        int32_data : typing.Optional[typing.Iterable[builtins.int]] = ...,
        byte_data : typing.Optional[builtins.bytes] = ...,
        string_data : typing.Optional[typing.Iterable[builtins.bytes]] = ...,
        double_data : typing.Optional[typing.Iterable[builtins.float]] = ...,
        int64_data : typing.Optional[typing.Iterable[builtins.int]] = ...,
        raw_data : typing.Optional[builtins.bytes] = ...,
        name : typing.Optional[typing.Text] = ...,
        device_detail : typing.Optional[global___DeviceOption] = ...,
        segment : typing.Optional[global___TensorProto.Segment] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"byte_data",b"byte_data",u"data_type",b"data_type",u"device_detail",b"device_detail",u"name",b"name",u"raw_data",b"raw_data",u"segment",b"segment"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"byte_data",b"byte_data",u"data_type",b"data_type",u"device_detail",b"device_detail",u"dims",b"dims",u"double_data",b"double_data",u"float_data",b"float_data",u"int32_data",b"int32_data",u"int64_data",b"int64_data",u"name",b"name",u"raw_data",b"raw_data",u"segment",b"segment",u"string_data",b"string_data"]) -> None: ...
global___TensorProto = TensorProto

class QTensorProto(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    DIMS_FIELD_NUMBER: builtins.int
    PRECISION_FIELD_NUMBER: builtins.int
    SCALE_FIELD_NUMBER: builtins.int
    BIAS_FIELD_NUMBER: builtins.int
    IS_SIGNED_FIELD_NUMBER: builtins.int
    DATA_FIELD_NUMBER: builtins.int
    NAME_FIELD_NUMBER: builtins.int
    DATA_TYPE_FIELD_NUMBER: builtins.int
    SCALES_FIELD_NUMBER: builtins.int
    BIASES_FIELD_NUMBER: builtins.int
    AXIS_FIELD_NUMBER: builtins.int
    IS_MULTIPARAM_FIELD_NUMBER: builtins.int
    dims: google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int] = ...
    precision: builtins.int = ...
    scale: builtins.float = ...
    bias: builtins.float = ...
    is_signed: builtins.bool = ...
    data: google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int] = ...
    name: typing.Text = ...
    data_type: global___TensorProto.DataType.V = ...
    scales: google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float] = ...
    biases: google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float] = ...
    axis: builtins.int = ...
    is_multiparam: builtins.bool = ...

    def __init__(self,
        *,
        dims : typing.Optional[typing.Iterable[builtins.int]] = ...,
        precision : typing.Optional[builtins.int] = ...,
        scale : typing.Optional[builtins.float] = ...,
        bias : typing.Optional[builtins.float] = ...,
        is_signed : typing.Optional[builtins.bool] = ...,
        data : typing.Optional[typing.Iterable[builtins.int]] = ...,
        name : typing.Optional[typing.Text] = ...,
        data_type : typing.Optional[global___TensorProto.DataType.V] = ...,
        scales : typing.Optional[typing.Iterable[builtins.float]] = ...,
        biases : typing.Optional[typing.Iterable[builtins.float]] = ...,
        axis : typing.Optional[builtins.int] = ...,
        is_multiparam : typing.Optional[builtins.bool] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"axis",b"axis",u"bias",b"bias",u"data_type",b"data_type",u"is_multiparam",b"is_multiparam",u"is_signed",b"is_signed",u"name",b"name",u"precision",b"precision",u"scale",b"scale"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"axis",b"axis",u"bias",b"bias",u"biases",b"biases",u"data",b"data",u"data_type",b"data_type",u"dims",b"dims",u"is_multiparam",b"is_multiparam",u"is_signed",b"is_signed",u"name",b"name",u"precision",b"precision",u"scale",b"scale",u"scales",b"scales"]) -> None: ...
global___QTensorProto = QTensorProto

class TensorProtos(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    PROTOS_FIELD_NUMBER: builtins.int

    @property
    def protos(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___TensorProto]: ...

    def __init__(self,
        *,
        protos : typing.Optional[typing.Iterable[global___TensorProto]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"protos",b"protos"]) -> None: ...
global___TensorProtos = TensorProtos

class TensorShape(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    DIMS_FIELD_NUMBER: builtins.int
    DATA_TYPE_FIELD_NUMBER: builtins.int
    UNKNOWN_DIMS_FIELD_NUMBER: builtins.int
    UNKNOWN_SHAPE_FIELD_NUMBER: builtins.int
    NAME_FIELD_NUMBER: builtins.int
    dims: google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int] = ...
    data_type: global___TensorProto.DataType.V = ...
    unknown_dims: google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int] = ...
    unknown_shape: builtins.bool = ...
    name: typing.Text = ...

    def __init__(self,
        *,
        dims : typing.Optional[typing.Iterable[builtins.int]] = ...,
        data_type : typing.Optional[global___TensorProto.DataType.V] = ...,
        unknown_dims : typing.Optional[typing.Iterable[builtins.int]] = ...,
        unknown_shape : typing.Optional[builtins.bool] = ...,
        name : typing.Optional[typing.Text] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"data_type",b"data_type",u"name",b"name",u"unknown_shape",b"unknown_shape"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"data_type",b"data_type",u"dims",b"dims",u"name",b"name",u"unknown_dims",b"unknown_dims",u"unknown_shape",b"unknown_shape"]) -> None: ...
global___TensorShape = TensorShape

class TensorShapes(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    SHAPES_FIELD_NUMBER: builtins.int

    @property
    def shapes(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___TensorShape]: ...

    def __init__(self,
        *,
        shapes : typing.Optional[typing.Iterable[global___TensorShape]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"shapes",b"shapes"]) -> None: ...
global___TensorShapes = TensorShapes

class TensorBoundShape(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    class _DimType(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[DimType.V], builtins.type):
        DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor = ...
        UNKNOWN = TensorBoundShape.DimType.V(0)
        CONSTANT = TensorBoundShape.DimType.V(1)
        BATCH = TensorBoundShape.DimType.V(2)
        BATCH_OF_FEATURE_MAX = TensorBoundShape.DimType.V(3)
        BATCH_OF_FEATURE_MAX_DEFAULT = TensorBoundShape.DimType.V(4)
        FEATURE_MAX = TensorBoundShape.DimType.V(5)
        FEATURE_MAX_DEFAULT = TensorBoundShape.DimType.V(6)
    class DimType(metaclass=_DimType):
        V = typing.NewType('V', builtins.int)
    UNKNOWN = TensorBoundShape.DimType.V(0)
    CONSTANT = TensorBoundShape.DimType.V(1)
    BATCH = TensorBoundShape.DimType.V(2)
    BATCH_OF_FEATURE_MAX = TensorBoundShape.DimType.V(3)
    BATCH_OF_FEATURE_MAX_DEFAULT = TensorBoundShape.DimType.V(4)
    FEATURE_MAX = TensorBoundShape.DimType.V(5)
    FEATURE_MAX_DEFAULT = TensorBoundShape.DimType.V(6)

    SHAPE_FIELD_NUMBER: builtins.int
    DIM_TYPE_FIELD_NUMBER: builtins.int
    NAME_FIELD_NUMBER: builtins.int
    SHAPE_IS_FINAL_FIELD_NUMBER: builtins.int
    dim_type: google.protobuf.internal.containers.RepeatedScalarFieldContainer[global___TensorBoundShape.DimType.V] = ...
    name: typing.Text = ...
    shape_is_final: builtins.bool = ...

    @property
    def shape(self) -> global___TensorShape: ...

    def __init__(self,
        *,
        shape : typing.Optional[global___TensorShape] = ...,
        dim_type : typing.Optional[typing.Iterable[global___TensorBoundShape.DimType.V]] = ...,
        name : typing.Optional[typing.Text] = ...,
        shape_is_final : typing.Optional[builtins.bool] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"name",b"name",u"shape",b"shape",u"shape_is_final",b"shape_is_final"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"dim_type",b"dim_type",u"name",b"name",u"shape",b"shape",u"shape_is_final",b"shape_is_final"]) -> None: ...
global___TensorBoundShape = TensorBoundShape

class TensorBoundShapes(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    SHAPES_FIELD_NUMBER: builtins.int
    MAX_BATCH_SIZE_FIELD_NUMBER: builtins.int
    MAX_FEATURE_LEN_FIELD_NUMBER: builtins.int
    max_batch_size: builtins.int = ...
    max_feature_len: builtins.int = ...

    @property
    def shapes(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___TensorBoundShape]: ...

    def __init__(self,
        *,
        shapes : typing.Optional[typing.Iterable[global___TensorBoundShape]] = ...,
        max_batch_size : typing.Optional[builtins.int] = ...,
        max_feature_len : typing.Optional[builtins.int] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"max_batch_size",b"max_batch_size",u"max_feature_len",b"max_feature_len"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"max_batch_size",b"max_batch_size",u"max_feature_len",b"max_feature_len",u"shapes",b"shapes"]) -> None: ...
global___TensorBoundShapes = TensorBoundShapes

class AOTConfig(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    MAX_BATCH_SIZE_FIELD_NUMBER: builtins.int
    MAX_SEQ_SIZE_FIELD_NUMBER: builtins.int
    IN_BATCH_BROADCAST_FIELD_NUMBER: builtins.int
    ONNXIFI_BLACKLIST_OPS_FIELD_NUMBER: builtins.int
    ONNXIFI_MIN_OPS_FIELD_NUMBER: builtins.int
    max_batch_size: builtins.int = ...
    max_seq_size: builtins.int = ...
    in_batch_broadcast: builtins.bool = ...
    onnxifi_blacklist_ops: typing.Text = ...
    onnxifi_min_ops: builtins.int = ...

    def __init__(self,
        *,
        max_batch_size : typing.Optional[builtins.int] = ...,
        max_seq_size : typing.Optional[builtins.int] = ...,
        in_batch_broadcast : typing.Optional[builtins.bool] = ...,
        onnxifi_blacklist_ops : typing.Optional[typing.Text] = ...,
        onnxifi_min_ops : typing.Optional[builtins.int] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"in_batch_broadcast",b"in_batch_broadcast",u"max_batch_size",b"max_batch_size",u"max_seq_size",b"max_seq_size",u"onnxifi_blacklist_ops",b"onnxifi_blacklist_ops",u"onnxifi_min_ops",b"onnxifi_min_ops"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"in_batch_broadcast",b"in_batch_broadcast",u"max_batch_size",b"max_batch_size",u"max_seq_size",b"max_seq_size",u"onnxifi_blacklist_ops",b"onnxifi_blacklist_ops",u"onnxifi_min_ops",b"onnxifi_min_ops"]) -> None: ...
global___AOTConfig = AOTConfig

class Argument(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    NAME_FIELD_NUMBER: builtins.int
    F_FIELD_NUMBER: builtins.int
    I_FIELD_NUMBER: builtins.int
    S_FIELD_NUMBER: builtins.int
    T_FIELD_NUMBER: builtins.int
    N_FIELD_NUMBER: builtins.int
    FLOATS_FIELD_NUMBER: builtins.int
    INTS_FIELD_NUMBER: builtins.int
    STRINGS_FIELD_NUMBER: builtins.int
    TENSORS_FIELD_NUMBER: builtins.int
    NETS_FIELD_NUMBER: builtins.int
    QTENSORS_FIELD_NUMBER: builtins.int
    name: typing.Text = ...
    f: builtins.float = ...
    i: builtins.int = ...
    s: builtins.bytes = ...
    floats: google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float] = ...
    ints: google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int] = ...
    strings: google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.bytes] = ...

    @property
    def t(self) -> global___TensorProto: ...

    @property
    def n(self) -> global___NetDef: ...

    @property
    def tensors(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___TensorProto]: ...

    @property
    def nets(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___NetDef]: ...

    @property
    def qtensors(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___QTensorProto]: ...

    def __init__(self,
        *,
        name : typing.Optional[typing.Text] = ...,
        f : typing.Optional[builtins.float] = ...,
        i : typing.Optional[builtins.int] = ...,
        s : typing.Optional[builtins.bytes] = ...,
        t : typing.Optional[global___TensorProto] = ...,
        n : typing.Optional[global___NetDef] = ...,
        floats : typing.Optional[typing.Iterable[builtins.float]] = ...,
        ints : typing.Optional[typing.Iterable[builtins.int]] = ...,
        strings : typing.Optional[typing.Iterable[builtins.bytes]] = ...,
        tensors : typing.Optional[typing.Iterable[global___TensorProto]] = ...,
        nets : typing.Optional[typing.Iterable[global___NetDef]] = ...,
        qtensors : typing.Optional[typing.Iterable[global___QTensorProto]] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"f",b"f",u"i",b"i",u"n",b"n",u"name",b"name",u"s",b"s",u"t",b"t"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"f",b"f",u"floats",b"floats",u"i",b"i",u"ints",b"ints",u"n",b"n",u"name",b"name",u"nets",b"nets",u"qtensors",b"qtensors",u"s",b"s",u"strings",b"strings",u"t",b"t",u"tensors",b"tensors"]) -> None: ...
global___Argument = Argument

class DeviceOption(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    DEVICE_TYPE_FIELD_NUMBER: builtins.int
    DEVICE_ID_FIELD_NUMBER: builtins.int
    RANDOM_SEED_FIELD_NUMBER: builtins.int
    NODE_NAME_FIELD_NUMBER: builtins.int
    NUMA_NODE_ID_FIELD_NUMBER: builtins.int
    EXTRA_INFO_FIELD_NUMBER: builtins.int
    device_type: builtins.int = ...
    device_id: builtins.int = ...
    random_seed: builtins.int = ...
    node_name: typing.Text = ...
    numa_node_id: builtins.int = ...
    extra_info: google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text] = ...

    def __init__(self,
        *,
        device_type : typing.Optional[builtins.int] = ...,
        device_id : typing.Optional[builtins.int] = ...,
        random_seed : typing.Optional[builtins.int] = ...,
        node_name : typing.Optional[typing.Text] = ...,
        numa_node_id : typing.Optional[builtins.int] = ...,
        extra_info : typing.Optional[typing.Iterable[typing.Text]] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"device_id",b"device_id",u"device_type",b"device_type",u"node_name",b"node_name",u"numa_node_id",b"numa_node_id",u"random_seed",b"random_seed"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"device_id",b"device_id",u"device_type",b"device_type",u"extra_info",b"extra_info",u"node_name",b"node_name",u"numa_node_id",b"numa_node_id",u"random_seed",b"random_seed"]) -> None: ...
global___DeviceOption = DeviceOption

class OperatorDef(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    INPUT_FIELD_NUMBER: builtins.int
    OUTPUT_FIELD_NUMBER: builtins.int
    NAME_FIELD_NUMBER: builtins.int
    TYPE_FIELD_NUMBER: builtins.int
    ARG_FIELD_NUMBER: builtins.int
    DEVICE_OPTION_FIELD_NUMBER: builtins.int
    ENGINE_FIELD_NUMBER: builtins.int
    CONTROL_INPUT_FIELD_NUMBER: builtins.int
    IS_GRADIENT_OP_FIELD_NUMBER: builtins.int
    DEBUG_INFO_FIELD_NUMBER: builtins.int
    DOMAIN_FIELD_NUMBER: builtins.int
    OP_VERSION_FIELD_NUMBER: builtins.int
    input: google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text] = ...
    output: google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text] = ...
    name: typing.Text = ...
    type: typing.Text = ...
    engine: typing.Text = ...
    control_input: google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text] = ...
    is_gradient_op: builtins.bool = ...
    debug_info: typing.Text = ...
    domain: typing.Text = ...
    op_version: builtins.int = ...

    @property
    def arg(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Argument]: ...

    @property
    def device_option(self) -> global___DeviceOption: ...

    def __init__(self,
        *,
        input : typing.Optional[typing.Iterable[typing.Text]] = ...,
        output : typing.Optional[typing.Iterable[typing.Text]] = ...,
        name : typing.Optional[typing.Text] = ...,
        type : typing.Optional[typing.Text] = ...,
        arg : typing.Optional[typing.Iterable[global___Argument]] = ...,
        device_option : typing.Optional[global___DeviceOption] = ...,
        engine : typing.Optional[typing.Text] = ...,
        control_input : typing.Optional[typing.Iterable[typing.Text]] = ...,
        is_gradient_op : typing.Optional[builtins.bool] = ...,
        debug_info : typing.Optional[typing.Text] = ...,
        domain : typing.Optional[typing.Text] = ...,
        op_version : typing.Optional[builtins.int] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"debug_info",b"debug_info",u"device_option",b"device_option",u"domain",b"domain",u"engine",b"engine",u"is_gradient_op",b"is_gradient_op",u"name",b"name",u"op_version",b"op_version",u"type",b"type"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"arg",b"arg",u"control_input",b"control_input",u"debug_info",b"debug_info",u"device_option",b"device_option",u"domain",b"domain",u"engine",b"engine",u"input",b"input",u"is_gradient_op",b"is_gradient_op",u"name",b"name",u"op_version",b"op_version",u"output",b"output",u"type",b"type"]) -> None: ...
global___OperatorDef = OperatorDef

class MapFieldEntry(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    KEY_FIELD_NUMBER: builtins.int
    VAL_FIELD_NUMBER: builtins.int
    key: typing.Text = ...
    val: typing.Text = ...

    def __init__(self,
        *,
        key : typing.Optional[typing.Text] = ...,
        val : typing.Optional[typing.Text] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"key",b"key",u"val",b"val"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"key",b"key",u"val",b"val"]) -> None: ...
global___MapFieldEntry = MapFieldEntry

class BackendOptions(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    BACKEND_NAME_FIELD_NUMBER: builtins.int
    OPTION_FIELD_NUMBER: builtins.int
    backend_name: typing.Text = ...

    @property
    def option(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___MapFieldEntry]: ...

    def __init__(self,
        *,
        backend_name : typing.Optional[typing.Text] = ...,
        option : typing.Optional[typing.Iterable[global___MapFieldEntry]] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"backend_name",b"backend_name"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"backend_name",b"backend_name",u"option",b"option"]) -> None: ...
global___BackendOptions = BackendOptions

class PartitionInfo(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    NAME_FIELD_NUMBER: builtins.int
    DEVICE_ID_FIELD_NUMBER: builtins.int
    EXTRA_INFO_FIELD_NUMBER: builtins.int
    BACKEND_OPTIONS_FIELD_NUMBER: builtins.int
    name: typing.Text = ...
    device_id: google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int] = ...
    extra_info: typing.Text = ...

    @property
    def backend_options(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___BackendOptions]: ...

    def __init__(self,
        *,
        name : typing.Optional[typing.Text] = ...,
        device_id : typing.Optional[typing.Iterable[builtins.int]] = ...,
        extra_info : typing.Optional[typing.Text] = ...,
        backend_options : typing.Optional[typing.Iterable[global___BackendOptions]] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"extra_info",b"extra_info",u"name",b"name"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"backend_options",b"backend_options",u"device_id",b"device_id",u"extra_info",b"extra_info",u"name",b"name"]) -> None: ...
global___PartitionInfo = PartitionInfo

class NetDef(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    NAME_FIELD_NUMBER: builtins.int
    OP_FIELD_NUMBER: builtins.int
    TYPE_FIELD_NUMBER: builtins.int
    NUM_WORKERS_FIELD_NUMBER: builtins.int
    DEVICE_OPTION_FIELD_NUMBER: builtins.int
    ARG_FIELD_NUMBER: builtins.int
    EXTERNAL_INPUT_FIELD_NUMBER: builtins.int
    EXTERNAL_OUTPUT_FIELD_NUMBER: builtins.int
    PARTITION_INFO_FIELD_NUMBER: builtins.int
    name: typing.Text = ...
    type: typing.Text = ...
    num_workers: builtins.int = ...
    external_input: google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text] = ...
    external_output: google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text] = ...

    @property
    def op(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___OperatorDef]: ...

    @property
    def device_option(self) -> global___DeviceOption: ...

    @property
    def arg(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Argument]: ...

    @property
    def partition_info(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___PartitionInfo]: ...

    def __init__(self,
        *,
        name : typing.Optional[typing.Text] = ...,
        op : typing.Optional[typing.Iterable[global___OperatorDef]] = ...,
        type : typing.Optional[typing.Text] = ...,
        num_workers : typing.Optional[builtins.int] = ...,
        device_option : typing.Optional[global___DeviceOption] = ...,
        arg : typing.Optional[typing.Iterable[global___Argument]] = ...,
        external_input : typing.Optional[typing.Iterable[typing.Text]] = ...,
        external_output : typing.Optional[typing.Iterable[typing.Text]] = ...,
        partition_info : typing.Optional[typing.Iterable[global___PartitionInfo]] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"device_option",b"device_option",u"name",b"name",u"num_workers",b"num_workers",u"type",b"type"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"arg",b"arg",u"device_option",b"device_option",u"external_input",b"external_input",u"external_output",b"external_output",u"name",b"name",u"num_workers",b"num_workers",u"op",b"op",u"partition_info",b"partition_info",u"type",b"type"]) -> None: ...
global___NetDef = NetDef

class ExecutionStep(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    NAME_FIELD_NUMBER: builtins.int
    SUBSTEP_FIELD_NUMBER: builtins.int
    NETWORK_FIELD_NUMBER: builtins.int
    NUM_ITER_FIELD_NUMBER: builtins.int
    CRITERIA_NETWORK_FIELD_NUMBER: builtins.int
    REPORT_NET_FIELD_NUMBER: builtins.int
    REPORT_INTERVAL_FIELD_NUMBER: builtins.int
    RUN_EVERY_MS_FIELD_NUMBER: builtins.int
    CONCURRENT_SUBSTEPS_FIELD_NUMBER: builtins.int
    SHOULD_STOP_BLOB_FIELD_NUMBER: builtins.int
    ONLY_ONCE_FIELD_NUMBER: builtins.int
    CREATE_WORKSPACE_FIELD_NUMBER: builtins.int
    NUM_CONCURRENT_INSTANCES_FIELD_NUMBER: builtins.int
    name: typing.Text = ...
    network: google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text] = ...
    num_iter: builtins.int = ...
    criteria_network: typing.Text = ...
    report_net: typing.Text = ...
    report_interval: builtins.int = ...
    run_every_ms: builtins.int = ...
    concurrent_substeps: builtins.bool = ...
    should_stop_blob: typing.Text = ...
    only_once: builtins.bool = ...
    create_workspace: builtins.bool = ...
    num_concurrent_instances: builtins.int = ...

    @property
    def substep(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___ExecutionStep]: ...

    def __init__(self,
        *,
        name : typing.Optional[typing.Text] = ...,
        substep : typing.Optional[typing.Iterable[global___ExecutionStep]] = ...,
        network : typing.Optional[typing.Iterable[typing.Text]] = ...,
        num_iter : typing.Optional[builtins.int] = ...,
        criteria_network : typing.Optional[typing.Text] = ...,
        report_net : typing.Optional[typing.Text] = ...,
        report_interval : typing.Optional[builtins.int] = ...,
        run_every_ms : typing.Optional[builtins.int] = ...,
        concurrent_substeps : typing.Optional[builtins.bool] = ...,
        should_stop_blob : typing.Optional[typing.Text] = ...,
        only_once : typing.Optional[builtins.bool] = ...,
        create_workspace : typing.Optional[builtins.bool] = ...,
        num_concurrent_instances : typing.Optional[builtins.int] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"concurrent_substeps",b"concurrent_substeps",u"create_workspace",b"create_workspace",u"criteria_network",b"criteria_network",u"name",b"name",u"num_concurrent_instances",b"num_concurrent_instances",u"num_iter",b"num_iter",u"only_once",b"only_once",u"report_interval",b"report_interval",u"report_net",b"report_net",u"run_every_ms",b"run_every_ms",u"should_stop_blob",b"should_stop_blob"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"concurrent_substeps",b"concurrent_substeps",u"create_workspace",b"create_workspace",u"criteria_network",b"criteria_network",u"name",b"name",u"network",b"network",u"num_concurrent_instances",b"num_concurrent_instances",u"num_iter",b"num_iter",u"only_once",b"only_once",u"report_interval",b"report_interval",u"report_net",b"report_net",u"run_every_ms",b"run_every_ms",u"should_stop_blob",b"should_stop_blob",u"substep",b"substep"]) -> None: ...
global___ExecutionStep = ExecutionStep

class PlanDef(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    NAME_FIELD_NUMBER: builtins.int
    NETWORK_FIELD_NUMBER: builtins.int
    EXECUTION_STEP_FIELD_NUMBER: builtins.int
    name: typing.Text = ...

    @property
    def network(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___NetDef]: ...

    @property
    def execution_step(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___ExecutionStep]: ...

    def __init__(self,
        *,
        name : typing.Optional[typing.Text] = ...,
        network : typing.Optional[typing.Iterable[global___NetDef]] = ...,
        execution_step : typing.Optional[typing.Iterable[global___ExecutionStep]] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"name",b"name"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"execution_step",b"execution_step",u"name",b"name",u"network",b"network"]) -> None: ...
global___PlanDef = PlanDef

class BlobProto(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    NAME_FIELD_NUMBER: builtins.int
    TYPE_FIELD_NUMBER: builtins.int
    TENSOR_FIELD_NUMBER: builtins.int
    CONTENT_FIELD_NUMBER: builtins.int
    QTENSOR_FIELD_NUMBER: builtins.int
    CONTENT_NUM_CHUNKS_FIELD_NUMBER: builtins.int
    CONTENT_CHUNK_ID_FIELD_NUMBER: builtins.int
    name: typing.Text = ...
    type: typing.Text = ...
    content: builtins.bytes = ...
    content_num_chunks: builtins.int = ...
    content_chunk_id: builtins.int = ...

    @property
    def tensor(self) -> global___TensorProto: ...

    @property
    def qtensor(self) -> global___QTensorProto: ...

    def __init__(self,
        *,
        name : typing.Optional[typing.Text] = ...,
        type : typing.Optional[typing.Text] = ...,
        tensor : typing.Optional[global___TensorProto] = ...,
        content : typing.Optional[builtins.bytes] = ...,
        qtensor : typing.Optional[global___QTensorProto] = ...,
        content_num_chunks : typing.Optional[builtins.int] = ...,
        content_chunk_id : typing.Optional[builtins.int] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"content",b"content",u"content_chunk_id",b"content_chunk_id",u"content_num_chunks",b"content_num_chunks",u"name",b"name",u"qtensor",b"qtensor",u"tensor",b"tensor",u"type",b"type"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"content",b"content",u"content_chunk_id",b"content_chunk_id",u"content_num_chunks",b"content_num_chunks",u"name",b"name",u"qtensor",b"qtensor",u"tensor",b"tensor",u"type",b"type"]) -> None: ...
global___BlobProto = BlobProto

class DBReaderProto(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    NAME_FIELD_NUMBER: builtins.int
    SOURCE_FIELD_NUMBER: builtins.int
    DB_TYPE_FIELD_NUMBER: builtins.int
    KEY_FIELD_NUMBER: builtins.int
    name: typing.Text = ...
    source: typing.Text = ...
    db_type: typing.Text = ...
    key: typing.Text = ...

    def __init__(self,
        *,
        name : typing.Optional[typing.Text] = ...,
        source : typing.Optional[typing.Text] = ...,
        db_type : typing.Optional[typing.Text] = ...,
        key : typing.Optional[typing.Text] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"db_type",b"db_type",u"key",b"key",u"name",b"name",u"source",b"source"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"db_type",b"db_type",u"key",b"key",u"name",b"name",u"source",b"source"]) -> None: ...
global___DBReaderProto = DBReaderProto

DeviceType = DeviceTypeProto

# These are freedom-patched into caffe2_pb2 in caffe2/proto/__init__.py
CPU: DeviceType = DeviceType.PROTO_CPU
CUDA: DeviceType = DeviceType.PROTO_CUDA
MKLDNN: DeviceType = DeviceType.PROTO_MKLDNN
OPENGL: DeviceType = DeviceType.PROTO_OPENGL
OPENCL: DeviceType = DeviceType.PROTO_OPENCL
IDEEP: DeviceType = DeviceType.PROTO_IDEEP
HIP: DeviceType = DeviceType.PROTO_HIP
COMPILE_TIME_MAX_DEVICE_TYPES: DeviceType = DeviceType.PROTO_COMPILE_TIME_MAX_DEVICE_TYPES
