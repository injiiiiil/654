#include "caffe2/operators/utility_ops.h"
#include <cmath>
#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

template <>
bool WeightedSumOp<CPUContext>::RunOnDevice() {
  return DoRunWithType<float>();
}

template <>
bool WeightedSumGradientOp<CPUContext>::RunOnDevice() {
  return DoRunWithType<float>();
}

REGISTER_CPU_OPERATOR(WallClockTime, WallClockTimeOp<CPUContext>);
REGISTER_CPU_OPERATOR(Print, PrintOp<CPUContext>);
REGISTER_CPU_OPERATOR(FlattenToVec, FlattenToVecOp<CPUContext>);
REGISTER_CPU_OPERATOR(Alias, AliasOp<CPUContext>);
REGISTER_CPU_OPERATOR(ResizeLike, ResizeLikeOp<CPUContext>);
REGISTER_CPU_OPERATOR(SumInt, SumOp<CPUContext>);
REGISTER_CPU_OPERATOR(WeightedSum, WeightedSumOp<CPUContext>);
REGISTER_CPU_OPERATOR(WeightedSumGradient, WeightedSumGradientOp<CPUContext>);
REGISTER_CPU_OPERATOR(
    ScatterWeightedSum,
    ScatterWeightedSumOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(ScatterAssign, ScatterAssignOp<CPUContext>);

// From CPU, copy it to whatever the current context
REGISTER_CPU_OPERATOR(
    CopyFromCPUInput,
    CopyOp<CPUContext, CPUContext, CPUContext>);
REGISTER_CPU_OPERATOR(
    CopyOnDeviceLike,
    CopyOnDeviceLikeOp<CPUContext, CPUContext, CPUContext>);
REGISTER_CPU_OPERATOR(Copy, CopyOp<CPUContext, CPUContext, CPUContext>);
REGISTER_CPU_OPERATOR(LengthsToShape, LengthsToShapeOp<CPUContext>);
REGISTER_CPU_OPERATOR(HasElements, HasElementsOp<CPUContext>);
REGISTER_CPU_OPERATOR(IsEmpty, IsEmptyOp<CPUContext>);
REGISTER_CPU_OPERATOR(GatherRanges, GatherRangesOp<CPUContext>);
REGISTER_CPU_OPERATOR(LengthsGather, LengthsGatherOp<CPUContext>);
REGISTER_CPU_OPERATOR(LengthsToSegmentIds, LengthsToSegmentIdsOp<CPUContext>);
REGISTER_CPU_OPERATOR(LengthsToRanges, LengthsToRangesOp<CPUContext>);
REGISTER_CPU_OPERATOR(SegmentIdsToLengths, SegmentIdsToLengthsOp<CPUContext>);
REGISTER_CPU_OPERATOR(SegmentIdsToRanges, SegmentIdsToRangesOp<CPUContext>);
REGISTER_CPU_OPERATOR(LengthsToWeights, LengthsToWeightsOp<CPUContext>);
REGISTER_CPU_OPERATOR(EnsureDense, EnsureDenseOp<CPUContext>);
REGISTER_CPU_OPERATOR(
    AccumulateHistogram,
    AccumulateHistogramOp<float, CPUContext>);

OPERATOR_SCHEMA(WallClockTime)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc("Time since epoch in nanoseconds.")
    .Output(0, "time", "The time in nanoseconds.");

REGISTER_CPU_OPERATOR(UnsafeCoalesce, UnsafeCoalesceOp<CPUContext>);

OPERATOR_SCHEMA(Print)
    .NumInputs(1)
    .NumOutputs(0)
    .SetDoc("Logs shape and contents of input tensor to stderr or to a file.")
    .Arg(
        "to_file",
        "(bool) if 1, saves contents to the root folder of the current "
        "workspace, appending the tensor contents to a file named after "
        "the blob name. Otherwise, logs to stderr.")
    .Arg(
        "limit",
        "(int, default 0) If set, prints the first `limit` elements of tensor. "
        "If 0, prints the first `k_limit_default`(1000) elements of tensor")
    .Arg(
        "every_n",
        "(int, default 1) Print tensor every `every_n` runs")
    .Input(0, "tensor", "The tensor to print.");

OPERATOR_SCHEMA(LengthsToShape)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
This operator takes a list of $N$ equal integers as input which represent the lengths of $N$ vectors. The output is the calculated shape of the matrix if the $N$ integers were combined into a single matrix.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.h
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "LengthsToShape",
    ["X"],
    ["Y"]
)

# Create X: Sample softmax output for 5-class model
X = np.array([2,2,2,2,2,2,2,2,2,2])
print("X:\n",X)

# Feed X into workspace
workspace.FeedBlob("X", X.astype(np.int32))

# Run op
workspace.RunOperatorOnce(op)

# Collect Output
print("Y:\n", workspace.FetchBlob("Y"))

```

**Result**

```

X:
 [2 2 2 2 2 2 2 2 2 2]
Y:
 [10  2]

```

</details>

    )DOC")
    .Input(
        0,
        "X",
        "List, of length $N$, of equal integers representing the lengths of several vectors.")
    .Output(
        0,
        "Y",
        "Vector of length 2 describing the dimensions of the data if the $N$ vectors from the input were combined to a single matrix.");
OPERATOR_SCHEMA(FlattenToVec)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /*def*/,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      int total = 1;
      for (auto d : in[0].dims()) {
        total *= d;
      }
      out[0].set_data_type(in[0].data_type());
      out[0].add_dims(total);
      return out;
    })
    .SetDoc(R"DOC(

The *FlattenToVec* op flattens the input tensor into a 1-D vector. The op accepts a single input tensor and returns a single output tensor.

Github Links:

- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.h


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "FlattenToVec",
    ["input"],
    ["output"],
)

workspace.FeedBlob("input", np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]).astype(np.float32))
print("input:\n", workspace.FetchBlob("input"))

workspace.RunOperatorOnce(op)
print("output: \n", workspace.FetchBlob("output"))

```

**Result**

```

input:
 [[ 1.  2.  3.]
 [ 4.  5.  6.]
 [ 7.  8.  9.]
 [10. 11. 12.]]
output:
 [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12.]

```

</details>

)DOC")
    .Input(0, "input", "A tensor of rank >= 1.")
    .Output(0, "output", "A tensor of rank 1 (vector) with the contents of the input tensor.");

OPERATOR_SCHEMA(Alias)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Makes the output and the input share the same underlying storage.

WARNING: in general, in caffe2's operator interface different tensors should
have different underlying storage, which is the assumption made by
components such as the dependency engine and memory optimization. Thus, in
normal situations you should not use the AliasOp, especially in a normal
forward-backward pass.

The Alias op is provided so one can achieve true asynchrony, such as
Hogwild, in a graph. But make sure you understand all the implications
similar to multi-thread computation before you use it explicitly.
)DOC")
    .Input(0, "input", "Input tensor whose storage will be shared.")
    .Output(0, "output", "Tensor of same shape as input, sharing its storage.");

OPERATOR_SCHEMA(ResizeLike)
    .NumInputs(2)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /*def*/,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      out.push_back(in[1]);
      out[0].set_data_type(in[0].data_type());
      return out;
    })
    .SetDoc(R"DOC(
Produces tensor containing data of first input and shape of second input.
)DOC")
    .Input(0, "data", "Tensor whose data will be copied into the output.")
    .Input(1, "shape_tensor", "Tensor whose shape will be applied to output.")
    .Output(0, "output", "Tensor with data of input 0 and shape of input 1.");

OPERATOR_SCHEMA(SumInt)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .InputsCanCrossDevices()
    .TensorInferenceFunction([](const OperatorDef& /*def*/,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      out.push_back(in[0]);
      out[0].set_data_type(TensorProto::INT32);
      return out;
    })
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(WeightedSum)
    .NumInputs([](int n) { return (n > 0 && n % 2 == 0); })
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShapeOfInput(0)
    .SetDoc(R"DOC(
Element-wise weighted sum of several data, weight tensor pairs.
Input should be in the form X_0, weight_0, X_1, weight_1, ... where X_i all
have the same shape, and weight_i are size 1 tensors that specifies the weight
of each vector. Note that if one wants to do in-place computation, it could
only be done with X_0 also as the output, but not other X_i.
)DOC")
    .Input(0, "data_0", "First of the input tensors.")
    .Input(0, "weight_0", "Weight of the first input in the sum.")
    .Output(0, "output", "Result containing weighted elem-wise sum of inputs.");

OPERATOR_SCHEMA(WeightedSumGradient)
    .NumInputs([](int n) { return (n > 0 && n % 2 == 1); })
    .NumOutputs(1, INT_MAX);

OPERATOR_SCHEMA(ScatterWeightedSum)
    .NumInputs([](int n) { return (n > 3 && (n - 3) % 2 == 0); })
    .NumOutputs(1)
    .EnforceInplace({{0, 0}})
    .SetDoc(R"DOC(
Similar to WeightedSum, computes the weighted sum of several tensors, with
the difference that inputs are sliced tensors. The first tensor has to be
in-place and only slices of it on the first dimension as indexed by INDICES
will be updated.

Note: The op pretty much ignores the exact shapes of the input arguments and
cares only about sizes. It's done for performance consideration to avoid
unnecessary reshapes. Only first dimension of X_0 is important, let's call it
N. If M is the total size of X_0 and K is the size of INDICES then X_i is
assumed to be of shape K x (M / N) regardless of the real shape.

Note: Each update in INDICES is applied independently which means that if
duplicated elements are present in INDICES the corresponding slice of X_0
will be scaled multiple times. Manual collapsing of INDICES is required
beforehand if necessary.

Note: Updates are applied sequentially by inputs which might have undesired
consequences if the input tensor is accessed concurrently by different op
(e.g. when doing Hogwild). Other threads might see intermediate results even
on individual slice level, e.g. X_0 scaled by weight_0 but without any
updates applied.

Currently only works on CPU because of access to INDICES.
)DOC")
    .Input(0, "X_0", "Tensor to be updated.")
    .Input(
        1,
        "Weight_0",
        "Scalar weight for X_0, applied only to slices affected.")
    .Input(
        2,
        "INDICES",
        "1-D list of indices on the first dimension of X_0 "
        "that need to be updated")
    .Input(3, "X_1", "Update slices, with shape len(INDICES) + shape(X_0)[1:]")
    .Input(4, "Weight_1", "Scalar weight for X_1 update")
    .Output(0, "X_0", "Has to be exactly the same tensor as the input 0")
    .EnforceInplace({{0, 0}});

OPERATOR_SCHEMA(ScatterAssign)
    .NumInputs(3)
    .NumOutputs(1)
    .EnforceInplace({{0, 0}})
    .SetDoc(R"DOC(
Update slices of the tensor in-place by overriding current value.

Note: The op pretty much ignores the exact shapes of the input arguments and
cares only about sizes. It's done for performance consideration to avoid
unnecessary reshapes. Only first dimension of X_0 is important, let's call it
N. If M is the total size of X_0 and K is the size of INDICES then X_i is
assumed to be of shape K x (M / N) regardless of the real shape.

Note: Each update in INDICES is applied independently which means that if
duplicated elements are present in INDICES arbitrary one will win.

Currently only works on CPU because of access to INDICES.
)DOC")
    .Input(0, "DATA", "Tensor to be updated.")
    .Input(
        1,
        "INDICES",
        "1-D list of indices on the first dimension"
        "of X_0 that need to be updated")
    .Input(
        2,
        "SLICES",
        "Update slices, with shape len(INDICES) + shape(X_0)[1:]")
    .Output(0, "DATA", "Has to be exactly the same tensor as the input 0");

OPERATOR_SCHEMA(Copy)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .InputsCanCrossDevices()
    .SetDoc(R"DOC(
Copy input tensor into output, potentially across devices.

Github Links:

- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.h


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Copy",
    ["input"],
    ["output"]
)

workspace.FeedBlob("input", np.random.rand(3,3))
print("input:", workspace.FetchBlob("input"))
workspace.RunOperatorOnce(op)
print("output:", workspace.FetchBlob("output"))

```

**Result**

```

input:
[[0.16826761 0.68168217 0.55196001]
 [0.19735483 0.34837823 0.69015595]
 [0.09448514 0.57390828 0.37097193]]
output:
[[0.16826761 0.68168217 0.55196001]
 [0.19735483 0.34837823 0.69015595]
 [0.09448514 0.57390828 0.37097193]]

```

</details>

)DOC")
    .Input(0, "input", "(*Tensor*): input tensor to copy")
    .Output(0, "output", "(*Tensor*): copy of input tensor");

OPERATOR_SCHEMA(CopyGPUToCPU)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .InputsCanCrossDevices()
    .DeviceInferenceFunction([](const OperatorDef& def) {
      CAFFE_ENFORCE(
          def.has_device_option(),
          "CopyGPUToCPU op should have cuda device option.");
      auto& cuda_option = def.device_option();
      auto cpu_option = DeviceOption();
      vector<DeviceOption> in_dev(def.input_size(), cuda_option);
      vector<DeviceOption> out_dev(def.output_size(), cpu_option);
      return std::make_pair(in_dev, out_dev);
    })
    .SetDoc(R"DOC(
Copy tensor for GPU to CPU context. Must be run under GPU device option.
)DOC")
    .Input(0, "input", "The input tensor.")
    .Output(0, "output", "Tensor that will contain a copy of the input.");

OPERATOR_SCHEMA(CopyCPUToGPU)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .InputsCanCrossDevices()
    .DeviceInferenceFunction([](const OperatorDef& def) {
      CAFFE_ENFORCE(
          def.has_device_option(),
          "CopyCPUToGPU op should have cuda device option.");
      auto& cuda_option = def.device_option();
      auto cpu_option = DeviceOption();
      vector<DeviceOption> in_dev(def.input_size(), cpu_option);
      vector<DeviceOption> out_dev(def.output_size(), cuda_option);
      return std::make_pair(in_dev, out_dev);
    })
    .SetDoc(R"DOC(
Copy tensor for CPU to GPU context. Must be run under GPU device option.
)DOC")
    .Input(0, "input", "The input tensor.")
    .Output(0, "output", "Tensor that will contain a copy of the input.");

OPERATOR_SCHEMA(CopyFromCPUInput)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .InputsCanCrossDevices()
    .DeviceInferenceFunction([](const OperatorDef& def) {
      auto op_device =
          def.has_device_option() ? def.device_option() : DeviceOption();
      auto cpu_option = DeviceOption();
      vector<DeviceOption> in_dev(def.input_size(), cpu_option);
      vector<DeviceOption> out_dev(def.output_size(), op_device);
      return std::make_pair(in_dev, out_dev);
    })
    .SetDoc(R"DOC(
Take a CPU input tensor and copy it to an output in the current
Context (GPU or CPU). This may involves cross-device MemCpy.
)DOC")
    .Input(0, "input", "The input CPU tensor.")
    .Output(0, "output", "either a TensorCUDA or a TensorCPU");

OPERATOR_SCHEMA(CopyOnDeviceLike)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc("Copy input tensor into output to the specific device.")
    .Input(0, "input", "The input tensor.")
    .Input(1, "dst", "Tensor, on which device the copy will be performed.")
    .Output(0, "output", "Tensor that will contain a copy of the input.");

OPERATOR_SCHEMA(HasElements)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
The *HasElements* op accepts a single input $tensor$, and produces a single boolean output $has\_elements$. The output is *True* if and only if $tensor$ has size > 0. Note, this op is the opposite of the *IsEmpty* op.

Github Links:

- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.h


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "HasElements",
    ["tensor"],
    ["has_elements"],
)

# Use a not-empty tensor
workspace.FeedBlob("tensor", np.random.randn(2, 2).astype(np.float32))
print("tensor:\n", workspace.FetchBlob("tensor"))

workspace.RunOperatorOnce(op)
print("has_elements: ", workspace.FetchBlob("has_elements"),"\n")

# Use an empty tensor
workspace.FeedBlob("tensor", np.empty(0))
print("tensor:\n", workspace.FetchBlob("tensor"))

workspace.RunOperatorOnce(op)
print("has_elements: ", workspace.FetchBlob("has_elements"))

```

**Result**

```

tensor:
 [[ 0.6116506  -0.54433197]
 [ 0.19406661 -0.7338629 ]]
has_elements:  True

tensor:
 []
has_elements:  False

```

</details>

)DOC")
    .Input(0, "tensor", "Input data tensor to check for elements.")
    .Output(0, "has_elements", "Output scalar boolean tensor. True if input has size > 0.");

OPERATOR_SCHEMA(IsEmpty)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
The *IsEmpty* op accepts a single input $tensor$, and produces a single boolean output $is\_empty$. The output is *True* if and only if $tensor$ has size == 0.

Github Links:

- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.h


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "IsEmpty",
    ["tensor"],
    ["is_empty"],
)

# Use a not-empty tensor
workspace.FeedBlob("tensor", np.random.randn(2, 2).astype(np.float32))
print("tensor:\n", workspace.FetchBlob("tensor"))

workspace.RunOperatorOnce(op)
print("is_empty: ", workspace.FetchBlob("is_empty"),"\n")

# Use an empty tensor
workspace.FeedBlob("tensor", np.empty(0))
print("tensor:\n", workspace.FetchBlob("tensor"))

workspace.RunOperatorOnce(op)
print("is_empty: ", workspace.FetchBlob("is_empty"))

```

**Result**

```

tensor:
 [[ 0.26018378  0.6778789 ]
 [-1.3097627  -0.40083608]]
is_empty:  False

tensor:
 []
is_empty:  True

```

</details>

)DOC")
    .ScalarType(::caffe2::TensorProto_DataType::TensorProto_DataType_BOOL)
    .Input(0, "tensor", "Input data tensor to check if empty.")
    .Output(0, "is_empty", "Output scalar boolean tensor. True if input has size == 0.");

OPERATOR_SCHEMA(GatherRanges)
    .NumInputs(2)
    .NumOutputs(2)
    .DisallowInputFillers()
    .SetDoc(R"DOC(
Given DATA tensor of rank 1, and RANGES tensor of rank 3, gather
corresponding ranges into a 1-D tensor OUTPUT.

RANGES dimentions description:
1: represents list of examples within a batch
2: represents list features
3: two values which are start and length or a range (to be applied on DATA)

Another output LENGTHS represents each example length within OUTPUT

Example:
  DATA  = [1, 2, 3, 4, 5, 6]
  RANGES = [
    [
      [0, 1],
      [2, 2],
    ],
    [
      [4, 1],
      [5, 1],
    ]
  ]
  OUTPUT = [1, 3, 4, 5, 6]
  LENGTHS = [3, 2]
)DOC")
    .Input(0, "DATA", "Tensor of rank 1.")
    .Input(
        1,
        "RANGES",
        "Tensor of int32/int64 ranges, of dims (N, M, 2). "
        "Where N is number of examples and M is a size of each example. "
        "Last dimension represents a range in the format (start, lengths)")
    .Output(0, "OUTPUT", "1-D tensor of size sum of range lengths")
    .Output(
        1,
        "LENGTHS",
        "1-D tensor of size N with lengths over gathered data"
        " for each row in a batch. sum(LENGTHS) == OUTPUT.size()")
    .TensorInferenceFunction([](const OperatorDef& /* unused */,
                                const vector<TensorShape>& in) {
      std::vector<TensorShape> out(2);

      int total = 1;
      for (auto d : in[0].dims()) {
        total *= d;
      }
      out[0].add_dims(total);
      out[0].set_data_type(in[0].data_type());
      out[1].add_dims(in[1].dims(0));
      out[1].set_data_type(in[1].data_type());
      return out;
    });

OPERATOR_SCHEMA(LengthsGather)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Gather items from sparse tensor. Sparse tensor is described by items and
lengths. This operator gathers items corresponding to lengths at the given
indices. This deliberately doesn't return lengths of OUTPUTS so that both lists
and maps can be supported without special cases. If you need lengths tensor for
 OUTPUT, use `Gather`.

Example:
  ITEMS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  LENGTHS = [0, 2, 3, 1, 4]
  INDICES = [0, 2, 4]

  OUTPUT = [2, 3, 4, 6, 7, 8, 9]
)DOC")
    .Input(0, "ITEMS", "items tensor")
    .Input(1, "LENGTHS", "lengths tensor")
    .Input(2, "INDICES", "indices into LENGTHS where items should be gathered")
    .Output(0, "OUTPUT", "1-D tensor containing gathered items");

OPERATOR_SCHEMA(LengthsToSegmentIds)
    .NumInputs(1)
    .NumOutputs(1)
    .DisallowInputFillers() // TODO: enable the filler
    .SetDoc(R"DOC(
Given a vector of segment lengths (*lengths*) the *LengthsToSegmentIds* op returns a zero-based, consecutive vector of segment ids (*segment_ids*). For example, *lengths=[1, 3, 0, 2]* will produce *segment_ids=[0, 1, 1, 1, 3, 3]*. In general, the inverse operation is *SegmentIdsToLengths*. Notice though that trailing empty sequence lengths can't be properly recovered from segment ids.

Github Links:

- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.h


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "LengthsToSegmentIds",
    ["lengths"],
    ["segment_ids"],
)

workspace.FeedBlob("lengths", np.array([1, 3, 0, 2]).astype(np.int32))
print("lengths:\n", workspace.FetchBlob("lengths"))

workspace.RunOperatorOnce(op)
print("segment_ids: \n", workspace.FetchBlob("segment_ids"))

```

**Result**

```

lengths:
 [1 3 0 2]
segment_ids:
 [0 1 1 1 3 3]

```

</details>

)DOC")
    .Input(0, "lengths", "1D tensor of int32 or int64 segment lengths.")
    .Output(0, "segment_ids", "1D tensor of length *sum(lengths)*");

OPERATOR_SCHEMA(LengthsToRanges)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* unused */,
                                const vector<TensorShape>& in) {
      vector<int> out_shape(in[0].dims().begin(), in[0].dims().end());
      out_shape.push_back(2);
      return vector<TensorShape>{
          CreateTensorShape(out_shape, in[0].data_type())};
    })
    .SetDoc(R"DOC(
Given a vector of segment lengths, calculates offsets of each segment and packs
them next to the lengths. For the input vector of length N the output is a Nx2
matrix with (offset, lengths) packaged for each segment.

For example, `[1, 3, 0, 2]` transforms into `[[0, 1], [1, 3], [4, 0], [4, 2]]`.
)DOC")
    .Input(0, "lengths", "1D tensor of int32 segment lengths.")
    .Output(
        0,
        "ranges",
        "2D tensor of shape len(lengths) X 2 and the same type as `lengths`");

OPERATOR_SCHEMA(SegmentIdsToLengths)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .DisallowInputFillers() // TODO: enable the filler
    .SetDoc(R"DOC(
Transfers a vector of segment ids to a vector of segment lengths. This operation
supports non-consecutive segment ids. Segments not appearing in the input vector
will have length 0. If the second input is provided, the number of segments =
the size of its first dimension. Otherwise, the number of segments = the last
index in the first input vector + 1.

In general, for consecutive, zero-based segment IDs, this is the inverse
operation of LengthsToSegmentIds, except that a vector of segment IDs
cannot represent empty segments at the end (if the second input is absent).
)DOC")
    .Input(0, "segment_ids", "1-D int32_t or int64_t tensor of segment ids")
    .Input(
        1,
        "data (optional)",
        "if provided, number of segments = the size of its first dimension")
    .Output(0, "lengths", "1-D int64_t tensor of segment lengths");

OPERATOR_SCHEMA(SegmentIdsToRanges)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .DisallowInputFillers() // TODO: enable the filler
    .SetDoc(R"DOC(
Transfers a vector of segment ids to a vector of segment ranges. This operation
supports non-consecutive segment ids. Segments not appearing in the input vector
will have length 0. If the second input is provided, the number of segments =
the size of its first dimension. Otherwise, the number of segments = the last
index in the first input vector + 1.
)DOC")
    .Input(0, "segment_ids", "1-D int32_t or int64_t tensor of segment ids")
    .Input(
        1,
        "data (optional)",
        "if provided, number of segments = the size of its first dimension")
    .Output(0, "lengths", "1-D int64_t tensor of segment lengths");

OPERATOR_SCHEMA(LengthsToWeights)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("power", "n of 1/pow(length,n) for normalization")
    .SetDoc(R"DOC(
Similar as LengthsToSegmentIds but output vector of segment
weights derived by lengths. i.e 1/pow(length, power)
)DOC")
    .Input(0, "lengths", "1-D int32_t or int64_t tensor of lengths")
    .Output(0, "a vector of weights", "1-D float tensor of weights by length");



SHOULD_NOT_DO_GRADIENT(WallClockTime);

OPERATOR_SCHEMA(UnsafeCoalesce)
    .NumInputsOutputs([](int inputs, int outputs) {
      return inputs + 1 == outputs;
    })
    .AllowInplace([](int input, int output) { return input == output; })
    .SetDoc(R"DOC(
Coalesce the N inputs into N outputs and a single coalesced output blob.

This allows operations that operate over multiple small kernels (e.g.
biases in a deep CNN) to be coalesced into a single larger operation,
amortizing the kernel launch overhead, synchronization costs for
distributed computation, etc.

The operator:

- computes the total size of the coalesced blob by summing the input sizes
- allocates the coalesced output blob as the total size
- copies the input vectors into the coalesced blob, at the correct offset.
- aliases each Output(i) to- point into the coalesced blob, at the corresponding offset for Input(i).

This is 'unsafe' as the output vectors are aliased, so use with
caution.

)DOC");

OPERATOR_SCHEMA(EnsureDense)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
This operator converts dense or sparse gradients to dense ones.
Therefore, sparse gradient can be back propagated to Operators that consume
dense gradients only (e.g., FCGradient).

The operator's behaviors:

- In forward, simply pass in place or copy input to the output.
- In backward, if the gradient passed-in is sparse gradient, change it to dense gradient in linear time; otherwise, simply pass the dense gradient.
)DOC")
    .Input(0, "input", "Input tensors.")
    .Output(0, "output", "Output tensor. Same dimension as inputs.");

OPERATOR_SCHEMA(AccumulateHistogram)
    .NumInputs(1)
    .NumOutputs(2)
    .SetDoc(R"DOC(
This operator calculate thes histogram of values in input tensor.
There're 2 outputs, one for histogram of current input tensor, and another
for histogram of the all input tensors accumulated through history.
The output would contain num_buckets + 2 values. index[1 ... num_buckets]
for values in [lower_bound, upper_bound) interval. And the rest 2 for values
smaller than lower_bound or greater than upper_bound respectively.
)DOC")
    .Input(0, "X", "Input tensor.")
    .Output(0, "CurHist", "Output histogram of the current tensor.")
    .Output(1, "AccHist", "Accumulated histogram of the history tensor.")
    .Arg("lower_bound", "the lower bound value")
    .Arg("upper_bound", "the upper bound value")
    .Arg(
        "num_buckets",
        "number of buckets to use in [lower_bound, upper_bound)");

class GetEnsureDenseGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE(
        GradOut(0).IsSparse() || GradOut(0).IsDense(),
        "Input gradient ",
        O(0),
        " should be either sparse or dense.");

    if (GradOut(0).IsDense()) {
      SetDense(0, GO(0));
      return vector<OperatorDef>();
    } else {
      return SingleGradientDef(
          "SparseToDense",
          "",
          vector<string>{GO_I(0), GO_V(0), I(0)},
          vector<string>{GI(0)});
    }
  }
};
REGISTER_GRADIENT(EnsureDense, GetEnsureDenseGradient);

SHOULD_NOT_DO_GRADIENT(Print);
SHOULD_NOT_DO_GRADIENT(HasElements);
SHOULD_NOT_DO_GRADIENT(IsEmpty);
SHOULD_NOT_DO_GRADIENT(LengthsToShape);
SHOULD_NOT_DO_GRADIENT(UnsafeCoalesce);

class GetAliasGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    // We will simply pass-along the gradient. Nothing needs to
    // be calculated.
    SetDense(0, GO(0));
    return vector<OperatorDef>();
  }
};
REGISTER_GRADIENT(Alias, GetAliasGradient);

SHOULD_NOT_DO_GRADIENT(ResizeLike);

class GetSumGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    for (auto i = 0; i < def_.input_size(); ++i) {
      SetDense(i, GO(0));
    }
    return vector<OperatorDef>();
  }
};
REGISTER_GRADIENT(Sum, GetSumGradient);

SHOULD_NOT_DO_GRADIENT(ScatterWeightedSum);
SHOULD_NOT_DO_GRADIENT(ScatterAssign);

class GetWeightedSumGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    ArgumentHelper argsHelper(def_);
    const bool grad_on_w = argsHelper.GetSingleArgument<bool>("grad_on_w", 0);

    auto inputs = vector<string>{GO(0)};
    auto outputs = vector<string>();
    for (int i = 0; i < def_.input_size(); i += 2) {
      inputs.push_back(I(i));
      inputs.push_back(I(i + 1));
      outputs.push_back(GI(i));
    }

    if (grad_on_w) {
      for (int i = 0; i < def_.input_size(); i += 2) {
        outputs.push_back(GI(i + 1));
      }
    }

    return SingleGradientDef("WeightedSumGradient", "", inputs, outputs);
  }
};
REGISTER_GRADIENT(WeightedSum, GetWeightedSumGradient);

struct GetFlattenToVecGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ResizeLike", "", vector<string>{GO(0), I(0)}, vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(FlattenToVec, GetFlattenToVecGradient);

struct GetCopyGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "CopyOnDeviceLike",
        "",
        vector<string>{GO(0), I(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Copy, GetCopyGradient);

struct GetGPUToCPUGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    if (g_output_[0].IsDense()) {
      return SingleGradientDef(
          "CopyCPUToGPU", "", vector<string>{GO(0)}, vector<string>{GI(0)});
    } else {
      return vector<OperatorDef>{CreateOperatorDef(
                                     "CopyCPUToGPU",
                                     "",
                                     std::vector<string>{GO_I(0)},
                                     std::vector<string>{GI_I(0)}),
                                 CreateOperatorDef(
                                     "CopyCPUToGPU",
                                     "",
                                     std::vector<string>{GO_V(0)},
                                     std::vector<string>{GI_V(0)})};
    }
  }
};
REGISTER_GRADIENT(CopyGPUToCPU, GetGPUToCPUGradient);

struct GetCPUToGPUGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    if (g_output_[0].IsDense()) {
      return SingleGradientDef(
          "CopyGPUToCPU", "", vector<string>{GO(0)}, vector<string>{GI(0)});
    } else {
      return vector<OperatorDef>{CreateOperatorDef(
                                     "CopyGPUToCPU",
                                     "",
                                     std::vector<string>{GO_I(0)},
                                     std::vector<string>{GI_I(0)}),
                                 CreateOperatorDef(
                                     "CopyGPUToCPU",
                                     "",
                                     std::vector<string>{GO_V(0)},
                                     std::vector<string>{GI_V(0)})};
    }
  }
};
REGISTER_GRADIENT(CopyCPUToGPU, GetCPUToGPUGradient);

SHOULD_NOT_DO_GRADIENT(LengthsToSegmentIds);
SHOULD_NOT_DO_GRADIENT(SegmentIdsToLengths);
SHOULD_NOT_DO_GRADIENT(SegmentIdsToRanges);
SHOULD_NOT_DO_GRADIENT(SegmentIdsToLengthWeights);
SHOULD_NOT_DO_GRADIENT(GatherRangesOp);
SHOULD_NOT_DO_GRADIENT(LengthsGather);
SHOULD_NOT_DO_GRADIENT(AccumulateHistogram);

template <>
bool NanCheckOp<CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  const int D = X.size();
  const float* data = X.data<float>();
  ConstEigenVectorMap<float> input_data(data, D);

  bool all_finite = input_data.allFinite();

  if (!all_finite) {
    std::cerr << "Tensor contained NaN or inf: [" << this->debug_def().input(0)
              << "]" << std::endl;

    for (int j = 0; j < InputSize(); j++) {
      std::cerr << "Tensor name: " << this->debug_def().input(j) << std::endl;
      std::cerr << "Input tensor:" << std::endl;
      tensorPrinter_.Print<float>(Input(j));
      std::cerr << "NaN idxs:" << std::endl;
      const float* x = Input(j).data<float>();
      for (size_t i = 0; i < Input(j).size(); ++i) {
        if (std::isnan(x[i]) || std::isinf(x[i])) {
          std::cerr << i << " ";
        }
      }
      std::cerr << std::endl;
    }
    return false;
  }

  if (&X != Y) {
    Y->CopyFrom(X, &context_);
  }
  return true;
}
REGISTER_CPU_OPERATOR(NanCheck, NanCheckOp<CPUContext>);
REGISTER_GRADIENT(NanCheck, GetNanCheckGradient);

OPERATOR_SCHEMA(NanCheck)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShapeOfInput(0)
    .SetDoc("Identity operator, but checks all values for nan or inf")
    .Input(0, "tensor", "Tensor to check for nan/inf")
    .Output(
        0,
        "output",
        "Tensor to copy input into if no NaNs or inf."
        " Can be in-place");

OPERATOR_SCHEMA(Size)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Return a 1D tensor of type *int64* that contains the number of elements of the input tensor.

Github Link:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Size",
    ["X"],
    ["size"],
)

workspace.FeedBlob("X", (np.random.randint(10, size=(3,3))))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("size:", workspace.FetchBlob("size"))

workspace.ResetWorkspace()

workspace.FeedBlob("X", (np.random.rand(6,4)))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("size:", workspace.FetchBlob("size"))

```

**Result**

```

X:
[[3 7 0]
 [0 1 6]
 [5 0 8]]
size: 9
X:
[[0.92017884 0.32115368 0.68692035 0.64135016]
 [0.8723328  0.77830265 0.80688656 0.25524236]
 [0.37970216 0.76407047 0.85689564 0.30692883]
 [0.69352573 0.42531502 0.16415212 0.59209324]
 [0.52684188 0.37094846 0.60670079 0.6489272 ]
 [0.94715906 0.34800557 0.61898769 0.28947359]]
size: 24

```

</details>

      )DOC")
    .Input(0, "X", "*(type: Tensor)* Input tensor to calculate number of elements.")
    .Output(
        0,
        "size",
        "*(type: Tensor)* 1D tensor of type int64 that contains the number of "
        "elements in the input tensor *X*.");

REGISTER_CPU_OPERATOR(Size, SizeOp<CPUContext>);
NO_GRADIENT(Size);

template <>
template <typename T>
bool RangeOp<CPUContext>::DoRunOnDevice(
    const T& start,
    const T& step,
    Tensor* output) {
  auto* output_data = output->template mutable_data<T>();
  for (int i = 0; i < output->size(); ++i) {
    output_data[i] = i * step + start;
  }
  return true;
}

OPERATOR_SCHEMA(Range)
    .NumInputs(1, 3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Generates an output tensor within the half-open interval $[start, stop)$ (the interval including start but excluding stop).
- The `start` input is optional, and defaults to 0 when not set.
- The `step` input is optional, and defaults to 1 when not set.
- The type of the `output` tensor is determined by the types of inputs used.

Github Links:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.h
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Range",
    ["start", "stop", "step"],
    ["output"]
)

workspace.FeedBlob("start", np.array(4, dtype=np.int32))
workspace.FeedBlob("stop", np.array(17, dtype=np.int32))
workspace.FeedBlob("step", np.array(2, dtype=np.int32))
print("start:", workspace.FetchBlob("start"))
print("stop:", workspace.FetchBlob("stop"))
print("step:", workspace.FetchBlob("step"))
workspace.RunOperatorOnce(op)
print("output:", workspace.FetchBlob("output"))

```

**Result**

```

start: 4
stop: 17
step: 2
output: [ 4  6  8 10 12 14 16]

```

</details>
        )DOC")
    .Input(
        0,
        "start",
        "(*Tensor*): [OPTIONAL] scalar tensor containing the start of the interval (inclusive) (default=0)")
    .Input(1, "stop", "(*Tensor*): scalar tensor containing the end of the interval (exclusive)")
    .Input(2, "step", "(*Tensor*): [OPTIONAL] scalar tensor specifying the spacing between values (default=1)")
    .Output(
        0,
        "output",
        "(*Tensor*): 1D tensor of same type as inputs that contains the sequence");

REGISTER_CPU_OPERATOR(Range, RangeOp<CPUContext>);
NO_GRADIENT(Range);

REGISTER_CPU_OPERATOR(ThrowException, ThrowExceptionOp);
OPERATOR_SCHEMA(ThrowException).NumInputs(0).NumOutputs(0);
SHOULD_NOT_DO_GRADIENT(ThrowException);

REGISTER_CPU_OPERATOR(ThrowChildThreadException, ThrowChildThreadExceptionOp);
OPERATOR_SCHEMA(ThrowChildThreadException).NumInputs(0).NumOutputs(0);
SHOULD_NOT_DO_GRADIENT(ThrowChildThreadException);

REGISTER_CPU_OPERATOR(LogFatal, LogFatalOp);
OPERATOR_SCHEMA(LogFatal).NumInputs(0).NumOutputs(0);
SHOULD_NOT_DO_GRADIENT(LogFatal);

REGISTER_CPU_OPERATOR(Fail, FailOp);
OPERATOR_SCHEMA(Fail).NumInputs(0).NumOutputs(0);
SHOULD_NOT_DO_GRADIENT(Fail);

} // namespace caffe2
