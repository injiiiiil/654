import unittest

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.pipeline.sync import Pipe
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import (
    requires_gloo,
    requires_nccl,
    skip_if_lt_x_gpu,
    skip_if_rocm,
)
from torch.testing._internal.dist_utils import INIT_METHOD_TEMPLATE, dist_init
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)


class PipeWithDDPTest(RpcAgentTestFixture):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    @dist_init
    @skip_if_rocm
    def test_basic_nccl_ckpt_never(self):
        self._run_basic_test("nccl", "never")

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    @dist_init
    @skip_if_rocm
    def test_basic_nccl_ckpt_never_find_unused(self):
        self._run_basic_test("nccl", "never", find_unused_parameters=True)

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    @dist_init
    @skip_if_rocm
    @unittest.skip("DDP doesn't work with checkpointing")
    def test_basic_nccl_ckpt_always(self):
        self._run_basic_test("nccl", "always")

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    @dist_init
    @skip_if_rocm
    @unittest.skip("DDP doesn't work with checkpointing")
    def test_basic_nccl_ckpt_except_last(self):
        self._run_basic_test("nccl", "except_last")

    @skip_if_lt_x_gpu(4)
    @requires_gloo()
    @dist_init
    @skip_if_rocm
    def test_basic_gloo_ckpt_never(self):
        self._run_basic_test("gloo", "never")

    @skip_if_lt_x_gpu(4)
    @requires_gloo()
    @dist_init
    @skip_if_rocm
    def test_basic_gloo_ckpt_never_find_unused(self):
        self._run_basic_test("gloo", "never", find_unused_parameters=True)

    @skip_if_lt_x_gpu(4)
    @requires_gloo()
    @dist_init
    @skip_if_rocm
    @unittest.skip("DDP doesn't work with checkpointing")
    def test_basic_gloo_ckpt_always(self):
        self._run_basic_test("gloo", "always")

    @skip_if_lt_x_gpu(4)
    @requires_gloo()
    @dist_init
    @skip_if_rocm
    @unittest.skip("DDP doesn't work with checkpointing")
    def test_basic_gloo_ckpt_except_last(self):
        self._run_basic_test("gloo", "except_last")

    def _run_basic_test(self, backend, checkpoint, find_unused_parameters=False):
        dist.init_process_group(
            backend="nccl",
            init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name),
            world_size=self.world_size,
            rank=self.rank,
        )

        # Use 4 GPUs, two replicas of a pipe across GPU 0 and 1 and another
        # pipe between GPU 2 and 3. Both replicas are replicated via DDP.
        fc1 = nn.Linear(16, 8, bias=False).cuda(2 * self.rank)

        class MyModule(nn.Module):
            def __init__(self, device):
                super(MyModule, self).__init__()
                self.fc2 = nn.Linear(8, 4, bias=False).cuda(device)
                self.fc3 = nn.Linear(4, 2, bias=False).cuda(device)

            def forward(self, inp):
                if find_unused_parameters:
                    return self.fc2(inp)
                else:
                    return self.fc3(self.fc2(inp))

        layer2 = MyModule(2 * self.rank + 1)
        model = nn.Sequential(
            fc1,
            layer2
        )
        model = Pipe(model, chunks=2, checkpoint=checkpoint)
        model = DistributedDataParallel(model, find_unused_parameters=find_unused_parameters)
        out = model(torch.rand(16, 16).cuda(2 * self.rank)).local_value()
        out.sum().backward()

        # Run forward again for find_unused_parameters to trigger any potential errors.
        if find_unused_parameters:
            model(torch.rand(16, 16).cuda(2 * self.rank))

        # Check grads
        output = [torch.empty_like(fc1.weight.grad), torch.empty_like(fc1.weight.grad)]
        dist.all_gather(output, fc1.weight.grad)
        self.assertEqual(output[0], output[1])

        output = [torch.empty_like(layer2.fc2.weight.grad), torch.empty_like(layer2.fc2.weight.grad)]
        dist.all_gather(output, layer2.fc2.weight.grad)
        self.assertEqual(output[0], output[1])

        if not find_unused_parameters:
            output = [torch.empty_like(layer2.fc3.weight.grad), torch.empty_like(layer2.fc3.weight.grad)]
            dist.all_gather(output, layer2.fc3.weight.grad)
            self.assertEqual(output[0], output[1])
