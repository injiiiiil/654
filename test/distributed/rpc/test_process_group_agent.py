#!/usr/bin/env python3

import sys

import torch
import torch.distributed as dist

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_utils import IS_IN_CI, run_tests
from torch.testing._internal.distributed.rpc.process_group_agent_test_fixture import (
    ProcessGroupRpcAgentTestFixture,
)
from torch.testing._internal.distributed.rpc_utils import (
    GENERIC_TESTS,
    PROCESS_GROUP_TESTS,
    MultiProcess,
    generate_tests,
)


# On CircleCI these tests are already run on CPU jobs, thus to save resources do
# not run them on GPU jobs, since thet wouldn't provide additional test signal.
if not (IS_IN_CI and torch.cuda.is_available()):
    globals().update(
        generate_tests(
            "ProcessGroup",
            ProcessGroupRpcAgentTestFixture,
            GENERIC_TESTS + PROCESS_GROUP_TESTS,
            MultiProcess.SPAWN,
            __name__,
        )
    )


if __name__ == "__main__":
    run_tests()
