"""
torch.distributed.launch is a helper module that spawns up multiple distributed
training processes on each of the training nodes.

The utility can be used for single-node distributed training, in which
each distributed process will be operating on a single GPU. This can achieve
well-improved single-node training performance. It can also be used in
multi-node distributed training, by spawning up multiple processes on each node
for well-improved multi-node distributed training performance as well.
This will especially be benefitial for systems with multiple Infiniband
interfaces since all of them can be utilized for aggregated communication
bandwidth.

In both cases, this utilily will launch a given number of processes
per node (nproc_per_node, which defaults to the number of GPUs on the node).
This number needs to be less or euqal to the number of GPUs on the current
system, and each process will be operating on a single GPU from GPU 0 to
GPU nproc_per_node - 1.

How to use this module:

    (1) Single Node multi-proc distributed training:
        python -m torch.distributed.launch YOUR_TRAINING_SCRIPT.py (--arg1
        --arg2 --arg3 and all other arguments of your training script)

    (2) Multi Node multi-proc distributed training: (e.g. two nodes)

    NODE1: (IP: 192.168.1.1, and has a free port: 1234)
        python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
        --num_node=2 --rank_node=0 --master_addr="192.168.1.1"
        --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and
        all other arguments of your training script)
    NODE2:
        python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
        --num_node=2 --rank_node=1 --master_addr="192.168.1.1"
        --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and
        all other arguments of your training script)

    (3) To look up what optional arguments this module offers:
        python -m torch.distributed.launch --help

Important Notices:

(1) This utilty and multi-process distributed (single node or
multi-node) GPU training currently only achieves the best performance using
the NCCL distributed backend. Thus NCCL backend is the recommended backend to
use for GPU training.

(2) In your training program, you are supposed to parse the command-line
argument: --device=DEVICE_TO_RUN, (which will be provided with this module) and
ensure that your code only runs on this device in your training program by:

    torch.cuda.set_device(arg.device)  # before your code runs

    or

    with torch.cuda.device(arg.device):
        # your code to run

(3) In your training program, you are supposed to call the following function
at the beginning to start the distributed backend. You need to make sure that
the init_method uses "env://", which is the only supported init_method by this
module:

    torch.distributed.init_process_group(backend='YOUR BACKEND',
                                         "init_method='env://')

(4) In your training program, you are supposed to convert your model to
DistributedDataParallel module using the following function. Please ensure
that device_ids argument is set to be the only GPU device id that your code
will be operating on. In other words, the device_ids needs to be [args.device]
in order to use this utility.

    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[arg.device])

(5) For multi-node training, current we only support nodes with identical number
of GPUs. In other words, the number of GPUs on each node needs to be the same.

"""


import subprocess
import os
import socket
from argparse import ArgumentParser, REMAINDER

import torch


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="PyTorch distributed training launch "
                                        "helper utilty that will spawn up "
                                        "multiple distributed processes")

    # Optional arguments for the launch helper
    parser.add_argument("--num_node", type=int, default=1,
                        help="The number of nodes to use for distributed "
                             "training")
    parser.add_argument("--rank_node", type=int, default=0,
                        help="The rank of the node for multi-node distributed "
                             "training")
    parser.add_argument("--nproc_per_node", type=int, default=-1,
                        help="The number of processes to launch on each node, "
                             "will default to the number of GPUs on your "
                             "system if not specified")
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communciation during distributed "
                             "training")

    # positional
    parser.add_argument("training_script", type=str,
                        help="The full path to the single GPU training "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the "
                             "training script")

    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()


args = parse_args()
num_gpus = torch.cuda.device_count()

if args.nproc_per_node == -1:
    args.nproc_per_node = num_gpus

if args.nproc_per_node > num_gpus:
    raise RuntimeError("Found: {} GPUs on host: {} with rank: {}, the "
                       "number of processes per node cannot be greater "
                       "than the number of GPUs availble on the host."
                       .format(num_gpus,
                               socket.gethostname(),
                               args.rank_node))

# world size in terms of number of processes
dist_world_size = args.nproc_per_node * args.num_node

# set PyTorch distributed related environmental variables
current_env = os.environ.copy()
current_env["MASTER_ADDR"] = args.master_addr
current_env["MASTER_PORT"] = str(args.master_port)
current_env["WORLD_SIZE"] = str(dist_world_size)

processes = []

for local_rank in range(0, args.nproc_per_node):
    # each process's rank
    dist_rank = args.nproc_per_node * args.rank_node + local_rank
    current_env["RANK"] = str(dist_rank)

    # spawn the processes
    cmd = ["python",
           args.training_script,
           "--device={}".format(local_rank)] + args.training_script_args

    process = subprocess.Popen(cmd, env=current_env)
    processes.append(process)

for process in processes:
    process.wait()
