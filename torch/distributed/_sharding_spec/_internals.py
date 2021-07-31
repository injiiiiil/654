from typing import List, Union
from dataclasses import dataclass

import torch
from torch.distributed.utils import _parse_remote_device

Device = Union[torch.device, int, str]

def is_valid_device(device):
    """
    Checks if this is a valid local/remote device.
    """
    # Check for torch.device
    try:
        torch.device(device)
        return True
    except Exception:
        pass

    # Check for remote device.
    try:
        _parse_remote_device(device)
        return True
    except Exception:
        pass

    return False


@dataclass
class ShardMetadata(object):
    """
    Represents a shard of the overall Tensor including its
    offsets, lengths and device placement.

    Args:
        shard_offsets(List[int]): Offsets in the orignal tensor indicating
            the start offsets for this shard. Should have the same rank as
            the original tensor.
        shard_lengths(List[int]): Lengths indicating the length of each
            dimension for this shard. Should have the same rank as the
            original tensor.
        placement(Device):
            Specifies the placement of this shard.

            The placement can be a local device or a remote device specified by one
            of the following remote formats:

                1. "rank:<rank>/<device>" (ex: "rank:0/cuda:0").
                2. "<worker_name>/<device>" (ex: "trainer0/cuda:0").
    """

    __slots__ = ['shard_offsets', 'shard_lengths', 'placement']

    shard_offsets: List[int]
    shard_lengths: List[int]
    placement: Device

    def __post_init__(self):
        if not is_valid_device(self.placement):
            raise ValueError(f'{self.placement} is not a valid device')

        if len(self.shard_offsets) != len(self.shard_lengths):
            raise ValueError(
                f'shard_offsets and shard_lengths should have '
                f'the same number of elements, found {len(self.shard_offsets)} '
                f'and {self.shard_lengths} respectively')

        for i in range(len(self.shard_offsets)):
            if self.shard_offsets[i] < 0:
                raise ValueError('shard_offsets should be >=0')
            if self.shard_lengths[i] <= 0:
                raise ValueError('shard_lengths should be > 0')



def _check_shard_metadata_pair_overlap(shard1: ShardMetadata, shard2: ShardMetadata):
    """
    Checks if two shards overlap.
    """

    # For each dim of each shard, check if one shard resides on the other
    # end of second shard with respect to that dim. As an example for a 2D
    # shard, we would check if one shard is above or on the left of the
    # other shard.
    ndims = len(shard1.shard_offsets)
    for i in range(ndims):
        if shard1.shard_offsets[i] >= shard2.shard_offsets[i] + shard2.shard_lengths[i]:
            return False
        if shard2.shard_offsets[i] >= shard1.shard_offsets[i] + shard1.shard_lengths[i]:
            return False

    return True

def validate_non_overlapping_shards_metadata(shards: List[ShardMetadata]):
    """
    Ensures none of the shards overlap with each other.

    Args:
        shards(List[ShardMetadata]): List of :class:`ShardMetadata` objects representing
            each shard.
    Raises:
        ``ValueError`` if there's overlap in any two shards.
    """
    # TODO: evaluate optimizing this if needed.
    for i in range(len(shards)):
        for j in range(i + 1, len(shards)):
            if _check_shard_metadata_pair_overlap(shards[i], shards[j]):
                raise ValueError(f'Shards {shards[i]} and {shards[j]} overlap')


def check_tensor(shards_metadata, tensor_dims) -> None:
    """
    Checks if the shards_metadata is compatible with the provided tensor dims.

    Args:
        shards_metadata(List[ShardMetadata]): List of :class:`ShardMetadata`
            objects representing each shard of the tensor.
        tensor_dims(Sequence of int): Dimensions of tensor to verify
    Raises:
        ``ValueError`` if not compatible.
    """

    # If the tensor's volume matches the total volume of all shards and
    # all shard boundaries are within tensor dims, we have a compatible
    # sharding spec for this tensor. Note that we have already verified
    # we don't have overlapping shards.
    tensor_rank = len(tensor_dims)
    shards_rank = len(shards_metadata[0].shard_offsets)
    if tensor_rank != shards_rank:
        raise ValueError(f'Rank of tensor is {tensor_rank}, but shards rank is {shards_rank}')

    total_shard_volume = 0
    for shard in shards_metadata:
        shard_volume = 1
        for i, shard_length in enumerate(shard.shard_lengths):
            shard_volume *= shard_length
            if shard.shard_offsets[i] + shard.shard_lengths[i] > tensor_dims[i]:
                raise ValueError(
                    f'Shard offset {shard.shard_offsets[i]} and length '
                    f'{shard.shard_lengths[i]} exceeds tensor dim: {tensor_dims[i]} for shard {shard}')
        total_shard_volume += shard_volume

    tensor_volume = 1
    for size in tensor_dims:
        tensor_volume *= size

    if total_shard_volume != tensor_volume:
        # TODO: Can we improve this error message to point out the gaps?
        raise ValueError(
            f'Total volume of shards: {total_shard_volume} '
            f'does not match tensor volume: {tensor_volume}, in other words '
            f'all the individual shards do not cover the entire tensor')
