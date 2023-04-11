# -*- coding: utf-8 -*-
# Owner(s): ["module: unknown"]

import itertools
import logging
import re

import torch
from torch import nn
from torch.ao.pruning import BasePruner, WeightNormPruner, NearlyDiagonalPruner, FakeSparsity
from torch.nn.utils.parametrize import is_parametrized

from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_pruning import SimpleLinear, MockSparseLinear, ImplementedPruner

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


class TestBasePruner(TestCase):
    def test_constructor(self):
        # Cannot instantiate the abstract base
        self.assertRaises(TypeError, BasePruner)
        # Can instantiate the model with no configs
        model = SimpleLinear()
        pruner = ImplementedPruner(test=3)
        pruner.prepare(model, config=None)
        assert len(pruner.groups) == 5
        pruner.step()
        # Can instantiate the model with configs
        pruner = ImplementedPruner(test=3)
        pruner.prepare(model, [{'tensor_fqn': 'linear1.weight'}])
        assert len(pruner.groups) == 1
        assert pruner.groups[0]['tensor_fqn'] == 'linear1.weight'
        assert 'test' in pruner.groups[0]
        assert pruner.groups[0]['test'] == 3

    def test_prepare_config(self):
        model = SimpleLinear()
        pruner = ImplementedPruner(test=3)
        # Make sure there are no parametrizations before `prepare`
        assert not hasattr(model.seq[0], 'parametrizations')
        assert not hasattr(model.linear1, 'parametrizations')
        assert not hasattr(model.linear2, 'parametrizations')
        pruner.prepare(model, config=[
            {'tensor_fqn': 'seq.0.weight', 'test': 42},
            # No 'linear1' to make sure it will be skipped in the sparsification
            {'tensor_fqn': 'linear2.weight'}
        ])
        assert len(pruner.groups) == 2
        # Check if default argument is not assigned if explicit
        assert pruner.groups[0]['tensor_fqn'] == 'seq.0.weight'
        assert pruner.groups[0]['test'] == 42
        # Check if FQN and module are pointing to the same location
        assert pruner.groups[1]['tensor_fqn'] == 'linear2.weight'
        assert pruner.groups[1]['module'] == model.linear2
        # Check if parameterizations are attached
        assert hasattr(model.seq[0], 'parametrizations')
        assert not hasattr(model.linear1, 'parametrizations')
        assert hasattr(model.linear2, 'parametrizations')

    def test_step(self):
        model = SimpleLinear()
        pruner = ImplementedPruner(test=3)
        pruner.enable_mask_update = True
        pruner.prepare(model, [{'tensor_fqn': 'linear1.weight'}])
        pruner.step()
        assert torch.all(model.linear1.parametrizations.weight[0].mask[0] == 0)

    def test_state_dict(self):
        step_count = 3
        model0 = SimpleLinear()
        pruner0 = ImplementedPruner(test=3)
        pruner0.prepare(model0, [{'tensor_fqn': 'linear1.weight'}])
        mask = model0.linear1.parametrizations['weight'][0].mask
        mask.data = torch.arange(mask.shape[0] * mask.shape[1]).reshape(mask.shape)
        for step in range(step_count):
            pruner0.step()
        state_dict = pruner0.state_dict()

        # Check the expected keys in the state_dict
        assert 'state' in state_dict
        assert 'step_count' in state_dict['state']['linear1.weight']
        assert state_dict['state']['linear1.weight']['step_count'] == 3
        assert 'groups' in state_dict
        assert 'test' in state_dict['groups'][0]
        assert 'tensor_fqn' in state_dict['groups'][0]
        assert state_dict['groups'][0]['tensor_fqn'] == 'linear1.weight'

        # Check loading static_dict creates an equivalent model
        model1 = SimpleLinear()
        pruner1 = ImplementedPruner()
        pruner1.prepare(model1, None)

        assert pruner0.state != pruner1.state

        # Make sure the masks are different in the beginning
        for mg in pruner0.groups:
            if mg['tensor_fqn'] == 'linear1.weight':
                mask0 = mg['module'].parametrizations.weight[0].mask
        for mg in pruner1.groups:
            if mg['tensor_fqn'] == 'linear1.weight':
                mask1 = mg['module'].parametrizations.weight[0].mask
        self.assertNotEqual(mask0, mask1)

        pruner1.load_state_dict(state_dict)

        # Make sure the states are loaded, and are correct
        assert pruner0.state == pruner1.state

        # Make sure the masks (and all dicts) are the same after loading
        assert len(pruner0.groups) == len(pruner1.groups)
        for idx in range(len(pruner0.groups)):
            mg0 = pruner0.groups[idx]
            mg1 = pruner1.groups[idx]
            for key in mg0.keys():
                assert key in mg1
                if key == 'module':
                    # We cannot compare modules as they are different
                    param0 = mg0[key].parametrizations.weight[0]
                    param1 = mg1[key].parametrizations.weight[0]
                    assert hasattr(param0, 'mask')
                    assert hasattr(param1, 'mask')
                    self.assertEqual(param0.__dict__, param1.__dict__)
                else:
                    assert mg0[key] == mg1[key]

    def test_convert(self):
        model = SimpleLinear()
        pruner = ImplementedPruner(test=3)
        pruner.prepare(model, [{'tensor_fqn': 'linear1.weight'}])
        new_model = pruner.convert(model, mapping={nn.Linear: MockSparseLinear}, inplace=False)

        assert isinstance(new_model.linear1, MockSparseLinear)
        assert isinstance(new_model.seq[0], nn.Linear)
        assert isinstance(new_model.linear2, nn.Linear)



    def test_mask_squash(self):
        model = SimpleLinear()
        pruner = ImplementedPruner(test=3)
        pruner.prepare(model, [{'tensor_fqn': 'linear1.weight'}])
        assert hasattr(model.linear1.parametrizations.weight[0], 'mask')
        assert is_parametrized(model.linear1, 'weight')
        assert not is_parametrized(model.seq[0], 'weight')

        pruner.squash_mask()
        assert not is_parametrized(model.seq[0], 'weight')
        assert not is_parametrized(model.linear1, 'weight')

    def test_mask_squash_with_params1(self):
        model = SimpleLinear()
        pruner = ImplementedPruner(foo=3, bar=2, baz=1)
        pruner.prepare(model, [{'tensor_fqn': 'linear1.weight'}, {'tensor_fqn': 'seq.0.weight'}])
        pruner.squash_mask(
            params_to_keep_per_layer={
                'linear1': ('foo', 'bar'),
                'seq.0': ('baz',)
            })
        assert not is_parametrized(model.seq[0], 'weight')
        assert not is_parametrized(model.linear1, 'weight')
        assert hasattr(model.seq[0], 'sparse_params')
        assert hasattr(model.linear1, 'sparse_params')
        assert model.seq[0].sparse_params.get('foo', None) is None
        assert model.seq[0].sparse_params.get('bar', None) is None
        assert model.seq[0].sparse_params.get('baz', None) == 1
        assert model.linear1.sparse_params.get('foo', None) == 3
        assert model.linear1.sparse_params.get('bar', None) == 2
        assert model.linear1.sparse_params.get('baz', None) is None

    def test_mask_squash_with_params2(self):
        model = SimpleLinear()
        pruner = ImplementedPruner(foo=3, bar=2, baz=1)
        pruner.prepare(model, [{'tensor_fqn': 'linear1.weight'}, {'tensor_fqn': 'seq.0.weight'}])
        pruner.squash_mask(params_to_keep=('foo', 'bar'))
        assert not is_parametrized(model.seq[0], 'weight')
        assert not is_parametrized(model.linear1, 'weight')
        assert hasattr(model.seq[0], 'sparse_params')
        assert hasattr(model.linear1, 'sparse_params')
        assert model.seq[0].sparse_params.get('foo', None) == 3
        assert model.seq[0].sparse_params.get('bar', None) == 2
        assert model.seq[0].sparse_params.get('baz', None) is None
        assert model.linear1.sparse_params.get('foo', None) == 3
        assert model.linear1.sparse_params.get('bar', None) == 2
        assert model.linear1.sparse_params.get('baz', None) is None

    def test_mask_squash_with_params3(self):
        model = SimpleLinear()
        pruner = ImplementedPruner(foo=3, bar=2, baz=1)
        pruner.prepare(model, [{'tensor_fqn': 'linear1.weight'}, {'tensor_fqn': 'seq.0.weight'}])
        pruner.squash_mask(
            params_to_keep=('foo', 'bar'),
            params_to_keep_per_layer={'seq.0': ('baz',)})
        assert not is_parametrized(model.seq[0], 'weight')
        assert not is_parametrized(model.linear1, 'weight')
        assert hasattr(model.seq[0], 'sparse_params')
        assert hasattr(model.linear1, 'sparse_params')
        assert model.seq[0].sparse_params.get('foo', None) == 3
        assert model.seq[0].sparse_params.get('bar', None) == 2
        assert model.seq[0].sparse_params.get('baz', None) == 1
        assert model.linear1.sparse_params.get('foo', None) == 3
        assert model.linear1.sparse_params.get('bar', None) == 2
        assert model.linear1.sparse_params.get('baz', None) is None


class TestWeightNormPruner(TestCase):
    def test_constructor(self):
        model = SimpleLinear()
        pruner = WeightNormPruner()
        pruner.prepare(model, config=None)
        for g in pruner.groups:
            assert isinstance(g['module'], nn.Linear)
            # The groups are unordered
            assert g['module_fqn'] in ('seq.0', 'seq.1', 'seq.2', 'linear1', 'linear2')

    def test_step(self):
        model = SimpleLinear()
        pruner = WeightNormPruner(sparsity_level=0.5)
        pruner.prepare(model, config=[{'tensor_fqn': 'linear1.weight'}])
        for g in pruner.groups:
            # Before step
            module = g['module']
            assert (1.0 - module.parametrizations['weight'][0].mask.mean()) == 0  # checking sparsity level is 0
        pruner.enable_mask_update = True
        pruner.step()
        self.assertAlmostEqual(model.linear1.parametrizations['weight'][0].mask.mean().item(), 0.5, places=2)
        for g in pruner.groups:
            # After step
            module = g['module']
            assert (1.0 - module.parametrizations['weight'][0].mask.mean()) > 0  # checking sparsity level has increased
        # Test if the mask collapses to all zeros if the weights are randomized
        iters_before_collapse = 1000
        for _ in range(iters_before_collapse):
            model.linear1.weight.data = torch.randn(model.linear1.weight.shape)
            pruner.step()
        for g in pruner.groups:
            # After step
            module = g['module']
            assert (1.0 - module.parametrizations['weight'][0].mask.mean()) > 0  # checking sparsity level did not collapse

    def test_step_2_of_4(self):
        model = SimpleLinear()
        pruner = WeightNormPruner(sparsity_level=1.0,
                                  sparse_block_shape=(1, 4),
                                  zeros_per_block=2)
        pruner.prepare(model, config=[{'tensor_fqn': 'linear1.weight'}])
        pruner.step()
        # make sure the sparsity level is approximately 50%
        mask = model.linear1.parametrizations['weight'][0].mask.to(torch.float)  # mean works on float only
        self.assertAlmostEqual(mask.mean().item(), 0.5, places=2)
        # Make sure each block has exactly 50% zeros
        module = pruner.groups[0]['module']
        mask = module.parametrizations['weight'][0].mask
        for row in mask:
            for idx in range(0, len(row), 4):
                block = row[idx:idx + 4]
                block, _ = block.sort()
                assert (block[:2] == 0).all()
                assert (block[2:] != 0).all()

    def test_prepare(self):
        model = SimpleLinear()
        pruner = WeightNormPruner()
        pruner.prepare(model, config=None)
        for g in pruner.groups:
            module = g['module']
            # Check mask exists
            assert hasattr(module.parametrizations['weight'][0], 'mask')
            # Check parametrization exists and is correct
            assert is_parametrized(module, 'weight')
            assert type(module.parametrizations.weight[0]) == FakeSparsity

    def test_mask_squash(self):
        model = SimpleLinear()
        pruner = WeightNormPruner()
        pruner.prepare(model, config=None)
        pruner.squash_mask()
        for g in pruner.groups:
            module = g['module']
            assert not is_parametrized(module, 'weight')
            assert not hasattr(module, 'mask')

    def test_sparsity_levels(self):
        sparsity_levels = [-1.0, 0.0, 0.5, 1.0, 2.0]
        sparse_block_shapes = [(1, 1), (1, 4), (2, 2), (4, 1)]
        zeros_per_blocks = [0, 1, 2, 3, 4]

        testcases = itertools.tee(itertools.product(sparsity_levels,
                                                    sparse_block_shapes,
                                                    zeros_per_blocks))
        # Create a config and model with all the testcases
        model = nn.Sequential()
        pruner = WeightNormPruner()

        sparsity_per_layer_config = []
        p = re.compile(r'[-\.\s]')
        for sl, sbs, zpb in testcases[0]:
            # Make sure the number of zeros is not > values in a block
            if zpb > sbs[0] * sbs[1]:
                continue
            layer_name = f'{sl}_{sbs}_{zpb}'
            layer_name = p.sub('_', layer_name)

            layer = nn.Linear(12, 12, bias=False)
            layer.weight = nn.Parameter(torch.ones(12, 12))
            model.add_module(layer_name, layer)
            config = {
                'tensor_fqn': layer_name + ".weight",
                'sparsity_level': sl,
                'sparse_block_shape': sbs,
                'zeros_per_block': zpb
            }
            sparsity_per_layer_config.append(config)

        pruner.prepare(model, sparsity_per_layer_config)
        pruner.step()
        pruner.squash_mask()
        model.eval()

        for sl, sbs, zpb in testcases[1]:
            if zpb > sbs[0] * sbs[1]:
                continue
            layer_name = f'{sl}_{sbs}_{zpb}'
            layer_name = p.sub('_', layer_name)
            layer = getattr(model, layer_name)

            # Level of sparsity is achieved
            sparse_mask = (layer.weight == 0).float()
            if zpb == 0:
                assert sparse_mask.mean() == 0
            else:
                # Ratio of individual zeros in the tensor
                true_sl = min(max(sl, 0.0), 1.0)
                true_sl = true_sl * zpb / sbs[0] / sbs[1]
                assert sparse_mask.mean() == true_sl


class TestNearlyDiagonalPruner(TestCase):
    def test_constructor(self):
        model = SimpleLinear()
        pruner = NearlyDiagonalPruner(nearliness=1)
        pruner.prepare(model, config=None)
        for g in pruner.groups:
            assert isinstance(g['module'], nn.Linear)
            # The groups are unordered
            assert g['module_fqn'] in ('seq.0', 'seq.1', 'seq.2', 'linear1', 'linear2')

    def test_step(self):
        model = SimpleLinear()
        pruner = NearlyDiagonalPruner(nearliness=1)
        pruner.prepare(model, config=[{'tensor_fqn': 'linear1.weight'}])

        for g in pruner.groups:
            # Before step
            module = g['module']
            assert (1.0 - module.parametrizations['weight'][0].mask.mean()) == 0  # checking sparsity level is 0

        pruner.enable_mask_update = True
        pruner.step()
        mask = module.parametrizations['weight'][0].mask
        height, width = mask.shape
        assert torch.all(mask == torch.eye(height, width))

        for g in pruner.groups:
            # After step
            module = g['module']
            assert (1.0 - module.parametrizations['weight'][0].mask.mean()) > 0  # checking sparsity level has increased

        # Test if the mask collapses to all zeros if the weights are randomized
        iters_before_collapse = 1000
        for _ in range(iters_before_collapse):
            model.linear1.weight.data = torch.randn(model.linear1.weight.shape)
            pruner.step()
        for g in pruner.groups:
            # After step
            module = g['module']
            assert (1.0 - module.parametrizations['weight'][0].mask.mean()) > 0  # checking sparsity level did not collapse

    def test_prepare(self):
        model = SimpleLinear()
        pruner = NearlyDiagonalPruner(nearliness=1)
        pruner.prepare(model, config=None)
        for g in pruner.groups:
            module = g['module']
            # Check mask exists
            assert hasattr(module.parametrizations['weight'][0], 'mask')
            # Check parametrization exists and is correct
            assert is_parametrized(module, 'weight')
            assert type(module.parametrizations.weight[0]) == FakeSparsity

    def test_mask_squash(self):
        model = SimpleLinear()
        pruner = NearlyDiagonalPruner(nearliness=1)
        pruner.prepare(model, config=None)
        pruner.step()
        pruner.squash_mask()
        for g in pruner.groups:
            module = g['module']
            assert not is_parametrized(module, 'weight')
            assert not hasattr(module, 'mask')
            weights = module.weight
            height, width = weights.shape
            assert torch.all(weights == torch.eye(height, width) * weights)  # only diagonal to be present


    def test_sparsity_levels(self):
        nearliness_levels = list(range(-1, 100))
        model = nn.Sequential()

        p = re.compile(r'[-\.\s]')
        for nearliness in nearliness_levels:
            pruner = NearlyDiagonalPruner(nearliness=1)
            layer_name = f'{nearliness}'
            layer_name = p.sub('_', layer_name)

            layer = nn.Linear(32, 32, bias=False)
            layer.weight = nn.Parameter(torch.ones(32, 32))
            width, height = layer.weight.shape
            model.add_module(layer_name, layer)
            config = {
                'tensor_fqn': layer_name + ".weight",
                'nearliness': nearliness
            }

            pruner.prepare(model, [config])
            # should raise a ValueError when nearliness arg is illegal
            if (nearliness > 0 and nearliness % 2 == 0) or (nearliness // 2 >= min(width, height)):
                with self.assertRaises(ValueError):
                    pruner.step()
            else:
                pruner.step()
                pruner.squash_mask()
                model.eval()

                layer = getattr(model, layer_name)
                # verify that mask created corresponds to the nearliness
                self._verify_nearliness(layer.weight, nearliness)

    # helper function to verify nearliness of a mask
    def _verify_nearliness(self, mask: torch.Tensor, nearliness: int):
        if nearliness <= 0:
            assert torch.all(mask == torch.zeros(mask.shape[0], mask.shape[1]))
        else:
            height, width = mask.shape
            dist_to_diagonal = nearliness // 2
            for row in range(0, height):
                for col in range(0, width):
                    if abs(row - col) <= dist_to_diagonal:
                        assert mask[row, col] == 1
                    else:
                        assert mask[row, col] == 0
