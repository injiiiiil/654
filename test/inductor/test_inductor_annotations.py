# Owner(s): ["module: inductor"]
import torch
import torch._inductor.config as inductor_config
from unittest import mock
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.triton_utils import requires_cuda
from torch._inductor.scheduler import Scheduler


class InductorAnnotationTestCase(TestCase):
    def get_code(self):
        def f(a, b):
            return a + b, a * b

        a = torch.randn(5, device="cuda")
        b = torch.randn(5, device="cuda")
        f_comp = torch.compile(f)

        _, code = run_and_get_code(f_comp, a, b)
        return code[0]


    @inductor_config.patch(annotate_training=True)
    @requires_cuda
    def test_training_annotation(self):
        code = self.get_code()

        self.assertEqual(code.count("training_annotation = nvtx.device_range_start('inference')"), 1)
        self.assertEqual(code.count("nvtx.device_range_end(training_annotation)"), 1)
        self.assertTrue("buffer_annotation" not in code)


    @inductor_config.patch(annotate_buffers=True)
    @requires_cuda
    def test_buffer_annotation(self):
        code = self.get_code()

        self.assertEqual(code.count("buffer_annotation = nvtx.device_range_start"), 1)
        self.assertEqual(code.count("nvtx.device_range_end(buffer_annotation)"), 1)
        self.assertTrue("training_annotation" not in code)


    @inductor_config.patch(annotate_buffers=True)
    @requires_cuda
    def test_buffer_annotation_no_fusion(self):
        def no_fusion(self, nodes):
            return nodes

        with mock.patch.object(Scheduler, "fuse_nodes", no_fusion):
            code = self.get_code()

            self.assertEqual(code.count("buffer_annotation = nvtx.device_range_start"), 2)
            self.assertEqual(code.count("nvtx.device_range_end(buffer_annotation)"), 2)
            self.assertTrue("training_annotation" not in code)


    @inductor_config.patch(annotate_buffers=True, annotate_training=True)
    @requires_cuda
    def test_buffer_and_training_annotation(self):
        code = self.get_code()

        self.assertEqual(code.count("training_annotation = nvtx.device_range_start('inference')"), 1)
        self.assertEqual(code.count("nvtx.device_range_end(training_annotation)"), 1)
        self.assertEqual(code.count("buffer_annotation = nvtx.device_range_start"), 1)
        self.assertEqual(code.count("nvtx.device_range_end(buffer_annotation)"), 1)
