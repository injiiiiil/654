from ..common import DeviceOpOverrides, register_device_op_overrides


class XPUDeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name):
        return f"from torch._C import _xpu_getCurrentRawStream as {name}"

    def set_device(self, device_idx):
        return f"torch.xpu.set_device({device_idx})"

    def synchronize(self):
        return "torch.xpu.synchronize()"

    def device_guard(self, device_idx):
        return f"torch.xpu._DeviceGuard({device_idx})"


register_device_op_overrides("xpu", XPUDeviceOpOverrides())
