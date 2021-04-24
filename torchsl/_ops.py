from torch._ops import _OpNamespace

from . import _HAS_OPS

__all__ = ['ops']


class _TorchSLOpNameSpace(_OpNamespace):

    def __init__(self):
        super(_TorchSLOpNameSpace, self).__init__('torchsl')

    def __getattr__(self, op_name):
        if not _HAS_OPS:
            raise ModuleNotFoundError("TorchSL C++ extension not found. To use torchsl::{} please install native C++ "
                                      "compiler with C++14 std (g++, msvc) \nand reinstall torchsl with:  "
                                      "python setup.py install".format(op_name))
        return super(_TorchSLOpNameSpace, self).__getattr__(op_name)


ops = _TorchSLOpNameSpace()
