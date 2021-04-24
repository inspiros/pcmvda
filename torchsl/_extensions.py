_HAS_OPS = False


def _has_ops():
    return False


def _register_extensions():
    import os
    import importlib
    import torch

    # load the custom_op_library and register the custom ops
    lib_dir = os.path.dirname(__file__)
    if os.name == 'nt':
        # Register the main torchvision library location on the default DLL path
        import ctypes
        import sys

        kernel32 = ctypes.WinDLL('kernel32.dll', use_last_error=True)
        with_load_library_flags = hasattr(kernel32, 'AddDllDirectory')
        prev_error_mode = kernel32.SetErrorMode(0x0001)

        if with_load_library_flags:
            kernel32.AddDllDirectory.restype = ctypes.c_void_p

        if sys.version_info >= (3, 8):
            os.add_dll_directory(lib_dir)
        elif with_load_library_flags:
            res = kernel32.AddDllDirectory(lib_dir)
            if res is None:
                err = ctypes.WinError(ctypes.get_last_error())
                err.strerror += f' Error adding "{lib_dir}" to the DLL directories.'
                raise err

        kernel32.SetErrorMode(prev_error_mode)

    loader_details = (
        importlib.machinery.ExtensionFileLoader,
        importlib.machinery.EXTENSION_SUFFIXES
    )

    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = extfinder.find_spec("_C")
    if ext_specs is None:
        raise ImportError
    torch.ops.load_library(ext_specs.origin)


try:
    _register_extensions()
    _HAS_OPS = True

    def _has_ops():  # noqa: F811
        return True
except (ImportError, OSError):
    pass


def _assert_has_ops():
    if not _has_ops():
        raise RuntimeError(
            "Couldn't load custom C++ ops. Please recompile with:\n"
            "   python setup.py build_ext --inplace"
        )
