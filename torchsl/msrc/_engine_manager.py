import os

__all__ = ['init_engine',
           'set_engine',
           'get_engine']
ENGINE = None


def init_engine():
    global ENGINE
    if ENGINE is None:
        try:
            import matlab
            import matlab.engine
        except ModuleNotFoundError:
            err_msg = f"Matlab engine not found. Consider installing it following the instructions "
            f"from https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html"
            raise ModuleNotFoundError(err_msg)
        print("[torchsl_matlab] Initializing Matlab engine...")
        ENGINE = matlab.engine.start_matlab()
        ENGINE.addpath(os.path.dirname(__file__))


def set_engine(engine):
    global ENGINE
    if ENGINE is not None:
        ENGINE.exit()
    ENGINE = engine


def get_engine():
    if ENGINE is None:
        init_engine()
    return ENGINE
