import torch

from .extensions import _HAS_OPS
from . import functional, modular, nn, optim
from .functional import *
from .msrc import *

# ops module
from ._ops import ops

del torch
