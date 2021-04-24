import torch

from ._extensions import _HAS_OPS
from . import ops, nn, optim
from .ops import *

# ops module
from ._ops import ops
from . import utils

del torch
