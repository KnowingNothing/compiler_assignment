"""petite_topi testing util funcs
Used to verify the correctness of operators in petite_topi
"""

from __future__ import absolute_import as _abs

from .conv2d_python import conv2d_python, conv2d_pytorch
from .conv2d_python import rconv2d_python, rconv2d_pytorch
from .relu_python import relu_python, relu_pytorch
from .relu_python import rrelu_python, rrelu_pytorch
from .pooling_python import pooling_python, pooling_pytorch
from .pooling_python import rpooling_python, rpooling_pytorch
from .flatten_python import flatten_python, flatten_pytorch
from .flatten_python import rflatten_python, rflatten_pytorch
from .fullyconn_python import fullyconn_python, fullyconn_pytorch
from .fullyconn_python import rfullyconn_python, rfullyconn_pytorch
