from __future__ import absolute_import as _abs

import tvm
import numpy as np
'''
The names of fucntions, parameters, return values and their orders are essential
to testing, so do not modify them.
The meaning of parameters are put in comments, and the order of return values are
specified in comments by 
'
Returns
-------
    Return val_1 : ...
    Return val_2 : ...
'
Here we also provide an example of conv2d to show how to build functions for the operators:
--------------------------------------------------------------------------
i_shape = (batch_size, image_height, image_width, in_channels)
f_shape = (out_channels, in_channels, kernel_height, kernel_width)
c_shape = (batch_size, image_height - kernel_height + 1, image_width - kernel_width + 1, out_channels)

Image = tvm.placeholder(i_shape, name='Image')
Filter = tvm.placeholder(f_shape, name='Filter')

Conv = conv2d(Image, Filter)
s =  tvm.create_schedule(Conv.op)

ctx = tvm.cpu(0)
f = tvm.build(s, [Image, Filter, Conv], 'llvm')

im = tvm.nd.array(np.random.uniform(size=i_shape).astype(Image.dtype), ctx)
fi = tvm.nd.array(np.random.uniform(size=f_shape).astype(Filter.dtype), ctx)
conv = tvm.nd.array(np.zeros(c_shape, dtype = Conv.dtype), ctx)
f(im, fi, conv)
---------------------------------------------------------------------------
You can use tvm.testing.assert_allclose function to test if the outputs are right
'''

def conv2d(Image, Filter):
    """
    Convolution operator in NHWC layout

    Parameters
    ----------
    Image : tvm.tensor.Tensor
        4-D with shape [batch_size, image_height, image_width, in_channels]
    Filter: tvm.tensor.Tensor
        4-D with shape [out_channels, in_channels, kernel_height, kernel_width]

    Returns
    -------
    Output: tvm.tensor.Tensor
        4-D with shape [batch_size, out_height, out_width, out_channels]
    """
    pass

def conv2db(Image, Filter, POutput):
    """
    convolution with NHWC layout backward

    Parameters
    ----------
    Image : tvm.tensor.Tensor
        4-D with shape [batch_size, image_height, image_width, in_channels]
    Filter: tvm.tensor.Tensor
        4-D with shape [out_channels, in_channels, kernel_height, kernel_width]
    POutput:tvm.tensor.Tensor, gradient of Output
        4-D with shape [batch_size, out_height, out_width, out_channels]

    Returns
    -------
    PImage :tvm.tensor.Tensor, gradient of Image
        4-D with shape (Image.shape)
    PFilter:tvm.tensor.Tensor, gradient of Filter
        4-D with shape (Filter.shape)
    """
    pass

def relu(Image):
    """
    ReLU operator

    Parameters
    ----------
    Image : tvm.tensor.Tensor
        4-D with shape [batch_size, image_height, image_width, in_channels]

    Returns
    -------
    Output: tvm.tensor.Tensor
        4-D with shape (Image.shape)
    """
    pass

def relub(Image, POutput):
    """
    ReLU operator backward

    Parameters
    ----------
    Image : tvm.tensor.Tensor
        4-D with shape [batch_size, image_height, image_width, in_channels]
    POutput:tvm.tensor.Tensor, gradient of Output
        4-D with shape [batch_size, image_height, image_width, in_channels]

    Returns
    -------
    PImage: tvm.tensor.Tensor
        4-D with shape (Image.shape)
    """
    pass

def pooling(Image):
    """
    2*2 max pooling adapted to the revised poolingb2

    Parameters
    ----------
    Image : tvm.tensor.Tensor
        4-D with shape [batch_size, image_height, image_width, in_channels]

    Returns
    -------
    Output: tvm.tensor.Tensor
        4-D with shape [batch_size, out_height, out_width, in_channels]
    """
    pass

def poolingb(Image, Index, POutput):
    """
    reverse 2*2 max pooling revised

    Parameters
    ----------
    Image : tvm.tensor.Tensor
        4-D with shape [batch_size, image_height, image_width, in_channels]
    Index : tvm.tensor.Tensor, specify where Output[i,j,k,l] is from, this follows the convention of 
        Numpy and PyTorch. You will need this tensor to compute the gradient.
        ------------------------------------------
        For example, if Image is of shape [1, 4, 4, 1] (batch 1 and channel 1), then the slice
        Image[0, :, :, 0] is

        [[0.7243, 0.3236, 0.0124, 0.4314],
        [0.4104, 0.3997, 0.4534, 0.1791],
        [0.0973, 0.2673, 0.6907, 0.9207],
        [0.9268, 0.6590, 0.0312, 0.2364]]

        and Index is of shape [1, 2, 2, 1] and the slice Index[0, :, :, 0] is 

        [[ 0,  6],
        [12, 11]]

        because 0 = 0 * 4 + 0 (0, 0)
                6 = 1 * 4 + 2 (1, 2)
                12= 3 * 4 + 0 (3, 0)
                11= 2 * 4 + 3 (2, 3)
        --------------------------------------------
        4-D with shape [batch_size, out_height, out_width, in_channels]
    POutput:tvm.tensor.Tensor, gradient of Output
        4-D with shape [batch_size, out_height, out_width, in_channels]

    Returns
    -------
    PImage: tvm.tensor.Tensor, gradient of Image
        4-D with shape (Image.shape)
    """
    pass

def flatten(Image):
    """
    flatten layer

    Parameters
    ----------
    Image : tvm.tensor.Tensor
        4-D with shape [batch_size, image_height, image_width, in_channels]

    Returns
    -------
    Output: tvm.tensor.Tensor
        2-D with shape [batch_size, out_size]
    """
    pass

def flattenb(Image, POutput):
    """
    reverse flatten

    Parameters
    ----------
    Image : tvm.tensor.Tensor
        4-D with shape [batch_size, image_height, image_width, in_channels]
    POutput:tvm.tensor.Tensor, gradient of Output
        4-D with shape [batch_size, out_size]

    Returns
    -------
    PImage: tvm.tensor.Tensor
        4-D with shape (Image.shape)
    """
    pass

def fullyconn(Input, Weight):
    """
    Fully Connected Layer

    Parameters
    ----------
    Input : tvm.tensor.Tensor
        2-D with shape [batch_size, input_size]
    Weight: tvm.tensor.Tensor
        2-D with shape [input_size, out_size]

    Returns
    -------
    Output: tvm.tensor.Tensor
        2-D with shape [batch_size, out_size]
    """
    pass

def fullyconnb(Input, Weight, POutput):
    """
    reverse fully connected

    Parameters
    ----------
    Input : tvm.tensor.Tensor
        2-D with shape [batch_size, input_size]
    Weight: tvm.tensor.Tensor
        2-D with shape [input_size, out_size]
    POutput:tvm.tensor.Tensor, gradient of Output
        2-D with shape [batch_size, out_size]

    Returns
    -------
    PWeight:tvm.tensor.Tensor, gradient of Weight
        2-D with shape (Weight.shape)
    PInput: tvm.tensor.Tensor, gradient of Input
        2-D with shape (Input.shape)
    """
    pass
