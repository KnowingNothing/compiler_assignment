# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals
"""convolution & reverse-Convolution in python"""

def conv2d_python(Image, Filter):
    """
    Convolution operator in NHWC layout

    Parameters
    ----------
    Image : numpy.ndarray
        4-D with shape [batch_size, image_height, image_width, in_channels]
    Filter: numpy ndarray
        4-D with shape [out_channels, in_channels, kernel_height, kernel_width]

    Returns
    -------
    Output: numpy.ndarray
        4-D with shape [batch_size, out_height, out_width, out_channels]
    """
    _fail = "[conv2d_python] fails: "
    try:
        import numpy as np
    except ImportError:
        print(_fail, "No Numpy found, exit...")
        return
    try:
        from scipy import signal
    except ImportError:
        print(_fail, "No Scipy found, exit...")
        return

    batch_size, image_height, image_width, in_channels = Image.shape
    out_channels, _, kernel_height, kernel_width = Filter.shape
    # compute the output shape
    out_height = image_height - kernel_height + 1
    out_width = image_width - kernel_width + 1
    # change the layout from NHWC to NCHW
    imaget = Image.transpose((0, 3, 1, 2))
    output = np.zeros((batch_size, out_channels, out_height, out_width))
    # computation
    for n in range(batch_size):
        for f in range(out_channels):
            for c in range(in_channels):
                impad = imaget[n, c]
                out = signal.convolve2d(
                        impad, np.rot90(np.rot90(Filter[f, c])), mode='valid')
                output[n, f] += out[::1, ::1]

    # change the layout from NCHW to NHWC
    return output.transpose((0, 2, 3, 1))


def rconv2d_python(Image, Filter, POutput):
    """
    reverse Convolution in NHWC layout implemented in numpy

    Parameters
    ----------
    Image : numpy.ndarray
        4-D with shape [batch_size, image_height, image_width, in_channels]
    Filter: numpy.ndarray
        4-D with shape [out_channels, in_channels, kernel_height, kernel_width]
    POutput:numpy.ndarray
        4-D with shape [batch_size, out_height, out_width, out_channels]

    Returns
    -------
    PFilter:numpy.ndarray
        4-D with shape (Filter.shape)
    PImage :numpy.ndarray
        4-D with shape (Image.shape)
    """
    _fail = "[rconv2d_python] fails: "
    try:
        import numpy as np
    except ImportError:
        print(_fail, "No Numpy found, exit...")
        return
    
    batch_size, image_height, image_width, in_channels = Image.shape
    out_channels, _, kernel_height, kernel_width = Filter.shape
    out_height = image_height - kernel_height + 1
    out_width = image_width - kernel_width + 1
    # pad the output
    pad_height = kernel_height - 1
    pad_width = kernel_width - 1
    ZPOutput = np.pad(POutput, ((0,0),(pad_height,pad_height),(pad_width,pad_width),(0,0)), 'constant')
    # change the layout to fit into conv2d_python 
    imaget = Image.transpose((3, 1, 2, 0))
    poutputt = POutput.transpose((3, 0, 1, 2))
    filtert = np.rot90(np.rot90(Filter, 1, (2,3)), 1, (2,3)).transpose((1, 0, 2, 3))

    # computation
    PFilter = conv2d_python(imaget, poutputt).transpose((3, 0, 1, 2))
    PImage = conv2d_python(ZPOutput, filtert)

    return PImage, PFilter

def conv2d_pytorch(Image, Filter):
    """
    Convolution operator in NHWC layout in PyTorch

    Parameters
    ----------
    Image : numpy.ndarray
        4-D with shape [batch_size, image_height, image_width, in_channels]
    Filter: numpy ndarray
        4-D with shape [out_channels, in_channels, kernel_height, kernel_width]

    Returns
    -------
    Output: numpy.ndarray
        4-D with shape [batch_size, out_height, out_width, out_channels]
    """
    _fail = "[conv2d_pytorch] fails: "
    try:
        import torch
    except ImportError:
        print(_fail, "PyTorch not found, exit...")
        return
    
    torch_img = torch.tensor(Image)
    torch_filt = torch.tensor(Filter)

    # change the layout, currently only found "NCHW" format in PyTorch
    permuted_img = torch_img.permute(0, 3, 1, 2)

    output = torch.nn.functional.conv2d(permuted_img, torch_filt)

    return output.permute(0, 2, 3, 1).numpy()

def rconv2d_pytorch(Image, Filter, POutput):
    """
    reverse Convolution in NHWC layout in PyTorch

    Parameters
    ----------
    Image : numpy.ndarray
        4-D with shape [batch_size, image_height, image_width, in_channels]
    Filter: numpy.ndarray
        4-D with shape [out_channels, in_channels, kernel_height, kernel_width]
    POutput:numpy.ndarray
        4-D with shape [batch_size, out_height, out_width, out_channels]

    Returns
    -------
    PFilter:numpy.ndarray
        4-D with shape (Filter.shape)
    PImage :numpy.ndarray
        4-D with shape (Image.shape)
    """
    _fail = "[rconv2d_pytorch] fails: "
    try:
        import torch
    except ImportError:
        print(_fail, "PyTorch not found, exit...")
        return
    if any(["int" in str(x.dtype) for x in [Image, Filter]]):
        print(_fail, "No grad support for 'int' tensors, exit...")
        return
    
    torch_img = torch.tensor(Image, dtype=torch.float32, requires_grad=True)
    torch_filt = torch.tensor(Filter, dtype=torch.float32, requires_grad=True)
    torch_pout = torch.tensor(POutput)

    # change the layout, currently only found "NCHW" format in PyTorch
    permuted_img = torch_img.permute(0, 3, 1, 2)
    permuted_pout = torch_pout.permute(0, 3, 1, 2)

    output = torch.nn.functional.conv2d(permuted_img, torch_filt)
    
    if output.shape != permuted_pout.shape:
        print(_fail, "Given gradient is not consistent with the outputs in shape, exit...")
        return
    output.backward(permuted_pout)
    return torch_img.grad.numpy(), torch_filt.grad.numpy()

