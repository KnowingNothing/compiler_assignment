"""2*2 pooling in python"""

def pooling_python(Image, return_indices=False):
    """
    2*2 max pooling operator

    Parameters
    ----------
    Image : numpy.ndarray
        4-D with shape [batch_size, image_height, image_width, in_channels]

    Returns
    -------
    Output: numpy.ndarray
        4-D with shape [batch_size,image_height//2,image_width//2,in_channels]
    """
    _fail = "[pooling_python] fails: "
    try:
        import numpy as np
    except ImportError:
        print(_fail, "No Numpy found, exit...")
        return
    
    batch_size, image_height, image_width, in_channels = Image.shape

    out_height = image_height // 2
    out_width = image_width // 2

    Output = np.zeros((batch_size, out_height, out_width, in_channels))
    if return_indices:
        Indices = np.zeros(Output.shape, dtype=np.int32)

    for n in range(batch_size):
        for h in range(out_height):
            for w in range(out_width):
                for c in range(in_channels):
                    Output[n, h, w, c] = np.max(Image[n, h * 2 : h * 2 + 2, w * 2 : w * 2 + 2, c])
                    if return_indices:
                        tmp = np.argmax(Image[n, h * 2 : h * 2 + 2, w * 2 : w * 2 + 2, c])
                        Indices[n, h, w, c] = (h * 2 + tmp // 2) * image_width + (w * 2 + tmp % 2)
    if return_indices:
        return Output, Indices
    else:
        return Output

def rpooling_python(Image, POutput):
    """
    reverse 2*2 max pooling operator

    Parameters
    ----------
    Image : numpy.ndarray
        4-D with shape [batch_size, image_height, image_width, in_channels]
    POutput:numpy.ndarray
        4-D with shape [batch_size, out_height, out_width, in_channels]

    Returns
    -------
    PImage: numpy.ndarray
        4-D with shape (Image.shape)
    """
    _fail = "[rpooling_python] fails: "
    try:
        import numpy as np
    except ImportError:
        print(_fail, "No Numpy found, exit...")
        return
    
    batch_size, image_height, image_width, in_channels = Image.shape

    mimage, indices = pooling_python(Image, return_indices=True)

    PImage = np.zeros((batch_size, image_height, image_width, in_channels))

    for n in range(batch_size):
        for h in range(image_height):
            for w in range(image_width):
                for c in range(in_channels):
                    if indices[n, h, w, c] == 1:
                        PImage[n, h, w, c] = POutput[n, h // 2, w // 2, c]

    return PImage, indices

def pooling_pytorch(Image):
    """
    2*2 max pooling operator in PyTorch

    Parameters
    ----------
    Image : numpy.ndarray
        4-D with shape [batch_size, image_height, image_width, in_channels]

    Returns
    -------
    Output: numpy.ndarray
        4-D with shape [batch_size,image_height//2,image_width//2,in_channels]
    """
    _fail = "[pooling_pytorch] fails: "
    try:
        import torch
    except ImportError:
        print(_fail, "No PyTorch found, exit...")
        return
    
    torch_img = torch.tensor(Image)

    # permute the layout of tensor to fit "NCHW"
    permuted_img = torch_img.permute(0, 3, 1, 2)

    output = torch.nn.functional.max_pool2d(permuted_img, 2)

    return output.numpy()

def rpooling_pytorch(Image, POutput):
    """
    reverse 2*2 max pooling operator in PyTorch

    Parameters
    ----------
    Image : numpy.ndarray
        4-D with shape [batch_size, image_height, image_width, in_channels]
    POutput:numpy.ndarray
        4-D with shape [batch_size, out_height, out_width, in_channels]

    Returns
    -------
    PImage: numpy.ndarray
        4-D with shape (Image.shape)
    """
    _fail = "[rpooling_pytorch] fails: "
    try:
        import torch
    except ImportError:
        print(_fail, "No PyTorch found, exit...")
        return
    
    if any(["int" in str(x.dtype) for x in [Image]]):
        print(_fail, "No grad support for 'int' tensors, exit...")
        return
    
    torch_img = torch.tensor(Image, dtype=torch.float32, requires_grad=True)
    torch_pout = torch.tensor(POutput)

    # permute the layout of tensor to fit "NCHW"
    permuted_img = torch_img.permute(0, 3, 1, 2)
    permuted_pout = torch_pout.permute(0, 3, 1, 2)

    output, indices = torch.nn.functional.max_pool2d(permuted_img, 2, return_indices=True)

    permuted_indices = indices.permute(0, 2, 3, 1)

    if output.shape != permuted_pout.shape:
        print(_fail, "Given gradient is not consistent with the outputs in shape, exit...")
        return
    output.backward(permuted_pout)
    return torch_img.grad.numpy(), permuted_indices.numpy()
