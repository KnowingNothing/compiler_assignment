"""ReLU in python"""

def relu_python(Image):
    """
    ReLU operator

    Parameters
    ----------
    Image : numpy.ndarray
        4-D with shape [batch_size, image_height, image_width, in_channels]

    Returns
    -------
    Output: numpy.ndarray
        4-D with shape (Image.shape)
    """
    _fail = "[relu_python] fails: "
    try:
        import numpy as np
    except ImportError:
        print(_fail, "No Numpy found, exit...")
        return
    
    Output = np.maximum(Image, 0.0)
    return Output

def rrelu_python(Image, POutput):
    """
    reverse ReLU operator

    Parameters
    ----------
    Image : numpy.ndarray
        4-D with shape [batch_size, image_height, image_width, in_channels]
    POutput:numpy.ndarray
        4-D with shape [batch_size, image_height, image_width, in_channels]

    Returns
    -------
    PImage: numpy.ndarray
        4-D with shape (Image.shape)
    """
    _fail = "[rrelu_python] fails: "
    try:
        import numpy as np
    except ImportError:
        print(_fail, "No Numpy found, exit...")
        return

    PImage = np.zeros(Image.shape)
    batch_size, image_height, image_width, in_channels = Image.shape
    for n in range(batch_size):
        for h in range(image_height):
            for w in range(image_width):
                for c in range(in_channels):
                    if Image[n, h, w, c] > 0.0:
                        PImage[n, h, w, c] = POutput[n, h, w, c]
    return PImage

def relu_pytorch(Image):
    """
    ReLU operator in PyTorch

    Parameters
    ----------
    Image : numpy.ndarray
        4-D with shape [batch_size, image_height, image_width, in_channels]

    Returns
    -------
    Output: numpy.ndarray
        4-D with shape (Image.shape)
    """
    _fail = "[relu_pytorch] fails: "
    try:
        import torch
    except ImportError:
        print(_fail, "No PyTorch found, exit...")
        return
    
    return torch.relu(torch.tensor(Image)).numpy()

def rrelu_pytorch(Image, POutput):
    """
    reverse ReLU operator in PyTorch

    Parameters
    ----------
    Image : numpy.ndarray
        4-D with shape [batch_size, image_height, image_width, in_channels]
    POutput:numpy.ndarray
        4-D with shape [batch_size, image_height, image_width, in_channels]

    Returns
    -------
    PImage: numpy.ndarray
        4-D with shape (Image.shape)
    """
    _fail = "[rrelu_pytorch] fails: "
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

    output = torch.relu(torch_img)

    if output.shape != torch_pout.shape:
        print(_fail, "Given gradient is not consistent with the outputs in shape, exit...")
        return
    output.backward(torch_pout)
    return torch_img.grad.numpy()
