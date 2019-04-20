"""flatten in python"""

def flatten_python(Image):
    """
    flatten layer

    Parameters
    ----------
    Image : numpy.ndarray
        4-D with shape [batch_size, image_height, image_width, in_channels]

    Returns
    -------
    Output: numpy.ndarray
        2-D with shape [batch_size, out_size]
    """
    batch_size, image_height, image_width, in_channels = Image.shape
    out_size = image_height * image_width * in_channels

    Output = Image.reshape(batch_size, out_size)

    return Output

def rflatten_python(Image, POutput):
    """
    reverse flatten

    Parameters
    ----------
    Image : numpy.ndarray
        4-D with shape [batch_size, image_height, image_width, in_channels]
    POutput:numpy.ndarray
        4-D with shape [batch_size, out_size]

    Returns
    -------
    PImage: numpy.ndarray
        4-D with shape (Image.shape)
    """
    PImage = POutput.reshape(Image.shape)

    return PImage

def flatten_pytorch(Image):
    """
    flatten layer in PyTroch

    Parameters
    ----------
    Image : numpy.ndarray
        4-D with shape [batch_size, image_height, image_width, in_channels]

    Returns
    -------
    Output: numpy.ndarray
        2-D with shape [batch_size, out_size]
    """
    _fail = "[flatten_pytorch] fails: "
    try:
        import torch
    except ImportError:
        print(_fail, "PyTorch not found, exit...")
        return

    return torch.flatten(torch.tensor(Image), start_dim=1).numpy()

def rflatten_pytorch(Image, POutput):
    """
    reverse flatten in Pytorch

    Parameters
    ----------
    Image : numpy.ndarray
        4-D with shape [batch_size, image_height, image_width, in_channels]
    POutput:numpy.ndarray
        4-D with shape [batch_size, out_size]

    Returns
    -------
    PImage: numpy.ndarray
        4-D with shape (Image.shape)
    """
    _fail = "[rflatten_pytorch] fails: "
    try:
        import torch
    except ImportError:
        print(_fail, "PyTorch not found, exit...")
        return
    if any(["int" in str(x.dtype) for x in [Image]]):
        print(_fail, "No grad support for 'int' tensors, exit...")
        return
    
    torch_img = torch.tensor(Image, dtype=torch.float32, requires_grad=True)
    torch_pout = torch.tensor(POutput)

    output = torch.flatten(torch_img, start_dim=1)

    if output.shape != torch_pout.shape:
        print(_fail, "Given gradient is not consistent with the outputs in shape, exit...")
        return
    
    output.backward(torch_pout)
    return torch_img.grad.numpy()
