"""fully connected in python"""

def fullyconn_python(Input, Weight):
    """
    Fully Connected operator

    Parameters
    ----------
    Input : numpy.ndarray
        2-D with shape [batch_size, input_size]
    Weight: numpy.ndarray
        2-D with shape [input_size, out_size]

    Returns
    -------
    Output: numpy.ndarray
        2-D with shape [batch_size, out_size]
    """
    _fail = "[fullyconn_python] fails: "
    try:
        import numpy as np
    except ImportError:
        print(_fail, "No Numpy found, exit...")
        return
    
    return np.dot(Input, Weight)

def rfullyconn_python(Input, Weight, POutput):
    """
    reverse fully connected

    Parameters
    ----------
    Input : numpy.ndarray
        2-D with shape [batch_size, input_size]
    Weight: numpy.ndarray
        2-D with shape [input_size, out_size]
    POutput:numpy.ndarray
        2-D with shape [batch_size, out_size]

    Returns
    -------
    PWeight:numpy.ndarray
        2-D with shape (Weight.shape)
    PInput: numpy.ndarray
        2-D with shape (Input.shape)
    """
    _fail = "[rfullyconn_python] fails: "
    try:
        import numpy as np
    except ImportError:
        print(_fail, "No Numpy found, exit...")
        return
    
    PWeight = np.dot(Input.T, POutput)
    PInput = np.dot(POutput, Weight.T)

    return PInput, PWeight

def fullyconn_pytorch(Input, Weight):
    """
    Fully Connected operation in PyTorch

    Parameters
    ----------
    Input : numpy.ndarray
        2-D with shape [batch_size, input_size]
    Weight: numpy.ndarray
        2-D with shape [input_size, out_size]

    Returns
    -------
    Output: numpy.ndarray
        2-D with shape [batch_size, out_size]
    """
    _fail = "[fullyconn_pytorch] fails: "
    try:
        import torch
    except ImportError:
        print(_fail, "No PyTorch found, exit...")
        return
    
    torch_input = torch.tensor(Input)
    torch_weight = torch.tensor(Weight)

    # permute the weight
    permuted_w = torch_weight.permute(1, 0)

    output = torch.nn.functional.linear(torch_input, permuted_w)
    return output.numpy()

def rfullyconn_pytorch(Input, Weight, POutput):
    """
    reverse fully connected in PyTorch

    Parameters
    ----------
    Input : numpy.ndarray
        2-D with shape [batch_size, input_size]
    Weight: numpy.ndarray
        2-D with shape [input_size, out_size]
    POutput:numpy.ndarray
        2-D with shape [batch_size, out_size]

    Returns
    -------
    PWeight:numpy.ndarray
        2-D with shape (Weight.shape)
    PInput: numpy.ndarray
        2-D with shape (Input.shape)
    """
    _fail = "[rfullyconn_pytorch] fails: "
    try:
        import torch
    except ImportError:
        print(_fail, "No PyTorch found, exit...")
        return

    if any(["int" in str(x.dtype) for x in [Input, Weight]]):
        print(_fail, "No grad support for 'int' tensors, exit...")
        return
    
    torch_input = torch.tensor(Input, dtype=torch.float32, requires_grad=True)
    torch_weight = torch.tensor(Weight, dtype=torch.float32, requires_grad=True)
    torch_pout = torch.tensor(POutput)

    # permute the weight
    permuted_w = torch_weight.permute(1, 0)

    output = torch.nn.functional.linear(torch_input, permuted_w)

    if output.shape != torch_pout.shape:
        print(_fail, "Given gradient is not consistent with the outputs in shape, exit...")
        return
    output.backward(torch_pout)

    return torch_input.grad.numpy(), torch_weight.grad.numpy()