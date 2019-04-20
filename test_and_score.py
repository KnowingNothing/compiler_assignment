import os
import sys
import time
import tvm
import numpy as np
import testing
import test_frame
from imp import reload

# verify the operators

def verify_conv2d(func, target, batch_size, image_height, image_width, in_channels, kernel_height, kernel_width, out_channels):
    i_shape = (batch_size, image_height, image_width, in_channels)
    f_shape = (out_channels, in_channels, kernel_height, kernel_width)
    c_shape = (batch_size, image_height - kernel_height + 1, image_width - kernel_width + 1, out_channels)
    
    Image = tvm.placeholder(i_shape, name='Image')
    Filter = tvm.placeholder(f_shape, name='Filter')

    Conv = func(Image, Filter)
    s =  tvm.create_schedule(Conv.op)

    ctx = tvm.cpu(0)
    f = tvm.build(s, [Image, Filter, Conv], 'llvm')

    im = tvm.nd.array(np.random.uniform(size=i_shape).astype(Image.dtype), ctx)
    fi = tvm.nd.array(np.random.uniform(size=f_shape).astype(Filter.dtype), ctx)
    conv = tvm.nd.array(np.zeros(c_shape, dtype = Conv.dtype), ctx)
    f(im, fi, conv)

    conv_np = target(im.asnumpy(), fi.asnumpy())

    passed = 1
    try:
        tvm.testing.assert_allclose(conv.asnumpy(), conv_np, rtol=1e-5)
    except AssertionError as e:
        passed = 0
    return passed

def test_conv2d(func, target):
    score = 0
    score += verify_conv2d(func, target, 1, 32, 32, 3, 3, 3, 16)
    score += verify_conv2d(func, target, 1, 32, 32, 3, 5, 5, 16)
    score += verify_conv2d(func, target, 1, 32, 32, 3, 1, 1, 16)
    score += verify_conv2d(func, target, 1, 7, 7, 8, 1, 1, 8)
    score += verify_conv2d(func, target, 1, 7, 7, 8, 3, 3, 16)
    score += verify_conv2d(func, target, 4, 32, 32, 3, 3, 3, 16)
    score += verify_conv2d(func, target, 4, 32, 32, 3, 5, 5, 16)
    score += verify_conv2d(func, target, 4, 32, 32, 3, 1, 1, 16)
    score += verify_conv2d(func, target, 4, 7, 7, 8, 1, 1, 8)
    score += verify_conv2d(func, target, 4, 7, 7, 8, 3, 3, 16)
    return score

def verify_rconv2d(func, target, batch_size, image_height, image_width, in_channels, kernel_height, kernel_width, out_channels):
    i_shape = (batch_size, image_height, image_width, in_channels)
    f_shape = (out_channels, in_channels, kernel_height, kernel_width)
    out_height = image_height - kernel_height + 1
    out_width = image_width - kernel_width + 1
    p_shape = (batch_size, out_height, out_width, out_channels)

    Image = tvm.placeholder(i_shape, name='Image')
    Filter = tvm.placeholder(f_shape, name='Filter')
    POutput = tvm.placeholder(p_shape, name='POutput')

    PImage, PFilter = func(Image, Filter, POutput)
    s1 = tvm.create_schedule(PImage.op)
    s2 = tvm.create_schedule(PFilter.op)

    ctx = tvm.cpu(0)
    f1 = tvm.build(s1, [Filter, POutput, PImage], 'llvm')
    f2 = tvm.build(s2, [Image, POutput, PFilter], 'llvm')

    im = tvm.nd.array(np.random.uniform(size=i_shape).astype(Image.dtype), ctx)
    fi = tvm.nd.array(np.random.uniform(size=f_shape).astype(Filter.dtype), ctx)
    po = tvm.nd.array(np.random.uniform(size=p_shape).astype(POutput.dtype), ctx)

    pf = tvm.nd.array(np.zeros(f_shape, dtype = PFilter.dtype), ctx)
    pi = tvm.nd.array(np.zeros(i_shape, dtype = PImage.dtype), ctx)

    f1(fi, po, pi)
    f2(im, po, pf)

    pi_np, pf_np = target(im.asnumpy(), fi.asnumpy(), po.asnumpy())

    passed = 1
    try:
        tvm.testing.assert_allclose(pf.asnumpy(), pf_np, rtol=1e-5)
        tvm.testing.assert_allclose(pi.asnumpy(), pi_np, rtol=1e-5)
    except AssertionError as e:
        passed = 0
    return passed

def test_rconv2d(func, target):
    score = 0
    score += verify_rconv2d(func, target, 1, 32, 32, 3, 3, 3, 16)
    score += verify_rconv2d(func, target, 1, 32, 32, 3, 5, 5, 16)
    score += verify_rconv2d(func, target, 1, 32, 32, 3, 1, 1, 16)
    score += verify_rconv2d(func, target, 1, 7, 7, 8, 1, 1, 8)
    score += verify_rconv2d(func, target, 1, 7, 7, 8, 3, 3, 16)
    score += verify_rconv2d(func, target, 4, 32, 32, 3, 3, 3, 16)
    score += verify_rconv2d(func, target, 4, 32, 32, 3, 5, 5, 16)
    score += verify_rconv2d(func, target, 4, 32, 32, 3, 1, 1, 16)
    score += verify_rconv2d(func, target, 4, 7, 7, 8, 1, 1, 8)
    score += verify_rconv2d(func, target, 4, 7, 7, 8, 3, 3, 16)
    return score

def verify_relu(func, target, batch_size, image_height, image_width, in_channels):
    i_shape = (batch_size, image_height, image_width, in_channels)
    
    Image = tvm.placeholder(i_shape, name='Image')

    Output = func(Image)
    s = tvm.create_schedule(Output.op)

    ctx = tvm.cpu(0)
    f = tvm.build(s, [Image, Output], 'llvm')

    im = tvm.nd.array(np.random.uniform(size=i_shape).astype(Image.dtype), ctx)

    ot = tvm.nd.array(np.zeros(i_shape, dtype = Output.dtype), ctx)

    f(im, ot)

    ot_np = target(im.asnumpy())

    passed = 1
    try:
        tvm.testing.assert_allclose(ot.asnumpy(), ot_np, rtol=1e-5)
    except AssertionError as e:
        passed = 0
    return passed

def test_relu(func, target):
    score = 0
    score += verify_relu(func, target, 1, 32, 32, 28)
    score += verify_relu(func, target, 1, 16, 16, 64)
    score += verify_relu(func, target, 1, 14, 14, 32)
    score += verify_relu(func, target, 1, 7, 7, 128)
    score += verify_relu(func, target, 1, 2, 2, 256)
    score += verify_relu(func, target, 4, 32, 32, 28)
    score += verify_relu(func, target, 4, 16, 16, 64)
    score += verify_relu(func, target, 4, 14, 14, 32)
    score += verify_relu(func, target, 4, 7, 7, 128)
    score += verify_relu(func, target, 4, 2, 2, 256)
    return score

def verify_rrelu(func, target, batch_size, image_height, image_width, in_channels):
    i_shape = (batch_size, image_height, image_width, in_channels)
    
    Image = tvm.placeholder(i_shape, name='Image')
    POutput = tvm.placeholder(i_shape, name='POutput')

    PImage = func(Image, POutput)
    s = tvm.create_schedule(PImage.op)

    ctx = tvm.cpu(0)
    f = tvm.build(s, [Image, POutput, PImage], 'llvm')

    im = tvm.nd.array(np.random.uniform(size=i_shape).astype(Image.dtype), ctx)
    po = tvm.nd.array(np.random.uniform(size=i_shape).astype(POutput.dtype), ctx)

    pi = tvm.nd.array(np.zeros(i_shape, dtype = PImage.dtype), ctx)

    f(im, po, pi)

    pi_np = target(im.asnumpy(), po.asnumpy())

    passed = 1
    try:
        tvm.testing.assert_allclose(pi.asnumpy(), pi_np, rtol=1e-5)
    except AssertionError as e:
        passed = 0
    return passed

def test_rrelu(func, target):
    score = 0
    score += verify_rrelu(func, target, 1, 32, 32, 28)
    score += verify_rrelu(func, target, 1, 16, 16, 64)
    score += verify_rrelu(func, target, 1, 14, 14, 32)
    score += verify_rrelu(func, target, 1, 7, 7, 128)
    score += verify_rrelu(func, target, 1, 2, 2, 256)
    score += verify_rrelu(func, target, 4, 32, 32, 28)
    score += verify_rrelu(func, target, 4, 16, 16, 64)
    score += verify_rrelu(func, target, 4, 14, 14, 32)
    score += verify_rrelu(func, target, 4, 7, 7, 128)
    score += verify_rrelu(func, target, 4, 2, 2, 256)
    return score

def verify_pooling(func, target, batch_size, image_height, image_width, in_channels):
    i_shape = (batch_size, image_height, image_width, in_channels)
    out_height = image_height // 2
    out_width = image_width // 2
    o_shape = (batch_size, out_height, out_width, in_channels)

    Image = tvm.placeholder(i_shape, name='Image')

    Output = func(Image)
    s = tvm.create_schedule(Output.op)

    ctx = tvm.cpu(0)
    f = tvm.build(s, [Image, Output], 'llvm')

    im = tvm.nd.array(np.random.uniform(size=i_shape).astype(Image.dtype), ctx)

    ot = tvm.nd.array(np.zeros(o_shape, dtype = Output.dtype), ctx)

    f(im, ot)

    ot_np = target(im.asnumpy()).transpose((0, 2, 3, 1))

    passed = 1
    try:
        tvm.testing.assert_allclose(ot.asnumpy(), ot_np, rtol=1e-5)
    except AssertionError as e:
        passed = 0
    return passed

def test_pooling(func, target):
    score = 0
    score += verify_pooling(func, target, 1, 32, 32, 28)
    score += verify_pooling(func, target, 1, 16, 16, 64)
    score += verify_pooling(func, target, 1, 14, 14, 32)
    score += verify_pooling(func, target, 1, 7, 7, 128)
    score += verify_pooling(func, target, 1, 2, 2, 256)
    score += verify_pooling(func, target, 4, 32, 32, 28)
    score += verify_pooling(func, target, 4, 16, 16, 64)
    score += verify_pooling(func, target, 4, 14, 14, 32)
    score += verify_pooling(func, target, 4, 7, 7, 128)
    score += verify_pooling(func, target, 4, 2, 2, 256)
    return score

def verify_rpooling(func, target, batch_size, image_height, image_width, in_channels):
    i_shape = (batch_size, image_height, image_width, in_channels)
    out_height = image_height // 2
    out_width = image_width // 2
    o_shape = (batch_size, out_height, out_width, in_channels)

    Image = tvm.placeholder(i_shape, name='Image')
    POutput = tvm.placeholder(o_shape, name='POutput')
    Index = tvm.placeholder(o_shape, name='Index', dtype="int32")

    PImage = func(Image, Index, POutput)
    s = tvm.create_schedule(PImage.op)

    ctx = tvm.cpu(0)
    f = tvm.build(s, [Image, Index, POutput, PImage], 'llvm')

    im = tvm.nd.array(np.random.uniform(size=i_shape).astype(Image.dtype), ctx)
    po = tvm.nd.array(np.random.uniform(size=o_shape).astype(POutput.dtype), ctx)

    pi = tvm.nd.array(np.zeros(i_shape, dtype = PImage.dtype), ctx)

    pi_np, ix1 = target(im.asnumpy(), po.asnumpy())
    ix = tvm.nd.array(ix1.astype(np.int32), ctx)

    f(im, ix, po, pi)

    passed = 1
    try:
        tvm.testing.assert_allclose(pi.asnumpy(), pi_np, rtol=1e-5)
    except AssertionError as e:
        passed = 0
    return passed

def test_rpooling(func, target):
    score = 0
    score += verify_rpooling(func, target, 1, 32, 32, 28)
    score += verify_rpooling(func, target, 1, 16, 16, 64)
    score += verify_rpooling(func, target, 1, 14, 14, 32)
    score += verify_rpooling(func, target, 1, 7, 7, 128)
    score += verify_rpooling(func, target, 1, 2, 2, 256)
    score += verify_rpooling(func, target, 4, 32, 32, 28)
    score += verify_rpooling(func, target, 4, 16, 16, 64)
    score += verify_rpooling(func, target, 4, 14, 14, 32)
    score += verify_rpooling(func, target, 4, 7, 7, 128)
    score += verify_rpooling(func, target, 4, 2, 2, 256)
    return score

def verify_flatten(func, target, batch_size, image_height, image_width, in_channels):
    i_shape = (batch_size, image_height, image_width, in_channels)
    out_size = image_height * image_width * in_channels
    o_shape = (batch_size, out_size)

    Image = tvm.placeholder(i_shape, name='Image')

    Output = func(Image)
    s = tvm.create_schedule(Output.op)

    ctx = tvm.cpu(0)
    f = tvm.build(s, [Image, Output], 'llvm')

    im = tvm.nd.array(np.random.uniform(size=i_shape).astype(Image.dtype), ctx)

    ot = tvm.nd.array(np.zeros(o_shape, dtype = Output.dtype), ctx)

    f(im, ot)

    ot_np = target(im.asnumpy())

    passed = 1
    try:
        tvm.testing.assert_allclose(ot.asnumpy(), ot_np, rtol=1e-5)
    except AssertionError as e:
        passed = 0
    return passed

def test_flatten(func, target):
    score = 0
    score += verify_flatten(func, target, 1, 32, 32, 28)
    score += verify_flatten(func, target, 1, 16, 16, 64)
    score += verify_flatten(func, target, 1, 14, 14, 32)
    score += verify_flatten(func, target, 1, 7, 7, 128)
    score += verify_flatten(func, target, 1, 2, 2, 256)
    score += verify_flatten(func, target, 4, 32, 32, 28)
    score += verify_flatten(func, target, 4, 16, 16, 64)
    score += verify_flatten(func, target, 4, 14, 14, 32)
    score += verify_flatten(func, target, 4, 7, 7, 128)
    score += verify_flatten(func, target, 4, 2, 2, 256)
    return score

def verify_rflatten(func, target, batch_size, image_height, image_width, in_channels):
    i_shape = (batch_size, image_height, image_width, in_channels)
    out_size = image_height * image_width * in_channels
    o_shape = (batch_size, out_size)

    Image = tvm.placeholder(i_shape, name='Image')
    POutput = tvm.placeholder(o_shape, name='POutput')

    PImage = func(Image, POutput)
    s = tvm.create_schedule(PImage.op)

    ctx = tvm.cpu(0)
    f = tvm.build(s, [Image, POutput, PImage], 'llvm')
    
    im = tvm.nd.array(np.random.uniform(size=i_shape).astype(Image.dtype), ctx)
    po = tvm.nd.array(np.random.uniform(size=o_shape).astype(POutput.dtype), ctx)

    pi = tvm.nd.array(np.zeros(i_shape, dtype = PImage.dtype), ctx)

    f(im, po, pi)

    pi_np = target(im.asnumpy(), po.asnumpy())

    passed = 1
    try:
        tvm.testing.assert_allclose(pi.asnumpy(), pi_np, rtol=1e-5)
    except AssertionError as e:
        passed = 0
    return passed

def test_rflatten(func, target, ):
    score = 0
    score += verify_rflatten(func, target, 1, 32, 32, 28)
    score += verify_rflatten(func, target, 1, 16, 16, 64)
    score += verify_rflatten(func, target, 1, 14, 14, 32)
    score += verify_rflatten(func, target, 1, 7, 7, 128)
    score += verify_rflatten(func, target, 1, 2, 2, 256)
    score += verify_rflatten(func, target, 4, 32, 32, 28)
    score += verify_rflatten(func, target, 4, 16, 16, 64)
    score += verify_rflatten(func, target, 4, 14, 14, 32)
    score += verify_rflatten(func, target, 4, 7, 7, 128)
    score += verify_rflatten(func, target, 4, 2, 2, 256)
    return score

def verify_fullyconn(func, target, batch_size, input_size, out_size):
    i_shape = (batch_size, input_size)
    w_shape = (input_size, out_size)
    o_shape = (batch_size, out_size)

    Input = tvm.placeholder(i_shape, name='Input')
    Weight = tvm.placeholder(w_shape, name='Weight')

    Output = func(Input, Weight)
    s = tvm.create_schedule(Output.op)

    ctx = tvm.cpu(0)
    f = tvm.build(s, [Input, Weight, Output], 'llvm')

    ip = tvm.nd.array(np.random.uniform(size=i_shape).astype(Input.dtype), ctx)
    wt = tvm.nd.array(np.random.uniform(size=w_shape).astype(Weight.dtype), ctx)

    ot = tvm.nd.array(np.zeros(o_shape, dtype = Output.dtype), ctx)

    f(ip, wt, ot)

    ot_np = target(ip.asnumpy(), wt.asnumpy())

    passed = 1
    try:
        tvm.testing.assert_allclose(ot.asnumpy(), ot_np, rtol=1e-5)
    except AssertionError as e:
        passed = 0
    return passed

def test_fullyconn(func, target):
    score = 0
    score += verify_fullyconn(func, target, 1, 100, 200)
    score += verify_fullyconn(func, target, 1, 1000, 200)
    score += verify_fullyconn(func, target, 1, 100, 2000)
    score += verify_fullyconn(func, target, 1, 128, 256)
    score += verify_fullyconn(func, target, 1, 1024, 1024)
    score += verify_fullyconn(func, target, 4, 100, 200)
    score += verify_fullyconn(func, target, 4, 1000, 200)
    score += verify_fullyconn(func, target, 4, 100, 2000)
    score += verify_fullyconn(func, target, 4, 128, 256)
    score += verify_fullyconn(func, target, 4, 1024, 1024)
    return score

def verify_rfullyconn(func, target, batch_size, input_size, out_size):
    i_shape = (batch_size, input_size)
    w_shape = (input_size, out_size)
    o_shape = (batch_size, out_size)

    Input = tvm.placeholder(i_shape, name='Input')
    Weight = tvm.placeholder(w_shape, name='Weight')
    POutput = tvm.placeholder(o_shape, name='POutput')

    PWeight, PInput = func(Input, Weight, POutput)
    s1 = tvm.create_schedule(PWeight.op)
    s2 = tvm.create_schedule(PInput.op)

    ctx = tvm.cpu(0)
    f1 = tvm.build(s1, [Input, POutput, PWeight], 'llvm')
    f2 = tvm.build(s2, [Weight, POutput, PInput], 'llvm')

    ip = tvm.nd.array(np.random.uniform(size=i_shape).astype(Input.dtype), ctx)
    wt = tvm.nd.array(np.random.uniform(size=w_shape).astype(Weight.dtype), ctx)
    po = tvm.nd.array(np.random.uniform(size=o_shape).astype(POutput.dtype), ctx)

    pw = tvm.nd.array(np.zeros(w_shape, dtype = PWeight.dtype), ctx)
    pi = tvm.nd.array(np.zeros(i_shape, dtype = PInput.dtype), ctx)

    f1(ip, po, pw)
    f2(wt, po, pi)

    pw_np, pi_np = target(ip.asnumpy(), wt.asnumpy(), po.asnumpy())

    passed = 1
    try:
        tvm.testing.assert_allclose(pw.asnumpy(), pi_np, rtol=1e-5)
        tvm.testing.assert_allclose(pi.asnumpy(), pw_np, rtol=1e-5)
    except AssertionError as e:
        passed = 0
    return passed

def test_rfullyconn(func, target, ):
    score = 0
    score += verify_rfullyconn(func, target, 1, 100, 200)
    score += verify_rfullyconn(func, target, 1, 1000, 200)
    score += verify_rfullyconn(func, target, 1, 100, 2000)
    score += verify_rfullyconn(func, target, 1, 128, 256)
    score += verify_rfullyconn(func, target, 1, 1024, 1024)
    score += verify_rfullyconn(func, target, 4, 100, 200)
    score += verify_rfullyconn(func, target, 4, 1000, 200)
    score += verify_rfullyconn(func, target, 4, 100, 2000)
    score += verify_rfullyconn(func, target, 4, 128, 256)
    score += verify_rfullyconn(func, target, 4, 1024, 1024)
    return score

def copy_ready(fromfile, tofile):
    first_line = ""
    with open(tofile, "r") as f:
        first_line += f.readline()
    first_line += "\n\n"
    with open(tofile, "w") as f:
        f.write(first_line)
        with open(fromfile, "r") as fin:
            for line in fin:
                f.write(line)

def clean_file(tofile):
    first_line = ""
    with open(tofile, "r") as f:
        first_line += f.readline()
    with open(tofile, "w") as f:
        f.write(first_line)
    
def write_score(id, score_name_lst, score_lst, filename):
    score = sum(score_lst)
    line = "{}: ".format(id)
    for i in range(len(score_name_lst)):
        line += "{}:{} ".format(score_name_lst[i], score_lst[i])
    line += "total:{}\n".format(score)
    with open(filename, "a") as f:
        f.write(line)


def test_and_score():
    sub_dir = "../submits"
    res_dir = "../results"
    res_file = "project_score_midterm.txt"
    res_full_path = os.path.join(res_dir, res_file)
    score_name_lst = ["conv2d", "conv2db", "relu", "relub", "pooling", 
            "poolingb", "flatten", "flattenb", "fullyconn", "fullyconnb"]
    if os.path.exists(res_dir) and os.path.isdir(res_dir) and os.listdir(res_dir):
        print("Warning:", res_dir, "is not empty, you'd better copy the contents and clean it.")
    if not os.path.exists(res_dir) or not os.path.isdir(res_dir):
        os.mkdir(res_dir)
    total_tasks = list(os.listdir(sub_dir))
    len_total_tasks = len(total_tasks)
    count_task = 0
    beg_time = time.time()
    for filename in total_tasks:
        # the task bar
        count_task += 1

        logs = "\rprocessing:{} | [finished/total] = [{}/{}] | [passed: {}s]".format(
            filename, count_task, len_total_tasks, int(time.time() - beg_time))
        sys.stdout.write(logs)
        sys.stdout.flush()

        full_path = os.path.join(sub_dir, filename)
        to_path = "test_frame.py"
        copy_ready(full_path, to_path)
        reload(test_frame)
        try:
            score1 = test_conv2d(test_frame.conv2d, testing.conv2d_pytorch)
        except Exception:
            score1 = 0
        # print('testing rconv2d...')
        try:
            score2 = test_rconv2d(test_frame.conv2db, testing.rconv2d_pytorch)
        except Exception:
            score2 = 0
        # print('testing ReLU...')
        try:
            score3 = test_relu(test_frame.relu, testing.relu_pytorch)
        except Exception:
            score3 = 0
        # print('testing rReLU...')
        try:
            score4 = test_rrelu(test_frame.relub, testing.rrelu_pytorch)
        except Exception:
            score4 = 0
        # print('testing 2*2 pooling...')
        try:
            score5 = test_pooling(test_frame.pooling, testing.pooling_pytorch)
        except Exception:
            score5 = 0
        # print('testing rpooling...')
        try:
            score6 = test_rpooling(test_frame.poolingb, testing.rpooling_pytorch)
        except Exception:
            score6 = 0
        # print('testing flatten...')
        try:
            score7 = test_flatten(test_frame.flatten, testing.flatten_pytorch)
        except Exception:
            score7 = 0
        # print('testing rflatten...')
        try:
            score8 = test_rflatten(test_frame.flattenb, testing.rflatten_pytorch)
        except Exception:
            score8 = 0
        # print('testing fullyconnected...')
        try:
            score9 = test_fullyconn(test_frame.fullyconn, testing.fullyconn_pytorch)
        except Exception:
            score9 = 0
        # print('testing rfullyconnected...')
        try:
            score10 = test_rfullyconn(test_frame.fullyconnb, testing.rfullyconn_pytorch)
        except Exception:
            score10 = 0
        score_lst = [score1, score2, score3, score4, score5, score6, score7, score8, score9, score10]
        student_id = filename.split(".")[0]
        write_score(student_id, score_name_lst, score_lst, res_full_path)
        clean_file(to_path)
    sys.stdout.write("\nall done! use {}s \n".format(time.time() - beg_time))
        

if __name__ == '__main__':
    test_and_score()


