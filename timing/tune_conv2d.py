import logging
import sys
import numpy as np
import tvm
from tvm import te, autotvm
from tvm.topi.testing import conv2d_nchw_python

@autotvm.template("conv2d_optim")
def conv2d_optim(ih, iw, kh, kw, co, ci, batches, stride, padding, dilation):
    image = te.placeholder((batches, ci, ih, iw), name="image", dtype="float32")
    kernel = te.placeholder((co, ci, kh, kw), name="kernel", dtype="float32")
    output = tvm.topi.nn.conv2d_nchw(image, kernel, stride, padding, dilation, out_dtype="float32")
    s = te.create_schedule([output.op])
    n, ch, y, x = s[output].op.axis

    # define the search space
    cfg = autotvm.get_config()
    cfg.define_knob("tile_y", [1, 2, 4, 8, 16])
    cfg.define_knob("tile_x", [1, 2, 4, 8, 16])
    # cfg.define_split("tile_y", y, num_outputs=2)
    # cfg.define_split("tile_x", x, num_outputs=2)

    # schedule according to config
    yo, yi = s[output].split(y, cfg["tile_y"].val)
    xo, xi = s[output].split(x, cfg["tile_x"].val)
    # yo, yi = cfg["tile_y"].apply(s, output, y)
    # xo, xi = cfg["tile_x"].apply(s, output, x)

    # print(tvm.lower(s, [image, kernel, output], simple_mode=True))
    return s, [image, kernel, output]

def gen_code():    
    batches = te.var("batches")
    ci = te.var("ci")
    co = te.var("co")
    ih = te.var("ih")
    iw = te.var("iw")
    kh = te.var("kh")
    kw = te.var("kw")
    image = te.placeholder((batches, ci, ih, iw), name="image")
    kernel = te.placeholder((co, ci, kh, kw), name="kernel")
    output = tvm.topi.nn.conv2d_nhwc(image, kernel, (1, 1), (0, 0), (1, 1), out_dtype="float32")
    s = te.create_schedule([output.op])
    n, ch, y, x = s[output].op.axis
    y_outer, y_inner = s[output].split(y, factor=8) 
    print(tvm.lower(s, [image, kernel, output], simple_mode=True))
    
if __name__ == "__main__":
    ih = iw = 128
    kh = kw = 5
    batches = 10
    ci = 64
    co = 32
    stride = (1, 1)
    dilation = (1, 1)
    padding = (0, 0)
    
    # set up a tuning task
    task = autotvm.task.create("conv2d_optim", args=(ih, iw, kh, kw, co, ci, batches, stride, padding, dilation), target="llvm")
    print(task.config_space)
    # printing tuning log
    logging.getLogger("autotvm").setLevel(logging.DEBUG)
    logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

    # use all CPU cores to compile the program
    # take 5 measurements in this case and average them
    measure_option = autotvm.measure_option(builder="local", runner=autotvm.LocalRunner(number=5))

    # actual tuning
    # change to XGBTuner for larger search spaces 
    tuner = autotvm.tuner.RandomTuner(task)
    tuner.tune(
            n_trial=10,
            measure_option=measure_option,
            callbacks=[autotvm.callback.log_to_file("conv2d_optim.log")],
            )
    # compile the program with the best config from tuning
    with autotvm.apply_history_best("conv2d_optim.log"):
        with tvm.target.Target("llvm"):
            s, args = conv2d_optim(ih, iw, kh, kw, co, ci, batches, stride, padding, dilation)
            func = tvm.build(s, args)

    # check correctness
    i_np = np.random.uniform(size=(batches, ci, ih, iw)).astype(np.float32)
    k_np = np.random.uniform(size=(co, ci, kh, kw)).astype(np.float32)
    res_np = conv2d_nchw_python(i_np, k_np, stride, padding, dilation)
    
    i_tvm = tvm.nd.array(i_np)
    k_tvm = tvm.nd.array(k_np)
    res_tvm = tvm.nd.empty(res_np.shape)
    func(i_tvm, k_tvm, res_tvm)
    # tvm.testing.assert_allclose(res_np, res_tvm.numpy(), rtol=1e-7)
    
    # evaluate running time
    evaluator = func.time_evaluator(func.entry_name, dev=tvm.cpu(), number=5)
    time_elapsed = evaluator(i_tvm, k_tvm, res_tvm).mean
    print(f"Time elapsed: {time_elapsed}")
    






