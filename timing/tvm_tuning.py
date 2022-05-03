import logging
import sys
import numpy as np
import json

import tvm
from tvm import te, topi, testing, autotvm
from tvm.topi.testing import conv2d_nchw_python
import tvm.testing


@autotvm.template("conv2d_optim")
def conv2d(N, H, W, CO, CI, KH, KW, stride, padding):

    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    s = te.create_schedule([conv.op])

    ##### space definition begin #####
    n, f, y, x = s[conv].op.axis
    rc, ry, rx = s[conv].op.reduce_axis

    cfg = autotvm.get_config()
    cfg.define_split("tile_f", f, num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=3)
    cfg.define_split("tile_ry", ry, num_outputs=3)
    cfg.define_split("tile_rx", rx, num_outputs=3)
    # cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    # cfg.define_knob("unroll_explicit", [0, 1])
    ##### space definition end #####

    # inline padding
    pad_data = s[conv].op.input_tensors[0]
    s[pad_data].compute_inline()
    data, raw_data = pad_data, data
    
    output = conv
    OL = s.cache_write(conv, "local")
    
    # create cache stage
    AA = s.cache_read(data, "shared", [OL])
    WW = s.cache_read(kernel, "shared", [OL])
    AL = s.cache_read(AA, "local", [OL])
    WL = s.cache_read(WW, "local", [OL])
    
    # tile and bind spatial axes
    n, f, y, x = s[output].op.axis
    
    bf, vf, _, _ = cfg["tile_f"].apply(s, output, f)
    by, vy, _, _ = cfg["tile_y"].apply(s, output, y)
    bx, vx, _, _ = cfg["tile_x"].apply(s, output, x)
    kernel_scope = n  # this is the scope to attach global config inside this kernel
    """
    
    s[output].bind(bf, te.thread_axis("blockIdx.z"))
    s[output].bind(by, te.thread_axis("blockIdx.y"))
    s[output].bind(bx, te.thread_axis("blockIdx.x"))
    s[output].bind(vf, te.thread_axis("vthread"))
    s[output].bind(vy, te.thread_axis("vthread"))
    s[output].bind(vx, te.thread_axis("vthread"))
    s[output].bind(tf, te.thread_axis("threadIdx.z"))
    s[output].bind(ty, te.thread_axis("threadIdx.y"))
    s[output].bind(tx, te.thread_axis("threadIdx.x"))
    s[output].reorder(n, bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
    s[OL].compute_at(s[output], tx)
    """
    # tile reduction axes
    n, f, y, x = s[OL].op.axis
    rc, ry, rx = s[OL].op.reduce_axis
    rco, rcm, rci = cfg["tile_rc"].apply(s, OL, rc)
    ryo, rym, ryi = cfg["tile_rx"].apply(s, OL, ry)
    rxo, rxm, rxi = cfg["tile_ry"].apply(s, OL, rx)
    
    s[OL].reorder(n, f, y, x, rco, ryo, rxo, rcm, rym, rxm, rci, ryi, rxi)
    fused = s[OL].fuse(n, f, y)
    s[OL].parallel(fused)
    s[OL].vectorize(x)
        
    """
    # parallelize n f loops
    fused = s[OL].fuse(n, f, y)
    s[OL].parallel(fused)
    s[OL].vectorize(x)
    
    s[AA].compute_at(s[OL], rxo)
    s[WW].compute_at(s[OL], rxo)
    s[AL].compute_at(s[OL], rxm)
    s[WL].compute_at(s[OL], rxm)

    # cooperative fetching
    for load in [AA, WW]:
        n, f, y, x = s[load].op.axis
        fused = s[load].fuse(n, f, y, x)
        tz, fused = s[load].split(fused, nparts=cfg["tile_f"].size[2])
        ty, fused = s[load].split(fused, nparts=cfg["tile_y"].size[2])
        tx, fused = s[load].split(fused, nparts=cfg["tile_x"].size[2])
        
        s[load].bind(tz, te.thread_axis("threadIdx.z"))
        s[load].bind(ty, te.thread_axis("threadIdx.y"))
        s[load].bind(tx, te.thread_axis("threadIdx.x"))
    
    # tune unroll
    s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    
    s[output].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)
    """
    return s, [raw_data, kernel, conv]


# logging config (for printing tuning log to screen)
logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

n_configs = 9
Ns = [10, 20, 10, 20, 10, 20, 10, 10, 32]
Hs = [100, 100, 200, 200, 300, 300, 200, 200, 128]
Ws = [100, 100, 200, 200, 300, 300, 200, 200, 128]
COs = [10, 10, 10, 10, 10, 10, 10, 10, 32]
CIs = [10, 10, 10, 10, 10, 10, 10, 10, 64]
KHs = [10, 10, 10, 10, 10, 10, 10, 10, 5]
KWs = [10, 10, 10, 10, 10, 10, 10, 10, 5]
padding_list = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (10, 10), (10, 10), (0, 0)]
stride_list = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (5, 5), (1, 1)]

best_configs = {}

for idx in range(n_configs):
    print(f"\n--------------- CONFIG {idx} --------------\n")
    # define parameters
    N, H, W, CO, CI, KH, KW, strides, padding = Ns[idx], Hs[idx], Ws[idx], COs[idx], CIs[idx], KHs[idx], KWs[idx], stride_list[idx], padding_list[idx]
    task = autotvm.task.create(
        "conv2d_optim", args=(N, H, W, CO, CI, KH, KW, strides, padding), target="llvm"
    )
    print(task.config_space)
    log_file = f"tvm_logs/conv2d_n{N}_h{H}_w{W}_co{CO}_ci{CI}_kh{KH}_kw{KW}_s{strides[0]}x{strides[1]}_p{padding[0]}x{padding[1]}.log"

    # Use local gpu, measure 10 times for every config to reduce variance
    # The timeout of compiling a program is 10 seconds, the timeout for running is 10000 seconds
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(number=20, timeout=100000),
    )

    # Begin tuning, log records to file `conv2d.log`
    # During tuning we will also try many invalid configs, so you are expected to
    # see many error reports. As long as you can see non-zero GFLOPS, it is okay.
    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(
        n_trial=20,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file(log_file)],
    )
    #########################################################################
    # Finally we can inspect the best config from log file, check correctness,
    # and measure running time.

    # inspect the best config
    dispatch_context = autotvm.apply_history_best(log_file)
    best_config = dispatch_context.query(task.target, task.workload)
    print("\nBest config:")
    print(best_config)
    best_configs[log_file] = {"best_config":best_config.to_json_dict()}

    # apply history best from log file
    with autotvm.apply_history_best(log_file):
        with tvm.target.Target("llvm"):
            s, arg_bufs = conv2d(N, H, W, CO, CI, KH, KW, strides, padding)
            # print(tvm.lower(s, arg_bufs, simple_mode=True))        
            func = tvm.build(s, arg_bufs)

    # check correctness
    a_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    w_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    c_np = conv2d_nchw_python(a_np, w_np, strides, padding)

    dev = tvm.cpu()
    a_tvm = tvm.nd.array(a_np, device=dev)
    w_tvm = tvm.nd.array(w_np, device=dev)
    c_tvm = tvm.nd.empty(c_np.shape, device=dev)
    func(a_tvm, w_tvm, c_tvm)

    tvm.testing.assert_allclose(c_np, c_tvm.numpy(), rtol=1e-2)

    # Evaluate running time. Here we choose a large repeat number (400) to reduce the noise
    # and the overhead of kernel launch. You can also use nvprof to validate the result.
    evaluator = func.time_evaluator(func.entry_name, dev, number=50)
    time_elapsed = evaluator(a_tvm, w_tvm, c_tvm).mean
    print("Time cost of this operator: %f" % time_elapsed)
    best_configs[log_file]["exec_time"] = time_elapsed

    with open("tvm_logs/best_configs.json", "w") as f:
        json.dump(best_configs, f)
