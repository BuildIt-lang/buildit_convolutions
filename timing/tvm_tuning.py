import logging
import sys
import os
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
    OL = conv
    
    # tile reduction axes
    n, f, y, x = s[OL].op.axis
    rc, ry, rx = s[OL].op.reduce_axis
    rco, rcm, rci = cfg["tile_rc"].apply(s, OL, rc)
    ryo, rym, ryi = cfg["tile_rx"].apply(s, OL, ry)
    rxo, rxm, rxi = cfg["tile_ry"].apply(s, OL, rx)
    #yo1, yo2, yi1, yi2 = cfg["tile_y"].apply(s, OL, y)
    #xo1, xo2, xi1, xi2 = cfg["tile_x"].apply(s, OL, x)
    
    s[OL].reorder(n, f, y, x, rco, ryo, rxo, rcm, rym, rxm, rci, ryi, rxi)
    fused = s[OL].fuse(n, f, y)
    s[OL].parallel(fused)
    s[OL].vectorize(x)
        
    return s, [raw_data, kernel, conv]

@autotvm.template("optim_template")
def optim_conv2d(N, H, W, CO, CI, KH, KW, stride, padding):

    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    s = te.create_schedule([conv.op])

    # define config space
    n, f, y, x = s[conv].op.axis
    rc, ry, rx = s[conv].op.reduce_axis
    
    # inline padding
    pad_data = s[conv].op.input_tensors[0]
    s[pad_data].compute_inline()
    data, raw_data = pad_data, data

    cfg = autotvm.get_config()
    
    # loop tiling
    #cfg.define_split("tile_f", f, num_outputs=2)
    cfg.define_split("tile_y", y, num_outputs=2)
    cfg.define_split("tile_x", x, num_outputs=2)
    cfg.define_split("tile_rc", rc, num_outputs=2)
    cfg.define_split("tile_ry", ry, num_outputs=2)
    cfg.define_split("tile_rx", rx, num_outputs=2)
    
    # tile main axes
    #fo, fi = cfg["tile_f"].apply(s, conv, f)
    yo, yi = cfg["tile_y"].apply(s, conv, y)
    xo, xi = cfg["tile_x"].apply(s, conv, x)
    # tile reduction axes
    rco, rci = cfg["tile_rc"].apply(s, conv, rc)
    ryo, ryi = cfg["tile_rx"].apply(s, conv, ry)
    rxo, rxi = cfg["tile_ry"].apply(s, conv, rx)
    
    # loop reordering
    all_loops = [rco, rci, yo, yi, xo, xi, ryo, ryi, rxo, rxi]
    cfg.define_reorder("reorder", all_loops, policy="all")
    ordered_loops = cfg["reorder"].apply(s, conv, all_loops)
    s[conv].reorder(n, f, ordered_loops[0], ordered_loops[1], ordered_loops[2],
            ordered_loops[3], ordered_loops[4], ordered_loops[5], ordered_loops[6],
            ordered_loops[7], ordered_loops[8], ordered_loops[9])
    
    # try unrolling or vectorizing the innermost loop
    cfg.define_annotate("unroll_vec", ordered_loops[-1:], policy="try_unroll_vec")
    annotated = cfg["unroll_vec"].apply(s, conv, ordered_loops[-1:]) 

    # parallelize the outermost loops
    fused = s[conv].fuse(n, f, ordered_loops[0])
    s[conv].parallel(fused)
    
        
    return s, [raw_data, kernel, conv]
@autotvm.template("baseline_template")
def baseline_conv2d(N, H, W, CO, CI, KH, KW, stride, padding):

    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    s = te.create_schedule([conv.op])

    n, f, y, x = s[conv].op.axis
    rc, ry, rx = s[conv].op.reduce_axis
    
    s[conv].reorder(n, f, rc, y, x, ry, rx)

    fused = s[conv].fuse(n, f, rc)
    s[conv].parallel(fused)
    
    return s, [data, kernel, conv]




# logging config (for printing tuning log to screen)
logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

n_configs = 9
Ns = [10, 20, 10, 20, 10, 20, 10, 10, 32]
Hs = [100, 100, 200, 200, 300, 300, 200, 200, 128]
Ws = [100, 100, 200, 200, 300, 300, 200, 200, 128]
COs = [10, 10, 10, 10, 10, 10, 10, 10, 16]
CIs = [10, 10, 10, 10, 10, 10, 10, 10, 64]
KHs = [10, 10, 10, 10, 10, 10, 10, 10, 5]
KWs = [10, 10, 10, 10, 10, 10, 10, 10, 5]
padding_list = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (10, 10), (10, 10), (0, 0)]
stride_list = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (5, 5), (1, 1)]
#templates = [(baseline_conv2d, "baseline_template")]
templates = [(optim_conv2d, "optim_template")]
for conv_func, templ in templates: 
    best_configs = {}
    dest_dir = os.path.join("tvm_logs", templ)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for idx in range(0, n_configs):
        print(f"\n--------------- CONFIG {idx} --------------\n")
        # define parameters
        N, H, W, CO, CI, KH, KW, strides, padding = Ns[idx], Hs[idx], Ws[idx], COs[idx], CIs[idx], KHs[idx], KWs[idx], stride_list[idx], padding_list[idx]
        task = autotvm.task.create(
            templ, args=(N, H, W, CO, CI, KH, KW, strides, padding), target="llvm"
        )
        print(task.config_space)
        config_name = f"conv2d_n{N}_h{H}_w{W}_co{CO}_ci{CI}_kh{KH}_kw{KW}_s{strides[0]}x{strides[1]}_p{padding[0]}x{padding[1]}"
        log_file = f"tvm_logs/{templ}/{config_name}.log"
        code_file = f"tvm_logs/{templ}/{config_name}.txt"

        # Use local gpu, measure 10 times for every config to reduce variance
        # The timeout of compiling a program is 10 seconds, the timeout for running is 10000 seconds
        measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.LocalRunner(number=50, timeout=100000),
        )

        # Begin tuning, log records to file `conv2d.log`
        # During tuning we will also try many invalid configs, so you are expected to
        # see many error reports. As long as you can see non-zero GFLOPS, it is okay.
        tuner = autotvm.tuner.XGBTuner(task)
        num_trials = 50
        tuner.tune(
            n_trial=num_trials,
            measure_option=measure_option,
            callbacks=[autotvm.callback.log_to_file(log_file),
                autotvm.callback.progress_bar(num_trials)],
        )
        #########################################################################
        # Finally we can inspect the best config from log file, check correctness,
        # and measure running time.

        # inspect the best config
        dispatch_context = autotvm.apply_history_best(log_file)
        best_config = dispatch_context.query(task.target, task.workload)
        print("\nBest config:")
        print(best_config)
        best_configs[config_name] = {"best_config":best_config.to_json_dict()}

        # apply history best from log file
        with autotvm.apply_history_best(log_file):
            with tvm.target.Target("llvm"):
                s, arg_bufs = conv_func(N, H, W, CO, CI, KH, KW, strides, padding)
                with open(code_file, "w") as f:
                    f.write(str(tvm.lower(s, arg_bufs, simple_mode=True)))
                    print(tvm.lower(s, arg_bufs, simple_mode=True))        
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
        best_configs[config_name]["exec_time"] = time_elapsed

        with open(os.path.join(dest_dir, "best_configs.json"), "w") as f:
            json.dump(best_configs, f)
