import gluoncv
import mxnet as mx
import multiprocessing
import numpy as np
import os
import sys

from mxnet.gluon.data.vision import transforms
from mxnet import gluon
from time import time

def get_use_tensorrt():
    return int(os.environ.get("MXNET_USE_TENSORRT", 0))

def set_use_tensorrt(status=False):
    os.environ["MXNET_USE_TENSORRT"] = str(int(status))

def get_cifar_classif_model(model_name='cifar_resnet56_v1', use_tensorrt=True,
                      ctx=mx.gpu(0), batch_size=128):

    set_use_tensorrt(use_tensorrt)
    net = gluoncv.model_zoo.get_model(model_name, pretrained=True)
    data = mx.sym.var('data')
    out = net(data)
    softmax = mx.sym.SoftmaxOutput(out, name='softmax')

    all_params = dict([(k, v.data()) for k, v in net.collect_params().items()])

    if not get_use_tensorrt():
        all_params = dict([(k, v.as_in_context(mx.gpu(0))) for k, v in all_params.items()])

    set_use_tensorrt(use_tensorrt)
    executor = softmax.simple_bind(ctx=ctx, data=(batch_size, 3, 32, 32), softmax_label=(batch_size,), grad_req='null',
                                   shared_buffer=all_params, force_rebind=True)
    return executor

def cifar10_infer(data_dir='./data', model_name='cifar_resnet56_v1', use_tensorrt=True, ctx=mx.gpu(0), batch_size=128, num_workers=1):

    executor = get_cifar_classif_model(model_name, use_tensorrt, ctx, batch_size)

    num_ex = 10000
    all_preds = np.zeros([num_ex, 10])

    all_label_test = np.zeros(num_ex)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    data_loader = lambda: gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)

    val_data = data_loader()

    for idx, (data, label) in enumerate(val_data):
        extent = data.shape[0]
        offset = idx*batch_size
        all_label_test[offset:offset+extent] = label.asnumpy()

        # warm-up, but don't use result
        if idx > 10:
            continue
        executor.arg_dict["data"][:extent, :] = data
        executor.forward(is_train=False)
        preds = executor.outputs[0].asnumpy()

    val_data = data_loader()
    example_ct = 0

    start = time()

    for idx, (data, label) in enumerate(val_data):
        extent = data.shape[0]
        executor.arg_dict["data"][:extent, :] = data
        executor.forward(is_train=False)
        preds = executor.outputs[0].asnumpy()
        offset = idx*batch_size
        all_preds[offset:offset+extent, :] = preds[:extent]
        example_ct += extent

    all_preds = np.argmax(all_preds, axis=1)
    matches = (all_preds[:example_ct] == all_label_test[:example_ct]).sum()
    duration = time() - start

    return duration, 100.0 * matches / example_ct

def run_experiment_for(model_name, batch_size, num_workers):
    print("\n===========================================")
    print("Model: %s" % model_name)
    print("===========================================")
    print("*** Running inference using pure MxNet ***\n")
    mx_duration, mx_pct = cifar10_infer(model_name=model_name, batch_size=batch_size, num_workers=num_workers, use_tensorrt=False)
    print("\nMxNet: time elapsed: %.3fs, accuracy: %.2f%%" % (mx_duration, mx_pct))

    print("\n*** Running inference using MxNet + TensorRT ***\n")
    trt_duration, trt_pct = cifar10_infer(model_name=model_name, batch_size=batch_size, num_workers=num_workers, use_tensorrt=True)
    print("TensorRT: time elapsed: %.3fs, accuracy: %.2f%%" % (trt_duration, trt_pct))
    speedup = mx_duration / trt_duration
    print("TensorRT speed-up (not counting compilation): %.2fx" % speedup)

    acc_diff = abs(mx_pct - trt_pct)
    print("Absolute accuracy difference: %f" % acc_diff)
    return speedup, acc_diff


def test_tensorrt_on_cifar_resnets():

    batch_size = 16
    tolerance = 0.1
    num_workers = int(multiprocessing.cpu_count() / 2)

    models = [
        'cifar_resnet20_v1',
        'cifar_resnet56_v1',
        'cifar_resnet110_v1',
        'cifar_resnet20_v2',
        'cifar_resnet56_v2',
        'cifar_resnet110_v2',
        'cifar_wideresnet16_10',
        'cifar_wideresnet28_10',
        'cifar_wideresnet40_8',
        'cifar_resnext29_16x64d'
    ]

    num_models = len(models)

    speedups = np.zeros(num_models, dtype=np.float32)
    acc_diffs = np.zeros(num_models, dtype=np.float32)

    test_start = time()

    for idx, model in enumerate(models):
        speedup, acc_diff = run_experiment_for(model, batch_size, num_workers)
        speedups[idx] = speedup
        acc_diffs[idx] = acc_diff
        assert acc_diff < 0.1, "Accuracy difference between MxNet and TensorRT > 0.1%% for model %s" % model

    print("Perf and correctness checks run on the following models:")
    print(models)
    mean_speedup = np.mean(speedups)
    std_speedup = np.std(speedups)
    print("\nSpeedups:")
    print(speedups)
    print("Speedup range: [%.2f, %.2f]" % (np.min(speedups), np.max(speedups)))
    print("Mean speedup: %.2f" % mean_speedup)
    print("St. dev. of speedups: %.2f" % std_speedup)
    print("\nAcc. differences: %s" % str(acc_diffs))

    test_duration = time() - test_start

    print("Test duration: %.2f seconds" % test_duration)

if __name__ == '__main__':
    test_tensorrt_on_cifar_resnets()
