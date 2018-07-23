# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import numpy as np
import mxnet as mx
from common import *
from lenet5_common import get_iters

def run_inference(sym, arg_params, aux_params, mnist, all_test_labels, batch_size):
    """Run inference with either MXNet or TensorRT"""

    shared_buffer = merge_dicts(arg_params, aux_params)
    if not get_use_tensorrt():
        shared_buffer = dict([(k, v.as_in_context(mx.gpu(0))) for k, v in shared_buffer.items()])
    executor = sym.simple_bind(ctx=mx.gpu(0),
                               data=(batch_size,) +  mnist['test_data'].shape[1:],
                               softmax_label=(batch_size,),
                               shared_buffer=shared_buffer,
                               grad_req='null',
                               force_rebind=True)

    # Get this value from all_test_labels
    # Also get classes from the dataset
    num_ex = 10000
    all_preds = np.zeros([num_ex, 10])
    test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

    example_ct = 0

    for idx, dbatch in enumerate(test_iter):
        executor.arg_dict["data"][:] = dbatch.data[0]
        executor.forward(is_train=False)
        offset = idx*batch_size
        extent = batch_size if num_ex - offset > batch_size else num_ex - offset
        all_preds[offset:offset+extent, :] = executor.outputs[0].asnumpy()[:extent]
        example_ct += extent

    all_preds = np.argmax(all_preds, axis=1)
    matches = (all_preds[:example_ct] == all_test_labels[:example_ct]).sum()

    percentage = 100.0 * matches / example_ct

    return percentage

def test_tensorrt_inference():
    """Run LeNet-5 inference comparison between MXNet and TensorRT."""
    check_tensorrt_installation()
    mnist = mx.test_utils.get_mnist()
    num_epochs = 10
    batch_size = 128
    model_name = 'lenet5'
    model_dir = os.getenv("LENET_MODEL_DIR", "/tmp")
    model_file = '%s/%s-symbol.json' % (model_dir, model_name)
    params_file = '%s/%s-%04d.params' % (model_dir, model_name, num_epochs)

    _, _, _, all_test_labels = get_iters(mnist, batch_size)

    # Load serialized MXNet model (model-symbol.json + model-epoch.params)
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_name, num_epochs)

    print("LeNet-5 test")
    print("Running inference in MXNet")
    set_use_tensorrt(False)
    mx_pct = run_inference(sym, arg_params, aux_params, mnist,
                           all_test_labels, batch_size=batch_size)

    print("Running inference in MXNet-TensorRT")
    set_use_tensorrt(True)
    trt_pct = run_inference(sym, arg_params, aux_params, mnist,
                            all_test_labels,  batch_size=batch_size)

    print("MXNet accuracy: %f" % mx_pct)
    print("MXNet-TensorRT accuracy: %f" % trt_pct)

    assert abs(mx_pct - trt_pct) < 1e-2, \
        """Diff. between MXNet & TensorRT accuracy too high:
           MXNet = %f, TensorRT = %f""" % (mx_pct, trt_pct)


if __name__ == '__main__':
    import nose
    nose.runmodule()
