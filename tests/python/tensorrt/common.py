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
# pylint: disable=unused-import
import unittest
# pylint: enable=unused-import
import numpy as np
import mxnet as mx
from ctypes.util import find_library

def check_tensorrt_installation():
    assert find_library('nvinfer') is not None, "Can't find the TensorRT shared library"

def get_use_tensorrt():
    return int(os.environ.get("MXNET_USE_TENSORRT", 0))

def set_use_tensorrt(status=False):
    os.environ["MXNET_USE_TENSORRT"] = str(int(status))

def merge_dicts(*dict_args):
    """Merge arg_params and aux_params to populate shared_buffer"""
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def get_fp16_infer_for_fp16_graph():
    return int(os.environ.get("MXNET_TENSORRT_USE_FP16_FOR_FP32", 0)) 

def set_fp16_infer_for_fp16_graph(status=False):
    os.environ["MXNET_TENSORRT_USE_FP16_FOR_FP32"] = str(int(status))
