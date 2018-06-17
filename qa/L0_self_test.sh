#!/bin/bash

apt-get update && apt-get install unzip
pip install nose

# We cannot move to latest scipy v1.1 due to a LO test failure. Repro with:
# MXNET_TEST_SEED=898846653 nosetests3 /opt/mxnet/tests/python/unittest/test_sparse_operator.py:test_sparse_mathematical_core
pip install scipy==1.0

nosetests --verbose /opt/mxnet/tests/python/gpu
nosetests --verbose /opt/mxnet/tests/python/train
