#!/bin/bash

apt-get update && apt-get install unzip
pip install nose

pip install scipy==1.0

nosetests --verbose /opt/mxnet/tests/python/tensorrt

