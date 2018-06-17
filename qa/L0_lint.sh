#!/bin/bash

pip install cpplint pylint==1.8.3 astroid==1.6.1
cd .. && make lint
