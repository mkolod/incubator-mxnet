FROM nvidia/cuda:9.1-cudnn7-devel

ENV MXNET_VERSION 1.2.0+
LABEL com.nvidia.mxnet.version="${MXNET_VERSION}"
ENV NVIDIA_MXNET_VERSION 18.07

ARG USE_TRT=1
ARG PYVER=3.5
ENV ONNX_NAMESPACE onnx

RUN PYSFX=`[ "$PYVER" != "2.7" ] && echo "$PYVER" | cut -c1-1 || echo ""` && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        wget \
        git \
        libatlas-base-dev \
        pkg-config \
        libtiff5-dev \
        libjpeg8-dev \
        zlib1g-dev \
        python$PYVER-dev \
        autoconf \
        automake \
        libtool \
        nasm \
        unzip && \
    rm -rf /var/lib/apt/lists/*

# Need a newer version of CMake for ONNX and onnx-tensorrt
RUN cd /usr/local/src && \
    wget https://cmake.org/files/v3.8/cmake-3.8.2.tar.gz && \
    tar -xvf cmake-3.8.2.tar.gz && \
    cd cmake-3.8.2 && \
    ./bootstrap && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf cmake*

# Make sure symlinks exist for either python 2 or 3
RUN rm -f /usr/bin/python && ln -s /usr/bin/python$PYVER /usr/bin/python
RUN MAJ=`echo "$PYVER" | cut -c1-1` && \
    rm -f /usr/bin/python$MAJ && ln -s /usr/bin/python$PYVER /usr/bin/python$MAJ

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# We need to force NumPy 1.13.3 because default is 1.14.1 right now
# and that issues MxNet warnings since it's not officially supported
# Install NumPy before the pip install --upgrade
RUN pip install numpy==1.13.3
RUN pip install --upgrade --no-cache-dir setuptools requests

# The following are needed for Sockeye on python 3+ only.
RUN [ "$PYVER" = "2.7" ] || pip install unidecode tqdm pyyaml

RUN OPENCV_VERSION=3.1.0 && \
    wget -q -O - https://github.com/Itseez/opencv/archive/${OPENCV_VERSION}.tar.gz | tar -xzf - && \
    cd /opencv-${OPENCV_VERSION} && \
    cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr \
          -DWITH_CUDA=OFF -DWITH_1394=OFF \
          -DBUILD_opencv_cudalegacy=OFF -DBUILD_opencv_stitching=OFF -DWITH_IPP=OFF . && \
    make -j"$(nproc)" install && \
    rm -rf /opencv-${OPENCV_VERSION}

# libjpeg-turbo
RUN JPEG_TURBO_VERSION=1.5.2 && \
    wget -q -O - https://github.com/libjpeg-turbo/libjpeg-turbo/archive/${JPEG_TURBO_VERSION}.tar.gz | tar -xzf - && \
    cd /libjpeg-turbo-${JPEG_TURBO_VERSION} && \
    autoreconf -fiv && \
    ./configure --enable-shared --prefix=/usr 2>&1 >/dev/null && \
    make -j"$(nproc)" install 2>&1 >/dev/null && \
    rm -rf /libjpeg-turbo-${JPEG_TURBO_VERSION}

WORKDIR /

# Install protoc 3.5 and build protobuf here (for onnx and onnx-tensorrt)
RUN if [ $USE_TRT = "1" ]; \
    then \
      echo "TensorRT build enabled. Installing Protobuf."; \
      git clone --recursive -b 3.5.1.1 https://github.com/google/protobuf.git; \
      cd protobuf; \
      ./autogen.sh; \
      ./configure; \
      make -j$(nproc); \
      make install; \
      ldconfig; \
    else \
      echo "TensorRT build disabled. Not installing Protobuf."; \
    fi

# Install TensorRT 4.0 for CUDA 9
RUN if [ $USE_TRT = "1" ]; \
    then \
      echo "TensorRT build enabled. Installing TensorRT."; \
      wget -qO tensorrt.deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvinfer-runtime-trt-repo-ubuntu1604-3.0.4-ga-cuda9.0_1.0-1_amd64.deb; \
      dpkg -i tensorrt.deb; \
      apt-get update; \
      apt-get install -y --allow-downgrades libnvinfer-dev; \
      rm tensorrt.deb; \
    else \
        echo "TensorRT build disabled. Not installing TensorRT."; \
    fi

WORKDIR /opt/mxnet
COPY . .

ENV MXNET_HOME "/opt/mxnet"
ENV MXNET_CUDNN_AUTOTUNE_DEFAULT 2

RUN cp make/config.mk . && \
   echo "USE_CUDA=1" >> config.mk && \
   echo "USE_CUDNN=1" >> config.mk && \
   echo "CUDA_ARCH :=" \
        "-gencode arch=compute_52,code=sm_52" \
        "-gencode arch=compute_60,code=sm_60" \
        "-gencode arch=compute_61,code=sm_61" \
        "-gencode arch=compute_70,code=sm_70" \
        "-gencode arch=compute_70,code=compute_70" >> config.mk && \
    echo "USE_CUDA_PATH=/usr/local/cuda" >> config.mk && \
    echo "USE_LIBJPEG_TURBO=1" >> config.mk && \
    echo "USE_LIBJPEG_TURBO_PATH=/usr" >> config.mk

RUN if [ $USE_TRT = "1" ]; \
    then \
      echo "TensorRT build enabled. Adding flags to config.mk."; \
      echo "USE_TENSORRT=1" >> config.mk; \
      echo "ONNX_NAMESPACE=$ONNX_NAMESPACE" >> config.mk; \
    else \
      echo "TensorRT build disabled. Not adding TensorRT flags to config.mk."; \
    fi

ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/lib

# Building ONNX, then onnx-tensorrt
WORKDIR /opt/mxnet/3rdparty/onnx-tensorrt/third_party/onnx

RUN if [ $USE_TRT = "1" ]; \
  then \
    echo "TensorRT build enabled. Installing ONNX."; \
    rm -rf build; \
    mkdir build; \
    cd build; \
    cmake -DCMAKE_CXX_FLAGS=-I/usr/include/python${PYVER} -DBUILD_SHARED_LIBS=ON ..; \
    make -j$(nproc); \
    make install; \
    ldconfig; \
    cd ..; \
    mkdir /usr/include/x86_64-linux-gnu/onnx; \
    cp build/onnx/onnx*pb.* /usr/include/x86_64-linux-gnu/onnx; \
    cp build/libonnx.so /usr/local/lib && ldconfig; \
  else \
    echo "TensorRT build disabled. Not installing ONNX."; \
  fi

WORKDIR /opt/mxnet/3rdparty/onnx-tensorrt

RUN if [ $USE_TRT = "1" ]; \
  then \
    echo "TensorRT build enabled. Installing onnx-tensorrt."; \
    mkdir build && cd build && cmake ..; \
    make -j$(nproc); \
    make install; \
    ldconfig; \
  else \
    echo "TensorRT build disabled. Not installing onnx-tensorrt."; \
  fi

WORKDIR /opt/mxnet

RUN make -j$(nproc) && \
    mv lib/libmxnet.so /usr/local/lib && \
    ldconfig && \
    make clean && \
    cd python && \
    pip install -e .
