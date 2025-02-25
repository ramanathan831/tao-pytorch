ARG PYTORCH_BASE_IMAGE=nvcr.io/nvidia/pytorch:25.01-py3
FROM ${PYTORCH_BASE_IMAGE}

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
COPY docker/requirements-apt.txt requirements-apt.txt
RUN apt-get upgrade && apt-get update && \
  xargs apt-get install -y < requirements-apt.txt && \
  rm requirements-apt.txt && \
  rm -rf /var/lib/apt/lists/*

# Installing custom packages in /opt.
WORKDIR /opt

# Clone and checkout TensorRT OSS
# Moving TensorRT to 10.8 branch.
ENV TRT_TAG="release/10.8"
# Install TRT OSS
RUN mkdir trt_oss_src && \
  cd trt_oss_src && \
  echo "$PWD Building TRT OSS..." && \
  git clone -b $TRT_TAG https://github.com/NVIDIA/TensorRT.git TensorRT && \
  cd TensorRT && \
  git submodule update --init --recursive && \
  mkdir -p build && cd build  && \
  cmake .. \
    -DGPU_ARCHS="53;60;61;70;75;80;86;90" \
    -DTRT_LIB_DIR=/usr/lib/x86_64-linux-gnu \
    -DTRT_OUT_DIR=`pwd`/out \
    -DCUDA_VERSION=12.8.0 \
    -DCUDNN_VERSION=9.7.0.66 && \
  make -j16 nvinfer_plugin nvinfer_plugin_static && \
  cp out/libnvinfer_plugin.so.10.8.0 /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.10.8.0 && \
  cp out/libnvinfer_plugin_static.a /usr/lib/x86_64-linux-gnu/libnvinfer_plugin_static.a && \
  cd ../../../ && \
  rm -rf trt_oss_src

COPY docker/requirements-pip.txt requirements-pip.txt
RUN  pip install --upgrade pip \
  && pip install -r requirements-pip.txt \
  && rm requirements-pip.txt

COPY docker/requirements-pip-pytorch.txt requirements-pip-pytorch.txt
RUN pip install --ignore-installed --no-deps -r requirements-pip-pytorch.txt \
  && rm requirements-pip-pytorch.txt

COPY docker/requirements-pip-odise.txt requirements-pip-odise.txt
RUN pip install --ignore-installed --no-deps -r requirements-pip-odise.txt \
  && rm requirements-pip-odise.txt

RUN mkdir xformers_src && \
  cd xformers_src && \
  git clone --recursive https://github.com/facebookresearch/xformers.git && \
  cd xformers && \
  git checkout v0.0.29.post2 && \
  git submodule update --init --recursive && \
  echo '' > requirements.txt && \
  MAX_JOBS=16 pip install . && \
  cd ../../ && \
  rm -rf xformers_src

# 25.01 DLFW stack includes 12.0+PTX which throws an error when building our custom CUDA kernels
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 10.0"

# Setup user account
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} taotoolkituser && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash taotoolkituser
RUN usermod -aG sudo taotoolkituser
RUN echo 'taotoolkituser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown taotoolkituser /workspace

WORKDIR /tao-pt

CMD [ "/bin/bash" ]