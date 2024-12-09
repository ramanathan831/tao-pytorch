ARG PYTORCH_BASE_IMAGE=nvcr.io/nvidia/pytorch:24.07-py3
FROM ${PYTORCH_BASE_IMAGE}

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
COPY docker/requirements-apt.txt requirements-apt.txt
RUN apt-get upgrade && apt-get update && \
  xargs apt-get install -y < requirements-apt.txt && \
  rm requirements-apt.txt && \
  rm -rf /var/lib/apt/lists/*

# uninstall stuff from base container
RUN pip uninstall -y sacrebleu torchtext
# Installing custom packages in /opt.
WORKDIR /opt

# Clone and checkout TensorRT OSS
# Moving TensorRT to 10.2 branch.
ENV TRT_TAG="release/10.2"
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
    -DCUDA_VERSION=12.5.1 \
    -DCUDNN_VERSION=9.2.1.18 && \
  make -j16 nvinfer_plugin nvinfer_plugin_static && \
  cp out/libnvinfer_plugin.so.10.2.0 /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.10.2.0 && \
  cp out/libnvinfer_plugin_static.a /usr/lib/x86_64-linux-gnu/libnvinfer_plugin_static.a && \
  cd ../../../ && \
  rm -rf trt_oss_src

COPY docker/requirements-pip.txt requirements-pip.txt
RUN  pip install --upgrade pip \
  && pip install -r requirements-pip.txt \
  && rm requirements-pip.txt

# Necessary to resolve this issue https://github.com/open-mmlab/mmdetection/issues/6765
# mmcv<2.2.0 compatible with mmpretrain 1.2.0
ENV FORCE_CUDA="1"
RUN mkdir mmcv_src && \
  cd mmcv_src && \
  echo "$PWD Building MMCV..." && \
  git clone -b v2.1.0 https://github.com/open-mmlab/mmcv.git mmcv && \
  cd mmcv && \
  MMCV_WITH_OPS=1 pip install -v -e . 

COPY docker/requirements-mim.txt requirements-mim.txt
RUN mim install -r requirements-mim.txt \
  && rm requirements-mim.txt

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
  git checkout v0.0.27 && \
  git submodule update --init --recursive && \
  echo '' > requirements.txt && \
  MAX_JOBS=16 pip install . && \
  cd ../../ && \
  rm -rf xformers_src

# Setup user account
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} taotoolkituser && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash taotoolkituser
RUN usermod -aG sudo taotoolkituser
RUN echo 'taotoolkituser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown taotoolkituser /workspace

WORKDIR /tao-pt

CMD [ "/bin/bash" ]