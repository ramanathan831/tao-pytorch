ARG PYTORCH_BASE_IMAGE=nvcr.io/nvidia/pytorch:25.09-py3
FROM ${PYTORCH_BASE_IMAGE}

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
COPY docker/requirements-apt.txt requirements-apt.txt
RUN apt-get upgrade && apt-get update && \
  xargs apt-get install -y < requirements-apt.txt && \
  rm requirements-apt.txt && \
  rm -rf /var/lib/apt/lists/*

# TAO-CORE microservice dependencies
WORKDIR /workspace/k8s
RUN wget https://github.com/kubernetes-client/python/archive/refs/tags/v23.6.0.zip && \
    unzip v23.6.0.zip && \
    cd python-23.6.0 && \
    sed -i "s/PACKAGE_NAME\ \=\ .*/PACKAGE_NAME = \"kubernetes-client\"/" setup.py && \
    pip install --disable-pip-version-check -r requirements.txt && \
    python setup.py install


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

##################
## build xformers
##################

# Xformers commit hash
ARG XFORMERS_COMMIT_HASH=c0eb310a0afbf782b8947501e4f835e30b5a47b1

# Adding 9.0a to the list of supported architectures.
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 9.0a 10.0 12.0+PTX"
ENV GPU_ARCHS="75 80 86 90 90a 100 120"


# flash attention issue: https://nvbugspro.nvidia.com/bug/5440235
ENV NVCC_FLAGS="--ptxas-options=-O2 --ptxas-options=-v"

COPY docker/xformers xformers_src

RUN echo "======Start building xformers from vllm gitlab repository======" && \
  cd xformers_src && \
  echo "======Calculating MAX_JOBS to build xformers parallelly======" && \
  # TLT-5330: 90 GB overhead, 20 GB/job, 4 job min and ${max_cpu} job max
  total_mem_gb=$(grep MemFree /proc/meminfo | awk '{print int($2 / 1024 / 1024)}'); \
  max_jobs=$(( (total_mem_gb - 90) / 20 )); \
  max_jobs=$(( max_jobs < 4 ? 4 : max_jobs )); \
  max_jobs=$(( max_jobs > $(nproc) ? $(nproc) : max_jobs )); \
  echo "======Using MAX_JOBS=${max_jobs} to build xformers======" && \
  MAX_JOBS=${max_jobs} python setup.py bdist_wheel -d build/ && \
  pip install --no-deps build/xformers*.whl && \
  cd ../ && \
  rm -rf xformers_src && \
  echo "======Xformers built and installed successfully======"



# Setup user account
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} taotoolkituser && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash taotoolkituser
RUN usermod -aG sudo taotoolkituser
RUN echo 'taotoolkituser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown taotoolkituser /workspace

WORKDIR /tao-pt

CMD [ "/bin/bash" ]