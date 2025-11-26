# FROM nvcr.io/nvstaging/tao/tao_pytorch_base_image@sha256:4d6c2f98bcb233a7ab7c87f187b3d5a3e1c405d1206fb2b0d3c7b410bb93e84d
# Select base image based on architecture
ARG TARGETARCH # provided by Docker if using BuildKit. Else pass in explicitly.
ARG ARM64_DIGEST=841a5b075f605312e63bdcab5eeeea66113353f1646745a1269935716f30c653
ARG X86_DIGEST=f0ae0442367e93df8b0644e125f3969c9e79c2677ec49a8e9de53acb5a49026c

FROM nvcr.io/nvstaging/tao/tao_pytorch_base_image@sha256:${ARM64_DIGEST} AS base-arm64
FROM nvcr.io/nvstaging/tao/tao_pytorch_base_image@sha256:${X86_DIGEST} AS base-amd64

FROM base-${TARGETARCH}

# Installing the TLT packages.
WORKDIR /opt/nvidia

# NGC Security scan related fixes.
# Low-level packages in apt are updated to resolve published security vulnerabilities ASAP
# We retrieve these updates and upgrade the critical/high vulnerability ones
RUN apt-get update && \
    apt --only-upgrade install linux-libc-dev -y && \
    apt --only-upgrade install curl -y && \
    apt --only-upgrade install libc-bin -y && \
    apt --only-upgrade install libc6-dev -y && \
    apt --only-upgrade install libc6 -y && \
    apt --only-upgrade install libc-dev-bin -y && \
    apt --only-upgrade install libcurl3-gnutls -y && \
    apt --only-upgrade install libcurl4 -y && \
    apt --only-upgrade install libcurl4-openssl-dev -y && \
    apt --only-upgrade install openssh-client -y

RUN rm -rf /opt/nvidia/entrypoint.d/*.txt
COPY release/docker/entrypoint.d/* /opt/nvidia/entrypoint.d/

# Installing TAO-Core
COPY tao-core tao-core
RUN cd tao-core && bash release/python/build_wheel.sh && \
    find dist/ -name "nvidia_tao_core*.whl" -type f | xargs -n 1 pip install && \
    cp nvidia_tao_core/microservices/nginx.conf /etc/nginx/ && \
    cd .. && rm -rf tao-core

COPY dist/*.whl /opt/nvidia/wheels/
RUN cd wheels && ls ./*.whl|xargs -I'{}' python -m pip install '{}' \
    && rm *.whl

# Copying tlt-converter and building it in the docker.
# ENV TRT_LIB_PATH="/usr/lib/x86_64-linux-gnu"
# ENV TRT_INCLUDE_PATH="/usr/include/x86_64-linux-gnu"
# COPY tao-converter /opt/nvidia/tools
# WORKDIR /opt/nvidia/tools
# RUN make \
#     && rm -rf Makefile *.cc *.h *.o
# RUN ln -s tao-converter converter

ENV NVIDIA_PRODUCT_NAME "TAO Toolkit"
ENV TAO_TOOLKIT_VERSION="6.25.10"
ENV NVIDIA_TAO_TOOLKIT_VERSION="${TAO_TOOLKIT_VERSION}-PyTorch"
ENV CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Defining the telemetry URL.
ENV TAO_TELEMETRY_SERVER="https://api.tao.ngc.nvidia.com"

EXPOSE 8000

# Microservices entrypoint
ENV FLASK_APP=nvidia_tao_core.microservices.app

ENV RUN_CLI=0

CMD if [ "$RUN_CLI" = "1" ]; then \
        /bin/bash; \
    else \
        /bin/bash $(get-microservice-script); \
    fi