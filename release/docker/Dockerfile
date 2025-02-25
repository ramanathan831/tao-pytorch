FROM nvcr.io/nvstaging/tao/tao_pytorch_base_image@sha256:00170d195bbd07a51a0d002180bc7eb5af911d6736bff3090a43968ba2b8357d

# Installing the TLT packages.
WORKDIR /opt/nvidia
COPY dist/*.whl /opt/nvidia/wheels/
RUN cd wheels && ls ./*.whl|xargs -I'{}' python -m pip install '{}' \
    && rm *.whl

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

COPY tao-core tao-core
RUN cd tao-core && bash release/python/build_wheel.sh && \
    find dist/ -name "nvidia_tao_core*.whl" -type f | xargs -n 1 pip install && \
    rm -rf *

# Copying tlt-converter and building it in the docker.
# ENV TRT_LIB_PATH="/usr/lib/x86_64-linux-gnu"
# ENV TRT_INCLUDE_PATH="/usr/include/x86_64-linux-gnu"
# COPY tao-converter /opt/nvidia/tools
# WORKDIR /opt/nvidia/tools
# RUN make \
#     && rm -rf Makefile *.cc *.h *.o
# RUN ln -s tao-converter converter

ENV NVIDIA_PRODUCT_NAME "TAO Toolkit"
ENV TAO_TOOLKIT_VERSION="5.2.0"
ENV NVIDIA_TAO_TOOLKIT_VERSION="${TAO_TOOLKIT_VERSION}-PyTorch"
ENV CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Defining the telemetry URL.
ENV TAO_TELEMETRY_SERVER="https://api.tao.ngc.nvidia.com"

# Microservices entrypoint
ENV FLASK_APP=/usr/local/lib/python3.12/dist-packages/nvidia_tao_pytorch/api/app.py 

# Include NGC CLI as an entrypoint.
CMD [ "/bin/bash" ]
