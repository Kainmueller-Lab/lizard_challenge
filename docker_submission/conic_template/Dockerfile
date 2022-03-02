# ! USER SPECIFIC
# <<<<<<<<<<<<<<<<<<<<<<<<<
# Edit the base image here, e.g., to use 
# TENSORFLOW (https://hub.docker.com/r/tensorflow/tensorflow/) 
# or a different PYTORCH (https://hub.docker.com/r/pytorch/pytorch/) base image

# alternative 1.9.0-cuda10.2-cudnn7-runtime
# 1.9.1-cuda11.1-cudnn8-runtime
# nvcr.io/nvidia/pytorch:22.02-py3
FROM nvcr.io/nvidia/pytorch:21.10-py3
# FROM conic-inference
ENV DEBIAN_FRONTEND noninteractive
# For running opencv-python
RUN apt-get update &&  apt-get install gcc ffmpeg libsm6 libxext6 openslide-tools -y

# >>>>>>>>>>>>>>>>>>>>>>>>>

# ! DO NOT MODIFY
# <<<<<<<<<<<<<<<<<<<<<<<<<
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output
USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip
# >>>>>>>>>>>>>>>>>>>>>>>>>

# ! DO NOT MODIFY - ORGANIZER SPECIFIC
# <<<<<<<<<<<<<<<<<<<<<<<<<
# Copy all required files such that they are available within the docker image (code, weights, ...)

COPY --chown=algorithm:algorithm process.py /opt/algorithm/

# >>>>>>>>>>>>>>>>>>>>>>>>>

# ! USER SPECIFIC
# <<<<<<<<<<<<<<<<<<<<<<<<<
# Copy all required files such that they are available within the docker image (code, weights, ...)
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm data/ /opt/algorithm/data/
COPY --chown=algorithm:algorithm source/ /opt/algorithm/source/
COPY --chown=algorithm:algorithm nvidia_tensorflow-1.15.5+nv21.10-cp38-cp38-linux_x86_64.whl /opt/algorithm/


# Install required python packages via pip - you may adapt the requirements.txt to your needs
RUN python -m pip install --user -r requirements.txt
Run python -m pip install --user -e /opt/algorithm/source/PatchPerPix_private
RUN python -m pip install --user --upgrade numpy
RUN mkdir -p /home/algorithm/.local/lib/python3.8/site-packages/gunpowder/tensorflow2

# Entrypoint to your python code - executes process.py as a script
ENTRYPOINT python -m process $0 $@
# >>>>>>>>>>>>>>>>>>>>>>>>>

## ALGORITHM LABELS ##

# ! USER SPECIFIC
# <<<<<<<<<<<<<<<<<<<<<<<<<
# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=ConicTemplate
# >>>>>>>>>>>>>>>>>>>>>>>>>

# ! DO NOT MODIFY - ORGANIZER SPECIFIC
# <<<<<<<<<<<<<<<<<<<<<<<<<
# These labels are required and describe what kind of hardware your algorithm requires to run.
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.count=4
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.capabilities=()
LABEL nl.diagnijmegen.rse.algorithm.hardware.memory=16G
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.cuda_compute_capability=11.1
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.memory=12G
# >>>>>>>>>>>>>>>>>>>>>>>>>
