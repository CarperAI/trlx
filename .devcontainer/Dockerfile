# Use Nvidia Ubuntu 20 base (includes CUDA if a supported GPU is present)
# https://hub.docker.com/r/nvidia/cuda
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04@sha256:55211df43bf393d3393559d5ab53283d4ebc3943d802b04546a24f3345825bd9

ARG USERNAME
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
# https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && usermod -a -G video user \
  && apt-get update \
  && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME

# Install dependencies
RUN sudo apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
  build-essential \
  python3.9 \
  python3.9-dev \
  python3.9-distutils \
  python3.9-venv \
  curl \
  git

# Install pip (we need the latest version not the standard Ubuntu version, to
# support modern wheels)
RUN sudo curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.9 get-pip.py

# Set python aliases
RUN sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# User the new user
USER $USERNAME

# Install python dev dependencies
RUN pip install \
  autopep8 \
  jedi \
  mypy \
  pytest \
  toml \
  yapf
