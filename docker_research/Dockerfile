FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Add UTF-8 support
RUN apt-get update && \
    apt-get install -y --no-install-recommends locales && \
    locale-gen en_US.UTF-8
ENV PYTHONIOENCODING='UTF-8' LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8' PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive

# install apt dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        bash \
        build-essential \
        cmake \
        git \
        ssh \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopenblas-dev \
        libprotobuf-dev \
        libsnappy-dev \
        make \
        protobuf-compiler \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-flask \
        python3-tk \
        python3-opencv \
        wget \
        tmux \
        htop \
        vim  \
        unzip

# create user
ARG USER
ARG USER_ID
ARG GROUP_ID
RUN useradd -m ${USER} --uid=${USER_ID} 

# install pyhton libs
USER ${USER_ID}:${GROUP_ID}
ENV PATH="/home/${USER}/.local/bin:${PATH}"
WORKDIR /home/${USER}

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
COPY requirements.txt /home/${USER}/requirements.txt
RUN python3 -m pip install --upgrade pip
RUN pip3 install --user --no-warn-script-location -r requirements.txt

# copy and set tmux config file
COPY tmux.conf /home/${USER}/.tmux.conf
# RUN tmux source-file /home/${USER}/.tmux.conf

# changing writes for some of the root directories
USER root
# RUN chown -R ${USER_ID}:${GROUP_ID} /.ssh/
RUN mkdir /.cache && chown -R ${USER_ID}:${GROUP_ID} /.cache
RUN mkdir /.cache/torch && chown -R ${USER_ID}:${GROUP_ID} /.cache/torch
RUN mkdir /.cache/torch/hub && chown -R ${USER_ID}:${GROUP_ID} /.cache/torch/hub
RUN mkdir /.config && chown -R ${USER_ID}:${GROUP_ID} /.config
RUN mkdir /.config/matplotlib && chown -R ${USER_ID}:${GROUP_ID} /.config/matplotlib
RUN touch /.netrc && chown -R ${USER_ID}:${GROUP_ID} /.netrc
RUN mkdir /.local && chown -R ${USER_ID}:${GROUP_ID} /.local

RUN rm -rf /var/lib/apt/lists/*