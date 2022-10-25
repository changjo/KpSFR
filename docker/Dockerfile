FROM nvcr.io/nvidia/pytorch:21.04-py3

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y tmux \
    && rm -rf /var/lib/apt/lists/*

COPY . /root/code
RUN cd /root/code && conda env update -f environment_new.yml
RUN pip install -U scikit-image==0.19.3

WORKDIR /root/code
