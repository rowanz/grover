FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libjpeg-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=3.6 tqdm numpy pyyaml scipy ipython mkl mkl-include cython typing h5py pandas && \
     /opt/conda/bin/conda clean -ya && /opt/conda/bin/pip install tensorflow-gpu==1.13.1

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PATH /opt/conda/bin:/usr/local/nvidia/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

WORKDIR /grover

COPY requirements-gpu.txt .
RUN pip install -r requirements-gpu.txt

ENV PYTHONPATH /grover

ADD . .

CMD ["/bin/bash"]
