FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.13.1-gpu-py39-cu117-ubuntu20.04-sagemaker
WORKDIR /root
RUN git clone https://github.com/NASA-IMPACT/hls-foundation-os.git
RUN wget https://github.com/open-mmlab/mmcv/archive/refs/tags/v1.6.2.tar.gz
RUN tar xvzf v1.6.2.tar.gz
WORKDIR /root/hls-foundation-os
RUN pip install -e . 
RUN pip install -U openmim 
WORKDIR /root/mmcv-1.6.2
RUN MMCV_WITH_OPS=1 pip install -e . -v

