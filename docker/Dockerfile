FROM tensorflow/tensorflow:2.9.0-gpu
MAINTAINER Jacob Czech <jacob@bioteam.net>

# Next 3 lines are needed due to https://github.com/NVIDIA/nvidia-docker/issues/1632
# RUN apt-key del 7fa2af80 && \
#     apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
#     apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub && \
#     apt-get update

RUN pip install prettytable art pillow==9.0.1 matplotlib mlflow
