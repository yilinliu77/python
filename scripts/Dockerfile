FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

#ENV http_proxy http://172.31.178.234:7890
#ENV https_proxy http://172.31.178.234:7890
# RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

RUN apt update && apt install openssh-server sudo -y

RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

RUN echo 'root:root' | chpasswd

RUN service ssh start

EXPOSE 22

CMD ["/usr/sbin/sshd","-D"]
