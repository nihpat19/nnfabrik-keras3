FROM tensorflow/tensorflow:2.15.0-gpu-jupyter

ADD . /src/nnfabrik
RUN apt-get remove -y python3-blinker
RUN pip3 install blinker==1.9.0
RUN pip3 install sphinx-rtd-theme
RUN pip3 install -e /src/nnfabrik
RUN pip3 install keras==3.0.0
RUN pip3 install torch==2.1.0
RUN pip3 install torchvision==0.16.0 
RUN pip3 install optuna 
RUN pip3 install datajoint
RUN pip3 install gitpython scipy 
ENV KERAS_BACKEND="tensorflow"
WORKDIR /notebooks

