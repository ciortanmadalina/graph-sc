FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
RUN apt update \
    && apt install -y \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*
RUN pip install setuptools==45.0.0 \
    jupyterlab==2.1.4 \
    notebook==6.0.3 \
    sklearn \
    lmdb \
    attrdict \
    h5py \
    scipy==1.6.0 \
    ipywidgets==7.5.1 \
    keras==2.3.1 \
    tensorflow-gpu==1.15.0 \
    tensorboard==1.15.0 \
    tensorboardX \
    scanpy==1.5.1 \
    jgraph \
    louvain \
    openpyxl \
    pandas==1.2.1 \
    dgl-cu101==0.5.3 \
    scikit_learn==0.22.2.post1 \
    xlrd==1.2.0 \
    leidenalg

# Jupyter notebook configuration
RUN pip install yapf==0.30.0 
RUN pip install jupyter_contrib_nbextensions==0.5.1
RUN pip install jupyter_highlight_selected_word==0.2.0
    
RUN apt-get update
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev


RUN jupyter contrib nbextension install --user
RUN jupyter nbextension install https://github.com/jfbercher/code_prettify/archive/master.zip --user
RUN jupyter nbextension enable code_prettify-master/code_prettify
RUN jupyter nbextension install --py jupyter_highlight_selected_word
RUN jupyter nbextension enable highlight_selected_word/main

EXPOSE 8080 8888 6006

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]
