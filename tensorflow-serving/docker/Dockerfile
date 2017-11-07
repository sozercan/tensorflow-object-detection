FROM sozercan/tensorflow-serving-devel

ENV PYTHON_BIN_PATH="/usr/bin/python"
ENV PYTHON_LIB_PATH="/usr/local/lib/python2.7/dist-packages"
ENV CC_OPT_FLAGS="-march=native"
ENV TF_NEED_JEMALLOC=1
ENV TF_NEED_GCP=0
ENV TF_NEED_HDFS=0
ENV TF_ENABLE_XLA=0
ENV TF_NEED_OPENCL=0
ENV TF_NEED_CUDA=0
ENV TF_CUDA_VERSION=v8.0
ENV TF_NEED_S3=0
ENV TF_NEED_GDR=0
ENV TF_NEED_VERBS=0
ENV TF_NEED_MPI=0

COPY object_detection-0.1.tar.gz /tmp/

RUN pip install /tmp/object_detection-0.1.tar.gz \
    && pip install requests grpcio tensorflow-serving-api \
    && git clone --recurse-submodules https://github.com/tensorflow/serving \
    && cd /serving/tensorflow \
    && ./configure \
    && cd /serving \
    && bazel build -c opt //tensorflow_serving/model_servers:tensorflow_model_server

COPY serving_client.py /serving/

WORKDIR /serving