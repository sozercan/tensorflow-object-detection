FROM tensorflow/tensorflow:1.3.0-devel

EXPOSE 6006

WORKDIR /tensorflow

# RUN git clone https://github.com/tensorflow/models
RUN git clone https://github.com/sozercan/models

WORKDIR /tensorflow/models/research

ENV PYTHONPATH "$PYTHONPATH:/tensorflow/models/research:/tensorflow/models/research/slim"
ENV PYTHON_HOME "$PYTHON_HOME:/tensorflow/models/research:/tensorflow/models/research/slim"

RUN curl -L -o /protoc-3.3.0-linux-x86_64.zip https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip \
    && unzip /protoc-3.3.0-linux-x86_64.zip \
    && rm /protoc-3.3.0-linux-x86_64.zip \
    && ./bin/protoc object_detection/protos/*.proto --python_out=. \
    && pip install Pillow lxml