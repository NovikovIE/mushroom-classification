FROM nvcr.io/nvidia/tritonserver:24.07-py3

RUN pip install --no-cache-dir \
    torchvision \
    pillow \
    numpy

CMD ["tritonserver"]
