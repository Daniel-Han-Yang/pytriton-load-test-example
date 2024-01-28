FROM nvcr.io/nvidia/tritonserver:23.12-pyt-python-py3

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt