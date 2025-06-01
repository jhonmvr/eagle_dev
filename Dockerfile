FROM tensorflow/tensorflow:2.12.0

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip && pip install -r requirements.txt
