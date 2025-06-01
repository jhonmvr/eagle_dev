FROM tensorflow/tensorflow:2.12.0

WORKDIR /app

# Instalar dependencias del sistema (incluye Java para pyarrow con HDFS)
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk && \
    apt-get clean

# Configurar JAVA_HOME para que pyarrow encuentre libjvm.so
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Copiar tu código y requerimientos
COPY . /app

# Instalar dependencias de Python
RUN pip install --upgrade pip && pip install -r requirements.txt
