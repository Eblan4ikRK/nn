# syntax=docker/dockerfile:1

FROM python:3.10

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential=12.9 gcc=4:10.2.1-1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY requirements.txt requirements.txt

RUN pip3.10 install --no-cache-dir -r requirements.txt

COPY runtime runtime
COPY models models
COPY routers.yml routers.yml
COPY configs configs

EXPOSE 8000

CMD [ "/bin/bash", "-c", "source configs/production.sh && python runtime/runnable.py" ]
