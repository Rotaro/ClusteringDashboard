FROM python:3.9-slim-buster

RUN apt-get update \
    && rm -rf /var/lib/apt/lists/*

ARG APP_DIR="/ClusteringDashboard"
WORKDIR ${APP_DIR}
COPY . .

RUN pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3"]
CMD ["dashboard/app.py"]
