FROM python:3.13.1-slim

RUN apt update && apt upgrade -y && apt install -y curl
RUN pip install mlflow[auth] psycopg2-binary boto3 cryptography pymysql

EXPOSE 5000