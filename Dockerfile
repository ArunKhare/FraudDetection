FROM python:3.10.11-slim-buster

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY . ./app

WORKDIR /app
RUN pip install --no-cache-dir -r requirements_deploy.txt

COPY .kaggle/kaggle.json /home/myuser/.kaggle/kaggle.json
RUN chmod 600 /home/myuser/.kaggle/kaggle.json

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "apps/trainingapp.py", "--server.port=8501"]