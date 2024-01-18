FROM python:3.10.11-slim-buster

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Default Mlflow URI
ENV MLFLOWTRACKINGURI=sqlite:///mlruns.db

COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir -r requirements_deploy.txt

RUN mkdir -p /root/.kaggle/
COPY .kaggle/kaggle.json /root/.kaggle/
RUN chmod 600 /root/.kaggle/kaggle.json

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run"]
CMD ["apps/trainingapp.py"]
