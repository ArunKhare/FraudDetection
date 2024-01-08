FROM python:3.10.11-slim-buster

ENV MLFLOW_TRACKING_URI=sqlite:///mlruns.db

WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
Run git clone https://github.com/ArunKhare/FraudDetection.git

# COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run"]
CMD ["trainingapp.py"]
