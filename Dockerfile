FROM python:3.10
COPY . /app
WORKER /app
RUN pip install -r requirements.txt
Expose $PORT
cmd gunicorn --worker=1 --bind 0.0.0.0:$PORT app:app