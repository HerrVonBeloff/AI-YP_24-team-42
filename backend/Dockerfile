FROM python:3.11
COPY requirements.txt app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
COPY . /app
EXPOSE 8000
CMD ["uvicorn","api:app","--host","0.0.0.0","--port","8000","--reload"]
