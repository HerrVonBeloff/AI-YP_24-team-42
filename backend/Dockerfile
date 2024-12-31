FROM python:3.11
COPY requirements.txt app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
COPY . /app
EXPOSE 8501
CMD ["uvicorn","api:app","--host","backend","--port","8000","--reload"]

      
