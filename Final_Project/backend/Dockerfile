FROM python:3.11-slim 

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt

EXPOSE 5000

CMD ["python", "app.py"]