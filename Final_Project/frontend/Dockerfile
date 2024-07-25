FROM python:3.11-slim 

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "RA_Program.py", "--server.port=8501"]