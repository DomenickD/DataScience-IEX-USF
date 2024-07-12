FROM python:3.11-slim 
#Might update file version
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD [ "flask", "run", "--host=0.0.0.0"] 
# Allow connections from anywhere 