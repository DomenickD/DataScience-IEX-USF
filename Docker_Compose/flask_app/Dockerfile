FROM python:3.11-slim 
#Might update file version
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
# Make port 5000 available to the world outside this container
EXPOSE 5000
# Run app.py when the container launches
CMD ["python", "app.py"]