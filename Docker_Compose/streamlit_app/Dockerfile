FROM python:3.11-slim
WORKDIR /app
COPY . /app
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Make port 8501 available to the world outside this container
EXPOSE 8501
# Run app.py when the container launches
CMD ["streamlit", "run", "app.py", "--server.port=8501"]