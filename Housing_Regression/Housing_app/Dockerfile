# Base image
FROM python:3.12

# Working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose the Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "Housing_App.py", "--server.port=8501", "--server.address=0.0.0.0"]

# docker build -t housing-app . 
# docker run -p 8501:8501 housing-app
