# Use a lightweight Python base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the Python file into the image
COPY HelloWorld.py .

# Define the command to execute when the container starts
CMD ["python", "HelloWorld.py"]

# # template:
# # docker build -t my-app . 

# # docker run -p 8080:80 --name my-app-container my-app 
