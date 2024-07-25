# Use a lightweight Debian-based image
FROM python:3.12-slim

# Install SQLite
RUN apt-get update && apt-get install -y sqlite3

# Set the working directory
WORKDIR /app

# Copy the application files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run create_db.py and keep the container running
CMD ["sh", "-c", "python3 /app/create_db.py && tail -f /dev/null"]
