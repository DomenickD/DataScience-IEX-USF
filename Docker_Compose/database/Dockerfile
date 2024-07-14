# Use a lightweight base image
FROM alpine:latest

# Install SQLite
RUN apk add --no-cache sqlite

# Set working directory
WORKDIR /app

# Copy your SQLite database file to the container
COPY ./titanic_data.db /app/titanic_data.db

# Command to keep the container running (adjust as needed)
CMD ["tail", "-f", "/dev/null"]
