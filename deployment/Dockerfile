# Use an official Python runtime
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Copy all contents into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
