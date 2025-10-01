# Use a lightweight Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the dependencies file
COPY requirements/requirements_gpu_environment.txt .

# Install dependencies and clean up
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache

# Keep the container running (useful for development)
CMD ["tail", "-f", "/dev/null"]