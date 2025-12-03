# Start from a Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your pipeline scripts into the container
# This makes them available at /app/<script_name>.py
COPY fetch_data.py .
COPY preprocess_features.py .
COPY train_model.py .
COPY evaluate_model.py .

# Define the entrypoint as the Python interpreter for all components
ENTRYPOINT ["python"]
