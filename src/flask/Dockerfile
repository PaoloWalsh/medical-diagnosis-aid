# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY app.py .
# Expose the port on which the Flask app will run
EXPOSE 5000

# Run the Flask application when the container starts
# The debug flag is set to False for production deployment
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
