# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
# Copy requirements.txt first to leverage Docker's cache
COPY flask/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
# This includes app.py and all your pickled model files
COPY flask/app.py .
# Assuming your models are named model1.pkl, model2.pkl, etc.
COPY models/*.pkl .
COPY models/*.csv .
# Expose the port on which the Flask app will run
EXPOSE 5000

# Run the Flask application when the container starts
# The debug flag is set to False for production deployment
CMD ["python", "app.py"]
