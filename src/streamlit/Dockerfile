# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt for Streamlit
# Crea un requirements.txt separato per Streamlit se non lo hai già
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit application code into the container
COPY streamlit_app.py .

# Expose the port on which the Streamlit app will run (default is 8501)
EXPOSE 8501

# Command to run the Streamlit application when the container starts
# --server.port 8501 è ridondante qui perché è la porta di default, ma esplicita
# --server.enableCORS false e --server.enableXsrfProtection false sono spesso necessari
# per la comunicazione tra frontend e backend in Docker Compose senza problemi CORS complessi.
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]