# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy only necessary files
COPY requirements.txt .
COPY src/ ./src

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port on which the Streamlit app will run
EXPOSE 8501

# Set the entry point to run the Streamlit app
ENTRYPOINT ["streamlit", "run"]

# Specify the Streamlit app file
CMD ["src/app.py"]
