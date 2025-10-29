# Dockerfile

FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the app files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Run the app with environment variable support
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
