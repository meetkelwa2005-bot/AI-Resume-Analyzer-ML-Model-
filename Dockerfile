FROM python:3.9-slim
# Set working directory
WORKDIR /app
# Install system dependencies required for some Python packages (like SpaCy)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Download Spacy model
RUN python -m spacy download en_core_web_sm
# Copy the rest of the application
COPY . .
# Run the training script to generate the models directory and .pkl files
RUN python train.py
# Expose the port the app runs on
EXPOSE 8000
# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
