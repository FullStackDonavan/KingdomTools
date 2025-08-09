FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy requirements and install everything
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

CMD ["python", "app.py"]
