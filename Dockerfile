# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install uv, a fast Python package installer
RUN pip install uv

# Copy only the project definition file first to leverage Docker's cache
COPY bayers_ts_forecasting/pyproject.toml .

# Install dependencies using uv.
RUN uv sync

# Copy the rest of the application code from the subdirectory
COPY bayers_ts_forecasting/ .

# Make port 8501 available
EXPOSE 8501

# Define environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT 8501
ENV STREAMLIT_SERVER_ADDRESS 0.0.0.0

# Run streamlit_app.py when the container launches
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]