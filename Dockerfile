# Load basic Docker image:
FROM python:3.11.4

# Create a "work folder"
WORKDIR /app

# Copy project into this folder
COPY . /app

# RUN:
# Installation of dependencies(setup.py) and local package
# Installation of pytest
RUN pip install . \
    && pip install pytest~=7.2.0
