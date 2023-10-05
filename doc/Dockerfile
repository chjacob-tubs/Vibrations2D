# Load basic Docker image:
FROM continuumio/miniconda3

# Create a "work folder"
WORKDIR /app

# Copy project into this folder
COPY . /app

# Installation of conversion programs
RUN pip install jupyter \
    && apt-get update \
    && apt-get install -y pandoc

# Convert .ipynb to .rst
RUN jupyter nbconvert --to rst How_to_2DIR_spectra.ipynb

# Installation of local package and sphinx + add ons
RUN pip install . \
    && pip install sphinx~=5.3.0 \
    && pip install sphinx_rtd_theme~=0.4.3 \
    && pip install sphinx_mdinclude~=0.5.3

# Copy converted Notebook-rst's to doc/
RUN cp How_to_2DIR_spectra.rst doc/source/ \
    && cp -rf How_to_2DIR_spectra_files doc/source/

# change work dir
WORKDIR /app/doc/source

# change header in Notebook *.rst
RUN python change_Howto_lines.py
