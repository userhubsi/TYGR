FROM continuumio/miniconda3:latest

WORKDIR /app

RUN apt-get update && apt-get install -y \
  build-essential \
  git \
  && rm -rf /var/lib/apt/lists/*

COPY environment.yml .

RUN conda env create -f environment.yml

ENV PATH="/opt/conda/envs/TYGR/bin:$PATH"

RUN pip install torch-scatter==2.0.6 torch-sparse==0.6.9 -f https://data.pyg.org/whl/torch-1.8.1+cpu.html

COPY . .

RUN chmod +x TYGR

ENTRYPOINT ["/bin/bash", "-c"]
