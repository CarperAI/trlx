FROM python:3.8-slim

# pip install -r docs/requirements.txt
# sphinx-build -b html docs docs/build/html -j auto
# sphinx-build -b html -D nb_execution_mode=off docs docs/build/html -j auto

RUN python -m pip install --upgrade --no-cache-dir pip
ADD docs/requirements.txt /tmp/requirements.txt
RUN python -m pip install --exists-action=w --no-cache-dir -r /tmp/requirements.txt
RUN apt-get update && apt-get install make imagemagick -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir /build
WORKDIR /build/
ADD . .
RUN python -m pip install -e .
RUN cd docs && make html
ENTRYPOINT [ "python", "-m", "http.server", "8000" ]
