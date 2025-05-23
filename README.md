# tccModel

This repository contains utilities for downloading and processing satellite imagery. The main entry point is `getTiles.py` which downloads data from Google Earth Engine and prepares training tiles.

## Docker usage

A sample `Dockerfile` is provided for running the code with Python 3.10.14. Build the image and run `getTiles.py` as follows:

```bash
docker build -t tccmodel .
docker run --rm tccmodel --lat 0.0 --lon 0.0 --outputdir /tmp/out
```

Adjust the command line arguments as needed. The container installs requirements from `requirements.txt` and sets `getTiles.py` as the default entrypoint.

For deployment to Cloud Run, push the built image to your container registry and create a Cloud Run service using that image.
