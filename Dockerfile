
FROM python:3.10-slim

# set workdir
WORKDIR /app
# copy files
COPY . /app

# install python deps
COPY requirements.txt /app/requirements.txt
RUN sh setup.sh

EXPOSE 8000
CMD ["python", "app.py"]
