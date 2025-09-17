
FROM python:3.10-slim

# set workdir
WORKDIR /app
# copy files
COPY . /app

# install python deps
RUN apt update
RUN apt install git
RUN sh setup.sh

EXPOSE 8000
CMD ["python", "app.py"]
