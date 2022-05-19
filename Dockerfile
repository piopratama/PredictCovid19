#Create a ubuntu base image with python 3 installed.
FROM python:3.8-alpine

#Set the working directory
WORKDIR /usr/src/app

#copy all the files
COPY . .

EXPOSE 5000
ENV FLASK_APP=app.py
#Install the dependencies
RUN pip install -r requirements.txt

#Run the command
ENTRYPOINT [ "flask"]
CMD [ "run", "--host", "0.0.0.0" ]