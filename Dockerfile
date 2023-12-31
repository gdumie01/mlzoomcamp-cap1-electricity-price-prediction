# First install the python 3.8, the slim version uses less space
FROM python:3.11-slim

# Install pipenv library in Docker 
RUN pip install pipenv

# create a directory in Docker named app and we're using it as work directory 
WORKDIR /app                                                                

# Copy the Pip files into our working derectory 
COPY ["Pipfile", "Pipfile.lock", "./"]

# install the pipenv dependencies for the project and deploy them.
RUN pipenv install --deploy --system

# Copy any python files
COPY ["*.py", "./"]

# Copy the model and the scaler
COPY ["model/", "./model/"]

# We need to expose the 9698 port because we're not able to communicate with Docker outside it
EXPOSE 9698

# If we run the Docker image, we want our prediction service to be running
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9698", "predict:app"]