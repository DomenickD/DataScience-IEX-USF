# Base Image: Choose a suitable one for your needs
FROM jupyter/datascience-notebook:latest

# Working directory within the container - jovyan is a default directory for jupyter notebooks named after a maintainer of the technology
WORKDIR /home/jovyan/work 

# Copy all required files from your project directory
COPY . /home/jovyan/work

# Expose the Jupyter Notebook server to the host
EXPOSE 8888

CMD ["start-notebook.sh", "--NotebookApp.token=''"]


# docker build -t housing_project . 
#                 -Container name

# docker run -p 8888:8888 housing_project
#                         -Conttainer name
