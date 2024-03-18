# Base Image: Choose a suitable one for your needs
FROM jupyter/datascience-notebook:latest

# Working directory within the container - jovyan is a default directory for jupyter notebooks named after a maintainer of the technology
WORKDIR /home/jovyan/work 

# Copy all required files from your project directory
COPY . /home/jovyan/work

# Expose the Jupyter Notebook server to the host
EXPOSE 8888

CMD ["start-notebook.sh", "--NotebookApp.token=''"]



# # Use an official Jupyter Notebook base image
# FROM jupyter/base-notebook:latest
# # Copy the notebook files into the container
# COPY . /home/jovyan/work
# # Install any additional dependencies
# # For example:
# # RUN pip install pandas matplotlib
# # Expose port 8888 for Jupyter Notebook
# EXPOSE 8888
# # Set notebook configuration (optional)
# # For example:
# # ENV JUPYTER_TOKEN=mysecret
# # Start Jupyter Notebook
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]


# docker build -t housing_project . 
#                 -Container name

# docker run -p 8888:8888 housing_project
#                         -Conttainer name

# docker build -t my-app . 

# docker run -d -p 8080:80 --name my-app-container my-app 


