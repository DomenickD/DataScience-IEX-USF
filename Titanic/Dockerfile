# Base Image: Choose a suitable one for your needs
FROM jupyter/datascience-notebook:latest

# Working directory within the container - jovyan is a default directory for jupyter notebooks named after a maintainer of the technology
WORKDIR /home/jovyan/work 

# Copy all required files from your project directory
COPY . /home/jovyan/work

RUN pip install tensorflow pytorch

# Expose the Jupyter Notebook server to the host
EXPOSE 8888

CMD ["start-notebook.sh", "--NotebookApp.token=''"]




# # Base Image: Choose a suitable one for your needs
# FROM jupyter/datascience-notebook

# # Working directory within the container
# WORKDIR /app

# # Copy all required files from your project directory
# COPY gender_submission.csv README.md test.csv titanic.ipynb train.csv ./

# # Expose the Jupyter Notebook server to the host
# EXPOSE 8888

# CMD [ "jupyter", "notebook", "--ip", "0.0.0.0", "--no-browser", "--allow-root", "--notebook-dir=/app" ]

