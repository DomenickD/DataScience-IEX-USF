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


# datascience-notebook includes
# NumPy: For numerical computations and array operations.
# Pandas: For data manipulation and analysis.
# Matplotlib: For plotting and visualization.
# SciPy: For scientific computing, including optimization, linear algebra, integration, interpolation, special functions, FFT, signal and image processing.
# Scikit-learn: For machine learning and data mining.
# Seaborn: For statistical data visualization based on matplotlib.
# Statsmodels: For statistical modeling and econometrics.
# Sympy: For symbolic mathematics.

# # Use an official Jupyter Notebook base image
# FROM jupyter/base-notebook:latest
# # Defines the base image for your Docker container. This line pulls the latest official Jupyter Notebook image, which has all the necessary components preinstalled.

# # Copy the notebook files into the container
# COPY . /home/jovyan/work
# # Copies all files and folders from your current directory (on the host) into the "/home/jovyan/work" directory inside the container. 
# # This is where your Jupyter Notebooks will reside.

# # Install any additional dependencies
# # For example:
# RUN pip install pandas numpy matplotlib seaborn scikit-learn torch 
# # A commented-out section to remind you to install dependencies. 
# # If your notebooks need libraries like pandas or matplotlib, uncomment the RUN line and include the packages.

# WORKDIR /app

# # Expose port 8888 for Jupyter Notebook
# EXPOSE 8888
# # Tells Docker that the container listens on port 8888 (the default Jupyter Notebook port). 
# # This is needed for accessing the notebook server from outside the container.

# # Set notebook configuration (optional)
# # For example:
# ENV JUPYTER_TOKEN=mysecret
# # demonstrating how to set an environment variable. This specific example sets a Jupyter token for enhanced security. Uncomment and change the value if you want to use a token.

# # Start Jupyter Notebook
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
# The command to execute when the container starts. It starts the Jupyter Notebook server with these options:
# --ip=0.0.0.0: Listen on all network interfaces.
# --no-browser: Prevent a browser from automatically opening when the server starts.
# --allow-root: Allow running Jupyter Notebook as root (often necessary within containers).

# # docker build -t housing_project . 
# #                 -Container name

# # docker run -p 8888:8888 housing_project
# #                         -Conttainer name

# # template:
# # docker build -t my-app . 

# # docker run -p 8080:80 --name my-app-container my-app 


