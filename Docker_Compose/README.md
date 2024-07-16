# Intro to Docker Compose

### Description:

This project demonstrates how to containerize a web application using Flask for the backend (with SQLite3 database) and Streamlit for the interactive frontend. Docker Compose simplifies development and deployment by managing multiple services (containers) seamlessly.

### Prerequisites:

- **Docker**: Make sure Docker is installed and running.
- **Docker Compose**: Install Docker Compose if it's not already included with your Docker installation.

## Setup

1. Clone this repository.
```bash
git clone https://your-repository-url.git
cd your-repository-name
```

2. Build and start the services using Docker Compose:

```bash
docker-compose up --build
```

This will:

- Build Docker images for the Flask app and Streamlit frontend.
- Create and start containers.
- Make the Streamlit frontend accessible at http://localhost:8501 by default.

3. Stop the conatiners

```bash
docker-compose down
```

---

### How it Works:

**Flask (Backend)**:

- Handles web requests and business logic.
- Interacts with the SQLite3 database for data storage.

**Streamlit (Frontend)**:

- Provides a user-friendly interactive interface.
- Communicates with the Flask backend to fetch and display data.

**Docker Compose**:

- Orchestrates the two services (Flask and Streamlit), ensuring they work together smoothly.
- Manages container dependencies (e.g., the Streamlit container depends on the Flask container being up).

---

## Demo

![address](Pictures\address.png)

This is the address to visit and see teh front end when the containers are loaded and up.

![main_page](Pictures\main_page.png)

This is the page that displays when you reached the frontend address in your browser.

![select_box](Pictures\select_box.png)

Select this box to test out the query box with a preloaded query. 

![results](Pictures\results.png)

See results below. 

![LIMIT5](Pictures\LIMIT5.png)

Add or change the query and hit submit to see the results in real time. 

### Enjoy!


![yaml](Pictures\yaml.png)

The Docker-Compose.yaml for reference. 


