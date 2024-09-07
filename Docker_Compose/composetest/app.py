"""Flask walkthrough for docker compose"""

import time

import redis
from flask import Flask

app = Flask(__name__)
cache = redis.Redis(host="redis", port=6379)


def get_hit_count():
    """Counts the times it is pinged"""
    retries = 5
    while True:
        try:
            return cache.incr("hits")
        except redis.exceptions.ConnectionError as exc:
            if retries == 0:
                raise exc
            retries -= 1
            time.sleep(0.5)


@app.route("/")
def hello():
    """Hello World style for Flask walkthrough"""
    count = get_hit_count()
    return f"Hello Docker! I have been seen {count} times.\n"
