import os
from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Get the database URL from the environment variable (set in docker-compose.yml)
db_url = os.environ.get('DATABASE_URL')  

# Ensure the database URL is provided
assert db_url is not None, "DATABASE_URL environment variable not set"

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable to reduce overhead

db = SQLAlchemy(app)

# Example Model (Define your database structure)
class Item(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)

# Create the database tables (if they don't exist)
with app.app_context():
    db.create_all()

# Example API Route
@app.route('/items', methods=['GET'])
def get_items():
    items = Item.query.all()
    return jsonify([{'id': item.id, 'name': item.name} for item in items])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
