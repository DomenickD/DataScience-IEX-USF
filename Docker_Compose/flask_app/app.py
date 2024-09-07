"""The flask Backend and logic"""

import sqlite3
from flask import Flask, request, jsonify

app = Flask(__name__)


def query_database(query_input):
    """Database query steps"""
    conn = sqlite3.connect("/app/titanic_data.db")  # Adjust path if needed
    cursor = conn.cursor()
    cursor.execute(query_input)
    columns = [description[0] for description in cursor.description]  # Get column names
    data = cursor.fetchall()
    conn.close()
    return {"columns": columns, "data": data}


@app.route("/query", methods=["POST"])
def query():
    """Query the database"""
    try:
        query_handle = request.json.get("query")
        if not query_handle:
            return jsonify({"error": "No query provided"}), 400
        data = query_database(query_handle)
        return jsonify(data)
    except sqlite3.DatabaseError as db_error:
        return jsonify({"error": f"Database error: {str(db_error)}"}), 500
    except ValueError as val_error:
        return jsonify({"error": f"Value error: {str(val_error)}"}), 400
    except KeyError as e:
        return jsonify({"error": f"KeyError: {str(e)}"}), 400  # Bad Request
    except IOError as e:
        return jsonify({"error": f"IOError: {str(e)}"}), 500  # Internal Server Error


if __name__ == "__main__":
    app.run(host="0.0.0.0")
