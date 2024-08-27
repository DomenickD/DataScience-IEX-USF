import sqlite3
from flask import Flask, request, jsonify

app = Flask(__name__)


def query_database(query):
    conn = sqlite3.connect("/app/titanic_data.db")  # Adjust path if needed
    cursor = conn.cursor()
    cursor.execute(query)
    columns = [description[0] for description in cursor.description]  # Get column names
    data = cursor.fetchall()
    conn.close()
    return {"columns": columns, "data": data}


@app.route("/query", methods=["POST"])
def query():
    try:
        query = request.json.get("query")
        if not query:
            return jsonify({"error": "No query provided"}), 400
        data = query_database(query)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0")
