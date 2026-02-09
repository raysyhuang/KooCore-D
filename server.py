"""
Simple web server to serve the HTML dashboard on Heroku.
"""
import os
from flask import Flask, send_from_directory

app = Flask(__name__)

@app.route('/')
def index():
    """Serve the main dashboard HTML file."""
    return send_from_directory('.', 'dashboard.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve any other static files if needed."""
    return send_from_directory('.', path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
