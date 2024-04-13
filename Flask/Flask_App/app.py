from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from models import db, User
from sklearn.externals import joblib  # This is for importing your pickle model

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Used to add additional security for session data

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Setup Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Home route
@app.route('/')
def home():
    return render_template('login.html')

# Dashboard route
@app.route('/dashboard')
@login_required
def dashboard():
    # Load your model and do something with it
    model = joblib.load('./model/your_model.pkl')
    # Example of using model
    # prediction = model.predict([data])
    return render_template('dashboard.html')

# Remaining routes (login, logout, register) will go here

with app.app_context():
    db.create_all()


if __name__ == '__main__':
    app.run(debug=True)
