from flask import Flask

def create_app():
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config['SECRET_KEY'] = 'your-secret-key-here'

    # Register routes
    from . import routes
    app.register_blueprint(routes.bp)

    return app
