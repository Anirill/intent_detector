from app.main import app
import gunicorn

if __name__ == "__main__":
    app.run()
