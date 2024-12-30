from uvicorn import run
from app import app  # Replace 'my_app' with the name of your FastAPI app module

if __name__ == "__main__":
    run(app, host="0.0.0.0", port=5000)
