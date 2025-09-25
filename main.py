import uvicorn
from dotenv import load_dotenv

from src.api.app import app

# Load environment variables from .env file
load_dotenv()


def main():
    """Run the FastAPI application."""
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )


if __name__ == "__main__":
    main()
