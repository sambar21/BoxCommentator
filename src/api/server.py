"""
Python FastAPI service: entry point.
Run with: uvicorn src.api.server:app --host 0.0.0.0 --port 8000
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(title="Box.IO Commentary Service", version="1.0.0")
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
