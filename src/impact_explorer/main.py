import os
import uvicorn

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .anthropic_client import lifespan
from .routes import app_routes

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

app.include_router(app_routes)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the FastAPI server with Anthropic API key")
    parser.add_argument("--api-key", required=True, help="Anthropic API key")
    args = parser.parse_args()

    os.environ["ANTHROPIC_API_KEY"] = args.api_key

    uvicorn.run("impact_explorer.main:app", host="0.0.0.0", port=8000, reload=True)
