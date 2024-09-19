import os
import uvicorn

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from anthropic_client import lifespan
from routes import app_routes

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

templates = Jinja2Templates(directory="templates")

app.include_router(app_routes)

if __name__ == "__main__":
    uvicorn.run("impact_explorer.main:app", host="0.0.0.0", port=8000, reload=True)
