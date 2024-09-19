from fastapi import APIRouter, Request
from .models import templates

app_routes = APIRouter()

@app_routes.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "message": "Hello, Impact Explorer!"})
