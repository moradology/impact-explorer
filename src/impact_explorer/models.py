from typing import Dict

from fastapi.templating import Jinja2Templates

MAX_CONTEXT_LENGTH = 20
user_contexts: Dict[str, Dict] = {}
templates = Jinja2Templates(directory="templates")