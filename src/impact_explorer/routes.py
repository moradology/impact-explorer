import json

from anthropic import AsyncAnthropic
from anthropic_client import get_anthropic_client
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import StreamingResponse
from models import templates, user_contexts
from utils import (
    get_user_session,
    prune_context,
)

app_routes = APIRouter()


@app_routes.get("/")
async def root(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "message": "Hello, Impact Explorer!"}
    )


@app_routes.post("/chat")
async def chat_to_anthropic(
    message: str = Form(...),
    session: str = Depends(get_user_session),
    client: AsyncAnthropic = Depends(get_anthropic_client),
):
    context = user_contexts[session]

    # Instead of injecting the reminder into the user's message, we'll use it as a system message
    system_message = f"{context['system_prompt']}"

    # Prepare the messages for the API call
    api_messages = []
    for msg in context["messages"]:
        if msg["role"] == "user":
            api_messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            api_messages.append({"role": "assistant", "content": msg["content"]})

    # Add the new user message
    api_messages.append({"role": "user", "content": message})

    async def event_generator():
        try:
            async with client.messages.stream(
                model="claude-3-sonnet-20240229",
                system=system_message,
                messages=api_messages,
                max_tokens=1000,
            ) as stream:
                full_response = ""
                async for chunk in stream:
                    if chunk.type == "content_block_delta":
                        delta_text = chunk.delta.text
                        full_response += delta_text
                        yield f"data: {json.dumps({'delta': delta_text})}\n\n"

                # Store the original message and AI's response in the context
                context["messages"].append({"role": "user", "content": message})
                context["messages"].append(
                    {"role": "assistant", "content": full_response}
                )
                context["messages"] = prune_context(context["messages"])

                yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            print(f"Error calling Anthropic API: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    response = StreamingResponse(event_generator(), media_type="text/event-stream")
    response.set_cookie(key="session_id", value=session)
    return response
