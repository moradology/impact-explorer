from fastapi.testclient import TestClient

from impact_explorer.main import app

client = TestClient(app)


def test_read_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["message"] == "Hello, Impact Explorer!"


def test_static_files():
    """Test that static files are being served."""
    response = client.get("/static/style.css")
    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]


def test_invalid_route():
    """Test accessing an invalid route."""
    response = client.get("/invalid")
    assert response.status_code == 404


def test_with_anthropic_key():
    """Test FastAPI app with Anthropic API key provided."""
    import os

    os.environ["ANTHROPIC_API_KEY"] = "mock_key"

    response = client.get("/")
    assert response.status_code == 200

    del os.environ["ANTHROPIC_API_KEY"]
