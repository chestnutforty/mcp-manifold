import os
import signal

from fastapi import FastAPI, Response
from server import mcp

# Shutdown state tracking for graceful connection draining
_shutting_down = False


def _handle_shutdown(signum, frame):
    global _shutting_down
    _shutting_down = True


# Register signal handlers for graceful shutdown detection
signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT, _handle_shutdown)

# Optional authentication via environment variable
token = os.environ.get("MCP_AUTH_TOKEN")
if token:
    from fastmcp.server.auth import BearerAuthProvider
    mcp.auth = BearerAuthProvider(token=token)

# Create ASGI app with MCP mounted at /mcp
mcp_app = mcp.http_app(path="/mcp")
app = FastAPI(lifespan=mcp_app.lifespan)


@app.get("/health")
async def health_check():
    """Health endpoint for load balancer and Docker healthcheck.

    Returns 503 during shutdown to signal Traefik to stop routing new requests,
    while existing connections continue to be served.
    """
    if _shutting_down:
        return Response(content="shutting down", status_code=503, media_type="text/plain")
    return Response(content="ok", status_code=200, media_type="text/plain")


app.mount("/", mcp_app)
