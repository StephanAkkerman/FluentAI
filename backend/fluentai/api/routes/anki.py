import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

anki_router = APIRouter()


@anki_router.post("/api/anki")
async def anki_proxy(request: Request):
    """
    Proxy API endpoint for forwarding requests to the Anki server.

    This function receives a JSON request from the client, forwards it to the Anki
    server running on localhost, and returns the response back to the client.

    HACK: This uses the backend as a proxy for when the frontend is deployed in GH Pages

    Parameters
    ----------
    request : Request
        The incoming HTTP request object containing the JSON payload to be forwarded.

    Returns
    -------
    JSONResponse
        A JSON response containing the Anki server response or an error message if
        the request fails.
    """
    try:
        # Forward the incoming request body to the Anki server
        request_body = await request.json()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://127.0.0.1:8765",  # Assuming Anki is running on localhost with default port
                json=request_body,
            )

        # Return the JSON response from Anki server
        return JSONResponse(content=response.json(), status_code=response.status_code)

    except httpx.RequestError as e:
        return JSONResponse(
            content={"error": "Failed to connect to Anki server.", "details": str(e)},
            status_code=500,
        )
    except Exception as e:
        return JSONResponse(
            content={"error": "An unexpected error occurred.", "details": str(e)},
            status_code=500,
        )
