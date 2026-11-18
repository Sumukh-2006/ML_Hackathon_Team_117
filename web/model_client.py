"""Every call from the website to the model server goes through here.

The website never imports the matchers; it only speaks the HTTP API in
docs/api.md. MODEL_API overrides the server address.
"""

import hashlib
import os

import requests

API: str = os.environ.get("MODEL_API", "http://127.0.0.1:8100/v1")
http = requests.Session()   # keep-alive: one TCP connection, reused per fix


class ModelServerDown(Exception):
    pass


def map_id(nodes: bytes, arcs: bytes) -> str:
    inner = hashlib.sha256(nodes).digest() + hashlib.sha256(arcs).digest()
    return hashlib.sha256(inner).hexdigest()[:16]


def ensure_map(nodes: bytes, arcs: bytes) -> requests.Response | str:
    """The map's id, uploading it only if the server lacks it; the
    server's error response if it rejects the map."""
    mid = map_id(nodes, arcs)
    if call("HEAD", f"/maps/{mid}").status_code == 404:
        response = call("POST", "/maps", files={"nodes": nodes, "arcs": arcs})
        if response.status_code >= 400:
            return response
    return mid


def call(method: str, path: str, wait: float = 0, **kwargs) -> requests.Response:
    """One request to the model server. `wait` is the long-poll time the
    server may hold the request, so the timeout allows for it."""
    try:
        return http.request(method, API + path, timeout=wait + 15, **kwargs)
    except requests.ConnectionError as error:
        raise ModelServerDown(str(error))


# The named calls of docs/api.md §5.2, for code that is not a plain proxy.

def open_session(map_id: str, models: list[str], truth=None) -> dict:
    options = {"geometry": True} | ({"truth": truth} if truth else {})
    return call("POST", "/sessions", json={"map_id": map_id, "models": models,
                                           "options": options}).json()


def push(sid: str, seq: int, fixes: list) -> dict:
    return call("POST", f"/sessions/{sid}/fixes",
                json={"seq": seq, "fixes": fixes}).json()


def finish(sid: str, wait: float = 25) -> dict:
    return call("POST", f"/sessions/{sid}/finish", wait=wait,
                params={"wait": wait}).json()


def state(sid: str, wait: float = 0) -> dict:
    return call("GET", f"/sessions/{sid}", wait=wait, params={"wait": wait}).json()
