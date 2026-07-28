"""A very small routing layer on top of ``http.server``.

Handlers take a :class:`Request` and return either a plain ``dict``/``list``
(sent as JSON) or a :class:`Response`. Raising :class:`HttpError` turns into a
JSON error with the given status.
"""

import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, unquote, urlparse


# Windows' registry regularly maps .js to text/plain, which browsers refuse to
# execute as a module -- so the types we serve are spelled out here.
CONTENT_TYPES = {
    ".css": "text/css",
    ".html": "text/html",
    ".js": "text/javascript",
    ".json": "application/json",
    ".map": "application/json",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


class HttpError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


class Response:
    def __init__(self, body=b"", status=200, content_type="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body)
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.body = body
        self.status = status
        self.content_type = content_type


class Request:
    def __init__(self, method, path, query, body):
        self.method = method
        self.path = path
        self.query = query  # {name: [values]}
        self._body = body

    def json(self):
        try:
            return json.loads(self._body or b"{}")
        except json.JSONDecodeError:
            return {}

    def get(self, name, default=""):
        return self.query.get(name, [default])[0]

    def get_int(self, name, default=0):
        try:
            return int(self.get(name, default))
        except (TypeError, ValueError):
            return default


class Router:
    def __init__(self):
        self._routes = {}
        self._static = []  # (url_prefix, directory)

    def add(self, method, path, handler):
        self._routes[(method.upper(), path)] = handler

    def get(self, path, handler):
        self.add("GET", path, handler)

    def post(self, path, handler):
        self.add("POST", path, handler)

    def mount_static(self, prefix, directory):
        self._static.append((prefix, directory))

    def dispatch(self, request):
        handler = self._routes.get((request.method, request.path))
        if handler is not None:
            result = handler(request)
            return result if isinstance(result, Response) else Response(result)
        if request.method == "GET":
            static = self._serve_static(request.path)
            if static is not None:
                return static
        raise HttpError(404, f"no route for {request.method} {request.path}")

    def _serve_static(self, path):
        for prefix, directory in self._static:
            if not path.startswith(prefix):
                continue
            target = (directory / unquote(path[len(prefix) :])).resolve()
            if directory.resolve() not in target.parents or not target.is_file():
                continue
            ctype = CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream")
            return Response(target.read_bytes(), content_type=ctype)
        return None


def make_handler(router):
    """Build the ``BaseHTTPRequestHandler`` subclass that feeds ``router``."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # the job log is the interesting output
            pass

        def _reply(self, response):
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(response.body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(response.body)

        def _handle(self, method):
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            request = Request(method, parsed.path, parse_qs(parsed.query), body)
            try:
                response = router.dispatch(request)
            except HttpError as exc:
                response = Response({"ok": False, "error": exc.message}, status=exc.status)
            except Exception as exc:  # noqa: BLE001 -- never take the server down
                response = Response({"ok": False, "error": str(exc)}, status=500)
            self._reply(response)

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

    return Handler
