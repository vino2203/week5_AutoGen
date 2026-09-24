"""HTTP client for the backend. All calls are synchronous; Streamlit reruns the script per interaction."""
import httpx


class ApiError(Exception):
    """A problem worth showing to the user as-is."""


class Api:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, timeout: float | None = None, **kwargs) -> dict:
        try:
            r = httpx.request(method, self.base_url + path, timeout=timeout or self.timeout, **kwargs)
        except httpx.ConnectError:
            raise ApiError(
                f"Cannot reach the backend at {self.base_url}. Start it with: "
                "`cd Backend && uvicorn agents.human_loop_agent:app --port 8710`"
            ) from None
        except httpx.TimeoutException:
            raise ApiError(f"The backend did not answer within {timeout or self.timeout:g} s.") from None
        except httpx.HTTPError as e:
            raise ApiError(f"Request failed: {e}") from None
        if r.status_code >= 400:
            raise ApiError(_explain(r))
        return r.json()

    def chat(self, **body) -> dict:
        return self._request("POST", "/chat", json=body)

    def status(self) -> dict:
        return self._request("GET", "/status", timeout=3)


def _explain(r: httpx.Response) -> str:
    try:
        detail = r.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, list):  # FastAPI validation errors
        return "Invalid input: " + "; ".join(
            f"{'.'.join(str(p) for p in d.get('loc', [])[1:])}: {d.get('msg')}" for d in detail
        )
    return str(detail or f"Backend returned HTTP {r.status_code}")
