"""HTTP client shared by the Finlex downloader and the model client."""

from __future__ import annotations

import httpx

from .config import USER_AGENT


def client(timeout: float = 60.0) -> httpx.Client:
    # local_address forces IPv4: some networks advertise IPv6 without routing it, and requests then hang.
    transport = httpx.HTTPTransport(local_address="0.0.0.0", retries=3)
    return httpx.Client(transport=transport, timeout=timeout, headers={"User-Agent": USER_AGENT}, follow_redirects=True)
