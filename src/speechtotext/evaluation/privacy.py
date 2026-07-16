from __future__ import annotations

import hashlib
import hmac


def protected_ref(ref_key: bytes, namespace: str, value: str) -> str:
    if len(ref_key) < 32:
        raise ValueError("ref_key debe tener al menos 32 bytes")
    if not namespace or any(
        char not in "abcdefghijklmnopqrstuvwxyz-" for char in namespace
    ):
        raise ValueError("namespace de referencia invalido")
    digest = hmac.new(
        ref_key,
        f"{namespace}\0{value}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{namespace}:{digest[:32]}"
