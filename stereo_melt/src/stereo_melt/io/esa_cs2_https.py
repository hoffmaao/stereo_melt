# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""HTTPS client for the ESA CryoSat-2 Science Server (science-pds.cryosat.esa.int).

The site exposes a JSON API behind a PHP front-end:

- ``GET /?do=list&file=Cry0Sat2_data/<PRODUCT>/<YYYY>/<MM>&maxfiles=N&pos=M`` →
  paginated JSON listing.
- ``GET /?do=download&file=<path>`` → raw file bytes.

The endpoints are gated by a ``PHPSESSID`` cookie that the front-end binds
after an OAuth2 Authorization Code dance against the ESA EOIAM IdP at
``eoiam-idp.eo.esa.int``. This module performs that dance lazily on the
first authenticated call, using the EO Sign credentials in ``~/.netrc``
(machine ``science-pds.cryosat.esa.int``).

The full URL/auth contract is documented in
``memory/reference_esa_cs2_https.md``.

Replaces the FTPS path used in earlier versions of the prefetcher; HTTPS
keeps the connection pool warm across many ``?do=download`` calls and
parallelizes cleanly across worker threads.
"""

from __future__ import annotations

import json
import netrc
import re
import threading
from pathlib import Path
from typing import Iterator

import requests

HOST = "science-pds.cryosat.esa.int"
BASE = f"https://{HOST}"
IDP = "https://eoiam-idp.eo.esa.int"

# Hidden-form-field regexes. The login.do HTML wraps `value=...` to a new
# line after `name="..."`, so re.DOTALL is required.
_RE_SESSION_DATA_KEY = re.compile(
    r'name="sessionDataKey"[^>]*value=["\']([0-9a-f-]+)', re.DOTALL,
)
_RE_SESSION_DATA_KEY_CONSENT = re.compile(
    r'name="sessionDataKeyConsent"[^>]*value=["\']([0-9a-f-]+)', re.DOTALL,
)
_RE_CONSENT_FIELDS = re.compile(r'name="(consent_\d+)"')


class EsaCryoSatHttpsClient:
    r"""Thread-safe HTTPS client with lazy OAuth login.

    One instance can be shared across worker threads; the underlying
    :class:`requests.Session` connection pool is thread-safe for sequential
    ops, and the auth dance is gated by a lock so only one thread ever
    runs the 5-step login. After login the cookie jar is stable.
    """

    def __init__(
        self,
        *,
        netrc_machine: str = HOST,
        timeout: int = 60,
        user_agent: str = "stereo_melt/cs2-https",
    ) -> None:
        self._timeout = timeout
        self._netrc_machine = netrc_machine
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self._auth_lock = threading.Lock()
        self._authed = False

    # ---- public API ------------------------------------------------------

    def list_dir(
        self, remote_path: str, *, max_per_page: int = 500,
    ) -> list[dict]:
        r"""List one remote directory, paging through all results.

        ``remote_path`` is everything under ``Cry0Sat2_data/`` — e.g.
        ``"SIR_SIN_L2/2019/01"`` (the prefix is added internally). Returns
        a list of dicts with at least ``name``, ``path``, ``mtime``,
        ``size``, ``is_dir`` per the front-end JSON schema.
        """
        self._ensure_authed()
        full_path = f"Cry0Sat2_data/{remote_path.lstrip('/')}"
        out: list[dict] = []
        pos = 0
        # Page until an empty page. The server caps a page at its OWN limit
        # (observed 500) regardless of the requested ``maxfiles``, so we must
        # NOT key termination off ``len(page) == max_per_page`` — that exits
        # after one page whenever ``max_per_page`` exceeds the server cap,
        # silently truncating the listing. Advance by however many the server
        # actually returned and stop only when it returns nothing.
        for _ in range(10_000):  # guard: ~5M entries, far beyond any CS2 month
            r = self.session.get(
                f"{BASE}/",
                params={"do": "list", "file": full_path,
                        "maxfiles": max_per_page, "pos": pos},
                timeout=self._timeout,
            )
            r.raise_for_status()
            # Content-Type is text/html even though body is JSON.
            data = json.loads(r.text)
            if not data.get("success"):
                msg = (data.get("error") or {}).get("msg", "unknown error")
                raise RuntimeError(f"ESA list {full_path!r} failed: {msg}")
            results = data.get("results", [])
            if not results:
                break
            out.extend(results)
            pos += len(results)
        else:
            raise RuntimeError(
                f"ESA list {full_path!r} did not terminate after 10000 pages "
                f"(server may be ignoring 'pos'); got {len(out)} entries")
        return out

    def download(self, remote_path: str, local_path: Path) -> int:
        r"""Stream one file to ``local_path``. Returns bytes written.

        ``remote_path`` is the full ``path`` field returned by
        :meth:`list_dir` (already includes the ``Cry0Sat2_data/`` prefix).
        Writes to ``<local_path>.part`` and renames on success so partial
        downloads never look like cache hits.
        """
        self._ensure_authed()
        part = local_path.with_suffix(local_path.suffix + ".part")
        with self.session.get(
            f"{BASE}/",
            params={"do": "download", "file": remote_path},
            stream=True, timeout=self._timeout,
        ) as r:
            r.raise_for_status()
            n = 0
            with open(part, "wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        n += len(chunk)
                        f.write(chunk)
        part.rename(local_path)
        return n

    # ---- auth ------------------------------------------------------------

    def _ensure_authed(self) -> None:
        if self._authed:
            return
        with self._auth_lock:
            if self._authed:
                return
            self._authenticate()
            self._authed = True

    def _authenticate(self) -> None:
        r"""Run the 5-step OAuth2 Authorization Code dance.

        See ``reference_esa_cs2_https.md`` for the protocol. This bakes
        in a few quirks: the IdP's ``commonauth`` redirect chain must be
        walked manually (curl-style ``-L`` re-POSTs the form Content-Type
        and triggers HTTP 415), and the consent page must be approved
        with all mandatory ``consent_N`` checkboxes set.
        """
        auth = netrc.netrc().authenticators(self._netrc_machine)
        if auth is None:
            raise RuntimeError(
                f"No ~/.netrc entry for {self._netrc_machine}. Add a line:\n"
                f"  machine {self._netrc_machine} login <eo-sign-email> password <pw>"
            )
        user, _, password = auth

        # Step 1: GET ?do=login → follow to IdP login form.
        r = self.session.get(
            f"{BASE}/", params={"do": "login"},
            allow_redirects=True, timeout=self._timeout,
        )
        r.raise_for_status()
        m = _RE_SESSION_DATA_KEY.search(r.text)
        if not m:
            raise RuntimeError("login form: sessionDataKey not found")
        session_data_key = m.group(1)

        # Step 2: POST credentials. Don't auto-follow — we'd re-POST the
        # form to /oauth2/authorize and get HTTP 415.
        r = self.session.post(
            f"{IDP}/commonauth",
            data={"username": user, "password": password,
                  "sessionDataKey": session_data_key},
            allow_redirects=False, timeout=self._timeout,
        )
        if r.status_code != 302 or "Location" not in r.headers:
            raise RuntimeError(
                f"commonauth POST failed: {r.status_code} (check credentials)"
            )

        # Step 3: GET the internal /oauth2/authorize redirect. This either
        # routes us to the consent page (first-time user, or consent expired)
        # or — if the user previously chose "approveAlways" — straight to
        # the science-pds callback with ?code=. Branch accordingly.
        r = self.session.get(
            r.headers["Location"], allow_redirects=False, timeout=self._timeout,
        )
        if r.status_code != 302:
            raise RuntimeError(f"oauth2/authorize step: {r.status_code}")
        callback = r.headers["Location"]

        if "/oauth2_consent.do" in callback:
            # Steps 4 & 5: fetch consent page, POST approval to get the code.
            r = self.session.get(
                callback, allow_redirects=False, timeout=self._timeout,
            )
            r.raise_for_status()
            m = _RE_SESSION_DATA_KEY_CONSENT.search(r.text)
            if not m:
                raise RuntimeError("consent page: sessionDataKeyConsent not found")
            consent_body = {
                "consent": "approveAlways",
                "sessionDataKeyConsent": m.group(1),
                "user_claims_consent": "true",
                "consent_select_all": "on",
            }
            for f in sorted(set(_RE_CONSENT_FIELDS.findall(r.text))):
                consent_body[f] = "on"
            r = self.session.post(
                f"{IDP}/oauth2/authorize", data=consent_body,
                allow_redirects=False, timeout=self._timeout,
            )
            if r.status_code != 302 or "code=" not in r.headers.get("Location", ""):
                raise RuntimeError(
                    f"consent POST failed: {r.status_code}; missing ?code= in callback"
                )
            callback = r.headers["Location"]

        if "code=" not in callback:
            raise RuntimeError(f"unexpected redirect after authorize: {callback}")

        # Final: GET callback → PHP exchanges code, binds to PHPSESSID.
        r = self.session.get(
            callback, allow_redirects=True, timeout=self._timeout,
        )
        r.raise_for_status()
        if "PHPSESSID" not in self.session.cookies.get_dict(domain=HOST):
            raise RuntimeError("auth dance completed but no PHPSESSID was set")


def iter_results(client: EsaCryoSatHttpsClient, remote_path: str) -> Iterator[dict]:
    """Convenience generator over a single directory listing."""
    yield from client.list_dir(remote_path)
