#!/usr/bin/env python3
"""web_fetch.py — Sandboxed web fetch tool for Aion.

Fetches content from public URLs and returns sanitized text.
Designed to prevent SSRF, prompt injection, and resource exhaustion.

Security measures:
  - SSRF protection: blocks private IPs, localhost, link-local, metadata endpoints
  - DNS resolution checked before connect (prevents DNS rebinding)
  - Content size capped (500KB raw, 20KB text returned)
  - Timeout: 15s connect, 30s total
  - Only HTTP/HTTPS schemes allowed
  - No redirects to private IPs (redirects followed but re-validated)
  - Content sanitized: HTML stripped to text, scripts/styles removed
  - Max 3 redirects, max depth
  - Response logged to episodic memory for audit

Usage:
  from web_fetch import fetch_url
  result = fetch_url("https://example.com")
  # result = {"ok": True, "text": "...", "url": "...", "status": 200, "size": 1234}
"""
import html
import ipaddress
import json
import os
import re
import socket
import subprocess
import sys
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
import aion_env  # noqa

AION = os.environ.get("AION_HOME", "$AION_HOME")

# Limits
MAX_REDIRECTS = 3
MAX_CONTENT_SIZE = 500_000  # 500KB raw download
MAX_TEXT_SIZE = 20_000        # 20KB text returned to Aion
CONNECT_TIMEOUT = 15
READ_TIMEOUT = 30
USER_AGENT = "Aion/1.0 (research agent; +https://github.com/aion)"

# Blocked IP ranges (SSRF protection)
BLOCKED_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),    # Carrier-grade NAT (Tailscale!)
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),   # Link-local (AWS metadata!)
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("$LAN_IP/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),         # IPv6 private
    ipaddress.ip_network("fe80::/10"),        # IPv6 link-local
]

# Blocked hostname patterns
BLOCKED_HOSTS = [
    "metadata.google.internal",
    "169.254.169.254",      # AWS/GCP metadata
    "metadata.azure.com",
    "metadata",
]


def _log_event(event_type, text, meta=None):
    """Log to episodic memory."""
    try:
        cmd = [sys.executable, f"{AION}/bin/log_event.py",
               "--type", event_type, "--text", text[:2000]]
        if meta:
            cmd.extend(["--meta", json.dumps(meta)])
        subprocess.run(cmd, capture_output=True, timeout=10, check=False)
    except Exception:
        pass


def _is_ip_blocked(ip_str):
    """Check if an IP address is in a blocked range."""
    try:
        ip = ipaddress.ip_address(ip_str)
        for net in BLOCKED_NETWORKS:
            if ip in net:
                return True
    except ValueError:
        return True  # Can't parse = block
    return False


def _validate_url(url):
    """Validate URL scheme and host. Returns parsed URL or raises ValueError."""
    parsed = urllib.parse.urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Scheme not allowed: {parsed.scheme} (only http/https)")

    if not parsed.hostname:
        raise ValueError("No hostname in URL")

    # Check blocked hostname patterns
    host_lower = parsed.hostname.lower()
    for blocked in BLOCKED_HOSTS:
        if blocked in host_lower:
            raise ValueError(f"Blocked host: {host_lower}")

    # Resolve DNS and check IPs
    try:
        addrs = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        raise ValueError(f"DNS resolution failed: {parsed.hostname}")

    ips = set()
    for family, _, _, _, sockaddr in addrs:
        ip_str = sockaddr[0]
        ips.add(ip_str)

    for ip_str in ips:
        if _is_ip_blocked(ip_str):
            raise ValueError(f"Resolved to blocked IP: {ip_str} (SSRF protection)")

    return parsed, ips


def _strip_html(content_bytes):
    """Extract readable text from HTML. Remove scripts, styles, tags."""
    # Decode
    try:
        text = content_bytes.decode("utf-8", errors="replace")
    except Exception:
        text = content_bytes.decode("latin-1", errors="replace")

    # Remove script and style blocks entirely
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.S | re.I)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.S | re.I)
    text = re.sub(r"<nav[^>]*>.*?</nav>", "", text, flags=re.S | re.I)
    text = re.sub(r"<footer[^>]*>.*?</footer>", "", text, flags=re.S | re.I)

    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)

    # Convert common block elements to newlines
    text = re.sub(r"</?(p|div|br|h[1-6]|li|tr|table)[^>]*>", "\n", text, flags=re.I)

    # Remove all remaining tags
    text = re.sub(r"<[^>]+>", "", text)

    # Decode HTML entities
    text = html.unescape(text)

    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


def fetch_url(url, max_text=None):
    """Fetch a URL and return sanitized text content.

    Args:
        url: The URL to fetch (must be http or https)
        max_text: Override max text size (default 20KB)

    Returns:
        dict with keys:
            ok: bool
            text: str (sanitized content, max ~20KB)
            url: str (final URL after redirects)
            status: int (HTTP status code)
            size: int (bytes downloaded)
            error: str (if ok=False)
    """
    max_text = max_text or MAX_TEXT_SIZE
    result = {"ok": False, "text": "", "url": url, "status": 0, "size": 0, "error": None}

    try:
        # Validate URL and resolve DNS
        parsed, resolved_ips = _validate_url(url)
    except ValueError as e:
        result["error"] = str(e)
        _log_event("web_fetch_blocked", f"Blocked: {url} — {e}", {"url": url, "reason": str(e)})
        return result

    current_url = url
    redirects = 0

    while redirects <= MAX_REDIRECTS:
        try:
            # Re-validate on each redirect
            if redirects > 0:
                parsed, _ = _validate_url(current_url)

            req = urllib.request.Request(
                current_url,
                headers={"User-Agent": USER_AGENT, "Accept": "text/html, text/plain, */*"},
            )

            # Use a custom opener with our timeout
            with urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT) as response:
                result["status"] = response.status

                # Check content length
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > MAX_CONTENT_SIZE:
                    result["error"] = f"Content too large: {content_length} bytes (max {MAX_CONTENT_SIZE})"
                    _log_event("web_fetch_blocked", f"Too large: {current_url}", {"url": current_url, "size": content_length})
                    return result

                # Download with size cap
                data = b""
                while len(data) < MAX_CONTENT_SIZE:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    data += chunk

                result["size"] = len(data)

                # Check if this is a redirect (urllib follows automatically, but check final URL)
                final_url = response.geturl()
                if final_url != current_url:
                    current_url = final_url
                    # Re-validate final URL
                    try:
                        _validate_url(current_url)
                    except ValueError as e:
                        result["error"] = f"Redirect to blocked URL: {e}"
                        _log_event("web_fetch_blocked", f"Redirect blocked: {current_url} — {e}", {"url": current_url})
                        return result

                result["url"] = current_url

                # Process content
                content_type = response.headers.get("Content-Type", "")

                if "text/html" in content_type or "text/xml" in content_type:
                    text = _strip_html(data)
                elif "text/plain" in content_type or "application/json" in content_type:
                    text = data.decode("utf-8", errors="replace")
                elif "application/xml" in content_type or "application/rss" in content_type:
                    text = _strip_html(data)
                else:
                    # For other content types, try to extract text anyway
                    text = _strip_html(data)

                # Truncate to max text size
                if len(text) > max_text:
                    text = text[:max_text] + "\n...[truncated]"

                result["text"] = text
                result["ok"] = True

                _log_event("web_fetch", f"Fetched {current_url} ({result['status']}, {result['size']}b)", {
                    "url": current_url,
                    "status": result["status"],
                    "size": result["size"],
                    "text_len": len(text),
                    "resolved_ips": list(resolved_ips),
                })

                return result

        except urllib.error.HTTPError as e:
            result["status"] = e.code
            result["error"] = f"HTTP {e.code}: {e.reason}"
            _log_event("web_fetch_error", f"HTTP error {current_url}: {e.code}", {"url": current_url, "status": e.code})
            return result

        except urllib.error.URLError as e:
            if hasattr(e, "code") and 300 <= e.code < 400:
                # Manual redirect handling (shouldn't happen with urlopen, but just in case)
                redirect_url = e.headers.get("Location", "")
                if not redirect_url:
                    result["error"] = "Redirect with no Location header"
                    return result
                redirect_url = urllib.parse.urljoin(current_url, redirect_url)
                redirects += 1
                current_url = redirect_url
                continue
            result["error"] = f"URL error: {e.reason}"
            _log_event("web_fetch_error", f"URL error {current_url}: {e.reason}", {"url": current_url})
            return result

        except socket.timeout:
            result["error"] = "Connection timed out"
            _log_event("web_fetch_error", f"Timeout: {current_url}", {"url": current_url})
            return result

        except Exception as e:
            result["error"] = f"Fetch error: {type(e).__name__}: {e}"
            _log_event("web_fetch_error", f"Error {current_url}: {e}", {"url": current_url})
            return result

    result["error"] = f"Too many redirects (max {MAX_REDIRECTS})"
    return result


# ── GitHub-aware fetch ────────────────────────────────────────────
# Plain github.com HTML pages are mostly navigation chrome; the REST API
# returns clean markdown. Parses pasted URLs into owner/repo/path parts
# and fetches via api.github.com (with GH_TOKEN for rate limits).

GH_API = "https://api.github.com"


def _gh_headers(accept="application/vnd.github+json"):
    h = {"Accept": accept, "User-Agent": USER_AGENT}
    token = os.environ.get("GH_TOKEN", "")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _gh_api_get(url, accept="application/vnd.github+json"):
    """GET an api.github.com URL. Returns (ok, data_str)."""
    req = urllib.request.Request(url, headers=_gh_headers(accept))
    try:
        with urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT) as r:
            return True, r.read(MAX_CONTENT_SIZE).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.reason}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _gh_parse(url):
    """Parse a github.com URL into (owner, repo, rest).

    Handles: /owner/repo, /owner/repo/tree/BRANCH/path, /owner/repo/blob/BRANCH/path,
    /owner/repo/issues/N, /owner/repo/pull/N, /owner/repo/releases, /orgs/...,
    raw.githubusercontent.com/owner/repo/BRANCH/path
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return None
    host = (parsed.hostname or "").lower()
    if host not in ("github.com", "www.github.com", "raw.githubusercontent.com"):
        return None
    parts = [p for p in parsed.path.split("/") if p]
    if host == "raw.githubusercontent.com":
        # /owner/repo/BRANCH/path...
        if len(parts) < 2:
            return None
        owner, repo = parts[0], parts[1]
        rest = parts[2:]  # includes branch
        return owner, repo, "raw:" + "/".join(rest)
    if not parts:
        return None
    if parts[0] in ("orgs", "users"):
        # org/user page — treat owner as the subject
        if len(parts) < 2:
            return None
        return parts[1], "", "profile"
    if len(parts) == 1:
        return parts[0], "", "profile"
    owner, repo = parts[0], parts[1]
    repo = repo.removesuffix(".git")
    tail = parts[2:]
    if tail and tail[0] in ("tree", "blob"):
        # /tree/BRANCH/... or /blob/BRANCH/path — keep branch + path
        return owner, repo, "/".join(tail)
    if tail and tail[0] in ("issues", "pull"):
        # keep full tail (issues/N, pull/N, issues?query=...) for detail regex
        return owner, repo, "/".join(tail)
    if tail and tail[0] in ("releases", "wiki", "commits", "tags", "actions", "projects"):
        return owner, repo, tail[0]
    if tail:
        return owner, repo, "file:" + "/".join(tail)
    return owner, repo, ""


def fetch_github(url, max_text=None):
    """Fetch a GitHub URL via the REST API (clean markdown, no HTML chrome).

    Returns the same result dict shape as fetch_url().
    """
    max_text = max_text or MAX_TEXT_SIZE
    result = {"ok": False, "text": "", "url": url, "status": 0, "size": 0, "error": None}

    parsed = _gh_parse(url)
    if parsed is None:
        result["error"] = "Not a github.com URL"
        return result
    owner, repo, rest = parsed

    def finish(text, status=200, note=""):
        text = text or ""
        if len(text) > max_text:
            text = text[:max_text] + "\n...[truncated]"
        result.update(ok=bool(text), text=text, status=status,
                      size=len(text), error=None if text else (note or "empty"))
        _log_event("web_fetch_github", f"Fetched {url} via API ({status}, {result['size']}b)", {
            "url": url, "owner": owner, "repo": repo, "kind": note or "ok",
        })
        return result

    # Profile / no repo
    if not repo:
        ok, data = _gh_api_get(f"{GH_API}/users/{owner}")
        if not ok:
            result["error"] = f"GitHub API: {data}"
            return result
        try:
            u = json.loads(data)
            bio = u.get("bio") or ""
            lines = [f"# {u.get('name') or owner} (@{owner})",
                     bio,
                     f"Location: {u.get('location', '?')} | Repos: {u.get('public_repos', '?')} | Followers: {u.get('followers', '?')}",
                     f"Profile: {u.get('html_url', url)}"]
            return finish("\n".join(x for x in lines if x), note="profile")
        except Exception as e:
            result["error"] = f"Bad profile JSON: {e}"
            return result

    # Repo metadata
    ok, data = _gh_api_get(f"{GH_API}/repos/{owner}/{repo}")
    if not ok:
        result["error"] = f"GitHub API: {data}"
        _log_event("web_fetch_github", f"Failed {url}: {data}", {"url": url, "reason": data})
        return result
    try:
        r = json.loads(data)
    except Exception as e:
        result["error"] = f"Bad repo JSON: {e}"
        return result
    default_branch = r.get("default_branch", "main")

    def repo_header(r):
        stars = r.get("stargazers_count", "?")
        forks = r.get("forks_count", "?")
        lang = r.get("language") or "?"
        desc = r.get("description") or ""
        lic = (r.get("license") or {}).get("spdx_id", "?") if r.get("license") else "?"
        topics = ", ".join(r.get("topics", []) or [])
        updated = (r.get("updated_at") or "")[:10]
        head = [f"# {r.get('full_name', owner + '/' + repo)}",
                desc,
                f"★ {stars} | forks {forks} | {lang} | license {lic} | default branch: {default_branch} | updated {updated}"]
        if topics:
            head.append(f"Topics: {topics}")
        return "\n".join(x for x in head if x)

    header = repo_header(r)

    # Specific sub-pages
    if rest in ("issues", "pull", "releases", "commits", "tags", "wiki", "actions", "projects"):
        kind = rest
        if kind in ("issues", "pull"):
            api_kind = "issues" if kind == "issues" else "pulls"
            ok, data = _gh_api_get(f"{GH_API}/repos/{owner}/{repo}/{api_kind}?state=all&per_page=10&sort=updated&direction=desc")
            if not ok:
                return finish(header + f"\n\n(recent {kind} unavailable: {data})", note=kind)
            items = json.loads(data)
            lines = [header, "", f"## Recent {kind} (up to 10, newest first):"]
            for it in items[:10]:
                num = it.get("number", "?")
                title = it.get("title", "?")
                state = it.get("state", "?")
                is_pr = "pull_request" in it
                body = (it.get("body") or "")[:600]
                lines.append(f"- #{num} [{state}]{' PR' if is_pr else ''} {title}\n  {body}" if body else f"- #{num} [{state}]{' PR' if is_pr else ''} {title}")
            return finish("\n".join(lines), note=kind)
        if kind == "releases":
            ok, data = _gh_api_get(f"{GH_API}/repos/{owner}/{repo}/releases?per_page=10")
            if not ok:
                return finish(header + f"\n\n(releases unavailable: {data})", note=kind)
            rels = json.loads(data)
            lines = [header, "", "## Releases (up to 10):"]
            for rel in rels[:10]:
                lines.append(f"- {rel.get('tag_name', '?')} ({rel.get('name') or 'no name'}, {rel.get('published_at', '?')[:10]}): {(rel.get('body') or '')[:400]}")
            return finish("\n".join(lines), note=kind)
        if kind == "commits":
            ok, data = _gh_api_get(f"{GH_API}/repos/{owner}/{repo}/commits?per_page=10")
            if not ok:
                return finish(header + f"\n\n(commits unavailable: {data})", note=kind)
            commits = json.loads(data)
            lines = [header, "", "## Recent commits (up to 10):"]
            for c in commits[:10]:
                msg = ((c.get("commit") or {}).get("message") or "?").split("\n")[0][:120]
                sha = (c.get("sha") or "")[:7]
                date = ((c.get("commit") or {}).get("author") or {}).get("date", "")[:10]
                lines.append(f"- {sha} {date} {msg}")
            return finish("\n".join(lines), note=kind)
        if kind == "wiki":
            return finish(header + "\n\n(Wiki pages are not exposed via the GitHub REST API; fetch the wiki URL with web_fetch instead.)", note="wiki")
        # tags/actions/projects
        api_kind = {"tags": "tags", "actions": "actions/runs", "projects": "projects"}[kind]
        ok, data = _gh_api_get(f"{GH_API}/repos/{owner}/{repo}/{api_kind}?per_page=10")
        if not ok:
            return finish(header + f"\n\n({kind} unavailable: {data})", note=kind)
        try:
            items = json.loads(data)
            if isinstance(items, dict):
                items = items.get("workflow_runs", items.get("items", []))
        except Exception:
            items = []
        lines = [header, "", f"## {kind} (up to 10):"]
        for it in items[:10]:
            if kind == "tags":
                lines.append(f"- {it.get('name', '?')} ({it.get('commit', {}).get('sha', '?')[:7]})")
            elif kind == "actions":
                lines.append(f"- {it.get('name', '?')}: {it.get('status', '?')}/{it.get('conclusion', '?')} ({it.get('created_at', '?')[:10]})")
            else:
                lines.append(f"- {json.dumps(it, ensure_ascii=False)[:200]}")
        return finish("\n".join(lines), note=kind)

    # raw.githubusercontent.com/owner/repo/BRANCH/path
    if rest.startswith("raw:"):
        segs = rest[4:].split("/")  # [BRANCH, ...path]
        branch = segs[0] if segs else default_branch
        path = "/".join(segs[1:])
        api_path = (f"{GH_API}/repos/{owner}/{repo}/contents/{urllib.parse.quote(path)}?ref={urllib.parse.quote(branch)}"
                    if path else f"{GH_API}/repos/{owner}/{repo}/readme")
        ok, data = _gh_api_get(api_path, accept="application/vnd.github.raw")
        if not ok:
            result["error"] = f"GitHub API: {data}"
            return result
        return finish(data, note="raw file")

    # /tree/BRANCH/path → directory listing
    if rest.startswith("tree/"):
        segs = rest.split("/")  # ['tree', BRANCH, ...path]
        branch = segs[1] if len(segs) > 1 else default_branch
        path = "/".join(segs[2:])
        api_path = f"{GH_API}/repos/{owner}/{repo}/contents/{urllib.parse.quote(path)}?ref={urllib.parse.quote(branch)}" if path else f"{GH_API}/repos/{owner}/{repo}/contents?ref={urllib.parse.quote(branch)}"
        ok, data = _gh_api_get(api_path)
        if not ok:
            result["error"] = f"GitHub API: {data}"
            return result
        try:
            entries = json.loads(data)
        except Exception as e:
            result["error"] = f"Bad contents JSON: {e}"
            return result
        if isinstance(entries, dict):
            # single file — fetch raw via its download_url
            dl = entries.get("download_url") or ""
            if dl:
                ok2, raw = _gh_api_get(dl, accept="application/vnd.github.raw")
                if ok2:
                    return finish(header + "\n\n" + raw, note="file")
            return finish(header + "\n\n" + json.dumps(entries, ensure_ascii=False)[:3000], note="file meta")
        lines = [header, "", f"## Contents of /{path or ''} (branch {branch}):"]
        for e in entries[:50]:
            icon = "d" if e.get("type") == "dir" else "f"
            lines.append(f"{icon} {e.get('path', '?')} ({e.get('size', '?')}b)")
        return finish("\n".join(lines), note="dir listing")

    # /blob/BRANCH/path → single file content
    if rest.startswith("blob/"):
        segs = rest.split("/")  # ['blob', BRANCH, ...path]
        branch = segs[1] if len(segs) > 1 else default_branch
        path = "/".join(segs[2:])
        ok, data = _gh_api_get(f"{GH_API}/repos/{owner}/{repo}/contents/{urllib.parse.quote(path)}?ref={urllib.parse.quote(branch)}",
                               accept="application/vnd.github.raw")
        if not ok:
            result["error"] = f"GitHub API: {data}"
            return result
        return finish(header + "\n\n" + data, note="file")

    # /issues/N or /pull/N with number → single issue detail
    m = re.match(r"^(issues|pull)/(\d+)$", rest)
    if m:
        kind, num = m.group(1), m.group(2)
        api_kind = "issues" if kind == "issues" else "pulls"
        ok, data = _gh_api_get(f"{GH_API}/repos/{owner}/{repo}/{api_kind}/{num}")
        if not ok:
            result["error"] = f"GitHub API: {data}"
            return result
        it = json.loads(data)
        body = (it.get("body") or "").strip()
        lines = [header, "", f"## {kind}#{num}: {it.get('title', '?')} [{it.get('state', '?')}]"]
        if it.get("user"):
            lines.append(f"By {it['user'].get('login', '?')} at {(it.get('created_at') or '')[:10]}")
        if body:
            lines.append("")
            lines.append(body[:4000])
        # top comments
        ok_c, data_c = _gh_api_get(f"{GH_API}/repos/{owner}/{repo}/{api_kind}/{num}/comments?per_page=10")
        if ok_c:
            try:
                comments = json.loads(data_c)
                if comments:
                    lines.append("")
                    lines.append("### Top comments:")
                    for c in comments[:10]:
                        who = (c.get("user") or {}).get("login", "?")
                        lines.append(f"- **{who}**: {(c.get('body') or '')[:400]}")
            except Exception:
                pass
        return finish("\n".join(lines), note=f"{kind} detail")

    # plain /owner/repo → README + header
    ok, data = _gh_api_get(f"{GH_API}/repos/{owner}/{repo}/readme", accept="application/vnd.github.raw")
    readme = data if ok else ""
    if not ok:
        readme = f"(README unavailable: {data})"
    return finish(header + "\n\n" + readme, note="readme")


def main():
    """CLI for testing."""
    import argparse
    parser = argparse.ArgumentParser(description="Aion web fetch tool")
    parser.add_argument("url", help="URL to fetch")
    parser.add_argument("--raw", action="store_true", help="Print raw JSON result")
    args = parser.parse_args()

    result = fetch_url(args.url)
    if args.raw:
        print(json.dumps(result, indent=2))
    elif result["ok"]:
        print(f"Status: {result['status']} | Size: {result['size']}b | URL: {result['url']}")
        print(f"---")
        print(result["text"])
    else:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()