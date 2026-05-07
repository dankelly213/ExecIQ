import asyncio
import base64
import json
import os
import re
import time
from typing import Optional

import anthropic
import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from jose import JWTError, jwk, jwt
from pydantic import BaseModel
from supabase import Client, create_client

app = FastAPI()

# ── Clients ────────────────────────────────────────────────────────────────────
_anthropic = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
_supabase: Optional[Client] = None


def get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_SERVICE_KEY", "")
        if not url or not key:
            raise HTTPException(
                status_code=503,
                detail="Database not configured. Set SUPABASE_URL and SUPABASE_SERVICE_KEY.",
            )
        _supabase = create_client(url, key)
    return _supabase


# ── Clerk JWT verification ─────────────────────────────────────────────────────
_jwks_cache: dict = {"data": None, "at": 0.0}


def _clerk_frontend_api() -> str:
    pk = os.environ.get("CLERK_PUBLISHABLE_KEY", "")
    if not pk:
        raise HTTPException(status_code=500, detail="CLERK_PUBLISHABLE_KEY not set")
    parts = pk.split("_", 2)
    if len(parts) != 3:
        raise HTTPException(status_code=500, detail="Invalid CLERK_PUBLISHABLE_KEY format")
    b64 = parts[2] + "=" * (4 - len(parts[2]) % 4)
    return base64.b64decode(b64).decode("utf-8").rstrip("$")


async def _get_jwks() -> dict:
    if _jwks_cache["data"] and time.time() - _jwks_cache["at"] < 3600:
        return _jwks_cache["data"]
    url = f"https://{_clerk_frontend_api()}/.well-known/jwks.json"
    async with httpx.AsyncClient() as c:
        r = await c.get(url, timeout=10)
        r.raise_for_status()
    _jwks_cache.update(data=r.json(), at=time.time())
    return _jwks_cache["data"]


async def verify_token(authorization: Optional[str] = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        keys = await _get_jwks()
        kid = jwt.get_unverified_header(token).get("kid")
        key_data = next((k for k in keys.get("keys", []) if k.get("kid") == kid), None)
        if not key_data:
            raise HTTPException(status_code=401, detail="Signing key not found")
        payload = jwt.decode(
            token,
            jwk.construct(key_data),
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
        uid = payload.get("sub")
        if not uid:
            raise HTTPException(status_code=401, detail="Token missing subject")
        return uid
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Token error: {e}")


# ── Models ─────────────────────────────────────────────────────────────────────
class ProfileRequest(BaseModel):
    name: str
    company: str


# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM = """You are an executive intelligence researcher for B2B sales teams.
Research executives and extract personal interests, hobbies, and passions OUTSIDE of work.
Focus on interviews, social media bios, charity boards, club memberships, sports teams, alumni networks.
Return ONLY valid JSON — no markdown fences, no extra text."""

SCHEMA = """{
  "name": "string",
  "title": "string (current role and company)",
  "summary": "string (2-3 sentences about personal interests outside work)",
  "sports_and_teams": ["array of strings"],
  "hobbies_and_lifestyle": ["array of strings"],
  "causes_and_philanthropy": ["array of strings"],
  "alma_mater": ["array of strings"],
  "outreach_angles": [
    {
      "title": "string (e.g. 'Invite to Knicks game')",
      "reasoning": "string (evidence-based, quote source where possible)",
      "category": "sports|culture|charity|networking|dining",
      "source_url": "string or null"
    }
  ],
  "sources": [{"title": "string", "url": "string"}],
  "confidence": "high|medium|low",
  "confidence_note": "string"
}"""


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(path) as f:
        html = f.read()
    pk = os.environ.get("CLERK_PUBLISHABLE_KEY", "")
    html = html.replace("__CLERK_PK__", pk)
    return HTMLResponse(content=html)


@app.post("/api/profile")
async def generate_profile(req: ProfileRequest, user_id: str = Depends(verify_token)):
    if not _anthropic.api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    prompt = f"""Research {req.name} at {req.company} and return their personal interests profile.

Use web search to find: interviews mentioning hobbies/passions, Twitter/X posts, charity boards,
alumni associations, sports affiliations, club memberships, and personal quotes.

RULES:
- Include source_url (exact URL) for each outreach_angle fact
- List all useful pages in sources array
- Quote or paraphrase the source in reasoning when possible
- Set confidence "low" if data is sparse

Return ONLY this JSON (no fences, exactly 3 outreach_angles):
{SCHEMA}"""

    try:
        resp = _anthropic.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=3000,
            system=SYSTEM,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
            messages=[{"role": "user", "content": prompt}],
        )

        search_sources = [
            {"url": getattr(item, "url", None), "title": getattr(item, "title", None)}
            for block in resp.content
            if block.type == "web_search_tool_result"
            for item in block.content
            if getattr(item, "url", None)
        ]

        text = "".join(b.text for b in resp.content if b.type == "text")
        if not text.strip():
            raise HTTPException(status_code=500, detail="Empty response from model")

        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        json_str = m.group(1) if m else text[text.find("{") : text.rfind("}") + 1]
        if not json_str:
            raise HTTPException(status_code=500, detail="No JSON in response")

        profile = json.loads(json_str)

        # Merge any sources Claude missed
        seen = {s.get("url") for s in profile.get("sources", [])}
        for s in search_sources:
            if s["url"] not in seen:
                profile.setdefault("sources", []).append(s)
                seen.add(s["url"])

        # Save to Supabase
        db = get_supabase()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: db.table("exec_profiles")
            .insert(
                {
                    "user_id": user_id,
                    "exec_name": req.name,
                    "company": req.company,
                    "profile_data": profile,
                }
            )
            .execute(),
        )

        return profile

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON parse error: {e}")
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit — wait 60 seconds and retry.")
    except anthropic.APIError as e:
        raise HTTPException(status_code=500, detail=f"API error: {e}")


@app.get("/api/profiles")
async def list_profiles(user_id: str = Depends(verify_token)):
    db = get_supabase()
    loop = asyncio.get_event_loop()
    res = await loop.run_in_executor(
        None,
        lambda: db.table("exec_profiles")
        .select("id, exec_name, company, profile_data, created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute(),
    )
    return res.data


@app.delete("/api/profiles/{pid}")
async def delete_profile(pid: str, user_id: str = Depends(verify_token)):
    db = get_supabase()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: db.table("exec_profiles")
        .delete()
        .eq("id", pid)
        .eq("user_id", user_id)
        .execute(),
    )
    return {"ok": True}


# Static files are served via the GET "/" route above.
# No mount needed — all assets are CDN-based.
