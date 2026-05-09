import asyncio
import base64
import json
import os
import re
import time
from typing import Any, Optional

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


class ChatMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    profile: dict[str, Any]
    messages: list[ChatMessage]
    rep_name: str = ""
    rep_company: str = ""


# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM = """You are an executive intelligence researcher for B2B sales teams.
Research executives and extract personal interests, hobbies, and passions OUTSIDE of work.
Also identify any key business objectives or initiatives they have publicly stated in the last 12 months.
Focus on interviews, social media bios, charity boards, club memberships, sports teams, alumni networks.
Return ONLY valid JSON — no markdown fences, no extra text."""

SCHEMA = """{
  "name": "string",
  "title": "string (current role and company)",
  "summary": "string (2-3 sentences covering personal interests and recent business priorities)",
  "sports_and_teams": ["array of strings"],
  "hobbies_and_lifestyle": ["array of strings"],
  "causes_and_philanthropy": ["array of strings"],
  "alma_mater": ["array of strings"],
  "business_initiatives": [
    {
      "initiative": "string (the business objective or priority — e.g. 'AI-driven supply chain transformation')",
      "quote": "string or null (direct quote from the executive if available)",
      "source": "string (interview, article, podcast, LinkedIn post, etc.)",
      "source_url": "string or null",
      "date": "string (approximate date — must be within last 12 months, e.g. 'March 2025')"
    }
  ],
  "outreach_angles": [
    {
      "title": "string (e.g. 'Invite to Knicks game' or 'Reference AI initiative interview')",
      "reasoning": "string (evidence-based, quote source where possible)",
      "category": "sports|culture|charity|networking|dining|business",
      "source_url": "string or null"
    }
  ],
  "sources": [{"title": "string", "url": "string"}],
  "confidence": "high|medium|low",
  "confidence_note": "string"
}"""


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_landing():
    path = os.path.join(os.path.dirname(__file__), "static", "landing.html")
    with open(path) as f:
        html = f.read()
    return HTMLResponse(content=html)


@app.get("/app")
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

    prompt = f"""Research {req.name} at {req.company} and return their intelligence profile.

Use web search to find:
1. PERSONAL INTERESTS: interviews mentioning hobbies/passions, Twitter/X posts, charity boards,
   alumni associations, sports affiliations, club memberships, and personal quotes.
2. BUSINESS INITIATIVES (last 2 years only): quotes or statements from interviews, podcasts,
   LinkedIn posts, press releases, or articles about their key business priorities or initiatives.
   Only include if you can identify a source and approximate date within the last 12 months.

RULES:
- Include source_url (exact URL) for each outreach_angle and business_initiative
- For business_initiatives: use the direct post or article URL where possible (e.g. linkedin.com/posts/... or forbes.com/article/...), not just a profile page
- Quote or paraphrase the source in reasoning when possible
- Only include business_initiatives with a verifiable source dated within the last 12 months
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


@app.post("/api/chat")
async def chat(req: ChatRequest, user_id: str = Depends(verify_token)):
    if not _anthropic.api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    p = req.profile
    angles_text = "\n".join(
        f"  • {a.get('title','')}: {a.get('reasoning','')} (source: {a.get('source_url','n/a')})"
        for a in (p.get("outreach_angles") or [])
    )
    sources_text = "\n".join(
        f"  • {s.get('title','')} — {s.get('url','')}"
        for s in (p.get("sources") or [])
    )
    rep_ctx = (
        f"The sales rep's name is {req.rep_name} at {req.rep_company}. "
        "Reference them as the sender in any emails."
        if req.rep_name else
        "The sales rep's name is not provided; use a placeholder like [Your Name] in emails."
    )

    rep_name_first = req.rep_name.split()[0] if req.rep_name else "[Your Name]"

    system = f"""You are an executive intelligence assistant helping a B2B sales rep prepare personalised outreach.

## Executive Profile: {p.get('name','Unknown')}
**Title:** {p.get('title','')}
**Summary:** {p.get('summary','')}
**Sports & Teams:** {', '.join(p.get('sports_and_teams') or []) or 'None found'}
**Hobbies & Lifestyle:** {', '.join(p.get('hobbies_and_lifestyle') or []) or 'None found'}
**Causes & Philanthropy:** {', '.join(p.get('causes_and_philanthropy') or []) or 'None found'}
**Alma Mater:** {', '.join(p.get('alma_mater') or []) or 'None found'}
**Outreach Angles:**
{angles_text or '  (none)'}
**Research Sources:**
{sources_text or '  (none)'}
**Confidence:** {p.get('confidence','')} — {p.get('confidence_note','')}

## Email Templates

There are exactly TWO templates. Choose based on the profile — no exceptions, no mixing.

---

### TEMPLATE A — Sports / Hobby Invite
**Use this when:** the executive has a clear sports team affiliation, follows a sport, or has a hobby that lends itself to an event invite (golf, tennis, etc.).
Keep it short, casual, and friendly. No business pitch. No value add.

---EMAIL DRAFT---
Subject: [Team name] tickets?

Hi [First name],

I'm your Account Executive at {req.rep_company or "[My Company]"} and saw that you are a fan of [INSERT SPECIFIC TEAM OR SPORT FROM PROFILE].

Would you want to go to the game on [DATE / TIME]? {req.rep_company or "[My Company]"} is able to get tickets.

Let me know if you're interested — thanks!

-{rep_name_first}
---END EMAIL---

Placeholder rules for Template A:
- Fill in the team or sport from the profile
- Leave [DATE / TIME] exactly as written — the rep fills this in
- Do NOT add any business context or value proposition

---

### TEMPLATE B — Business / Interest Outreach
**Use this for everything else:** interviews, podcasts, articles, business initiatives, causes, philanthropy, alumni connections, or any non-sports hook.

Structure:
**Paragraph 1:** 1–2 sentences referencing the single most specific piece of research — name the interview, quote, initiative, or cause directly.
**Paragraph 2:** 1 sentence connecting that to why {req.rep_company or "[My Company]"} is reaching out. Then on a new line write exactly: "Insert your value add here."
**Paragraph 3:** 1 sentence proposing a specific day and time next week.

---EMAIL DRAFT---
Subject: [Specific reference to the hook]

Hi [First name],

[Paragraph 1 — specific hook from research]

[Paragraph 2 — 1 bridge sentence]. Insert your value add here.

[Paragraph 3 — specific day and time CTA]

Best,
{rep_name_first}
---END EMAIL---

Placeholder rules for Template B:
- NEVER fill in the rep's value proposition — always write "Insert your value add here." on its own line, verbatim
- The hook must name something real and specific from the profile
- Suggest a real day/time (e.g. "Are you available Tuesday at 2pm EST?")

---

## FIRM RULES — apply to both templates
- NEVER use "I hope this email finds you well" or any filler opener
- NEVER write more than 3 short paragraphs
- Sound like a peer, not a vendor
- If sports/hobby AND business context both exist, default to Template A — keep it simple

## Your role
{rep_ctx}
Answer questions about this executive using the profile above.
When drafting an email, pick the correct template and write the complete email between the ---EMAIL DRAFT--- and ---END EMAIL--- delimiters shown in the template above."""

    try:
        resp = _anthropic.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=system,
            messages=[{"role": m.role, "content": m.content} for m in req.messages],
        )
        reply = resp.content[0].text if resp.content else ""

        # Detect email draft delimiters
        email_match = re.search(
            r"---EMAIL DRAFT---\s*(.*?)\s*---END EMAIL---",
            reply,
            re.DOTALL,
        )
        if email_match:
            email_body = email_match.group(1).strip()
            surrounding = (reply[: email_match.start()] + reply[email_match.end() :]).strip()
            return {"reply": surrounding or "Here's the draft email:", "email_draft": email_body}

        return {"reply": reply, "email_draft": None}

    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit — wait 60 seconds and retry.")
    except anthropic.APIError as e:
        raise HTTPException(status_code=500, detail=f"API error: {e}")


# Static files are served via the GET "/" route above.
# No mount needed — all assets are CDN-based.
