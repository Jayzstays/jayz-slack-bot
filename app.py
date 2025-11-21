import os
import json
import re
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaInMemoryUpload


import requests
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler
from slack_sdk import WebClient
from openai import OpenAI
import difflib

# Load environment variables
load_dotenv()

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GUESTY_CLIENT_ID = os.environ.get("GUESTY_CLIENT_ID")
GUESTY_CLIENT_SECRET = os.environ.get("GUESTY_CLIENT_SECRET")


# Timezone for "today"
TZ = ZoneInfo("America/Los_Angeles")

# OpenAI client
ai = OpenAI(api_key=OPENAI_API_KEY)

# Slack Bolt app (for events & slash commands)
bolt_app = App(
    token=SLACK_BOT_TOKEN,
    signing_secret=SLACK_SIGNING_SECRET,
)

# Slack WebClient (for cron & helper calls)
slack_client = WebClient(token=SLACK_BOT_TOKEN)


def build_conversation_text(messages):
    """Turn Slack messages into a simple text transcript."""
    lines = []
    for m in messages:
        if m.get("subtype") == "bot_message":
            continue
        text = m.get("text", "")
        user = m.get("user", "unknown")
        if text:
            lines.append(f"<@{user}>: {text}")
    return "\n".join(lines) or "(no relevant messages)"


def summarize_text_for_mode(mode: str, convo_text: str) -> str:
    """Call OpenAI with different prompts based on mode."""
    if mode == "thread":
        system_content = (
            "You are an assistant that summarizes Slack *threads* for the "
            "Jayz Stays team. Provide a concise summary with key points, "
            "decisions, and action items."
        )
        user_content = f"Summarize this Slack thread:\n\n{convo_text}"
    elif mode == "channel_recent":
        system_content = (
            "You are an assistant that summarizes recent activity in a Slack *channel* "
            "for the Jayz Stays team. Focus on:\n"
            "- Key topics\n- Important decisions\n- Action items with owners (if any)\n"
            "Be concise."
        )
        user_content = f"Summarize the recent activity in this Slack channel:\n\n{convo_text}"
    elif mode == "channel_today":
        system_content = (
            "You are an assistant that summarizes *today's* activity in a Slack channel "
            "for the Jayz Stays team (based on the messages provided). "
            "Produce:\n- Main topics discussed today\n- Important decisions\n"
            "- Action items with owners and any dates mentioned.\n"
            "If there was little or no activity, say so."
        )
        user_content = f"Summarize *today's* activity in this Slack channel:\n\n{convo_text}"
    elif mode == "sop_suggestions":
        # NEW MODE: find implicit processes that should become SOPs
        system_content = (
            "You are a process design assistant for the Jayz Stays team. "
            "Your job is to scan Slack conversations and suggest which recurring "
            "processes or workflows should be turned into Standard Operating Procedures (SOPs). "
            "Respond in PLAIN TEXT ONLY (no markdown, no *, no #).\n\n"
            "When asked, you should:\n"
            "- Look for repeated patterns, checklists, troubleshooting steps, or decisions.\n"
            "- Suggest up to 3 SOP candidates.\n"
            "- For each candidate, give a short title and one-line description.\n"
            "- If there are no strong candidates, say: No strong SOP candidates from this channel today."
        )
        user_content = (
            "Based only on the Slack messages below, identify up to 3 potential SOPs "
            "the team might want to formalize. These should be recurring processes or "
            "clear workflows (not one-off issues).\n\n"
            "Format your answer as plain text, one SOP per line, like:\n"
            "1) Title: short description\n"
            "2) ...\n\n"
            "If there are no strong SOP candidates, reply exactly:\n"
            "No strong SOP candidates from this channel today.\n\n"
            f"SLACK MESSAGES:\n{convo_text}"
        )
    else:
        system_content = (
            "You are a helpful Slack assistant for the Jayz Stays team. "
            "Answer clearly and concisely."
        )
        user_content = convo_text

    resp = ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content



def get_today_messages(slack_api_client, channel_id: str):
    """Fetch messages from start of today in this channel."""
    now = datetime.now(TZ)
    start_of_day = datetime.combine(now.date(), time.min, tzinfo=TZ)
    oldest_ts = start_of_day.timestamp()

    history = slack_api_client.conversations_history(
        channel=channel_id,
        oldest=oldest_ts,
        limit=500,
    )
    return history.get("messages", [])


def guesty_get_access_token():
    """
    Get an OAuth access token from Guesty using client ID + secret
    via the Open API (form-encoded, as per Guesty docs).
    """
    if not GUESTY_CLIENT_ID or not GUESTY_CLIENT_SECRET:
        raise RuntimeError("Guesty client ID/secret not configured")

    token_url = "https://open-api.guesty.com/oauth2/token"

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {
        "grant_type": "client_credentials",
        "scope": "open-api",
        "client_id": GUESTY_CLIENT_ID,
        "client_secret": GUESTY_CLIENT_SECRET,
    }

    resp = requests.post(token_url, headers=headers, data=data, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    access_token = data.get("access_token")
    if not access_token:
        raise RuntimeError(f"Guesty token response missing access_token: {data}")
    return access_token

def get_drive_service():
    """
    Build a Google Drive service using a service account JSON stored in
    GOOGLE_SERVICE_ACCOUNT_JSON.
    """
    service_account_raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not service_account_raw:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON not set")

    info = json.loads(service_account_raw)

    creds = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/drive"]
    )

    return build("drive", "v3", credentials=creds)


def create_sop_file_in_drive(title: str, content: str) -> dict:
    """
    Create a Google Doc in the SOP_DRIVE_FOLDER_ID folder with the given content.
    Returns a dict with id and webViewLink.
    """
    folder_id = os.environ.get("SOP_DRIVE_FOLDER_ID")
    if not folder_id:
        raise RuntimeError("SOP_DRIVE_FOLDER_ID not set")

    drive = get_drive_service()

    file_metadata = {
        "name": title,
        "mimeType": "application/vnd.google-apps.document",
        "parents": [folder_id],
    }

    media = MediaInMemoryUpload(
        content.encode("utf-8"),
        mimetype="text/plain",
        resumable=False,
    )

    file = drive.files().create(
        body=file_metadata,
        media_body=media,
        fields="id, name, webViewLink",
    ).execute()

    return file


def search_sops_in_drive(query: str, max_results: int = 5) -> list[dict]:
    """
    Search SOP Google Docs in the SOP folder by full text / title.
    Returns a list of {id, name, webViewLink}.
    """
    folder_id = os.environ.get("SOP_DRIVE_FOLDER_ID")
    if not folder_id:
        raise RuntimeError("SOP_DRIVE_FOLDER_ID not set")

    drive = get_drive_service()

    # Basic escaping of single quotes
    safe_query = query.replace("'", "\\'")
    q = (
        f"'{folder_id}' in parents and "
        f"trashed = false and "
        f"fullText contains '{safe_query}'"
    )

    resp = drive.files().list(
        q=q,
        pageSize=max_results,
        fields="files(id, name, webViewLink)",
    ).execute()

    return resp.get("files", [])


def get_sop_content(file_id: str) -> str:
    """
    Export a Google Doc SOP as plain text.
    """
    drive = get_drive_service()
    # Google Docs must be exported to text
    data = drive.files().export(
        fileId=file_id,
        mimeType="text/plain",
    ).execute()

    if isinstance(data, bytes):
        return data.decode("utf-8", errors="ignore")
    # googleapiclient may already give str
    return str(data)


def guesty_get_todays_reservations():
    """
    Fetch today's check-ins and check-outs from Guesty using Open API.

    - Check-ins:  checkInDateLocalized == today AND status in ["confirmed", "reserved"]
    - Check-outs: checkOutDateLocalized == today AND status in ["confirmed", "reserved"]
    """
    access_token = guesty_get_access_token()

    today = datetime.now(TZ).date().isoformat()  # "YYYY-MM-DD", Guesty's expected format

    base_url = "https://open-api.guesty.com/v1/reservations"
    headers = {
        "Authorization": f"Bearer {access_token}",
    }

    # We only ask for the fields we care about
    fields = (
        "status "
        "checkInDateLocalized "
        "checkOutDateLocalized "
        "listing "
        "guest "
        "confirmationCode "
        "numberOfGuests "
        "preCheckIn "
        "customFields "
        "money "
        "price "
        "balance "
    )


    # ---- Today’s CHECK-INS ----
    filters_checkins = json.dumps([
        {
            "field": "checkInDateLocalized",
            "operator": "$eq",
            "value": today,
        },
        {
            "field": "status",
            "operator": "$in",
            "value": ["confirmed", "reserved"],
        },
    ])

    params_checkins = {
        "fields": fields,
        "filters": filters_checkins,
        "sort": "_id",
        "limit": 100,
        "skip": 0,
    }

    resp_in = requests.get(
        base_url,
        headers=headers,
        params=params_checkins,
        timeout=30,
    )
    resp_in.raise_for_status()
    checkins = resp_in.json()

    # ---- Today’s CHECK-OUTS ----
    filters_checkouts = json.dumps([
        {
            "field": "checkOutDateLocalized",
            "operator": "$eq",
            "value": today,
        },
        {
            "field": "status",
            "operator": "$in",
            "value": ["confirmed", "reserved"],
        },
    ])

    params_checkouts = {
        "fields": fields,
        "filters": filters_checkouts,
        "sort": "_id",
        "limit": 100,
        "skip": 0,
    }

    resp_out = requests.get(
        base_url,
        headers=headers,
        params=params_checkouts,
        timeout=30,
    )
    resp_out.raise_for_status()
    checkouts = resp_out.json()

    # Return both sets so the caller can summarize
    return {
        "checkins": checkins,
        "checkouts": checkouts,
    }

# >>> INSERT START: yesterday reservations detail <<<

def guesty_get_yesterday_reservations_detail():
    """
    Fetch reservations that were created yesterday (local TZ) and return a list
    of detailed reservation objects for reporting.
    """
    access_token = guesty_get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}
    base_url = "https://open-api.guesty.com/v1/reservations"

    now_local = datetime.now(TZ)
    today_local = now_local.date()
    yesterday_local = today_local - timedelta(days=1)

    start_local = datetime.combine(yesterday_local, time.min, tzinfo=TZ)
    end_local = datetime.combine(today_local, time.min, tzinfo=TZ)

    # Convert to UTC ISO strings
    start_utc = start_local.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    end_utc = end_local.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    fields = (
        "status "
        "createdAt "
        "checkInDateLocalized "
        "checkOutDateLocalized "
        "listing "
        "guest "
        "money "
        "price "
        "balance "
        "confirmationCode "
        "numberOfGuests "
        "preCheckIn "
        "customFields "
    )

    filters = json.dumps([
        {
            "field": "createdAt",
            "operator": "$gte",
            "value": start_utc,
        },
        {
            "field": "createdAt",
            "operator": "$lt",
            "value": end_utc,
        },
        {
            "field": "status",
            "operator": "$in",
            "value": ["confirmed", "reserved"],
        },
    ])

    params = {
        "fields": fields,
        "filters": filters,
        "sort": "_id",
        "limit": 200,
        "skip": 0,
    }

    try:
        resp = requests.get(
            base_url,
            headers=headers,
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return _extract_reservations_list(data)
    except Exception as e:
        print(f"Error fetching yesterday reservations from Guesty: {e}")
        return []

# <<< INSERT END

def _extract_reservations_list(raw):
    """
    Guesty Open API may return either:
      - {"results": [...], ...}
      - or a plain list [...]
    This normalizes it into a list.
    """
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return raw.get("results", [])
    return []

def _extract_revenue_from_reservation(reservation) -> float:
    """
    Try several common fields in a Guesty reservation to estimate total revenue.
    Returns 0.0 if we can't find anything.
    """
    def _to_float(val):
        try:
            return float(val)
        except Exception:
            return 0.0

    money = (
        reservation.get("money")
        or reservation.get("price")
        or reservation.get("balance")
        or {}
    )

    # Direct keys on money dict
    for key in ("total", "grossTotal", "balanceDue", "nativeTotal"):
        if isinstance(money, dict) and key in money:
            return _to_float(money.get(key))

    # Nested native dict
    if isinstance(money, dict):
        native = money.get("native")
        if isinstance(native, dict):
            for key in ("total", "grossTotal"):
                if key in native:
                    return _to_float(native.get(key))

    return 0.0

def _extract_revenue_from_reservation(reservation) -> float:
    """
    Try several common fields in a Guesty reservation to estimate total revenue.
    Returns 0.0 if we can't find anything.
    """
    def _to_float(val):
        try:
            return float(val)
        except Exception:
            return 0.0

    money = (
        reservation.get("money")
        or reservation.get("price")
        or reservation.get("balance")
        or {}
    )

    # Direct keys on money dict
    for key in ("total", "grossTotal", "balanceDue", "nativeTotal"):
        if isinstance(money, dict) and key in money:
            return _to_float(money.get(key))

    # Nested native dict
    if isinstance(money, dict):
        native = money.get("native")
        if isinstance(native, dict):
            for key in ("total", "grossTotal"):
                if key in native:
                    return _to_float(native.get(key))

    return 0.0
def guesty_get_yesterday_bookings_stats():
    """
    Fetch reservations that were created yesterday and compute:
      - number of bookings
      - approximate total revenue

    If Guesty rejects the createdAt filter, this will fall back to zeros
    and log the error.
    """
    access_token = guesty_get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}
    base_url = "https://open-api.guesty.com/v1/reservations"

    today = datetime.now(TZ).date()
    yesterday = today - timedelta(days=1)

    start_iso = f"{yesterday.isoformat()}T00:00:00.000Z"
    end_iso = f"{today.isoformat()}T00:00:00.000Z"

    fields = "status createdAt listing money price balance confirmationCode"

    filters = json.dumps([
        {
            "field": "createdAt",
            "operator": "$gte",
            "value": start_iso,
        },
        {
            "field": "createdAt",
            "operator": "$lt",
            "value": end_iso,
        },
        {
            "field": "status",
            "operator": "$in",
            "value": ["confirmed", "reserved"],
        },
    ])

    params = {
        "fields": fields,
        "filters": filters,
        "sort": "_id",
        "limit": 200,
        "skip": 0,
    }

    try:
        resp = requests.get(
            base_url,
            headers=headers,
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        reservations = _extract_reservations_list(data)

        count = len(reservations)
        revenue_total = 0.0
        for r in reservations:
            revenue_total += _extract_revenue_from_reservation(r)

        return {"count": count, "revenue": revenue_total}
    except Exception as e:
        print(f"Error fetching yesterday bookings from Guesty: {e}")
        return {"count": 0, "revenue": 0.0}


# -------------------------------------------------------------------
# NEW: Guesty + Slack helpers for listing/channel discovery + mapping
# -------------------------------------------------------------------
def guesty_get_all_properties():
    """
    Fetch all (or first N) Guesty listings/properties using Open API.
    """
    access_token = guesty_get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}

    url = "https://open-api.guesty.com/v1/listings"
    params = {"limit": 200}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # open-api usually returns a dict with "results"
        return data.get("results", [])
    except Exception as e:
        print(f"Error fetching Guesty properties: {e}")
        return []


def slack_get_all_channels():
    """
    Fetch Slack channels (public + private) that the bot can see.
    """
    try:
        result = slack_client.conversations_list(
            types="public_channel,private_channel",
            limit=200,
        )
        return result.get("channels", [])
    except Exception as e:
        print(f"Error fetching Slack channels: {e}")
        return []


def _normalize_name_for_match(s):
    """
    Normalize a name for matching:
    - lowercase
    - remove special chars
    - expand a few common abbreviations
    """
    if not s:
        return ""
    s = s.lower().strip()

    # Replace some separators with spaces
    for ch in ["-", "_", ".", ","]:
        s = s.replace(ch, " ")

    # Common abbreviations / cleanup
    replacements = {
        "apt": "apartment",
        "bdr": "bedroom",
        "br": "bedroom",
        "st ": "street ",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)

    # Collapse multiple spaces
    s = " ".join(s.split())
    return s



def suggest_property_channel_mapping(min_score: float = 0.6):
    """
    Returns a dict of {Guesty_property_id: Slack_channel_id} suggestions
    based on:
      1) Guesty 'nickname' vs Slack 'name' (preferred),
      2) 'contains' match on normalized names,
      3) fuzzy ratio fallback.
    """
    props = guesty_get_all_properties()
    channels = slack_get_all_channels()

    # Build normalized channel list
    norm_channels = []
    for c in channels:
        cid = c.get("id")
        cname = c.get("name") or ""
        norm_name = _normalize_name_for_match(cname)
        if not cid or not norm_name:
            continue
        norm_channels.append({"id": cid, "name": cname, "norm": norm_name})

    suggestions = {}

    for p in props:
        pid = p.get("_id")
        if not pid:
            continue

        # IMPORTANT: prefer nickname; only fall back to title if no nickname
        nickname = p.get("nickname")
        title = p.get("title")
        raw_name = nickname or title or f"property-{pid}"
        norm_prop = _normalize_name_for_match(raw_name)
        if not norm_prop:
            continue

        best_channel_id = None
        best_score = 0.0

        # 1) Try "contains" match first: nickname similar to channel name
        for c in norm_channels:
            if norm_prop and (norm_prop == c["norm"] or
                              norm_prop in c["norm"] or
                              c["norm"] in norm_prop):
                best_channel_id = c["id"]
                best_score = 1.0  # treat as perfect
                break

        # 2) If no contains match, use fuzzy ratio
        if not best_channel_id:
            for c in norm_channels:
                score = difflib.SequenceMatcher(None, norm_prop, c["norm"]).ratio()
                if score > best_score:
                    best_score = score
                    best_channel_id = c["id"]

        if best_channel_id and best_score >= min_score:
            suggestions[str(pid)] = best_channel_id

    return suggestions



def get_property_channel_map():
    """
    Load a manual/confirmed mapping from PROPERTY_CHANNEL_MAP_JSON env var.
    {Guesty_property_id: Slack_channel_id}
    """
    raw = os.environ.get("PROPERTY_CHANNEL_MAP_JSON")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception as e:
        print(f"Error parsing PROPERTY_CHANNEL_MAP_JSON: {e}")
        return {}


def get_slack_channel_for_reservation(reservation):
    """
    Given a Guesty reservation, return the mapped Slack channel ID
    (if configured via PROPERTY_CHANNEL_MAP_JSON).
    """
    mapping = get_property_channel_map()
    if not mapping:
        return None

    # Guesty reservation listing/property id (adjust if your shape differs)
    listing = reservation.get("listing") or {}
    listing_id = (
        listing.get("_id")
        or reservation.get("listingId")
        or reservation.get("listing_id")
        or reservation.get("propertyId")
        or reservation.get("property_id")
    )

    if not listing_id:
        return None

    return mapping.get(str(listing_id))

def _extract_reservations_list(raw):
    """
    Guesty Open API may return either:
      - {"results": [...], ...}
      - or a plain list [...]
    This normalizes it into a list.
    """
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return raw.get("results", [])
    return []


def build_property_daily_ops_text(property_name, checkins, checkouts):
    """
    Build a plain-text daily ops summary for a single property.
    No markdown, no emojis.
    """
    lines = []
    lines.append(f"Property: {property_name}")
    lines.append(f"Check-ins today: {len(checkins)}")

    if checkins:
        for r in checkins:
            guest = ""
            guest_obj = r.get("guest") or {}
            guest = (
                guest_obj.get("fullName")
                or guest_obj.get("firstName")
                or guest_obj.get("lastName")
                or "Guest"
            )
            conf = r.get("confirmationCode") or "n/a"
            num_guests = r.get("numberOfGuests") or 0
            lines.append(f"- Check-in: {guest}, {num_guests} guests, confirmation {conf}")
    else:
        lines.append("No check-ins today.")

    lines.append(f"Check-outs today: {len(checkouts)}")

    if checkouts:
        for r in checkouts:
            guest = ""
            guest_obj = r.get("guest") or {}
            guest = (
                guest_obj.get("fullName")
                or guest_obj.get("firstName")
                or guest_obj.get("lastName")
                or "Guest"
            )
            conf = r.get("confirmationCode") or "n/a"
            num_guests = r.get("numberOfGuests") or 0
            lines.append(f"- Check-out: {guest}, {num_guests} guests, confirmation {conf}")
    else:
        lines.append("No check-outs today.")

    return "\n".join(lines)


def build_property_ops_messages_by_channel(guesty_data):
    """
    Take today's Guesty check-ins/check-outs and return a dict:
      { slack_channel_id: [message_text, ...], ... }

    Uses get_slack_channel_for_reservation() directly so we don't depend
    on the exact shape of listing IDs here.
    """
    raw_checkins = guesty_data.get("checkins")
    raw_checkouts = guesty_data.get("checkouts")

    checkins = _extract_reservations_list(raw_checkins)
    checkouts = _extract_reservations_list(raw_checkouts)

    # messages_by_channel[channel_id] = {
    #   "property_name": str,
    #   "checkins": [...],
    #   "checkouts": [...],
    # }
    messages_by_channel = {}

    def add_reservation(reservation, kind: str):
        channel_id = get_slack_channel_for_reservation(reservation)
        if not channel_id:
            return

        listing = reservation.get("listing") or {}
        prop_name = (
            listing.get("nickname")
            or listing.get("title")
            or "Property"
        )

        entry = messages_by_channel.setdefault(
            channel_id,
            {"property_name": prop_name, "checkins": [], "checkouts": []},
        )

        if kind == "checkin":
            entry["checkins"].append(reservation)
        else:
            entry["checkouts"].append(reservation)

    # Add all check-ins
    for r in checkins:
        add_reservation(r, "checkin")

    # Add all check-outs
    for r in checkouts:
        add_reservation(r, "checkout")

    # Build final text per channel
    channel_texts = {}
    for channel_id, info in messages_by_channel.items():
        text = build_property_daily_ops_text(
            property_name=info["property_name"],
            checkins=info["checkins"],
            checkouts=info["checkouts"],
        )
        channel_texts.setdefault(channel_id, []).append(text)

    return channel_texts



@bolt_app.event("app_mention")
def handle_app_mention(event, say, client, logger):
    text = (event.get("text") or "").strip()
    user = event.get("user")
    channel_id = event.get("channel")
    thread_ts = event.get("thread_ts") or event["ts"]

    lower_text = text.lower()

    # --- Mode: draft an SOP from this conversation/thread ---
    if (
        "draft sop" in lower_text
        or "store this as an sop" in lower_text
        or "create sop" in lower_text
    ):
        try:
            # Get the whole thread this mention is in
            replies = client.conversations_replies(
                channel=channel_id,
                ts=thread_ts,
                limit=100,
            )
            messages = replies.get("messages", [])
            convo_text = build_conversation_text(messages)

            # 1) Draft SOP body
            sop_prompt = (
                "You are an assistant for the Jayz Stays team.\n"
                "Draft a clear, structured Standard Operating Procedure (SOP) based solely on the conversation below.\n"
                "Requirements:\n"
                "- Plain text only (no markdown, no *, no #)\n"
                "- Include: Purpose, Required Materials (if applicable), Step-by-step process\n"
                "- Use concise, professional, operational language.\n\n"
                f"CONVERSATION:\n{convo_text}\n"
            )

            sop_draft = summarize_text_for_mode("qa", sop_prompt)

            # 2) Suggest SOP title
            title_prompt = (
                "Based on the conversation below, generate a clear, professional SOP title.\n"
                "Make it short, descriptive, and capitalized.\n"
                "Do NOT add quotes. Do NOT add extra text.\n\n"
                f"CONVERSATION:\n{convo_text}\n"
            )

            suggested_title = summarize_text_for_mode("qa", title_prompt)
            suggested_title = suggested_title.replace("\n", " ").strip()

            say(
                f"Here is a proposed SOP based on this conversation:\n\n"
                f"Suggested Title: {suggested_title}\n\n"
                f"SOP DRAFT START\n{sop_draft}\nSOP DRAFT END\n\n"
                f"If this looks good, reply in this thread with:\n"
                f"save sop: {suggested_title}",
                thread_ts=thread_ts,
            )
        except Exception as e:
            logger.error(f"Error drafting SOP with title suggestion: {e}")
            say(
                "Sorry, I could not draft an SOP from this conversation.",
                thread_ts=thread_ts,
            )
        return

    # --- Mode: save the last SOP draft in this thread to Google Drive ---
    if lower_text.startswith("save sop:"):
        try:
            # Extract title from "save sop: <title>"
            title = text.split("save sop:", 1)[1].strip()
            if not title:
                title = "Untitled SOP"

            # Fetch the full thread to find the most recent SOP draft
            replies = client.conversations_replies(
                channel=channel_id,
                ts=thread_ts,
                limit=100,
            )
            messages = replies.get("messages", [])

            sop_text = None

            # Look from newest to oldest for a bot message containing SOP DRAFT START/END
            for m in reversed(messages):
                if m.get("subtype") != "bot_message":
                    continue
                body_text = m.get("text", "")
                if "SOP DRAFT START" in body_text and "SOP DRAFT END" in body_text:
                    start_idx = body_text.index("SOP DRAFT START") + len("SOP DRAFT START")
                    end_idx = body_text.index("SOP DRAFT END", start_idx)
                    sop_text = body_text[start_idx:end_idx].strip()
                    break

            if not sop_text:
                say(
                    "I could not find a SOP draft in this thread. "
                    "Try asking me to 'draft sop' again first.",
                    thread_ts=thread_ts,
                )
                return

            # Save to Google Drive as a Google Doc
            file = create_sop_file_in_drive(title, sop_text)
            name = file.get("name", title)
            link = file.get("webViewLink", "(no link)")

            say(
                f"SOP '{name}' has been saved to Google Drive.\n"
                f"Link: {link}",
                thread_ts=thread_ts,
            )
        except Exception as e:
            logger.error(f"Error saving SOP to Drive: {e}")
            say(
                "Sorry, I could not save the SOP to Google Drive. Please check the logs.",
                thread_ts=thread_ts,
            )
        return

    # --- Mode 1: summarize this thread ---
    if "summarize this thread" in lower_text or "summarise this thread" in lower_text:
        try:
            replies = client.conversations_replies(
                channel=channel_id,
                ts=thread_ts,
                limit=100,
            )
            messages = replies.get("messages", [])
            convo_text = build_conversation_text(messages)
            summary = summarize_text_for_mode("thread", convo_text)
            say(
                f"Here’s a summary of this *thread*, <@{user}>:\n\n{summary}",
                thread_ts=thread_ts,
            )
        except Exception as e:
            logger.error(f"Error summarizing thread: {e}")
            say(
                f"Sorry <@{user}>, I couldn’t summarize this thread due to an internal error.",
                thread_ts=thread_ts,
            )
        return

    # --- Mode 2: summarize recent channel (last ~100 messages) ---
    if "summarize this channel" in lower_text or "summarise this channel" in lower_text:
        try:
            history = client.conversations_history(
                channel=channel_id,
                limit=100,
            )
            messages = history.get("messages", [])
            convo_text = build_conversation_text(messages)
            summary = summarize_text_for_mode("channel_recent", convo_text)
            say(
                f"Here’s a summary of recent activity in this *channel*, <@{user}>:\n\n{summary}",
                thread_ts=thread_ts,
            )
        except Exception as e:
            logger.error(f"Error summarizing channel: {e}")
            say(
                f"Sorry <@{user}>, I couldn’t summarize this channel due to an internal error.",
                thread_ts=thread_ts,
            )
        return

    # --- Mode 3: summarize *today* in this channel ---
    if "summarize today" in lower_text or "summarise today" in lower_text:
        try:
            messages = get_today_messages(client, channel_id)
            convo_text = build_conversation_text(messages)
            summary = summarize_text_for_mode("channel_today", convo_text)
            say(
                f"Here’s a summary of *today* in this channel, <@{user}>:\n\n{summary}",
                thread_ts=thread_ts,
            )
        except Exception as e:
            logger.error(f"Error summarizing today: {e}")
            say(
                f"Sorry <@{user}>, I couldn’t summarize today’s activity due to an internal error.",
                thread_ts=thread_ts,
            )
        return

    # --- Mode 4: normal Q&A / chat ---
    answer = summarize_text_for_mode("qa", text)
    say(
        f"<@{user}> {answer[:3500]}",
        thread_ts=thread_ts,
    )


# Slash command: /daily_summary_now (manual daily summary for this channel)
@bolt_app.command("/daily_summary_now")
def daily_summary_now(ack, body, respond, client, logger):
    ack()

    channel_id = body.get("channel_id")
    user_id = body.get("user_id")

    try:
        messages = get_today_messages(client, channel_id)
        convo_text = build_conversation_text(messages)
        summary = summarize_text_for_mode("channel_today", convo_text)

        client.chat_postMessage(
            channel=channel_id,
            text=f"📋 End-of-day summary for today in this channel "
                 f"(requested by <@{user_id}>):\n\n{summary}",
        )
        respond("I’ve posted today’s summary in this channel ✅")
    except Exception as e:
        logger.error(f"Error in /daily_summary_now: {e}")
        respond("Sorry, I couldn’t generate the daily summary due to an error.")


@bolt_app.command("/ops_today")
def ops_today(ack, body, respond, logger):
    """
    Show today's Guesty check-ins and check-outs.
    """
    ack()

    user_id = body.get("user_id")

    try:
        guesty_data = guesty_get_todays_reservations()
        checkins = guesty_data.get("checkins")
        checkouts = guesty_data.get("checkouts")

        prompt_text = (
            "You are an assistant for the Jayz Stays operations team. "
            "Write the summary in PLAIN TEXT ONLY. "
            "Do NOT use Markdown or formatting. No asterisks, no hash symbols, "
            "no bold, no headings, no lists using hyphens and spaces like '- '. "
            "Instead, write each line as a simple sentence or bullet using only hyphens with NO space after them.\n\n"
            "Provide:\n"
            "-Number of check-ins today\n"
            "-Number of check-outs today\n"
            "-Patterns by property or channel\n"
            "-Operational notes written in plain text with no formatting.\n\n"
            f"CHECK-INS JSON:\n{checkins}\n\n"
            f"CHECK-OUTS JSON:\n{checkouts}\n"
        )

        summary = summarize_text_for_mode("qa", prompt_text)

        respond(
            f"📋 *Today's Guesty operations overview* "
            f"(requested by <@{user_id}>):\n\n{summary}"
        )
    except Exception as e:
        logger.error(f"/ops_today error: {e}")
        respond(f"⚠️ Error talking to Guesty: `{e}`")

@bolt_app.command("/save_sop")
def save_sop(ack, body, respond, logger):
    """
    Save an SOP from Slack into Google Drive as a Google Doc.

    Usage from Slack:
      /save_sop Title of SOP | step 1, step 2, step 3...

    If no '|' is provided, the whole text is used as both the title and content.
    """
    ack()

    user_id = body.get("user_id")
    raw_text = (body.get("text") or "").strip()

    if not raw_text:
        respond(
            "Please provide SOP text.\n"
            "Example: /save_sop Cleaning Checklist | Step 1, step 2, step 3..."
        )
        return

    if "|" in raw_text:
        title_part, content_part = raw_text.split("|", 1)
        title = title_part.strip()
        content = content_part.strip()
    else:
        title = raw_text[:80].strip()
        content = raw_text

    if not title:
        title = "Untitled SOP"

    try:
        file = create_sop_file_in_drive(title, content)
        name = file.get("name", title)
        link = file.get("webViewLink", "(no link)")

        respond(
            f"SOP saved as '{name}'.\n"
            f"Drive link: {link}\n"
            f"Requested by <@{user_id}>."
        )
    except Exception as e:
        logger.error(f"/save_sop error: {e}")
        respond("Sorry, I could not save the SOP to Google Drive. Check the logs.")


# FastAPI wrapper for Slack events
app = FastAPI()
handler = SlackRequestHandler(bolt_app)

# Logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.post("/slack/events")
async def slack_events(req: Request):
    return await handler.handle(req)


@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.get("/debug/test-property-map")
def debug_test_property_map():
    # Fake a Guesty reservation that matches your known property
    test_reservation = {
        "listing": {"_id": "63c1b699841d8b00503ce4bf"}
    }

    try:
        result = get_slack_channel_for_reservation(test_reservation)
    except Exception as e:
        return {"error": str(e)}

    return {
        "expected": "C05AGSB18GZ",
        "got": result
    }
@app.get("/debug/533-today")
def debug_533_today():
    """
    Debug endpoint to see if 533 South 51st St has any
    check-ins or check-outs today and what Slack channel
    we would post to.
    """
    try:
        guesty_data = guesty_get_todays_reservations()
    except Exception as e:
        return {"error": f"Error calling guesty_get_todays_reservations: {e}"}

    raw_checkins = guesty_data.get("checkins")
    raw_checkouts = guesty_data.get("checkouts")

    checkins = _extract_reservations_list(raw_checkins)
    checkouts = _extract_reservations_list(raw_checkouts)

    target_id = "63c1b699841d8b00503ce4bf"

    cin_533 = []
    cout_533 = []

    for r in checkins:
        listing = r.get("listing") or {}
        if listing.get("_id") == target_id:
            cin_533.append(r)

    for r in checkouts:
        listing = r.get("listing") or {}
        if listing.get("_id") == target_id:
            cout_533.append(r)

    # Also test what channel we'd route to if there IS at least one reservation
    channel_for_first = None
    if cin_533:
        channel_for_first = get_slack_channel_for_reservation(cin_533[0])
    elif cout_533:
        channel_for_first = get_slack_channel_for_reservation(cout_533[0])

    return {
        "property_id": target_id,
        "checkins_count": len(cin_533),
        "checkouts_count": len(cout_533),
        "channel_for_first_reservation": channel_for_first,
        "checkins_sample": cin_533[:2],
        "checkouts_sample": cout_533[:2],
    }

@bolt_app.command("/sop_help")
def sop_help(ack, body, respond, logger):
    """
    Help employees by searching SOPs in Google Drive and answering
    their question using the best-matching SOP.
    """
    ack()

    user_id = body.get("user_id")
    query = (body.get("text") or "").strip()

    if not query:
        respond(
            "Please provide a question or topic.\n"
            "Example: /sop_help late checkout procedure"
        )
        return

    try:
        files = search_sops_in_drive(query, max_results=3)
    except Exception as e:
        logger.error(f"/sop_help Drive search error: {e}")
        respond("Sorry, I could not search SOPs in Google Drive.")
        return

    if not files:
        respond(f"I could not find any SOPs matching: {query}")
        return

    # Take the best match (first file)
    target = files[0]
    file_id = target["id"]
    file_name = target.get("name", "SOP")
    link = target.get("webViewLink", "")

    try:
        sop_text = get_sop_content(file_id)
    except Exception as e:
        logger.error(f"/sop_help error getting content: {e}")
        respond(
            f"I found an SOP named '{file_name}', "
            f"but could not read its content. Link: {link}"
        )
        return

    # Ask OpenAI to answer using the SOP text
    prompt = (
        "You are an assistant for the Jayz Stays team. "
        "Answer the employee's question using ONLY the SOP text below. "
        "If the SOP does not cover the question, say you are not sure.\n\n"
        f"Employee question:\n{query}\n\n"
        "SOP TEXT:\n"
        f"{sop_text}\n"
    )

    try:
        answer = summarize_text_for_mode("qa", prompt)
    except Exception as e:
        logger.error(f"/sop_help OpenAI error: {e}")
        respond(
            f"I found an SOP named '{file_name}', but had trouble generating guidance. "
            f"You can read it here: {link}"
        )
        return

    full_reply = (
        f"Based on SOP '{file_name}', here is the guidance:\n\n"
        f"{answer}\n\n"
        f"Full SOP: {link}"
    )

    respond(full_reply[:3000])  # keep under Slack limits


# ---------------------------------------
# NEW DEBUG: show properties and channels
# ---------------------------------------
@app.get("/debug/properties-and-channels")
def debug_properties_and_channels():
    props = guesty_get_all_properties()
    channels = slack_get_all_channels()

    output = []

    output.append("GUESTY PROPERTIES:\n")
    for p in props:
        pid = p.get("_id")
        name = p.get("title") or p.get("nickname") or "Unnamed Property"
        output.append(f"- {pid}: {name}")

    output.append("\nSLACK CHANNELS:\n")
    for c in channels:
        cid = c.get("id")
        name = c.get("name")
        output.append(f"- {cid}: {name}")

    return PlainTextResponse("\n".join(output))

@app.get("/debug/guesty-full-listing")
def debug_guesty_full_listing():
    """
    Fetch ONE listing from Guesty with ALL fields so we can inspect
    lock-related information (if any).
    """
    try:
        access_token = guesty_get_access_token()
    except Exception as e:
        return {"error": f"Token error: {e}"}

    url = "https://open-api.guesty.com/v1/listings"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {
        "limit": 1,
        "sort": "-createdAt"
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data
    except Exception as e:
        return {"error": str(e)}


# -----------------------------------------------
# NEW DEBUG: suggest property -> channel mapping
# -----------------------------------------------
@app.get("/debug/suggest-property-channel-map")
def debug_suggest_property_channel_map():
    try:
        mapping = suggest_property_channel_mapping(min_score=0.6)
        text = json.dumps(mapping, indent=2)
        return PlainTextResponse(text)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        # Return the error as plain text so we can see what went wrong
        error_text = f"ERROR in suggest_property_channel_mapping:\n{e}\n\nTRACEBACK:\n{tb}"
        return PlainTextResponse(error_text)

def _is_guesty_app_signed(reservation) -> bool:
    """
    Determine whether the guest has completed/signed the Guesty app / pre-check-in.
    This is a placeholder and should be adjusted once you know the exact field
    from your Guesty reservation JSON.
    """
    pre_checkin = (
        reservation.get("preCheckIn")
        or reservation.get("preCheckin")
        or reservation.get("pre_check_in")
        or {}
    )

    if isinstance(pre_checkin, dict):
        # Common patterns – adjust once you see real structure
        status = (pre_checkin.get("status") or "").lower()
        if status in ("completed", "signed", "done", "finished"):
            return True
        if pre_checkin.get("isCompleted") is True:
            return True

    # Fallback: assume not signed
    return False

def format_yesterday_reservations_section(reservations: list[dict]) -> str:
    """
    Build a plain-text section describing yesterday's new reservations:
    - property nickname
    - booking dates
    - total revenue
    """
    if not reservations:
        return "Yesterday's new reservations: None."

    lines = []
    lines.append("Yesterday's new reservations:")

    total_revenue = 0.0

    for r in reservations:
        listing = r.get("listing") or {}
        prop_name = listing.get("nickname") or listing.get("title") or "Unknown property"

        checkin = r.get("checkInDateLocalized") or "?"
        checkout = r.get("checkOutDateLocalized") or "?"

        revenue = _extract_revenue_from_reservation(r)
        total_revenue += revenue

        guest = r.get("guest") or {}
        guest_name = (
            guest.get("fullName")
            or guest.get("firstName")
            or guest.get("lastName")
            or "Guest"
        )

        lines.append(
            f"- {prop_name} | {checkin} to {checkout} | {guest_name} | "
            f"Revenue approx: ${revenue:,.2f}"
        )

    lines.append(f"Total revenue from yesterday's new reservations: ${total_revenue:,.2f}")
    return "\n".join(lines)

def format_today_missing_app_section(todays_checkins_raw) -> str:
    """
    Among today's check-ins, list which guests have NOT signed / completed
    the Guesty app / pre-check-in.
    """
    checkins = _extract_reservations_list(todays_checkins_raw)
    missing = []

    for r in checkins:
        if _is_guesty_app_signed(r):
            continue

        listing = r.get("listing") or {}
        prop_name = listing.get("nickname") or listing.get("title") or "Unknown property"

        checkin = r.get("checkInDateLocalized") or "?"
        checkout = r.get("checkOutDateLocalized") or "?"

        guest = r.get("guest") or {}
        guest_name = (
            guest.get("fullName")
            or guest.get("firstName")
            or guest.get("lastName")
            or "Guest"
        )

        missing.append((prop_name, guest_name, checkin, checkout))

    if not missing:
        return "Guesty app / pre-check-in: All guests checking in today appear to be completed."

    lines = []
    lines.append("Guests checking in today who have NOT completed the Guesty app / pre-check-in:")
    for prop_name, guest_name, checkin, checkout in missing:
        lines.append(
            f"- {prop_name} | {checkin} to {checkout} | {guest_name}"
        )

    return "\n".join(lines)

@app.get("/debug/guesty-full-reservation")
def debug_guesty_full_reservation():
    """
    Fetch ONE reservation from Guesty with ALL fields (no 'fields' filter)
    so we can inspect the complete structure.
    """
    try:
        access_token = guesty_get_access_token()
    except Exception as e:
        return {"error": f"Token error: {e}"}

    url = "https://open-api.guesty.com/v1/reservations"

    headers = {"Authorization": f"Bearer {access_token}"}

    # Just get the most recent reservation
    params = {
        "limit": 1,
        "sort": "-createdAt"   # newest first
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data
    except Exception as e:
        return {"error": str(e)}


# -------------------------------
# UTIL: Fetch today's messages in a channel
# -------------------------------
def get_today_messages(slack_client, channel_id):
    """Fetch today’s Slack messages from a channel."""
    today = datetime.now(TZ).date().isoformat()

    resp = slack_client.conversations_history(
        channel=channel_id,
        limit=1000,
    )

    messages = []
    for m in resp.get("messages", []):
        ts = float(m.get("ts", 0))
        d = datetime.fromtimestamp(ts, TZ).date().isoformat()
        if d == today:
            messages.append(m)

    return messages


def build_conversation_text(messages):
    """Convert messages list into plain text."""
    lines = []
    for m in messages:
        user = m.get("user", "unknown")
        text = m.get("text", "")
        lines.append(f"{user}: {text}")
    return "\n".join(lines)


# -------------------------------
# CRON: Daily Slack Channel Summary
# -------------------------------
@app.api_route("/cron/daily-summary", methods=["POST", "GET"])
async def cron_daily_summary(request: Request):
    logger.info("cron_daily_summary triggered via %s", request.method)
    try:
        # List channels
        channels = []
        cursor = None
        while True:
            resp = slack_client.conversations_list(
                types="public_channel,private_channel",
                limit=200,
                cursor=cursor,
            )
            channels.extend(resp.get("channels", []))
            cursor = resp.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

        # Summarize each channel where bot is a member
        for ch in channels:
            if not ch.get("is_member"):
                continue

            channel_id = ch["id"]
            channel_name = ch.get("name")
            messages = get_today_messages(slack_client, channel_id)

            if not messages:
                continue

            convo_text = build_conversation_text(messages)

            # Regular end-of-day summary
            summary = summarize_text_for_mode("channel_today", convo_text)

            # NEW: suggested SOPs
            try:
                sop_suggestions = summarize_text_for_mode("sop_suggestions", convo_text)
            except Exception as e:
                logger.error(f"Error generating SOP suggestions for channel {channel_id}: {e}")
                sop_suggestions = "Could not generate SOP suggestions today."

            text = (
                f"📋 End-of-Day Summary for Today (#{channel_name}):\n\n"
                f"{summary}\n\n"
                f"Suggested SOPs from today's discussion:\n"
                f"{sop_suggestions}"
            )

            slack_client.chat_postMessage(
                channel=channel_id,
                text=text[:3900],  # stay under Slack hard limits
            )

        return {"ok": True}

    except Exception:
        logger.exception("cron_daily_summary failed")
        return {"ok": False, "error": "daily summary failed"}



# -------------------------------
# CRON: Daily Ops Today (8am CST)
# -------------------------------
@app.api_route("/cron/ops-today", methods=["POST", "GET"])
async def cron_ops_today(request: Request):
    logger.info("cron_ops_today triggered via %s", request.method)
    channel_id = os.environ.get("OPS_TODAY_CHANNEL_ID")

    if not channel_id:
        return {"ok": False, "error": "OPS_TODAY_CHANNEL_ID not set"}

    try:
        # 1) Today's Guesty data (check-ins / check-outs)
        guesty_data = guesty_get_todays_reservations()
        checkins = guesty_data.get("checkins")
        checkouts = guesty_data.get("checkouts")

        # 2) Yesterday's new reservations (details + revenue)
        yesterday_reservations = guesty_get_yesterday_reservations_detail()
        yesterday_section = format_yesterday_reservations_section(yesterday_reservations)

        # 3) Today's check-ins missing Guesty app / pre-check-in
        missing_app_section = format_today_missing_app_section(checkins)

        # 4) Global overview via OpenAI (today's ops)
        prompt_text = (
            "You are an assistant for the Jayz Stays operations team. "
            "Provide a clear, concise summary in PLAIN TEXT ONLY. "
            "Do NOT use *, #, markdown lists, or bold.\n\n"
            "Include:\n"
            "- Number of check-ins today\n"
            "- Number of check-outs today\n"
            "- Patterns by property or channel\n"
            "- Operational notes in plain text\n\n"
            f"CHECK-INS JSON:\n{checkins}\n\n"
            f"CHECK-OUTS JSON:\n{checkouts}\n"
        )

        summary = summarize_text_for_mode("qa", prompt_text)

        # Additional cleaning
        summary_clean = re.sub(r"^#+\s*", "", summary, flags=re.MULTILINE)
        summary_clean = summary_clean.replace("*", "")
        summary_clean = summary_clean.replace("#", "")

        # 5) Compose final text
        final_text = (
            "8am CST - Today's Operations Overview:\n\n"
            f"{yesterday_section}\n\n"
            f"{missing_app_section}\n\n"
            f"Today's check-ins and check-outs summary:\n"
            f"{summary_clean}"
        )

        # 6) Post to Slack main ops channel
        slack_client.chat_postMessage(
            channel=channel_id,
            text=final_text,
        )

        # (Optional) if you already had per-property posts, keep that block here

        return {"ok": True}

    except Exception:
        logger.exception("cron_ops_today failed")
        return {"ok": False, "error": "ops today failed"}

