import os
import json
import re
from datetime import datetime, time
from zoneinfo import ZoneInfo

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
        "numberOfGuests"
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
    Only includes properties that have a mapping in PROPERTY_CHANNEL_MAP_JSON.
    """
    raw_checkins = guesty_data.get("checkins")
    raw_checkouts = guesty_data.get("checkouts")

    checkins = _extract_reservations_list(raw_checkins)
    checkouts = _extract_reservations_list(raw_checkouts)

    # Aggregate by property ID
    props = {}

    # Handle check-ins
    for r in checkins:
        listing = r.get("listing") or {}
        pid = listing.get("_id") or listing.get("id") or None
        if not pid:
            continue

        prop_name = listing.get("nickname") or listing.get("title") or f"Property {pid}"
        entry = props.setdefault(pid, {"name": prop_name, "checkins": [], "checkouts": [], "channel": None})
        entry["checkins"].append(r)
        # Use our mapping helper to get the Slack channel
        if entry["channel"] is None:
            entry["channel"] = get_slack_channel_for_reservation(r)

    # Handle check-outs
    for r in checkouts:
        listing = r.get("listing") or {}
        pid = listing.get("_id") or listing.get("id") or None
        if not pid:
            continue

        prop_name = listing.get("nickname") or listing.get("title") or f"Property {pid}"
        entry = props.setdefault(pid, {"name": prop_name, "checkins": [], "checkouts": [], "channel": None})
        entry["checkouts"].append(r)
        if entry["channel"] is None:
            entry["channel"] = get_slack_channel_for_reservation(r)

    # Now build messages grouped by Slack channel
    messages_by_channel = {}

    for pid, info in props.items():
        channel_id = info.get("channel")
        if not channel_id:
            # No mapping for this property, skip for now
            continue

        text = build_property_daily_ops_text(
            property_name=info["name"],
            checkins=info["checkins"],
            checkouts=info["checkouts"],
        )

        messages_by_channel.setdefault(channel_id, []).append(text)

    return messages_by_channel


@bolt_app.event("app_mention")
def handle_app_mention(event, say, client, logger):
    text = (event.get("text") or "").strip()
    user = event.get("user")
    channel_id = event.get("channel")
    thread_ts = event.get("thread_ts") or event["ts"]

    lower_text = text.lower()

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
            summary = summarize_text_for_mode("channel_today", convo_text)

            slack_client.chat_postMessage(
                channel=channel_id,
                text=f"📋 End-of-Day Summary for Today (#{channel_name}):\n\n{summary}"
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
        # Get today's Guesty data
        guesty_data = guesty_get_todays_reservations()
        checkins = guesty_data.get("checkins")
        checkouts = guesty_data.get("checkouts")

        # 1) Global overview (same as before, sent to main ops channel)
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

        # Post global summary to main ops channel
        slack_client.chat_postMessage(
            channel=channel_id,
            text=f"8am CST - Today's Operations Overview:\n\n{summary_clean}",
        )

        # 2) Property-specific messages to each mapped property channel
        messages_by_channel = build_property_ops_messages_by_channel(guesty_data)

        for prop_channel_id, messages in messages_by_channel.items():
            # Combine messages for that property (or multiple properties mapped to same channel)
            text = "\n\n".join(messages)
            slack_client.chat_postMessage(
                channel=prop_channel_id,
                text=text,
            )

        return {"ok": True}

    except Exception:
        logger.exception("cron_ops_today failed")
        return {"ok": False, "error": "ops today failed"}

