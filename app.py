import os
from datetime import datetime, time
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler
from slack_sdk import WebClient
from openai import OpenAI

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
    For now: fetch a batch of recent reservations from Guesty via Open API.
    Once this works, we can add filters for today's check-ins/check-outs.
    """
    access_token = guesty_get_access_token()

    reservations_url = "https://open-api.guesty.com/v1/reservations"

    headers = {
        "Authorization": f"Bearer {access_token}",
    }

    params = {
        # Basic "all reservations" query; Guesty expects filters as a JSON-encoded string
        "filters": "[]",
        "skip": 0,
        "limit": 25,
        "sort": "_id",
    }

    resp = requests.get(
        reservations_url,
        headers=headers,
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()



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
    Show today's operational picture from Guesty (currently 'recent reservations').
    """
    ack()

    user_id = body.get("user_id")

    try:
        guesty_data = guesty_get_todays_reservations()

        prompt_text = (
            "You are an assistant for the Jayz Stays operations team. "
            "I will give you raw reservation data from Guesty. "
            "Summarize it clearly. Include:\n"
            "- Number of reservations\n"
            "- Any notable patterns (length of stay, properties, channels)\n"
            "- Anything that looks operationally important.\n\n"
            f"Raw data:\n{guesty_data}"
        )

        summary = summarize_text_for_mode("qa", prompt_text)

        respond(
            f"📋 *Guesty reservations overview* "
            f"(requested by <@{user_id}>):\n\n{summary}"
        )
    except Exception as e:
        logger.error(f"/ops_today error: {e}")
        respond("Sorry, I couldn’t fetch today’s data from Guesty. Check the logs for details.")



# FastAPI wrapper for Slack events
app = FastAPI()
handler = SlackRequestHandler(bolt_app)


@app.post("/slack/events")
async def slack_events(req: Request):
    return await handler.handle(req)


@app.get("/healthz")
async def healthz():
    return {"ok": True}


# Cron endpoint: summarize today for all channels the bot is in
@app.post("/cron/daily-summary")
async def cron_daily_summary():
    # List public & private channels the bot can see
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

    # For each channel where the bot is a member, summarize today
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
            text=(
                f"📋 Automated end-of-day summary for *today* in this channel"
                + (f" (#{channel_name})" if channel_name else "")
                + f":\n\n{summary}"
            ),
        )

    return {"ok": True}
