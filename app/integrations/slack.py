import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL = os.getenv("SLACK_CHANNEL", "#general")


def create_war_room(topic: str = "War Room", members=None):
    """Create or post to a Slack channel as war room automation.

    Requires env var SLACK_BOT_TOKEN.
    """
    if not SLACK_BOT_TOKEN:
        return {"error": "SLACK_BOT_TOKEN not configured"}

    client = WebClient(token=SLACK_BOT_TOKEN)
    channel = SLACK_CHANNEL

    try:
        msg = f"🚨 War room opened: {topic}. Participants: {members or 'TBD'}"
        response = client.chat_postMessage(channel=channel, text=msg)
        return {
            "room_url": f"https://slack.com/app_redirect?channel={channel.strip('#')}",
            "ts": response.get("ts"),
            "message": msg,
        }
    except SlackApiError as e:
        return {"error": str(e), "details": e.response.get("error")}
