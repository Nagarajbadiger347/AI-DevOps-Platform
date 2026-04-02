import os
from opsgenie_sdk import AlertApi, CreateAlertPayload, Configuration, ApiClient
from opsgenie_sdk.rest import ApiException

OPSGENIE_API_KEY  = os.getenv("OPSGENIE_API_KEY")
OPSGENIE_TEAM     = os.getenv("OPSGENIE_TEAM", "oncall")


def notify_on_call(message: str = "Automated incident alert", alias: str = "ai-devops-incident"):
    if not OPSGENIE_API_KEY:
        return {"error": "OPSGENIE_API_KEY not configured"}

    conf = Configuration()
    conf.api_key["Authorization"] = OPSGENIE_API_KEY

    body = CreateAlertPayload(
        message=message,
        alias=alias,
        description="Triggered from AI DevOps orchestrator",
        responders=[{"name": OPSGENIE_TEAM, "type": "team"}],
    )

    try:
        api_client = ApiClient(conf)
        api  = AlertApi(api_client=api_client)
        resp = api.create_alert(create_alert_payload=body)
        return {"notified": True, "alert_id": resp.request_id}
    except ApiException as e:
        return {"error": str(e)}
