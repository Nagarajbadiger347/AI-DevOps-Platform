from app.integrations.aws_ops import list_ec2_instances, list_s3_buckets, list_cloudwatch_alarms


def check_aws_infrastructure():
    """Check basic AWS infrastructure health using the shared aws_ops integration."""
    try:
        ec2 = list_ec2_instances()
        s3 = list_s3_buckets()
        alarms = list_cloudwatch_alarms()

        if not ec2.get("success") or not s3.get("success") or not alarms.get("success"):
            errors = [
                r.get("error") for r in (ec2, s3, alarms) if not r.get("success") and r.get("error")
            ]
            return {"status": "error", "details": "; ".join(errors)}

        return {
            "status": "healthy",
            "details": {
                "ec2_instances": ec2.get("count", 0),
                "s3_buckets": s3.get("count", 0),
                "cloudwatch_alarms": alarms.get("count", 0),
            },
        }
    except Exception as e:
        return {"status": "error", "details": str(e)}
