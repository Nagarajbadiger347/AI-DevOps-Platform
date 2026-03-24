import os
import boto3
from botocore.exceptions import BotoCoreError, ClientError

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


def check_aws_infrastructure():
    """Check basic AWS infrastructure health using boto3.

    Requires AWS credentials configured (env vars or IAM).
    """
    try:
        # Check EC2 instances
        ec2 = boto3.client("ec2", region_name=AWS_REGION)
        instances = ec2.describe_instances()
        instance_count = sum(len(res["Instances"]) for res in instances["Reservations"])

        # Check S3 buckets
        s3 = boto3.client("s3", region_name=AWS_REGION)
        buckets = s3.list_buckets()
        bucket_count = len(buckets["Buckets"])

        # Check CloudWatch alarms
        cloudwatch = boto3.client("cloudwatch", region_name=AWS_REGION)
        alarms = cloudwatch.describe_alarms()
        alarm_count = len(alarms["MetricAlarms"])

        return {
            "status": "healthy",
            "details": {
                "ec2_instances": instance_count,
                "s3_buckets": bucket_count,
                "cloudwatch_alarms": alarm_count,
            }
        }
    except (BotoCoreError, ClientError) as e:
        return {"status": "error", "details": str(e)}
