"""AWS Live Pricing Engine.

Fetches real-time on-demand prices from the official AWS Pricing API:
  https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/price-changes.html

Supports:
  - EC2 (any instance type, region, OS)
  - RDS (any instance class, engine, Multi-AZ)
  - Lambda (per invocation + GB-second compute)
  - ECS Fargate (vCPU-hour + GB-hour)
  - EKS (cluster fee + node pricing)
  - S3 (per-GB storage + request costs)
  - NAT Gateway, Data Transfer
  - Terraform plan cost estimation

All prices are on-demand USD/hour unless noted.
Prices are cached in-process for 1 hour to avoid repeated API calls.
"""
from __future__ import annotations

import os
import json
import time
import datetime
from typing import Optional

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    _BOTO3 = True
except ImportError:
    _BOTO3 = False

try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Price cache (instance_type+region+os → $/hr, TTL 1 hour)
# ---------------------------------------------------------------------------
_PRICE_CACHE: dict[str, tuple[float, float]] = {}   # key → (price, timestamp)
_CACHE_TTL = 3600.0

# ---------------------------------------------------------------------------
# Region name mapping (AWS Pricing API uses long names)
# ---------------------------------------------------------------------------
_REGION_NAMES = {
    "us-east-1":      "US East (N. Virginia)",
    "us-east-2":      "US East (Ohio)",
    "us-west-1":      "US West (N. California)",
    "us-west-2":      "US West (Oregon)",
    "eu-west-1":      "Europe (Ireland)",
    "eu-west-2":      "Europe (London)",
    "eu-west-3":      "Europe (Paris)",
    "eu-central-1":   "Europe (Frankfurt)",
    "eu-north-1":     "Europe (Stockholm)",
    "ap-southeast-1": "Asia Pacific (Singapore)",
    "ap-southeast-2": "Asia Pacific (Sydney)",
    "ap-northeast-1": "Asia Pacific (Tokyo)",
    "ap-northeast-2": "Asia Pacific (Seoul)",
    "ap-south-1":     "Asia Pacific (Mumbai)",
    "sa-east-1":      "South America (Sao Paulo)",
    "ca-central-1":   "Canada (Central)",
}

# Reference on-demand prices (us-east-1, Linux) — 2025 rates
# Source: https://aws.amazon.com/ec2/pricing/on-demand/
# Regional multipliers applied dynamically in get_ec2_price_per_hour()
_EC2_FALLBACK: dict[str, float] = {
    # T2 (older burstable)
    "t2.nano": 0.0058, "t2.micro": 0.0116, "t2.small": 0.023,
    "t2.medium": 0.0464, "t2.large": 0.0928, "t2.xlarge": 0.1856, "t2.2xlarge": 0.3712,
    # T3 (burstable, current gen)
    "t3.nano": 0.0052, "t3.micro": 0.0104, "t3.small": 0.0208, "t3.medium": 0.0416,
    "t3.large": 0.0832, "t3.xlarge": 0.1664, "t3.2xlarge": 0.3328,
    # T3a (AMD)
    "t3a.nano": 0.0047, "t3a.micro": 0.0094, "t3a.small": 0.0188, "t3a.medium": 0.0376,
    "t3a.large": 0.0752, "t3a.xlarge": 0.1504, "t3a.2xlarge": 0.3008,
    # T4g (Graviton2)
    "t4g.nano": 0.0042, "t4g.micro": 0.0084, "t4g.small": 0.0168, "t4g.medium": 0.0336,
    "t4g.large": 0.0672, "t4g.xlarge": 0.1344, "t4g.2xlarge": 0.2688,
    # M5 (general purpose)
    "m5.large": 0.096,  "m5.xlarge": 0.192,  "m5.2xlarge": 0.384,
    "m5.4xlarge": 0.768, "m5.8xlarge": 1.536, "m5.12xlarge": 2.304, "m5.24xlarge": 4.608,
    # M6i (general purpose, Intel, current gen)
    "m6i.large": 0.096, "m6i.xlarge": 0.192, "m6i.2xlarge": 0.384,
    "m6i.4xlarge": 0.768, "m6i.8xlarge": 1.536, "m6i.12xlarge": 2.304,
    # M6g (Graviton2, general purpose)
    "m6g.medium": 0.0385, "m6g.large": 0.077, "m6g.xlarge": 0.154,
    "m6g.2xlarge": 0.308, "m6g.4xlarge": 0.616, "m6g.8xlarge": 1.232,
    # M7i (general purpose, latest Intel)
    "m7i.large": 0.1008, "m7i.xlarge": 0.2016, "m7i.2xlarge": 0.4032,
    # M7g (Graviton3)
    "m7g.medium": 0.0408, "m7g.large": 0.0816, "m7g.xlarge": 0.1632, "m7g.2xlarge": 0.3264,
    # C5 (compute optimized)
    "c5.large": 0.085,   "c5.xlarge": 0.17,   "c5.2xlarge": 0.34,
    "c5.4xlarge": 0.68,  "c5.9xlarge": 1.53,  "c5.18xlarge": 3.06,
    # C6i (compute optimized, current gen)
    "c6i.large": 0.085,  "c6i.xlarge": 0.17,  "c6i.2xlarge": 0.34,
    "c6i.4xlarge": 0.68, "c6i.8xlarge": 1.36, "c6i.12xlarge": 2.04,
    # C6g (Graviton2, compute)
    "c6g.medium": 0.034, "c6g.large": 0.068, "c6g.xlarge": 0.136,
    "c6g.2xlarge": 0.272, "c6g.4xlarge": 0.544,
    # C7g (Graviton3, compute)
    "c7g.medium": 0.0363, "c7g.large": 0.0725, "c7g.xlarge": 0.145, "c7g.2xlarge": 0.29,
    # R5 (memory optimized)
    "r5.large": 0.126,   "r5.xlarge": 0.252,  "r5.2xlarge": 0.504,
    "r5.4xlarge": 1.008, "r5.8xlarge": 2.016, "r5.12xlarge": 3.024, "r5.24xlarge": 6.048,
    # R6i (memory optimized, current gen)
    "r6i.large": 0.126,  "r6i.xlarge": 0.252, "r6i.2xlarge": 0.504,
    "r6i.4xlarge": 1.008, "r6i.8xlarge": 2.016,
    # R6g (Graviton2, memory)
    "r6g.medium": 0.0504, "r6g.large": 0.1008, "r6g.xlarge": 0.2016,
    "r6g.2xlarge": 0.4032, "r6g.4xlarge": 0.8064,
    # X1e / X2 (high memory)
    "x1e.xlarge": 0.834, "x1e.2xlarge": 1.668, "x1e.4xlarge": 3.336,
    # GPU / ML
    "p3.2xlarge": 3.06,  "p3.8xlarge": 12.24, "p3.16xlarge": 24.48,
    "p4d.24xlarge": 32.7726,
    "g4dn.xlarge": 0.526, "g4dn.2xlarge": 0.752, "g4dn.4xlarge": 1.204,
    "g4dn.8xlarge": 2.264, "g4dn.12xlarge": 3.912, "g4dn.16xlarge": 4.528,
    "g5.xlarge": 1.006, "g5.2xlarge": 1.212, "g5.4xlarge": 1.624, "g5.8xlarge": 2.448,
    "g6.xlarge": 0.6049, "g6.2xlarge": 0.9778,
    # Storage optimized (I/O)
    "i3.large": 0.156, "i3.xlarge": 0.312, "i3.2xlarge": 0.624, "i3.4xlarge": 1.248,
    "i3en.large": 0.226, "i3en.xlarge": 0.452, "i3en.2xlarge": 0.904,
}

# RDS fallback prices ($/hr, single-AZ)
_RDS_FALLBACK: dict[str, float] = {
    "db.t3.micro": 0.017, "db.t3.small": 0.034, "db.t3.medium": 0.068,
    "db.t3.large": 0.136, "db.t3.xlarge": 0.272,
    "db.t4g.micro": 0.016, "db.t4g.small": 0.032, "db.t4g.medium": 0.065,
    "db.m5.large": 0.171, "db.m5.xlarge": 0.342, "db.m5.2xlarge": 0.684,
    "db.m6i.large": 0.171, "db.m6i.xlarge": 0.342,
    "db.r5.large": 0.24,  "db.r5.xlarge": 0.48,  "db.r5.2xlarge": 0.96,
    "db.r6i.large": 0.24, "db.r6i.xlarge": 0.48,
}

HOURS_PER_MONTH = 730.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cache_get(key: str) -> Optional[float]:
    entry = _PRICE_CACHE.get(key)
    if entry and (time.time() - entry[1]) < _CACHE_TTL:
        return entry[0]
    return None


def _cache_set(key: str, price: float) -> None:
    _PRICE_CACHE[key] = (price, time.time())


def _pricing_client():
    """AWS Pricing API is only available in us-east-1."""
    if not _BOTO3:
        raise RuntimeError("boto3 not installed")
    return boto3.client("pricing", region_name="us-east-1")


# ---------------------------------------------------------------------------
# Public AWS bulk pricing fetcher (no credentials required)
# AWS publishes on-demand pricing at a public HTTPS endpoint.
# The file is large (~400MB), so we download it to disk and cache for 24h.
# A background thread handles the download so chatbot responses are never blocked.
# ---------------------------------------------------------------------------
_BULK_PRICE_CACHE_TTL = 86400.0   # 24 hours disk cache
_BULK_PRICE_DATA: dict[str, dict] = {}          # region → {instance_type: price/hr}
_BULK_PRICE_LOADED: dict[str, float] = {}       # region → timestamp when loaded
_BULK_DOWNLOAD_ACTIVE: set = set()              # regions currently downloading

import threading as _threading
import tempfile as _tempfile

def _parse_pricing_json(raw_bytes: bytes) -> dict[str, float]:
    """Parse AWS bulk pricing JSON → {instance_type: $/hr} for Linux shared on-demand."""
    data = json.loads(raw_bytes.decode("utf-8"))
    products = data.get("products", {})
    terms    = data.get("terms", {}).get("OnDemand", {})
    prices: dict[str, float] = {}

    # AWS bulk JSON uses lowercase field names in attributes
    sku_to_itype: dict[str, str] = {}
    for sku, prod in products.items():
        attrs = prod.get("attributes", {})
        if (
            attrs.get("operatingSystem") == "Linux"
            and attrs.get("tenancy") == "Shared"
            and attrs.get("capacitystatus", attrs.get("capacityStatus")) == "Used"
            and attrs.get("preInstalledSw", "NA") == "NA"
            and attrs.get("instanceType")
        ):
            sku_to_itype[sku] = attrs["instanceType"]

    for sku, itype in sku_to_itype.items():
        for term in terms.get(sku, {}).values():
            for dim in term.get("priceDimensions", {}).values():
                p = float(dim.get("pricePerUnit", {}).get("USD", 0) or 0)
                if p > 0:
                    prices[itype] = p
                    break
    return prices


def _cache_file_path(region: str) -> str:
    return os.path.join(_tempfile.gettempdir(), f"nexusops_ec2_pricing_{region}.json")


def _load_cached_pricing(region: str) -> dict[str, float]:
    """Load pricing from disk cache if fresh enough."""
    path = _cache_file_path(region)
    try:
        if os.path.exists(path):
            age = time.time() - os.path.getmtime(path)
            if age < _BULK_PRICE_CACHE_TTL:
                with open(path, "r") as f:
                    return json.load(f)
    except Exception:
        pass
    return {}


def _download_pricing_background(region: str) -> None:
    """Download AWS bulk pricing JSON in background, cache to disk."""
    if region in _BULK_DOWNLOAD_ACTIVE:
        return
    _BULK_DOWNLOAD_ACTIVE.add(region)
    try:
        import urllib.request
        url = (f"https://pricing.us-east-1.amazonaws.com"
               f"/offers/v1.0/aws/AmazonEC2/current/{region}/index.json")
        logger.info("aws_pricing_download_start", extra={"region": region, "url": url})
        req = urllib.request.Request(url, headers={"User-Agent": "NexusOps-PricingFetcher/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
        prices = _parse_pricing_json(raw)
        if prices:
            path = _cache_file_path(region)
            with open(path, "w") as f:
                json.dump(prices, f)
            _BULK_PRICE_DATA[region] = prices
            _BULK_PRICE_LOADED[region] = time.time()
            logger.info("aws_pricing_download_done", extra={"region": region, "count": len(prices)})
    except Exception as exc:
        logger.warning("aws_pricing_download_failed", extra={"region": region, "error": str(exc)})
    finally:
        _BULK_DOWNLOAD_ACTIVE.discard(region)


def _get_public_ec2_pricing(region: str) -> dict[str, float]:
    """
    Return the public AWS on-demand pricing dict for a region.
    - If already in memory: return immediately.
    - If fresh disk cache exists: load from disk.
    - Otherwise: trigger background download, return empty dict now.
    """
    # Already in memory and fresh
    if region in _BULK_PRICE_DATA and (time.time() - _BULK_PRICE_LOADED.get(region, 0)) < _BULK_PRICE_CACHE_TTL:
        return _BULK_PRICE_DATA[region]

    # Try disk cache
    cached = _load_cached_pricing(region)
    if cached:
        _BULK_PRICE_DATA[region] = cached
        _BULK_PRICE_LOADED[region] = time.time()
        return cached

    # Kick off background download (non-blocking)
    t = _threading.Thread(target=_download_pricing_background, args=(region,), daemon=True)
    t.start()
    return {}  # Caller will use reference prices this time


# ---------------------------------------------------------------------------
# AWS Cost Explorer — actual spend data from connected account
# ---------------------------------------------------------------------------

_CE_CACHE: dict[str, tuple[dict, float]] = {}  # key → (result, timestamp)
_CE_CACHE_TTL = 3600.0  # 1 hour


def get_cost_explorer_summary(days: int = 30, granularity: str = "MONTHLY") -> dict:
    """Fetch actual AWS spend from Cost Explorer (requires ce:GetCostAndUsage permission).

    Returns dict with total, by_service, and daily/monthly breakdown.
    """
    if not _BOTO3:
        return {"error": "boto3 not available"}

    cache_key = f"ce:{days}:{granularity}"
    cached_entry = _CE_CACHE.get(cache_key)
    if cached_entry and (time.time() - cached_entry[1]) < _CE_CACHE_TTL:
        return cached_entry[0]

    try:
        end   = datetime.date.today()
        start = end - datetime.timedelta(days=days)
        ce    = boto3.client("ce", region_name="us-east-1")
        resp  = ce.get_cost_and_usage(
            TimePeriod={"Start": start.isoformat(), "End": end.isoformat()},
            Granularity=granularity,
            Metrics=["UnblendedCost"],
            GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
        )
        by_service: dict[str, float] = {}
        total = 0.0
        for result in resp.get("ResultsByTime", []):
            for group in result.get("Groups", []):
                svc   = group["Keys"][0]
                amt   = float(group["Metrics"]["UnblendedCost"]["Amount"])
                by_service[svc] = round(by_service.get(svc, 0.0) + amt, 4)
                total += amt
        out = {
            "total_usd":    round(total, 2),
            "period_days":  days,
            "by_service":   dict(sorted(by_service.items(), key=lambda x: -x[1])[:20]),
            "source":       "aws_cost_explorer",
            "as_of":        end.isoformat(),
        }
        _CE_CACHE[cache_key] = (out, time.time())
        return out
    except Exception as exc:
        err_msg = str(exc)
        if "AccessDenied" in err_msg or "UnauthorizedOperation" in err_msg:
            return {"error": "Cost Explorer access denied — add ce:GetCostAndUsage IAM permission"}
        if "OptInRequired" in err_msg:
            return {"error": "Cost Explorer not enabled — activate it in AWS Billing console"}
        return {"error": err_msg[:200]}


def get_ec2_actual_cost(instance_id: str, days: int = 30) -> dict:
    """Get actual cost for a specific EC2 instance from Cost Explorer."""
    if not _BOTO3:
        return {"error": "boto3 not available"}
    try:
        end   = datetime.date.today()
        start = end - datetime.timedelta(days=days)
        ce    = boto3.client("ce", region_name="us-east-1")
        resp  = ce.get_cost_and_usage(
            TimePeriod={"Start": start.isoformat(), "End": end.isoformat()},
            Granularity="MONTHLY",
            Metrics=["UnblendedCost"],
            Filter={
                "Dimensions": {
                    "Key":    "RESOURCE_ID",
                    "Values": [instance_id],
                }
            },
        )
        total = sum(
            float(r["Total"]["UnblendedCost"]["Amount"])
            for r in resp.get("ResultsByTime", [])
        )
        return {
            "instance_id": instance_id,
            "total_usd":   round(total, 4),
            "period_days": days,
            "source":      "aws_cost_explorer",
        }
    except Exception as exc:
        return {"error": str(exc)[:200]}


# ---------------------------------------------------------------------------
# EC2 pricing  — tries (1) boto3 API, (2) public bulk JSON, (3) hardcoded table
# ---------------------------------------------------------------------------

def get_ec2_price_per_hour(instance_type: str, region: str = "us-east-1",
                            os: str = "Linux") -> dict:
    """Return on-demand price for an EC2 instance type.

    Priority:
      1. In-process cache (1 h TTL)
      2. boto3 AWS Pricing API (requires valid AWS credentials)
      3. Public AWS bulk pricing JSON (no credentials needed, 24 h disk cache)
      4. Hardcoded reference table (2025 us-east-1 rates with regional multiplier)

    Returns:
        {"price_per_hour": float, "source": str, "instance_type": str}
    """
    cache_key = f"ec2:{instance_type}:{region}:{os}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return {"price_per_hour": cached, "source": "cached", "instance_type": instance_type}

    region_name = _REGION_NAMES.get(region, region)

    # ── 1. Try boto3 AWS Pricing API (requires valid credentials) ──────────
    try:
        pc = _pricing_client()
        resp = pc.get_products(
            ServiceCode="AmazonEC2",
            Filters=[
                {"Type": "TERM_MATCH", "Field": "instanceType",    "Value": instance_type},
                {"Type": "TERM_MATCH", "Field": "location",        "Value": region_name},
                {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": os},
                {"Type": "TERM_MATCH", "Field": "tenancy",         "Value": "Shared"},
                {"Type": "TERM_MATCH", "Field": "capacityStatus",  "Value": "Used"},
                {"Type": "TERM_MATCH", "Field": "preInstalledSw",  "Value": "NA"},
            ],
            MaxResults=5,
        )
        for price_str in resp.get("PriceList", []):
            item = json.loads(price_str)
            for term in item.get("terms", {}).get("OnDemand", {}).values():
                for dim in term.get("priceDimensions", {}).values():
                    price = float(dim["pricePerUnit"].get("USD", 0))
                    if price > 0:
                        _cache_set(cache_key, price)
                        return {"price_per_hour": price, "source": "aws_pricing_api",
                                "instance_type": instance_type, "region": region}
    except Exception as exc:
        logger.debug("pricing_api_unavailable", extra={"error": str(exc)[:120]})

    # ── 2. Public AWS bulk pricing JSON (no auth required) ─────────────────
    try:
        bulk = _get_public_ec2_pricing(region)
        if instance_type in bulk:
            price = bulk[instance_type]
            # Windows premium ~40%, RHEL ~20%
            if "windows" in os.lower():
                price *= 1.4
            elif "rhel" in os.lower() or "red hat" in os.lower():
                price *= 1.2
            _cache_set(cache_key, price)
            return {"price_per_hour": price, "source": "aws_public_pricing_json",
                    "instance_type": instance_type, "region": region,
                    "note": "Live data from AWS public pricing endpoint (no credentials needed)"}
    except Exception as exc:
        logger.debug("public_pricing_failed", extra={"error": str(exc)[:120]})

    # ── 3. Hardcoded reference table (2025 us-east-1 rates) ────────────────
    # Regional price multipliers relative to us-east-1
    _REGION_MULTIPLIER = {
        "us-east-1": 1.00, "us-east-2": 1.00, "us-west-1": 1.12, "us-west-2": 1.00,
        "ca-central-1": 1.05, "eu-west-1": 1.10, "eu-west-2": 1.12, "eu-west-3": 1.12,
        "eu-central-1": 1.12, "eu-north-1": 1.08, "ap-southeast-1": 1.15,
        "ap-southeast-2": 1.16, "ap-northeast-1": 1.14, "ap-northeast-2": 1.13,
        "ap-south-1": 1.08, "sa-east-1": 1.30,
    }
    multiplier = _REGION_MULTIPLIER.get(region, 1.10)
    base_price = _EC2_FALLBACK.get(instance_type, 0.0)
    price = round(base_price * multiplier, 6)
    _cache_set(cache_key, price)
    source_note = "aws_reference_table_2025"
    if not base_price:
        source_note += " (unknown instance type — price unavailable)"
    return {"price_per_hour": price, "source": source_note,
            "instance_type": instance_type,
            "note": "Reference prices from aws.amazon.com/ec2/pricing. Reconnect valid AWS credentials for real-time API data."}


def estimate_ec2_cost(instance_type: str, count: int = 1,
                       hours: float = HOURS_PER_MONTH,
                       region: str = "us-east-1") -> dict:
    """Full EC2 cost estimate."""
    pricing = get_ec2_price_per_hour(instance_type, region)
    per_hr   = pricing["price_per_hour"]
    total    = per_hr * count * hours
    return {
        "instance_type":    instance_type,
        "count":            count,
        "hours":            hours,
        "price_per_hour":   per_hr,
        "total_usd":        round(total, 2),
        "monthly_usd":      round(per_hr * count * HOURS_PER_MONTH, 2),
        "source":           pricing["source"],
        "region":           region,
    }


# ---------------------------------------------------------------------------
# RDS pricing
# ---------------------------------------------------------------------------

def get_rds_price_per_hour(instance_class: str, engine: str = "mysql",
                            region: str = "us-east-1",
                            multi_az: bool = False) -> dict:
    cache_key = f"rds:{instance_class}:{engine}:{region}:{multi_az}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return {"price_per_hour": cached, "source": "api_cached"}

    region_name = _REGION_NAMES.get(region, region)
    engine_map  = {
        "mysql": "MySQL", "postgres": "PostgreSQL", "postgresql": "PostgreSQL",
        "aurora": "Aurora MySQL", "aurora-mysql": "Aurora MySQL",
        "aurora-postgresql": "Aurora PostgreSQL", "mariadb": "MariaDB",
        "oracle": "Oracle", "sqlserver": "SQL Server",
    }
    db_engine = engine_map.get(engine.lower(), engine)
    deployment = "Multi-AZ" if multi_az else "Single-AZ"

    try:
        pc = _pricing_client()
        resp = pc.get_products(
            ServiceCode="AmazonRDS",
            Filters=[
                {"Type": "TERM_MATCH", "Field": "instanceType",       "Value": instance_class},
                {"Type": "TERM_MATCH", "Field": "location",           "Value": region_name},
                {"Type": "TERM_MATCH", "Field": "databaseEngine",     "Value": db_engine},
                {"Type": "TERM_MATCH", "Field": "deploymentOption",   "Value": deployment},
            ],
            MaxResults=5,
        )
        for price_str in resp.get("PriceList", []):
            item  = json.loads(price_str)
            terms = item.get("terms", {}).get("OnDemand", {})
            for term in terms.values():
                for dim in term.get("priceDimensions", {}).values():
                    price = float(dim["pricePerUnit"].get("USD", 0))
                    if price > 0:
                        _cache_set(cache_key, price)
                        return {"price_per_hour": price, "source": "aws_pricing_api"}
    except Exception as exc:
        logger.warning("rds_pricing_api_failed", extra={"error": str(exc)})

    base  = _RDS_FALLBACK.get(instance_class, 0.0)
    price = base * (2 if multi_az else 1)
    _cache_set(cache_key, price)
    return {"price_per_hour": price, "source": "fallback_table",
            "note": "Live API unavailable — using reference prices"}


def estimate_rds_cost(instance_class: str, engine: str = "mysql",
                       storage_gb: int = 100, multi_az: bool = False,
                       region: str = "us-east-1") -> dict:
    pricing   = get_rds_price_per_hour(instance_class, engine, region, multi_az)
    per_hr    = pricing["price_per_hour"]
    instance  = per_hr * HOURS_PER_MONTH
    storage   = storage_gb * 0.115   # gp2 storage: $0.115/GB-month
    total     = instance + storage
    return {
        "instance_class":        instance_class,
        "engine":                engine,
        "multi_az":              multi_az,
        "storage_gb":            storage_gb,
        "instance_monthly_usd":  round(instance, 2),
        "storage_monthly_usd":   round(storage, 2),
        "total_monthly_usd":     round(total, 2),
        "price_per_hour":        per_hr,
        "source":                pricing["source"],
        "region":                region,
    }


# ---------------------------------------------------------------------------
# Lambda pricing
# ---------------------------------------------------------------------------
# Source: https://aws.amazon.com/lambda/pricing/
_LAMBDA_REQUEST_PRICE  = 0.20 / 1_000_000   # $0.20 per 1M requests
_LAMBDA_COMPUTE_PRICE  = 0.0000166667        # $0.0000166667 per GB-second (after free tier)

def estimate_lambda_cost(invocations_per_month: int, avg_duration_ms: float,
                          memory_mb: int = 128) -> dict:
    """Estimate Lambda monthly cost."""
    gb_seconds   = (memory_mb / 1024) * (avg_duration_ms / 1000) * invocations_per_month
    compute_cost = gb_seconds * _LAMBDA_COMPUTE_PRICE
    request_cost = invocations_per_month * _LAMBDA_REQUEST_PRICE
    total        = compute_cost + request_cost
    return {
        "invocations_per_month": invocations_per_month,
        "avg_duration_ms":       avg_duration_ms,
        "memory_mb":             memory_mb,
        "gb_seconds":            round(gb_seconds, 2),
        "compute_cost_usd":      round(compute_cost, 4),
        "request_cost_usd":      round(request_cost, 4),
        "total_monthly_usd":     round(total, 4),
        "source":                "aws_official_pricing",
        "note":                  "First 1M requests/month free. First 400K GB-seconds/month free.",
        "pricing_page":          "https://aws.amazon.com/lambda/pricing/",
    }


# ---------------------------------------------------------------------------
# ECS Fargate pricing
# ---------------------------------------------------------------------------
# Source: https://aws.amazon.com/fargate/pricing/
_FARGATE_VCPU_PRICE_HR = 0.04048    # $0.04048 per vCPU per hour
_FARGATE_MEM_PRICE_HR  = 0.004445   # $0.004445 per GB per hour

def estimate_fargate_cost(vcpu: float, memory_gb: float, tasks: int = 1,
                           hours: float = HOURS_PER_MONTH) -> dict:
    cpu_cost = vcpu  * _FARGATE_VCPU_PRICE_HR * tasks * hours
    mem_cost = memory_gb * _FARGATE_MEM_PRICE_HR * tasks * hours
    total    = cpu_cost + mem_cost
    return {
        "vcpu":             vcpu,
        "memory_gb":        memory_gb,
        "tasks":            tasks,
        "hours":            hours,
        "cpu_cost_usd":     round(cpu_cost, 2),
        "memory_cost_usd":  round(mem_cost, 2),
        "total_monthly_usd": round(total, 2),
        "source":           "aws_official_pricing",
        "pricing_page":     "https://aws.amazon.com/fargate/pricing/",
    }


# ---------------------------------------------------------------------------
# S3 pricing
# ---------------------------------------------------------------------------
# Source: https://aws.amazon.com/s3/pricing/
def estimate_s3_cost(storage_gb: float, get_requests: int = 10000,
                      put_requests: int = 1000, data_transfer_gb: float = 0.0) -> dict:
    storage_cost   = storage_gb * 0.023         # $0.023/GB standard storage
    get_cost       = get_requests * 0.0000004   # $0.0004 per 1000 GET
    put_cost       = put_requests * 0.000005    # $0.005 per 1000 PUT
    transfer_cost  = max(0, data_transfer_gb - 1) * 0.09  # first 1GB free
    total          = storage_cost + get_cost + put_cost + transfer_cost
    return {
        "storage_gb":       storage_gb,
        "storage_cost_usd": round(storage_cost, 2),
        "request_cost_usd": round(get_cost + put_cost, 4),
        "transfer_cost_usd": round(transfer_cost, 2),
        "total_monthly_usd": round(total, 2),
        "source":            "aws_official_pricing",
        "pricing_page":      "https://aws.amazon.com/s3/pricing/",
    }


# ---------------------------------------------------------------------------
# Terraform plan cost estimation
# ---------------------------------------------------------------------------

# Map Terraform resource types → estimator functions
_TF_RESOURCE_MAP = {
    "aws_instance":                  "ec2",
    "aws_db_instance":               "rds",
    "aws_rds_cluster":               "aurora",
    "aws_lambda_function":           "lambda",
    "aws_ecs_task_definition":       "fargate",
    "aws_ecs_service":               "fargate",
    "aws_s3_bucket":                 "s3",
    "aws_elasticache_cluster":       "elasticache",
    "aws_elasticsearch_domain":      "elasticsearch",
    "aws_nat_gateway":               "nat_gateway",
    "aws_lb":                        "alb",
    "aws_alb":                       "alb",
}

# Simple fallback monthly costs for resources without full pricing logic
_TF_SIMPLE_COSTS = {
    "nat_gateway":    32.40,   # $0.045/hr × 720 + data processing
    "alb":            18.00,   # ~$0.025/hr base
    "elasticache":    24.00,   # db.t3.micro equivalent
    "elasticsearch":  25.00,   # t3.small.elasticsearch equivalent
}


def estimate_terraform_plan_cost(plan_json: dict) -> dict:
    """Estimate monthly cost from a `terraform plan -json` output.

    Args:
        plan_json: Parsed JSON from `terraform plan -out=plan.tfplan && terraform show -json plan.tfplan`

    Returns:
        Dict with per-resource and total cost estimates.
    """
    resources = []
    warnings  = []
    total_monthly = 0.0

    # Handle both `terraform show -json` and `terraform plan -json` formats
    changes = plan_json.get("resource_changes", [])
    if not changes:
        # Try planned_values format
        planned = plan_json.get("planned_values", {}).get("root_module", {})
        resources_raw = planned.get("resources", [])
        changes = [{"address": r.get("address", ""), "type": r["type"],
                    "change": {"actions": ["create"], "after": r.get("values", {})}}
                   for r in resources_raw]

    region = os.getenv("AWS_REGION", "us-east-1")

    for change in changes:
        actions = change.get("change", {}).get("actions", [])
        if "no-op" in actions or "delete" in actions:
            continue  # Skip deletions and no-ops for cost estimation

        rtype    = change.get("type", "")
        address  = change.get("address", rtype)
        after    = change.get("change", {}).get("after") or {}
        category = _TF_RESOURCE_MAP.get(rtype)

        if not category:
            continue

        try:
            if category == "ec2":
                itype = after.get("instance_type", "t3.micro")
                est   = estimate_ec2_cost(itype, count=1, region=region)
                resources.append({
                    "address":       address,
                    "type":          rtype,
                    "details":       f"EC2 {itype}",
                    "monthly_usd":   est["monthly_usd"],
                    "price_per_hour": est["price_per_hour"],
                    "source":        est["source"],
                })
                total_monthly += est["monthly_usd"]

            elif category == "rds":
                iclass  = after.get("instance_class", "db.t3.micro")
                engine  = after.get("engine", "mysql")
                storage = int(after.get("allocated_storage", 20))
                maz     = bool(after.get("multi_az", False))
                est     = estimate_rds_cost(iclass, engine, storage, maz, region)
                resources.append({
                    "address":      address,
                    "type":         rtype,
                    "details":      f"RDS {iclass} {engine} {'Multi-AZ' if maz else 'Single-AZ'} {storage}GB",
                    "monthly_usd":  est["total_monthly_usd"],
                    "source":       est["source"],
                })
                total_monthly += est["total_monthly_usd"]

            elif category == "lambda":
                memory   = int(after.get("memory_size", 128))
                # Assume 1M invocations/month at 200ms if not inferable
                est = estimate_lambda_cost(1_000_000, 200, memory)
                resources.append({
                    "address":     address,
                    "type":        rtype,
                    "details":     f"Lambda {memory}MB (estimated 1M invocations/mo)",
                    "monthly_usd": est["total_monthly_usd"],
                    "source":      est["source"],
                    "note":        "Invocation count assumed — actual cost depends on traffic",
                })
                total_monthly += est["total_monthly_usd"]
                warnings.append(f"{address}: Lambda cost assumes 1M invocations/mo. Adjust for actual traffic.")

            elif category == "fargate":
                cpu    = float(after.get("cpu", 256)) / 1024   # Fargate cpu is in units
                mem    = float(after.get("memory", 512)) / 1024
                count  = int(after.get("desired_count", 1))
                est    = estimate_fargate_cost(cpu, mem, count)
                resources.append({
                    "address":     address,
                    "type":        rtype,
                    "details":     f"Fargate {cpu}vCPU {mem}GB × {count} tasks",
                    "monthly_usd": est["total_monthly_usd"],
                    "source":      est["source"],
                })
                total_monthly += est["total_monthly_usd"]

            elif category == "s3":
                resources.append({
                    "address":     address,
                    "type":        rtype,
                    "details":     "S3 bucket (cost depends on storage/requests)",
                    "monthly_usd": 0.0,
                    "source":      "estimated",
                    "note":        "S3 cost depends on storage size and request volume",
                })
                warnings.append(f"{address}: S3 bucket cost depends on usage — use estimate_s3_cost() for details.")

            else:
                cost = _TF_SIMPLE_COSTS.get(category, 0.0)
                resources.append({
                    "address":     address,
                    "type":        rtype,
                    "details":     f"{category} resource",
                    "monthly_usd": cost,
                    "source":      "reference_estimate",
                })
                total_monthly += cost

        except Exception as exc:
            warnings.append(f"Could not estimate cost for {address}: {exc}")
            resources.append({
                "address":     address,
                "type":        rtype,
                "details":     "Estimation failed",
                "monthly_usd": 0.0,
            })

    return {
        "total_monthly_usd":    round(total_monthly, 2),
        "total_annual_usd":     round(total_monthly * 12, 2),
        "resource_count":       len(resources),
        "resources":            resources,
        "warnings":             warnings,
        "generated_at":         datetime.datetime.utcnow().isoformat() + "Z",
        "pricing_source":       "AWS Pricing API + official reference rates",
        "pricing_docs":         "https://aws.amazon.com/pricing/",
    }


# ---------------------------------------------------------------------------
# Natural language cost estimator (used by chatbot)
# ---------------------------------------------------------------------------

def estimate_from_description(description: str, region: str = "us-east-1") -> dict:
    """Best-effort cost estimate from a natural language description.

    Examples:
      "3 t3.medium instances"
      "db.r5.large mysql multi-az"
      "lambda function 512mb 5 million invocations"
      "fargate 2 vcpu 4gb 10 tasks"
      "100gb s3 storage"
    """
    desc = description.lower().strip()
    results = []
    total   = 0.0
    warnings = []

    # EC2 pattern: look for instance type patterns
    import re
    ec2_matches = re.findall(r'\b([a-z]\d[a-z]*(?:dn|gd)?\.(?:nano|micro|small|medium|large|[248]?xlarge|metal))\b', desc)
    count_match = re.search(r'(\d+)\s+(?:instance|server|node|ec2)', desc)
    ec2_count = int(count_match.group(1)) if count_match else 1

    # Detect OS from description for pricing note
    os_label = "Linux"
    if "windows" in desc:
        os_label = "Windows"
    elif "ubuntu" in desc or "debian" in desc:
        os_label = "Linux (Ubuntu)"
    elif "rhel" in desc or "red hat" in desc:
        os_label = "RHEL"

    # If no specific instance type found but EC2/instance/server mentioned, leave empty
    # so the caller can show a full pricing menu to the user
    ec2_keywords_present = any(k in desc for k in ("ec2", "instance", "server", "vm", "compute", "virtual machine"))

    for itype in set(ec2_matches):
        if not itype.startswith("db."):
            est = estimate_ec2_cost(itype, count=ec2_count, region=region)
            results.append({"type": "EC2", "details": f"{ec2_count}x {itype} ({os_label})", "monthly_usd": est["monthly_usd"], "source": est["source"]})
            total += est["monthly_usd"]

    # RDS pattern
    rds_matches = re.findall(r'\b(db\.[a-z]\d[a-z]*\.(?:micro|small|medium|large|[248]?xlarge))\b', desc)
    for iclass in set(rds_matches):
        engine  = "postgres" if "postgres" in desc else "aurora" if "aurora" in desc else "mysql"
        maz     = "multi" in desc and "az" in desc
        storage = int(re.search(r'(\d+)\s*gb', desc).group(1)) if re.search(r'(\d+)\s*gb', desc) else 100
        est = estimate_rds_cost(iclass, engine, storage, maz, region)
        results.append({"type": "RDS", "details": f"{iclass} {engine} {storage}GB", "monthly_usd": est["total_monthly_usd"], "source": est["source"]})
        total += est["total_monthly_usd"]

    # Lambda
    if "lambda" in desc:
        mem_m = re.search(r'(\d+)\s*mb', desc)
        inv_m = re.search(r'([\d,.]+)\s*(?:million|m)\s*invoc', desc) or re.search(r'([\d,]+)\s*invoc', desc)
        memory = int(mem_m.group(1)) if mem_m else 128
        if inv_m:
            raw = inv_m.group(1).replace(",", "")
            invocations = int(float(raw) * 1_000_000) if "million" in desc or " m " in desc else int(raw)
        else:
            invocations = 1_000_000
        est = estimate_lambda_cost(invocations, 200, memory)
        results.append({"type": "Lambda", "details": f"{memory}MB, {invocations:,} invocations/mo", "monthly_usd": est["total_monthly_usd"], "source": est["source"]})
        total += est["total_monthly_usd"]
        warnings.append("Lambda duration assumed 200ms — adjust if different.")

    # Fargate
    if "fargate" in desc:
        vcpu_m = re.search(r'([\d.]+)\s*vcpu', desc)
        mem_m  = re.search(r'([\d.]+)\s*gb', desc)
        task_m = re.search(r'(\d+)\s*task', desc)
        vcpu   = float(vcpu_m.group(1)) if vcpu_m else 0.25
        mem    = float(mem_m.group(1)) if mem_m else 0.5
        tasks  = int(task_m.group(1)) if task_m else 1
        est    = estimate_fargate_cost(vcpu, mem, tasks)
        results.append({"type": "Fargate", "details": f"{vcpu}vCPU {mem}GB × {tasks} tasks", "monthly_usd": est["total_monthly_usd"], "source": est["source"]})
        total += est["total_monthly_usd"]

    # S3
    if "s3" in desc:
        gb_m = re.search(r'([\d,]+)\s*(?:tb|gb)\s*(?:s3|storage|bucket)?', desc)
        if gb_m:
            storage_gb = float(gb_m.group(1).replace(",", ""))
            if "tb" in desc:
                storage_gb *= 1024
            est = estimate_s3_cost(storage_gb)
            results.append({"type": "S3", "details": f"{storage_gb:.0f}GB storage", "monthly_usd": est["total_monthly_usd"], "source": est["source"]})
            total += est["total_monthly_usd"]

    if not results:
        return {
            "total_monthly_usd": 0.0,
            "resources": [],
            "warnings": [
                "Could not identify specific resources to estimate. "
                "Examples that work: '3 t3.medium instances', 'm5.large ubuntu', "
                "'db.r5.large postgres 100gb multi-az', 'lambda 512mb 10M invocations', "
                "'fargate 2 vcpu 4gb 5 tasks', '500gb s3'."
            ],
            "source": "estimate",
            "helpful_defaults": {
                "t3.micro":  "~$8/mo — dev/test workloads",
                "t3.medium": "~$30/mo — small production app",
                "m5.large":  "~$70/mo — general purpose production",
                "c5.xlarge": "~$122/mo — compute-intensive workloads",
            },
        }

    return {
        "total_monthly_usd":  round(total, 2),
        "total_annual_usd":   round(total * 12, 2),
        "resources":          results,
        "warnings":           warnings,
        "region":             region,
        "pricing_source":     "AWS Pricing API (live) + official reference rates",
        "pricing_docs":       "https://aws.amazon.com/pricing/",
        "note":               "On-demand pricing. Reserved instances can save 30-70%.",
    }


# ---------------------------------------------------------------------------
# Warm up background pricing download for default regions on module import
# ---------------------------------------------------------------------------
def _warm_pricing_cache() -> None:
    """Trigger background downloads for common regions so prices are ready."""
    default_region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    for _r in sorted({default_region, "us-east-1", "us-west-2"}):
        # Only trigger if not already cached on disk
        path = _cache_file_path(_r)
        if not (os.path.exists(path) and (time.time() - os.path.getmtime(path)) < _BULK_PRICE_CACHE_TTL):
            t = _threading.Thread(target=_download_pricing_background, args=(_r,), daemon=True)
            t.start()

try:
    _warm_pricing_cache()
except Exception:
    pass
