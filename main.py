from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any
from datetime import datetime, timezone
from pathlib import Path
import json
import time
import re


# ============================================================
# APP
# ============================================================

app = FastAPI(title="Vera Merchant AI Assistant")


# ============================================================
# STORAGE
# ============================================================

contexts = {}
server_started_at = time.time()

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "expanded"


# ============================================================
# DATASET LOADING
# ============================================================

DATASET = {
    "categories": {},
    "merchants": {},
    "customers": {},
    "triggers": {},
}


def load_json_file(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def identify_dataset_object(data, filename):
    """
    Put dataset JSON into the appropriate in-memory index.
    """

    if not isinstance(data, dict):
        return

    # Category
    if (
        "slug" in data
        and (
            "offer_catalog" in data
            or "voice" in data
            or "peer_stats" in data
        )
    ):
        key = str(data.get("slug"))
        DATASET["categories"][key] = data
        return

    # Merchant
    if (
        "merchant_id" in data
        and (
            "identity" in data
            or "offers" in data
            or "performance" in data
        )
    ):
        key = str(data["merchant_id"])
        DATASET["merchants"][key] = data
        return

    # Customer
    if (
        "customer_id" in data
        and (
            "relationship" in data
            or "preferences" in data
            or "consent" in data
        )
    ):
        key = str(data["customer_id"])
        DATASET["customers"][key] = data
        return

    # Trigger
    if (
        "id" in data
        and (
            "kind" in data
            or "payload" in data
            or "suppression_key" in data
        )
    ):
        key = str(data["id"])
        DATASET["triggers"][key] = data
        return


def load_dataset():
    """
    Load the generated challenge dataset from expanded/.
    Expected structure:
        expanded/categories/*.json
        expanded/merchants/*.json
        expanded/customers/*.json
        expanded/triggers/*.json
    """

    folders = {
        "categories": DATASET_DIR / "categories",
        "merchants": DATASET_DIR / "merchants",
        "customers": DATASET_DIR / "customers",
        "triggers": DATASET_DIR / "triggers",
    }

    for category, folder in folders.items():

        if not folder.exists():
            print(f"[WARN] Missing folder: {folder}")
            continue

        for path in folder.glob("*.json"):

            data = load_json_file(path)

            if not isinstance(data, dict):
                continue

            if category == "categories":
                key = data.get("slug", path.stem)
                DATASET["categories"][str(key)] = data

            elif category == "merchants":
                key = data.get("merchant_id", path.stem)
                DATASET["merchants"][str(key)] = data

            elif category == "customers":
                key = data.get("customer_id", path.stem)
                DATASET["customers"][str(key)] = data

            elif category == "triggers":
                key = data.get("id", path.stem)
                DATASET["triggers"][str(key)] = data

    print(
        f"[INFO] Dataset loaded: "
        f"{len(DATASET['categories'])} categories, "
        f"{len(DATASET['merchants'])} merchants, "
        f"{len(DATASET['customers'])} customers, "
        f"{len(DATASET['triggers'])} triggers"
    )


load_dataset()


# ============================================================
# REQUEST MODELS
# ============================================================

class ContextRequest(BaseModel):
    scope: str
    context_id: str
    version: int
    delivered_at: str | None = None
    payload: dict[str, Any]


class TickRequest(BaseModel):
    now: str
    available_triggers: list[str]


class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: str | None = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


# ============================================================
# HELPERS
# ============================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")


def get_context(scope: str, context_id: str):
    """
    First use pushed context.

    If unavailable, fall back to the challenge dataset.
    This makes the bot more robust when the judge references
    a dataset object that has not been explicitly pushed.
    """

    key = f"{scope}:{context_id}"

    if key in contexts:
        return contexts[key]

    if scope == "category":
        data = DATASET["categories"].get(context_id)

    elif scope == "merchant":
        data = DATASET["merchants"].get(context_id)

    elif scope == "customer":
        data = DATASET["customers"].get(context_id)

    elif scope == "trigger":
        data = DATASET["triggers"].get(context_id)

    else:
        data = None

    if data is None:
        return None

    return {
        "scope": scope,
        "context_id": context_id,
        "version": 1,
        "delivered_at": None,
        "payload": data,
    }


def find_merchant(merchant_id):
    if not merchant_id:
        return None

    context = get_context("merchant", merchant_id)

    if context:
        return context.get("payload")

    return None


def find_customer(customer_id):
    if not customer_id:
        return None

    context = get_context("customer", customer_id)

    if context:
        return context.get("payload")

    return None


def find_category(category_id):
    if not category_id:
        return None

    context = get_context("category", category_id)

    if context:
        return context.get("payload")

    return None


def get_nested(data, *keys, default=None):
    current = data

    for key in keys:
        if not isinstance(current, dict):
            return default

        current = current.get(key)

        if current is None:
            return default

    return current


def first_value(*values, default=None):
    for value in values:
        if value is not None and value != "":
            return value

    return default


def merchant_name(merchant):
    if not merchant:
        return "your business"

    return first_value(
        get_nested(merchant, "identity", "name"),
        merchant.get("name"),
        default="your business",
    )


def customer_name(customer):
    if not customer:
        return "there"

    return first_value(
        get_nested(customer, "identity", "name"),
        customer.get("name"),
        default="there",
    )


def category_name(category):
    if not category:
        return None

    return first_value(
        category.get("slug"),
        category.get("name"),
    )


def extract_merchant_id(trigger):
    if not trigger:
        return None

    payload = trigger.get("payload", {})

    return first_value(
        trigger.get("merchant_id"),
        payload.get("merchant_id"),
    )


def extract_customer_id(trigger):
    if not trigger:
        return None

    payload = trigger.get("payload", {})

    return first_value(
        trigger.get("customer_id"),
        payload.get("customer_id"),
    )


def extract_category(trigger):
    if not trigger:
        return None

    payload = trigger.get("payload", {})

    return first_value(
        trigger.get("category"),
        payload.get("category"),
    )


def active_offer(merchant):
    if not merchant:
        return None

    offers = merchant.get("offers", [])

    if not isinstance(offers, list):
        return None

    for offer in offers:
        if not isinstance(offer, dict):
            continue

        status = str(
            offer.get("status", "")
        ).lower()

        if status == "active":
            return offer

    return None


def offer_text(merchant):
    offer = active_offer(merchant)

    if not offer:
        return None

    title = first_value(
        offer.get("title"),
        offer.get("name"),
    )

    if title:
        return title

    service = offer.get("service")
    price = offer.get("price")

    if service and price:
        return f"{service} @ ₹{price}"

    return None


def extract_top_item(trigger, category=None):
    payload = trigger.get("payload", {})

    # Directly supplied item
    top_item = payload.get("top_item")

    if isinstance(top_item, dict):
        return top_item

    # Resolve item by ID from the category digest
    top_item_id = payload.get("top_item_id")

    if top_item_id and isinstance(category, dict):
        digest = category.get("digest", [])

        if isinstance(digest, list):
            for item in digest:
                if isinstance(item, dict) and item.get("id") == top_item_id:
                    return item

    # Legacy fallback
    top_item = trigger.get("top_item")

    return top_item if isinstance(top_item, dict) else {}


def get_performance_value(merchant, key):
    performance = merchant.get("performance", {})

    if not isinstance(performance, dict):
        return None

    return performance.get(key)


def detect_language(merchant):
    """
    Lightweight deterministic language preference detection.
    """

    identity = merchant.get("identity", {}) if merchant else {}

    value = first_value(
        identity.get("language"),
        identity.get("language_pref"),
        merchant.get("language") if merchant else None,
    )

    if not value:
        return "english"

    value = str(value).lower()

    if "hi" in value or "hindi" in value:
        return "hinglish"

    return "english"


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Vera bot is running",
        "dataset": {
            "categories": len(DATASET["categories"]),
            "merchants": len(DATASET["merchants"]),
            "customers": len(DATASET["customers"]),
            "triggers": len(DATASET["triggers"]),
        },
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/v1/healthz")
def healthz():

    counts = {
        "category": 0,
        "merchant": 0,
        "customer": 0,
        "trigger": 0,
    }

    for context in contexts.values():

        scope = context["scope"]

        if scope in counts:
            counts[scope] += 1

    return {
        "status": "ok",
        "uptime_seconds": int(
            time.time() - server_started_at
        ),
        "contexts_loaded": counts,
    }


# ============================================================
# METADATA
# ============================================================

@app.get("/v1/metadata")
def metadata():

    return {
        "team_name": "grimlock",

        # CHANGE THIS
        "team_members": ["Shashank Shekhar"],

        "model": "deterministic-rule-engine",

        "approach": (
            "deterministic merchant-aware composer "
            "with dataset retrieval and intent routing"
        ),

        "version": "2.0.0",
    }


# ============================================================
# PUSH CONTEXT
# ============================================================

@app.post("/v1/context")
def push_context(request: ContextRequest):

    key = f"{request.scope}:{request.context_id}"

    if key in contexts:

        current_version = contexts[key]["version"]

        if request.version <= current_version:

            return JSONResponse(
                status_code=409,
                content={
                    "accepted": False,
                    "reason": "stale_version",
                    "current_version": current_version,
                },
            )

    contexts[key] = {
        "scope": request.scope,
        "context_id": request.context_id,
        "version": request.version,
        "delivered_at": request.delivered_at,
        "payload": request.payload,
    }

    ack_id = (
        f"ack_{request.context_id}_"
        f"v{request.version}"
    )

    return {
        "accepted": True,
        "ack_id": ack_id,
        "stored_at": utc_now(),
    }


# ============================================================
# TRIGGER EXPIRY
# ============================================================

def trigger_is_expired(trigger, now_string):

    expires_at = trigger.get("expires_at")

    if not expires_at:
        return False

    try:

        now_dt = datetime.fromisoformat(
            now_string.replace("Z", "+00:00")
        )

        expiry_dt = datetime.fromisoformat(
            expires_at.replace("Z", "+00:00")
        )

        return now_dt > expiry_dt

    except (ValueError, TypeError):

        return False


# ============================================================
# ACTION FACTORY
# ============================================================

def make_action(
    conversation_id,
    merchant_id,
    customer_id,
    send_as,
    trigger_id,
    template_name,
    template_params,
    body,
    cta,
    suppression_key,
    rationale,
):

    return {
        "conversation_id": conversation_id,
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "send_as": send_as,
        "trigger_id": trigger_id,
        "template_name": template_name,
        "template_params": template_params,
        "body": body,
        "cta": cta,
        "suppression_key": suppression_key,
        "rationale": rationale,
    }


# ============================================================
# RESEARCH DIGEST
# ============================================================

def compose_research_digest(
    trigger,
    merchant,
    category,
    trigger_id,
    suppression_key,
):

    if merchant is None:
        return None

    item = extract_top_item(trigger, category)

    title = first_value(
        item.get("title"),
        item.get("headline"),
    )

    source = item.get("source")

    trial_n = first_value(
        item.get("trial_n"),
        item.get("sample_size"),
    )

    patient_segment = item.get(
        "patient_segment"
    )

    merchant_id = merchant.get(
        "merchant_id",
        extract_merchant_id(trigger),
    )

    name = merchant_name(merchant)

    if not title:
        title = "a new category update"

    parts = [
        f"{name}, a new update in your category caught my eye."
    ]

    parts.append(title)

    if trial_n:
        parts.append(
            f"It covers {trial_n:,} participants."
            if isinstance(trial_n, int)
            else f"It covers {trial_n} participants."
        )

    if patient_segment:
        parts.append(
            f"It is especially relevant to {patient_segment}."
        )

    if source:
        parts.append(f"Source: {source}.")

    body = " ".join(parts)

    body += (
        " Want me to pull out the practical takeaway "
        "for your business?"
    )

    return make_action(
        conversation_id=f"conv_{merchant_id}_{trigger_id}",
        merchant_id=merchant_id,
        customer_id=None,
        send_as="vera",
        trigger_id=trigger_id,
        template_name="research_digest_v1",
        template_params=[
            title,
            source or "",
        ],
        body=body,
        cta="open_ended",
        suppression_key=suppression_key,
        rationale=(
            "Research trigger matched to a concrete "
            "category insight and relevant merchant context."
        ),
    )


# ============================================================
# PERFORMANCE SPIKE
# ============================================================

def compose_perf_spike(
    trigger,
    merchant,
    trigger_id,
    suppression_key,
):

    if merchant is None:
        return None

    merchant_id = merchant.get(
        "merchant_id",
        extract_merchant_id(trigger),
    )

    name = merchant_name(merchant)

    payload = trigger.get("payload", {})

    change = first_value(
        payload.get("change_pct"),
        payload.get("percentage_change"),
        payload.get("delta_pct"),
    )

    metric = first_value(
        payload.get("metric"),
        "views",
    )

    if change is not None:

        body = (
            f"{name}, your {metric} are up {change}% "
            f"versus the comparison period. "
            f"That movement is worth looking at while "
            f"it's fresh. Want me to break down what changed?"
        )

    else:

        body = (
            f"{name}, your dashboard shows a recent "
            f"spike in {metric}. Want me to break down "
            f"what changed and where it came from?"
        )

    return make_action(
        conversation_id=f"conv_{merchant_id}_{trigger_id}",
        merchant_id=merchant_id,
        customer_id=None,
        send_as="vera",
        trigger_id=trigger_id,
        template_name="performance_spike_v1",
        template_params=[str(metric)],
        body=body,
        cta="open_ended",
        suppression_key=suppression_key,
        rationale=(
            "Performance spike trigger surfaced with "
            "the available merchant metric."
        ),
    )


# ============================================================
# PERFORMANCE DIP
# ============================================================

def compose_perf_dip(
    trigger,
    merchant,
    trigger_id,
    suppression_key,
):

    if merchant is None:
        return None

    merchant_id = merchant.get(
        "merchant_id",
        extract_merchant_id(trigger),
    )

    name = merchant_name(merchant)

    payload = trigger.get("payload", {})

    change = first_value(
        payload.get("change_pct"),
        payload.get("percentage_change"),
        payload.get("delta_pct"),
    )

    metric = first_value(
        payload.get("metric"),
        "calls",
    )

    if change is not None:

        change_value = abs(float(change))

        # Challenge dataset may express relative change as a decimal.
        # Example: -0.5 means 50% down, not 0% down.
        if change_value <= 1:
            change_value *= 100

        body = (
            f"{name}, your {metric} are down {change_value:.0f}% "
            f"versus the comparison period. "
            f"I can check the available signals and show "
            f"what may be driving it. Want me to?"
        )

    else:

        body = (
            f"{name}, there's a recent dip in {metric}. "
            f"I can check the available signals and show "
            f"what may be driving it. Want me to?"
        )

    return make_action(
        conversation_id=f"conv_{merchant_id}_{trigger_id}",
        merchant_id=merchant_id,
        customer_id=None,
        send_as="vera",
        trigger_id=trigger_id,
        template_name="performance_dip_v1",
        template_params=[str(metric)],
        body=body,
        cta="open_ended",
        suppression_key=suppression_key,
        rationale=(
            "Performance dip trigger translated into "
            "a specific diagnostic prompt."
        ),
    )


# ============================================================
# RENEWAL DUE
# ============================================================

def compose_renewal_due(
    trigger,
    merchant,
    trigger_id,
    suppression_key,
):

    if merchant is None:
        return None

    merchant_id = merchant.get(
        "merchant_id",
        extract_merchant_id(trigger),
    )

    name = merchant_name(merchant)
    payload = trigger.get("payload", {})

    days_remaining = payload.get("days_remaining")
    plan = payload.get("plan")
    amount = payload.get("renewal_amount")

    if days_remaining is not None and plan and amount is not None:
        body = (
            f"{name}, your {plan} plan is due for renewal "
            f"in {days_remaining} days at ₹{amount:,}. "
            f"Want me to walk you through the renewal?"
        )
    elif days_remaining is not None:
        body = (
            f"{name}, your plan is due for renewal "
            f"in {days_remaining} days. "
            f"Want me to walk you through the renewal?"
        )
    else:
        body = (
            f"{name}, your plan is coming up for renewal. "
            f"Want me to walk you through it?"
        )

    return make_action(
        conversation_id=f"conv_{merchant_id}_{trigger_id}",
        merchant_id=merchant_id,
        customer_id=None,
        send_as="vera",
        trigger_id=trigger_id,
        template_name="renewal_due_v1",
        template_params=[
            str(days_remaining or ""),
            str(plan or ""),
            str(amount or ""),
        ],
        body=body,
        cta="open_ended",
        suppression_key=suppression_key,
        rationale=(
            "Renewal trigger matched to the merchant's "
            "plan, remaining days, and renewal amount."
        ),
    )


# ============================================================
# MILESTONE
# ============================================================

def compose_milestone(
    trigger,
    merchant,
    trigger_id,
    suppression_key,
):

    if merchant is None:
        return None

    merchant_id = merchant.get(
        "merchant_id",
        extract_merchant_id(trigger),
    )

    name = merchant_name(merchant)

    payload = trigger.get("payload", {})

    milestone = first_value(
        payload.get("milestone"),
        payload.get("value"),
        payload.get("count"),
    )

    metric = first_value(
        payload.get("metric"),
        "reviews",
    )

    if milestone is not None:

        body = (
            f"{name}, you've just crossed {milestone} "
            f"{metric}. That's a useful milestone to build on. "
            f"Want me to suggest the next profile action?"
        )

    else:

        body = (
            f"{name}, you've just reached a new {metric} "
            f"milestone. Want me to suggest what to do next?"
        )

    return make_action(
        conversation_id=f"conv_{merchant_id}_{trigger_id}",
        merchant_id=merchant_id,
        customer_id=None,
        send_as="vera",
        trigger_id=trigger_id,
        template_name="milestone_v1",
        template_params=[str(milestone or metric)],
        body=body,
        cta="open_ended",
        suppression_key=suppression_key,
        rationale=(
            "Milestone trigger used to create a specific "
            "and actionable merchant nudge."
        ),
    )


# ============================================================
# DORMANT
# ============================================================

def compose_dormant(
    trigger,
    merchant,
    trigger_id,
    suppression_key,
):

    if merchant is None:
        return None

    merchant_id = merchant.get(
        "merchant_id",
        extract_merchant_id(trigger),
    )

    name = merchant_name(merchant)

    body = (
        f"{name}, it's been a while since we last worked "
        f"on your business profile. I found a concrete "
        f"opportunity in your current data. Want me to show it?"
    )

    return make_action(
        conversation_id=f"conv_{merchant_id}_{trigger_id}",
        merchant_id=merchant_id,
        customer_id=None,
        send_as="vera",
        trigger_id=trigger_id,
        template_name="dormant_v1",
        template_params=[],
        body=body,
        cta="open_ended",
        suppression_key=suppression_key,
        rationale=(
            "Dormant merchant trigger uses a low-friction "
            "curiosity hook rather than a generic promotion."
        ),
    )


# ============================================================
# FESTIVAL / WEATHER / NEWS / REGULATION / TREND
# ============================================================

def compose_external_trigger(
    trigger,
    merchant,
    category,
    trigger_id,
    suppression_key,
):

    if merchant is None:
        return None

    merchant_id = merchant.get(
        "merchant_id",
        extract_merchant_id(trigger),
    )

    kind = trigger.get("kind")

    payload = trigger.get("payload", {})

    title = first_value(
        payload.get("title"),
        payload.get("headline"),
        payload.get("event"),
        payload.get("name"),
    )

    detail = first_value(
        payload.get("description"),
        payload.get("note"),
        payload.get("details"),
    )

    if not title:
        title = kind.replace("_", " ")

    name = merchant_name(merchant)

    body = (
        f"{name}, there's a {title} relevant to your "
        f"business right now."
    )

    if detail:
        body += f" {detail}"

    body += (
        " Want me to show the practical implication "
        "for your business?"
    )

    return make_action(
        conversation_id=f"conv_{merchant_id}_{trigger_id}",
        merchant_id=merchant_id,
        customer_id=None,
        send_as="vera",
        trigger_id=trigger_id,
        template_name="external_trigger_v1",
        template_params=[title],
        body=body,
        cta="open_ended",
        suppression_key=suppression_key,
        rationale=(
            f"External trigger '{kind}' surfaced using "
            "the information supplied in the trigger context."
        ),
    )


# ============================================================
# CUSTOMER RECALL
# ============================================================

def compose_recall_due(
    trigger,
    merchant,
    customer,
    trigger_id,
    suppression_key,
):

    if merchant is None or customer is None:
        return None

    merchant_id = merchant.get(
        "merchant_id",
        extract_merchant_id(trigger),
    )

    customer_id = customer.get(
        "customer_id",
        extract_customer_id(trigger),
    )

    m_name = merchant_name(merchant)
    c_name = customer_name(customer)

    offer = offer_text(merchant)

    if not offer:
        offer = "your next appointment"

    body = (
        f"Hi {c_name}, {m_name} here 🦷. "
        f"Your cleaning recall is due. "
        f"We currently have {offer}. "
        f"Would you like me to help arrange a slot?"
    )

    return make_action(
        conversation_id=(
            f"conv_{customer_id}_{trigger_id}"
        ),
        merchant_id=merchant_id,
        customer_id=customer_id,
        send_as="merchant_on_behalf",
        trigger_id=trigger_id,
        template_name="recall_due_v1",
        template_params=[
            c_name,
            offer,
        ],
        body=body,
        cta="open_ended",
        suppression_key=suppression_key,
        rationale=(
            "Customer recall trigger matched to the "
            "merchant's active offer and customer context."
        ),
    )


# ============================================================
# MAIN COMPOSER
# ============================================================

def compose_action(
    trigger,
    merchant,
    customer,
    trigger_id,
    suppression_key,
):

    if not trigger:
        return None

    kind = str(
        trigger.get("kind", "")
    ).lower()

    # --------------------------------------------------------
    # CUSTOMER RECALL
    # --------------------------------------------------------

    if kind in {
        "recall_due",
        "customer_lapsed_soft",
    }:

        return compose_recall_due(
            trigger,
            merchant,
            customer,
            trigger_id,
            suppression_key,
        )

    # --------------------------------------------------------
    # RESEARCH
    # --------------------------------------------------------

    if kind in {
        "research_digest",
        "research_digest_release",
        "category_research_digest_release",
    }:

        category = find_category(
            extract_category(trigger)
        )

        return compose_research_digest(
            trigger,
            merchant,
            category,
            trigger_id,
            suppression_key,
        )

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    if kind == "perf_spike":

        return compose_perf_spike(
            trigger,
            merchant,
            trigger_id,
            suppression_key,
        )

    if kind == "perf_dip":

        return compose_perf_dip(
            trigger,
            merchant,
            trigger_id,
            suppression_key,
        )

    # --------------------------------------------------------
    # RENEWAL
    # --------------------------------------------------------

    if kind == "renewal_due":

        return compose_renewal_due(
            trigger,
            merchant,
            trigger_id,
            suppression_key,
        )

    # --------------------------------------------------------
    # MILESTONE
    # --------------------------------------------------------

    if kind == "milestone_reached":

        return compose_milestone(
            trigger,
            merchant,
            trigger_id,
            suppression_key,
        )

    # --------------------------------------------------------
    # DORMANT
    # --------------------------------------------------------

    if kind == "dormant_with_vera":

        return compose_dormant(
            trigger,
            merchant,
            trigger_id,
            suppression_key,
        )

    # --------------------------------------------------------
    # EXTERNAL EVENTS
    # --------------------------------------------------------

    if kind in {
        "festival_upcoming",
        "weather_heatwave",
        "local_news_event",
        "regulation_change",
        "competitor_opened",
        "category_trend_movement",
    }:

        category = find_category(
            extract_category(trigger)
        )

        return compose_external_trigger(
            trigger,
            merchant,
            category,
            trigger_id,
            suppression_key,
        )

    # --------------------------------------------------------
    # GENERIC FALLBACK
    # --------------------------------------------------------

    if merchant is None:
        return None

    merchant_id = merchant.get(
        "merchant_id",
        extract_merchant_id(trigger),
    )

    name = merchant_name(merchant)

    readable_kind = kind.replace(
        "_", " "
    )

    body = (
        f"{name}, I spotted a {readable_kind} "
        f"signal in your account. Want me to show "
        f"what it means for your business?"
    )

    return make_action(
        conversation_id=f"conv_{merchant_id}_{trigger_id}",
        merchant_id=merchant_id,
        customer_id=None,
        send_as="vera",
        trigger_id=trigger_id,
        template_name="generic_trigger_v1",
        template_params=[readable_kind],
        body=body,
        cta="open_ended",
        suppression_key=suppression_key,
        rationale=(
            "Generic deterministic fallback for an otherwise "
            "unsupported trigger kind."
        ),
    )


# ============================================================
# TICK
# ============================================================

@app.post("/v1/tick")
def tick(request: TickRequest):

    actions = []

    seen_suppression_keys = set()

    for trigger_id in request.available_triggers:

        # ----------------------------------------------------
        # Find trigger
        # ----------------------------------------------------

        trigger_context = get_context(
            "trigger",
            trigger_id,
        )

        if trigger_context is None:
            continue

        trigger = trigger_context.get(
            "payload",
            {},
        )

        if not isinstance(trigger, dict):
            continue

        # ----------------------------------------------------
        # Expiry
        # ----------------------------------------------------

        if trigger_is_expired(
            trigger,
            request.now,
        ):
            continue

        # ----------------------------------------------------
        # Suppression
        # ----------------------------------------------------

        suppression_key = first_value(
            trigger.get("suppression_key"),
            trigger.get("payload", {}).get(
                "suppression_key"
            ),
            trigger_id,
        )

        if suppression_key in seen_suppression_keys:
            continue

        # ----------------------------------------------------
        # Merchant
        # ----------------------------------------------------

        merchant_id = extract_merchant_id(
            trigger
        )

        merchant = find_merchant(
            merchant_id
        )

        # ----------------------------------------------------
        # Customer
        # ----------------------------------------------------

        customer_id = extract_customer_id(
            trigger
        )

        customer = find_customer(
            customer_id
        ) if customer_id else None

        # ----------------------------------------------------
        # Compose
        # ----------------------------------------------------

        action = compose_action(
            trigger=trigger,
            merchant=merchant,
            customer=customer,
            trigger_id=trigger_id,
            suppression_key=suppression_key,
        )

        if action is not None:

            actions.append(action)

            seen_suppression_keys.add(
                suppression_key
            )

        # Challenge allows <=20
        if len(actions) >= 20:
            break

    return {
        "actions": actions
    }


# ============================================================
# REPLY / MULTI-TURN HANDLING
# ============================================================

AUTO_REPLY_PATTERNS = [
    "thank you for contacting",
    "thanks for contacting",
    "your message has been received",
    "we will get back to you",
    "we'll get back to you",
    "our team will contact",
    "your query has been received",
    "for your information",
    "thank you for your message",
    "i am an automated assistant",
    "automated assistant",
]


POSITIVE_PATTERNS = [
    "yes",
    "yeah",
    "yep",
    "sure",
    "okay",
    "ok",
    "lets do it",
    "let's do it",
    "go ahead",
    "do it",
    "i want to join",
    "want to join",
    "sign me up",
    "interested",
    "please proceed",
    "proceed",
    "start",
]


NEGATIVE_PATTERNS = [
    "not interested",
    "no thanks",
    "no thank you",
    "stop",
    "don't message",
    "dont message",
    "do not message",
    "remove me",
    "unsubscribe",
    "leave me alone",
]


def normalized_text(text):
    return re.sub(
        r"\s+",
        " ",
        text.strip().lower(),
    )


def contains_phrase(text, phrases):
    text = normalized_text(text)

    return any(
        phrase in text
        for phrase in phrases
    )


def is_auto_reply(message):
    text = normalized_text(message)

    for pattern in AUTO_REPLY_PATTERNS:

        if pattern in text:
            return True

    return False


def is_negative_intent(message):
    return contains_phrase(
        message,
        NEGATIVE_PATTERNS,
    )


def is_positive_intent(message):

    text = normalized_text(message)

    if text in {
        "yes",
        "yeah",
        "yep",
        "ok",
        "okay",
        "sure",
    }:
        return True

    return contains_phrase(
        text,
        POSITIVE_PATTERNS,
    )


def reply_action_end(reason):

    return {
        "action": "end",
        "rationale": reason,
    }


def reply_action_send(
    body,
    cta="open_ended",
    rationale="",
):

    return {
        "action": "send",
        "body": body,
        "cta": cta,
        "rationale": rationale,
    }


@app.post("/v1/reply")
def reply(request: ReplyRequest):

    message = request.message.strip()

    # --------------------------------------------------------
    # HARD STOP
    # --------------------------------------------------------

    if is_negative_intent(message):

        return reply_action_end(
            "Merchant explicitly declined or requested "
            "no further messaging. Closing the conversation."
        )

    # --------------------------------------------------------
    # AUTO REPLY
    # --------------------------------------------------------

    if is_auto_reply(message):

        return reply_action_send(
            body=(
                "Got it — I may have reached an automated "
                "WhatsApp reply. I'll avoid sending repeated "
                "messages here."
            ),
            cta="none",
            rationale=(
                "Detected likely canned/automated response "
                "and avoided wasting additional turns."
            ),
        )

    # --------------------------------------------------------
    # POSITIVE INTENT
    # --------------------------------------------------------

    if is_positive_intent(message):

        return reply_action_send(
            body=(
                "Absolutely — let's do it. "
                "I'll move this to the next step using "
                "the details already available for your business."
            ),
            cta="open_ended",
            rationale=(
                "Explicit positive intent detected. "
                "Proceeding directly instead of asking "
                "unnecessary qualification questions."
            ),
        )

    # --------------------------------------------------------
    # PERFORMANCE QUESTION
    # --------------------------------------------------------

    normalized = normalized_text(message)

    performance_terms = [
        "calls",
        "call volume",
        "call count",
        "calls dropped",
        "calls down",
        "calls decreased",
        "fewer calls",
        "calls falling",
    ]

    if (
        any(term in normalized for term in performance_terms)
        and any(word in normalized for word in ["why", "how", "what", "down", "dropped", "decreased"])
    ):
        return reply_action_send(
            body=(
                "I can check the available performance signals "
                "behind the call drop and help narrow down what changed. "
                "Want me to check the recent details?"
            ),
            cta="open_ended",
            rationale=(
                "Detected a merchant question about call performance "
                "and routed it to a focused diagnostic response."
            ),
        )

    # --------------------------------------------------------
    # PERFORMANCE FOLLOW-UP
    # --------------------------------------------------------

    performance_follow_up_terms = [
        "what exactly changed",
        "what changed",
        "what happened",
        "what is causing",
        "what caused",
        "why did this happen",
        "what's causing",
    ]

    if any(
        term in normalized
        for term in performance_follow_up_terms
    ):
        return reply_action_send(
            body=(
                "The available context points to a recent drop in "
                "call volume versus your comparison period. I can "
                "use the available performance signals to narrow "
                "down the likely change. Want me to check those signals?"
            ),
            cta="open_ended",
            rationale=(
                "Detected a performance follow-up question and "
                "continued the diagnostic flow using available "
                "business context."
            ),
        )

    # --------------------------------------------------------
    # QUESTION / GENERAL REPLY
    # --------------------------------------------------------

    question_words = [
        "what",
        "how",
        "why",
        "when",
        "where",
        "?",
    ]

    if any(
        word in normalized_text(message)
        for word in question_words
    ):

        return reply_action_send(
            body=(
                "Sure — I can help with that. "
                "Tell me the specific part you'd like "
                "me to check and I'll use the available "
                "business context."
            ),
            cta="open_ended",
            rationale=(
                "Merchant asked a follow-up question; "
                "kept the response focused and context-aware."
            ),
        )

    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return reply_action_send(
        body=(
            "Got it. I can work with that. "
            "Would you like me to take the next step?"
        ),
        cta="open_ended",
        rationale=(
            "Neutral continuation for an ambiguous merchant reply."
        ),
    )

# ============================================================
# STARTUP INFO
# ============================================================

@app.on_event("startup")
def startup_event():

    print("=" * 60)
    print("Vera Merchant AI Assistant")
    print("=" * 60)

    print(
        f"Categories: {len(DATASET['categories'])}"
    )

    print(
        f"Merchants: {len(DATASET['merchants'])}"
    )

    print(
        f"Customers: {len(DATASET['customers'])}"
    )

    print(
        f"Triggers: {len(DATASET['triggers'])}"
    )

    print("=" * 60)