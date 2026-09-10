"""
Intent Definitions for AmazonHelp
====================================
8 intents derived by reading 500+ real AmazonHelp customer messages
from the cleaned dataset (amazon_customers.csv).

Each intent has:
  - name        : machine-readable slug (used in code)
  - display     : human-readable label
  - description : used in LLM classifier system prompt
  - keywords    : weak signal hints for sampling/filtering only
  - examples    : 3 real messages from the dataset per intent (few-shot)
"""

INTENTS = [
    {
        "name": "delivery_issue",
        "display": "Delivery Issue",
        "description": (
            "Customer's package has not arrived, is delayed, shows wrong delivery status, "
            "was delivered to wrong address, delivery was attempted but missed, or tracking "
            "info is not updating. This is the highest volume intent."
        ),
        "keywords": [
            "not delivered", "not arrived", "where is my package", "out for delivery",
            "tracking", "delayed", "delivery", "courier", "shipped", "arrived",
            "delivery date", "expected", "missing", "lost", "late", "parcel"
        ],
        "examples": [
            "Item has not been delivered but tracking says it was handed to me over an hour ago. 2nd time this has happened.",
            "I have a package that is showing as delivered last Friday, however it has not been and there is no note.",
            "My Prime expedited delivery has been out for delivery for 70 hours. What is going on?",
        ],
    },
    {
        "name": "return_refund",
        "display": "Return / Refund",
        "description": (
            "Customer wants to return a product, get a refund, request a replacement, "
            "exchange an item, or is unhappy with the refund amount received. "
            "Also includes pickup not being arranged for a return."
        ),
        "keywords": [
            "return", "refund", "replace", "replacement", "exchange", "money back",
            "send back", "pickup", "pick up", "return policy", "reimburse", "want to return"
        ],
        "examples": [
            "The pick up has been arranged 5 times now and cancelled at the last moment without citing any reason!",
            "I purchased redmi4 few functions of phone not working and camera quality not good. Hence dont want to replace this with same product.",
            "Pur 50 Led TV with extended warranty from Amazon which is giving problem. Unable to talk to someone in Amazon.",
        ],
    },
    {
        "name": "account_access",
        "display": "Account / Login Issue",
        "description": (
            "Customer cannot log in, forgot password, account is locked or suspended, "
            "two-factor authentication not working, account hacked or compromised, "
            "or cannot update account details like email address."
        ),
        "keywords": [
            "login", "log in", "password", "account locked", "locked", "suspended",
            "sign in", "cant access", "hacked", "unauthorized", "two step", "2 step",
            "verification", "otp", "email address", "account closed"
        ],
        "examples": [
            "i reset my password 3 times and it still says incorrect you gotta be shitting me",
            "I cant log in because its locked.",
            "I can't log in to my account to even do that. That's the issue. I have Zero access cause it's been hacked.",
        ],
    },
    {
        "name": "billing_payment",
        "display": "Billing / Payment Issue",
        "description": (
            "Unexpected or incorrect charges, payment taken multiple times, "
            "charged for a service not used, cashback not received, "
            "Amazon Pay balance issues, or payment failure at checkout."
        ),
        "keywords": [
            "charged", "charge", "payment", "billing", "invoice", "cashback",
            "cash back", "refund", "deducted", "amazon pay", "false charge",
            "unknown charge", "double charged", "taken money", "subscription charge"
        ],
        "examples": [
            "Haha has been sneakily charging me $9.99 SINCE FEBRUARY for a service I never use $90 gone!!!",
            "i have had a payment taken THREE times and no product!",
            "Even after your teams promise, I have not received my cashback of INR1500 before deadline given by them.",
        ],
    },
    {
        "name": "product_issue",
        "display": "Product / Item Issue",
        "description": (
            "Product arrived damaged, broken, defective, or not as described. "
            "Wrong item was sent. Item is counterfeit or fake. "
            "Package was tampered with or opened before delivery."
        ),
        "keywords": [
            "broken", "damaged", "defective", "wrong item", "not as described",
            "fake", "counterfeit", "tampered", "opened", "not working",
            "dead on arrival", "cracked", "faulty", "doesnt work", "item damaged"
        ],
        "examples": [
            "my package was accidentally opened. 4 items missing worth GBP 97. You need better delivery drivers!!",
            "Someone opened my amazon package and retaped it. I dont know if it was the delivery driver or someone in the neighborhood.",
            "I received the product in damaged condition, its tempered from outside though material was not spilling.",
        ],
    },
    {
        "name": "prime_subscription",
        "display": "Prime / Subscription Issue",
        "description": (
            "Questions or complaints about Amazon Prime membership — benefits not working, "
            "Prime delivery not being free, Prime Video access issues, "
            "unexpected Prime charges, wanting to cancel Prime, "
            "or questions about what Prime includes."
        ),
        "keywords": [
            "prime", "prime membership", "prime video", "prime delivery",
            "cancel prime", "prime benefits", "prime now", "prime charge",
            "subscribe", "subscription", "prime trial", "prime shipping"
        ],
        "examples": [
            "being charged for amazon prime & when I go to cancel it its saying Im not a member",
            "I am unable to Use Prime Video even though I am Prime Member it tells me to sign up for prime Membership",
            "sure is frustrating to pay $100 a year for Prime and not have my Prime items delivered after 5 days.",
        ],
    },
    {
        "name": "device_tech_support",
        "display": "Device / Tech Support",
        "description": (
            "Customer needs help with an Amazon device — Kindle, Echo, Alexa, Fire Tablet, "
            "Fire Stick, Echo Dot, Echo Show. Issues include setup, not connecting, "
            "features not working, app issues, or software bugs."
        ),
        "keywords": [
            "kindle", "echo", "alexa", "fire tablet", "fire stick", "firestick",
            "fire tv", "echo dot", "echo show", "device", "not working", "setup",
            "connect", "bluetooth", "wifi", "app", "alexa skill", "buffering", "freezing"
        ],
        "examples": [
            "Bought an Echo Show and it wont recognize a single account in our household. WTF guys?",
            "My Kindle not working properly is breaking my heart!",
            "my echo keeps telling me that I need to link my premium spotify account to play music from there but I have already done that.",
        ],
    },
    {
        "name": "other",
        "display": "Other / General",
        "description": (
            "Message does not clearly fit any of the above categories. "
            "Includes vague complaints, general feedback, compliments, "
            "seller-related questions, ticket purchases, or unclear intent. "
            "When in doubt, classify as other."
        ),
        "keywords": [],
        "examples": [
            "Way to drop the ball on customer service so pissed right now!",
            "Amazon just doesnt seem to get it together poor customer service warranty is a scam",
            "please teach Alexa slang. Ive been trying to play Momentz by gorillaz for the last 30 minutes.",
        ],
    },
]

# ── Helper exports used by agent/pipeline.py ──────────────────────────────
INTENT_NAMES = [i["name"] for i in INTENTS]
INTENT_MAP   = {i["name"]: i for i in INTENTS}


def get_intent_list_block() -> str:
    """
    Builds the numbered intent list for the LLM classifier system prompt.
    Example output:
      1. delivery_issue — Customer's package has not arrived...
      2. return_refund  — Customer wants to return...
    """
    lines = []
    for idx, intent in enumerate(INTENTS, 1):
        lines.append(f"{idx}. {intent['name']} — {intent['description']}")
    return "\n".join(lines)


def get_few_shot_block() -> str:
    """
    Builds few-shot examples block for the LLM classifier system prompt.
    3 real examples per intent, clearly labeled.
    """
    lines = []
    for intent in INTENTS:
        for example in intent["examples"]:
            lines.append(f'Message: "{example}"')
            lines.append(f'Intent: {intent["name"]}')
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick sanity check
    print(f"Total intents defined : {len(INTENTS)}")
    print(f"Intent names          : {INTENT_NAMES}")
    print()
    print("=== Intent List Block (for LLM prompt) ===")
    print(get_intent_list_block())
    print()
    print("=== Few-shot Block (first 3 examples) ===")
    lines = get_few_shot_block().split("\n")
    print("\n".join(lines[:12]))
