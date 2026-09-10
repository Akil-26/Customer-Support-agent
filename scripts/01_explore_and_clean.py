"""
Step 1: Data Exploration & Cleaning (FIXED VERSION)
=====================================================
Fixes applied:
1. Better emoji removal using unicode ranges + variation selectors
2. Filter out non-English messages (keep only ASCII + common punctuation)
3. Remove agent signature codes like ^AG, ^TN, ^CH from brand replies
4. Stricter min_length filter (20 chars) to drop meaningless short messages
5. Cleaner output columns — only what we actually need
6. Deduplication on cleaned text

Outputs:
  data/processed/amazon_customers.csv   <- cleaned customer messages only
  data/processed/amazon_pairs.csv       <- (customer msg, brand reply) pairs
  data/processed/exploration_report.txt <- summary stats
"""

import pandas as pd
import re
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
RAW_PATH      = Path("data/raw/twcs/twcs.csv")
OUT_CUSTOMERS = Path("data/processed/amazon_customers.csv")
OUT_PAIRS     = Path("data/processed/amazon_pairs.csv")
OUT_REPORT    = Path("data/processed/exploration_report.txt")

BRAND      = "AmazonHelp"
CHUNK_SIZE = 100_000


# ── Text Cleaning ──────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Full cleaning pipeline.
    
    Order matters:
    1. @mentions first — before anything else so we don't accidentally
       clean parts of a mention and leave fragments
    2. URLs — t.co links carry no meaning
    3. Agent codes like ^AG ^TN — Amazon's internal agent signatures
       that appear at the end of every brand reply. Pure noise.
    4. HTML entities — &amp; &lt; &gt; from Twitter API encoding
    5. Emojis — broad unicode range pattern covers all emoji blocks
    6. Non-ASCII characters — removes Japanese, German umlauts, Arabic etc.
       We keep only English messages for this project.
       Why? Our LLM classifier and RAG are English-only. Non-English
       messages would confuse intent classification badly.
    7. Collapse whitespace — result of all above removals leaving gaps
    """
    if not isinstance(text, str):
        return ""

    # 1. Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # 2. Remove URLs — also remove dangling colon/punctuation left after URL removal
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove trailing colon anywhere at end — catches both "here:" and "here: "
    text = re.sub(r"[:,]\s*$", "", text.strip())

    # 3. Remove Amazon agent signatures like ^AG ^TN ^CH ^KM
    text = re.sub(r"\^[A-Z]{2,3}", "", text)

    # 4. Decode HTML entities
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")

    # 5. Remove emojis (comprehensive unicode ranges)
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"   # emoticons
        u"\U0001F300-\U0001F5FF"   # symbols & pictographs
        u"\U0001F680-\U0001F6FF"   # transport & map symbols
        u"\U0001F700-\U0001F77F"   # alchemical symbols
        u"\U0001F780-\U0001F7FF"   # geometric shapes extended
        u"\U0001F800-\U0001F8FF"   # supplemental arrows
        u"\U0001F900-\U0001F9FF"   # supplemental symbols
        u"\U0001FA00-\U0001FA6F"   # chess symbols
        u"\U0001FA70-\U0001FAFF"   # symbols and pictographs extended
        u"\U00002702-\U000027B0"   # dingbats
        u"\U000024C2-\U0001F251"   # enclosed characters
        u"\uFE00-\uFE0F"           # variation selectors (emoji modifiers)
        u"\U0001F1E0-\U0001F1FF"   # flags
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub("", text)

    # 6. Replace common currency symbols before ASCII encoding so numbers stay meaningful
    text = text.replace("£", "GBP ")
    text = text.replace("€", "EUR ")
    text = text.replace("¥", "JPY ")

    # Keep only ASCII printable characters (removes remaining non-ASCII)
    text = text.encode("ascii", errors="ignore").decode("ascii")

    # 7. Remove leftover special chars except basic punctuation
    text = re.sub(r"[^\w\s\.\,\!\?\'\"\-\#\$\@\&\(\)\:]", "", text)

    # 8. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Mid-conversation follow-up signals — these are replies to previous tweets
# with no standalone intent. Useless for classification and RAG.
MID_CONVERSATION_SIGNALS = [
    "details sent", "already handled", "just venting", "never mind",
    "forget it", "sorted now", "all good now", "it arrived", "thanks anyway",
    "got it thanks", "ok thanks", "okay thanks", "thank you bye",
    "yes please", "no thanks", "sounds good", "will do", "done that",
    "tried that", "still waiting", "any update", "following up",
    "as i said", "like i said", "as mentioned",
]

def is_useful(text: str, min_len: int = 20) -> bool:
    """
    Checks:
    1. Min 20 chars and at least 4 words — filters one-liners with no intent
       (raised from 3 words to 4 to catch 'Hi ready for some help')
    2. Not a mid-conversation follow-up with no standalone meaning
    """
    words = text.split()
    if len(text) < 20 or len(words) < 4:
        return False
    text_lower = text.lower()
    for signal in MID_CONVERSATION_SIGNALS:
        if text_lower.startswith(signal) or text_lower == signal:
            return False
    return True


# Common non-English words that survive ASCII encoding check
NON_ENGLISH_SIGNALS = [
    # German
    "danke", "bitte", "nicht", "und", "ist", "ich", "wie", "beim", "wird",
    "haben", "Sie", "oder", "aber", "noch", "wir", "das", "der", "die",
    # French
    "merci", "bonjour", "pourquoi", "je suis", "vous", "est", "une", "mon",
    # Spanish
    "gracias", "hola", "por favor", "no puedo", "tengo", "este",
    # Portuguese
    "obrigado", "nao", "voce", "para",
]

def is_english(text: str) -> bool:
    """
    Two-layer English check BEFORE cleaning:
    1. If more than 20% of characters are non-ASCII → reject
    2. If common non-English words are found → reject
    Stricter than before (was 30%) to catch German messages
    that survived because special chars got stripped to ASCII.
    """
    if not isinstance(text, str) or len(text) == 0:
        return False

    # Layer 1: non-ASCII character ratio (stricter: 20%)
    non_ascii = sum(1 for c in text if ord(c) > 127)
    ratio = non_ascii / len(text)
    if ratio >= 0.20:
        return False

    # Layer 2: keyword check for non-English words
    text_lower = text.lower()
    for signal in NON_ENGLISH_SIGNALS:
        if signal in text_lower:
            return False

    return True


# ── Load AmazonHelp rows ───────────────────────────────────────────────────
def load_amazon_data(path: Path):
    print(f"Reading {path} in chunks of {CHUNK_SIZE:,}...")
    amazon_chunks = []
    total_rows = 0

    for chunk in pd.read_csv(path, dtype=str, chunksize=CHUNK_SIZE):
        total_rows += len(chunk)
        amazon_mask = chunk["author_id"] == BRAND
        amazon_chunks.append(chunk[amazon_mask])

        if total_rows % 500_000 == 0:
            print(f"  Processed {total_rows:,} rows...")

    print(f"  Total rows in dataset  : {total_rows:,}")
    df = pd.concat(amazon_chunks, ignore_index=True)
    print(f"  AmazonHelp rows found  : {len(df):,}")
    return df, total_rows


# ── Get customer messages ──────────────────────────────────────────────────
def get_customer_messages(amazon_df: pd.DataFrame, path: Path) -> pd.DataFrame:
    replied_to_ids = set(amazon_df["in_response_to_tweet_id"].dropna().tolist())
    print(f"\n  AmazonHelp replied to {len(replied_to_ids):,} unique tweets")
    print("  Second pass — collecting customer messages...")

    customer_chunks = []
    for chunk in pd.read_csv(path, dtype=str, chunksize=CHUNK_SIZE):
        mask = (
            (chunk["inbound"] == "True") &
            (chunk["tweet_id"].isin(replied_to_ids))
        )
        customer_chunks.append(chunk[mask])

    customer_df = pd.concat(customer_chunks, ignore_index=True)
    print(f"  Customer messages found : {customer_df.shape[0]:,}")
    return customer_df


# ── Build pairs ────────────────────────────────────────────────────────────
def build_pairs(customer_df: pd.DataFrame, amazon_df: pd.DataFrame) -> pd.DataFrame:
    amazon_reply_map = (
        amazon_df
        .dropna(subset=["in_response_to_tweet_id"])
        .set_index("in_response_to_tweet_id")["text"]
        .to_dict()
    )

    rows = []
    skipped_language = 0
    skipped_tooshort = 0

    for _, row in customer_df.iterrows():
        tweet_id     = row["tweet_id"]
        customer_raw = row["text"]

        # Skip non-English BEFORE cleaning
        if not is_english(customer_raw):
            skipped_language += 1
            continue

        customer_clean = clean_text(customer_raw)

        # Skip too-short or meaningless after cleaning
        if not is_useful(customer_clean):
            skipped_tooshort += 1
            continue

        brand_reply_raw   = amazon_reply_map.get(tweet_id, "")
        brand_reply_clean = clean_text(brand_reply_raw)

        # Fix 3: Filter out non-English brand replies
        # A customer message passed English check but Amazon may have replied
        # in another language (e.g. German support team replied in German)
        # Such replies are useless for RAG — skip the pair entirely
        if brand_reply_clean and not is_english(brand_reply_clean):
            skipped_language += 1
            continue

        rows.append({
            "tweet_id":          tweet_id,
            "created_at":        row["created_at"],
            "customer_clean":    customer_clean,
            "brand_reply_clean": brand_reply_clean,
            "has_reply":         bool(brand_reply_clean and is_useful(brand_reply_clean)),
        })

    print(f"\n  Skipped (non-English)  : {skipped_language:,}")
    print(f"  Skipped (too short)    : {skipped_tooshort:,}")
    print(f"  Valid pairs            : {len(rows):,}")

    pairs_df = pd.DataFrame(rows)

    # Deduplicate on cleaned customer message
    before = len(pairs_df)
    pairs_df = pairs_df.drop_duplicates(subset=["customer_clean"]).reset_index(drop=True)
    print(f"  After deduplication    : {len(pairs_df):,} (removed {before - len(pairs_df):,} duplicates)")

    return pairs_df


# ── Exploration Report ─────────────────────────────────────────────────────
def write_report(pairs_df, total_rows, path):
    lines = []
    lines.append("=" * 60)
    lines.append("DATA EXPLORATION REPORT — AmazonHelp (CLEANED)")
    lines.append("=" * 60)
    lines.append(f"\nTotal rows in full dataset      : {total_rows:,}")
    lines.append(f"Final clean pairs               : {len(pairs_df):,}")
    lines.append(f"Pairs WITH brand reply          : {pairs_df['has_reply'].sum():,}")
    lines.append(f"Pairs WITHOUT brand reply       : {(~pairs_df['has_reply']).sum():,}")
    lines.append(f"\nAvg customer message length     : {pairs_df['customer_clean'].str.len().mean():.1f} chars")
    lines.append(f"Min length                      : {pairs_df['customer_clean'].str.len().min()}")
    lines.append(f"Max length                      : {pairs_df['customer_clean'].str.len().max()}")

    lines.append(f"\nDate range:")
    lines.append(f"  Earliest : {pairs_df['created_at'].min()}")
    lines.append(f"  Latest   : {pairs_df['created_at'].max()}")

    lines.append(f"\nSample cleaned pairs (English only, no emojis, no agent codes, no mid-conversation):")
    for i, row in pairs_df[pairs_df["has_reply"]].head(8).iterrows():
        lines.append(f"\n  Customer : {row['customer_clean']}")
        lines.append(f"  Amazon   : {row['brand_reply_clean']}")
        lines.append(f"  {'-'*50}")

    report = "\n".join(lines)
    path.write_text(report, encoding="utf-8")
    print(report)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    OUT_CUSTOMERS.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load AmazonHelp brand rows
    amazon_df, total_rows = load_amazon_data(RAW_PATH)

    # 2. Get customer messages
    customer_df = get_customer_messages(amazon_df, RAW_PATH)

    # 3. Build clean pairs (filtering + cleaning happens inside)
    print("\nBuilding clean pairs...")
    pairs_df = build_pairs(customer_df, amazon_df)
    pairs_df.to_csv(OUT_PAIRS, index=False)
    print(f"Saved → {OUT_PAIRS}")

    # 4. Save clean customers only (just tweet_id + clean text)
    customers_out = pairs_df[["tweet_id", "created_at", "customer_clean"]].copy()
    customers_out.to_csv(OUT_CUSTOMERS, index=False)
    print(f"Saved → {OUT_CUSTOMERS}")

    # 5. Write report
    print("\nWriting report...")
    write_report(pairs_df, total_rows, OUT_REPORT)
    print(f"Saved → {OUT_REPORT}")
    print("\nStep 1 COMPLETE!")


if __name__ == "__main__":
    main()
