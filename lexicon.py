# lexicon_bp.py (bilingual: English + Spanish)
import io
import re
import unicodedata
from collections import Counter
from urllib.parse import urlparse
from typing import Optional, Tuple, List, Dict

from flask import Blueprint, request, render_template, redirect, url_for, flash, send_file
import pandas as pd
import numpy as np
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from io import BytesIO
from datetime import datetime  # fixed import

# NEW: optional lxml import for HTML parsing
try:
    from lxml import html as lxml_html
except Exception:
    lxml_html = None

# NEW: truthy parser for request flags
def _truthy(v) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "on", "yes", "y", "t"}

# --- NLTK resources ---
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

# --- English sentiment (VADER) ---
SIA_EN = SentimentIntensityAnalyzer()

# --- Try loading spaCy models; fall back to blank pipelines if absent ---
def _load_spacy(lang: str):
    try:
        if lang == "es":
            return spacy.load("es_core_news_sm")
        return spacy.load("en_core_web_sm")
    except Exception:
        # Fallback: blank pipeline (no NER), keeps everything else working
        return spacy.blank("es" if lang == "es" else "en")

# Pre-load cache; load lazily depending on flags
NLP_CACHE = {"en_full": None, "es_full": None, "en_blank": None, "es_blank": None}

def _get_nlp(lang: str, use_ner: bool = True):
    code = "es" if lang == "es" else "en"
    key = f"{code}_{'full' if use_ner else 'blank'}"
    if NLP_CACHE[key] is None:
        NLP_CACHE[key] = _load_spacy(code) if use_ner else spacy.blank(code)
    return NLP_CACHE[key]

# --- Stopword sets ---
STOP_EN = set(stopwords.words("english"))
STOP_ES = set(stopwords.words("spanish"))

# --- Spanish negative lexicon (compact, extensible) ---
# Stored accent-free for robust matching; we strip accents from text before scoring.
SPANISH_NEG_LEXICON = {
    # general polarity
    "malo", "terrible", "horrible", "pesimo", "pésimo", "nefasto", "negativo",
    "odio", "odiar", "toxico", "tóxico", "desastre", "caos", "crisis", "peligro",
    "amenaza", "riesgo", "fracaso", "estafa", "fraude", "corrupcion", "corrupción",
    "mentira", "enganio", "engaño", "falso", "violencia", "ataque", "boicot",
    "ilegal", "criminal", "escandalo", "escándalo", "panico", "pánico",
    # discourse / online harms
    "acoso", "hostigamiento", "insulto", "odio", "difamacion", "difamación",
}

def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )

def _parse_language() -> str:
    lang = (request.form.get("lang") or "en").strip().lower()
    return "es" if lang.startswith("es") else "en"

def _get_stopwords(lang: str):
    return STOP_ES if lang == "es" else STOP_EN

def _split_terms(q: str) -> List[str]:
    """Split a query string on commas or newlines; trim blanks."""
    parts = re.split(r"[,|\n]+", q or "")
    return [p.strip() for p in parts if p and p.strip()]

def _build_filter_mask(
    series: pd.Series,
    query: str,
    *,
    regex: bool = False,
    logic: str = "any",              # "any" or "all"
    case_insensitive: bool = True,
    accent_insensitive: bool = True,
) -> pd.Series:
    """
    Build a boolean mask over 'series' matching 'query' terms.
    - Multi-term queries are ORed ("any") or ANDed ("all").
    - Optional regex mode.
    - Accent-insensitive by default: both haystack and needles are stripped of accents.
    """
    terms = _split_terms(query)
    if not terms:
        return pd.Series(True, index=series.index)

    text = series.fillna("").astype(str)
    if accent_insensitive:
        text_proc = text.apply(_strip_accents)
        terms_proc = [_strip_accents(t) for t in terms]
    else:
        text_proc = text
        terms_proc = terms

    flags = re.IGNORECASE if case_insensitive else 0
    masks: List[pd.Series] = []

    for term in terms_proc:
        if regex:
            try:
                pat = re.compile(term, flags)
                masks.append(text_proc.str.contains(pat, na=False))
            except re.error:
                # Fallback: escape broken regex to literal match
                pat = re.compile(re.escape(term), flags)
                masks.append(text_proc.str.contains(pat, na=False))
        else:
            masks.append(text_proc.str.contains(term, case=not case_insensitive, regex=False, na=False))

    if logic.lower() == "all":
        return pd.concat(masks, axis=1).all(axis=1) if masks else pd.Series(True, index=series.index)
    return pd.concat(masks, axis=1).any(axis=1) if masks else pd.Series(True, index=series.index)

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    # Remove links, @mentions, bare "#"
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    return text.strip()

def extract_meta(text: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    if pd.isna(text):
        return [], [], [], []
    mentions = re.findall(r"@(\w+)", text)
    hashtags = re.findall(r"#(\w+)", text)
    links = re.findall(r"https?://\S+", text)
    domains = []
    for link in links:
        try:
            parsed = urlparse(link)
            domains.append(parsed.netloc)
        except Exception:
            continue
    return mentions, hashtags, links, domains

def extract_embedded_links(article_clean_top_node):
    # Find all URLs in the HTML content
    urls = article_clean_top_node.xpath("//a/@href")

    return urls

def extract_tweet_details(article_clean_top_node):
    # Find all embedded tweets using the appropriate class name
    embedded_tweets = article_clean_top_node.xpath(".//blockquote[contains(@class, 'twitter-tweet')]")

    tweets_data = []

    for tweet in embedded_tweets:
        tweet_data = {}

        # The tweet URL is typically contained within the last <a> tag in the blockquote.
        tweet_link = tweet.xpath(".//a/@href")[-1] if tweet.xpath(".//a/@href") else None
        if tweet_link:
            tweet_data['tweetLink'] = tweet_link
            
            # Extract tweet ID from the tweet URL
            tweet_data['tweetId'] = tweet_link.split('/')[-1]
            
            # Extract the screen name from the tweet URL
            parts = tweet_link.split('/')
            tweet_data['screenName'] = parts[-3] if len(parts) > 3 else None

        # The full text of the tweet is contained in the <p> tag within the blockquote
        tweet_text_list = tweet.xpath(".//p//text()")
        tweet_data['tweetText'] = ''.join(tweet_text_list).strip() if tweet_text_list else None

        tweets_data.append(tweet_data)

    return tweets_data


def tokenize(text: str, lang: str) -> List[str]:
    """
    Language-aware tokenization:
    - Lowercase
    - NLTK word_tokenize fallback to regex on error
    - Remove stopwords, keep alphabetic tokens
    """
    sw = _get_stopwords(lang)
    txt = text.lower()
    try:
        toks = word_tokenize(txt, language="spanish" if lang == "es" else "english")
    except Exception:
        toks = re.findall(r"[a-záéíóúüñ]+", txt)
    return [t for t in toks if t.isalpha() and t not in sw]

def get_ngrams(tokens, n):
    return ["_".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

def build_frequency_series(list_of_token_lists, include_ngrams=(1, 2, 3)):
    counter = Counter()
    total_equivalent = 0
    for tokens in list_of_token_lists:
        if 1 in include_ngrams:
            counter.update(tokens)
            total_equivalent += len(tokens)
        if 2 in include_ngrams:
            bigrams = get_ngrams(tokens, 2)
            counter.update(bigrams)
            total_equivalent += max(len(tokens) - 1, 0)
        if 3 in include_ngrams:
            trigrams = get_ngrams(tokens, 3)
            counter.update(trigrams)
            total_equivalent += max(len(tokens) - 2, 0)
    raw = pd.Series(counter).sort_values(ascending=False)
    normalized = (raw / total_equivalent * 1000) if total_equivalent > 0 else raw * 0.0
    return {
        "raw": raw,
        "normalized": normalized,
        "total_token_equivalents": total_equivalent,
    }

def differential_dataframe(group_freqs, group_a, group_b, top_n=50):
    a_norm = group_freqs[group_a]["normalized"]
    b_norm = group_freqs[group_b]["normalized"]
    candidates = set(
        list(group_freqs[group_a]["raw"].head(top_n).index)
        + list(group_freqs[group_b]["raw"].head(top_n).index)
    )
    records = []
    for term in candidates:
        a_raw = int(group_freqs[group_a]["raw"].get(term, 0))
        b_raw = int(group_freqs[group_b]["raw"].get(term, 0))
        a_n = float(a_norm.get(term, 0.0))
        b_n = float(b_norm.get(term, 0.0))
        diff_norm = a_n - b_n
        records.append(
            {
                "term": term,
                f"{group_a}_raw": a_raw,
                f"{group_b}_raw": b_raw,
                f"{group_a}_norm": round(a_n, 4),
                f"{group_b}_norm": round(b_n, 4),
                "differential_norm": round(diff_norm, 4),
            }
        )
    df_diff = pd.DataFrame(records).sort_values("differential_norm", ascending=False)
    return df_diff

# --- Sentiment (language-aware) ---

def _neg_intensity_en(text: str) -> float:
    return float(SIA_EN.polarity_scores(text)["neg"])

def _neg_intensity_es(text: str) -> float:
    """
    Very lightweight Spanish negative-intensity proxy:
    Count negative-lexicon tokens per total tokens (accent-insensitive).
    """
    txt = _strip_accents(text.lower())
    toks = re.findall(r"[a-zñ]+", txt)
    if not toks:
        return 0.0
    neg_hits = sum(1 for t in toks if t in { _strip_accents(w) for w in SPANISH_NEG_LEXICON })
    return float(neg_hits) / float(len(toks))

def sentiment_summary(df: pd.DataFrame, text_col: str, group_col: str, lang: str) -> pd.DataFrame:
    if lang == "es":
        df["negIntensity"] = df[text_col].apply(_neg_intensity_es)
    else:
        df["negIntensity"] = df[text_col].apply(_neg_intensity_en)

    records = []
    for group, sub in df.groupby(group_col):
        neg_nonzero = sub["negIntensity"][sub["negIntensity"] > 0]
        n = len(neg_nonzero)
        if n == 0:
            continue
        mean = neg_nonzero.mean()
        std = neg_nonzero.std(ddof=1)
        se = std / np.sqrt(n) if n > 0 else np.nan
        z = 1.96
        records.append(
            {
                "group": group,
                "count_nonzero": int(n),
                "mean_neg_intensity": float(mean),
                "std_dev": float(std),
                "std_error": float(se),
                "95ci_lower": float(mean - z * se),
                "95ci_upper": float(mean + z * se),
            }
        )
    return pd.DataFrame(records)

def cum_counts(series_of_lists: pd.Series) -> pd.Series:
    flat = sum(series_of_lists.tolist(), [])
    return pd.Series(flat).value_counts()

lexicon_bp = Blueprint("lexicon", __name__, template_folder="templates", url_prefix="/lexicon")


def _load_dataframe_from_upload(file_storage, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Load a DataFrame from an uploaded file.
    Supports: CSV (.csv) and Excel (.xlsx, .xlsm, .xlsb, .xls).
    - If 'sheet_name' is provided, tries to load that sheet; otherwise loads the first sheet.
    - For CSV, attempts normal read; if that fails, retries with delimiter sniffing.
    """
    if not file_storage or not file_storage.filename:
        raise ValueError("No file provided.")

    filename = (file_storage.filename or "").lower().strip()
    content = file_storage.read()
    bio = io.BytesIO(content)

    # Excel?
    if filename.endswith((".xlsx", ".xlsm", ".xlsb", ".xls")):
        # Try engines appropriate to extension
        ext = filename.rsplit(".", 1)[-1]
        engine_candidates = []
        if ext in ("xlsx", "xlsm", "xlsb"):
            engine_candidates = ["openpyxl", None]  # None lets pandas choose
        elif ext == "xls":
            engine_candidates = ["xlrd", None]

        last_err = None
        for eng in engine_candidates:
            try:
                bio.seek(0)
                if sheet_name:
                    return pd.read_excel(bio, sheet_name=sheet_name, engine=eng)
                # No sheet specified → load first sheet
                bio.seek(0)
                xls = pd.ExcelFile(bio, engine=eng)
                first = xls.sheet_names[0] if xls.sheet_names else 0
                return pd.read_excel(xls, sheet_name=first)
            except Exception as e:
                last_err = e

        # If we reach here, we failed to read Excel
        hint = ("Install 'openpyxl' for .xlsx/.xlsm/.xlsb or 'xlrd' for .xls "
                "if the default engine is unavailable.")
        raise ValueError(f"Failed to read Excel file. {hint} Error: {last_err}")

    # CSV (default)
    try:
        bio.seek(0)
        return pd.read_csv(bio)
    except Exception:
        # Retry with delimiter sniffing using the Python engine
        bio.seek(0)
        return pd.read_csv(bio, sep=None, engine="python")
    
@lexicon_bp.route("/", methods=["GET", "POST"])
def upload_and_process():
    results: Dict = {}
    columns: List[str] = []
    lang = _parse_language()  # default 'en' on GET or missing

    # Defaults so we can include them in META even on GET
    filter_query = (request.form.get("filter_query") or "").strip()
    filter_logic = (request.form.get("filter_logic") or "any").lower()   # "any" or "all"
    filter_regex = bool(request.form.get("filter_regex"))

    if request.method == "POST":
        uploaded = request.files.get("csv_file")
        if not uploaded or not uploaded.filename:
            flash("No file uploaded", "warning")
            return redirect(url_for("lexicon.upload_and_process"))

        try:
            excel_sheet = (request.form.get("excel_sheet") or "").strip() or None
            df = _load_dataframe_from_upload(uploaded, sheet_name=excel_sheet)
        except Exception as e:
            flash(f"Failed to read file: {e}", "danger")
            return redirect(url_for("lexicon.upload_and_process"))

        columns = df.columns.tolist()
        text_col = request.form.get("text_column")
        group_col = request.form.get("group_column") or None

        # NEW: toggles for optional NLP work
        enable_tokens = _truthy(request.form.get("enable_tokens", "true"))
        enable_entities = _truthy(request.form.get("enable_entities", "true"))
        enable_sentiment = _truthy(request.form.get("enable_sentiment", "true"))
        # NEW: toggle for social profile extraction
        enable_social_profiles = _truthy(request.form.get("enable_social_profiles", "true"))

        # NEW: optional HTML text column name (for XPath parsing)
        html_col = (request.form.get("html_column") or "").strip() or None
        if html_col and html_col not in df.columns:
            flash(f"HTML column '{html_col}' not found in data", "warning")
            html_col = None
        if html_col and lxml_html is None:
            flash("lxml is not available; HTML parsing is disabled. Install 'lxml' to enable.", "warning")
            html_col = None

        if not text_col or text_col not in df.columns:
            flash("Text column not provided or invalid", "danger")
            return redirect(url_for("lexicon.upload_and_process"))

        # ---------------------------
        # NEW: optional filtering
        # ---------------------------
        if filter_query:
            mask = _build_filter_mask(
                df[text_col],
                filter_query,
                regex=filter_regex,
                logic=filter_logic,
                case_insensitive=True,
                accent_insensitive=True,
            )
            kept = int(mask.sum())
            total = int(len(mask))
            df = df[mask].copy()
            results["filter_info"] = {
                "query": filter_query,
                "logic": filter_logic,
                "regex": filter_regex,
                "kept": kept,
                "total": total,
            }

            if kept == 0:
                # Still proceed, but give a friendly heads-up
                flash("Filter returned zero rows. Try different terms or disable the filter.", "warning")

        # --- Clean + extract ---
        df["clean_text"] = df[text_col].apply(clean_text)

        # Tokens (optional)
        if enable_tokens:
            df["tokens"] = df["clean_text"].apply(lambda t: tokenize(t, lang))
        else:
            df["tokens"] = [[] for _ in range(len(df))]

        # Entities (optional, lazy spaCy)
        if enable_entities:
            nlp = _get_nlp(lang, use_ner=True)
            df["entities"] = df["clean_text"].apply(
                lambda t: [ent.text for ent in nlp(t).ents] if nlp.has_pipe("ner") else []
            )
        else:
            df["entities"] = [[] for _ in range(len(df))]

        # Links, mentions, hashtags, domains
        meta = df[text_col].apply(extract_meta)
        df["mentions"] = meta.apply(lambda x: x[0])
        df["hashtags"] = meta.apply(lambda x: x[1])
        df["links"] = meta.apply(lambda x: x[2])
        df["domains"] = meta.apply(lambda x: x[3])

        # NEW: Social media profiles (from original text with URLs)
        if enable_social_profiles:
            df["social_profiles"] = df[text_col].apply(extract_social_media_profiles)
            df["social_profile_pairs"] = df["social_profiles"].apply(
                lambda d: [f"{platform}:{handle}" for platform, handles in (d or {}).items() for handle in handles]
            )
            df["social_platforms"] = df["social_profiles"].apply(
                lambda d: [platform for platform, handles in (d or {}).items() for _ in handles]
            )

        # NEW: HTML parsing with XPath (embedded links + tweets)
        if html_col:
            df["__html_root"] = df[html_col].apply(_safe_parse_html)
            df["embedded_links_xpath"] = df["__html_root"].apply(
                lambda node: (extract_embedded_links(node) if node is not None else [])
            )
            df["embedded_tweets"] = df["__html_root"].apply(
                lambda node: (extract_tweet_details(node) if node is not None else [])
            )
            # Convenience lists for counting
            df["tweet_ids"] = df["embedded_tweets"].apply(
                lambda lst: [d.get("tweetId") for d in lst if isinstance(d, dict) and d.get("tweetId")]
            )
            df["tweet_screen_names"] = df["embedded_tweets"].apply(
                lambda lst: [d.get("screenName") for d in lst if isinstance(d, dict) and d.get("screenName")]
            )
            # Clean up heavy roots to keep memory down
            del df["__html_root"]

        # Grouping
        if group_col and group_col in df.columns:
            groups = df[group_col].dropna().unique()
        else:
            group_col = "__all__"
            df[group_col] = "all"
            groups = ["all"]

        # Frequencies (optional; only when tokens are enabled)
        if enable_tokens:
            group_freqs = {}
            for g in groups:
                token_lists = df[df[group_col] == g]["tokens"].tolist()
                group_freqs[g] = build_frequency_series(token_lists, include_ngrams=(1, 2, 3))

            # Summaries per group
            summary_per_group = {}
            for g in group_freqs:
                norm = group_freqs[g]["normalized"].sort_values(ascending=False).head(1000)
                raw = group_freqs[g]["raw"].sort_values(ascending=False).head(1000)
                summary_per_group[g] = {
                    "top_normalized": norm.round(4).to_dict(),
                    "top_raw": raw.to_dict(),
                    "total_token_equivalents": group_freqs[g]["total_token_equivalents"],
                }
            results["group_freqs"] = summary_per_group

            # Differential if exactly two groups
            if len(groups) == 2:
                a, b = list(groups)
                diff_df = differential_dataframe(group_freqs, a, b, top_n=100)
                results["differential"] = diff_df.to_dict(orient="records")
                results["differential_groups"] = (a, b)

        # Sentiment (optional, language-aware)
        if enable_sentiment:
            sentiment_df = sentiment_summary(df, "clean_text", group_col, lang)
            results["sentiment_summary"] = sentiment_df.to_dict(orient="records")
        else:
            results["sentiment_summary"] = []

        # Meta counts
        meta_summary = {}
        for g in groups:
            subset = df[df[group_col] == g]
            meta_summary[g] = {
                "mentions": cum_counts(subset["mentions"]).to_dict(),
                "hashtags": cum_counts(subset["hashtags"]).to_dict(),
                "links": cum_counts(subset["links"]).to_dict(),
                "domains": cum_counts(subset["domains"]).to_dict(),
                "entities": cum_counts(subset["entities"]).to_dict(),
            }
            # NEW: add HTML-derived meta if present
            if "embedded_links_xpath" in df.columns:
                meta_summary[g]["embedded_links_xpath"] = cum_counts(subset["embedded_links_xpath"]).to_dict()
            if "tweet_ids" in df.columns:
                meta_summary[g]["tweet_ids"] = cum_counts(subset["tweet_ids"]).to_dict()
            if "tweet_screen_names" in df.columns:
                meta_summary[g]["tweet_screen_names"] = cum_counts(subset["tweet_screen_names"]).to_dict()
            # NEW: add social profiles if present
            if "social_profile_pairs" in df.columns:
                meta_summary[g]["social_profile_pairs"] = cum_counts(subset["social_profile_pairs"]).to_dict()
            if "social_platforms" in df.columns:
                meta_summary[g]["social_platforms"] = cum_counts(subset["social_platforms"]).to_dict()
        results["meta"] = meta_summary

    # Excel export (unified group handling + filter META)
    if request.form.get("format") == "excel":
        sheets: Dict[str, pd.DataFrame] = {}

        # === Raw & normalized top terms (single tables with 'group' col) ===
        raw_rows, norm_rows = [], []
        for group, summary in results.get("group_freqs", {}).items():
            for term, count in summary.get("top_raw", {}).items():
                raw_rows.append({"group": group, "term": term, "count_raw": count})
            for term, val in summary.get("top_normalized", {}).items():
                norm_rows.append({"group": group, "term": term, "normalized_per_thousand": val})

        if raw_rows:
            sheets["TOP_RAW"] = (
                pd.DataFrame(raw_rows)
                .sort_values(["group", "count_raw"], ascending=[True, False])
                .reset_index(drop=True)
            )
        if norm_rows:
            sheets["TOP_NORMALIZED"] = (
                pd.DataFrame(norm_rows)
                .sort_values(["group", "normalized_per_thousand"], ascending=[True, False])
                .reset_index(drop=True)
            )

        # === Group totals ===
        totals_rows = []
        for group, summary in results.get("group_freqs", {}).items():
            totals_rows.append({
                "group": group,
                "total_token_equivalents": summary.get("total_token_equivalents", 0)
            })
        if totals_rows:
            sheets["GROUP_TOTALS"] = pd.DataFrame(totals_rows).sort_values("group").reset_index(drop=True)

        # === Meta counts (mentions, hashtags, links, domains, entities, + HTML) ===
        meta_keys = ["mentions", "hashtags", "links", "domains", "entities"]
        # Add optional HTML-derived keys if present
        any_meta = results.get("meta") or {}
        if any(k in (any_meta.get(g, {}) or {}) for g in any_meta for k in ["embedded_links_xpath"]):
            meta_keys.append("embedded_links_xpath")
        if any(k in (any_meta.get(g, {}) or {}) for g in any_meta for k in ["tweet_ids"]):
            meta_keys.append("tweet_ids")
        if any(k in (any_meta.get(g, {}) or {}) for g in any_meta for k in ["tweet_screen_names"]):
            meta_keys.append("tweet_screen_names")
        # NEW: add social profile meta keys if present
        if any(k in (any_meta.get(g, {}) or {}) for g in any_meta for k in ["social_profile_pairs"]):
            meta_keys.append("social_profile_pairs")
        if any(k in (any_meta.get(g, {}) or {}) for g in any_meta for k in ["social_platforms"]):
            meta_keys.append("social_platforms")

        for meta_key in meta_keys:
            meta_rows = []
            for group, metas in results.get("meta", {}).items():
                for val, count in (metas or {}).get(meta_key, {}).items():
                    meta_rows.append({"group": group, meta_key[:-1] if meta_key.endswith("s") else meta_key: val, "count": count})
            if meta_rows:
                sheets[meta_key.upper()] = (
                    pd.DataFrame(meta_rows)
                    .sort_values(["group", "count"], ascending=[True, False])
                    .reset_index(drop=True)
                )

        # === Differential (own sheet) ===
        if "differential" in results and results.get("differential_groups"):
            sheets["DIFFERENTIAL"] = pd.DataFrame(results["differential"])

        # === Sentiment (own sheet) ===
        if results.get("sentiment_summary"):
            sheets["SENTIMENT_SUMMARY"] = pd.DataFrame(results["sentiment_summary"])

        # NEW: Detailed embedded tweets sheet (flattened)
        if "embedded_tweets" in df.columns:
            tweet_rows = []
            for _, row in df.iterrows():
                grp = row[group_col]
                for t in row["embedded_tweets"]:
                    if not isinstance(t, dict):
                        continue
                    tweet_rows.append({
                        "group": grp,
                        "tweetId": t.get("tweetId"),
                        "screenName": t.get("screenName"),
                        "tweetLink": t.get("tweetLink"),
                        "tweetText": t.get("tweetText"),
                    })
            if tweet_rows:
                sheets["EMBEDDED_TWEETS"] = pd.DataFrame(tweet_rows).sort_values(["group", "screenName"]).reset_index(drop=True)

        # === META (record filter + language + html column + flags) ===
        fi = results.get("filter_info", {})
        sheets["META"] = pd.DataFrame([{
            "language": lang,
            "filter_query": fi.get("query", ""),
            "filter_logic": fi.get("logic", ""),
            "filter_regex": fi.get("regex", False),
            "filtered_rows_kept": fi.get("kept", None),
            "filtered_rows_total": fi.get("total", None),
            "html_column": (request.form.get("html_column") or "").strip(),
            "enable_tokens": _truthy(request.form.get("enable_tokens", "true")),
            "enable_entities": _truthy(request.form.get("enable_entities", "true")),
            "enable_sentiment": _truthy(request.form.get("enable_sentiment", "true")),
            # NEW: record social profile flag
            "enable_social_profiles": _truthy(request.form.get("enable_social_profiles", "true")),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }])

        filename = f"lexical_sentiment_analysis_{lang}_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".xlsx"
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            for sheet_name, data in sheets.items():
                safe = sheet_name[:31].replace("/", "_").replace("\\", "_")
                data.to_excel(writer, sheet_name=safe, index=False)
        output.seek(0)
        return send_file(
            output,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name=filename,
        )

    return render_template(
        "lexicon.html",
        results=results,
        columns=columns,
        request=request,
    )

