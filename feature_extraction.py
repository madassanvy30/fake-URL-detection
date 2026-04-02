"""
=============================================================
  feature_extraction.py
  Fake Website Detection System
  Role: Extract numerical features from raw URL strings
=============================================================
"""

import re
import socket
from urllib.parse import urlparse
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────

def _safe_parse(url: str) -> urlparse:
    """Parse URL; prepend http:// if scheme is missing."""
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return urlparse(url)


def _has_ip_address(url: str) -> int:
    """Return 1 if the URL hostname looks like an IPv4 address."""
    parsed = _safe_parse(url)
    hostname = parsed.hostname or ""
    ipv4_pattern = re.compile(
        r"^(\d{1,3}\.){3}\d{1,3}$"
    )
    return int(bool(ipv4_pattern.match(hostname)))


def _domain_resolves(url: str) -> int:
    """
    Return 1 if the domain resolves via DNS (basic reachability check).
    Returns 0 on failure or timeout.
    NOTE: This makes a live network call – disable if offline.
    """
    try:
        parsed = _safe_parse(url)
        hostname = parsed.hostname or ""
        if hostname:
            socket.setdefaulttimeout(2)
            socket.gethostbyname(hostname)
            return 1
    except Exception:
        pass
    return 0


# ─────────────────────────────────────────────────────────────
# Core feature extraction
# ─────────────────────────────────────────────────────────────

SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq",   # Free Freenom TLDs abused heavily
    ".xyz", ".top", ".ru", ".cn", ".pw",
    ".club", ".online", ".site", ".info",
    ".biz", ".link", ".click",
}

BRAND_KEYWORDS = [
    "paypal", "amazon", "google", "apple", "microsoft",
    "facebook", "instagram", "twitter", "netflix", "ebay",
    "bank", "secure", "login", "verify", "account",
    "update", "confirm", "billing", "payment", "support",
]


def extract_features(url: str, dns_check: bool = False) -> dict:
    """
    Extract a fixed set of numerical features from a single URL.

    Parameters
    ----------
    url       : raw URL string
    dns_check : set True to perform a live DNS lookup (slower)

    Returns
    -------
    dict of feature_name → numeric value
    """
    parsed   = _safe_parse(url)
    hostname = parsed.hostname or ""
    path     = parsed.path     or ""
    full_url = url.lower()

    # ── 1. Length-based features ───────────────────────────
    url_length      = len(url)
    hostname_length = len(hostname)
    path_length     = len(path)

    # ── 2. Protocol ────────────────────────────────────────
    has_https = int(parsed.scheme == "https")

    # ── 3. Dot / subdomain count ───────────────────────────
    num_dots        = url.count(".")
    num_subdomains  = max(0, len(hostname.split(".")) - 2)

    # ── 4. Special-character counts ────────────────────────
    num_hyphens     = url.count("-")
    num_at_signs    = url.count("@")
    num_underscores = url.count("_")
    num_slashes     = url.count("/")
    num_question    = url.count("?")
    num_equals      = url.count("=")
    num_ampersand   = url.count("&")
    num_percent     = url.count("%")        # URL-encoded chars → obfuscation
    num_digits_in_domain = sum(c.isdigit() for c in hostname)

    # ── 5. IP address in URL ────────────────────────────────
    has_ip_address = _has_ip_address(url)

    # ── 6. Suspicious TLD ──────────────────────────────────
    has_suspicious_tld = int(
        any(hostname.endswith(tld) for tld in SUSPICIOUS_TLDS)
    )

    # ── 7. Brand impersonation keywords in domain ──────────
    num_brand_keywords = sum(
        kw in hostname for kw in BRAND_KEYWORDS
    )

    # ── 8. Path / query entropy (randomness) ───────────────
    def _char_entropy(s: str) -> float:
        if not s:
            return 0.0
        probs = [s.count(c) / len(s) for c in set(s)]
        return -sum(p * np.log2(p) for p in probs)

    path_entropy = _char_entropy(path)

    # ── 9. Redirect double-slash ───────────────────────────
    has_double_slash_redirect = int("//" in path)

    # ── 10. Prefix / suffix hyphen in domain ───────────────
    has_prefix_suffix_hyphen = int("-" in hostname)

    # ── 11. Short URL service indicators ───────────────────
    SHORT_SERVICES = {"bit.ly", "tinyurl.com", "goo.gl", "ow.ly",
                      "t.co", "is.gd", "cli.gs", "buff.ly"}
    is_short_url = int(hostname in SHORT_SERVICES or url_length < 22)

    # ── 12. Digit ratio in full URL ────────────────────────
    digit_ratio = sum(c.isdigit() for c in url) / max(len(url), 1)

    # ── 13. DNS resolution (optional, live) ────────────────
    dns_resolves = _domain_resolves(url) if dns_check else -1

    features = {
        # Length
        "url_length":               url_length,
        "hostname_length":          hostname_length,
        "path_length":              path_length,
        # Protocol
        "has_https":                has_https,
        # Dots / subdomains
        "num_dots":                 num_dots,
        "num_subdomains":           num_subdomains,
        # Special characters
        "num_hyphens":              num_hyphens,
        "num_at_signs":             num_at_signs,
        "num_underscores":          num_underscores,
        "num_slashes":              num_slashes,
        "num_question":             num_question,
        "num_equals":               num_equals,
        "num_ampersand":            num_ampersand,
        "num_percent":              num_percent,
        "num_digits_in_domain":     num_digits_in_domain,
        # IP & TLD
        "has_ip_address":           has_ip_address,
        "has_suspicious_tld":       has_suspicious_tld,
        # Brand impersonation
        "num_brand_keywords":       num_brand_keywords,
        # Entropy / obfuscation
        "path_entropy":             round(path_entropy, 4),
        "digit_ratio":              round(digit_ratio, 4),
        # Structure
        "has_double_slash_redirect":has_double_slash_redirect,
        "has_prefix_suffix_hyphen": has_prefix_suffix_hyphen,
        "is_short_url":             is_short_url,
        # DNS (set to -1 when not checked)
        "dns_resolves":             dns_resolves,
    }
    return features


def extract_features_dataframe(urls: pd.Series,
                                dns_check: bool = False) -> pd.DataFrame:
    """
    Vectorised wrapper: extract features for every URL in a Series.
    Returns a DataFrame (one row per URL, one column per feature).
    """
    records = [extract_features(url, dns_check=dns_check) for url in urls]
    df = pd.DataFrame(records)

    # Drop DNS column if we didn't use it (all -1s add noise)
    if not dns_check:
        df = df.drop(columns=["dns_resolves"])

    print(f"[FeatureExtraction] Shape: {df.shape}  "
          f"Features: {list(df.columns)}")
    return df


# ── Quick self-test ──────────────────────────────────────────
if __name__ == "__main__":
    test_urls = [
        "https://www.google.com",
        "http://paypa1.com-secure-login.tk/account",
        "http://192.168.1.1/bank-login",
    ]
    for u in test_urls:
        feats = extract_features(u)
        print(f"\nURL: {u}")
        for k, v in feats.items():
            print(f"  {k:35s}: {v}")
