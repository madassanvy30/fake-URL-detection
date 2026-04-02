"""
=============================================================
  data_preprocessing.py
  Fake Website Detection System
  Role: Generate / load dataset and preprocess it for ML
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle, os

# ── Reproducibility ─────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ─────────────────────────────────────────────────────────────
# 1.  SAMPLE DATASET
#     In production you would replace this with the UCI
#     Phishing Dataset or any real labelled CSV.
#     Label: 1 = Legitimate,  0 = Fake / Phishing
# ─────────────────────────────────────────────────────────────

LEGITIMATE_URLS = [
    "https://www.google.com",
    "https://www.github.com",
    "https://www.microsoft.com",
    "https://www.amazon.com",
    "https://www.wikipedia.org",
    "https://www.stackoverflow.com",
    "https://www.linkedin.com",
    "https://www.twitter.com",
    "https://www.facebook.com",
    "https://www.apple.com",
    "https://www.netflix.com",
    "https://www.reddit.com",
    "https://www.youtube.com",
    "https://www.instagram.com",
    "https://www.dropbox.com",
    "https://www.spotify.com",
    "https://www.adobe.com",
    "https://www.salesforce.com",
    "https://www.oracle.com",
    "https://www.ibm.com",
    "https://www.paypal.com",
    "https://www.ebay.com",
    "https://www.walmart.com",
    "https://www.target.com",
    "https://www.cnn.com",
    "https://www.bbc.com",
    "https://www.nytimes.com",
    "https://www.weather.com",
    "https://www.zoom.us",
    "https://www.slack.com",
    "https://docs.python.org",
    "https://www.kaggle.com",
    "https://www.coursera.org",
    "https://www.udemy.com",
    "https://www.medium.com",
]

PHISHING_URLS = [
    "http://paypa1.com-secure-login.tk/account",
    "http://192.168.1.1/bank-login",
    "http://www.amazon-security-alert.com/verify",
    "http://secure-paypal-update.com/login",
    "http://google.com.phishing-site.ru/signin",
    "http://faceb00k-login.tk/secure",
    "http://apple-id-locked.com/verify-now",
    "http://netflix-billing-update.xyz/payment",
    "http://microsoft-account-alert.com/login",
    "http://wellsfargo-secure.tk/online-banking",
    "http://update-your-info.net/bank-login",
    "http://secure-ebay-account.com/signin",
    "http://login-amazon-support.com/account",
    "http://alert-bankofamerica.tk/account",
    "http://chase-secure-login.com/verify",
    "http://172.16.0.1/phishing-page",
    "http://irs-gov-refund.tk/claim",
    "http://free-gift-amazon.tk/win",
    "http://urgent-account-verify.com/login",
    "http://suspended-account-restore.com/reactivate",
    "http://instagram-winner.tk/prize",
    "http://whatsapp-update-required.net/verify",
    "http://dropbox-share-file.com-hack.ru/login",
    "http://linkedin-security-alert.tk/verify",
    "http://twitter-account-suspended.com/restore",
    "http://paypal-limitation.tk/resolve",
    "http://citibank-alert.com/secure-login",
    "http://usps-tracking-update.com/package",
    "http://steam-trade-offer.tk/accept",
    "http://coinbase-verification.com.phish.net/verify",
    "http://crypto-wallet-recover.tk/login",
    "http://zoom-meeting-invite.com-phish.ru/join",
    "http://microsoft365-renewal.tk/billing",
    "http://icloud-locked.com/unlock",
    "http://support-apple-id.tk/verify",
]


def generate_dataset() -> pd.DataFrame:
    """
    Build a labelled DataFrame from the sample URL lists above.
    Returns a DataFrame with columns: [url, label]
    """
    legit_df = pd.DataFrame({"url": LEGITIMATE_URLS, "label": 1})
    fake_df  = pd.DataFrame({"url": PHISHING_URLS,   "label": 0})
    df = pd.concat([legit_df, fake_df], ignore_index=True)

    # Shuffle rows so classes are interleaved
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"[Dataset] Total samples: {len(df)}  "
          f"(Legitimate: {df['label'].sum()}, Fake: {(df['label']==0).sum()})")
    return df


def load_or_generate_dataset(csv_path: str = None) -> pd.DataFrame:
    """
    If a CSV path is provided and the file exists, load it.
    Otherwise, fall back to the built-in sample dataset.
    Expected CSV columns: url, label  (1=Legit, 0=Fake)
    """
    if csv_path and os.path.exists(csv_path):
        print(f"[Dataset] Loading from {csv_path}")
        df = pd.read_csv(csv_path)
        # Basic sanity checks
        assert "url"   in df.columns, "CSV must have a 'url' column"
        assert "label" in df.columns, "CSV must have a 'label' column"
        df = df.dropna(subset=["url", "label"])
        df["label"] = df["label"].astype(int)
    else:
        print("[Dataset] No external CSV found – using built-in sample dataset.")
        df = generate_dataset()
    return df


def split_and_scale(X: pd.DataFrame, y: pd.Series,
                    test_size: float = 0.2):
    """
    Train/test split → StandardScaler fit on train → transform both sets.
    Returns: X_train, X_test, y_train, y_test, scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Persist scaler for use in the app
    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("[Preprocessing] Scaler saved → models/scaler.pkl")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ── Quick self-test ──────────────────────────────────────────
if __name__ == "__main__":
    df = load_or_generate_dataset()
    print(df.head())
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")
