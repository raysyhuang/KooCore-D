"""
Analysis Functions

Headline analysis, keyword detection, and other analytical utilities.
"""

from __future__ import annotations
from typing import Optional


# Dilution risk keywords
DILUTION_KEYWORDS = [
    "offering", "secondary", "at-the-market", "atm", "shelf", "convertible",
    "warrant", "dilution", "share issuance", "equity raise", "capital raise"
]

# Catalyst keywords by category
CATALYST_KEYWORDS = {
    "earnings": ["earnings", "eps", "revenue", "guidance", "beat", "miss"],
    "fda": ["fda", "approval", "pdufa", "nda", "clinical", "trial"],
    "merger": ["merger", "acquisition", "m&a", "takeover", "deal"],
    "product": ["launch", "product", "partnership", "contract", "deal"],
    "upgrade": ["upgrade", "downgrade", "initiate", "price target", "pt raise"],
    "regulatory": ["sec", "regulatory", "approval", "clearance"],
}


def analyze_headlines(titles: list[str]) -> dict:
    """
    Analyze headlines for dilution risk and catalyst keywords.
    
    Args:
        titles: List of headline title strings
    
    Returns:
        Dict with:
            - dilution_flag: 0 or 1
            - catalyst_tags: comma-separated string of catalyst categories
    """
    if not titles:
        return {"dilution_flag": 0, "catalyst_tags": ""}
    
    # Combine all titles into searchable text
    text = " | ".join([str(x).lower() for x in titles if str(x).strip()])
    
    # Check for dilution keywords
    dilution = any(k in text for k in DILUTION_KEYWORDS)
    
    # Check for catalyst keywords
    tags = []
    for category, keywords in CATALYST_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            tags.append(category)
    
    return {
        "dilution_flag": int(dilution),
        "catalyst_tags": ",".join(sorted(set(tags)))
    }

