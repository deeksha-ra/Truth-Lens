"""
TruthLens Backend Classifier Engine
-----------------------------------
Implements tiered classification:
  1. Fine-tuned Transformer (PyTorch/HuggingFace) if checkpoint available
  2. TF-IDF + Logistic Regression baseline model if saved joblib available
  3. Built-in Heuristic Pattern Engine as guaranteed offline fallback
"""

import os
import re
import math
from typing import Dict, Any, List, Tuple

# Heuristic patterns
SENSATIONAL_PATTERNS = [
    r"\b(shocking|mind-blowing|unbelievable|miracle cure|secret revealed|exposed|they don'?t want you to know)\b",
    r"\b(guaranteed|100% effective|cure all|magic pill|instant cure|burns \d+ calories)\b",
    r"\b(secret lab|covert operation|conspiracy|illuminati|microchips in|chemtrails|flat earth|hollow earth)\b",
    r"\b(doctors are furious|banned in \d+ states|leaked documents show|bombshell revelation)\b"
]

HEDGE_PATTERNS = [
    r"\b(sources say|reportedly|unconfirmed reports|allegedly|rumors suggest|insiders claim)\b",
    r"\b(some people believe|it is rumored|anonymous sources|may possibly|could potentially)\b",
    r"\b(speculation suggests|purportedly|unverified claims|whispers in)\b"
]

VERIFIED_PATTERNS = [
    r"\b(published in (nature|science|the lancet|cell|new england journal of medicine))\b",
    r"\b(according to (official data|the bureau of labor statistics|census bureau|nasa|who|cdc|federal reserve))\b",
    r"\b(peer-reviewed (study|research|journal)|statistically significant|randomized controlled trial)\b",
    r"\b(data released by|official statistics show|treaty ratified by|passed legislation)\b"
]

class TruthLensClassifier:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.model_tier = "heuristic"
        self.transformer_model = None
        self.transformer_tokenizer = None
        self.baseline_bundle = None
        
        self._load_available_models()

    def _load_available_models(self):
        # 1. Try PyTorch / HuggingFace Transformer
        transformer_path = os.path.join(self.model_dir, "distilbert_truthlens")
        if os.path.exists(transformer_path):
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self.transformer_tokenizer = AutoTokenizer.from_pretrained(transformer_path)
                self.transformer_model = AutoModelForSequenceClassification.from_pretrained(transformer_path)
                self.transformer_model.eval()
                self.model_tier = "transformer"
                print(f"[Classifier] Successfully loaded Transformer model from {transformer_path}")
                return
            except Exception as e:
                print(f"[Classifier] Could not load transformer: {e}")

        # 2. Try Baseline TF-IDF + Logistic Regression
        baseline_path = os.path.join(self.model_dir, "baseline_tfidf.joblib")
        if os.path.exists(baseline_path):
            try:
                import joblib
                self.baseline_bundle = joblib.load(baseline_path)
                self.model_tier = "baseline_tfidf"
                print(f"[Classifier] Successfully loaded Baseline model from {baseline_path}")
                return
            except Exception as e:
                print(f"[Classifier] Could not load baseline: {e}")

        # 3. Fallback to Heuristic
        self.model_tier = "heuristic"
        print("[Classifier] Initialized with Heuristic Rule Engine fallback.")

    def classify_heuristic(self, text: str) -> Tuple[str, float, str]:
        """Pattern-based heuristic classification with reasoning."""
        lower_text = text.lower()

        sensational_matches = []
        for pattern in SENSATIONAL_PATTERNS:
            found = re.findall(pattern, lower_text, flags=re.IGNORECASE)
            if found:
                sensational_matches.extend(found)

        hedge_matches = []
        for pattern in HEDGE_PATTERNS:
            found = re.findall(pattern, lower_text, flags=re.IGNORECASE)
            if found:
                hedge_matches.extend(found)

        verified_matches = []
        for pattern in VERIFIED_PATTERNS:
            found = re.findall(pattern, lower_text, flags=re.IGNORECASE)
            if found:
                verified_matches.extend(found)

        # Check all-caps words (excluding short acronyms)
        words = text.split()
        caps_words = [w for w in words if len(w) > 3 and w.isupper() and w.isalpha()]
        excessive_exclamation = text.count("!") >= 2 or "?!" in text

        # Scoring Logic
        if sensational_matches or (excessive_exclamation and caps_words):
            score = 0.85 + (0.05 * min(len(sensational_matches), 2))
            reason = "Contains highly sensationalized phrasing or extreme claims."
            if sensational_matches:
                reason += f" Detected keyword: '{sensational_matches[0]}'."
            return "likely-false", min(score, 0.96), reason

        if hedge_matches:
            score = 0.70 + (0.05 * min(len(hedge_matches), 3))
            reason = f"Relies on unverified hedge framing ('{hedge_matches[0]}')."
            return "uncertain", min(score, 0.88), reason

        if verified_matches:
            score = 0.88 + (0.04 * min(len(verified_matches), 2))
            reason = f"Grounding in institutional or peer-reviewed citation ('{verified_matches[0]}')."
            return "verified", min(score, 0.97), reason

        # Neutral statement without obvious red flags -> default verified with moderate confidence
        return "verified", 0.78, "Standard descriptive statement with no obvious red-flag patterns."

    def classify(self, text: str, sensitivity: str = "medium") -> Dict[str, Any]:
        """Classify a single claim and apply sensitivity calibration."""
        tier = "verified"
        confidence = 0.80
        reason = ""
        model_used = self.model_tier

        # 1. Transformer inference
        if self.transformer_model is not None and self.transformer_tokenizer is not None:
            try:
                import torch
                inputs = self.transformer_tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=128, padding=True
                )
                with torch.no_grad():
                    outputs = self.transformer_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1).squeeze().tolist()
                
                # Probs index: 0=verified, 1=uncertain, 2=likely-false
                idx = int(torch.argmax(outputs.logits, dim=1).item())
                labels = ["verified", "uncertain", "likely-false"]
                tier = labels[idx]
                confidence = float(probs[idx])
                
                if tier == "likely-false":
                    reason = "Deep-learning transformer model identified statistical patterns consistent with misinformation."
                elif tier == "uncertain":
                    reason = "Claim exhibits ambiguous or weakly-substantiated assertions."
                else:
                    reason = "Statement exhibits high semantic consistency with verified reports."
            except Exception as e:
                print(f"[Classifier] Transformer inference error: {e}, falling back.")
                tier, confidence, reason = self.classify_heuristic(text)
                model_used = "heuristic_fallback"

        # 2. Baseline TF-IDF inference
        elif self.baseline_bundle is not None:
            try:
                vec = self.baseline_bundle["vectorizer"]
                clf = self.baseline_bundle["classifier"]
                X = vec.transform([text])
                pred_idx = int(clf.predict(X)[0])
                probs = clf.predict_proba(X)[0]
                labels = ["verified", "uncertain", "likely-false"]
                tier = labels[pred_idx]
                confidence = float(probs[pred_idx])
                
                if tier == "likely-false":
                    reason = "N-gram statistical baseline flagged uncharacteristic phrasing associated with unverified claims."
                elif tier == "uncertain":
                    reason = "Statistical confidence falls in intermediate uncertainty range."
                else:
                    reason = "N-gram signature aligns with verified informational text."
            except Exception as e:
                print(f"[Classifier] Baseline inference error: {e}, falling back.")
                tier, confidence, reason = self.classify_heuristic(text)
                model_used = "heuristic_fallback"

        # 3. Heuristic engine
        else:
            tier, confidence, reason = self.classify_heuristic(text)

        # Apply Sensitivity Filter:
        # False positives on verified content must be minimized.
        # If sensitivity is 'low', require higher confidence (> 0.82) to flag as 'likely-false' or 'uncertain'.
        if sensitivity == "low":
            if tier in ["likely-false", "uncertain"] and confidence < 0.82:
                tier = "verified"
                confidence = 0.72
                reason = "De-escalated to verified under low sensitivity threshold."
        elif sensitivity == "high":
            # Under high sensitivity, even lower confidence signals get flagged
            if tier == "verified" and confidence < 0.65:
                tier = "uncertain"
                confidence = 0.68
                reason = "Elevated to uncertain under strict high-sensitivity inspection."

        return {
            "claim": text,
            "tier": tier,
            "confidence": round(confidence, 4),
            "reason": reason,
            "model_tier": model_used,
            "sensitivity": sensitivity
        }

    def classify_batch(self, texts: List[str], sensitivity: str = "medium") -> List[Dict[str, Any]]:
        """Batch classification for on-screen sentence batches."""
        return [self.classify(t, sensitivity=sensitivity) for t in texts]
