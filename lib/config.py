#!/usr/bin/env python3
"""
Configuration for Cross-Script Rescue Experiments
==================================================
Date: January 2, 2026 (Updated)

This module centralizes all experimental configuration with proper methodology.

Updates:
- Added all Gemma 3 model sizes (270M, 1B, 4B, 12B)
- Expanded word list to 200+ pairs
- Added Gemma Scope transcoder mappings
"""

from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional
import json
import os
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for a model."""

    name: str
    hf_id: str
    scope_repo: Optional[str]  # May not exist for all models
    n_layers: int
    d_model: int

    # Layers to test (based on model depth)
    target_layers: List[int] = field(default_factory=list)

    # Expected behavior
    expected_hindi_accuracy: float = 0.7  # Should know Hindi
    expected_ood_accuracy: float = 0.3  # Should fail on OOD


# Model configurations
# Gemma Scope 2 provides SAEs and transcoders for all Gemma 3 models
# Cross-Layer Transcoders (CLTs) available for 270M and 1B only
# Repo format: google/gemma-scope-2-{size}-{pt|it}
MODELS = {
    "270m": ModelConfig(
        name="270m",
        hf_id="google/gemma-3-270m-it",
        scope_repo="google/gemma-scope-2-270m-it",  # Gemma Scope 2
        n_layers=18,
        d_model=1024,
        target_layers=list(range(18)),  # ALL layers
        expected_hindi_accuracy=0.3,  # Smallest model, lowest expectation
        expected_ood_accuracy=0.1,
    ),
    "1b": ModelConfig(
        name="1b",
        hf_id="google/gemma-3-1b-it",
        scope_repo="google/gemma-scope-2-1b-it",  # Gemma Scope 2
        n_layers=26,
        d_model=2048,
        target_layers=list(range(26)),  # ALL layers for comprehensive sweep
        expected_hindi_accuracy=0.5,  # Smaller model, lower expectation
        expected_ood_accuracy=0.2,
    ),
    "4b": ModelConfig(
        name="4b",
        hf_id="google/gemma-3-4b-it",
        scope_repo="google/gemma-scope-2-4b-it",  # Gemma Scope 2
        n_layers=34,
        d_model=2560,
        target_layers=list(range(34)),  # ALL layers
        expected_hindi_accuracy=0.7,
        expected_ood_accuracy=0.25,
    ),
    "12b": ModelConfig(
        name="12b",
        hf_id="google/gemma-3-12b-it",
        scope_repo="google/gemma-scope-2-12b-it",  # Gemma Scope 2
        n_layers=48,
        d_model=3840,
        target_layers=list(range(48)),  # ALL layers
        expected_hindi_accuracy=0.85,  # Larger model, better Hindi
        expected_ood_accuracy=0.4,  # May already handle some OOD
    ),
}


@dataclass
class ExperimentConfig:
    """Master experiment configuration."""

    # Random seeds for reproducibility
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1337])

    # ICL configuration
    n_icl_examples: int = 5

    # Test configuration
    n_test_samples: int = 50  # Per seed (non-overlapping with ICL)

    # Patching configuration
    topk_values: List[int] = field(default_factory=lambda: [5, 10, 25, 50, 100])

    # Token inflation threshold
    max_token_inflation: float = 5.0

    # Statistical thresholds
    significance_alpha: float = 0.05

    # Device
    device: str = "cuda"
    dtype: str = "bfloat16"

    # Output
    results_dir: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(__file__), "..", "results", "fresh"
        )
    )


# ============================================================================
# EXPANDED WORD LIST (200+ pairs)
# ============================================================================
# Organized by category for analysis
# Each entry: Hindi (Devanagari), Telugu-script transliteration, English meaning

HINDI_TELUGU_WORDS = [
    # ==================== GREETINGS (10) ====================
    {"hindi": "नमस्ते", "telugu": "నమస్తే", "english": "hello", "category": "greeting"},
    {
        "hindi": "धन्यवाद",
        "telugu": "ధన్యవాద్",
        "english": "thank you",
        "category": "greeting",
    },
    {"hindi": "कृपया", "telugu": "కృపయా", "english": "please", "category": "greeting"},
    {"hindi": "स्वागत", "telugu": "స్వాగత్", "english": "welcome", "category": "greeting"},
    {"hindi": "अलविदा", "telugu": "అల్విదా", "english": "goodbye", "category": "greeting"},
    {"hindi": "शुभ", "telugu": "శుభ్", "english": "auspicious", "category": "greeting"},
    {
        "hindi": "प्रणाम",
        "telugu": "ప్రణామ్",
        "english": "greetings",
        "category": "greeting",
    },
    {
        "hindi": "नमस्कार",
        "telugu": "నమస్కార్",
        "english": "salutation",
        "category": "greeting",
    },
    {"hindi": "जय", "telugu": "జయ్", "english": "victory", "category": "greeting"},
    {"hindi": "शांति", "telugu": "శాంతి", "english": "peace", "category": "greeting"},
    # ==================== FAMILY (15) ====================
    {"hindi": "माता", "telugu": "మాతా", "english": "mother", "category": "family"},
    {"hindi": "पिता", "telugu": "పితా", "english": "father", "category": "family"},
    {"hindi": "भाई", "telugu": "భాఈ", "english": "brother", "category": "family"},
    {"hindi": "बहन", "telugu": "బహన్", "english": "sister", "category": "family"},
    {"hindi": "बेटा", "telugu": "బేటా", "english": "son", "category": "family"},
    {"hindi": "बेटी", "telugu": "బేటీ", "english": "daughter", "category": "family"},
    {"hindi": "दादा", "telugu": "దాదా", "english": "grandfather", "category": "family"},
    {"hindi": "दादी", "telugu": "దాదీ", "english": "grandmother", "category": "family"},
    {"hindi": "चाचा", "telugu": "చాచా", "english": "uncle", "category": "family"},
    {"hindi": "चाची", "telugu": "చాచీ", "english": "aunt", "category": "family"},
    {"hindi": "पति", "telugu": "పతి", "english": "husband", "category": "family"},
    {"hindi": "पत्नी", "telugu": "పత్నీ", "english": "wife", "category": "family"},
    {"hindi": "बच्चा", "telugu": "బచ్చా", "english": "child", "category": "family"},
    {"hindi": "परिवार", "telugu": "పరివార్", "english": "family", "category": "family"},
    {
        "hindi": "रिश्ता",
        "telugu": "రిశ్తా",
        "english": "relationship",
        "category": "family",
    },
    # ==================== NUMBERS (20) ====================
    {"hindi": "एक", "telugu": "ఏక్", "english": "one", "category": "number"},
    {"hindi": "दो", "telugu": "దో", "english": "two", "category": "number"},
    {"hindi": "तीन", "telugu": "తీన్", "english": "three", "category": "number"},
    {"hindi": "चार", "telugu": "చార్", "english": "four", "category": "number"},
    {"hindi": "पांच", "telugu": "పాంచ్", "english": "five", "category": "number"},
    {"hindi": "छह", "telugu": "ఛహ్", "english": "six", "category": "number"},
    {"hindi": "सात", "telugu": "సాత్", "english": "seven", "category": "number"},
    {"hindi": "आठ", "telugu": "ఆఠ్", "english": "eight", "category": "number"},
    {"hindi": "नौ", "telugu": "నౌ", "english": "nine", "category": "number"},
    {"hindi": "दस", "telugu": "దస్", "english": "ten", "category": "number"},
    {"hindi": "ग्यारह", "telugu": "గ్యారహ్", "english": "eleven", "category": "number"},
    {"hindi": "बारह", "telugu": "బారహ్", "english": "twelve", "category": "number"},
    {"hindi": "बीस", "telugu": "బీస్", "english": "twenty", "category": "number"},
    {"hindi": "सौ", "telugu": "సౌ", "english": "hundred", "category": "number"},
    {"hindi": "हजार", "telugu": "హజార్", "english": "thousand", "category": "number"},
    {
        "hindi": "लाख",
        "telugu": "లాఖ్",
        "english": "hundred thousand",
        "category": "number",
    },
    {"hindi": "करोड़", "telugu": "కరోడ్", "english": "ten million", "category": "number"},
    {"hindi": "पहला", "telugu": "పహలా", "english": "first", "category": "number"},
    {"hindi": "दूसरा", "telugu": "దూసరా", "english": "second", "category": "number"},
    {"hindi": "आधा", "telugu": "ఆధా", "english": "half", "category": "number"},
    # ==================== COMMON NOUNS (40) ====================
    {"hindi": "पानी", "telugu": "పానీ", "english": "water", "category": "noun"},
    {"hindi": "खाना", "telugu": "ఖానా", "english": "food", "category": "noun"},
    {"hindi": "घर", "telugu": "ఘర్", "english": "house", "category": "noun"},
    {"hindi": "बाजार", "telugu": "బాజార్", "english": "market", "category": "noun"},
    {"hindi": "किताब", "telugu": "కితాబ్", "english": "book", "category": "noun"},
    {"hindi": "दोस्त", "telugu": "దోస్త్", "english": "friend", "category": "noun"},
    {"hindi": "समय", "telugu": "సమయ్", "english": "time", "category": "noun"},
    {"hindi": "काम", "telugu": "కామ్", "english": "work", "category": "noun"},
    {"hindi": "रोटी", "telugu": "రోటీ", "english": "bread", "category": "noun"},
    {"hindi": "दूध", "telugu": "దూధ్", "english": "milk", "category": "noun"},
    {"hindi": "चाय", "telugu": "చాయ్", "english": "tea", "category": "noun"},
    {"hindi": "फल", "telugu": "ఫల్", "english": "fruit", "category": "noun"},
    {"hindi": "सब्जी", "telugu": "సబ్జీ", "english": "vegetable", "category": "noun"},
    {"hindi": "चावल", "telugu": "చావల్", "english": "rice", "category": "noun"},
    {"hindi": "नमक", "telugu": "నమక్", "english": "salt", "category": "noun"},
    {"hindi": "चीनी", "telugu": "చీనీ", "english": "sugar", "category": "noun"},
    {"hindi": "तेल", "telugu": "తేల్", "english": "oil", "category": "noun"},
    {"hindi": "आग", "telugu": "ఆగ్", "english": "fire", "category": "noun"},
    {"hindi": "हवा", "telugu": "హవా", "english": "air", "category": "noun"},
    {"hindi": "धरती", "telugu": "ధరతీ", "english": "earth", "category": "noun"},
    {"hindi": "आकाश", "telugu": "ఆకాశ్", "english": "sky", "category": "noun"},
    {"hindi": "सूरज", "telugu": "సూరజ్", "english": "sun", "category": "noun"},
    {"hindi": "चांद", "telugu": "చాంద్", "english": "moon", "category": "noun"},
    {"hindi": "तारा", "telugu": "తారా", "english": "star", "category": "noun"},
    {"hindi": "बारिश", "telugu": "బారిశ్", "english": "rain", "category": "noun"},
    {"hindi": "नदी", "telugu": "నదీ", "english": "river", "category": "noun"},
    {"hindi": "पहाड़", "telugu": "పహాడ్", "english": "mountain", "category": "noun"},
    {"hindi": "जंगल", "telugu": "జంగల్", "english": "forest", "category": "noun"},
    {"hindi": "पेड़", "telugu": "పేడ్", "english": "tree", "category": "noun"},
    {"hindi": "फूल", "telugu": "ఫూల్", "english": "flower", "category": "noun"},
    {"hindi": "पत्ता", "telugu": "పత్తా", "english": "leaf", "category": "noun"},
    {"hindi": "रास्ता", "telugu": "రాస్తా", "english": "road", "category": "noun"},
    {"hindi": "दरवाजा", "telugu": "దర్వాజా", "english": "door", "category": "noun"},
    {"hindi": "खिड़की", "telugu": "ఖిడ్కీ", "english": "window", "category": "noun"},
    {"hindi": "कुर्सी", "telugu": "కుర్సీ", "english": "chair", "category": "noun"},
    {"hindi": "मेज", "telugu": "మేజ్", "english": "table", "category": "noun"},
    {"hindi": "बिस्तर", "telugu": "బిస్తర్", "english": "bed", "category": "noun"},
    {"hindi": "कपड़ा", "telugu": "కపడా", "english": "cloth", "category": "noun"},
    {"hindi": "जूता", "telugu": "జూతా", "english": "shoe", "category": "noun"},
    {"hindi": "टोपी", "telugu": "టోపీ", "english": "hat", "category": "noun"},
    # ==================== BODY PARTS (15) ====================
    {"hindi": "सिर", "telugu": "సిర్", "english": "head", "category": "body"},
    {"hindi": "आंख", "telugu": "ఆంఖ్", "english": "eye", "category": "body"},
    {"hindi": "कान", "telugu": "కాన్", "english": "ear", "category": "body"},
    {"hindi": "नाक", "telugu": "నాక్", "english": "nose", "category": "body"},
    {"hindi": "मुंह", "telugu": "ముంహ్", "english": "mouth", "category": "body"},
    {"hindi": "हाथ", "telugu": "హాథ్", "english": "hand", "category": "body"},
    {"hindi": "पैर", "telugu": "పైర్", "english": "foot", "category": "body"},
    {"hindi": "दिल", "telugu": "దిల్", "english": "heart", "category": "body"},
    {"hindi": "पेट", "telugu": "పేట్", "english": "stomach", "category": "body"},
    {"hindi": "गला", "telugu": "గలా", "english": "throat", "category": "body"},
    {"hindi": "बाल", "telugu": "బాల్", "english": "hair", "category": "body"},
    {"hindi": "चेहरा", "telugu": "చేహరా", "english": "face", "category": "body"},
    {"hindi": "गर्दन", "telugu": "గర్దన్", "english": "neck", "category": "body"},
    {"hindi": "कंधा", "telugu": "కంధా", "english": "shoulder", "category": "body"},
    {"hindi": "उंगली", "telugu": "ఉంగ్లీ", "english": "finger", "category": "body"},
    # ==================== ANIMALS (15) ====================
    {"hindi": "कुत्ता", "telugu": "కుత్తా", "english": "dog", "category": "animal"},
    {"hindi": "बिल्ली", "telugu": "బిల్లీ", "english": "cat", "category": "animal"},
    {"hindi": "गाय", "telugu": "గాయ్", "english": "cow", "category": "animal"},
    {"hindi": "घोड़ा", "telugu": "ఘోడా", "english": "horse", "category": "animal"},
    {"hindi": "हाथी", "telugu": "హాథీ", "english": "elephant", "category": "animal"},
    {"hindi": "शेर", "telugu": "శేర్", "english": "lion", "category": "animal"},
    {"hindi": "बंदर", "telugu": "బందర్", "english": "monkey", "category": "animal"},
    {"hindi": "मछली", "telugu": "మచ్ఛ్లీ", "english": "fish", "category": "animal"},
    {"hindi": "पक्षी", "telugu": "పక్షీ", "english": "bird", "category": "animal"},
    {"hindi": "मोर", "telugu": "మోర్", "english": "peacock", "category": "animal"},
    {"hindi": "सांप", "telugu": "సాంప్", "english": "snake", "category": "animal"},
    {"hindi": "मेंढक", "telugu": "మేంఢక్", "english": "frog", "category": "animal"},
    {"hindi": "चूहा", "telugu": "చూహా", "english": "mouse", "category": "animal"},
    {"hindi": "बकरी", "telugu": "బక్రీ", "english": "goat", "category": "animal"},
    {"hindi": "भैंस", "telugu": "భైంస్", "english": "buffalo", "category": "animal"},
    # ==================== PLACES (15) ====================
    {"hindi": "शहर", "telugu": "శహర్", "english": "city", "category": "place"},
    {"hindi": "गांव", "telugu": "గాంవ్", "english": "village", "category": "place"},
    {"hindi": "अस्पताल", "telugu": "అస్పతాల్", "english": "hospital", "category": "place"},
    {"hindi": "दुकान", "telugu": "దుకాన్", "english": "shop", "category": "place"},
    {"hindi": "स्कूल", "telugu": "స్కూల్", "english": "school", "category": "place"},
    {"hindi": "मंदिर", "telugu": "మందిర్", "english": "temple", "category": "place"},
    {"hindi": "मस्जिद", "telugu": "మస్జిద్", "english": "mosque", "category": "place"},
    {"hindi": "चर्च", "telugu": "చర్చ్", "english": "church", "category": "place"},
    {"hindi": "पुल", "telugu": "పుల్", "english": "bridge", "category": "place"},
    {"hindi": "स्टेशन", "telugu": "స్టేశన్", "english": "station", "category": "place"},
    {"hindi": "होटल", "telugu": "హోటల్", "english": "hotel", "category": "place"},
    {"hindi": "बैंक", "telugu": "బైంక్", "english": "bank", "category": "place"},
    {"hindi": "पार्क", "telugu": "పార్క్", "english": "park", "category": "place"},
    {"hindi": "समुद्र", "telugu": "సముద్ర్", "english": "ocean", "category": "place"},
    {"hindi": "देश", "telugu": "దేశ్", "english": "country", "category": "place"},
    # ==================== ADJECTIVES (25) ====================
    {"hindi": "अच्छा", "telugu": "అచ్ఛా", "english": "good", "category": "adjective"},
    {"hindi": "बुरा", "telugu": "బురా", "english": "bad", "category": "adjective"},
    {"hindi": "बड़ा", "telugu": "బడా", "english": "big", "category": "adjective"},
    {"hindi": "छोटा", "telugu": "ఛోటా", "english": "small", "category": "adjective"},
    {"hindi": "नया", "telugu": "నయా", "english": "new", "category": "adjective"},
    {"hindi": "पुराना", "telugu": "పురానా", "english": "old", "category": "adjective"},
    {"hindi": "लाल", "telugu": "లాల్", "english": "red", "category": "adjective"},
    {"hindi": "नीला", "telugu": "నీలా", "english": "blue", "category": "adjective"},
    {"hindi": "हरा", "telugu": "హరా", "english": "green", "category": "adjective"},
    {"hindi": "पीला", "telugu": "పీలా", "english": "yellow", "category": "adjective"},
    {"hindi": "काला", "telugu": "కాలా", "english": "black", "category": "adjective"},
    {"hindi": "सफेद", "telugu": "సఫేద్", "english": "white", "category": "adjective"},
    {
        "hindi": "सुंदर",
        "telugu": "సుందర్",
        "english": "beautiful",
        "category": "adjective",
    },
    {"hindi": "खुश", "telugu": "ఖుశ్", "english": "happy", "category": "adjective"},
    {"hindi": "उदास", "telugu": "ఉదాస్", "english": "sad", "category": "adjective"},
    {"hindi": "गर्म", "telugu": "గర్మ్", "english": "hot", "category": "adjective"},
    {"hindi": "ठंडा", "telugu": "ఠండా", "english": "cold", "category": "adjective"},
    {"hindi": "मीठा", "telugu": "మీఠా", "english": "sweet", "category": "adjective"},
    {"hindi": "खट्टा", "telugu": "ఖట్టా", "english": "sour", "category": "adjective"},
    {"hindi": "तेज", "telugu": "తేజ్", "english": "fast", "category": "adjective"},
    {"hindi": "धीमा", "telugu": "ధీమా", "english": "slow", "category": "adjective"},
    {"hindi": "भारी", "telugu": "భారీ", "english": "heavy", "category": "adjective"},
    {"hindi": "हल्का", "telugu": "హల్కా", "english": "light", "category": "adjective"},
    {"hindi": "साफ", "telugu": "సాఫ్", "english": "clean", "category": "adjective"},
    {"hindi": "गंदा", "telugu": "గందా", "english": "dirty", "category": "adjective"},
    # ==================== VERBS (25) ====================
    {"hindi": "जाना", "telugu": "జానా", "english": "to go", "category": "verb"},
    {"hindi": "आना", "telugu": "ఆనా", "english": "to come", "category": "verb"},
    {"hindi": "देखना", "telugu": "దేఖ్నా", "english": "to see", "category": "verb"},
    {"hindi": "सुनना", "telugu": "సున్నా", "english": "to listen", "category": "verb"},
    {"hindi": "बोलना", "telugu": "బోల్నా", "english": "to speak", "category": "verb"},
    {"hindi": "खाना", "telugu": "ఖానా", "english": "to eat", "category": "verb"},
    {"hindi": "पीना", "telugu": "పీనా", "english": "to drink", "category": "verb"},
    {"hindi": "सोना", "telugu": "సోనా", "english": "to sleep", "category": "verb"},
    {"hindi": "उठना", "telugu": "ఉఠ్నా", "english": "to wake up", "category": "verb"},
    {"hindi": "चलना", "telugu": "చల్నా", "english": "to walk", "category": "verb"},
    {"hindi": "दौड़ना", "telugu": "దౌడ్నా", "english": "to run", "category": "verb"},
    {"hindi": "पढ़ना", "telugu": "పఢ్నా", "english": "to read", "category": "verb"},
    {"hindi": "लिखना", "telugu": "లిఖ్నా", "english": "to write", "category": "verb"},
    {
        "hindi": "समझना",
        "telugu": "సమఝ్నా",
        "english": "to understand",
        "category": "verb",
    },
    {"hindi": "सोचना", "telugu": "సోచ్నా", "english": "to think", "category": "verb"},
    {"hindi": "करना", "telugu": "కర్నా", "english": "to do", "category": "verb"},
    {"hindi": "लेना", "telugu": "లేనా", "english": "to take", "category": "verb"},
    {"hindi": "देना", "telugu": "దేనా", "english": "to give", "category": "verb"},
    {"hindi": "रखना", "telugu": "రఖ్నా", "english": "to keep", "category": "verb"},
    {"hindi": "मिलना", "telugu": "మిల్నా", "english": "to meet", "category": "verb"},
    {"hindi": "बैठना", "telugu": "బైఠ్నా", "english": "to sit", "category": "verb"},
    {"hindi": "खड़ा होना", "telugu": "ఖడా హోనా", "english": "to stand", "category": "verb"},
    {"hindi": "रोना", "telugu": "రోనా", "english": "to cry", "category": "verb"},
    {"hindi": "हंसना", "telugu": "హంస్నా", "english": "to laugh", "category": "verb"},
    {
        "hindi": "प्यार करना",
        "telugu": "ప్యార్ కర్నా",
        "english": "to love",
        "category": "verb",
    },
    # ==================== TIME (15) ====================
    {"hindi": "आज", "telugu": "ఆజ్", "english": "today", "category": "time"},
    {"hindi": "कल", "telugu": "కల్", "english": "tomorrow", "category": "time"},
    {
        "hindi": "परसों",
        "telugu": "పర్సోం",
        "english": "day after tomorrow",
        "category": "time",
    },
    {"hindi": "अभी", "telugu": "అభీ", "english": "now", "category": "time"},
    {"hindi": "बाद", "telugu": "బాద్", "english": "later", "category": "time"},
    {"hindi": "पहले", "telugu": "పహలే", "english": "before", "category": "time"},
    {"hindi": "सुबह", "telugu": "సుబహ్", "english": "morning", "category": "time"},
    {"hindi": "दोपहर", "telugu": "దోపహర్", "english": "afternoon", "category": "time"},
    {"hindi": "शाम", "telugu": "శామ్", "english": "evening", "category": "time"},
    {"hindi": "रात", "telugu": "రాత్", "english": "night", "category": "time"},
    {"hindi": "घंटा", "telugu": "ఘంటా", "english": "hour", "category": "time"},
    {"hindi": "मिनट", "telugu": "మినట్", "english": "minute", "category": "time"},
    {"hindi": "दिन", "telugu": "దిన్", "english": "day", "category": "time"},
    {"hindi": "हफ्ता", "telugu": "హఫ్తా", "english": "week", "category": "time"},
    {"hindi": "महीना", "telugu": "మహీనా", "english": "month", "category": "time"},
    # ==================== QUESTIONS (10) ====================
    {"hindi": "क्या", "telugu": "క్యా", "english": "what", "category": "question"},
    {"hindi": "कौन", "telugu": "కౌన్", "english": "who", "category": "question"},
    {"hindi": "कहां", "telugu": "కహాం", "english": "where", "category": "question"},
    {"hindi": "कब", "telugu": "కబ్", "english": "when", "category": "question"},
    {"hindi": "कैसे", "telugu": "కైసే", "english": "how", "category": "question"},
    {"hindi": "क्यों", "telugu": "క్యోం", "english": "why", "category": "question"},
    {"hindi": "कितना", "telugu": "కిత్నా", "english": "how much", "category": "question"},
    {"hindi": "कौनसा", "telugu": "కౌన్సా", "english": "which", "category": "question"},
    {"hindi": "किसका", "telugu": "కిస్కా", "english": "whose", "category": "question"},
    {
        "hindi": "किधर",
        "telugu": "కిధర్",
        "english": "which direction",
        "category": "question",
    },
    # ==================== ABSTRACT (15) ====================
    {"hindi": "सच", "telugu": "సచ్", "english": "truth", "category": "abstract"},
    {"hindi": "झूठ", "telugu": "ఝూఠ్", "english": "lie", "category": "abstract"},
    {"hindi": "प्यार", "telugu": "ప్యార్", "english": "love", "category": "abstract"},
    {"hindi": "नफरत", "telugu": "నఫ్రత్", "english": "hate", "category": "abstract"},
    {"hindi": "डर", "telugu": "డర్", "english": "fear", "category": "abstract"},
    {"hindi": "खुशी", "telugu": "ఖుశీ", "english": "happiness", "category": "abstract"},
    {"hindi": "दुख", "telugu": "దుఖ్", "english": "sorrow", "category": "abstract"},
    {"hindi": "गुस्सा", "telugu": "గుస్సా", "english": "anger", "category": "abstract"},
    {"hindi": "आशा", "telugu": "ఆశా", "english": "hope", "category": "abstract"},
    {"hindi": "विश्वास", "telugu": "విశ్వాస్", "english": "faith", "category": "abstract"},
    {"hindi": "सपना", "telugu": "సప్నా", "english": "dream", "category": "abstract"},
    {"hindi": "याद", "telugu": "యాద్", "english": "memory", "category": "abstract"},
    {"hindi": "ज्ञान", "telugu": "జ్ఞాన్", "english": "knowledge", "category": "abstract"},
    {"hindi": "बुद्धि", "telugu": "బుద్ధి", "english": "wisdom", "category": "abstract"},
    {"hindi": "शक्ति", "telugu": "శక్తి", "english": "power", "category": "abstract"},
]


HINDI_TELUGU_WORDS_JSON_PATH = os.environ.get("HINDI_TELUGU_WORDS_JSON", "").strip()


def _load_words_from_json(path: str) -> List[Dict[str, str]]:
    p = Path(path).expanduser()
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list, got: {type(raw).__name__}")

    out: List[Dict[str, str]] = []
    seen_ids: set[str] = set()

    for i, rec in enumerate(raw):
        if not isinstance(rec, dict):
            continue

        hindi = rec.get("hindi") or rec.get("native") or rec.get("native word")
        telugu = rec.get("telugu") or rec.get("ood") or rec.get("target")

        if not isinstance(hindi, str) or not hindi.strip():
            continue
        if not isinstance(telugu, str) or not telugu.strip():
            continue

        english = (
            rec.get("english") or rec.get("id") or rec.get("english word") or hindi
        )
        if not isinstance(english, str) or not english.strip():
            english = hindi

        category = rec.get("category")
        if not isinstance(category, str) or not category.strip():
            category = "expanded"

        source = rec.get("source")
        if not isinstance(source, str) or not source.strip():
            source = "unknown"

        english_id = str(english).strip()
        if english_id in seen_ids:
            english_id = f"{english_id}__{i}"
        seen_ids.add(english_id)

        out.append(
            {
                "hindi": str(hindi).strip(),
                "telugu": str(telugu).strip(),
                "english": english_id,
                "category": str(category).strip(),
                "source": str(source).strip(),
            }
        )

    if not out:
        raise ValueError(f"No valid entries found in {path}")

    return out


if HINDI_TELUGU_WORDS_JSON_PATH:
    try:
        _loaded = _load_words_from_json(HINDI_TELUGU_WORDS_JSON_PATH)
        if len(_loaded) >= 150:
            HINDI_TELUGU_WORDS = _loaded
    except Exception:
        pass


def get_experiment_config() -> ExperimentConfig:
    """Get default experiment configuration."""
    return ExperimentConfig()


def get_model_config(model_size: str) -> ModelConfig:
    """Get model configuration by size."""
    if model_size not in MODELS:
        raise ValueError(
            f"Unknown model size: {model_size}. Available: {list(MODELS.keys())}"
        )
    return MODELS[model_size]


def get_words_by_category(category: str) -> List[Dict]:
    """Get words filtered by category."""
    return [w for w in HINDI_TELUGU_WORDS if w.get("category") == category]


def get_category_stats() -> Dict[str, int]:
    """Get count of words per category."""
    stats = {}
    for word in HINDI_TELUGU_WORDS:
        cat = word.get("category", "unknown")
        stats[cat] = stats.get(cat, 0) + 1
    return stats


# Print stats when run directly
if __name__ == "__main__":
    print(f"Total words: {len(HINDI_TELUGU_WORDS)}")
    print("\nBy category:")
    for cat, count in sorted(get_category_stats().items()):
        print(f"  {cat}: {count}")

    print("\nModels available:")
    for name, config in MODELS.items():
        print(f"  {name}: {config.hf_id} ({config.n_layers} layers)")
