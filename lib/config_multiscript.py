#!/usr/bin/env python3
"""
Multi-Script Cross-Language Configuration
==========================================
Date: January 2, 2026

CRITICAL FIX: We need multiple script pairs, not just Hindi-Telugu.

This file defines:
1. Multiple source languages (that model knows)
2. Multiple target scripts (OOD combinations)
3. Proper control conditions

A mech interp researcher would require this generalization.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import warnings


try:
    from aksharamukha import transliterate as _akshar_transliterate
except Exception:
    _akshar_transliterate = None

try:
    from indic_transliteration import sanscript as _sanscript
    from indic_transliteration.sanscript import transliterate as _indic_transliterate
except Exception:
    _sanscript = None
    _indic_transliterate = None

try:
    from wordfreq import top_n_list as _wordfreq_top_n_list
except Exception:
    _wordfreq_top_n_list = None


_SCRIPT_RANGES = {
    # Common scripts we use in SCRIPT_PAIRS. These are *heuristics* to guard
    # against "no-op" transliterations being silently accepted.
    "Arabic": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "Cyrillic": [(0x0400, 0x04FF), (0x0500, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)],
    "Devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "Bengali": [(0x0980, 0x09FF)],
    "Gurmukhi": [(0x0A00, 0x0A7F)],
    "Gujarati": [(0x0A80, 0x0AFF)],
    "Oriya": [(0x0B00, 0x0B7F)],
    "Georgian": [(0x10A0, 0x10FF), (0x2D00, 0x2D2F)],
    "Greek": [(0x0370, 0x03FF), (0x1F00, 0x1FFF)],
    "Tamil": [(0x0B80, 0x0BFF)],
    "Telugu": [(0x0C00, 0x0C7F)],
    "Kannada": [(0x0C80, 0x0CFF)],
    "Malayalam": [(0x0D00, 0x0D7F)],
    "Thai": [(0x0E00, 0x0E7F)],
    # Non-Indic scripts supported by aksharamukha (when installed).
    "Hebrew": [(0x0590, 0x05FF)],
    "Cyrillic (Russian)": [(0x0400, 0x04FF), (0x0500, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)],
    "Japanese (Hiragana)": [(0x3040, 0x309F)],
    "Japanese (Katakana)": [(0x30A0, 0x30FF), (0xFF66, 0xFF9D)],
    "Khmer (Cambodian)": [(0x1780, 0x17FF)],
    "Burmese (Myanmar)": [(0x1000, 0x109F)],
    "Lao": [(0x0E80, 0x0EFF)],
}


def _contains_target_script_chars(text: str, target_script: str) -> bool:
    ranges = _SCRIPT_RANGES.get(target_script)
    if not ranges:
        return True
    for ch in text:
        cp = ord(ch)
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return True
    return False


def _safe_transliterate(text: str, source_script: str, target_script: str) -> Optional[str]:
    """Best-effort transliteration helper (avoids import-time hard failures)."""
    def _aliases(s: str) -> List[str]:
        """
        Generate a small set of script-name aliases to make aksharamukha calls
        more robust across naming conventions.

        Examples:
          "Cyrillic (Russian)" → ["Cyrillic (Russian)", "Cyrillic", "Russian", "CyrillicRussian", "Cyrillic_Russian"]
          "Japanese (Katakana)" → ["Japanese (Katakana)", "Japanese", "Katakana", "JapaneseKatakana", "Japanese_Katakana"]
        """
        s = str(s or "").strip()
        if not s:
            return []
        out: List[str] = [s]

        base = s
        inside = ""
        if "(" in s and ")" in s and s.index("(") < s.index(")"):
            base = s.split("(", 1)[0].strip()
            inside = s.split("(", 1)[1].split(")", 1)[0].strip()
            if base:
                out.append(base)
            if inside:
                out.append(inside)

        # Common normalizations.
        for a in (base, s):
            a = str(a).strip()
            if not a:
                continue
            out.append(a.replace(" ", ""))
            out.append(a.replace(" ", "_"))
            out.append(a.replace("-", ""))
            out.append(a.replace("-", "_"))
        if base and inside:
            out.append(f"{base}{inside}".replace(" ", ""))
            out.append(f"{base}_{inside}".replace(" ", "_"))

        # De-dupe, preserve order.
        seen = set()
        uniq: List[str] = []
        for a in out:
            a = str(a).strip()
            if not a or a in seen:
                continue
            seen.add(a)
            uniq.append(a)
        return uniq

    def _accept(src: str, tgt: Optional[str]) -> Optional[str]:
        if tgt is None:
            return None
        tgt2 = str(tgt).strip()
        if not tgt2:
            return None
        # Many transliteration libs return the input unchanged when a script name
        # is unsupported; treat that as failure so we can fall back.
        if tgt2.casefold() == str(src).strip().casefold():
            return None
        # For a set of common scripts, require that the output actually contains
        # characters from that script's Unicode blocks. This avoids accepting
        # outputs like Latin-with-diacritics when the library silently fails.
        if target_script not in {"Latin", "ISO"}:
            if not _contains_target_script_chars(tgt2, target_script):
                # If we know the target script, reject outright.
                if target_script in _SCRIPT_RANGES:
                    return None
                # Otherwise, fall back to a weaker non-ASCII heuristic.
                if not any(ord(ch) > 127 for ch in tgt2):
                    return None
        return tgt2

    if _akshar_transliterate is not None:
        try:
            # aksharamukha emits UserWarnings for unsupported scripts and may
            # return the input unchanged. We treat those as "no-op" and fall
            # back, so suppress the warning to keep logs clean.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Target script: .* not found in the list of scripts supported.*",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message=r"Source script: .* not found in the list of scripts supported.*",
                    category=UserWarning,
                )
                for src in _aliases(source_script):
                    for tgt in _aliases(target_script):
                        out = _akshar_transliterate.process(src, tgt, text)
                        ok = _accept(text, out)
                        if ok is not None:
                            return ok
        except Exception:
            pass

    if _indic_transliterate is not None and _sanscript is not None:
        try:
            script_map = {
                "Devanagari": _sanscript.DEVANAGARI,
                "Telugu": _sanscript.TELUGU,
                "Tamil": _sanscript.TAMIL,
                "Kannada": _sanscript.KANNADA,
                "Malayalam": _sanscript.MALAYALAM,
                "Bengali": _sanscript.BENGALI,
                "Gujarati": _sanscript.GUJARATI,
                "Oriya": _sanscript.ORIYA,
                "Gurmukhi": _sanscript.GURMUKHI,
            }
            src = script_map.get(source_script)
            tgt = script_map.get(target_script)
            if src and tgt:
                out = _indic_transliterate(text, src, tgt)
                ok = _accept(text, out)
                if ok is not None:
                    return ok
        except Exception:
            pass

    return None


@dataclass
class ScriptPair:
    """Defines a language-script pair for testing."""
    name: str                    # e.g., "hindi_telugu"
    source_language: str         # Language the model should know
    source_script: str           # Native script for that language
    target_script: str           # OOD script we're testing
    words: List[Dict]            # word entries with source_script and target_script versions
    is_control: bool = False     # True if this is a control condition
    expected_rescue: bool = True # Do we expect rescue to work?


def _external_only_pair(
    name: str,
    *,
    source_language: str,
    target_script: str,
) -> ScriptPair:
    """
    Declare an external-first pair with no builtin fallback rows.

    This is used for the publication-oriented workshop lane where benchmark
    semantics matter more than having a local smoke-test pool. If no external
    JSONL is present under `data/transliteration/`, ingestion should fail
    instead of silently substituting synthetic data.
    """
    return ScriptPair(
        name=name,
        source_language=source_language,
        source_script="Latin",
        target_script=target_script,
        words=[],
        is_control=False,
        expected_rescue=True,
    )


# ============================================================================
# HINDI TRANSLITERATIONS INTO MULTIPLE SCRIPTS
# ============================================================================

# Hindi words with transliterations into various scripts
# Source: Hindi (Devanagari) - a language Gemma likely knows well
# Target: Various scripts Gemma can tokenize but doesn't associate with Hindi

HINDI_WORDS_BASE = [
    # Each entry: english, hindi (devanagari)
    ("water", "पानी"),
    ("food", "खाना"),
    ("house", "घर"),
    ("book", "किताब"),
    ("mother", "माता"),
    ("father", "पिता"),
    ("brother", "भाई"),
    ("sister", "बहन"),
    ("sun", "सूरज"),
    ("moon", "चांद"),
    ("star", "तारा"),
    ("sky", "आकाश"),
    ("earth", "धरती"),
    ("fire", "आग"),
    ("air", "हवा"),
    ("tree", "पेड़"),
    ("flower", "फूल"),
    ("river", "नदी"),
    ("mountain", "पहाड़"),
    ("rain", "बारिश"),
    ("dog", "कुत्ता"),
    ("cat", "बिल्ली"),
    ("cow", "गाय"),
    ("bird", "पक्षी"),
    ("fish", "मछली"),
    ("good", "अच्छा"),
    ("bad", "बुरा"),
    ("big", "बड़ा"),
    ("small", "छोटा"),
    ("new", "नया"),
    ("old", "पुराना"),
    ("hot", "गर्म"),
    ("cold", "ठंडा"),
    ("one", "एक"),
    ("two", "दो"),
    ("three", "तीन"),
    ("four", "चार"),
    ("five", "पांच"),
    ("today", "आज"),
    ("tomorrow", "कल"),
    ("hello", "नमस्ते"),
    ("thank you", "धन्यवाद"),
    ("please", "कृपया"),
    ("yes", "हां"),
    ("no", "नहीं"),
    ("what", "क्या"),
    ("who", "कौन"),
    ("where", "कहां"),
    ("when", "कब"),
    ("why", "क्यों"),
]

# Hindi written in Telugu script (phonetic transliteration)
HINDI_IN_TELUGU = {
    "water": "పానీ",
    "food": "ఖానా",
    "house": "ఘర్",
    "book": "కితాబ్",
    "mother": "మాతా",
    "father": "పితా",
    "brother": "భాఈ",
    "sister": "బహన్",
    "sun": "సూరజ్",
    "moon": "చాంద్",
    "star": "తారా",
    "sky": "ఆకాశ్",
    "earth": "ధరతీ",
    "fire": "ఆగ్",
    "air": "హవా",
    "tree": "పేడ్",
    "flower": "ఫూల్",
    "river": "నదీ",
    "mountain": "పహాడ్",
    "rain": "బారిశ్",
    "dog": "కుత్తా",
    "cat": "బిల్లీ",
    "cow": "గాయ్",
    "bird": "పక్షీ",
    "fish": "మచ్ఛ్లీ",
    "good": "అచ్ఛా",
    "bad": "బురా",
    "big": "బడా",
    "small": "ఛోటా",
    "new": "నయా",
    "old": "పురానా",
    "hot": "గర్మ్",
    "cold": "ఠండా",
    "one": "ఏక్",
    "two": "దో",
    "three": "తీన్",
    "four": "చార్",
    "five": "పాంచ్",
    "today": "ఆజ్",
    "tomorrow": "కల్",
    "hello": "నమస్తే",
    "thank you": "ధన్యవాద్",
    "please": "కృపయా",
    "yes": "హాం",
    "no": "నహీం",
    "what": "క్యా",
    "who": "కౌన్",
    "where": "కహాం",
    "when": "కబ్",
    "why": "క్యోం",
}

# Hindi written in Tamil script (phonetic transliteration)
HINDI_IN_TAMIL = {
    "water": "பானீ",
    "food": "கானா",
    "house": "கர்",
    "book": "கிதாப்",
    "mother": "மாதா",
    "father": "பிதா",
    "brother": "பாய்",
    "sister": "பஹன்",
    "sun": "சூரஜ்",
    "moon": "சாந்த்",
    "star": "தாரா",
    "sky": "ஆகாஷ்",
    "earth": "தரதீ",
    "fire": "ஆக்",
    "air": "ஹவா",
    "tree": "பேட்",
    "flower": "பூல்",
    "river": "நதீ",
    "mountain": "பஹாட்",
    "rain": "பாரிஷ்",
    "dog": "குத்தா",
    "cat": "பில்லீ",
    "cow": "காய்",
    "bird": "பக்ஷீ",
    "fish": "மச்ச்லீ",
    "good": "அச்சா",
    "bad": "புரா",
    "big": "பட்டா",
    "small": "சோடா",
    "new": "நயா",
    "old": "புராணா",
    "hot": "கர்ம்",
    "cold": "தண்டா",
    "one": "ஏக்",
    "two": "தோ",
    "three": "தீன்",
    "four": "சார்",
    "five": "பாஞ்ச்",
    "today": "ஆஜ்",
    "tomorrow": "கல்",
    "hello": "நமஸ்தே",
    "thank you": "தன்யவாத்",
    "please": "க்ரிபயா",
    "yes": "ஹாம்",
    "no": "நஹீம்",
    "what": "க்யா",
    "who": "கௌன்",
    "where": "கஹாம்",
    "when": "கப்",
    "why": "க்யோம்",
}

# Hindi written in Arabic script (Urdu-style, right-to-left)
HINDI_IN_ARABIC = {
    "water": "پانی",
    "food": "کھانا",
    "house": "گھر",
    "book": "کتاب",
    "mother": "ماتا",
    "father": "پتا",
    "brother": "بھائی",
    "sister": "بہن",
    "sun": "سورج",
    "moon": "چاند",
    "star": "تارا",
    "sky": "آکاش",
    "earth": "دھرتی",
    "fire": "آگ",
    "air": "ہوا",
    "tree": "پیڑ",
    "flower": "پھول",
    "river": "ندی",
    "mountain": "پہاڑ",
    "rain": "بارش",
    "dog": "کتا",
    "cat": "بلی",
    "cow": "گائے",
    "bird": "پکشی",
    "fish": "مچھلی",
    "good": "اچھا",
    "bad": "برا",
    "big": "بڑا",
    "small": "چھوٹا",
    "new": "نیا",
    "old": "پرانا",
    "hot": "گرم",
    "cold": "ٹھنڈا",
    "one": "ایک",
    "two": "دو",
    "three": "تین",
    "four": "چار",
    "five": "پانچ",
    "today": "آج",
    "tomorrow": "کل",
    "hello": "نمستے",
    "thank you": "دھنیہ واد",
    "please": "کرپیا",
    "yes": "ہاں",
    "no": "نہیں",
    "what": "کیا",
    "who": "کون",
    "where": "کہاں",
    "when": "کب",
    "why": "کیوں",
}

# Hindi written in Cyrillic script (phonetic)
HINDI_IN_CYRILLIC = {
    "water": "пани",
    "food": "кхана",
    "house": "гхар",
    "book": "китаб",
    "mother": "мата",
    "father": "пита",
    "brother": "бхаи",
    "sister": "бахан",
    "sun": "сурадж",
    "moon": "чанд",
    "star": "тара",
    "sky": "акаш",
    "earth": "дхарти",
    "fire": "аг",
    "air": "хава",
    "tree": "пед",
    "flower": "пхул",
    "river": "нади",
    "mountain": "пахад",
    "rain": "бариш",
    "dog": "кутта",
    "cat": "билли",
    "cow": "гай",
    "bird": "пакши",
    "fish": "маччхли",
    "good": "аччха",
    "bad": "бура",
    "big": "бада",
    "small": "чхота",
    "new": "ная",
    "old": "пурана",
    "hot": "гарм",
    "cold": "тханда",
    "one": "эк",
    "two": "до",
    "three": "тин",
    "four": "чар",
    "five": "панч",
    "today": "адж",
    "tomorrow": "кал",
    "hello": "намасте",
    "thank you": "дханьявад",
    "please": "крипая",
    "yes": "хан",
    "no": "нахин",
    "what": "кья",
    "who": "каун",
    "where": "кахан",
    "when": "каб",
    "why": "кьон",
}

# Hindi written in Greek script (phonetic)
HINDI_IN_GREEK = {
    "water": "πανι",
    "food": "κανα",
    "house": "γκαρ",
    "book": "κιταμπ",
    "mother": "ματα",
    "father": "πιτα",
    "brother": "μπαι",
    "sister": "μπαχαν",
    "sun": "σουραζ",
    "moon": "τσαντ",
    "star": "ταρα",
    "sky": "ακας",
    "earth": "νταρτι",
    "fire": "αγκ",
    "air": "χαβα",
    "tree": "πεντ",
    "flower": "φουλ",
    "river": "ναντι",
    "mountain": "παχαντ",
    "rain": "μπαρις",
    "dog": "κουττα",
    "cat": "μπιλλι",
    "cow": "γκαι",
    "bird": "πακσι",
    "fish": "ματσλι",
    "good": "ατστσα",
    "bad": "μπουρα",
    "big": "μπαντα",
    "small": "τσοτα",
    "new": "ναγια",
    "old": "πουρανα",
    "hot": "γκαρμ",
    "cold": "τσαντα",
    "one": "εκ",
    "two": "ντο",
    "three": "τιν",
    "four": "τσαρ",
    "five": "παντς",
    "today": "αζ",
    "tomorrow": "καλ",
    "hello": "ναμαστε",
    "thank you": "ντανιαβαντ",
    "please": "κριπαγια",
    "yes": "χαν",
    "no": "ναχιν",
    "what": "κια",
    "who": "καουν",
    "where": "καχαν",
    "when": "καμπ",
    "why": "κιον",
}

# ============================================================================
# ENGLISH TRANSLITERATIONS INTO NON-LATIN SCRIPTS (SECOND SOURCE LANGUAGE)
# ============================================================================

DEFAULT_ENGLISH_WORDS_BASE = [
    "hello",
    "thank you",
    "please",
    "water",
    "food",
    "house",
    "book",
    "friend",
    "love",
    "world",
    "time",
    "life",
    "work",
    "good",
    "great",
    "bad",
    "morning",
    "evening",
    "night",
    "day",
    "sun",
    "moon",
    "star",
    "sky",
    "rain",
    "tree",
    "river",
    "city",
    "car",
    "bus",
    "train",
    "doctor",
    "teacher",
    "student",
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "big",
    "small",
    "hot",
    "cold",
    "happy",
    "sad",
    "yes",
    "no",
    "one",
    "two",
]


def _build_wordfreq_words(
    lang_code: str,
    *,
    n: int = 220,
    min_len: int = 3,
    max_len: int = 12,
    ascii_only: bool = True,
) -> List[str]:
    """
    Build a word list from the `wordfreq` package (optional dependency).

    This gives us larger, more diverse word pools for additional source
    languages (e.g., Spanish/French) without hardcoding big lists.
    """
    if _wordfreq_top_n_list is None:
        return []
    try:
        raw = list(_wordfreq_top_n_list(lang_code, int(n * 8)))
    except Exception:
        return []

    out: List[str] = []
    seen = set()
    for w in raw:
        if not isinstance(w, str):
            continue
        w = w.strip()
        if not w or w in seen:
            continue
        # Keep simple "word" items only (no spaces/hyphens).
        if any(ch.isspace() for ch in w) or "-" in w:
            continue
        if len(w) < int(min_len) or len(w) > int(max_len):
            continue
        # Keep alphabetic tokens (filters digits/punctuation).
        if not w.isalpha():
            continue
        if ascii_only and not all(ord(ch) < 128 for ch in w):
            continue
        lw = w.lower()
        if lw in seen:
            continue
        seen.add(lw)
        out.append(lw)
        if len(out) >= int(n):
            break
    return out


def _build_english_words_base() -> List[str]:
    """
    Build an English word list for "English-in-OOD-script → English-in-Latin" tasks.

    To keep *English* experiments as statistically comparable to the 220-word
    Hindi-Telugu default task, we opportunistically reuse the English glosses
    from config.py (when available). This keeps the word pool large without
    hand-maintaining another list.

    If config.py isn't importable (e.g., minimal environment), fall back to a
    small built-in list so experiments still run.
    """
    try:
        from config import HINDI_TELUGU_WORDS

        seen = set()
        out: List[str] = []
        for w in HINDI_TELUGU_WORDS:
            e = w.get("english")
            if not isinstance(e, str):
                continue
            e = e.strip()
            if not e or e in seen:
                continue
            seen.add(e)
            out.append(e)
        # Match the Hindi default pool size (~220) when possible.
        return out[:220] if out else DEFAULT_ENGLISH_WORDS_BASE
    except Exception:
        return DEFAULT_ENGLISH_WORDS_BASE


ENGLISH_WORDS_BASE = _build_english_words_base()

# Hand-crafted phonetic approximations for English written in Georgian script.
# We prefer library-generated transliterations (when available) for scale; this
# fallback keeps experiments runnable offline.
ENGLISH_GEORGIAN_FALLBACK = {
    "hello": "ჰელო",
    "thank you": "თენქ იუ",
    "please": "ფლიზ",
    "water": "ვოთერ",
    "food": "ფუდ",
    "house": "ჰაუს",
    "book": "ბუქ",
    "friend": "ფრენდ",
    "love": "ლავ",
    "world": "ვორლდ",
    "time": "თაიმ",
    "life": "ლაიფ",
    "work": "ვორქ",
    "good": "გუდ",
    "great": "გრეით",
    "bad": "ბედ",
    "morning": "მორნინგ",
    "evening": "ივნინგ",
    "night": "ნაით",
    "day": "დეი",
    "sun": "სან",
    "moon": "მუნ",
    "star": "სთარ",
    "sky": "სქაი",
    "rain": "რეინ",
    "tree": "თრი",
    "river": "რივერ",
    "city": "სითი",
    "car": "ქარ",
    "bus": "ბას",
    "doctor": "დოქთორ",
    "teacher": "თიჩერ",
    "red": "რედ",
    "blue": "ბლუ",
    "green": "გრინ",
    "happy": "ჰეფი",
    "sad": "სედ",
    "yes": "იეს",
    "no": "ნო",
    "one": "ვან",
}

# Thai/Arabic/Indic scripts can be generated programmatically via aksharamukha (if installed).
# If not available, we fall back to a small hand-crafted list so experiments can still run.

def _build_latin_script_map(words: List[str], target_script: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    attempts = 0
    successes = 0
    for w in words:
        attempts += 1
        # Latin-script source languages are inconsistently named across
        # transliteration libraries. We try a short list of candidate "source
        # script" names and accept the first non-noop output.
        tr: Optional[str] = None
        for src in ("Latin", "ISO"):
            tr = _safe_transliterate(w, src, target_script)
            if tr:
                break
        if tr:
            out[w] = tr
            successes += 1
        # If the first chunk yields zero valid outputs, assume the target script
        # isn't supported (or is being returned as a no-op) and bail out early
        # instead of looping across ~220 words.
        if attempts >= 20 and successes == 0:
            break
    return out


ENGLISH_DEVANAGARI_FALLBACK = {
    "hello": "हेलो",
    "thank you": "थैंक यू",
    "please": "प्लीज़",
    "water": "वॉटर",
    "food": "फूड",
    "house": "हाउस",
    "book": "बुक",
    "friend": "फ्रेंड",
    "love": "लव",
    "world": "वर्ल्ड",
}

ENGLISH_TELUGU_FALLBACK = {
    "hello": "హెలో",
    "thank you": "థ్యాంక్ యూ",
    "please": "ప్లీజ్",
    "water": "వాటర్",
    "food": "ఫుడ్",
    "house": "హౌస్",
    "book": "బుక్",
    "friend": "ఫ్రెండ్",
    "love": "లవ్",
    "world": "వరల్డ్",
}

ENGLISH_IN_GEORGIAN = _build_latin_script_map(ENGLISH_WORDS_BASE, "Georgian") or ENGLISH_GEORGIAN_FALLBACK
ENGLISH_IN_DEVANAGARI = _build_latin_script_map(ENGLISH_WORDS_BASE, "Devanagari") or ENGLISH_DEVANAGARI_FALLBACK
ENGLISH_IN_TELUGU = _build_latin_script_map(ENGLISH_WORDS_BASE, "Telugu") or ENGLISH_TELUGU_FALLBACK

ENGLISH_THAI_FALLBACK = {
    "hello": "เฮลโล",
    "thank you": "แธงค์ยู",
    "water": "วอเตอร์",
    "food": "ฟู้ด",
    "house": "เฮาส์",
    "book": "บุ๊ค",
    "friend": "เฟรนด์",
    "love": "เลิฟ",
    "world": "เวิลด์",
    "computer": "คอมพิวเตอร์",
}

ENGLISH_ARABIC_FALLBACK = {
    "hello": "هيلو",
    "thank you": "ثانك يو",
    "water": "ووتر",
    "food": "فود",
    "house": "هاوس",
    "book": "بوك",
    "friend": "فريند",
    "love": "لوف",
    "computer": "كومبيوتر",
    "internet": "انترنت",
}

ENGLISH_IN_THAI = _build_latin_script_map(ENGLISH_WORDS_BASE, "Thai") or ENGLISH_THAI_FALLBACK
ENGLISH_IN_ARABIC = _build_latin_script_map(ENGLISH_WORDS_BASE, "Arabic") or ENGLISH_ARABIC_FALLBACK

# Additional non-Indic script variants (optional; use check_multiscript_pairs.py --min-n to validate).
ENGLISH_CYRILLIC_FALLBACK = {
    "hello": "хелло",
    "thank you": "сэнк ю",
    "please": "плиз",
    "water": "вотер",
    "food": "фуд",
    "house": "хаус",
    "book": "бук",
    "friend": "френд",
    "love": "лав",
    "world": "ворлд",
}
ENGLISH_HEBREW_FALLBACK = {
    "hello": "הלו",
    "thank you": "ת'אנק יו",
    "please": "פליז",
    "water": "ווטר",
    "food": "פוד",
    "house": "האוס",
    "book": "בוק",
    "friend": "פרנד",
    "love": "לאב",
    "world": "וורלד",
}
ENGLISH_KATAKANA_FALLBACK = {
    "hello": "ハロー",
    "thank you": "サンキュー",
    "please": "プリーズ",
    "water": "ウォーター",
    "food": "フード",
    "house": "ハウス",
    "book": "ブック",
    "friend": "フレンド",
    "love": "ラブ",
    "world": "ワールド",
}
ENGLISH_IN_CYRILLIC = _build_latin_script_map(ENGLISH_WORDS_BASE, "Cyrillic (Russian)") or ENGLISH_CYRILLIC_FALLBACK
ENGLISH_IN_HEBREW = _build_latin_script_map(ENGLISH_WORDS_BASE, "Hebrew") or ENGLISH_HEBREW_FALLBACK
ENGLISH_IN_KATAKANA = _build_latin_script_map(ENGLISH_WORDS_BASE, "Japanese (Katakana)") or ENGLISH_KATAKANA_FALLBACK

# ============================================================================
# ADDITIONAL SOURCE LANGUAGES (Latin script) FOR DIVERSITY
# ============================================================================

DEFAULT_SPANISH_WORDS_BASE = [
    "hola",
    "gracias",
    "porfavor",
    "agua",
    "comida",
    "casa",
    "libro",
    "amigo",
    "amor",
    "mundo",
    "tiempo",
    "vida",
    "trabajo",
    "bueno",
    "malo",
    "dia",
    "noche",
    "sol",
    "luna",
    "lluvia",
]

SPANISH_WORDS_BASE = _build_wordfreq_words("es") or DEFAULT_SPANISH_WORDS_BASE

SPANISH_TELUGU_FALLBACK = {
    "hola": "హొలా",
    "gracias": "గ్రాసియస్",
    "agua": "ఆగ్వా",
    "casa": "కాసా",
    "libro": "లిబ్రో",
}

SPANISH_DEVANAGARI_FALLBACK = {
    "hola": "होला",
    "gracias": "ग्रासियस",
    "agua": "आग्वा",
    "casa": "कासा",
    "libro": "लिब्रो",
}

SPANISH_THAI_FALLBACK = {
    "hola": "โฮลา",
    "gracias": "กราเซียส",
    "agua": "อากวา",
    "casa": "กาซา",
    "libro": "ลิบโร",
}

SPANISH_IN_TELUGU = _build_latin_script_map(SPANISH_WORDS_BASE, "Telugu") or SPANISH_TELUGU_FALLBACK
SPANISH_IN_DEVANAGARI = _build_latin_script_map(SPANISH_WORDS_BASE, "Devanagari") or SPANISH_DEVANAGARI_FALLBACK
SPANISH_IN_THAI = _build_latin_script_map(SPANISH_WORDS_BASE, "Thai") or SPANISH_THAI_FALLBACK
SPANISH_IN_CYRILLIC = _build_latin_script_map(SPANISH_WORDS_BASE, "Cyrillic (Russian)")
SPANISH_IN_HEBREW = _build_latin_script_map(SPANISH_WORDS_BASE, "Hebrew")
SPANISH_IN_KATAKANA = _build_latin_script_map(SPANISH_WORDS_BASE, "Japanese (Katakana)")
SPANISH_IN_ARABIC = _build_latin_script_map(SPANISH_WORDS_BASE, "Arabic")

DEFAULT_FRENCH_WORDS_BASE = [
    "bonjour",
    "merci",
    "svp",
    "eau",
    "maison",
    "livre",
    "ami",
    "amour",
    "monde",
    "temps",
    "vie",
    "travail",
    "bon",
    "mauvais",
    "jour",
    "nuit",
    "soleil",
    "lune",
    "pluie",
]

FRENCH_WORDS_BASE = _build_wordfreq_words("fr") or DEFAULT_FRENCH_WORDS_BASE
FRENCH_TELUGU_FALLBACK = {
    "bonjour": "బోంజూర్",
    "merci": "మెర్సీ",
    "svp": "ఎస్‌వీపీ",
    "eau": "ఓ",
    "maison": "మెజోన్",
}
FRENCH_DEVANAGARI_FALLBACK = {
    "bonjour": "बोंजूर",
    "merci": "मर्सी",
    "svp": "एसवीपी",
    "eau": "ओ",
    "maison": "मेज़ों",
}
FRENCH_THAI_FALLBACK = {
    "bonjour": "บงชูร์",
    "merci": "แมร์ซี",
    "svp": "เอสวีพี",
    "eau": "โอ",
    "maison": "เมซง",
}
FRENCH_IN_TELUGU = _build_latin_script_map(FRENCH_WORDS_BASE, "Telugu") or FRENCH_TELUGU_FALLBACK
FRENCH_IN_DEVANAGARI = _build_latin_script_map(FRENCH_WORDS_BASE, "Devanagari") or FRENCH_DEVANAGARI_FALLBACK
FRENCH_IN_THAI = _build_latin_script_map(FRENCH_WORDS_BASE, "Thai") or FRENCH_THAI_FALLBACK
FRENCH_IN_CYRILLIC = _build_latin_script_map(FRENCH_WORDS_BASE, "Cyrillic (Russian)")
FRENCH_IN_HEBREW = _build_latin_script_map(FRENCH_WORDS_BASE, "Hebrew")
FRENCH_IN_KATAKANA = _build_latin_script_map(FRENCH_WORDS_BASE, "Japanese (Katakana)")
FRENCH_IN_ARABIC = _build_latin_script_map(FRENCH_WORDS_BASE, "Arabic")


def build_word_list(english_hindi: List[tuple], transliteration_map: Dict[str, str]) -> List[Dict]:
    """Build word list from base words and transliteration map."""
    words = []
    for english, hindi in english_hindi:
        if english in transliteration_map:
            words.append({
                "english": english,
                "source": hindi,
                "target": transliteration_map[english],
            })
    return words


def _build_hindi_words_via_transliteration(
    *,
    target_script: str,
    fallback_transliteration_map: Dict[str, str],
) -> List[Dict]:
    """
    Build a Hindi word list for a given target script.

    Preferred: expand to the full 220-word Hindi pool (from config.py) by
    transliterating the Devanagari targets using `_safe_transliterate`.

    Fallback: use the smaller hand-maintained 50-word pool for robustness when
    transliteration libraries are unavailable.
    """
    # Try to expand to the 220-word pool used by the main Hindi default task.
    try:
        from config import HINDI_TELUGU_WORDS as _HINDI_220

        expanded: List[Dict] = []
        for w in _HINDI_220:
            hindi = w.get("hindi", "")
            english = w.get("english", "")
            if not isinstance(hindi, str) or not hindi.strip():
                continue
            if not isinstance(english, str) or not english.strip():
                continue
            ood = _safe_transliterate(hindi, "Devanagari", target_script)
            if not ood:
                continue
            expanded.append({"english": english, "source": hindi, "target": ood})

        # Only use expanded pool if it is reasonably complete; otherwise keep
        # experiments deterministic via the fallback list.
        if len(expanded) >= 150:
            return expanded
    except Exception:
        pass

    return build_word_list(HINDI_WORDS_BASE, fallback_transliteration_map)


def _build_hindi_telugu_words() -> List[Dict]:
    """Prefer the curated 220-word Hindi-in-Telugu list from config.py when available."""
    try:
        from config import HINDI_TELUGU_WORDS as _HINDI_220

        out: List[Dict] = []
        for w in _HINDI_220:
            hindi = w.get("hindi", "")
            telugu = w.get("telugu", "")
            english = w.get("english", "")
            if not isinstance(hindi, str) or not hindi.strip():
                continue
            if not isinstance(telugu, str) or not telugu.strip():
                continue
            if not isinstance(english, str) or not english.strip():
                continue
            out.append({"english": english, "source": hindi, "target": telugu})

        if len(out) >= 150:
            return out
    except Exception:
        pass

    return build_word_list(HINDI_WORDS_BASE, HINDI_IN_TELUGU)


# ============================================================================
# SCRIPT PAIR DEFINITIONS
# ============================================================================

SCRIPT_PAIRS = {
    # PRIMARY: Hindi in Telugu (our main experiment)
    "hindi_telugu": ScriptPair(
        name="hindi_telugu",
        source_language="Hindi",
        source_script="Devanagari",
        target_script="Telugu",
        words=_build_hindi_telugu_words(),
        is_control=False,
        expected_rescue=True,
    ),

    # GENERALIZATION: Hindi in Tamil
    "hindi_tamil": ScriptPair(
        name="hindi_tamil",
        source_language="Hindi",
        source_script="Devanagari",
        target_script="Tamil",
        words=_build_hindi_words_via_transliteration(
            target_script="Tamil",
            fallback_transliteration_map=HINDI_IN_TAMIL,
        ),
        is_control=False,
        expected_rescue=True,
    ),

    # GENERALIZATION: Hindi in Kannada (Indic script; tests script transfer)
    "hindi_kannada": ScriptPair(
        name="hindi_kannada",
        source_language="Hindi",
        source_script="Devanagari",
        target_script="Kannada",
        words=_build_hindi_words_via_transliteration(
            target_script="Kannada",
            fallback_transliteration_map={},
        ),
        is_control=False,
        expected_rescue=True,
    ),

    # GENERALIZATION: Hindi in Malayalam
    "hindi_malayalam": ScriptPair(
        name="hindi_malayalam",
        source_language="Hindi",
        source_script="Devanagari",
        target_script="Malayalam",
        words=_build_hindi_words_via_transliteration(
            target_script="Malayalam",
            fallback_transliteration_map={},
        ),
        is_control=False,
        expected_rescue=True,
    ),

    # GENERALIZATION: Hindi in Bengali
    "hindi_bengali": ScriptPair(
        name="hindi_bengali",
        source_language="Hindi",
        source_script="Devanagari",
        target_script="Bengali",
        words=_build_hindi_words_via_transliteration(
            target_script="Bengali",
            fallback_transliteration_map={},
        ),
        is_control=False,
        expected_rescue=True,
    ),

    # GENERALIZATION: Hindi in Gujarati
    "hindi_gujarati": ScriptPair(
        name="hindi_gujarati",
        source_language="Hindi",
        source_script="Devanagari",
        target_script="Gujarati",
        words=_build_hindi_words_via_transliteration(
            target_script="Gujarati",
            fallback_transliteration_map={},
        ),
        is_control=False,
        expected_rescue=True,
    ),

    # GENERALIZATION: Hindi in Gurmukhi
    "hindi_gurmukhi": ScriptPair(
        name="hindi_gurmukhi",
        source_language="Hindi",
        source_script="Devanagari",
        target_script="Gurmukhi",
        words=_build_hindi_words_via_transliteration(
            target_script="Gurmukhi",
            fallback_transliteration_map={},
        ),
        is_control=False,
        expected_rescue=True,
    ),

    # GENERALIZATION: Hindi in Thai (non-Indic script; tests true script shift)
    "hindi_thai": ScriptPair(
        name="hindi_thai",
        source_language="Hindi",
        source_script="Devanagari",
        target_script="Thai",
        words=_build_hindi_words_via_transliteration(
            target_script="Thai",
            fallback_transliteration_map={},
        ),
        is_control=False,
        expected_rescue=True,
    ),

    # GENERALIZATION: Hindi in Arabic (very different script)
    "hindi_arabic": ScriptPair(
        name="hindi_arabic",
        source_language="Hindi",
        source_script="Devanagari",
        target_script="Arabic",
        words=_build_hindi_words_via_transliteration(
            target_script="Arabic",
            fallback_transliteration_map=HINDI_IN_ARABIC,
        ),
        is_control=False,
        expected_rescue=True,  # Interesting test - different script family
    ),

    # DISTANT: Hindi in Cyrillic
    "hindi_cyrillic": ScriptPair(
        name="hindi_cyrillic",
        source_language="Hindi",
        source_script="Devanagari",
        target_script="Cyrillic",
        words=_build_hindi_words_via_transliteration(
            target_script="Cyrillic",
            fallback_transliteration_map=HINDI_IN_CYRILLIC,
        ),
        is_control=False,
        expected_rescue=True,  # Should be harder
    ),

    # DISTANT: Hindi in Greek
    "hindi_greek": ScriptPair(
        name="hindi_greek",
        source_language="Hindi",
        source_script="Devanagari",
        target_script="Greek",
        words=_build_hindi_words_via_transliteration(
            target_script="Greek",
            fallback_transliteration_map=HINDI_IN_GREEK,
        ),
        is_control=False,
        expected_rescue=True,  # Should be hardest
    ),

    # WORKSHOP LANE: benchmark-backed romanized/native transliteration pairs.
    # These must be populated from external JSONL manifests (for example
    # Aksharantar) rather than from repo-local transliteration-library output.
    "latin_hindi": _external_only_pair(
        "latin_hindi",
        source_language="Hindi",
        target_script="Devanagari",
    ),
    "latin_tamil": _external_only_pair(
        "latin_tamil",
        source_language="Tamil",
        target_script="Tamil",
    ),
    "latin_telugu": _external_only_pair(
        "latin_telugu",
        source_language="Telugu",
        target_script="Telugu",
    ),

    # SECOND SOURCE LANGUAGE: English in Devanagari
    "english_devanagari": ScriptPair(
        name="english_devanagari",
        source_language="English",
        source_script="Latin",
        target_script="Devanagari",
        words=[{"english": w, "source": w, "target": ENGLISH_IN_DEVANAGARI[w]} for w in ENGLISH_IN_DEVANAGARI],
        is_control=False,
        expected_rescue=True,
    ),

    # SECOND SOURCE LANGUAGE: English in Telugu
    "english_telugu": ScriptPair(
        name="english_telugu",
        source_language="English",
        source_script="Latin",
        target_script="Telugu",
        words=[{"english": w, "source": w, "target": ENGLISH_IN_TELUGU[w]} for w in ENGLISH_IN_TELUGU],
        is_control=False,
        expected_rescue=True,
    ),

    # SECOND SOURCE LANGUAGE: English in Georgian (cross-family)
    "english_georgian": ScriptPair(
        name="english_georgian",
        source_language="English",
        source_script="Latin",
        target_script="Georgian",
        words=[
            {"english": w, "source": w, "target": ENGLISH_IN_GEORGIAN[w]}
            for w in ENGLISH_IN_GEORGIAN
        ],
        is_control=False,
        expected_rescue=True,
    ),

    # English in Thai (library-generated via aksharamukha when available)
    "english_thai": ScriptPair(
        name="english_thai",
        source_language="English",
        source_script="Latin",
        target_script="Thai",
        words=[{"english": w, "source": w, "target": ENGLISH_IN_THAI[w]} for w in ENGLISH_IN_THAI],
        is_control=False,
        expected_rescue=True,
    ),

    # English in Cyrillic (Russian) (library-generated via aksharamukha when available)
    "english_cyrillic": ScriptPair(
        name="english_cyrillic",
        source_language="English",
        source_script="Latin",
        target_script="Cyrillic (Russian)",
        words=[{"english": w, "source": w, "target": ENGLISH_IN_CYRILLIC[w]} for w in ENGLISH_IN_CYRILLIC],
        is_control=False,
        expected_rescue=True,
    ),

    # English in Hebrew (library-generated via aksharamukha when available)
    "english_hebrew": ScriptPair(
        name="english_hebrew",
        source_language="English",
        source_script="Latin",
        target_script="Hebrew",
        words=[{"english": w, "source": w, "target": ENGLISH_IN_HEBREW[w]} for w in ENGLISH_IN_HEBREW],
        is_control=False,
        expected_rescue=True,
    ),

    # English in Japanese Katakana (library-generated via aksharamukha when available)
    "english_katakana": ScriptPair(
        name="english_katakana",
        source_language="English",
        source_script="Latin",
        target_script="Japanese (Katakana)",
        words=[{"english": w, "source": w, "target": ENGLISH_IN_KATAKANA[w]} for w in ENGLISH_IN_KATAKANA],
        is_control=False,
        expected_rescue=True,
    ),

    # English in Arabic (library-generated via aksharamukha when available)
    "english_arabic": ScriptPair(
        name="english_arabic",
        source_language="English",
        source_script="Latin",
        target_script="Arabic",
        words=[{"english": w, "source": w, "target": ENGLISH_IN_ARABIC[w]} for w in ENGLISH_IN_ARABIC],
        is_control=False,
        expected_rescue=True,
    ),

    # THIRD SOURCE LANGUAGE: Spanish in non-Latin scripts
    "spanish_telugu": ScriptPair(
        name="spanish_telugu",
        source_language="Spanish",
        source_script="Latin",
        target_script="Telugu",
        words=[{"english": w, "source": w, "target": SPANISH_IN_TELUGU[w]} for w in SPANISH_IN_TELUGU],
        is_control=False,
        expected_rescue=True,
    ),
    "spanish_devanagari": ScriptPair(
        name="spanish_devanagari",
        source_language="Spanish",
        source_script="Latin",
        target_script="Devanagari",
        words=[{"english": w, "source": w, "target": SPANISH_IN_DEVANAGARI[w]} for w in SPANISH_IN_DEVANAGARI],
        is_control=False,
        expected_rescue=True,
    ),
    "spanish_thai": ScriptPair(
        name="spanish_thai",
        source_language="Spanish",
        source_script="Latin",
        target_script="Thai",
        words=[{"english": w, "source": w, "target": SPANISH_IN_THAI[w]} for w in SPANISH_IN_THAI],
        is_control=False,
        expected_rescue=True,
    ),
    "spanish_cyrillic": ScriptPair(
        name="spanish_cyrillic",
        source_language="Spanish",
        source_script="Latin",
        target_script="Cyrillic (Russian)",
        words=[{"english": w, "source": w, "target": SPANISH_IN_CYRILLIC[w]} for w in SPANISH_IN_CYRILLIC],
        is_control=False,
        expected_rescue=True,
    ),
    "spanish_hebrew": ScriptPair(
        name="spanish_hebrew",
        source_language="Spanish",
        source_script="Latin",
        target_script="Hebrew",
        words=[{"english": w, "source": w, "target": SPANISH_IN_HEBREW[w]} for w in SPANISH_IN_HEBREW],
        is_control=False,
        expected_rescue=True,
    ),
    "spanish_katakana": ScriptPair(
        name="spanish_katakana",
        source_language="Spanish",
        source_script="Latin",
        target_script="Japanese (Katakana)",
        words=[{"english": w, "source": w, "target": SPANISH_IN_KATAKANA[w]} for w in SPANISH_IN_KATAKANA],
        is_control=False,
        expected_rescue=True,
    ),
    "spanish_arabic": ScriptPair(
        name="spanish_arabic",
        source_language="Spanish",
        source_script="Latin",
        target_script="Arabic",
        words=[{"english": w, "source": w, "target": SPANISH_IN_ARABIC[w]} for w in SPANISH_IN_ARABIC],
        is_control=False,
        expected_rescue=True,
    ),

    # FOURTH SOURCE LANGUAGE: French in non-Latin scripts (wordfreq-generated when available)
    "french_telugu": ScriptPair(
        name="french_telugu",
        source_language="French",
        source_script="Latin",
        target_script="Telugu",
        words=[{"english": w, "source": w, "target": FRENCH_IN_TELUGU[w]} for w in FRENCH_IN_TELUGU],
        is_control=False,
        expected_rescue=True,
    ),
    "french_devanagari": ScriptPair(
        name="french_devanagari",
        source_language="French",
        source_script="Latin",
        target_script="Devanagari",
        words=[{"english": w, "source": w, "target": FRENCH_IN_DEVANAGARI[w]} for w in FRENCH_IN_DEVANAGARI],
        is_control=False,
        expected_rescue=True,
    ),
    "french_thai": ScriptPair(
        name="french_thai",
        source_language="French",
        source_script="Latin",
        target_script="Thai",
        words=[{"english": w, "source": w, "target": FRENCH_IN_THAI[w]} for w in FRENCH_IN_THAI],
        is_control=False,
        expected_rescue=True,
    ),
    "french_cyrillic": ScriptPair(
        name="french_cyrillic",
        source_language="French",
        source_script="Latin",
        target_script="Cyrillic (Russian)",
        words=[{"english": w, "source": w, "target": FRENCH_IN_CYRILLIC[w]} for w in FRENCH_IN_CYRILLIC],
        is_control=False,
        expected_rescue=True,
    ),
    "french_hebrew": ScriptPair(
        name="french_hebrew",
        source_language="French",
        source_script="Latin",
        target_script="Hebrew",
        words=[{"english": w, "source": w, "target": FRENCH_IN_HEBREW[w]} for w in FRENCH_IN_HEBREW],
        is_control=False,
        expected_rescue=True,
    ),
    "french_katakana": ScriptPair(
        name="french_katakana",
        source_language="French",
        source_script="Latin",
        target_script="Japanese (Katakana)",
        words=[{"english": w, "source": w, "target": FRENCH_IN_KATAKANA[w]} for w in FRENCH_IN_KATAKANA],
        is_control=False,
        expected_rescue=True,
    ),
    "french_arabic": ScriptPair(
        name="french_arabic",
        source_language="French",
        source_script="Latin",
        target_script="Arabic",
        words=[{"english": w, "source": w, "target": FRENCH_IN_ARABIC[w]} for w in FRENCH_IN_ARABIC],
        is_control=False,
        expected_rescue=True,
    ),

    # CONTROL: Hindi in Devanagari (native - ceiling)
    "hindi_native": ScriptPair(
        name="hindi_native",
        source_language="Hindi",
        source_script="Devanagari",
        target_script="Devanagari",
        words=[{"english": e, "source": h, "target": h} for e, h in HINDI_WORDS_BASE],
        is_control=True,
        expected_rescue=False,  # No rescue needed
    ),
}


def get_script_pair(name: str) -> ScriptPair:
    """Get a script pair by name."""
    if name not in SCRIPT_PAIRS:
        raise ValueError(f"Unknown script pair: {name}. Available: {list(SCRIPT_PAIRS.keys())}")
    return SCRIPT_PAIRS[name]


def get_all_script_pairs() -> Dict[str, ScriptPair]:
    """Get all script pairs."""
    return SCRIPT_PAIRS


def get_experimental_pairs() -> List[ScriptPair]:
    """Get non-control script pairs for experiments."""
    return [p for p in SCRIPT_PAIRS.values() if not p.is_control]


def get_control_pairs() -> List[ScriptPair]:
    """Get control script pairs."""
    return [p for p in SCRIPT_PAIRS.values() if p.is_control]


# Summary when run directly
if __name__ == "__main__":
    print("=" * 60)
    print("MULTI-SCRIPT CONFIGURATION")
    print("=" * 60)
    print(f"\nTotal script pairs: {len(SCRIPT_PAIRS)}")

    print("\nExperimental pairs:")
    for pair in get_experimental_pairs():
        print(f"  {pair.name}: {pair.source_language} ({pair.source_script}) → {pair.target_script}")
        print(f"    Words: {len(pair.words)}")

    print("\nControl pairs:")
    for pair in get_control_pairs():
        print(f"  {pair.name}: {pair.source_language} ({pair.source_script}) → {pair.target_script}")
        print(f"    Words: {len(pair.words)}")
