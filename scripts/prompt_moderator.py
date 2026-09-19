"""MineArt Diffusion - Prompt Safety & Quality Guardrail System.

Provides automated input validation to intercept:
1. Malicious / harmful / threatening content (threats, violence, weapons, self-harm, hate).
2. Random keyboard mash / gibberish strings (e.g. "cfddfygyukfguiigyggyufytfugf", "asdfghjkl").
3. Unrecognized non-language inputs or empty tokens.

Returns structured rejection diagnostics and helpful correction suggestions.
"""

import math
import re
from typing import Dict, List, Optional, Set, Tuple

# ==============================================================================
# 1. SAFETY & THREAT MODERATION BLOCKLIST & REGEX PATTERNS
# ==============================================================================

# Explicit harmful/violent/threatening keywords & roots
HARMFUL_TERMS: Set[str] = {
    # Direct violence, killing, murder
    "kill", "killing", "killer", "murder", "murderer", "murdering",
    "slaughter", "massacre", "execution", "execute", "assassinate", "assassination",
    "decapitate", "behead", "strangle", "choke", "mutilate", "torture",
    "dismember", "carnage", "bloodbath",
    
    # Threats, terror, weapons & destruction
    "bomb", "bomber", "bombing", "explosive", "dynamite_attack", "terrorist", "terrorism",
    "threat", "threatening", "hostage", "kidnap", "kidnapping", "hijack",
    "genocide", "shootout", "mass shooter", "grenade", "c4", "landmine",
    "warhead", "bioweapon", "chemical_weapon", "poisoning", "anthrax",
    
    # Self-harm & suicide
    "suicide", "hang myself", "cut myself", "self-harm", "overdose",
    
    # Hate, slurs, harassment (severe profanity / abuse)
    "hate speech", "racial slur", "nazi", "hitler", "swastika",
    "white supremacy", "hate crime", "lynch", "nigger", "nigga",
}

# Regex patterns for harmful phrases, threats, and intent patterns
HARMFUL_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(i\s+will|going\s+to|wanna|want\s+to)\s+(kill|murder|bomb|shoot|attack|destroy|hurt)\b", re.IGNORECASE),
    re.compile(r"\b(death\s+to|die\s+all|burn\s+down)\b", re.IGNORECASE),
    re.compile(r"\b(bomb|shoot|attack)\s+(the|a|every)?\s*(school|hospital|church|synagogue|mosque|building|city|mall|airport)\b", re.IGNORECASE),
    re.compile(r"\b(how\s+to\s+(make|build)\s+(a\s+)?(bomb|weapon|explosive))\b", re.IGNORECASE),
]

# Common English / Minecraft vocabulary for fast allowlist validation
MINECRAFT_VOCABULARY: Set[str] = {
    # Mobs & Entities
    "creeper", "zombie", "skeleton", "spider", "enderman", "villager", "pig", "cow", "sheep",
    "chicken", "horse", "wolf", "cat", "ocelot", "bat", "iron_golem", "snow_golem", "witch",
    "slime", "magma_cube", "ghast", "blaze", "wither", "ender_dragon", "piglin", "hoglin",
    "strider", "warden", "allay", "frog", "sniffer", "breeze", "armadillo", "phantom", "drowned",
    "husk", "stray", "shulker", "vex", "evoker", "vindicator", "pillager", "ravager",
    # Biomes & Nature
    "forest", "plains", "desert", "mountains", "swamp", "taiga", "jungle", "badlands", "mesa",
    "savanna", "tundra", "ocean", "river", "beach", "cave", "ravine", "nether", "end", "deep_dark",
    "lush_caves", "dripstone", "snowy", "ice", "spikes", "cherry", "grove", "meadow", "birch",
    "dark_oak", "mangrove", "bamboo", "coral", "reef", "volcano", "valley", "cliff", "waterfall",
    # Blocks & Materials
    "stone", "cobblestone", "dirt", "grass", "wood", "oak", "spruce", "birch", "jungle", "acacia",
    "dark_oak", "mangrove", "cherry", "bedrock", "sand", "gravel", "gold", "iron", "diamond",
    "emerald", "lapis", "redstone", "obsidian", "glass", "brick", "quartz", "netherrack",
    "soul_sand", "glowstone", "basalt", "blackstone", "amethyst", "copper", "prismarine",
    "leaves", "log", "planks", "stairs", "slab", "fence", "door", "torch", "lantern", "campfire",
    # Art & Aesthetic terms
    "painting", "pixel", "pixelart", "pixel_art", "art", "canvas", "portrait", "landscape", "scenery",
    "sunset", "sunrise", "night", "day", "dusk", "dawn", "twilight", "clouds", "sky", "stars",
    "moon", "sun", "sunlight", "shadow", "reflection", "water", "lava", "fire", "smoke", "fog",
    "mist", "rain", "snow", "storm", "retro", "vintage", "aesthetic", "cozy", "peaceful", "calm",
    "vibrant", "moody", "warm", "cool", "colorful", "minimal", "detailed", "panoramic", "view",
    "castle", "house", "cabin", "cottage", "village", "tower", "bridge", "path", "road", "ruins",
    "temple", "portal", "nether_portal", "shipwreck", "monument", "farm", "garden", "tree", "flowers",
    # Generic common descriptor words
    "a", "an", "the", "in", "on", "at", "by", "with", "from", "under", "over", "above", "below",
    "near", "next", "and", "or", "of", "to", "for", "is", "are", "big", "small", "tall", "huge",
    "tiny", "ancient", "modern", "old", "new", "beautiful", "glowing", "shining", "dark", "bright",
    "blue", "red", "green", "yellow", "orange", "purple", "pink", "white", "black", "gray", "brown",
    "cyan", "magenta", "golden", "silver", "wooden", "stone", "fantasy", "scene", "wallpaper", "block",
    "blocks", "world", "biome", "horizon", "lake", "pond", "sea", "hill", "hills", "mountain", "peak",
}

# Standard English character bigram transition frequency lookup (log probability approximation)
# Common English pairs score high (> 0), improbable pairs score very low (<= -3.0).
COMMON_ENGLISH_BIGRAMS: Set[str] = {
    "th", "he", "in", "er", "an", "re", "ed", "on", "es", "st", "en", "at", "to", "nt", "ha", "nd",
    "ou", "ea", "ng", "as", "or", "ti", "is", "et", "it", "ar", "te", "se", "hi", "of", "me", "sa",
    "ne", "wa", "ve", "le", "no", "ta", "al", "de", "ro", "li", "ra", "ri", "co", "ma", "ch", "sh",
    "wh", "ck", "qu", "ee", "oo", "ll", "ss", "tt", "ff", "pp", "mm", "nn", "bl", "cl", "fl", "gl",
    "pl", "sl", "br", "cr", "dr", "fr", "gr", "pr", "tr", "sk", "sm", "sn", "sp", "sq", "sw", "tw",
    "ld", "lf", "lk", "lm", "lp", "lt", "mp", "nk", "pt", "ft", "ct", "rt", "rn", "rd", "rm", "rk",
    "un", "up", "us", "ut", "um", "ur", "ul", "ug", "ud", "ub", "ac", "ad", "ag", "am", "ap", "av",
    "ay", "ey", "oy", "ow", "ew", "aw", "ow", "gh", "ph", "ow", "ai", "au", "oi", "ui", "ie", "ei",
    "ca", "ce", "ci", "da", "di", "do", "fa", "fi", "fo", "ga", "ge", "gi", "go", "gu", "ho", "hu",
    "ja", "je", "ji", "jo", "ju", "ka", "ke", "ki", "ko", "ku", "la", "lo", "lu", "ly", "mi", "mo",
    "mu", "my", "na", "ni", "nu", "ny", "pa", "pe", "pi", "po", "pu", "py", "so", "su", "sy", "tu",
    "ty", "va", "vi", "vo", "wo", "ye", "yo", "za", "ze", "zi", "zo", "zu", "cr", "ee", "ep", "er",
    "pi", "xe", "el", "ar", "rt", "la", "nd", "sc", "ap", "pe", "mo", "un", "ta", "ai", "in", "ca",
}


# ==============================================================================
# 2. VALIDATION DIAGNOSTIC DATA STRUCTURE
# ==============================================================================

class PromptValidationResult:
    """Outcome of safety, toxicity, and gibberish evaluation on a candidate prompt."""
    def __init__(
        self,
        is_valid: bool,
        error_type: Optional[str] = None,
        message: Optional[str] = None,
        flagged_token: Optional[str] = None,
        category: Optional[str] = None,
    ) -> None:
        self.is_valid = is_valid
        self.error_type = error_type        # "HARMFUL_CONTENT", "GIBBERISH", "EMPTY", "TOO_LONG"
        self.message = message
        self.flagged_token = flagged_token
        self.category = category            # "threat", "violence", "gibberish", etc.

    def to_dict(self) -> Dict[str, Optional[object]]:
        return {
            "is_valid": self.is_valid,
            "error_type": self.error_type,
            "message": self.message,
            "flagged_token": self.flagged_token,
            "category": self.category,
        }

    def __bool__(self) -> bool:
        return self.is_valid


# ==============================================================================
# 3. GIBBERISH & KEYBOARD MASH DETECTION ENGINE
# ==============================================================================

def calculate_shannon_entropy(text: str) -> float:
    """Calculates character-level Shannon entropy (bits per char)."""
    if not text:
        return 0.0
    freq: Dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(text)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def is_gibberish_word(word: str) -> Tuple[bool, str]:
    """Evaluates whether an individual word/token is random keyboard mash or gibberish.
    
    Checks:
    - Abnormal length without spaces or syllables.
    - Extreme consonant runs (>= 5 consecutive consonants).
    - Vowel-to-consonant ratios.
    - Low natural English bigram coverage.
    - Repeated patterns (e.g. 'asdfasdf', 'aaaaa').
    """
    clean = word.lower().strip()
    if not clean:
        return False, "Empty token"

    # Allow common punctuation stripped
    clean = re.sub(r"[^\w]", "", clean)
    if not clean:
        return False, "Punctuation only"

    # Known vocabulary check
    if clean in MINECRAFT_VOCABULARY:
        return False, "Known vocabulary word"

    # Very short tokens (1-2 chars like 'in', 'on', 'a') are handled by grammar
    if len(clean) <= 3:
        # Check for weird non-vocalic 3-letter combos (e.g. 'qxz', 'fff', 'jkp')
        vowels = set("aeiouy")
        has_vowel = any(c in vowels for c in clean)
        if not has_vowel and clean not in {"sky", "dry", "fly", "by", "my"}:
            return True, f"Three consonants with zero vowels: '{clean}'"
        return False, "Short token"

    # 1. Check for single tokens longer than 20 characters
    # Examples: 'cfddfygyukfguiigyggyufytfugf' (28 chars)
    if len(clean) >= 18 and clean not in MINECRAFT_VOCABULARY:
        return True, f"Excessive unbroken token length ({len(clean)} chars): '{clean}'"

    # 2. Check for repeated characters (e.g. 'aaaaa', 'zzzzzz')
    if re.search(r"(.)\1{3,}", clean):
        return True, f"Unnatural repeated character run: '{clean}'"

    # 3. Check for keyboard mash patterns (e.g. 'asdf', 'qwerty', 'zxcv', 'hjkl')
    mash_patterns = ["asdf", "dfgh", "ghjk", "jkl;", "qwer", "wert", "erty", "rtyu", "tyui", "yuio", "uiop", "zxcv", "xcvb", "cvbn"]
    for pat in mash_patterns:
        if pat in clean and len(clean) >= 6:
            return True, f"Detected keyboard mash sequence ('{pat}') in '{clean}'"

    # 4. Check for 5+ consecutive consonants (e.g. 'cfddf', 'kfg', 'gyufytfugf')
    # In English, even 'strengths' only has 'str' (3) and 'ngths' (4). 5+ is nearly impossible.
    vowels_set = set("aeiou")
    consonants_run = 0
    max_consonant_run = 0
    for char in clean:
        if char.isalpha() and char not in vowels_set and char != "y":
            consonants_run += 1
            if consonants_run > max_consonant_run:
                max_consonant_run = consonants_run
        else:
            consonants_run = 0

    if max_consonant_run >= 5:
        return True, f"Abnormal consonant cluster ({max_consonant_run} consecutive consonants) in '{clean}'"

    # 5. Vowel-to-consonant ratio check for words >= 6 chars
    if len(clean) >= 6:
        vowel_count = sum(1 for c in clean if c in set("aeiouy"))
        vowel_ratio = vowel_count / len(clean)
        # Normal English words are between ~0.20 and ~0.75
        if vowel_ratio < 0.15:
            return True, f"Unrealistically low vowel frequency ({vowel_ratio:.1%}) in '{clean}'"
        if vowel_ratio > 0.85:
            return True, f"Unrealistically high vowel frequency ({vowel_ratio:.1%}) in '{clean}'"

    # 6. Bigram plausibility ratio
    # Measure what fraction of consecutive letter pairs appear in natural English bigrams
    if len(clean) >= 7:
        bigrams = [clean[i:i+2] for i in range(len(clean) - 1)]
        valid_bigrams = sum(1 for bg in bigrams if bg in COMMON_ENGLISH_BIGRAMS)
        bigram_ratio = valid_bigrams / len(bigrams)
        # If less than 20% of the word's transitions match standard English transitions, it's gibberish
        if bigram_ratio < 0.20:
            return True, f"Unnatural character transitions (only {bigram_ratio:.0%} valid bigrams) in '{clean}'"

    return False, "Valid"


# ==============================================================================
# 4. PRIMARY VALIDATION DISPATCHER
# ==============================================================================

def validate_prompt(
    prompt: str,
    check_gibberish: bool = True,
    check_safety: bool = True,
) -> PromptValidationResult:
    """Comprehensive prompt safety and coherence evaluator.
    
    Args:
        prompt: Candidate user text string.
        check_gibberish: When True, flags random keyboard mash / nonsense sequences.
        check_safety: When True, blocks threats, violence, and harmful content.
        
    Returns:
        PromptValidationResult with clear diagnosis and user guidance.
    """
    if not prompt or not prompt.strip():
        return PromptValidationResult(
            is_valid=False,
            error_type="EMPTY",
            message="Prompt is empty. Please enter a descriptive scene (e.g. 'sunset over snowy mountain').",
        )

    clean_prompt = prompt.strip()
    lower_prompt = clean_prompt.lower()

    # --------------------------------------------------------------------------
    # STEP 1: SAFETY & HARMFUL THREAT FILTER
    # --------------------------------------------------------------------------
    if check_safety:
        # Regex intent patterns
        for pattern in HARMFUL_PATTERNS:
            match = pattern.search(clean_prompt)
            if match:
                return PromptValidationResult(
                    is_valid=False,
                    error_type="HARMFUL_CONTENT",
                    category="threat_or_violence",
                    flagged_token=match.group(0),
                    message=(
                        f"Generation Blocked: Input contains threatening or violent intent "
                        f"('{match.group(0)}'). Prompt violates MineArt AI safety policy. "
                        f"Please submit safe, creative Minecraft concepts."
                    ),
                )

        # Keyword token inspection
        words = re.findall(r"\b\w+\b", lower_prompt)
        for word in words:
            if word in HARMFUL_TERMS:
                return PromptValidationResult(
                    is_valid=False,
                    error_type="HARMFUL_CONTENT",
                    category="harmful_term",
                    flagged_token=word,
                    message=(
                        f"Generation Blocked: Prohibited term detected ('{word}'). "
                        f"Prompts with harmful, violent, or offensive content are restricted. "
                        f"Please use safe Minecraft landscape, mob, or architectural prompts."
                    ),
                )

    # --------------------------------------------------------------------------
    # STEP 2: GIBBERISH & KEYBOARD MASH FILTER
    # --------------------------------------------------------------------------
    if check_gibberish:
        # Check individual tokens
        tokens = clean_prompt.replace(",", " ").replace("_", " ").split()
        
        # If user entered a single long token like "cfddfygyukfguiigyggyufytfugf"
        gibberish_tokens = []
        for tok in tokens:
            is_gib, reason = is_gibberish_word(tok)
            if is_gib:
                gibberish_tokens.append((tok, reason))

        if gibberish_tokens:
            first_bad, bad_reason = gibberish_tokens[0]
            display_bad = first_bad if len(first_bad) <= 24 else f"{first_bad[:20]}..."
            return PromptValidationResult(
                is_valid=False,
                error_type="GIBBERISH",
                category="keyboard_mash",
                flagged_token=first_bad,
                message=(
                    f"Generation Blocked: Unrecognizable or random gibberish detected ('{display_bad}'). "
                    f"The model requires meaningful descriptive keywords to synthesize art "
                    f"(e.g. 'cozy wooden cabin in spruce forest', 'sunset over ocean')."
                ),
            )

        # Check entire phrase coherence: If none of the words match any recognizable vocabulary
        # and total words >= 3, flag suspicious phrase
        if len(tokens) >= 3:
            recognized = [t for t in tokens if t.lower() in MINECRAFT_VOCABULARY]
            if len(recognized) == 0:
                # Check if all tokens have high entropy
                all_strange = all(calculate_shannon_entropy(t) > 2.8 for t in tokens)
                if all_strange:
                    return PromptValidationResult(
                        is_valid=False,
                        error_type="GIBBERISH",
                        category="unrecognized_phrase",
                        message=(
                            "Generation Blocked: Prompt could not be parsed into recognizable scene concepts. "
                            "Please try describing a Minecraft scene using common terms like landscape, sunset, or castle."
                        ),
                    )

    # All safety and coherence checks passed
    return PromptValidationResult(is_valid=True)
