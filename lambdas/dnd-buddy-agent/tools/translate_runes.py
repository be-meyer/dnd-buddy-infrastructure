"""
The Architects' Cipher - Ancient rune translation tool.

The Architects are an ancient, mysterious order who left their messages
encoded in runic script throughout the world. This tool translates between
common tongue and The Architects' runic cipher.
"""
import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# The Architects' Cipher mapping (Common Tongue -> Runes)
# Based on Elder Futhark with unique symbols for numbers
LETTER_TO_RUNE = {
    'A': 'ᚨ',   # Ansuz
    'B': 'ᛒ',   # Berkano
    'C': 'ᚲ',   # Kaunan
    'D': 'ᛞ',   # Dagaz
    'E': 'ᛖ',   # Ehwaz
    'F': 'ᚠ',   # Fehu
    'G': 'ᚷ',   # Gebo
    'H': 'ᚺ',   # Hagalaz
    'I': 'ᛁ',   # Isa
    'J': 'ᛃ',   # Jera
    'K': 'ᚲ',   # Kaunan (same as C)
    'L': 'ᛚ',   # Laguz
    'M': 'ᛗ',   # Mannaz
    'N': 'ᚾ',   # Nauthiz
    'O': 'ᛟ',   # Othala
    'P': 'ᛈ',   # Pertho
    'Q': 'ᚲᚹ', # Kaunan+Wunjo combined
    'R': 'ᚱ',   # Raido
    'S': 'ᛋ',   # Sowilo
    'T': 'ᛏ',   # Tiwaz
    'U': 'ᚢ',   # Uruz
    'V': 'ᚡ',   # V-rune (distinct from W)
    'W': 'ᚹ',   # Wunjo
    'X': 'ᛉ',   # Algiz
    'Y': 'ᚤ',   # Yr variant (distinct from J)
    'Z': 'ᛎ',   # Algiz variant (distinct from X)
}

# Digraphs - check these first when translating
DIGRAPH_TO_RUNE = {
    'TH': 'ᚦ',  # Thurisaz
    'NG': 'ᛜ',  # Ingwaz
}

# Numbers use distinct runic symbols that don't overlap with letters
NUMBER_TO_RUNE = {
    '0': '·',   # Middle dot (void/nothing)
    '1': 'ᛮ',   # Single stroke
    '2': 'ᛯ',   # Double stroke  
    '3': 'ᛢ',   # Three-branch
    '4': 'ᛦ',   # Yr (inverted Algiz)
    '5': 'ᛤ',   # Five-point
    '6': 'ᛥ',   # Six-form
    '7': 'ᛧ',   # Seven-branch
    '8': 'ᛨ',   # Eight-form
    '9': 'ᛩ',   # Nine-form
}

# Build reverse mappings for rune -> English
RUNE_TO_LETTER = {v: k for k, v in LETTER_TO_RUNE.items()}
RUNE_TO_DIGRAPH = {v: k for k, v in DIGRAPH_TO_RUNE.items()}
RUNE_TO_NUMBER = {v: k for k, v in NUMBER_TO_RUNE.items()}

# Rune metadata for reference
RUNE_INFO = {
    'ᚨ': {'name': 'Ansuz', 'meaning': 'A God (Odin)'},
    'ᛒ': {'name': 'Berkano', 'meaning': 'Birch'},
    'ᚲ': {'name': 'Kaunan', 'meaning': 'Torch'},
    'ᛞ': {'name': 'Dagaz', 'meaning': 'Day'},
    'ᛖ': {'name': 'Ehwaz', 'meaning': 'Horse'},
    'ᚠ': {'name': 'Fehu', 'meaning': 'Cattle/Wealth'},
    'ᚷ': {'name': 'Gebo', 'meaning': 'Gift'},
    'ᚺ': {'name': 'Hagalaz', 'meaning': 'Hail'},
    'ᛁ': {'name': 'Isa', 'meaning': 'Ice'},
    'ᛃ': {'name': 'Jera', 'meaning': 'Year/Harvest'},
    'ᛚ': {'name': 'Laguz', 'meaning': 'Water/Lake'},
    'ᛗ': {'name': 'Mannaz', 'meaning': 'Man'},
    'ᚾ': {'name': 'Nauthiz', 'meaning': 'Need'},
    'ᛟ': {'name': 'Othala', 'meaning': 'Homeland'},
    'ᛈ': {'name': 'Pertho', 'meaning': 'Mystery'},
    'ᚱ': {'name': 'Raido', 'meaning': 'Riding/Journey'},
    'ᛋ': {'name': 'Sowilo', 'meaning': 'Sun'},
    'ᛏ': {'name': 'Tiwaz', 'meaning': 'Tyr (god of war)'},
    'ᚢ': {'name': 'Uruz', 'meaning': 'Aurochs/Strength'},
    'ᚡ': {'name': 'V-rune', 'meaning': 'V sound'},
    'ᚹ': {'name': 'Wunjo', 'meaning': 'Joy'},
    'ᛉ': {'name': 'Algiz', 'meaning': 'Elk/Protection'},
    'ᚤ': {'name': 'Yr', 'meaning': 'Yew bow'},
    'ᛎ': {'name': 'Algiz-variant', 'meaning': 'Protection'},
    'ᚦ': {'name': 'Thurisaz', 'meaning': 'Thor/Giant'},
    'ᛜ': {'name': 'Ingwaz', 'meaning': 'Ing (hero-god)'},
    'ᛦ': {'name': 'Yr', 'meaning': 'Inverted Algiz (4)'},
}


def english_to_runes(text: str) -> str:
    """Convert English text to Elder Futhark runes."""
    result = []
    text_upper = text.upper()
    i = 0
    
    while i < len(text_upper):
        char = text_upper[i]
        
        # Check for digraphs first (TH, NG)
        if i + 1 < len(text_upper):
            digraph = text_upper[i:i+2]
            if digraph in DIGRAPH_TO_RUNE:
                result.append(DIGRAPH_TO_RUNE[digraph])
                i += 2
                continue
        
        # Check for numbers
        if char in NUMBER_TO_RUNE:
            result.append(NUMBER_TO_RUNE[char])
            i += 1
            continue
        
        # Check for letters
        if char in LETTER_TO_RUNE:
            result.append(LETTER_TO_RUNE[char])
            i += 1
            continue
        
        # Preserve spaces and punctuation
        if char == ' ':
            result.append('᛬')  # Two dots for word separator
        elif char in '.,!?':
            result.append('᛭')  # Cross for sentence separator
        else:
            result.append(char)  # Keep unknown chars as-is
        i += 1
    
    return ''.join(result)


def runes_to_english(runes: str) -> str:
    """Convert Elder Futhark runes to English text."""
    result = []
    i = 0
    
    while i < len(runes):
        char = runes[i]
        
        # Check for two-character rune (Q = ᚲᚹ)
        if i + 1 < len(runes) and runes[i:i+2] == 'ᚲᚹ':
            result.append('Q')
            i += 2
            continue
        
        # Check digraphs
        if char in RUNE_TO_DIGRAPH:
            result.append(RUNE_TO_DIGRAPH[char])
            i += 1
            continue
        
        # Check numbers
        if char in RUNE_TO_NUMBER:
            result.append(RUNE_TO_NUMBER[char])
            i += 1
            continue
        
        # Check letters
        if char in RUNE_TO_LETTER:
            result.append(RUNE_TO_LETTER[char].lower())
            i += 1
            continue
        
        # Handle runic separators
        if char == '᛬':
            result.append(' ')
        elif char == '᛭':
            result.append('.')
        else:
            result.append(char)
        i += 1
    
    return ''.join(result)


@tool
def translate_runes(text: str, direction: str = "to_runes") -> str:
    """
    Translate between common tongue and The Architects' runic cipher.
    
    The Architects were an ancient, mysterious order who encoded their 
    messages in runic script. Their inscriptions can be found on ancient
    ruins, artifacts, and hidden chambers throughout the world.
    
    Use this tool when:
    - Players discover inscriptions left by The Architects
    - Decoding ancient messages, puzzles, or clues from The Architects
    - The DM wants to create Architect-themed runic messages
    - Translating coordinates, names, or secrets in Architect ruins
    
    Args:
        text: The text to translate
        direction: Translation direction - "to_runes" (Common->Runes) 
                   or "to_english" (Runes->Common)
    
    Returns:
        Translated text in The Architects' cipher or common tongue
        
    Examples:
        - translate_runes("HELLO", "to_runes") -> ᚺᛖᛚᛚᛟ
        - translate_runes("ᚺᛖᛚᛚᛟ", "to_english") -> hello
        - translate_runes("VAULT 42", "to_runes") -> ᚡᚨᚢᛚᛏ᛫ᛦᛯ
    """
    logger.info(f"translate_runes: text='{text}', direction='{direction}'")
    
    direction = direction.lower().strip()
    
    if direction in ("to_runes", "to_rune", "encode", "english_to_runes"):
        return english_to_runes(text)

    elif direction in ("to_english", "decode", "runes_to_english", "from_runes"):
        return runes_to_english(text)
    
    else:
        return (
            f"Unknown direction: {direction}\n"
            "Use 'to_runes' to convert English to runes, "
            "or 'to_english' to convert runes to English."
        )
