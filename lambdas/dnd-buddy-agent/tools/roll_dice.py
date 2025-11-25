"""
Dice rolling tool for D&D.
"""
import re
import random
import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@tool
def roll_dice(dice_notation: str) -> str:
    """
    Roll dice using standard D&D notation.
    
    Use this tool when the user asks to roll dice for:
    - Ability checks (e.g., "roll perception" -> use 1d20)
    - Attack rolls (e.g., "roll to hit" -> use 1d20+modifier)
    - Damage rolls (e.g., "roll damage for longsword" -> use 1d8+modifier)
    - Initiative (e.g., "roll initiative" -> use 1d20+modifier)
    - Any other dice rolls requested
    
    Args:
        dice_notation: Standard D&D dice notation
                      Format: [number]d[sides][+/-modifier]
                      Examples: '2d6+3', '1d20', '4d8-2', 'd20'
        
    Returns:
        Roll results showing individual dice values and total
        
    Examples:
        - '2d6+3' rolls two 6-sided dice and adds 3
        - '1d20' rolls one 20-sided die  
        - 'd20' rolls one 20-sided die (1 is implied)
        - '3d8+5' rolls three 8-sided dice and adds 5
    """
    logger.info(f"roll_dice: {dice_notation}")
    
    # Parse dice notation
    pattern = r'^(\d*)d(\d+)([+-]\d+)?$'
    match = re.match(pattern, dice_notation.lower().strip())
    
    if not match:
        logger.warning(f"Invalid dice notation: {dice_notation}")
        return f"Invalid dice notation: {dice_notation}. Use format like '2d6+3', '1d20', or 'd20'"
    
    num_dice = int(match.group(1)) if match.group(1) else 1
    die_size = int(match.group(2))
    modifier = int(match.group(3)) if match.group(3) else 0
    
    # Validate
    if num_dice < 1 or num_dice > 100:
        return "Number of dice must be between 1 and 100"
    if die_size < 2 or die_size > 1000:
        return "Die size must be between 2 and 1000"
    
    # Roll dice
    rolls = [random.randint(1, die_size) for _ in range(num_dice)]
    total = sum(rolls) + modifier
    
    # Format result
    rolls_str = ", ".join(str(r) for r in rolls)
    modifier_str = f" {modifier:+d}" if modifier != 0 else ""
    
    result = f"🎲 Rolling {dice_notation}:\n"
    result += f"Rolls: [{rolls_str}]{modifier_str}\n"
    result += f"Total: {total}"
    
    return result
