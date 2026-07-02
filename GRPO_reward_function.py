"""
Reward Functions for ISO 29148 Requirements Compliance
Based on 42 INCOSE transformation rules
"""

import re
from typing import Dict, List


def reward_xml_format(text: str) -> float:
    """
    Reward properly formatted XML output with <answer> tags.
    
    Args:
        text: The generated text to check
        
    Returns:
        float: 1.0 if properly formatted, 0.0 otherwise
    """
    text = text.strip()
    
    # Check for both opening and closing tags
    has_opening = "<answer>" in text.lower()
    has_closing = "</answer>" in text.lower()
    
    if has_opening and has_closing:
        # Check if opening comes before closing
        opening_idx = text.lower().find("<answer>")
        closing_idx = text.lower().find("</answer>")
        
        if opening_idx < closing_idx:
            return 1.0
    
    return 0.0


def extract_xml_answer(text: str) -> str:
    """
    Extract the content between <answer> tags.
    
    Args:
        text: The generated text containing XML tags
        
    Returns:
        str: The extracted answer or the original text if no tags found
    """
    text = text.strip()
    
    # Try to extract content between <answer> tags
    match = re.search(r'<answer>(.*?)</answer>', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If no tags found, return original text
    return text


def reward_iso_compliance(text: str) -> float:
    """
    Comprehensive reward based on all 42 INCOSE transformation rules.
    
    Categories:
    1. Shall Language (Rules 1-5)
    2. Structure & Clarity (Rules 6-10)
    3. Specificity & Precision (Rules 11-15)
    4. Verifiability & Measurability (Rules 16-20)
    5. Completeness (Rules 21-25)
    6. Consistency (Rules 26-30)
    7. Traceability (Rules 31-35)
    8. Quality Attributes (Rules 36-42)
    
    Returns:
        float: Score between 0.0 and 1.0
    """
    # Extract answer from XML if present
    answer = extract_xml_answer(text)
    answer_lower = answer.lower()
    
    score = 0.0
    total_checks = 42
    
    # ============================================================
    # Category 1: Shall Language (Rules 1-5) - 5 points
    # ============================================================
    # Rule 1: Use "shall" for mandatory requirements
    if "shall" in answer_lower:
        score += 1
    
    # Rule 2: Avoid "should" (weak requirement)
    if "should" not in answer_lower:
        score += 1
    
    # Rule 3: Avoid "may" (optional)
    if "may" not in answer_lower:
        score += 1
    
    # Rule 4: Avoid "will" (declaration, not requirement)
    if "will" not in answer_lower:
        score += 1
    
    # Rule 5: Use present tense for shall statements
    if "shall" in answer_lower and not re.search(r'shall\s+have\s+\w+ed', answer_lower):
        score += 1
    
    # ============================================================
    # Category 2: Structure & Clarity (Rules 6-10) - 5 points
    # ============================================================
    # Rule 6: One requirement per statement
    sentences = re.split(r'[.!?]', answer)
    valid_sentences = [s for s in sentences if s.strip()]
    if len(valid_sentences) <= 3:  # Prefer concise, focused requirements
        score += 1
    
    # Rule 7: Avoid compound requirements (and/or)
    if not re.search(r'\s+and\s+', answer_lower) or answer_lower.count(' and ') <= 1:
        score += 1
    
    # Rule 8: Avoid negative requirements (shall not be vague, prefer positive)
    negative_patterns = len(re.findall(r'shall not|must not|cannot', answer_lower))
    if negative_patterns <= 1:
        score += 1
    
    # Rule 9: Use active voice
    passive_indicators = ['is required', 'are required', 'be provided', 'be performed']
    if not any(ind in answer_lower for ind in passive_indicators):
        score += 1
    
    # Rule 10: Clear subject-verb-object structure
    if re.search(r'(the system|the product|the software|the \w+)\s+shall', answer_lower):
        score += 1
    
    # ============================================================
    # Category 3: Specificity & Precision (Rules 11-15) - 5 points
    # ============================================================
    # Rule 11: Avoid vague terms
    vague_terms = ['appropriate', 'adequate', 'user-friendly', 'easy', 'fast', 
                   'efficient', 'flexible', 'robust', 'good', 'bad', 'nice',
                   'normal', 'typical', 'sufficient', 'suitable']
    vague_count = sum(1 for term in vague_terms if term in answer_lower)
    if vague_count == 0:
        score += 1
    
    # Rule 12: Use specific quantities
    has_numbers = bool(re.search(r'\d+', answer))
    if has_numbers:
        score += 1
    
    # Rule 13: Include units for measurements
    unit_patterns = r'\d+\s*(m|km|s|ms|kg|g|mb|gb|%|degrees?|celsius|fahrenheit)'
    if re.search(unit_patterns, answer_lower) or not has_numbers:
        score += 1
    
    # Rule 14: Define acronyms on first use
    acronyms = re.findall(r'\b[A-Z]{2,}\b', answer)
    if len(acronyms) <= 2:  # Penalize excessive unexplained acronyms
        score += 1
    
    # Rule 15: Avoid ambiguous pronouns (it, this, that without antecedent)
    ambiguous_pronouns = len(re.findall(r'\b(it|this|that)\s+shall\b', answer_lower))
    if ambiguous_pronouns == 0:
        score += 1
    
    # ============================================================
    # Category 4: Verifiability & Measurability (Rules 16-20) - 5 points
    # ============================================================
    # Rule 16: Include measurable criteria
    if has_numbers or any(word in answer_lower for word in ['within', 'at least', 'no more than', 'between']):
        score += 1
    
    # Rule 17: Specify conditions and constraints
    if any(word in answer_lower for word in ['when', 'if', 'under', 'during', 'while']):
        score += 1
    
    # Rule 18: Define acceptable ranges
    if re.search(r'\d+\s*(?:to|-)\s*\d+', answer) or 'between' in answer_lower:
        score += 1
    
    # Rule 19: Specify tolerances
    if '+/-' in answer or '±' in answer or 'tolerance' in answer_lower:
        score += 1
    elif not has_numbers:  # Don't penalize if no measurements
        score += 1
    
    # Rule 20: Define success criteria
    if any(word in answer_lower for word in ['accurate', 'correct', 'complete', 'successful']):
        score += 1
    
    # ============================================================
    # Category 5: Completeness (Rules 21-25) - 5 points
    # ============================================================
    # Rule 21: Include all necessary conditions
    word_count = len(answer.split())
    if word_count >= 10:  # Sufficient detail
        score += 1
    
    # Rule 22: Specify all inputs and outputs
    if any(word in answer_lower for word in ['input', 'output', 'provide', 'receive', 'accept']):
        score += 1
    
    # Rule 23: Define operational modes
    if any(word in answer_lower for word in ['mode', 'state', 'condition', 'operation']):
        score += 1
    
    # Rule 24: Include environmental conditions
    if any(word in answer_lower for word in ['temperature', 'environment', 'condition', 'operating']):
        score += 1
    
    # Rule 25: Specify interfaces
    if any(word in answer_lower for word in ['interface', 'connection', 'communication', 'api']):
        score += 1
    
    # ============================================================
    # Category 6: Consistency (Rules 26-30) - 5 points
    # ============================================================
    # Rule 26: Use consistent terminology
    # (Difficult to check without context, give benefit of doubt)
    score += 1
    
    # Rule 27: Maintain consistent format
    if "shall" in answer_lower:
        score += 1
    
    # Rule 28: Use standard units
    if not re.search(r'\d+\s*(?:feet|inches|miles|pounds)', answer_lower):
        score += 1
    
    # Rule 29: Follow naming conventions
    # (Give benefit of doubt if no obvious violations)
    score += 1
    
    # Rule 30: Consistent requirement structure
    if answer.strip().endswith('.'):
        score += 1
    
    # ============================================================
    # Category 7: Traceability (Rules 31-35) - 5 points
    # ============================================================
    # Rule 31: Unique identifier capability
    # (Assumed to be handled externally)
    score += 1
    
    # Rule 32: Reference to source requirement
    # (Give benefit for being transformable)
    score += 1
    
    # Rule 33: Link to parent requirement
    # (Assumed to be handled externally)
    score += 1
    
    # Rule 34: Version control compatibility
    # (Give benefit of doubt)
    score += 1
    
    # Rule 35: Change tracking capability
    # (Give benefit of doubt)
    score += 1
    
    # ============================================================
    # Category 8: Quality Attributes (Rules 36-42) - 7 points
    # ============================================================
    # Rule 36: Atomic (single requirement)
    if valid_sentences and len(valid_sentences) <= 2:
        score += 1
    
    # Rule 37: Complete (self-contained)
    if word_count >= 8:
        score += 1
    
    # Rule 38: Consistent (no contradictions)
    # (Give benefit of doubt)
    score += 1
    
    # Rule 39: Correct (technically accurate)
    # (Assume correct if well-formed)
    if "shall" in answer_lower:
        score += 1
    
    # Rule 40: Feasible (implementable)
    # (Give benefit of doubt)
    score += 1
    
    # Rule 41: Necessary (adds value)
    # (Give benefit of doubt)
    score += 1
    
    # Rule 42: Unambiguous (clear meaning)
    if vague_count == 0 and ambiguous_pronouns == 0:
        score += 1
    
    # Normalize to 0-1 range
    return score / total_checks


def reward_no_vague_terms(text: str) -> float:
    """
    Penalize use of vague, subjective terms.
    
    Returns:
        float: 1.0 if no vague terms, 0.0 if many vague terms
    """
    answer = extract_xml_answer(text)
    answer_lower = answer.lower()
    
    vague_terms = [
        'appropriate', 'adequate', 'user-friendly', 'easy', 'fast', 
        'efficient', 'flexible', 'robust', 'good', 'bad', 'nice',
        'normal', 'typical', 'sufficient', 'suitable', 'reasonable',
        'acceptable', 'satisfactory', 'optimal', 'better', 'worse',
        'high', 'low', 'large', 'small', 'quick', 'slow'
    ]
    
    vague_count = sum(1 for term in vague_terms if term in answer_lower)
    
    # Exponential penalty
    if vague_count == 0:
        return 1.0
    elif vague_count == 1:
        return 0.5
    else:
        return max(0.0, 1.0 - (vague_count * 0.3))


def reward_measurability(text: str) -> float:
    """
    Reward presence of measurable, verifiable criteria.
    
    Returns:
        float: Score based on measurability indicators
    """
    answer = extract_xml_answer(text)
    answer_lower = answer.lower()
    
    score = 0.0
    
    # Check for numerical values
    if re.search(r'\d+', answer):
        score += 0.4
    
    # Check for units
    unit_patterns = r'\d+\s*(m|km|s|ms|kg|g|mb|gb|%|degrees?|celsius|fahrenheit|hz|mhz|ghz)'
    if re.search(unit_patterns, answer_lower):
        score += 0.3
    
    # Check for ranges or comparisons
    if re.search(r'\d+\s*(?:to|-|\.\.)\s*\d+', answer):
        score += 0.2
    elif any(word in answer_lower for word in ['at least', 'no more than', 'between', 'within', 'maximum', 'minimum']):
        score += 0.2
    
    # Check for conditions
    if any(word in answer_lower for word in ['when', 'if', 'under', 'during']):
        score += 0.1
    
    return min(1.0, score)


def reward_shall_language(text: str) -> float:
    """
    Reward proper use of "shall" language and penalize weak terms.
    
    Returns:
        float: Score based on requirement language strength
    """
    answer = extract_xml_answer(text)
    answer_lower = answer.lower()
    
    score = 0.0
    
    # Strong positive: uses "shall"
    if "shall" in answer_lower:
        score += 0.6
    
    # Weak negatives
    if "should" in answer_lower:
        score -= 0.3
    if "may" in answer_lower:
        score -= 0.2
    if "will" in answer_lower:
        score -= 0.2
    if "might" in answer_lower:
        score -= 0.3
    
    return max(0.0, min(1.0, score))


def reward_appropriate_length(text: str, min_words: int = 10, max_words: int = 50) -> float:
    """
    Reward appropriate requirement length.
    Too short = incomplete, too long = compound requirement
    
    Args:
        text: The text to evaluate
        min_words: Minimum acceptable word count
        max_words: Maximum acceptable word count
        
    Returns:
        float: Score based on length appropriateness
    """
    answer = extract_xml_answer(text)
    word_count = len(answer.split())
    
    if word_count < min_words:
        # Penalize too short
        return word_count / min_words
    elif word_count > max_words:
        # Penalize too long
        excess = word_count - max_words
        penalty = min(0.5, excess * 0.02)
        return max(0.5, 1.0 - penalty)
    else:
        # Ideal range
        return 1.0


def compute_combined_reward(text: str, weights: Dict[str, float] = None) -> Dict[str, float]:
    """
    Compute combined reward from all reward functions.
    
    Args:
        text: The generated text to evaluate
        weights: Optional dictionary of weights for each reward component
        
    Returns:
        dict: Dictionary containing individual rewards and total
    """
    if weights is None:
        weights = {
            'xml_format': 0.10,
            'iso_compliance': 0.40,
            'no_vague': 0.15,
            'measurability': 0.15,
            'shall_language': 0.10,
            'length': 0.10
        }
    
    rewards = {
        'xml_format': reward_xml_format(text),
        'iso_compliance': reward_iso_compliance(text),
        'no_vague': reward_no_vague_terms(text),
        'measurability': reward_measurability(text),
        'shall_language': reward_shall_language(text),
        'length': reward_appropriate_length(text)
    }
    
    # Compute weighted total
    total_reward = sum(rewards[key] * weights[key] for key in rewards.keys())
    
    rewards['total'] = total_reward
    rewards['weights'] = weights
    
    return rewards


if __name__ == "__main__":
    # Test the reward functions
    test_cases = [
        {
            'name': 'Good Requirement',
            'text': '<answer>The system shall respond to user input within 200 milliseconds under normal operating conditions.</answer>'
        },
        {
            'name': 'Vague Requirement',
            'text': '<answer>The system should be fast and user-friendly.</answer>'
        },
        {
            'name': 'No XML Format',
            'text': 'The system shall process data quickly.'
        }
    ]
    
    print("=" * 80)
    print("REWARD FUNCTION TEST RESULTS")
    print("=" * 80)
    
    for test in test_cases:
        print(f"\n{test['name']}:")
        print(f"Text: {test['text']}")
        print("-" * 80)
        
        rewards = compute_combined_reward(test['text'])
        
        print(f"XML Format:      {rewards['xml_format']:.3f}")
        print(f"ISO Compliance:  {rewards['iso_compliance']:.3f}")
        print(f"No Vague Terms:  {rewards['no_vague']:.3f}")
        print(f"Measurability:   {rewards['measurability']:.3f}")
        print(f"Shall Language:  {rewards['shall_language']:.3f}")
        print(f"Length:          {rewards['length']:.3f}")
        print(f"TOTAL REWARD:    {rewards['total']:.3f}")
