import re


STOP_PATTERNS = [
    r"\bstop\s+(?:by|near|at|in front of|beside|under|next to)\b",
    r"\byou(?:'ll| will)\s+stop\b",
    r"\bhalt\b",
]


def split_instruction_sentences(instruction: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\s{2,}", instruction.strip())
    return [part.strip(" .") for part in parts if part and part.strip(" .")]


def extract_stop_phrase(instruction: str) -> str:
    sentences = split_instruction_sentences(instruction)
    for sentence in reversed(sentences):
        lowered = sentence.lower()
        if any(re.search(pattern, lowered) for pattern in STOP_PATTERNS):
            return sentence
    return sentences[-1] if sentences else instruction.strip()


def extract_stop_object_phrase(instruction: str) -> str:
    phrase = extract_stop_phrase(instruction).strip(" .")
    lowered = phrase.lower()
    patterns = [
        r"\bstop\s+(?:by|near|at|beside|under|next to|in front of)\s+(.*)$",
        r"\byou(?:'ll| will)\s+stop\s+(?:by|near|at|beside|under|next to|in front of)\s+(.*)$",
        r"\bwait\s+(?:by|near|at|beside|under|next to|in front of|in)\s+(.*)$",
        r"\bhalt\s+(?:by|near|at|beside|under|next to|in front of)\s+(.*)$",
        r"\bstop\s+when\s+you\s+are\s+inside\s+(.*)$",
        r"\bstop\s+once\s+you\s+are\s+inside\s+(.*)$",
        r"\bstop\s+once\s+you\s+are\s+just\s+past\s+(.*)$",
        r"\bstop\s+after\s+you\s+pass\s+(.*)$",
        r"\bwait\s+(.*)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            start = match.start(1)
            candidate = phrase[start:].strip(" .")
            if candidate:
                return candidate
    if " archway" in lowered:
        return "archway"
    if " doorway" in lowered:
        return "doorway"
    if " cabinet" in lowered:
        tail = lowered[lowered.rfind("cabinet") :]
        return phrase[len(phrase) - len(tail) :].strip(" .")
    if " door" in lowered:
        tail = lowered[lowered.rfind("door") :]
        return phrase[len(phrase) - len(tail) :].strip(" .")
    if " room" in lowered:
        room_match = re.search(r"((?:the\s+)?[a-zA-Z -]*room)\b", phrase, re.IGNORECASE)
        if room_match:
            return room_match.group(1).strip(" .")
    return phrase


def is_usable_stop_object_phrase(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered:
        return False
    weak_patterns = [
        r"^stop there$",
        r"^that's where you will wait$",
        r"^there$",
        r"\bstop\b",
        r"\bwait\b",
        r"\bhalt\b",
    ]
    if any(re.search(pattern, lowered) for pattern in weak_patterns):
        return False
    if len(lowered.split()) > 8:
        return False
    return True


def build_stop_alignment_prompt(stop_phrase: str) -> str:
    return (
        "You are verifying whether the current view matches the final stopping condition.\n"
        f"Target stopping description: {stop_phrase}\n"
        "Focus on the final landmark/object and the relative position described there."
    )
