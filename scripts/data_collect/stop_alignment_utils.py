import re


STOP_PATTERNS = [
    r"\bstop\s+(?:by|near|at|in front of|beside|under|next to|on|inside)\b",
    r"\byou(?:'ll| will)\s+stop\b",
    r"\bhalt\b",
]

RELATION_PATTERNS = [
    "in front of",
    "next to",
    "just before",
    "inside",
    "under",
    "near",
    "beside",
    "before",
    "by",
    "at",
    "on",
    "in",
]

WEAK_ELEMENT_PATTERNS = [
    r"^there$",
    r"^here$",
    r"^it$",
    r"^them$",
    r"^the room$",
    r"^room$",
    r"^area$",
    r"^place$",
    r"^location$",
    r"^left$",
    r"^right$",
    r"^straight$",
]

ACTION_PREFIX_PATTERNS = [
    r"^(?:walk|go|head|move|continue|proceed|travel|turn|take|make|veer|keep|exit|enter|pass|passed)\b.*?\b(?:to|toward|towards|into|inside|past|through|before|near|beside|by|at|on|in)\s+",
    r"^(?:walk|go|head|move|continue|proceed|travel|pass|passed)\s+straight\s+",
    r"^(?:turn|take|make)\s+(?:a\s+)?(?:left|right)\b\s*(?:and\s+)?",
    r"^(?:stop|wait|halt)\s+(?:there|here)\b\s*",
]

ACTION_WORD_PATTERNS = [
    r"\bwalk\b",
    r"\bgo\b",
    r"\bturn\b",
    r"\btake\b",
    r"\bmake\b",
    r"\bcontinue\b",
    r"\bproceed\b",
    r"\bmove\b",
    r"\bkeep\b",
    r"\bstraight\b",
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
        r"\bstop\s+(?:by|near|at|beside|under|next to|in front of|on|inside)\s+(.*)$",
        r"\byou(?:'ll| will)\s+stop\s+(?:by|near|at|beside|under|next to|in front of|on|inside)\s+(.*)$",
        r"\bwait\s+(?:by|near|at|beside|under|next to|in front of|in|on|inside)\s+(.*)$",
        r"\bhalt\s+(?:by|near|at|beside|under|next to|in front of|on|inside)\s+(.*)$",
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


def extract_relation_hint(text: str) -> str:
    lowered = text.strip().lower()
    for relation in RELATION_PATTERNS:
        pattern = r"\b" + re.escape(relation).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, lowered):
            return relation
    return ""


def normalize_element_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"^[,.;:()\[\]{}\-\s]+|[,.;:()\[\]{}\-\s]+$", "", text)
    text = re.sub(r"\b(?:the|a|an)\b\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_action_prefix(text: str) -> str:
    cleaned = text.strip()
    previous = None
    while cleaned and cleaned != previous:
        previous = cleaned
        for pattern in ACTION_PREFIX_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip(" .")
    return cleaned or text.strip()


def _split_candidate_elements(text: str) -> list[str]:
    pieces = re.split(
        r"\s*(?:,| and | or | with | past | just before | before | next to | near | beside | under | inside | in front of )\s*",
        text,
        flags=re.IGNORECASE,
    )
    return [piece.strip(" .") for piece in pieces if piece.strip(" .")]


def _is_usable_element(text: str) -> bool:
    lowered = normalize_element_text(text)
    if not lowered:
        return False
    if any(re.search(pattern, lowered) for pattern in WEAK_ELEMENT_PATTERNS):
        return False
    if len(lowered) <= 1:
        return False
    if len(lowered.split()) > 6:
        return False
    if any(re.search(pattern, lowered) for pattern in ACTION_WORD_PATTERNS):
        return False
    return True


def extract_stop_target_elements(instruction: str, max_elements: int = 4) -> list[str]:
    stop_phrase = extract_stop_phrase(instruction)
    stop_object_phrase = extract_stop_object_phrase(instruction)
    source = stop_object_phrase if is_usable_stop_object_phrase(stop_object_phrase) else stop_phrase
    working = strip_action_prefix(source.strip(" ."))
    lowered = working.lower()

    relation = extract_relation_hint(working)
    if relation:
        match = re.search(r"\b" + re.escape(relation).replace(r"\ ", r"\s+") + r"\b", lowered)
        if match and match.start() == 0:
            working = working[match.end() :].strip(" .")

    candidates = _split_candidate_elements(working)
    if not candidates:
        candidates = [working]

    deduped = []
    seen = set()
    for candidate in candidates:
        candidate = strip_action_prefix(candidate)
        norm = normalize_element_text(candidate)
        if not _is_usable_element(norm):
            continue
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(norm)
        if len(deduped) >= max_elements:
            break

    if not deduped and _is_usable_element(stop_object_phrase):
        deduped = [normalize_element_text(stop_object_phrase)]
    return deduped


def extract_stop_action_text(instruction: str) -> str:
    stop_phrase = extract_stop_phrase(instruction).strip(" .")
    stop_object_phrase = extract_stop_object_phrase(instruction).strip(" .")
    relation = extract_relation_hint(stop_phrase)

    if stop_object_phrase and stop_object_phrase.lower() != stop_phrase.lower():
        lowered_phrase = stop_phrase.lower()
        lowered_object = stop_object_phrase.lower()
        idx = lowered_phrase.find(lowered_object)
        if idx >= 0:
            action_text = (stop_phrase[:idx] + stop_phrase[idx + len(stop_object_phrase) :]).strip(" ,.")
        else:
            action_text = stop_phrase
    else:
        action_text = stop_phrase

    action_text = re.sub(r"\s+", " ", action_text).strip(" ,.")
    if relation and relation not in action_text.lower():
        action_text = f"{action_text} {relation}".strip()
    if not action_text:
        action_text = stop_phrase
    return action_text


def build_stop_alignment_prompt(stop_phrase: str) -> str:
    return (
        "You are verifying whether the current view matches the final stopping condition.\n"
        f"Target stopping description: {stop_phrase}\n"
        "Focus on the final landmark/object and the relative position described there."
    )
