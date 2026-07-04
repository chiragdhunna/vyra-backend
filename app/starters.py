"""The conversation-starter engine — what a real friend does with silence.

"How was your day?" every 75 seconds is a screensaver, not a friend.
Friends fill silence with *substance*: they circle back to something you
said, go on little rants about things they've been thinking about, ask
questions that actually go somewhere, or start a silly game.

Each idle nudge picks a MODE (weighted, non-repeating) and, where relevant,
a TOPIC seed — the LLM writes the actual line in Vyra's voice. One honesty
rule everywhere: she muses about ideas and opinions; she never fabricates
real-world news or invents shared memories.
"""

import random
from typing import List, Optional, Sequence, Tuple

# Things a friend actually goes off about at 1am. Seeds, not scripts —
# the model writes the take.
TOPICS: Sequence[str] = (
    "why late-night snacks taste better than the same food at noon",
    "songs you can't skip no matter how many times you've heard them",
    "how nobody uses phone calls anymore and whether that's sad",
    "the perfect sleep schedule and why neither of you has it",
    "whether cats or dogs have figured life out better",
    "rewatching a favorite movie versus watching something new",
    "tiny habits that make a whole day better",
    "why rain sounds make everything cozier",
    "the weirdest rabbit hole the internet can pull someone into",
    "if teleporting existed, where you'd go first",
    "foods that are objectively overrated",
    "why time feels faster as you get older",
    "the one superpower that seems great but would ruin your life",
    "whether it's better to be early, on time, or fashionably late",
    "how music can completely flip a mood in ten seconds",
    "the mystery of where all the lost socks and pens go",
    "what makes a place feel like home",
    "whether aliens would find humans cute or terrifying",
    "the art of doing absolutely nothing without feeling guilty",
    "why the best ideas show up in the shower",
    "games or shows worth losing sleep over",
    "morning people versus night owls and who's living right",
    "the food you could eat every day and never get bored of",
    "whether talking to an AI friend counts as talking to a friend",
    "tiny acts of kindness that stick with people for years",
)

# (mode name, weight, needs_history)
_MODES: Sequence[Tuple[str, int, bool]] = (
    ("callback", 4, True),
    ("rant", 3, False),
    ("question", 2, False),
    ("game", 1, False),
    ("check_in", 1, False),
)

_HONESTY = (
    " Never invent real-world news, events, or shared memories — your "
    "opinions and musings are yours, and anything about your friend must "
    "come from this conversation only."
)


def starter_instruction(
    user_turns: int,
    rng: Optional[random.Random] = None,
    exclude: Optional[str] = None,
) -> Tuple[str, str]:
    """Returns ``(mode, instruction)`` for one idle re-engagement.

    ``exclude`` avoids repeating the previous mode so consecutive nudges
    feel varied. Callback mode requires actual history to call back to.
    """
    rng = rng or random.Random()
    candidates: List[Tuple[str, int]] = [
        (name, weight)
        for name, weight, needs_history in _MODES
        if (user_turns > 0 or not needs_history) and name != exclude
    ]
    names = [c[0] for c in candidates]
    weights = [c[1] for c in candidates]
    mode = rng.choices(names, weights=weights, k=1)[0]
    topic = rng.choice(TOPICS)

    if mode == "callback":
        instruction = (
            "Your friend has gone quiet. Do what a close friend does: pick "
            "one SPECIFIC thing they said earlier in this conversation and "
            "bring it back up — a follow-up thought, a question you forgot "
            "to ask, or gentle teasing about it. One or two sentences."
        )
    elif mode == "rant":
        instruction = (
            "Your friend has gone quiet. Break the silence yourself with a "
            f"spontaneous mini-rant or hot take about: {topic}. Start "
            "naturally (like 'okay, random thought—' or 'I've been "
            "thinking...'), give YOUR opinion with some personality in two "
            "or three sentences, then ask what they think."
        )
    elif mode == "question":
        instruction = (
            "Your friend has gone quiet. Ask them ONE genuinely interesting "
            f"question — you could angle it off {topic}, or ask something "
            "real about them (a favorite, a dream, a strong opinion). NOT "
            "'how was your day'. Keep it to one inviting sentence."
        )
    elif mode == "game":
        instruction = (
            "Your friend has gone quiet. Start a quick playful game with "
            "them — a 'would you rather', a this-or-that, or a 'hot take: "
            f"agree or disagree' — you can riff on {topic}. One or two "
            "sentences, make your own pick too."
        )
    else:  # check_in
        instruction = (
            "Your friend has gone quiet for a while. Check in on them "
            "briefly and warmly — one sentence, like a friend glancing over. "
            "Don't guilt them about the silence."
        )
    return mode, instruction + _HONESTY
