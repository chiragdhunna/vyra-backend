from app.conversation import SessionHistory, build_messages, context_note
from app.schemas import ChatTurn, VisionContext


def turns(*pairs):
    return [ChatTurn(role=r, content=c) for r, c in pairs]


def test_system_prompt_first_and_history_appended():
    messages = build_messages(turns(("user", "hi"), ("assistant", "hey!")))
    assert messages[0]["role"] == "system"
    assert "Vyra" in messages[0]["content"]
    assert "[emotion: X]" in messages[0]["content"]
    assert messages[1:] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hey!"},
    ]


def test_history_window_keeps_most_recent():
    history = turns(*[("user", f"m{i}") for i in range(30)])
    messages = build_messages(history, max_history_turns=10)
    assert len(messages) == 11  # system + 10
    assert messages[1]["content"] == "m20"
    assert messages[-1]["content"] == "m29"


def test_context_note_variants():
    assert context_note() == ""
    assert "Chirag" in context_note(user_name="Chirag")
    smiling = context_note(vision=VisionContext(present=True, smiling=True))
    assert "smiling" in smiling
    away = context_note(vision=VisionContext(present=False))
    assert "can't see" in away


def test_context_and_instruction_land_in_system():
    messages = build_messages(
        turns(("user", "hi")),
        user_name="Chirag",
        vision=VisionContext(present=True, smiling=False),
        extra_instruction="Greet them first.",
    )
    system = messages[0]["content"]
    assert "Chirag" in system
    assert "present" in system
    assert "Greet them first." in system


def test_instruction_with_empty_history_becomes_user_message():
    # System-only prompts make instruct models (llama3.1 etc.) emit an
    # instant end-of-turn -> empty reply. Greeting/nudge instructions must
    # therefore arrive as a user turn when there is no history yet.
    messages = build_messages([], extra_instruction="Greet them first.")
    assert messages[0]["role"] == "system"
    assert "Greet them first." not in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "Greet them first."}


def test_session_history_bounded_and_counted():
    history = SessionHistory(max_turns=4)
    for i in range(6):
        history.add_user(f"u{i}")
        history.add_assistant(f"a{i}")
    assert len(history.turns) == 4
    assert history.turns[-1].content == "a5"
    assert history.user_turn_count == 2


def test_noise_transcript_filter():
    from app.realtime.session import is_noise_transcript

    # Whisper's greatest hits on silence/echo:
    assert is_noise_transcript("You")
    assert is_noise_transcript("you.")
    assert is_noise_transcript("and and and and and and and")
    assert is_noise_transcript("We'll see you in the next one.")
    assert is_noise_transcript("Thanks for watching!")
    assert is_noise_transcript("  ")
    # Real speech must pass:
    assert not is_noise_transcript("Hi there, am I audible?")
    assert not is_noise_transcript("Oh, really?")
    assert not is_noise_transcript("you know what happened today")
    assert not is_noise_transcript("No, tell me more")


def test_shared_whisper_singleton_shape():
    # build_stt('fake') returns fresh engines; whisper path is cached.
    from app.config import Settings
    from app.realtime import stt as stt_mod

    fake_settings = Settings(stt_provider="fake")
    a = stt_mod.build_stt(fake_settings)
    b = stt_mod.build_stt(fake_settings)
    assert a is not b  # fakes are per-session (tests rely on fresh queues)
    disabled = stt_mod.build_stt(Settings(stt_provider="client"))
    assert disabled is None
