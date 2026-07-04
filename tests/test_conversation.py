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


def test_session_history_bounded_and_counted():
    history = SessionHistory(max_turns=4)
    for i in range(6):
        history.add_user(f"u{i}")
        history.add_assistant(f"a{i}")
    assert len(history.turns) == 4
    assert history.turns[-1].content == "a5"
    assert history.user_turn_count == 2
