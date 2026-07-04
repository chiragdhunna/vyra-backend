from app.personality import EMOTIONS, EmotionTagFilter, parse_emotion


class TestParseEmotion:
    def test_strips_trailing_tag(self):
        text, emotion = parse_emotion("So glad you're here!\n[emotion: happy]")
        assert text == "So glad you're here!"
        assert emotion == "happy"

    def test_case_and_whitespace_insensitive(self):
        text, emotion = parse_emotion("Hey.[ Emotion :  ANGRY ]")
        assert text == "Hey."
        assert emotion == "angry"

    def test_last_tag_wins_and_all_are_stripped(self):
        raw = "One [emotion: sad] two [emotion: excited]"
        text, emotion = parse_emotion(raw)
        assert text == "One  two"
        assert emotion == "excited"

    def test_unknown_emotion_falls_back_to_neutral(self):
        text, emotion = parse_emotion("Hmm [emotion: banana]")
        assert text == "Hmm"
        assert emotion == "neutral"

    def test_no_tag(self):
        text, emotion = parse_emotion("Just words.")
        assert text == "Just words."
        assert emotion == "neutral"

    def test_angry_is_a_known_emotion(self):
        assert "angry" in EMOTIONS


class TestEmotionTagFilter:
    def _run(self, deltas):
        f = EmotionTagFilter()
        out = []
        for d in deltas:
            out.extend(f.feed(d))
        out.append(f.flush())
        return "".join(out), f.emotion

    def test_tag_split_across_deltas_is_suppressed(self):
        text, emotion = self._run(["Hi there!", "\n[emo", "tion: ca", "ring]"])
        assert text.strip() == "Hi there!"
        assert emotion == "caring"

    def test_plain_brackets_pass_through(self):
        text, emotion = self._run(["I like [most] fruits [a lot]"])
        assert text == "I like [most] fruits [a lot]"
        assert emotion is None

    def test_text_after_tag_still_flows(self):
        text, emotion = self._run(["A[emotion: happy]B"])
        assert text == "AB"
        assert emotion == "happy"

    def test_unterminated_bracket_flushes(self):
        text, emotion = self._run(["Wait [emotio"])
        assert text == "Wait [emotio"
        assert emotion is None
