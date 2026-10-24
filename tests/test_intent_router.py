"""Tests for core.intent turn router."""

from core.intent import (
    is_small_talk,
    is_sub_registrar_domain,
    resolve_turn_intent,
    select_persona,
)


def test_scanned_with_is_general_not_sub_registrar():
    assert not is_sub_registrar_domain("scanned with ?")
    ti = resolve_turn_intent("scanned with ?", channel="chat", has_doc_scope=True)
    assert ti.domain == "general"
    assert ti.persona == "general"
    assert ti.need_retrieval is True


def test_section_34_is_sub_registrar_chat():
    assert is_sub_registrar_domain("what is section 34 means ?")
    ti = resolve_turn_intent(
        "what is section 34 means ?", channel="chat", has_doc_scope=True
    )
    assert ti.domain == "sub_registrar"
    assert ti.persona == "sub_registrar"
    assert ti.retrieval_intent == "section_lookup"


def test_voice_always_nova_persona():
    ti = resolve_turn_intent(
        "what is section 34 means ?", channel="voice", has_doc_scope=True
    )
    assert ti.persona == "nova"
    assert select_persona(channel="voice", domain="sub_registrar") == "nova"


def test_identity_intent():
    ti = resolve_turn_intent("what is her name ?", channel="chat", has_doc_scope=True)
    assert ti.retrieval_intent == "identity_lookup"
    assert ti.need_retrieval is True


def test_small_talk_no_retrieval():
    assert is_small_talk("hello")
    ti = resolve_turn_intent("hello", channel="voice", has_doc_scope=False)
    assert ti.route == "small_talk"
    assert ti.need_retrieval is False


def test_summary_before_field():
    ti = resolve_turn_intent(
        "summarize this property document", channel="chat", has_doc_scope=True
    )
    assert ti.retrieval_intent == "summary"
