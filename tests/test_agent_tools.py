"""Tool-use tests (Phase 9).

Covers: the root jail against every way a path can leave it, argument
validation against the sloppy types a model emits, dispatch never raising, the
generate/act/feed-back loop and its step cap, and the vocabulary contract that
decides whether a tool name is reachable at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from astra.tools import (
    AgentSession,
    Param,
    RootJail,
    ToolError,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    extract_calls,
    make_file_tools,
    protocol_words,
    tool,
    unknown_words,
    validate_args,
)

ARTIFACT = Path(__file__).resolve().parents[1] / "tokenizer/artifacts/prose_chat_word.json"


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("\n".join(f"line {i}" for i in range(1, 30)))
    (tmp_path / "src" / "util.py").write_text("def helper():\n    return 42\n")
    (tmp_path / "README.md").write_text("# project\n\ndef main():\n    pass\n")
    (tmp_path / "notes.txt").write_text("remember the milk\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.pyc").write_bytes(b"\x00\x01")
    return tmp_path


@pytest.fixture
def registry(tree):
    return ToolRegistry(make_file_tools(RootJail(tree)))


# ----- the jail -----


@pytest.mark.parametrize(
    "escape",
    [
        "../outside.txt",
        "../../etc/passwd",
        "src/../../outside.txt",
        "/etc/passwd",
        "src/../../../etc/shadow",
        "./src/../src/../../outside.txt",
    ],
)
def test_jail_refuses_paths_outside_the_root(registry, tmp_path, escape):
    result = registry.call("read_file", {"path": escape})
    assert not result.ok
    assert "outside the tool root" in result.content


def test_jail_follows_symlinks_before_checking(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("do not leak me")
    root = tmp_path / "project"
    root.mkdir()
    (root / "link.txt").symlink_to(secret)
    (root / "ok.txt").write_text("fine")

    reg = ToolRegistry(make_file_tools(RootJail(root)))
    assert not reg.call("read_file", {"path": "link.txt"}).ok
    assert reg.call("read_file", {"path": "ok.txt"}).ok


def test_jail_allows_absolute_paths_already_inside_the_root(registry, tree):
    inside = tree / "notes.txt"
    result = registry.call("read_file", {"path": str(inside)})
    assert result.ok
    assert "milk" in result.content


def test_jail_prunes_noise_directories(registry):
    listed = registry.call("find_files", {"pattern": "**/*"}).content
    assert ".git" not in listed
    assert "__pycache__" not in listed
    assert "README.md" in listed


def test_root_must_be_a_directory(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("x")
    with pytest.raises(ToolError, match="not a directory"):
        RootJail(target)


# ----- the three tools -----


def test_read_file_slices_and_reports_the_next_start(registry):
    result = registry.call("read_file", {"path": "src/main.py", "start": 6, "limit": 3})
    assert result.ok
    assert result.content.startswith("line 7\nline 8\nline 9")
    assert "start=9" in result.note
    assert result.meta["total_lines"] == 29


def test_read_file_past_the_end_is_an_error_not_an_empty_string(registry):
    result = registry.call("read_file", {"path": "notes.txt", "start": 500})
    assert not result.ok
    assert "past the end" in result.content


def test_search_text_finds_matches_with_paths_and_lines(registry):
    result = registry.call("search_text", {"pattern": "def main"})
    assert result.ok
    assert "README.md:3: def main():" in result.content


def test_search_text_falls_back_to_literal_when_the_pattern_is_not_a_regex(registry):
    result = registry.call("search_text", {"pattern": "def main("})
    assert result.ok
    assert "literal" in result.meta["match_mode"]
    assert "README.md" in result.content


def test_search_text_respects_its_limit(registry):
    result = registry.call("search_text", {"pattern": "line", "limit": 3})
    assert result.meta["matches"] == 3
    assert result.meta["truncated"] is True


def test_find_files_glob(registry):
    result = registry.call("find_files", {"pattern": "src/*"})
    assert set(result.content.split("\n")) == {"src/main.py", "src/util.py"}


def test_find_files_reports_no_match_without_pretending(registry):
    result = registry.call("find_files", {"pattern": "nope/*"})
    assert result.ok
    assert "no paths matched" in result.content


def test_tools_refuse_patterns_the_tokenizer_cannot_spell(registry):
    """A <unk> in a pattern would otherwise match nothing and read as 'no such file'."""
    result = registry.call("find_files", {"pattern": "**/*.<unk>"})
    assert not result.ok
    assert "<unk>" in result.content


def test_read_file_refuses_binary(tmp_path):
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe\x00\x01binary")
    reg = ToolRegistry(make_file_tools(RootJail(tmp_path)))
    result = reg.call("read_file", {"path": "blob.bin"})
    assert not result.ok
    assert "utf-8" in result.content


# ----- argument validation -----


def test_unknown_argument_is_refused_not_ignored(registry):
    result = registry.call("read_file", {"path": "notes.txt", "offest": 3})
    assert not result.ok
    assert "no argument(s) offest" in result.content


def test_missing_required_argument(registry):
    result = registry.call("search_text", {})
    assert not result.ok
    assert "requires the argument pattern" in result.content


@pytest.mark.parametrize("value", ["200", 200.0, 200])
def test_integer_arguments_accept_the_loose_types_a_model_emits(registry, value):
    """A model asked for an integer answers "200" or 200.0 about half the time."""
    spec = registry.get("read_file")
    assert validate_args(spec, {"path": "a", "limit": value})["limit"] == 200
    assert isinstance(validate_args(spec, {"path": "a", "limit": value})["limit"], int)


def test_non_numeric_integer_is_refused_with_the_value_quoted(registry):
    result = registry.call("read_file", {"path": "notes.txt", "limit": "many"})
    assert not result.ok
    assert "whole number" in result.content


def test_booleans_accept_true_false_strings(registry):
    spec = registry.get("search_text")
    assert validate_args(spec, {"pattern": "x", "ignore_case": "TRUE"})["ignore_case"] is True
    assert validate_args(spec, {"pattern": "x", "ignore_case": "false"})["ignore_case"] is False


def test_search_text_ignore_case_changes_what_matches(registry):
    upper = registry.call("search_text", {"pattern": "REMEMBER"})
    lower = registry.call("search_text", {"pattern": "REMEMBER", "ignore_case": True})
    assert "notes.txt" not in upper.content
    assert "notes.txt" in lower.content


def test_validate_args_fills_defaults():
    spec = ToolSpec(
        name="x",
        description="d",
        params=(Param("a", "string", "", required=True), Param("b", "integer", "", default=7)),
        fn=lambda a, b: ToolResult(ok=True, content=f"{a}{b}"),
    )
    assert validate_args(spec, {"a": "x"}) == {"a": "x", "b": 7}


# ----- registry and dispatch -----


def test_call_returns_an_error_result_and_never_raises(registry):
    assert registry.call("no_such_tool", {}).ok is False
    assert registry.call("read_file", {}).ok is False


def test_registering_a_duplicate_name_is_refused():
    reg = ToolRegistry()
    spec = ToolSpec(name="a", description="", params=(), fn=lambda: ToolResult(ok=True, content=""))
    reg.register(spec)
    with pytest.raises(ToolError, match="already registered"):
        reg.register(spec)


def test_tool_returning_a_bare_value_is_wrapped(registry):
    reg = ToolRegistry()
    spec = ToolSpec(name="n", description="", params=(), fn=lambda: 7)
    reg.register(spec)
    assert reg.call("n", {}).content == "7"


def test_json_schemas_are_well_formed(registry):
    schemas = registry.json_schemas()
    assert {s["function"]["name"] for s in schemas} == {"read_file", "search_text", "find_files"}
    for schema in schemas:
        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "path" not in params["required"] or schema["function"]["name"] != "find_files"
    assert json.dumps(schemas)  # serialisable for a hosted-API call


def test_prompt_block_names_every_tool(registry):
    block = registry.prompt_block()
    for name in ("read_file", "search_text", "find_files"):
        assert f"tool {name}(" in block


# ----- the loop -----


def test_loop_feeds_a_result_back_and_answers(registry):
    scripted = [
        'Looking.\ntool search_text {"pattern": "line 7"}',
        'Reading it.\ntool read_file {"path": "src/main.py", "start": 6, "limit": 1}',
        "Line 7 holds the value you asked about.",
    ]
    prompts: list[str] = []

    def gen(prompt):
        prompts.append(prompt)
        return scripted[len(prompts) - 1]

    turn = AgentSession(registry, gen).ask("what is on line 7?")

    assert turn.text == scripted[2]
    assert [c.name for c in turn.calls] == ["search_text", "read_file"]
    assert not turn.exhausted
    assert "result {" in prompts[2]
    assert "line 7" in prompts[2]


def test_loop_stops_a_model_that_only_ever_calls_tools(registry):
    calls = []

    def gen(prompt):
        calls.append(prompt)
        return 'tool find_files {"pattern": "src/*"}'

    turn = AgentSession(registry, gen, max_steps=3).ask("go")
    assert turn.steps == 3
    assert turn.exhausted
    assert len(calls) == 3


def test_loop_returns_immediately_when_no_tool_is_called(registry):
    turn = AgentSession(registry, lambda p: "just an answer").ask("hi")
    assert turn.steps == 0
    assert turn.text == "just an answer"
    assert not turn.exhausted


def test_unknown_tool_name_is_answered_with_the_real_list(registry):
    turn = AgentSession(registry, lambda p: 'tool grep {"pattern": "x"}').ask("grep")
    assert 'there is no tool named' in turn.results[0]
    assert "search_text" in turn.results[0]


def test_tool_error_is_handed_back_rather_than_raised(registry):
    turn = AgentSession(registry, lambda p: 'tool read_file {"path": "../../etc/passwd"}').ask("go")
    assert "outside the tool root" in turn.results[0]


def test_malformed_json_is_reported_without_ending_the_turn(registry):
    turn = AgentSession(registry, lambda p: 'tool read_file {"path": broken').ask("go")
    assert "not valid JSON" in turn.results[0]


def test_trailing_prose_after_the_json_is_tolerated(registry):
    calls = extract_calls('tool find_files {"pattern": "src/*"} and that is all', registry)
    assert len(calls) == 1
    assert calls[0].args == {"pattern": "src/*"}


def test_a_call_missing_its_braces_is_still_extracted(registry):
    calls = extract_calls("tool read_file", registry)
    assert len(calls) == 1
    assert "no arguments" in calls[0].error


def test_result_is_truncated_to_the_configured_budget(registry):
    turn = AgentSession(
        registry,
        lambda p: 'tool read_file {"path": "src/main.py", "limit": 200}',
        max_result_chars=120,
    ).ask("go")
    assert len(turn.results[0]) <= 120
    assert "truncated" in turn.results[0]


# ----- the vocabulary contract -----


@pytest.mark.skipif(not ARTIFACT.exists(), reason="tokenizer artifact not present")
def test_every_tool_name_and_argument_is_spelled_by_the_tokenizer(registry):
    """The whole feature is unreachable if the model cannot write the words.

    The sampler forbids special ids, so a name absent from the vocabulary is a
    name the model can never emit. Measured against the prose artifact, `grep`
    and `glob` are absent and `offset` and `max_matches` are too, which is why
    the tools are named search_text, find_files, and take start and limit.
    """
    from astra.tokenizer.word import WordLevel

    tok = WordLevel.load(str(ARTIFACT))
    unspellable = [w for w in protocol_words(registry.specs) if unknown_words(w, tok)]
    assert unspellable == [], (
        f"these protocol words are absent from the tokenizer vocabulary and so can "
        f"never be generated: {unspellable}. Rename the argument, or add the word "
        f"to the tokenizer fold-in."
    )


@pytest.mark.skipif(not ARTIFACT.exists(), reason="tokenizer artifact not present")
def test_the_protocol_markers_and_envelope_are_spelled(registry):
    from astra.tokenizer.word import WordLevel

    tok = WordLevel.load(str(ARTIFACT))
    for spec in registry.specs:
        sample = f'tool {spec.name} {{"path": "notes.txt"}}\nresult {{"content": "found it"}}'
        assert unknown_words(sample, tok) == [], f"{spec.name} protocol line is not emittable"


@pytest.mark.skipif(not ARTIFACT.exists(), reason="tokenizer artifact not present")
def test_result_envelope_avoids_words_the_model_cannot_read():
    """`ok` and `meta` are both absent from the prose vocabulary."""
    from astra.tokenizer.word import WordLevel

    tok = WordLevel.load(str(ARTIFACT))
    success = ToolResult(ok=True, content="fine").to_json()
    failure = ToolResult(ok=False, content="no such file").to_json()
    assert unknown_words(success, tok) == []
    assert unknown_words(failure, tok) == []
