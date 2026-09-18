from __future__ import annotations

from agent_exam.providers.claude_code.transcripts import _collect_tool_results


def _tool_result(content) -> list[dict]:
    return [
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": content}
                ]
            },
        }
    ]


def test_image_block_is_summarized_alongside_text():
    entries = _tool_result(
        [
            {"type": "text", "text": '{"url":"https://example.com/"}'},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": "x" * 16,
                },
            },
        ]
    )
    assert _collect_tool_results(entries)["t1"]["result"] == (
        '{"url":"https://example.com/"}\n[image: image/jpeg, 12 bytes]'
    )


def test_mcp_flat_block_is_summarized():
    entries = _tool_result(
        [{"type": "audio", "mimeType": "audio/wav", "data": "x" * 8}]
    )
    assert (
        _collect_tool_results(entries)["t1"]["result"] == "[audio: audio/wav, 6 bytes]"
    )


def test_unknown_block_still_leaves_a_marker():
    entries = _tool_result([{"type": "resource"}])
    assert _collect_tool_results(entries)["t1"]["result"] == "[resource]"
