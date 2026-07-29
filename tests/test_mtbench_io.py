from __future__ import annotations

import pandas as pd
import pytest

from biases.mtbench_io import (
    canonical_conversation_json,
    serialize_conversation_columns,
)


def test_serialize_conversation_columns_emits_canonical_json_without_mutation() -> None:
    frame = pd.DataFrame(
        {
            "conversation_a": [
                [{"role": "user", "content": "Question?"}],
            ],
            "conversation_b": [
                {
                    "messages": [
                        {"role": "assistant", "content": "Answer."},
                    ]
                },
            ],
            "winner": ["model_a"],
        }
    )

    serialized = serialize_conversation_columns(frame)

    assert serialized.loc[0, "conversation_a"] == (
        '[{"content":"Question?","role":"user"}]'
    )
    assert serialized.loc[0, "conversation_b"] == (
        '{"messages":[{"content":"Answer.","role":"assistant"}]}'
    )
    assert isinstance(frame.loc[0, "conversation_a"], list)
    assert isinstance(frame.loc[0, "conversation_b"], dict)


@pytest.mark.parametrize("value", ("plain text", 42, None))
def test_canonical_conversation_json_rejects_scalar_values(value: object) -> None:
    with pytest.raises(TypeError, match="list or dict"):
        canonical_conversation_json(value)
