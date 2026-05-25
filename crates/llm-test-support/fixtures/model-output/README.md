# Model-Output Fixture Packs

These JSON packs capture small, deterministic model-facing behavior for one
model family at a time. They are intentionally text-only and must not require a
downloaded model snapshot or accelerator hardware.

Each pack contains:

- `schema_version`: fixture format version, currently `1`.
- `family` and `case_name`: stable selectors for consumers.
- `tokenizer_fixture`: repository-relative tokenizer fixture used for token ID
  assertions.
- `prompt_options`: family template options used by the consuming test.
- `tools` and `messages_before_assistant`: OpenAI-compatible request data before
  the model emits a tool call.
- `assistant_output`: raw family-specific assistant markup to feed the tool
  parser.
- `tool_result_content` and `follow_up_user`: turns appended after parsing the
  assistant tool call.
- `expected.rendered_prompt`: exact rendered prompt string after the tool
  history round trip.
- `expected.token_ids`: exact token IDs for the rendered prompt with
  `add_special_tokens = false`.
- `expected.parsed_*`: expected parser result for `assistant_output`.

To add another family, create a sibling directory under `model-output/`, keep
the chat/tool example short, add a loader in `llm_test_support::model_output`,
and add focused tokenizer/template/parser tests that consume the loader.
