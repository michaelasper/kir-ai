# Issue #179: Structured Chat Context Dropped When Tool Messages Present

> Superseded by the Issue #215 lossless chat context fix. The current backend
> contract stores `llm_api::ChatMessage` directly in `BackendChatContext` and
> forwards the structured OpenAI history to MLX `/v1/chat/completions`,
> including assistant `tool_calls`, `tool` role messages, `tool_call_id`, and
> optional `name` fields. Rendered prompts remain cache/fallback context, not
> the source of truth for MLX chat requests when structured `chat_context` is
> present.

## Problem

`backend_chat_message()` in `crates/llm-runtime/src/adapters.rs:160-174` returns `None` for:
1. Assistant messages with `tool_calls` (line 161-163)
2. Messages with `ChatRole::Tool` (line 168)

`backend_chat_context()` at line 126-129 uses `.map(backend_chat_message).collect::<Option<Vec<_>>>()` which short-circuits on the first `None`. Any tool or tool-call message poisons the entire collection, causing `backend_chat_context()` to return `None`.

When `BackendRequest.chat_context` is `None`, the MLX sidecar's protocol selector (`protocol.rs:70`) falls back from `ChatCompletions` to `Completions` for Llama models, and the request builder (`request.rs:92-106`) wraps the rendered prompt string in a single synthetic `user` message — discarding all structured multi-turn context.

The prompt string (rendered by the tokenizer) is unaffected and always contains the full conversation including tool turns.

## Approach: Filter + Observability

Filter tool/tool-call messages from the structured context instead of propagating `None`. Add `tracing::debug!` when filtering occurs.

### Rationale

- Minimal code change — only `backend_chat_context()` in `adapters.rs`
- No changes to `BackendChatMessage`, `BackendChatRole`, `MlxChatMessage`, or protocol layer
- The prompt string already carries the full conversation; `chat_context` is supplementary
- The filtered context (system/user/assistant turns) still provides valuable multi-turn structure to MLX sidecars

### Changes

#### `crates/llm-runtime/src/adapters.rs`

Replace the `collect::<Option<Vec<_>>>()` with `filter_map`:

```rust
fn backend_chat_context(
    self,
    messages: &[ChatMessage],
    _tools: &[ToolDefinition],
) -> Option<BackendChatContext> {
    let total = messages.len();
    let messages: Vec<_> = messages
        .iter()
        .filter_map(backend_chat_message)
        .collect();
    let filtered = total - messages.len();
    if filtered > 0 {
        tracing::debug!(
            filtered_count = filtered,
            remaining_count = messages.len(),
            "filtered tool/tool-call messages from structured chat context"
        );
    }
    if messages.is_empty() {
        return None;
    }
    Some(BackendChatContext { messages })
}
```

Key points:
- `filter_map` drops `None` results (tool/tool-call messages) instead of poisoning the collection
- Empty guard prevents sending an empty context
- `backend_chat_message()` is unchanged — it correctly returns `None` for tool/tool-call messages
- `tracing::debug!` gives operators visibility without noise

#### `crates/llm-runtime/tests/runtime_contract/tool_validation.rs`

Add a test that verifies a conversation with tool messages produces a filtered (non-`None`) chat context with only system/user/assistant messages retained.

### What Stays the Same

- `backend_chat_message()` — unchanged
- `BackendChatMessage`, `BackendChatRole` — unchanged
- `MlxChatMessage`, `request.rs`, `protocol.rs` — unchanged
- Prompt rendering path (`render_prompt`) — unchanged
- All existing tests — pass without modification

## Success Criteria

- Conversations with tool messages produce a non-`None` `chat_context` containing all system/user/assistant messages
- Conversations without tool messages produce identical results to the current behaviour
- `tracing::debug!` fires when filtering occurs
- All existing contract tests pass unchanged
