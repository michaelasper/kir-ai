# Structured Chat Context Filter Implementation Plan

> Superseded by the Issue #215 lossless chat context implementation. The
> runtime now preserves the full OpenAI message history in
> `BackendChatContext`, and MLX chat requests use that structured context
> directly instead of a filtered projection or rendered-prompt reconstruction.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `backend_chat_context()` to preserve system/user/assistant messages when tool/tool-call messages are present, instead of silently dropping the entire structured context.

**Architecture:** Replace the `collect::<Option<Vec<_>>>()` short-circuit with `filter_map` so tool and tool-call messages are filtered out individually while remaining messages are preserved. Add `tracing::debug!` for observability.

**Tech Stack:** Rust, tokio, tracing, llm-runtime crate

**Design spec:** `docs/superpowers/specs/2026-05-11-structured-chat-context-filter-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `crates/llm-runtime/src/adapters.rs` | Modify | Change `backend_chat_context()` to use `filter_map` instead of `map...collect<Option>` |
| `crates/llm-runtime/tests/runtime_contract/tool_validation.rs` | Modify | Add test verifying tool messages are filtered, context is preserved |

No new files. No changes to other crates.

---

### Task 1: Add `tracing` dependency to `llm-runtime`

**Files:**
- Modify: `crates/llm-runtime/Cargo.toml`

- [ ] **Step 1: Check if `tracing` is already a dependency**

Run: `grep 'tracing' crates/llm-runtime/Cargo.toml`

If `tracing` is already present, skip to Task 2.

- [ ] **Step 2: Add `tracing` dependency**

Add `tracing` to the `[dependencies]` section of `crates/llm-runtime/Cargo.toml`.

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p llm-runtime`
Expected: Compiles without errors

- [ ] **Step 4: Commit**

```bash
git add crates/llm-runtime/Cargo.toml
git commit -m "chore: add tracing dependency to llm-runtime"
```

---

### Task 2: Write failing test for filtered tool messages

**Files:**
- Modify: `crates/llm-runtime/tests/runtime_contract/tool_validation.rs`

- [ ] **Step 1: Write the failing test**

Add this test at the end of `crates/llm-runtime/tests/runtime_contract/tool_validation.rs`:

```rust
#[tokio::test]
async fn runtime_preserves_chat_context_when_tool_messages_are_present() {
    let observed = Arc::new(Mutex::new(None));
    let runtime = Runtime::new(RecordingChatContextBackend {
        observed: observed.clone(),
        family: "gemma",
    });

    runtime
        .chat(ChatCompletionRequest {
            model: "local-gemma4".to_owned(),
            messages: vec![
                ChatMessage::system("You are a helpful assistant."),
                ChatMessage::user("lookup rust"),
                ChatMessage::assistant_tool_call(
                    "call_1",
                    "lookup",
                    json!({"query": "rust"}),
                ),
                ChatMessage::tool("call_1", "Rust is a systems programming language."),
                ChatMessage::user("tell me more"),
            ],
            tools: vec![ToolDefinition::function("lookup", "lookup", json!({}))],
            max_tokens: Some(16),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("Gemma chat with tool messages succeeds");

    let observed = observed
        .lock()
        .expect("observed request lock")
        .clone()
        .expect("backend request captured");
    let chat_context = observed
        .chat_context
        .expect("structured chat context must be present even when tool messages exist");
    assert_eq!(
        chat_context.messages.len(),
        3,
        "should have system, first user, and second user messages (tool messages filtered)"
    );
    assert_eq!(chat_context.messages[0].role, BackendChatRole::System);
    assert_eq!(
        chat_context.messages[0].content, "You are a helpful assistant."
    );
    assert_eq!(chat_context.messages[1].role, BackendChatRole::User);
    assert_eq!(chat_context.messages[1].content, "lookup rust");
    assert_eq!(chat_context.messages[2].role, BackendChatRole::User);
    assert_eq!(chat_context.messages[2].content, "tell me more");
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p llm-runtime --test runtime_contract runtime_preserves_chat_context_when_tool_messages_are_present -- --nocapture`
Expected: FAIL — `chat_context` will be `None` because the current `collect::<Option<Vec<_>>>()` short-circuits on the tool-call message.

- [ ] **Step 3: Commit the failing test**

```bash
git add crates/llm-runtime/tests/runtime_contract/tool_validation.rs
git commit -m "test: add failing test for tool message chat context preservation"
```

---

### Task 3: Fix `backend_chat_context()` to filter instead of propagate

**Files:**
- Modify: `crates/llm-runtime/src/adapters.rs:121-131`

- [ ] **Step 1: Replace `backend_chat_context()` implementation**

Replace the `backend_chat_context` method body in the `ChatAdapter for SelectedChatAdapter` impl block (lines 121-131) with:

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

- [ ] **Step 2: Run the failing test to verify it passes**

Run: `cargo test -p llm-runtime --test runtime_contract runtime_preserves_chat_context_when_tool_messages_are_present -- --nocapture`
Expected: PASS

- [ ] **Step 3: Run all existing contract tests to verify no regressions**

Run: `cargo test -p llm-runtime --test runtime_contract`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add crates/llm-runtime/src/adapters.rs
git commit -m "fix: filter tool messages instead of dropping entire chat context

backend_chat_context() now uses filter_map instead of
collect::<Option<Vec<_>>>(), preserving system/user/assistant
messages when tool or tool-call messages are present."
```

---

### Task 4: Verify full test suite and lint

- [ ] **Step 1: Run clippy**

Run: `cargo clippy -p llm-runtime --all-features`
Expected: No warnings or errors

- [ ] **Step 2: Run full test suite**

Run: `cargo test -p llm-runtime --all-features`
Expected: All tests PASS

- [ ] **Step 3: Run formatting check**

Run: `cargo fmt -p llm-runtime -- --check`
Expected: No formatting changes needed
