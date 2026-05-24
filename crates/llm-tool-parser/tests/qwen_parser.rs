use llm_tool_parser::{QwenParser, ToolParserFamily, parse_assistant_for_parser_family};
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};
use tracing::field::{Field, Visit};
use tracing::{Event, Id, Metadata, Subscriber, span};

#[test]
fn parses_reasoning_content_and_hermes_tool_call() {
    let parsed = QwenParser
        .parse_complete(
            "<think>Need a file read.</think><tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"Cargo.toml\"}}</tool_call>",
        )
        .expect("tool call parses");

    assert_eq!(parsed.reasoning.as_deref(), Some("Need a file read."));
    assert_eq!(parsed.content, "");
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.tool_calls[0].function.name, "read_file");
    assert_eq!(
        parsed.tool_calls[0].function.arguments["path"],
        "Cargo.toml"
    );
}

#[test]
fn parser_family_entrypoint_emits_trace_metadata() {
    let capture = TraceCapture::start();
    let parsed = parse_assistant_for_parser_family(
        ToolParserFamily::Qwen,
        "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"Cargo.toml\"}}</tool_call>",
    )
    .expect("tool call parses");
    let events = capture.events();

    assert_eq!(parsed.tool_calls.len(), 1);
    assert!(
        events.iter().any(|event| {
            event.has_field("operation", "parse_assistant")
                && event.has_field("parser_family", "Qwen")
                && event.has_field("tool_call_count", "1")
        }),
        "parser should emit structured trace metadata, got {events:?}"
    );
}

#[test]
fn parses_parameters_alias_in_hermes_tool_call() {
    let parsed = QwenParser
        .parse_complete(
            "<tool_call>{\"name\":\"read_file\",\"parameters\":{\"path\":\"Cargo.toml\",\"_i\":0}}</tool_call>",
        )
        .expect("parameters alias parses");

    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.tool_calls[0].function.name, "read_file");
    assert_eq!(
        parsed.tool_calls[0].function.arguments,
        serde_json::json!({"path": "Cargo.toml", "_i": 0})
    );
}

#[test]
fn parses_qwen_coder_xml_tool_call() {
    let parsed = QwenParser
        .parse_complete(
            "<tool_call><function=bash><parameter=cmd>cargo test --workspace</parameter></function></tool_call>",
        )
        .expect("xml tool call parses");

    assert_eq!(parsed.reasoning, None);
    assert_eq!(parsed.content, "");
    assert_eq!(parsed.tool_calls.len(), 1);
    assert!(parsed.tool_calls[0].id.starts_with("call_"));
    assert!(
        !parsed.tool_calls[0].id["call_".len()..]
            .chars()
            .all(|character| character.is_ascii_digit())
    );
    assert_eq!(parsed.tool_calls[0].function.name, "bash");
    assert_eq!(
        parsed.tool_calls[0].function.arguments,
        serde_json::json!({"cmd": "cargo test --workspace"})
    );
}

#[test]
fn preserves_plain_assistant_whitespace() {
    let parsed = QwenParser
        .parse_complete("  keep leading space\n    indented line\n")
        .expect("plain text parses");

    assert_eq!(parsed.content, "  keep leading space\n    indented line\n");
    assert!(parsed.tool_calls.is_empty());
}

#[test]
fn preserves_content_around_reasoning_and_tool_tags() {
    let parsed = QwenParser
        .parse_complete(
            "  before\n<think>private chain</think>\ninside\n<tool_call>{\"name\":\"lookup\",\"arguments\":{\"query\":\"rust\"}}</tool_call>\n  after\n",
        )
        .expect("tagged output parses");

    assert_eq!(parsed.reasoning.as_deref(), Some("private chain"));
    assert_eq!(parsed.content, "  before\n\ninside\n\n  after\n");
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.tool_calls[0].function.name, "lookup");
}

#[test]
fn parses_truncated_reasoning_as_partial_reasoning() {
    let parsed = QwenParser
        .parse_complete("visible prefix\n<think>Need more tokens")
        .expect("truncated reasoning parses as partial output");

    assert_eq!(parsed.reasoning.as_deref(), Some("Need more tokens"));
    assert_eq!(parsed.content, "visible prefix\n");
    assert!(parsed.tool_calls.is_empty());
}

#[test]
fn fails_when_tool_markup_is_malformed() {
    let err = QwenParser
        .parse_complete("<tool_call>{\"name\":\"read_file\",\"arguments\":")
        .expect_err("malformed tool call fails");

    assert_eq!(err.code(), "malformed_tool_call");
}

#[derive(Clone, Debug)]
struct RecordedEvent {
    fields: Vec<(String, String)>,
}

impl RecordedEvent {
    fn has_field(&self, name: &str, value: &str) -> bool {
        self.fields
            .iter()
            .any(|(field, recorded)| field == name && recorded == value)
    }
}

static TRACE_EVENTS: OnceLock<Arc<Mutex<Vec<RecordedEvent>>>> = OnceLock::new();

struct TraceCapture {
    events: Arc<Mutex<Vec<RecordedEvent>>>,
}

impl TraceCapture {
    fn start() -> Self {
        let events = Arc::clone(TRACE_EVENTS.get_or_init(|| {
            let events = Arc::new(Mutex::new(Vec::new()));
            let subscriber = RecordingSubscriber {
                events: Arc::clone(&events),
            };
            tracing::subscriber::set_global_default(subscriber)
                .expect("trace test subscriber installs once");
            events
        }));
        events.lock().expect("recorded events lock").clear();
        tracing::callsite::rebuild_interest_cache();
        Self { events }
    }

    fn events(&self) -> Vec<RecordedEvent> {
        self.events.lock().expect("recorded events lock").clone()
    }
}

struct RecordingSubscriber {
    events: Arc<Mutex<Vec<RecordedEvent>>>,
}

impl Subscriber for RecordingSubscriber {
    fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
        true
    }

    fn register_callsite(
        &self,
        _metadata: &'static Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        tracing::subscriber::Interest::always()
    }

    fn max_level_hint(&self) -> Option<tracing::level_filters::LevelFilter> {
        Some(tracing::level_filters::LevelFilter::TRACE)
    }

    fn new_span(&self, _span: &span::Attributes<'_>) -> Id {
        Id::from_u64(1)
    }

    fn record(&self, _span: &Id, _values: &span::Record<'_>) {}

    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

    fn event(&self, event: &Event<'_>) {
        let mut visitor = FieldRecorder::default();
        event.record(&mut visitor);
        self.events
            .lock()
            .expect("recorded events lock")
            .push(RecordedEvent {
                fields: visitor.fields,
            });
    }

    fn enter(&self, _span: &Id) {}

    fn exit(&self, _span: &Id) {}
}

#[derive(Default)]
struct FieldRecorder {
    fields: Vec<(String, String)>,
}

impl FieldRecorder {
    fn record_value(&mut self, field: &Field, value: String) {
        self.fields.push((field.name().to_owned(), value));
    }
}

impl Visit for FieldRecorder {
    fn record_bool(&mut self, field: &Field, value: bool) {
        self.record_value(field, value.to_string());
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.record_value(field, value.to_string());
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.record_value(field, value.to_string());
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.record_value(field, value.to_owned());
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.record_value(field, format!("{value:?}"));
    }
}
