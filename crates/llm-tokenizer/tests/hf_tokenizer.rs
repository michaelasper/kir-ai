use llm_tokenizer::HuggingFaceTokenizer;
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};
use tracing::field::{Field, Visit};
use tracing::{Event, Id, Metadata, Subscriber, span};

const TOKENIZER_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../fixtures/qwen36/tokenizer.json"
);

#[test]
fn official_qwen36_tokenizer_round_trips_text() {
    let tokenizer =
        HuggingFaceTokenizer::from_file(TOKENIZER_PATH).expect("official tokenizer loads");

    let ids = tokenizer
        .encode("hello rust tokenizer", false)
        .expect("text encodes");
    assert!(!ids.is_empty());

    let decoded = tokenizer.decode(&ids, false).expect("text decodes");
    assert_eq!(decoded, "hello rust tokenizer");
}

#[test]
fn official_qwen36_tokenizer_uses_regex_pretokenizer_covered_by_onig() {
    let tokenizer_json: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/qwen36/tokenizer.json"
    )))
    .expect("official tokenizer fixture is JSON");
    let regex_pattern = tokenizer_json
        .pointer("/pre_tokenizer/pretokenizers/0/pattern/Regex")
        .and_then(serde_json::Value::as_str)
        .expect("official Qwen tokenizer uses a regex split pre-tokenizer");

    assert!(regex_pattern.contains("\\p{L}"));
    HuggingFaceTokenizer::from_file(TOKENIZER_PATH)
        .expect("tokenizers/onig loads the supported regex pre-tokenizer");
}

#[test]
fn tokenizer_decode_emits_trace_metadata() {
    let tokenizer =
        HuggingFaceTokenizer::from_file(TOKENIZER_PATH).expect("official tokenizer loads");
    let ids = tokenizer
        .encode("hello rust tokenizer", false)
        .expect("text encodes");

    let capture = TraceCapture::start();
    let decoded = tokenizer.decode(&ids, false).expect("text decodes");
    let events = capture.events();

    assert_eq!(decoded, "hello rust tokenizer");
    assert!(
        events.iter().any(|event| {
            event.has_field("operation", "decode")
                && event.has_field("token_count", &ids.len().to_string())
                && event.has_field("skip_special_tokens", "false")
        }),
        "decode should emit structured trace metadata, got {events:?}"
    );
}

#[test]
fn official_qwen36_tokenizer_preserves_chatml_special_tokens() {
    let tokenizer =
        HuggingFaceTokenizer::from_file(TOKENIZER_PATH).expect("official tokenizer loads");

    assert_eq!(tokenizer.token_to_id("<|im_start|>"), Some(248_045));
    assert_eq!(tokenizer.token_to_id("<|im_end|>"), Some(248_046));

    let ids = tokenizer
        .encode("<|im_start|>user\nhi<|im_end|>\n", false)
        .expect("chatml encodes");
    assert!(ids.contains(&248_045));
    assert!(ids.contains(&248_046));
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
