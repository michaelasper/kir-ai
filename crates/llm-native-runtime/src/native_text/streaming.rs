use futures::stream::{BoxStream, StreamExt};
use llm_backend_contracts::{BackendError, BackendStreamChunk};
use llm_tokenizer::{HuggingFaceDecodeStream, HuggingFaceTokenizer};

pub(crate) trait NativeTextStreamDecoder {
    fn step(&mut self, token_id: u32) -> Result<Option<String>, BackendError>;
}

pub(crate) struct NativeTokenizerStreamDecoder<'tokenizer> {
    inner: HuggingFaceDecodeStream<'tokenizer>,
}

impl<'tokenizer> NativeTokenizerStreamDecoder<'tokenizer> {
    pub(crate) fn new(tokenizer: &'tokenizer HuggingFaceTokenizer) -> Self {
        Self {
            inner: tokenizer.decode_stream(false),
        }
    }
}

impl NativeTextStreamDecoder for NativeTokenizerStreamDecoder<'_> {
    fn step(&mut self, token_id: u32) -> Result<Option<String>, BackendError> {
        self.inner
            .step(token_id)
            .map_err(|err| BackendError::other(err.to_string()))
    }
}

#[cfg(test)]
#[derive(Default)]
pub(crate) struct NativeStreamTextDeltas {
    emitted: String,
    pending: Option<String>,
}

#[cfg(test)]
impl NativeStreamTextDeltas {
    pub(crate) fn observe(&mut self, decoded: String) -> Result<Option<String>, BackendError> {
        if !decoded.starts_with(&self.emitted) {
            return Err(non_prefix_stream_error());
        }
        let Some(pending) = self.pending.take() else {
            self.pending = Some(decoded);
            return Ok(None);
        };
        if !pending.starts_with(&self.emitted) {
            return Err(non_prefix_stream_error());
        }
        let stable_len = common_prefix_len(&pending, &decoded);
        let delta = if stable_len > self.emitted.len() {
            let stable = pending[..stable_len].to_owned();
            let delta = stable[self.emitted.len()..].to_owned();
            self.emitted = stable;
            non_empty(delta)
        } else {
            None
        };
        self.pending = Some(decoded);
        Ok(delta)
    }

    pub(crate) fn finish(&mut self, decoded: String) -> Result<Option<String>, BackendError> {
        self.pending = None;
        if !decoded.starts_with(&self.emitted) {
            return Err(non_prefix_stream_error());
        }
        let delta = decoded[self.emitted.len()..].to_owned();
        self.emitted = decoded;
        Ok(non_empty(delta))
    }

    pub(crate) fn observe_incremental(&mut self, decoded_piece: String) -> Option<String> {
        let delta = self.pending.replace(decoded_piece).and_then(non_empty);
        if let Some(delta) = &delta {
            self.emitted.push_str(delta);
        }
        delta
    }

    pub(crate) fn finish_incremental(&mut self) -> Option<String> {
        let delta = self.pending.take().and_then(non_empty);
        if let Some(delta) = &delta {
            self.emitted.push_str(delta);
        }
        delta
    }
}

#[cfg(test)]
fn non_empty(value: String) -> Option<String> {
    (!value.is_empty()).then_some(value)
}

#[cfg(test)]
fn common_prefix_len(left: &str, right: &str) -> usize {
    let mut len = 0;
    let mut right_chars = right.chars();
    for (index, left_char) in left.char_indices() {
        let Some(right_char) = right_chars.next() else {
            break;
        };
        if left_char != right_char {
            break;
        }
        len = index + left_char.len_utf8();
    }
    len
}

#[cfg(test)]
fn non_prefix_stream_error() -> BackendError {
    BackendError::other(
        "native tokenizer streaming decode became non-prefix after emitted delta".to_owned(),
    )
}

pub(crate) fn native_text_worker_stream(
    label: &'static str,
    rx: tokio::sync::mpsc::Receiver<Result<BackendStreamChunk, BackendError>>,
    worker: tokio::task::JoinHandle<()>,
) -> BoxStream<'static, Result<BackendStreamChunk, BackendError>> {
    async_stream::stream! {
        let mut rx = rx;
        let mut worker = Some(worker);
        loop {
            let Some(handle) = worker.as_mut() else {
                match rx.recv().await {
                    Some(item) => {
                        yield item;
                        continue;
                    }
                    None => break,
                }
            };
            tokio::select! {
                item = rx.recv() => {
                    match item {
                        Some(item) => yield item,
                        None => {
                            if let Some(handle) = worker.take() {
                                let result = handle.await;
                                if let Err(err) = result {
                                    yield Err(BackendError::other(format!(
                                        "{label} streaming worker failed: {err}"
                                    )));
                                }
                            }
                            break;
                        }
                    }
                }
                result = handle => {
                    worker = None;
                    if let Err(err) = result {
                        yield Err(BackendError::other(format!(
                            "{label} streaming worker failed: {err}"
                        )));
                        break;
                    }
                }
            }
        }
    }
    .boxed()
}
