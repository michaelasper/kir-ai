use futures::stream::{BoxStream, StreamExt};
use llm_backend::{BackendError, BackendStreamChunk};

#[derive(Default)]
pub(crate) struct NativeStreamTextDeltas {
    emitted: String,
    pending: Option<String>,
}

impl NativeStreamTextDeltas {
    pub(crate) fn observe(&mut self, decoded: String) -> Result<Option<String>, BackendError> {
        if !decoded.starts_with(&self.emitted) {
            return Err(non_prefix_stream_error());
        }
        let Some(pending) = self.pending.take() else {
            self.pending = Some(decoded);
            return Ok(None);
        };
        let delta = if pending.starts_with(&self.emitted) && decoded.starts_with(&pending) {
            let delta = pending[self.emitted.len()..].to_owned();
            self.emitted = pending;
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
}

fn non_empty(value: String) -> Option<String> {
    (!value.is_empty()).then_some(value)
}

fn non_prefix_stream_error() -> BackendError {
    BackendError::Other(
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
                                    yield Err(BackendError::Other(format!(
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
                        yield Err(BackendError::Other(format!(
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
