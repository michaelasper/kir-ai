use axum::body::{Body, Bytes};
use futures::StreamExt;
use std::{error::Error, fmt, time::Duration};
use tokio::time::Instant as TokioInstant;

type BoxBodyError = Box<dyn Error + Send + Sync>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RequestBodyTimeoutError {
    timeout: Duration,
}

impl RequestBodyTimeoutError {
    pub(super) fn new(timeout: Duration) -> Self {
        Self { timeout }
    }

    pub(super) fn timeout(self) -> Duration {
        self.timeout
    }
}

impl fmt::Display for RequestBodyTimeoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "request body exceeded timeout of {} ms",
            self.timeout.as_millis()
        )
    }
}

impl Error for RequestBodyTimeoutError {}

pub(super) fn with_request_body_timeout(body: Body, timeout: Duration) -> Body {
    let deadline = TokioInstant::now() + timeout;
    let mut body = body.into_data_stream();
    let stream = async_stream::stream! {
        loop {
            match tokio::time::timeout_at(deadline, body.next()).await {
                Ok(Some(Ok(bytes))) => yield Ok::<Bytes, BoxBodyError>(bytes),
                Ok(Some(Err(err))) => {
                    yield Err::<Bytes, BoxBodyError>(Box::new(err));
                    break;
                }
                Ok(None) => break,
                Err(_) => {
                    yield Err::<Bytes, BoxBodyError>(Box::new(RequestBodyTimeoutError::new(timeout)));
                    break;
                }
            }
        }
    };
    Body::from_stream(stream)
}

pub(super) fn request_body_timeout(err: &(dyn Error + 'static)) -> Option<Duration> {
    let mut current = Some(err);
    while let Some(err) = current {
        if let Some(timeout) = err.downcast_ref::<RequestBodyTimeoutError>() {
            return Some(timeout.timeout());
        }
        current = err.source();
    }
    None
}
