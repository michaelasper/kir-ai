use async_trait::async_trait;
use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use futures::StreamExt;
use llm_backend::ProtocolTestBackend;
use llm_backend_contracts::{
    BackendCapabilities, BackendError, BackendFinishReason, BackendHealth, BackendModelMetadata,
    BackendOutput, BackendRequest, BackendStreamChunk, BackendStreamProgress, BackendToolCallDelta,
    BackendToolCallFunctionDelta, BackendToolCallType, ModelBackend,
};
use llm_engine::{EngineOptions, PublicInferenceRateLimit, build_router, router_builder};
use llm_hub::{HubFile, HubRepoId, ModelProfile, ModelStore, build_download_plan};
use serde_json::{Value, json};
use std::{
    io::{Read, Write},
    net::TcpListener,
    path::{Path, PathBuf},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::Duration,
};
use tokio::sync::{Notify, Semaphore, mpsc};
use tokio_util::sync::CancellationToken;
use tower::ServiceExt;
