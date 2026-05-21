mod engine;
mod sync_ext;

pub use axum::Router as ServerRouter;
use axum::serve::Listener;
pub use engine::*;
pub use llm_util::defaults::DEFAULT_MODEL_ID;
use serde_json::Value;
use std::{
    collections::HashMap,
    error::Error,
    fmt, io,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use tokio_rustls::{
    TlsAcceptor,
    rustls::{
        ServerConfig,
        pki_types::{CertificateDer, PrivateKeyDer, pem::PemObject},
    },
};

#[derive(Clone, Debug, Default)]
pub struct ServerBackendMetricsSnapshot {
    pub metrics: HashMap<String, Value>,
}

pub trait ServerBackendMetrics: Send + Sync {
    fn snapshot(&self) -> ServerBackendMetricsSnapshot;
}

#[derive(Debug, Default)]
pub struct NoopServerBackendMetrics;

impl ServerBackendMetrics for NoopServerBackendMetrics {
    fn snapshot(&self) -> ServerBackendMetricsSnapshot {
        ServerBackendMetricsSnapshot::default()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TlsConfig {
    cert_path: PathBuf,
    key_path: PathBuf,
}

impl TlsConfig {
    pub fn new(cert_path: impl Into<PathBuf>, key_path: impl Into<PathBuf>) -> Self {
        Self {
            cert_path: cert_path.into(),
            key_path: key_path.into(),
        }
    }

    pub fn cert_path(&self) -> &Path {
        &self.cert_path
    }

    pub fn key_path(&self) -> &Path {
        &self.key_path
    }
}

#[derive(Debug)]
pub struct TlsConfigError {
    message: String,
}

impl TlsConfigError {
    fn read(kind: &str, path: &Path, source: io::Error) -> Self {
        Self {
            message: format!(
                "TLS {kind} file `{}` could not be read: {source}",
                path.display()
            ),
        }
    }

    fn invalid(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for TlsConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for TlsConfigError {}

async fn load_tls_server_config(config: &TlsConfig) -> Result<Arc<ServerConfig>, TlsConfigError> {
    let cert_bytes = tokio::fs::read(config.cert_path())
        .await
        .map_err(|err| TlsConfigError::read("certificate", config.cert_path(), err))?;
    let key_bytes = tokio::fs::read(config.key_path())
        .await
        .map_err(|err| TlsConfigError::read("private key", config.key_path(), err))?;

    let cert_chain = CertificateDer::pem_slice_iter(&cert_bytes)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| {
            TlsConfigError::invalid(format!(
                "TLS certificate file `{}` contains invalid PEM data: {err}",
                config.cert_path().display()
            ))
        })?;
    if cert_chain.is_empty() {
        return Err(TlsConfigError::invalid(format!(
            "TLS certificate file `{}` does not contain any CERTIFICATE PEM blocks",
            config.cert_path().display()
        )));
    }

    let private_key = PrivateKeyDer::from_pem_slice(&key_bytes).map_err(|err| {
        TlsConfigError::invalid(format!(
            "TLS private key file `{}` must contain one PRIVATE KEY, RSA PRIVATE KEY, or EC PRIVATE KEY PEM block: {err}",
            config.key_path().display()
        ))
    })?;

    let server_config = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(cert_chain, private_key)
        .map_err(|err| {
            TlsConfigError::invalid(format!(
                "TLS certificate/key configuration is invalid for certificate `{}` and key `{}`: {err}",
                config.cert_path().display(),
                config.key_path().display()
            ))
        })?;

    Ok(Arc::new(server_config))
}

struct TlsListener {
    listener: tokio::net::TcpListener,
    acceptor: TlsAcceptor,
}

impl Listener for TlsListener {
    type Io = tokio_rustls::server::TlsStream<tokio::net::TcpStream>;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            let (stream, addr) = match self.listener.accept().await {
                Ok(accepted) => accepted,
                Err(err) => {
                    handle_accept_error(err).await;
                    continue;
                }
            };

            match self.acceptor.accept(stream).await {
                Ok(tls_stream) => return (tls_stream, addr),
                Err(err) => {
                    tracing::warn!(remote_addr = %addr, error = %err, "TLS handshake failed");
                }
            }
        }
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        self.listener.local_addr()
    }
}

async fn handle_accept_error(err: io::Error) {
    if matches!(
        err.kind(),
        io::ErrorKind::ConnectionRefused
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::ConnectionReset
    ) {
        return;
    }

    tracing::error!(error = %err, "accept error");
    tokio::time::sleep(Duration::from_secs(1)).await;
}

pub async fn serve(listener: tokio::net::TcpListener, router: ServerRouter) -> io::Result<()> {
    axum::serve(listener, router).await
}

pub async fn serve_tls(
    listener: tokio::net::TcpListener,
    router: ServerRouter,
    tls_config: TlsConfig,
) -> io::Result<()> {
    let server_config = load_tls_server_config(&tls_config)
        .await
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;
    let tls_listener = TlsListener {
        listener,
        acceptor: TlsAcceptor::from(server_config),
    };
    axum::serve(tls_listener, router).await
}
