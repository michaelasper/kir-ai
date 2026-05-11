use std::path::Path;

pub(crate) async fn read_optional_bytes(path: &Path) -> anyhow::Result<Option<Vec<u8>>> {
    match tokio::fs::read(path).await {
        Ok(bytes) => Ok(Some(bytes)),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(err.into()),
    }
}

pub(crate) async fn read_optional_string(path: &Path) -> anyhow::Result<Option<String>> {
    match tokio::fs::read_to_string(path).await {
        Ok(s) => Ok(Some(s)),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(err.into()),
    }
}
