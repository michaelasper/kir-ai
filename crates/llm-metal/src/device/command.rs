use super::MetalError;
use metal::{CommandBufferRef, MTLCommandBufferStatus};

pub(crate) async fn finish_command_buffer_async(
    command_buffer: &CommandBufferRef,
    kernel_name: &str,
) -> Result<(), MetalError> {
    let rx = {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let tx = std::sync::Mutex::new(Some(tx));
        let block = block::ConcreteBlock::new(move |cb: &CommandBufferRef| {
            if let Ok(mut guard) = tx.lock()
                && let Some(tx) = guard.take()
            {
                let _ = tx.send(cb.status());
            }
        })
        .copy();
        command_buffer.add_completed_handler(&block);
        command_buffer.commit();
        rx
    };
    let status = rx.await.unwrap_or(MTLCommandBufferStatus::Error);
    command_buffer_status_result(status, kernel_name)
}

pub(crate) fn command_buffer_status_result(
    status: MTLCommandBufferStatus,
    kernel_name: &str,
) -> Result<(), MetalError> {
    match status {
        MTLCommandBufferStatus::Completed => Ok(()),
        MTLCommandBufferStatus::Error => Err(MetalError::Execution(format!(
            "{kernel_name} command buffer failed with status {status:?}"
        ))),
        other => Err(MetalError::Execution(format!(
            "{kernel_name} command buffer finished with unexpected status {other:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal::MTLCommandBufferStatus;

    #[test]
    fn command_buffer_status_result_accepts_completed_status() {
        assert!(
            command_buffer_status_result(MTLCommandBufferStatus::Completed, "matvec_f32").is_ok()
        );
    }

    #[test]
    fn command_buffer_status_result_rejects_error_status() {
        let err = command_buffer_status_result(MTLCommandBufferStatus::Error, "softmax_f32")
            .expect_err("error status should fail");

        assert!(matches!(err, MetalError::Execution(_)));
        assert!(err.to_string().contains("softmax_f32"));
    }

    #[test]
    fn command_buffer_status_result_rejects_unfinished_status() {
        let err = command_buffer_status_result(MTLCommandBufferStatus::Scheduled, "top_k_f32")
            .expect_err("unfinished status should fail");

        assert!(matches!(err, MetalError::Execution(_)));
        assert!(err.to_string().contains("unexpected status"));
    }
}
