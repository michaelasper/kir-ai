use super::MetalError;
use metal::{CommandBufferRef, MTLCommandBufferStatus};
use std::{
    sync::{Arc, Condvar, Mutex, MutexGuard},
    time::Duration,
};

const DEFAULT_COMMAND_BUFFER_TIMEOUT: Duration = Duration::from_secs(30);
const COMMAND_BUFFER_TIMEOUT_MS_ENV: &str = "LLM_ENGINE_METAL_COMMAND_BUFFER_TIMEOUT_MS";

pub(crate) async fn finish_command_buffer_async(
    synchronization: &Arc<MetalSynchronization>,
    command_buffer: &CommandBufferRef,
    kernel_name: &str,
) -> Result<(), MetalError> {
    let rx = {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let tx = std::sync::Mutex::new(Some(tx));
        let in_flight = std::sync::Mutex::new(Some(synchronization.begin_command()));
        let block = block::ConcreteBlock::new(move |cb: &CommandBufferRef| {
            if let Ok(mut guard) = tx.lock()
                && let Some(tx) = guard.take()
            {
                let _ = tx.send(cb.status());
            }
            if let Ok(mut guard) = in_flight.lock() {
                guard.take();
            }
        })
        .copy();
        command_buffer.add_completed_handler(&block);
        command_buffer.commit();
        rx
    };
    command_buffer_completion_result(rx, kernel_name, command_buffer_timeout()).await
}

#[derive(Debug, Default)]
pub(crate) struct MetalSynchronization {
    state: Mutex<MetalSynchronizationState>,
    idle: Condvar,
}

impl MetalSynchronization {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn begin_command(self: &Arc<Self>) -> MetalCommandInFlight {
        let mut state = self.lock_state();
        while state.cpu_accessing || state.pending_cpu_access > 0 {
            state = self.wait(state);
        }
        state.in_flight_commands += 1;
        MetalCommandInFlight {
            synchronization: Arc::clone(self),
        }
    }

    pub(crate) fn begin_cpu_access(self: &Arc<Self>) -> MetalCpuAccessGuard {
        let mut state = self.lock_state();
        // StorageModeShared buffers are visible to both the CPU and GPU. Treat
        // command submission and CPU copies as one device-wide critical section
        // so no command can start while a CPU copy is active, and CPU copies wait
        // for already-submitted commands to finish.
        state.pending_cpu_access += 1;
        while state.cpu_accessing || state.in_flight_commands > 0 {
            state = self.wait(state);
        }
        state.pending_cpu_access -= 1;
        state.cpu_accessing = true;
        MetalCpuAccessGuard {
            synchronization: Arc::clone(self),
        }
    }

    fn finish_command(&self) {
        let mut state = self.lock_state();
        state.in_flight_commands = state.in_flight_commands.saturating_sub(1);
        if state.in_flight_commands == 0 {
            self.idle.notify_all();
        }
    }

    fn finish_cpu_access(&self) {
        let mut state = self.lock_state();
        state.cpu_accessing = false;
        self.idle.notify_all();
    }

    fn lock_state(&self) -> MutexGuard<'_, MetalSynchronizationState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn wait<'a>(
        &self,
        state: MutexGuard<'a, MetalSynchronizationState>,
    ) -> MutexGuard<'a, MetalSynchronizationState> {
        self.idle
            .wait(state)
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[derive(Debug, Default)]
struct MetalSynchronizationState {
    in_flight_commands: usize,
    pending_cpu_access: usize,
    cpu_accessing: bool,
}

#[derive(Debug)]
pub(crate) struct MetalCommandInFlight {
    synchronization: Arc<MetalSynchronization>,
}

impl Drop for MetalCommandInFlight {
    fn drop(&mut self) {
        self.synchronization.finish_command();
    }
}

#[derive(Debug)]
pub(crate) struct MetalCpuAccessGuard {
    synchronization: Arc<MetalSynchronization>,
}

impl Drop for MetalCpuAccessGuard {
    fn drop(&mut self) {
        self.synchronization.finish_cpu_access();
    }
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

async fn command_buffer_completion_result(
    rx: tokio::sync::oneshot::Receiver<MTLCommandBufferStatus>,
    kernel_name: &str,
    timeout: Duration,
) -> Result<(), MetalError> {
    let status = tokio::time::timeout(timeout, rx)
        .await
        .map_err(|_| {
            MetalError::Execution(format!(
                "{kernel_name} command buffer timed out after {timeout:?}"
            ))
        })?
        .unwrap_or(MTLCommandBufferStatus::Error);
    command_buffer_status_result(status, kernel_name)
}

fn command_buffer_timeout() -> Duration {
    std::env::var(COMMAND_BUFFER_TIMEOUT_MS_ENV)
        .ok()
        .and_then(|value| parse_command_buffer_timeout_ms(&value))
        .unwrap_or(DEFAULT_COMMAND_BUFFER_TIMEOUT)
}

fn parse_command_buffer_timeout_ms(value: &str) -> Option<Duration> {
    let millis = value.trim().parse::<u64>().ok()?;
    (millis > 0).then(|| Duration::from_millis(millis))
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal::MTLCommandBufferStatus;
    use std::sync::mpsc;
    use std::time::Duration;

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

    #[tokio::test]
    async fn command_buffer_completion_times_out_when_handler_never_reports_status() {
        let (_tx, rx) = tokio::sync::oneshot::channel();

        let err = command_buffer_completion_result(rx, "hung_kernel", Duration::from_millis(1))
            .await
            .expect_err("missing completion status should time out");

        assert!(matches!(err, MetalError::Execution(_)));
        assert!(err.to_string().contains("hung_kernel"));
        assert!(err.to_string().contains("timed out"));
    }

    #[test]
    fn command_buffer_timeout_parser_accepts_positive_milliseconds() {
        assert_eq!(
            parse_command_buffer_timeout_ms("250"),
            Some(Duration::from_millis(250))
        );
    }

    #[test]
    fn command_buffer_timeout_parser_rejects_zero_or_invalid_values() {
        assert_eq!(parse_command_buffer_timeout_ms("0"), None);
        assert_eq!(parse_command_buffer_timeout_ms("not-a-number"), None);
    }

    #[test]
    fn cpu_access_waits_for_in_flight_command() {
        let synchronization = Arc::new(MetalSynchronization::new());
        let in_flight = synchronization.begin_command();
        let (tx, rx) = mpsc::channel();
        let worker_sync = Arc::clone(&synchronization);

        let worker = std::thread::spawn(move || {
            let _guard = worker_sync.begin_cpu_access();
            tx.send(()).expect("signal sends");
        });

        assert!(rx.recv_timeout(Duration::from_millis(20)).is_err());
        drop(in_flight);
        rx.recv_timeout(Duration::from_secs(1))
            .expect("cpu access starts after command finishes");
        worker.join().expect("worker joins");
    }

    #[test]
    fn command_submission_waits_for_cpu_access() {
        let synchronization = Arc::new(MetalSynchronization::new());
        let cpu_access = synchronization.begin_cpu_access();
        let (tx, rx) = mpsc::channel();
        let worker_sync = Arc::clone(&synchronization);

        let worker = std::thread::spawn(move || {
            let _in_flight = worker_sync.begin_command();
            tx.send(()).expect("signal sends");
        });

        assert!(rx.recv_timeout(Duration::from_millis(20)).is_err());
        drop(cpu_access);
        rx.recv_timeout(Duration::from_secs(1))
            .expect("command starts after cpu access finishes");
        worker.join().expect("worker joins");
    }
}
