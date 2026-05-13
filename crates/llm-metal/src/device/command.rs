use super::MetalError;
use metal::{CommandBufferRef, MTLCommandBufferStatus};
use std::{
    sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock},
    time::Duration,
};

const DEFAULT_COMMAND_BUFFER_TIMEOUT: Duration = Duration::from_secs(30);
const COMMAND_BUFFER_TIMEOUT_MS_ENV: &str = "LLM_ENGINE_METAL_COMMAND_BUFFER_TIMEOUT_MS";
static COMMAND_BUFFER_TIMEOUT: CommandBufferTimeout = CommandBufferTimeout::new();

pub(crate) async fn finish_command_buffer_async(
    synchronization: &Arc<MetalSynchronization>,
    command_buffer: &CommandBufferRef,
    kernel_name: &str,
) -> Result<(), MetalError> {
    let rx = {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let tx = std::sync::Mutex::new(Some(tx));
        let in_flight = std::sync::Mutex::new(Some(synchronization.begin_command()));
        let kernel_name = kernel_name.to_owned();
        let block = block::ConcreteBlock::new(move |cb: &CommandBufferRef| {
            report_command_buffer_status(&tx, cb.status(), &kernel_name);
            finish_in_flight_command(&in_flight, &kernel_name);
        })
        .copy();
        command_buffer.add_completed_handler(&block);
        command_buffer.commit();
        rx
    };
    command_buffer_completion_result(rx, kernel_name, command_buffer_timeout()).await
}

fn report_command_buffer_status(
    tx: &Mutex<Option<tokio::sync::oneshot::Sender<MTLCommandBufferStatus>>>,
    status: MTLCommandBufferStatus,
    kernel_name: &str,
) {
    if !matches!(status, MTLCommandBufferStatus::Completed) {
        tracing::warn!(
            kernel = %kernel_name,
            ?status,
            "Metal command buffer completed with non-success status"
        );
    }

    let mut guard = match tx.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::warn!(
                kernel = %kernel_name,
                "Metal command buffer status sender lock was poisoned; recovering status sender"
            );
            poisoned.into_inner()
        }
    };
    let Some(tx) = guard.take() else {
        tracing::warn!(
            kernel = %kernel_name,
            ?status,
            "Metal command buffer status was reported after sender was already consumed"
        );
        return;
    };
    if let Err(status) = tx.send(status) {
        tracing::warn!(
            kernel = %kernel_name,
            ?status,
            "Metal command buffer status receiver was dropped before status could be reported"
        );
    }
}

fn finish_in_flight_command(in_flight: &Mutex<Option<MetalCommandInFlight>>, kernel_name: &str) {
    let mut guard = match in_flight.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::warn!(
                kernel = %kernel_name,
                "Metal command in-flight lock was poisoned; recovering synchronization guard"
            );
            poisoned.into_inner()
        }
    };
    if guard.take().is_none() {
        tracing::warn!(
            kernel = %kernel_name,
            "Metal command in-flight guard was already released before completion callback"
        );
    }
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
    COMMAND_BUFFER_TIMEOUT.get()
}

struct CommandBufferTimeout {
    value: OnceLock<Duration>,
}

impl CommandBufferTimeout {
    const fn new() -> Self {
        Self {
            value: OnceLock::new(),
        }
    }

    fn get(&self) -> Duration {
        self.get_or_init_with(|| std::env::var(COMMAND_BUFFER_TIMEOUT_MS_ENV).ok())
    }

    fn get_or_init_with(&self, read_source: impl FnOnce() -> Option<String>) -> Duration {
        *self.value.get_or_init(|| {
            read_source()
                .and_then(|value| parse_command_buffer_timeout_ms(&value))
                .unwrap_or(DEFAULT_COMMAND_BUFFER_TIMEOUT)
        })
    }
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
    fn report_command_buffer_status_clears_sender_when_receiver_is_dropped() {
        let (tx, rx) = tokio::sync::oneshot::channel();
        drop(rx);
        let tx = Mutex::new(Some(tx));

        report_command_buffer_status(&tx, MTLCommandBufferStatus::Error, "failed_kernel");

        assert!(tx.lock().expect("status sender lock").is_none());
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
    fn command_buffer_timeout_cache_reads_source_once() {
        let cache = CommandBufferTimeout::new();
        let reads = std::sync::atomic::AtomicUsize::new(0);

        let first = cache.get_or_init_with(|| {
            reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Some("125".to_owned())
        });
        let second = cache.get_or_init_with(|| {
            reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Some("250".to_owned())
        });

        assert_eq!(first, Duration::from_millis(125));
        assert_eq!(second, first);
        assert_eq!(reads.load(std::sync::atomic::Ordering::SeqCst), 1);
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
