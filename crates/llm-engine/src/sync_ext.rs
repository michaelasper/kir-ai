use std::sync::{Mutex, MutexGuard};

pub(crate) trait RecoverPoisonedMutex<T> {
    fn lock_or_recover(&self, name: &'static str) -> MutexGuard<'_, T>;
}

impl<T> RecoverPoisonedMutex<T> for Mutex<T> {
    fn lock_or_recover(&self, name: &'static str) -> MutexGuard<'_, T> {
        match self.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                tracing::error!(lock = name, "recovering poisoned mutex");
                poisoned.into_inner()
            }
        }
    }
}
