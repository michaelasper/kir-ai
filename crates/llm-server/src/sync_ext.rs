use std::sync::{Mutex, MutexGuard};

pub(crate) trait FailPoisonedMutex<T> {
    fn lock_or_panic(&self, name: &'static str) -> MutexGuard<'_, T>;
}

impl<T> FailPoisonedMutex<T> for Mutex<T> {
    fn lock_or_panic(&self, name: &'static str) -> MutexGuard<'_, T> {
        match self.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                tracing::error!(
                    lock = name,
                    error = %poisoned,
                    "poisoned mutex detected; refusing access"
                );
                panic!("poisoned mutex `{name}`: {poisoned}; refusing to recover protected state");
            }
        }
    }
}
