use std::sync::{Mutex, MutexGuard};

pub trait FailPoisonedMutex<T> {
    /// Lock a mutex without letting poison escalate into a process panic.
    ///
    /// The legacy method name is retained for existing call sites, but poison
    /// is treated as recoverable for server/runtime state guarded by this helper.
    fn lock_or_panic(&self, name: &'static str) -> MutexGuard<'_, T>;
}

impl<T> FailPoisonedMutex<T> for Mutex<T> {
    #[track_caller]
    fn lock_or_panic(&self, name: &'static str) -> MutexGuard<'_, T> {
        match self.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                tracing::warn!(mutex = name, "recovering poisoned mutex protected state");
                self.clear_poison();
                poisoned.into_inner()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FailPoisonedMutex;
    use std::sync::Mutex;

    #[test]
    fn lock_or_panic_recovers_and_clears_poisoned_mutex() {
        let mutex = Mutex::new(7_u32);
        let poison_result = std::panic::catch_unwind(|| {
            let mut guard = mutex.lock_or_panic("test");
            *guard = 13;
            panic!("poison test mutex");
        });
        assert!(poison_result.is_err());
        assert!(mutex.is_poisoned());

        let recovered = std::panic::catch_unwind(|| {
            let guard = mutex.lock_or_panic("test");
            assert_eq!(*guard, 13);
        });

        assert!(
            recovered.is_ok(),
            "poisoned locks should recover protected state"
        );
        assert!(!mutex.is_poisoned());
    }
}
