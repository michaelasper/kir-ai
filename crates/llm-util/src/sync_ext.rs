use std::sync::{Mutex, MutexGuard};

pub trait FailPoisonedMutex<T> {
    fn lock_or_panic(&self, name: &'static str) -> MutexGuard<'_, T>;
}

impl<T> FailPoisonedMutex<T> for Mutex<T> {
    #[track_caller]
    fn lock_or_panic(&self, name: &'static str) -> MutexGuard<'_, T> {
        match self.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                panic!("poisoned mutex `{name}`: {poisoned}; refusing to recover protected state");
            }
        }
    }
}
