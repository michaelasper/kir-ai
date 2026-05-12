mod admin;
#[cfg(test)]
mod admin_tests;
mod config;
mod error;
mod inference;
mod lifecycle;
mod metrics;
#[cfg(feature = "test-utils")]
// Security: gated behind non-default feature to prevent production exposure (GH#139).
mod protocol;
mod request;
mod requests;
mod router;
mod scheduler;
mod state;
mod streaming;

pub use config::{EngineConfigError, EngineOptions};
#[cfg(feature = "test-utils")]
pub use router::build_router_with_protocol_test_backend;
pub use router::{
    build_router, build_router_with_backend, build_router_with_backend_and_concurrency,
    build_router_with_backend_and_options,
    build_router_with_backend_and_options_allowing_unauthenticated_admin,
};

use error::{EngineError, runtime_error_metadata};
use request::parse_json_request;
use state::AppState;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync_ext::RecoverPoisonedMutex;
    use std::sync::Mutex;

    #[test]
    fn poisoned_mutex_lock_recovers_inner_state() {
        let mutex = Mutex::new(7_u32);
        let _ = std::panic::catch_unwind(|| {
            let _guard = mutex.lock().expect("test lock");
            panic!("poison test mutex");
        });

        *mutex.lock_or_recover("test") += 1;

        assert_eq!(*mutex.lock_or_recover("test"), 8);
    }

    #[test]
    fn admin_model_profile_accepts_built_in_profiles() {
        for name in llm_hub::ModelProfile::builtin_names() {
            let profile =
                admin::model_profile(name).expect("admin profile matcher accepts profile");

            assert_eq!(profile.name, name);
        }
    }
}
