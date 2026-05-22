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
mod rate_limit;
mod request;
mod requests;
mod router;
mod scheduler;
mod state;
mod streaming;

pub use config::{
    EngineConfigError, EngineOptions, PublicInferenceRateLimit, configured_hub_client,
};
#[cfg(feature = "test-utils")]
pub use router::build_router_with_protocol_test_backend;
#[allow(deprecated)]
pub use router::{
    RouterBuilder, build_router, build_router_with_backend,
    build_router_with_backend_and_concurrency, build_router_with_backend_and_options,
    build_router_with_backend_and_options_allowing_unauthenticated_admin,
    build_router_with_backend_and_options_allowing_unauthenticated_admin_and_backend_metrics,
    build_router_with_backend_and_options_and_backend_metrics,
};

use error::{EngineError, EngineErrorBody};
use request::parse_json_request;
use state::AppState;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync_ext::FailPoisonedMutex;
    use std::sync::Mutex;

    #[test]
    fn poisoned_mutex_lock_recovers_inner_state() {
        let mutex = Mutex::new(7_u32);
        let _ = std::panic::catch_unwind(|| {
            let _guard = mutex.lock().expect("test lock");
            panic!("poison test mutex");
        });

        let guard = mutex.lock_or_panic("test");
        assert_eq!(*guard, 7);
        drop(guard);
        assert!(!mutex.is_poisoned());
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
