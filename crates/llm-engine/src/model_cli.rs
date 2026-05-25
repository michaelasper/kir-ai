//! Compatibility facade for model-management CLI helpers used by tests and
//! external tooling.

pub use crate::cli::model::{
    ModelPlanOptions, PruneMode, model_home_from_args, model_inspect_json,
    model_lifecycle_request_from_args, model_list_json, model_list_json_with_mode,
    model_plan_options_from_args, model_prune_json, model_verify_json, prune_mode_from_args,
    prune_policy_from_args, run, snapshot_readiness_mode_from_args,
};
