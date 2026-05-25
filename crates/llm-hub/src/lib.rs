#[cfg(feature = "remote")]
mod client;
mod error;
mod lifecycle;
mod manifest;
mod model_info;
mod plan;
mod profile;
mod prune;
mod repo;
mod store;

#[cfg(feature = "remote")]
pub use client::{HubClient, HubTimeouts};
pub use error::HubError;
#[cfg(feature = "remote")]
pub use lifecycle::ModelLifecycleService;
pub use lifecycle::{DEFAULT_MODEL_REVISION, ModelLifecyclePlanOptions, ModelLifecycleRequest};
pub use manifest::{
    ManifestFile, PromotedSnapshot, SNAPSHOT_MANIFEST_FILE, SNAPSHOT_VERIFICATION_FILE,
    SnapshotManifest, SnapshotVerification, SnapshotVerificationMode, SnapshotVerificationStamp,
    VerificationStampFile,
};
pub use model_info::HubModelInfo;
pub use plan::{ArtifactClass, DownloadPlan, PlannedFile, build_download_plan};
pub use profile::{DEFAULT_MODEL_PROFILE_NAME, ModelProfile};
pub use prune::{
    DeletedSnapshot, ProtectedSnapshot, PruneCandidate, PrunePlan, PrunePolicy, PruneReport,
};
pub use repo::{HubFile, HubRepoId, RepoType};
pub use store::{
    ModelAlias, ModelStore, QuarantineMetadata, QuarantinedSnapshot, SnapshotInventory,
    SnapshotReadiness, SnapshotReadinessMode, SnapshotRecord, SnapshotUsage,
};
