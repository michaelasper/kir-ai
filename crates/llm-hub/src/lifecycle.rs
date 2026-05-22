use crate::{
    DEFAULT_MODEL_PROFILE_NAME, DownloadPlan, HubClient, HubError, HubRepoId, ModelProfile,
    ModelStore, PromotedSnapshot,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const DEFAULT_MODEL_REVISION: &str = "main";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ModelLifecycleRequest {
    pub repo_id: String,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub profile: Option<String>,
    #[serde(default)]
    pub metadata_only: bool,
}

impl ModelLifecycleRequest {
    pub fn new(repo_id: impl Into<String>) -> Self {
        Self {
            repo_id: repo_id.into(),
            revision: None,
            profile: None,
            metadata_only: false,
        }
    }

    pub fn with_revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = Some(revision.into());
        self
    }

    pub fn with_profile(mut self, profile: impl Into<String>) -> Self {
        self.profile = Some(profile.into());
        self
    }

    pub fn with_metadata_only(mut self, metadata_only: bool) -> Self {
        self.metadata_only = metadata_only;
        self
    }

    pub fn resolve(&self) -> Result<ModelLifecyclePlanOptions, HubError> {
        let repo_id = HubRepoId::model(self.repo_id.clone())?;
        let revision = self
            .revision
            .clone()
            .unwrap_or_else(|| DEFAULT_MODEL_REVISION.to_owned());
        let profile_name = self
            .profile
            .as_deref()
            .unwrap_or(DEFAULT_MODEL_PROFILE_NAME);
        let profile = ModelProfile::builtin(profile_name).ok_or_else(|| {
            HubError::invalid_request(format!("unknown model profile `{profile_name}`"))
        })?;
        Ok(ModelLifecyclePlanOptions {
            repo_id,
            revision,
            profile,
            metadata_only: self.metadata_only,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelLifecyclePlanOptions {
    pub repo_id: HubRepoId,
    pub revision: String,
    pub profile: ModelProfile,
    pub metadata_only: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct ModelLifecycleService<'a> {
    client: &'a HubClient,
    token: Option<&'a str>,
}

impl<'a> ModelLifecycleService<'a> {
    pub fn new(client: &'a HubClient, token: Option<&'a str>) -> Self {
        Self { client, token }
    }

    pub async fn plan(&self, request: &ModelLifecycleRequest) -> Result<DownloadPlan, HubError> {
        let options = request.resolve()?;
        self.plan_resolved(options).await
    }

    pub async fn pull(
        &self,
        store: &ModelStore,
        request: &ModelLifecycleRequest,
    ) -> Result<PromotedSnapshot, HubError> {
        let plan = self.plan(request).await?;
        self.pull_plan(store, &plan).await
    }

    pub async fn plan_resolved(
        &self,
        options: ModelLifecyclePlanOptions,
    ) -> Result<DownloadPlan, HubError> {
        let mut plan = self
            .client
            .plan_model(
                options.repo_id,
                &options.revision,
                options.profile,
                self.token,
            )
            .await?;
        if options.metadata_only {
            plan = plan.metadata_only();
        }
        Ok(plan)
    }

    pub async fn pull_plan(
        &self,
        store: &ModelStore,
        plan: &DownloadPlan,
    ) -> Result<PromotedSnapshot, HubError> {
        store.pull_plan(self.client, plan, self.token).await
    }
}
