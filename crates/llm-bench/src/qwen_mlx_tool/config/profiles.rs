use super::super::super::cli::flag_values;
use super::{
    DefaultOrU32, DefaultOrU64, MlxLmSettings, NormalizedLaneConfig, NormalizedLaneKind,
    NormalizedModelAddressing, NormalizedSweepProfile, NormalizedTemplatePolicy, UnsetOrU64,
};
use crate::MlxToolParserMode;
use std::path::PathBuf;

const PROFILE_PROXY_MODEL_ID: &str = "local-qwen36-mlx";
const PROFILE_CACHE_BYTES_1G: u64 = 1_073_741_824;

pub(super) fn expand_sweep_profile(
    profile: NormalizedSweepProfile,
    args: &[String],
) -> anyhow::Result<Vec<NormalizedLaneConfig>> {
    let snapshot = required_profile_snapshot(args)?;
    Ok(match profile {
        NormalizedSweepProfile::QwenMlxCachePrefill => qwen_mlx_cache_prefill_lanes(snapshot),
        NormalizedSweepProfile::QwenMlxPrefill135k => qwen_mlx_prefill_135k_lanes(snapshot),
        NormalizedSweepProfile::QwenMlxPrefill135kExperimental => {
            qwen_mlx_prefill_135k_experimental_lanes(snapshot)
        }
        NormalizedSweepProfile::QwenMlxStablePrefix => qwen_mlx_stable_prefix_lanes(snapshot),
    })
}

fn required_profile_snapshot(args: &[String]) -> anyhow::Result<&str> {
    let snapshots = flag_values(args, "--snapshot");
    match snapshots.as_slice() {
        [snapshot] => Ok(snapshot),
        [] => anyhow::bail!("--sweep-profile requires --snapshot <path>"),
        _ => anyhow::bail!("--snapshot may only be provided once for --sweep-profile"),
    }
}

fn qwen_mlx_cache_prefill_lanes(snapshot: &str) -> Vec<NormalizedLaneConfig> {
    vec![
        profile_direct_lane("mlx-default", 8080, MlxLmSettings::default(), snapshot),
        profile_direct_lane(
            "mlx-cache-size-4096",
            8081,
            MlxLmSettings {
                prompt_cache_size: DefaultOrU64::Value(4096),
                ..MlxLmSettings::default()
            },
            snapshot,
        ),
        profile_direct_lane(
            "mlx-cache-bytes-1g",
            8082,
            MlxLmSettings {
                prompt_cache_bytes: UnsetOrU64::Value(PROFILE_CACHE_BYTES_1G),
                ..MlxLmSettings::default()
            },
            snapshot,
        ),
        profile_direct_lane(
            "mlx-prefill-2048",
            8083,
            MlxLmSettings {
                prefill_step_size: DefaultOrU64::Value(2048),
                ..MlxLmSettings::default()
            },
            snapshot,
        ),
        profile_direct_lane(
            "mlx-prefill-4096",
            8084,
            MlxLmSettings {
                prefill_step_size: DefaultOrU64::Value(4096),
                ..MlxLmSettings::default()
            },
            snapshot,
        ),
        profile_direct_lane(
            "mlx-prefill-8192",
            8085,
            MlxLmSettings {
                prefill_step_size: DefaultOrU64::Value(8192),
                ..MlxLmSettings::default()
            },
            snapshot,
        ),
        profile_direct_lane(
            "mlx-concurrent-4x2",
            8086,
            MlxLmSettings {
                prompt_concurrency: DefaultOrU32::Value(4),
                decode_concurrency: DefaultOrU32::Value(2),
                ..MlxLmSettings::default()
            },
            snapshot,
        ),
        NormalizedLaneConfig {
            name: "kir-proxy".to_owned(),
            endpoint: "http://127.0.0.1:3000".to_owned(),
            declared_model_id: PROFILE_PROXY_MODEL_ID.to_owned(),
            launched_model_id: Some(snapshot.to_owned()),
            snapshot_path: Some(PathBuf::from(snapshot)),
            kind: NormalizedLaneKind::KirAiProxy,
            model_addressing: NormalizedModelAddressing::DefaultModel,
            template: NormalizedTemplatePolicy::SidecarChatTemplateArgs,
            tool_parser: MlxToolParserMode::Auto,
            mlx_lm_settings: MlxLmSettings::default(),
            experimental: false,
        },
    ]
}

fn qwen_mlx_prefill_135k_lanes(snapshot: &str) -> Vec<NormalizedLaneConfig> {
    let steps = [
        ("default", DefaultOrU64::Default),
        ("512", DefaultOrU64::Value(512)),
        ("1024", DefaultOrU64::Value(1024)),
        ("2048", DefaultOrU64::Value(2048)),
        ("4096", DefaultOrU64::Value(4096)),
        ("8192", DefaultOrU64::Value(8192)),
    ];
    let mut lanes = Vec::with_capacity(steps.len() * 2);
    for (index, (label, prefill_step_size)) in steps.into_iter().enumerate() {
        let port_offset = index as u16;
        let settings = MlxLmSettings {
            prefill_step_size,
            ..MlxLmSettings::default()
        };
        lanes.push(profile_direct_lane(
            &format!("mlx-prefill-{label}"),
            8080 + port_offset,
            settings,
            snapshot,
        ));
        lanes.push(profile_proxy_lane(
            &format!("kir-prefill-{label}"),
            3000 + port_offset,
            settings,
            snapshot,
        ));
    }
    lanes
}

fn qwen_mlx_prefill_135k_experimental_lanes(snapshot: &str) -> Vec<NormalizedLaneConfig> {
    let steps = [
        ("8192-control", DefaultOrU64::Value(8192), false),
        ("experimental-12288", DefaultOrU64::Value(12288), true),
        ("experimental-16384", DefaultOrU64::Value(16384), true),
        ("experimental-32768", DefaultOrU64::Value(32768), true),
    ];
    let mut lanes = Vec::with_capacity(steps.len() * 2);
    for (index, (label, prefill_step_size, experimental)) in steps.into_iter().enumerate() {
        let port_offset = index as u16;
        let settings = MlxLmSettings {
            prefill_step_size,
            ..MlxLmSettings::default()
        };
        let mut direct = profile_direct_lane(
            &format!("mlx-prefill-{label}"),
            8080 + port_offset,
            settings,
            snapshot,
        );
        direct.experimental = experimental;
        lanes.push(direct);

        let mut proxy = profile_proxy_lane(
            &format!("kir-prefill-{label}"),
            3000 + port_offset,
            settings,
            snapshot,
        );
        proxy.experimental = experimental;
        lanes.push(proxy);
    }
    lanes
}

fn qwen_mlx_stable_prefix_lanes(snapshot: &str) -> Vec<NormalizedLaneConfig> {
    vec![
        NormalizedLaneConfig {
            name: "mlx-stable-prefix".to_owned(),
            endpoint: "http://127.0.0.1:8080/v1".to_owned(),
            declared_model_id: snapshot.to_owned(),
            launched_model_id: Some(snapshot.to_owned()),
            snapshot_path: Some(PathBuf::from(snapshot)),
            kind: NormalizedLaneKind::DirectMlx,
            model_addressing: NormalizedModelAddressing::ServerDefault,
            template: NormalizedTemplatePolicy::QwenNoThinking,
            tool_parser: MlxToolParserMode::Auto,
            mlx_lm_settings: MlxLmSettings::default(),
            experimental: false,
        },
        NormalizedLaneConfig {
            name: "kir-stable-prefix".to_owned(),
            endpoint: "http://127.0.0.1:3000".to_owned(),
            declared_model_id: PROFILE_PROXY_MODEL_ID.to_owned(),
            launched_model_id: Some(snapshot.to_owned()),
            snapshot_path: Some(PathBuf::from(snapshot)),
            kind: NormalizedLaneKind::KirAiProxy,
            model_addressing: NormalizedModelAddressing::DefaultModel,
            template: NormalizedTemplatePolicy::SidecarChatTemplateArgs,
            tool_parser: MlxToolParserMode::Auto,
            mlx_lm_settings: MlxLmSettings::default(),
            experimental: false,
        },
    ]
}

fn profile_direct_lane(
    name: &str,
    port: u16,
    mlx_lm_settings: MlxLmSettings,
    snapshot: &str,
) -> NormalizedLaneConfig {
    NormalizedLaneConfig {
        name: name.to_owned(),
        endpoint: format!("http://127.0.0.1:{port}/v1"),
        declared_model_id: snapshot.to_owned(),
        launched_model_id: Some(snapshot.to_owned()),
        snapshot_path: Some(PathBuf::from(snapshot)),
        kind: NormalizedLaneKind::DirectMlx,
        model_addressing: NormalizedModelAddressing::ServerDefault,
        template: NormalizedTemplatePolicy::SidecarChatTemplateArgs,
        tool_parser: MlxToolParserMode::Auto,
        mlx_lm_settings,
        experimental: false,
    }
}

fn profile_proxy_lane(
    name: &str,
    port: u16,
    mlx_lm_settings: MlxLmSettings,
    snapshot: &str,
) -> NormalizedLaneConfig {
    NormalizedLaneConfig {
        name: name.to_owned(),
        endpoint: format!("http://127.0.0.1:{port}"),
        declared_model_id: PROFILE_PROXY_MODEL_ID.to_owned(),
        launched_model_id: Some(snapshot.to_owned()),
        snapshot_path: Some(PathBuf::from(snapshot)),
        kind: NormalizedLaneKind::KirAiProxy,
        model_addressing: NormalizedModelAddressing::DefaultModel,
        template: NormalizedTemplatePolicy::SidecarChatTemplateArgs,
        tool_parser: MlxToolParserMode::Auto,
        mlx_lm_settings,
        experimental: false,
    }
}
