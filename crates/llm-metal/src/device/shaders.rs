pub(crate) const METAL_SOURCE: &str = concat!(
    include_str!("shaders/generic.metal"),
    include_str!("shaders/transformer.metal"),
    include_str!("shaders/matvec.metal"),
    include_str!("shaders/reductions.metal"),
);

include!(concat!(env!("OUT_DIR"), "/shader_metallib.rs"));

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum ShaderLibraryLoadCandidate {
    EmbeddedMetallib,
    Source,
}

const EMBEDDED_LOAD_PLAN: &[ShaderLibraryLoadCandidate] = &[
    ShaderLibraryLoadCandidate::EmbeddedMetallib,
    ShaderLibraryLoadCandidate::Source,
];
const SOURCE_LOAD_PLAN: &[ShaderLibraryLoadCandidate] = &[ShaderLibraryLoadCandidate::Source];

pub(crate) fn embedded_metallib() -> Option<&'static [u8]> {
    EMBEDDED_METALLIB
}

pub(crate) fn shader_library_load_plan() -> &'static [ShaderLibraryLoadCandidate] {
    if embedded_metallib().is_some() {
        EMBEDDED_LOAD_PLAN
    } else {
        SOURCE_LOAD_PLAN
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn shader_source_digest_matches_current_source() {
        assert_eq!(SHADER_SOURCE_SHA256, sha256_hex(METAL_SOURCE.as_bytes()));
    }

    #[test]
    fn shader_library_load_plan_preserves_source_fallback() {
        let plan = shader_library_load_plan();

        assert_eq!(plan.last(), Some(&ShaderLibraryLoadCandidate::Source));
        if embedded_metallib().is_some() {
            assert_eq!(
                plan.first(),
                Some(&ShaderLibraryLoadCandidate::EmbeddedMetallib)
            );
        } else {
            assert_eq!(plan, [ShaderLibraryLoadCandidate::Source]);
        }
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        let digest = Sha256::digest(bytes);
        let mut encoded = String::with_capacity(digest.len() * 2);
        for byte in digest {
            use std::fmt::Write as _;
            write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
        }
        encoded
    }
}
