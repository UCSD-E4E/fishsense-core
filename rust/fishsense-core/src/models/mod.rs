//! Model identity: which weight files this crate knows, and checking that a
//! file is one of them.
//!
//! The manifest (`models.toml` at the crate root) is compiled in. It names each
//! model version's file per [`Target`] with its sha256 and size. Nothing here
//! touches the network: fetching is the caller's business (a server pulls from
//! its registry, a phone reads its app bundle), and this module only answers
//! "is this the file I expect?"

use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::sync::LazyLock;

use serde::Deserialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

/// The manifest source, exactly as shipped. Python reads it through
/// `_native.models.manifest_toml()`, so both languages see the same bytes.
pub const MANIFEST_TOML: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/models.toml"));

/// Where a model artifact runs. A model version has at most one artifact per
/// target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Target {
    /// Full precision, server runtimes (CPU / CUDA).
    Server,
    /// On-device: ONNX Runtime with the CoreML execution provider.
    MobileCoreml,
}

impl Target {
    pub fn as_str(&self) -> &'static str {
        match self {
            Target::Server => "server",
            Target::MobileCoreml => "mobile-coreml",
        }
    }
}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("invalid model manifest: {0}")]
    Manifest(String),
    #[error("unknown model {name}/{version} for target {target}; known: {known}")]
    Unknown {
        name: String,
        version: String,
        target: Target,
        known: String,
    },
    #[error("{id}: expected {expected} bytes, got {got}")]
    SizeMismatch { id: String, expected: u64, got: u64 },
    #[error("{id}: expected sha256 {expected}, got {got}")]
    HashMismatch {
        id: String,
        expected: String,
        got: String,
    },
    #[error("reading model file: {0}")]
    Io(#[from] std::io::Error),
}

/// One weight file: a resolved `(name, version, target)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRef {
    pub name: String,
    pub version: String,
    pub target: Target,
    pub filename: String,
    pub sha256: String,
    pub size: u64,
}

impl ModelRef {
    /// Provenance string recorded with every result: `name/version@sha256[:12]`.
    ///
    /// The target is left out on purpose: the hash already tells two builds
    /// apart, and one file can serve several targets (FishIAL's ONNX runs on
    /// both), so the target would name how it was resolved, not what ran.
    pub fn id(&self) -> String {
        format!("{}/{}@{}", self.name, self.version, &self.sha256[..12])
    }
}

/// How hard [`verify`] looks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verify {
    /// Length only. For a file whose integrity something else already vouches
    /// for (a code-signed app bundle), where hashing a large model on every
    /// cold start would cost seconds for nothing.
    Size,
    /// Length, then sha256.
    Full,
}

// ── manifest ────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawManifest {
    #[serde(default)]
    model: Vec<RawModel>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawModel {
    name: String,
    version: String,
    #[serde(default)]
    default: bool,
    // Read by Python stores, not by Rust; accepted so the schema stays strict.
    #[serde(default)]
    #[allow(dead_code)]
    origin: BTreeMap<String, BTreeMap<String, String>>,
    #[serde(default)]
    artifact: Vec<RawArtifact>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawArtifact {
    targets: Vec<Target>,
    filename: String,
    sha256: String,
    size: u64,
}

/// The parsed, validated manifest.
#[derive(Debug)]
pub struct Manifest {
    refs: Vec<ModelRef>,
    defaults: BTreeMap<String, String>,
}

static BUILTIN: LazyLock<Manifest> = LazyLock::new(|| {
    // A bad compiled-in manifest is a build defect, and the test suite parses
    // it (`builtin_manifest_is_valid`), so this cannot fire in a released crate.
    Manifest::parse(MANIFEST_TOML).expect("compiled-in models.toml is invalid")
});

impl Manifest {
    /// The manifest compiled into this crate.
    pub fn builtin() -> &'static Manifest {
        &BUILTIN
    }

    pub fn parse(text: &str) -> Result<Manifest, ModelError> {
        let raw: RawManifest =
            toml::from_str(text).map_err(|e| ModelError::Manifest(e.to_string()))?;
        let bad = |msg: String| Err(ModelError::Manifest(msg));

        let mut refs = Vec::new();
        let mut defaults = BTreeMap::new();
        let mut seen_versions = HashSet::new();
        for m in raw.model {
            if !seen_versions.insert((m.name.clone(), m.version.clone())) {
                return bad(format!("{}/{} listed twice", m.name, m.version));
            }
            if m.default && defaults.insert(m.name.clone(), m.version.clone()).is_some() {
                return bad(format!("{} has more than one default version", m.name));
            }
            let mut seen_targets = HashSet::new();
            for a in m.artifact {
                if a.sha256.len() != 64 || !a.sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return bad(format!("{}/{}: sha256 is not 64 hex digits", m.name, m.version));
                }
                if a.size == 0 {
                    return bad(format!("{}/{}: size is 0", m.name, m.version));
                }
                if a.targets.is_empty() {
                    return bad(format!("{}/{}: artifact has no targets", m.name, m.version));
                }
                for target in a.targets {
                    if !seen_targets.insert(target) {
                        return bad(format!(
                            "{}/{}: two artifacts for target {target}",
                            m.name, m.version
                        ));
                    }
                    refs.push(ModelRef {
                        name: m.name.clone(),
                        version: m.version.clone(),
                        target,
                        filename: a.filename.clone(),
                        sha256: a.sha256.to_ascii_lowercase(),
                        size: a.size,
                    });
                }
            }
        }
        for (name, _) in &seen_versions {
            if !defaults.contains_key(name) {
                return bad(format!("{name} has no default version"));
            }
        }
        Ok(Manifest { refs, defaults })
    }

    /// The artifact for `(name, version, target)`; `version = None` means the
    /// pinned default. An unknown combination is an error that lists what is
    /// known. It never falls back to another version or target.
    pub fn resolve(
        &self,
        name: &str,
        version: Option<&str>,
        target: Target,
    ) -> Result<ModelRef, ModelError> {
        let version = match version {
            Some(v) => Some(v),
            None => self.defaults.get(name).map(String::as_str),
        };
        if let Some(version) = version
            && let Some(r) = self
                .refs
                .iter()
                .find(|r| r.name == name && r.version == version && r.target == target)
        {
            return Ok(r.clone());
        }
        Err(ModelError::Unknown {
            name: name.to_string(),
            version: version.unwrap_or("<default>").to_string(),
            target,
            known: self
                .refs
                .iter()
                .map(|r| format!("{}/{} ({})", r.name, r.version, r.target))
                .collect::<Vec<_>>()
                .join(", "),
        })
    }

    /// Every resolvable artifact.
    pub fn refs(&self) -> &[ModelRef] {
        &self.refs
    }
}

// ── verification ────────────────────────────────────────────────────────────

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// Check that `bytes` are the file `model` names.
pub fn verify(model: &ModelRef, bytes: &[u8], how: Verify) -> Result<(), ModelError> {
    let got = bytes.len() as u64;
    if got != model.size {
        return Err(ModelError::SizeMismatch {
            id: model.id(),
            expected: model.size,
            got,
        });
    }
    if how == Verify::Full {
        let got = sha256_hex(bytes);
        if got != model.sha256 {
            return Err(ModelError::HashMismatch {
                id: model.id(),
                expected: model.sha256.clone(),
                got,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ABC_SHA: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    fn tiny() -> Manifest {
        Manifest::parse(&format!(
            r#"
            [[model]]
            name = "m"
            version = "1"
            default = true
            [[model.artifact]]
            targets = ["server", "mobile-coreml"]
            filename = "m.onnx"
            sha256 = "{ABC_SHA}"
            size = 3

            [[model]]
            name = "m"
            version = "2"
            [[model.artifact]]
            targets = ["server"]
            filename = "m2.onnx"
            sha256 = "{ABC_SHA}"
            size = 3
            "#
        ))
        .unwrap()
    }

    #[test]
    fn builtin_manifest_is_valid() {
        let m = Manifest::builtin();
        assert!(m.resolve("fishial", None, Target::Server).is_ok());
        assert!(m.resolve("fishial", None, Target::MobileCoreml).is_ok());
        assert!(m.resolve("laser-detector", None, Target::Server).is_ok());
    }

    #[test]
    fn none_resolves_to_the_default_version() {
        assert_eq!(tiny().resolve("m", None, Target::Server).unwrap().version, "1");
        assert_eq!(
            tiny().resolve("m", Some("2"), Target::Server).unwrap().filename,
            "m2.onnx"
        );
    }

    #[test]
    fn unknown_target_is_an_error_not_a_fallback() {
        // m/2 has a server artifact only; asking for mobile must not hand back
        // the server file.
        let err = tiny()
            .resolve("m", Some("2"), Target::MobileCoreml)
            .unwrap_err();
        assert!(matches!(err, ModelError::Unknown { .. }), "{err}");
    }

    #[test]
    fn unknown_name_lists_the_known_models() {
        let err = tiny().resolve("nope", None, Target::Server).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("m/1 (server)"), "{msg}");
    }

    #[test]
    fn id_is_name_version_and_short_hash() {
        let r = tiny().resolve("m", None, Target::Server).unwrap();
        assert_eq!(r.id(), "m/1@ba7816bf8f01");
    }

    #[test]
    fn verify_accepts_the_right_bytes() {
        let r = tiny().resolve("m", None, Target::Server).unwrap();
        verify(&r, b"abc", Verify::Full).unwrap();
    }

    #[test]
    fn verify_rejects_wrong_bytes_of_the_right_size() {
        let r = tiny().resolve("m", None, Target::Server).unwrap();
        let err = verify(&r, b"abd", Verify::Full).unwrap_err();
        assert!(matches!(err, ModelError::HashMismatch { .. }), "{err}");
        // Size-only checking is exactly that: it cannot see this.
        verify(&r, b"abd", Verify::Size).unwrap();
    }

    #[test]
    fn verify_rejects_the_wrong_size_before_hashing() {
        let r = tiny().resolve("m", None, Target::Server).unwrap();
        let err = verify(&r, b"abcd", Verify::Size).unwrap_err();
        assert!(matches!(err, ModelError::SizeMismatch { .. }), "{err}");
    }

    fn parse_err(body: &str) -> String {
        Manifest::parse(body).unwrap_err().to_string()
    }

    #[test]
    fn manifest_rejects_a_short_hash() {
        let msg = parse_err(
            r#"[[model]]
            name = "m"
            version = "1"
            default = true
            [[model.artifact]]
            targets = ["server"]
            filename = "f"
            sha256 = "abc"
            size = 3"#,
        );
        assert!(msg.contains("64 hex"), "{msg}");
    }

    #[test]
    fn manifest_rejects_a_name_without_a_default() {
        let msg = parse_err(&format!(
            r#"[[model]]
            name = "m"
            version = "1"
            [[model.artifact]]
            targets = ["server"]
            filename = "f"
            sha256 = "{ABC_SHA}"
            size = 3"#
        ));
        assert!(msg.contains("no default"), "{msg}");
    }

    #[test]
    fn manifest_rejects_two_artifacts_for_one_target() {
        let msg = parse_err(&format!(
            r#"[[model]]
            name = "m"
            version = "1"
            default = true
            [[model.artifact]]
            targets = ["server"]
            filename = "a"
            sha256 = "{ABC_SHA}"
            size = 3
            [[model.artifact]]
            targets = ["server"]
            filename = "b"
            sha256 = "{ABC_SHA}"
            size = 3"#
        ));
        assert!(msg.contains("two artifacts"), "{msg}");
    }

    #[test]
    fn manifest_rejects_an_unknown_target() {
        let msg = parse_err(&format!(
            r#"[[model]]
            name = "m"
            version = "1"
            default = true
            [[model.artifact]]
            targets = ["toaster"]
            filename = "a"
            sha256 = "{ABC_SHA}"
            size = 3"#
        ));
        assert!(msg.contains("toaster"), "{msg}");
    }
}
