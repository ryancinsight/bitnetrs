//! NPU detection for the BitNet b1.58 inference engine.
//!
//! # Strategy
//!
//! Neural Processing Units (NPUs) are not universally exposed through a single
//! standardised API.  This module uses wgpu adapter enumeration combined with
//! name-based heuristics to identify NPU-like adapters on the current system.
//!
//! On Windows with DirectML-backed wgpu, NPU adapters appear alongside GPU
//! adapters in the enumerated list.  Qualcomm Snapdragon X NPUs, Intel NPUs
//! (Meteor Lake, Lunar Lake), and AMD NPUs (Hawk Point, Strix Point) all appear
//! in this list when their drivers are installed.
//!
//! # Detection Heuristics
//!
//! An adapter is classified as an NPU if its reported name contains any of the
//! following substrings (case-insensitive):
//!
//! | Vendor   | Substrings checked                              |
//! |----------|-------------------------------------------------|
//! | Intel    | `"npu"`, `"neural"`, `"vpu"`                   |
//! | AMD      | `"npu"`, `"neural"`                             |
//! | Qualcomm | `"npu"`, `"neural"`, `"hexagon"`, `"adreno"`   |
//! | Apple    | `"neural engine"`, `"ane"`                      |
//! | Generic  | `"npu"`, `"neural"`, `"accelerator"`           |
//!
//! The heuristic is conservative: a false negative (missing a real NPU) is
//! preferred over a false positive (treating a GPU as an NPU).
//!
//! # Adapter Priority
//!
//! When multiple NPU-like adapters are found, they are ranked by:
//! 1. Non-CPU device type (hardware NPU preferred over software simulation)
//! 2. Alphabetical name (for determinism)
//!
//! The adapter at `device_id = 0` in [`detect_npu`] is the highest-priority NPU.
//!
//! # Cross-Platform Notes
//!
//! - **Windows**: Full detection via wgpu + DirectML adapter enumeration.
//! - **macOS/Linux**: Limited detection; most NPUs are not exposed via wgpu.
//!   The Apple Neural Engine is not accessible through wgpu/Metal compute.
//! - **Web**: Not supported.
//!
//! # Invariants
//!
//! - [`detect_npu`] never panics regardless of the hardware configuration.
//! - Returns `None` (not an error) when no NPU is found.
//! - The `adapter_index` field in [`NpuInfo`] is always a valid index into
//!   the wgpu adapter list at the time of detection.

use tracing::{debug, trace};
use wgpu::{Adapter, Backends, Instance, InstanceDescriptor, InstanceFlags};

// ---------------------------------------------------------------------------
// NpuVendor
// ---------------------------------------------------------------------------

/// Known NPU vendor classifications.
///
/// Used for logging and potential vendor-specific optimisation paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NpuVendor {
    /// Intel NPU (Meteor Lake VPU, Lunar Lake, etc.)
    Intel,
    /// AMD NPU (XDNA architecture: Hawk Point, Strix Point)
    Amd,
    /// Qualcomm Hexagon NPU (Snapdragon X Elite, Snapdragon 8cx Gen 3)
    Qualcomm,
    /// Apple Neural Engine (not accessible via wgpu; included for completeness)
    Apple,
    /// Samsung Exynos NPU
    Samsung,
    /// MediaTek APU / Dimensity NPU
    MediaTek,
    /// Unknown / unclassified NPU-like device
    Unknown,
}

impl std::fmt::Display for NpuVendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Intel => write!(f, "Intel"),
            Self::Amd => write!(f, "AMD"),
            Self::Qualcomm => write!(f, "Qualcomm"),
            Self::Apple => write!(f, "Apple"),
            Self::Samsung => write!(f, "Samsung"),
            Self::MediaTek => write!(f, "MediaTek"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

// ---------------------------------------------------------------------------
// NpuAdapterType
// ---------------------------------------------------------------------------

/// The wgpu device type of the detected NPU adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpuAdapterType {
    /// Discrete hardware NPU (has its own memory / die).
    DiscreteNpu,
    /// Integrated NPU (on same die as CPU).
    IntegratedNpu,
    /// Virtual or emulated NPU.
    Virtual,
    /// Software-emulated NPU (fallback only; not recommended).
    Software,
    /// Device type could not be determined.
    Unknown,
}

impl std::fmt::Display for NpuAdapterType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DiscreteNpu => write!(f, "DiscreteNpu"),
            Self::IntegratedNpu => write!(f, "IntegratedNpu"),
            Self::Virtual => write!(f, "Virtual"),
            Self::Software => write!(f, "Software"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

// ---------------------------------------------------------------------------
// NpuInfo
// ---------------------------------------------------------------------------

/// Information about a detected NPU adapter.
///
/// Returned by [`detect_npu`] when an NPU is found.  The `adapter_index` field
/// is a direct index into the wgpu adapter list and can be passed to
/// `bitnet_gpu::create_wgpu_backend_blocking(adapter_index)` to create a backend
/// targeting the NPU through the GPU facade.
///
/// # Invariants
///
/// - `name` is always non-empty.
/// - `adapter_index` is valid at the time [`detect_npu`] was called.
///   It may become invalid if adapters are dynamically added/removed (rare).
#[derive(Debug, Clone)]
pub struct NpuInfo {
    /// Human-readable adapter name as reported by the driver.
    ///
    /// Examples:
    /// - `"Intel(R) AI Boost"`
    /// - `"Qualcomm(R) AI Accelerator"`
    /// - `"AMD NPU"`
    pub name: String,

    /// Identified vendor.
    pub vendor: NpuVendor,

    /// Adapter device type (integrated, discrete, etc.)
    pub adapter_type: NpuAdapterType,

    /// wgpu backend used to enumerate this adapter.
    ///
    /// Examples: `"Dx12"`, `"Vulkan"`, `"Metal"`.
    pub backend: String,

    /// Zero-based index into the sorted wgpu adapter list at detection time.
    ///
    /// Pass this to `bitnet_gpu::create_wgpu_backend_blocking(adapter_index)` to
    /// target the NPU through the GPU facade.
    pub adapter_index: u32,

    /// PCI vendor ID (0 if unavailable or non-PCI device).
    pub pci_vendor_id: u32,

    /// PCI device ID (0 if unavailable or non-PCI device).
    pub pci_device_id: u32,
}

impl std::fmt::Display for NpuInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} [vendor={}, type={}, backend={}]",
            self.name, self.vendor, self.adapter_type, self.backend
        )
    }
}

// ---------------------------------------------------------------------------
// NPU keyword lists
// ---------------------------------------------------------------------------

/// Substrings in adapter names that suggest NPU presence (case-insensitive).
///
/// These are checked against the lowercased adapter name.
const NPU_NAME_KEYWORDS: &[&str] = &[
    "npu",
    "neural",
    "vpu",
    "hexagon",
    "ai accelerator",
    "ai boost",
    "xdna",
    "neural engine",
    "ane",
    "apu",      // MediaTek APU, generic APUs
    "exynos",   // Samsung Exynos NPU
    "mediatek", // MediaTek NPU
    "samsung",  // Samsung NPU branding
];

/// Substrings that identify Intel NPUs.
const INTEL_NPU_KEYWORDS: &[&str] = &["intel", "npu", "ai boost", "vpu"];

/// Substrings that identify AMD NPUs.
const AMD_NPU_KEYWORDS: &[&str] = &["amd", "xdna"];

/// Substrings that identify Qualcomm NPUs.
const QUALCOMM_NPU_KEYWORDS: &[&str] = &["qualcomm", "snapdragon", "hexagon", "adreno"];

/// Substrings that identify Apple NPUs.
const APPLE_NPU_KEYWORDS: &[&str] = &["apple", "neural engine", "ane"];

/// Substrings that identify Samsung NPUs (Exynos NPU, Samsung NPU branding).
const SAMSUNG_NPU_KEYWORDS: &[&str] = &["samsung", "exynos"];

/// Substrings that identify MediaTek NPUs (APU, Dimensity).
const MEDIATEK_NPU_KEYWORDS: &[&str] = &["mediatek", "dimensity", "apu"];

// ---------------------------------------------------------------------------
// detect_npu
// ---------------------------------------------------------------------------

/// Probe the system for NPU adapters and return the highest-priority one.
///
/// Enumerates all wgpu-visible compute adapters, applies name-based heuristics
/// to identify NPU-like devices, and returns a [`NpuInfo`] for the best
/// candidate.
///
/// # Returns
///
/// - `Some(NpuInfo)` if at least one NPU-like adapter was found.
/// - `None` if no NPU was detected or if wgpu adapter enumeration fails.
///
/// # Panics
///
/// Never panics.  All internal errors result in `None`.
///
/// # Example
///
/// ```
/// use bitnet_npu::detect::{detect_npu, NpuVendor};
///
/// match bitnet_npu::detect::detect_npu() {
///     Some(info) => println!("Found NPU: {info}"),
///     None => println!("No NPU detected; will use CPU fallback"),
/// }
/// ```
pub fn detect_npu() -> Option<NpuInfo> {
    // ── Environment variable override ─────────────────────────────────────────
    //
    // BITNET_NPU_ADAPTER: comma-separated list of adapter name substrings.
    // If set, an adapter is considered an NPU candidate if its lowercased name
    // contains any of the specified substrings (in addition to the built-in
    // keyword list).
    //
    // Example: BITNET_NPU_ADAPTER=qualcomm,hexagon
    let env_override: Option<Vec<String>> = std::env::var("BITNET_NPU_ADAPTER")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| s.split(',').map(|kw| kw.trim().to_lowercase()).collect());
    let env_kws: &[String] = env_override.as_deref().unwrap_or(&[]);

    // Create a wgpu instance that queries all available backends.
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::all(),
        flags: InstanceFlags::default(),
        ..Default::default()
    });

    // Enumerate all available adapters.
    let adapters: Vec<Adapter> = instance.enumerate_adapters(Backends::all());

    if adapters.is_empty() {
        debug!("No wgpu adapters found; NPU detection returning None");
        return None;
    }

    // Log all adapters at trace level for debugging.
    for (idx, adapter) in adapters.iter().enumerate() {
        let info = adapter.get_info();
        trace!(
            index = idx,
            name = %info.name,
            backend = ?info.backend,
            device_type = ?info.device_type,
            "wgpu adapter found during NPU detection"
        );
    }

    // Filter adapters to NPU candidates, preserving their wgpu adapter index.
    // Checks both the built-in NPU_NAME_KEYWORDS and any BITNET_NPU_ADAPTER
    // environment variable overrides.
    let mut candidates: Vec<(usize, &Adapter)> = adapters
        .iter()
        .enumerate()
        .filter(|(_, adapter)| {
            let info = adapter.get_info();
            is_npu_adapter_extended(&info.name, env_kws)
        })
        .collect();

    if candidates.is_empty() {
        debug!("No NPU-like adapters detected in wgpu adapter list");
        return None;
    }

    // Sort candidates by priority:
    // 1. Hardware (non-CPU) adapters first.
    // 2. Smaller adapter index (earlier in the enumeration = higher priority).
    candidates.sort_by_key(|(idx, adapter)| {
        let info = adapter.get_info();
        let is_software = matches!(info.device_type, wgpu::DeviceType::Cpu);
        (is_software as u8, *idx)
    });

    // Take the highest-priority candidate.
    let (adapter_index, best_adapter) = candidates.into_iter().next()?;
    let raw_info = best_adapter.get_info();

    let name_lower = raw_info.name.to_lowercase();
    let vendor = classify_vendor(&name_lower, raw_info.vendor);
    let adapter_type = classify_adapter_type(raw_info.device_type);

    let npu_info = NpuInfo {
        name: raw_info.name.clone(),
        vendor,
        adapter_type,
        backend: format!("{:?}", raw_info.backend),
        adapter_index: adapter_index as u32,
        pci_vendor_id: raw_info.vendor,
        pci_device_id: raw_info.device,
    };

    debug!(
        npu = %npu_info,
        "NPU detection complete"
    );

    Some(npu_info)
}

/// Detect all NPU adapters (not just the best one).
///
/// Useful for diagnostics or when the caller wants to select a specific NPU
/// from a list.
///
/// # Returns
///
/// A (possibly empty) `Vec<NpuInfo>` sorted by priority (best first).
pub fn detect_all_npus() -> Vec<NpuInfo> {
    // Honour the same BITNET_NPU_ADAPTER env override as detect_npu so that
    // both functions remain consistent.
    let env_override: Option<Vec<String>> = std::env::var("BITNET_NPU_ADAPTER")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| s.split(',').map(|kw| kw.trim().to_lowercase()).collect());
    let env_kws: &[String] = env_override.as_deref().unwrap_or(&[]);

    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::all(),
        flags: InstanceFlags::default(),
        ..Default::default()
    });

    let adapters: Vec<Adapter> = instance.enumerate_adapters(Backends::all());

    let mut results: Vec<NpuInfo> = adapters
        .iter()
        .enumerate()
        .filter(|(_, adapter)| {
            let info = adapter.get_info();
            is_npu_adapter_extended(&info.name, env_kws)
        })
        .map(|(idx, adapter)| {
            let info = adapter.get_info();
            let name_lower = info.name.to_lowercase();
            NpuInfo {
                name: info.name.clone(),
                vendor: classify_vendor(&name_lower, info.vendor),
                adapter_type: classify_adapter_type(info.device_type),
                backend: format!("{:?}", info.backend),
                adapter_index: idx as u32,
                pci_vendor_id: info.vendor,
                pci_device_id: info.device,
            }
        })
        .collect();

    // Sort: hardware before software, then by index.
    results.sort_by_key(|info| {
        let is_software = matches!(info.adapter_type, NpuAdapterType::Software);
        (is_software as u8, info.adapter_index)
    });

    results
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Returns `true` if `name` matches any built-in NPU keyword heuristic **or**
/// any caller-supplied extra keywords.
///
/// # Arguments
///
/// - `name`:           Raw adapter name (any case; lowercased internally).
/// - `extra_keywords`: Additional lowercased substrings to match against,
///                     typically sourced from the `BITNET_NPU_ADAPTER`
///                     environment variable.
///
/// This is the core detection predicate.  It uses a simple substring match
/// on the lowercased adapter name.  The predicate is intentionally conservative:
/// generic GPU names like `"NVIDIA GeForce RTX 4090"` do not match.
fn is_npu_adapter_extended(name: &str, extra_keywords: &[String]) -> bool {
    let lower = name.to_lowercase();
    NPU_NAME_KEYWORDS.iter().any(|&kw| lower.contains(kw))
        || extra_keywords.iter().any(|kw| lower.contains(kw.as_str()))
}

/// Returns `true` if the adapter name matches the built-in NPU keyword
/// heuristic (no extra keywords).
///
/// Thin wrapper around [`is_npu_adapter_extended`] for call sites that do not
/// need environment-variable overrides.  Only compiled in test builds because
/// production call-sites use [`is_npu_adapter_extended`] directly (passing the
/// `BITNET_NPU_ADAPTER` env-override keywords).
#[cfg(test)]
fn is_npu_adapter(name: &str) -> bool {
    is_npu_adapter_extended(name, &[])
}

/// Classify the NPU vendor from the adapter name and PCI vendor ID.
///
/// PCI vendor IDs:
/// - Intel: 0x8086
/// - AMD: 0x1002
/// - Qualcomm: 0x17CB / 0x5143
/// - Apple: 0x106B
fn classify_vendor(name_lower: &str, pci_vendor_id: u32) -> NpuVendor {
    // PCI ID-based classification (most reliable).
    match pci_vendor_id {
        0x8086 => return NpuVendor::Intel,
        0x1002 => return NpuVendor::Amd,
        0x17CB | 0x5143 => return NpuVendor::Qualcomm,
        0x106B => return NpuVendor::Apple,
        _ => {}
    }

    // Name-based fallback.
    if INTEL_NPU_KEYWORDS.iter().any(|&kw| name_lower.contains(kw)) && name_lower.contains("intel")
    {
        return NpuVendor::Intel;
    }
    if AMD_NPU_KEYWORDS.iter().any(|&kw| name_lower.contains(kw)) {
        return NpuVendor::Amd;
    }
    if QUALCOMM_NPU_KEYWORDS
        .iter()
        .any(|&kw| name_lower.contains(kw))
    {
        return NpuVendor::Qualcomm;
    }
    if APPLE_NPU_KEYWORDS.iter().any(|&kw| name_lower.contains(kw)) {
        return NpuVendor::Apple;
    }
    // Samsung: check by name since there is no standardised PCI vendor ID for
    // Samsung NPUs exposed through wgpu.
    if SAMSUNG_NPU_KEYWORDS
        .iter()
        .any(|&kw| name_lower.contains(kw))
    {
        return NpuVendor::Samsung;
    }
    // MediaTek
    if MEDIATEK_NPU_KEYWORDS
        .iter()
        .any(|&kw| name_lower.contains(kw))
    {
        return NpuVendor::MediaTek;
    }

    NpuVendor::Unknown
}

/// Map a wgpu [`DeviceType`] to our [`NpuAdapterType`].
fn classify_adapter_type(device_type: wgpu::DeviceType) -> NpuAdapterType {
    match device_type {
        wgpu::DeviceType::DiscreteGpu => NpuAdapterType::DiscreteNpu,
        wgpu::DeviceType::IntegratedGpu => NpuAdapterType::IntegratedNpu,
        wgpu::DeviceType::VirtualGpu => NpuAdapterType::Virtual,
        wgpu::DeviceType::Cpu => NpuAdapterType::Software,
        wgpu::DeviceType::Other => NpuAdapterType::Unknown,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // NpuVendor
    // -----------------------------------------------------------------------

    #[test]
    fn npu_vendor_display_intel() {
        assert_eq!(NpuVendor::Intel.to_string(), "Intel");
    }

    #[test]
    fn npu_vendor_display_amd() {
        assert_eq!(NpuVendor::Amd.to_string(), "AMD");
    }

    #[test]
    fn npu_vendor_display_qualcomm() {
        assert_eq!(NpuVendor::Qualcomm.to_string(), "Qualcomm");
    }

    #[test]
    fn npu_vendor_display_apple() {
        assert_eq!(NpuVendor::Apple.to_string(), "Apple");
    }

    #[test]
    fn npu_vendor_display_unknown() {
        assert_eq!(NpuVendor::Unknown.to_string(), "Unknown");
    }

    // -----------------------------------------------------------------------
    // NpuAdapterType
    // -----------------------------------------------------------------------

    #[test]
    fn npu_adapter_type_display_all_variants() {
        let cases = [
            (NpuAdapterType::DiscreteNpu, "DiscreteNpu"),
            (NpuAdapterType::IntegratedNpu, "IntegratedNpu"),
            (NpuAdapterType::Virtual, "Virtual"),
            (NpuAdapterType::Software, "Software"),
            (NpuAdapterType::Unknown, "Unknown"),
        ];
        for (variant, expected) in cases {
            assert_eq!(
                variant.to_string(),
                expected,
                "NpuAdapterType::Display mismatch"
            );
        }
    }

    // -----------------------------------------------------------------------
    // NpuInfo
    // -----------------------------------------------------------------------

    #[test]
    fn npu_info_display_format() {
        let info = NpuInfo {
            name: "Intel(R) AI Boost".to_string(),
            vendor: NpuVendor::Intel,
            adapter_type: NpuAdapterType::IntegratedNpu,
            backend: "Dx12".to_string(),
            adapter_index: 2,
            pci_vendor_id: 0x8086,
            pci_device_id: 0x7D1D,
        };
        let display = info.to_string();
        assert!(display.contains("Intel(R) AI Boost"), "must contain name");
        assert!(display.contains("Intel"), "must contain vendor");
        assert!(display.contains("IntegratedNpu"), "must contain type");
        assert!(display.contains("Dx12"), "must contain backend");
    }

    #[test]
    fn npu_info_debug_non_empty() {
        let info = NpuInfo {
            name: "Test NPU".to_string(),
            vendor: NpuVendor::Unknown,
            adapter_type: NpuAdapterType::Unknown,
            backend: "Vulkan".to_string(),
            adapter_index: 0,
            pci_vendor_id: 0,
            pci_device_id: 0,
        };
        assert!(!format!("{info:?}").is_empty());
    }

    #[test]
    fn npu_info_clone() {
        let info = NpuInfo {
            name: "AMD NPU".to_string(),
            vendor: NpuVendor::Amd,
            adapter_type: NpuAdapterType::IntegratedNpu,
            backend: "Vulkan".to_string(),
            adapter_index: 1,
            pci_vendor_id: 0x1002,
            pci_device_id: 0x1900,
        };
        let cloned = info.clone();
        assert_eq!(cloned.name, info.name);
        assert_eq!(cloned.adapter_index, info.adapter_index);
    }

    // -----------------------------------------------------------------------
    // classify_vendor
    // -----------------------------------------------------------------------

    #[test]
    fn classify_vendor_by_pci_intel() {
        let v = classify_vendor("some device", 0x8086);
        assert_eq!(v, NpuVendor::Intel);
    }

    #[test]
    fn classify_vendor_by_pci_amd() {
        let v = classify_vendor("some device", 0x1002);
        assert_eq!(v, NpuVendor::Amd);
    }

    #[test]
    fn classify_vendor_by_pci_qualcomm() {
        let v = classify_vendor("some device", 0x17CB);
        assert_eq!(v, NpuVendor::Qualcomm);
        let v2 = classify_vendor("some device", 0x5143);
        assert_eq!(v2, NpuVendor::Qualcomm);
    }

    #[test]
    fn classify_vendor_by_pci_apple() {
        let v = classify_vendor("some device", 0x106B);
        assert_eq!(v, NpuVendor::Apple);
    }

    #[test]
    fn classify_vendor_by_name_qualcomm_snapdragon() {
        let v = classify_vendor("qualcomm snapdragon x elite npu", 0);
        assert_eq!(v, NpuVendor::Qualcomm);
    }

    #[test]
    fn classify_vendor_by_name_amd_xdna() {
        let v = classify_vendor("amd xdna npu", 0);
        assert_eq!(v, NpuVendor::Amd);
    }

    #[test]
    fn classify_vendor_unknown_pci_and_unknown_name() {
        let v = classify_vendor("mystery accelerator", 0x9999);
        assert_eq!(v, NpuVendor::Unknown);
    }

    // -----------------------------------------------------------------------
    // classify_adapter_type
    // -----------------------------------------------------------------------

    #[test]
    fn classify_adapter_type_discrete_gpu() {
        let t = classify_adapter_type(wgpu::DeviceType::DiscreteGpu);
        assert_eq!(t, NpuAdapterType::DiscreteNpu);
    }

    #[test]
    fn classify_adapter_type_integrated_gpu() {
        let t = classify_adapter_type(wgpu::DeviceType::IntegratedGpu);
        assert_eq!(t, NpuAdapterType::IntegratedNpu);
    }

    #[test]
    fn classify_adapter_type_virtual_gpu() {
        let t = classify_adapter_type(wgpu::DeviceType::VirtualGpu);
        assert_eq!(t, NpuAdapterType::Virtual);
    }

    #[test]
    fn classify_adapter_type_cpu() {
        let t = classify_adapter_type(wgpu::DeviceType::Cpu);
        assert_eq!(t, NpuAdapterType::Software);
    }

    #[test]
    fn classify_adapter_type_other() {
        let t = classify_adapter_type(wgpu::DeviceType::Other);
        assert_eq!(t, NpuAdapterType::Unknown);
    }

    // -----------------------------------------------------------------------
    // NPU keyword list checks
    // -----------------------------------------------------------------------

    #[test]
    fn npu_name_keywords_are_lowercase() {
        for &kw in NPU_NAME_KEYWORDS {
            assert_eq!(kw, kw.to_lowercase(), "keyword '{kw}' must be lowercase");
        }
    }

    #[test]
    fn npu_name_keywords_are_non_empty() {
        for &kw in NPU_NAME_KEYWORDS {
            assert!(!kw.is_empty(), "NPU keyword must not be empty");
        }
    }

    // -----------------------------------------------------------------------
    // detect_npu (integration test — graceful on machines without NPU)
    // -----------------------------------------------------------------------

    /// Verify that detect_npu() never panics, regardless of hardware.
    #[test]
    fn detect_npu_does_not_panic() {
        let _result = detect_npu(); // must not panic
    }

    /// Verify that detect_npu() returns a valid NpuInfo when it succeeds.
    ///
    /// On machines without an NPU, this returns None — which is also valid.
    #[test]
    fn detect_npu_returns_valid_info_when_some() {
        if let Some(info) = detect_npu() {
            assert!(!info.name.is_empty(), "NpuInfo.name must not be empty");
            assert!(
                !info.backend.is_empty(),
                "NpuInfo.backend must not be empty"
            );
            // The adapter_index must be reachable (< total adapter count).
            let instance = Instance::new(InstanceDescriptor {
                backends: Backends::all(),
                flags: InstanceFlags::default(),
                ..Default::default()
            });
            let adapters: Vec<Adapter> = instance.enumerate_adapters(Backends::all());
            assert!(
                (info.adapter_index as usize) < adapters.len(),
                "adapter_index {} must be < adapter count {}",
                info.adapter_index,
                adapters.len()
            );
        }
        // None is a valid result — no assertion needed.
    }

    // -----------------------------------------------------------------------
    // detect_all_npus
    // -----------------------------------------------------------------------

    #[test]
    fn detect_all_npus_does_not_panic() {
        let _results = detect_all_npus();
    }

    #[test]
    fn detect_all_npus_consistent_with_detect_npu() {
        let all = detect_all_npus();
        let best = detect_npu();

        match (all.is_empty(), best) {
            (true, None) => {} // both found nothing — consistent
            (true, Some(_)) => {
                panic!("detect_all_npus returned empty but detect_npu returned Some")
            }
            (false, None) => panic!("detect_all_npus returned items but detect_npu returned None"),
            (false, Some(b)) => {
                // The first element in all must match detect_npu's result.
                assert_eq!(
                    all[0].adapter_index, b.adapter_index,
                    "First element of detect_all_npus must match detect_npu"
                );
                assert_eq!(
                    all[0].name, b.name,
                    "Names must match between detect_all_npus[0] and detect_npu"
                );
            }
        }
    }

    #[test]
    fn detect_all_npus_sorted_by_priority() {
        let results = detect_all_npus();
        // Software adapters (NpuAdapterType::Software) must come after hardware ones.
        let mut saw_hardware = false;
        let mut saw_software = false;
        for info in &results {
            let is_software = matches!(info.adapter_type, NpuAdapterType::Software);
            if !is_software {
                saw_hardware = true;
                assert!(
                    !saw_software,
                    "Hardware NPU appeared after software NPU — sort order violated"
                );
            } else {
                saw_software = true;
            }
        }
        // If we have both hardware and software, hardware must come first.
        // (This assertion is vacuously true if only one type exists.)
        let _ = saw_hardware;
    }

    #[test]
    fn detect_all_npus_adapter_indices_are_unique() {
        let results = detect_all_npus();
        let mut seen_indices = std::collections::HashSet::new();
        for info in &results {
            assert!(
                seen_indices.insert(info.adapter_index),
                "Duplicate adapter_index {} in detect_all_npus results",
                info.adapter_index
            );
        }
    }

    #[test]
    fn detect_all_npus_all_names_non_empty() {
        for info in detect_all_npus() {
            assert!(
                !info.name.is_empty(),
                "All NpuInfo.name fields must be non-empty"
            );
        }
    }

    // -----------------------------------------------------------------------
    // is_npu_adapter heuristic boundary tests
    // -----------------------------------------------------------------------

    /// These tests use real wgpu adapters (if available) to verify the
    /// heuristic does not falsely classify common GPU names.
    #[test]
    fn common_gpu_names_do_not_match_npu_heuristic() {
        let common_gpu_names = [
            "nvidia geforce rtx 4090",
            "amd radeon rx 7900 xtx",
            "intel arc a770",
            "apple m2 gpu",
            "llvmpipe (llvm 15.0.0, 256 bits)",
            "microsoft basic render driver",
            "warp software rasterizer",
            "swiftshader device (subzero)",
        ];
        for name in &common_gpu_names {
            let name_lower = name.to_lowercase();
            let matches = NPU_NAME_KEYWORDS.iter().any(|&kw| name_lower.contains(kw));
            // "Intel Arc" contains "arc" which is not in our keyword list.
            // None of these should match NPU keywords.
            assert!(
                !matches || name_lower.contains("npu") || name_lower.contains("neural"),
                "GPU name '{name}' should not match NPU heuristic"
            );
        }
    }

    /// Verify that known NPU name patterns do match the heuristic.
    #[test]
    fn known_npu_names_match_heuristic() {
        let npu_names = [
            "intel(r) ai boost",
            "qualcomm(r) ai accelerator",
            "amd npu",
            "intel npu accelerator",
            "hexagon npu",
            "neural processing unit",
            "vpu device",
        ];
        for name in &npu_names {
            let name_lower = name.to_lowercase();
            let matches = NPU_NAME_KEYWORDS.iter().any(|&kw| name_lower.contains(kw));
            assert!(
                matches,
                "Known NPU name '{name}' should match NPU keyword heuristic"
            );
        }
    }

    // -----------------------------------------------------------------------
    // NpuVendor — new variants
    // -----------------------------------------------------------------------

    #[test]
    fn npu_vendor_display_samsung() {
        assert_eq!(NpuVendor::Samsung.to_string(), "Samsung");
    }

    #[test]
    fn npu_vendor_display_mediatek() {
        assert_eq!(NpuVendor::MediaTek.to_string(), "MediaTek");
    }

    // -----------------------------------------------------------------------
    // is_npu_adapter heuristic — Samsung / MediaTek names
    // -----------------------------------------------------------------------

    #[test]
    fn samsung_exynos_name_matches_npu_heuristic() {
        assert!(
            is_npu_adapter("Exynos Neural Processing Unit"),
            "Exynos NPU name must match heuristic"
        );
        assert!(
            is_npu_adapter("Samsung NPU"),
            "Samsung NPU name must match heuristic"
        );
    }

    #[test]
    fn mediatek_apu_name_matches_npu_heuristic() {
        assert!(
            is_npu_adapter("MediaTek APU"),
            "MediaTek APU name must match heuristic"
        );
        assert!(
            is_npu_adapter("Dimensity APU"),
            "Dimensity APU name must match heuristic"
        );
    }

    // -----------------------------------------------------------------------
    // is_npu_adapter_extended — BITNET_NPU_ADAPTER env override
    // -----------------------------------------------------------------------

    #[test]
    fn env_override_adds_custom_keywords() {
        // An adapter whose name contains a user-supplied keyword must match.
        let extra = vec!["myvendor".to_string()];
        assert!(
            is_npu_adapter_extended("MyVendor SuperNPU 9000", &extra),
            "Adapter matching env override keyword must be detected"
        );
        // An adapter that matches neither built-in keywords nor the extra
        // keyword must not match.
        assert!(
            !is_npu_adapter_extended("Intel HD Graphics 4000", &extra),
            "Ordinary GPU with no matching keyword must not be detected as NPU"
        );
    }
}
