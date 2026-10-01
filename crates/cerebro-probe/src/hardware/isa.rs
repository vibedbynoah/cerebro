//! CPU instruction-set extensions that change which llama.cpp CPU kernels can run.
//!
//! On x86-64 an extension counts only when the CPU implements it **and** the OS has enabled its
//! register state (`XCR0`): AVX-512 or AMX reported by CPUID but not enabled by the OS faults on
//! first use. Decoding is a pure function of the raw CPUID registers and `XCR0`, so every rule is
//! tested against register values for CPUs Cerebro has never run on. Bit positions are from the
//! Intel SDM, volume 2A, `CPUID` (leaf 1, leaf 7 sub-leaves 0 and 1) and volume 1, 13.3 (`XCR0`).

use std::collections::BTreeSet;
use std::fmt;

/// One instruction-set extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Isa {
    /// SSE4.2.
    Sse42,
    /// AVX.
    Avx,
    /// AVX2.
    Avx2,
    /// FMA3.
    Fma,
    /// F16C half-precision conversion.
    F16c,
    /// AVX-512 Foundation.
    Avx512F,
    /// AVX-512 Byte and Word.
    Avx512Bw,
    /// AVX-512 Vector Length.
    Avx512Vl,
    /// AVX-512 VNNI (int8 dot products).
    Avx512Vnni,
    /// AVX-512 BF16.
    Avx512Bf16,
    /// AVX-VNNI (VEX-encoded VNNI, Alder Lake and newer).
    AvxVnni,
    /// AMX tile architecture.
    AmxTile,
    /// AMX int8.
    AmxInt8,
    /// AMX bf16.
    AmxBf16,
    /// Arm Advanced SIMD (NEON).
    Neon,
    /// Arm dot product (`SDOT`/`UDOT`).
    DotProd,
    /// Arm int8 matrix multiply (`SMMLA`).
    I8mm,
    /// Arm Scalable Vector Extension.
    Sve,
    /// Arm SVE2.
    Sve2,
}

impl Isa {
    /// Conventional name, e.g. `AVX-512F`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Sse42 => "SSE4.2",
            Self::Avx => "AVX",
            Self::Avx2 => "AVX2",
            Self::Fma => "FMA",
            Self::F16c => "F16C",
            Self::Avx512F => "AVX-512F",
            Self::Avx512Bw => "AVX-512BW",
            Self::Avx512Vl => "AVX-512VL",
            Self::Avx512Vnni => "AVX-512 VNNI",
            Self::Avx512Bf16 => "AVX-512 BF16",
            Self::AvxVnni => "AVX-VNNI",
            Self::AmxTile => "AMX-TILE",
            Self::AmxInt8 => "AMX-INT8",
            Self::AmxBf16 => "AMX-BF16",
            Self::Neon => "NEON",
            Self::DotProd => "DotProd",
            Self::I8mm => "I8MM",
            Self::Sve => "SVE",
            Self::Sve2 => "SVE2",
        }
    }
}

impl fmt::Display for Isa {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The extensions a CPU has, usable by this process.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IsaSet(BTreeSet<Isa>);

impl IsaSet {
    /// A set from a list (fixtures and the aarch64 detector).
    pub fn from_slice(items: &[Isa]) -> Self {
        Self(items.iter().copied().collect())
    }

    /// Whether `isa` is present.
    pub fn has(&self, isa: Isa) -> bool {
        self.0.contains(&isa)
    }

    /// Every extension present, in a stable order.
    pub fn iter(&self) -> impl Iterator<Item = Isa> + '_ {
        self.0.iter().copied()
    }
}

/// The raw x86 registers the decoder reads. Fields for leaves the CPU does not implement are 0.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuidRegs {
    /// `CPUID.1:ECX`.
    pub leaf1_ecx: u32,
    /// `CPUID.(7,0):EBX`.
    pub leaf7_ebx: u32,
    /// `CPUID.(7,0):ECX`.
    pub leaf7_ecx: u32,
    /// `CPUID.(7,0):EDX`.
    pub leaf7_edx: u32,
    /// `CPUID.(7,1):EAX`.
    pub leaf7_1_eax: u32,
    /// `XCR0` as read by `XGETBV(0)`; 0 when the OS has not set `OSXSAVE`.
    pub xcr0: u64,
}

/// `XCR0` bits 1 (SSE) and 2 (AVX upper halves): the OS saves AVX state.
const XCR0_AVX_STATE: u64 = 0b110;
/// `XCR0` bits 5-7 (opmask, upper ZMM0-15, ZMM16-31): the OS saves AVX-512 state.
const XCR0_AVX512_STATE: u64 = 0b1110_0000;
/// `XCR0` bits 17 (XTILECFG) and 18 (XTILEDATA): the OS saves AMX tile state.
const XCR0_AMX_STATE: u64 = 0b110 << 16;

const fn bit(reg: u32, n: u32) -> bool {
    reg & (1 << n) != 0
}

/// Decode the extensions a process may use from raw CPUID registers and `XCR0`.
pub fn decode_x86(r: &CpuidRegs) -> IsaSet {
    let mut set = BTreeSet::new();
    let osxsave = bit(r.leaf1_ecx, 27);
    let os_avx = osxsave && r.xcr0 & XCR0_AVX_STATE == XCR0_AVX_STATE;
    let os_avx512 = os_avx && r.xcr0 & XCR0_AVX512_STATE == XCR0_AVX512_STATE;
    let os_amx = osxsave && r.xcr0 & XCR0_AMX_STATE == XCR0_AMX_STATE;

    let mut add = |cond: bool, isa: Isa| {
        if cond {
            set.insert(isa);
        }
    };
    add(bit(r.leaf1_ecx, 20), Isa::Sse42);
    add(os_avx && bit(r.leaf1_ecx, 28), Isa::Avx);
    add(os_avx && bit(r.leaf1_ecx, 12), Isa::Fma);
    add(os_avx && bit(r.leaf1_ecx, 29), Isa::F16c);
    add(os_avx && bit(r.leaf7_ebx, 5), Isa::Avx2);
    add(os_avx && bit(r.leaf7_1_eax, 4), Isa::AvxVnni);
    add(os_avx512 && bit(r.leaf7_ebx, 16), Isa::Avx512F);
    add(os_avx512 && bit(r.leaf7_ebx, 30), Isa::Avx512Bw);
    add(os_avx512 && bit(r.leaf7_ebx, 31), Isa::Avx512Vl);
    add(os_avx512 && bit(r.leaf7_ecx, 11), Isa::Avx512Vnni);
    add(os_avx512 && bit(r.leaf7_1_eax, 5), Isa::Avx512Bf16);
    add(os_amx && bit(r.leaf7_edx, 24), Isa::AmxTile);
    add(os_amx && bit(r.leaf7_edx, 25), Isa::AmxInt8);
    add(os_amx && bit(r.leaf7_edx, 22), Isa::AmxBf16);
    IsaSet(set)
}

/// Read the registers [`decode_x86`] needs from this CPU.
#[cfg(target_arch = "x86_64")]
pub fn read_cpuid() -> CpuidRegs {
    use std::arch::x86_64::{__cpuid, __cpuid_count};

    let max_leaf = __cpuid(0).eax;
    let leaf1 = __cpuid(1);
    let mut regs = CpuidRegs {
        leaf1_ecx: leaf1.ecx,
        ..CpuidRegs::default()
    };
    if max_leaf >= 7 {
        let leaf7 = __cpuid_count(7, 0);
        regs.leaf7_ebx = leaf7.ebx;
        regs.leaf7_ecx = leaf7.ecx;
        regs.leaf7_edx = leaf7.edx;
        // Leaf 7 sub-leaf 0 EAX is the highest sub-leaf implemented.
        if leaf7.eax >= 1 {
            regs.leaf7_1_eax = __cpuid_count(7, 1).eax;
        }
    }
    if bit(leaf1.ecx, 27) {
        regs.xcr0 = read_xcr0();
    }
    regs
}

/// `XGETBV(0)`. Called only when `CPUID.1:ECX.OSXSAVE` is set, which is the architectural
/// guarantee that the instruction exists and is enabled.
#[cfg(target_arch = "x86_64")]
fn read_xcr0() -> u64 {
    #[target_feature(enable = "xsave")]
    unsafe fn xgetbv0() -> u64 {
        std::arch::x86_64::_xgetbv(0)
    }
    // SAFETY: the only caller checked CPUID.1:ECX bit 27 (OSXSAVE), which means the CPU supports
    // XSAVE and the OS has set CR4.OSXSAVE, so XGETBV executes without faulting; that is the one
    // requirement the `xsave` target feature places on the call.
    unsafe { xgetbv0() }
}

/// The extensions this process may use on this CPU.
#[cfg(target_arch = "x86_64")]
pub fn detect() -> IsaSet {
    decode_x86(&read_cpuid())
}

/// The extensions this process may use on this CPU. NEON is architecturally mandatory on
/// AArch64; the rest come from the OS's feature report.
#[cfg(target_arch = "aarch64")]
pub fn detect() -> IsaSet {
    let mut items = vec![Isa::Neon];
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        items.push(Isa::DotProd);
    }
    if std::arch::is_aarch64_feature_detected!("i8mm") {
        items.push(Isa::I8mm);
    }
    if std::arch::is_aarch64_feature_detected!("sve") {
        items.push(Isa::Sve);
    }
    if std::arch::is_aarch64_feature_detected!("sve2") {
        items.push(Isa::Sve2);
    }
    IsaSet::from_slice(&items)
}

/// No detector for this architecture: an empty set, which the caller reports as unknown.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn detect() -> IsaSet {
    IsaSet::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// XCR0 with x87, SSE and AVX state enabled (no AVX-512, no AMX).
    const XCR0_AVX: u64 = 0b111;
    /// Plus AVX-512 state.
    const XCR0_AVX512: u64 = 0b1110_0111;
    /// Plus AMX tile state.
    const XCR0_AMX: u64 = 0b1110_0111 | 0x6_0000;

    const OSXSAVE: u32 = 1 << 27;
    const LEAF1_AVX_FAMILY: u32 = (1 << 20) | (1 << 28) | (1 << 12) | (1 << 29) | OSXSAVE;
    const LEAF7_AVX2: u32 = 1 << 5;
    const LEAF7_AVX512_FBWVL: u32 = (1 << 16) | (1 << 30) | (1 << 31);

    /// Alder Lake (i9-12900H, the development machine): AVX2 + AVX-VNNI, AVX-512 fused off.
    fn alder_lake() -> CpuidRegs {
        CpuidRegs {
            leaf1_ecx: LEAF1_AVX_FAMILY,
            leaf7_ebx: LEAF7_AVX2,
            leaf7_ecx: 0,
            leaf7_edx: 0,
            leaf7_1_eax: 1 << 4,
            xcr0: XCR0_AVX,
        }
    }

    #[test]
    fn alder_lake_has_avx2_and_avx_vnni_but_no_avx512() {
        let set = decode_x86(&alder_lake());
        for isa in [Isa::Sse42, Isa::Avx, Isa::Avx2, Isa::Fma, Isa::F16c, Isa::AvxVnni] {
            assert!(set.has(isa), "{isa}");
        }
        assert!(!set.has(Isa::Avx512F));
        assert!(!set.has(Isa::AmxInt8));
    }

    #[test]
    fn zen4_has_avx512_with_vnni_and_bf16() {
        let set = decode_x86(&CpuidRegs {
            leaf1_ecx: LEAF1_AVX_FAMILY,
            leaf7_ebx: LEAF7_AVX2 | LEAF7_AVX512_FBWVL,
            leaf7_ecx: 1 << 11,
            leaf7_edx: 0,
            leaf7_1_eax: 1 << 5,
            xcr0: XCR0_AVX512,
        });
        for isa in [Isa::Avx512F, Isa::Avx512Bw, Isa::Avx512Vl, Isa::Avx512Vnni, Isa::Avx512Bf16] {
            assert!(set.has(isa), "{isa}");
        }
        assert!(!set.has(Isa::AvxVnni));
    }

    #[test]
    fn sapphire_rapids_amx_needs_os_tile_state() {
        let regs = CpuidRegs {
            leaf1_ecx: LEAF1_AVX_FAMILY,
            leaf7_ebx: LEAF7_AVX2 | LEAF7_AVX512_FBWVL,
            leaf7_ecx: 1 << 11,
            leaf7_edx: (1 << 22) | (1 << 24) | (1 << 25),
            leaf7_1_eax: (1 << 4) | (1 << 5),
            xcr0: XCR0_AMX,
        };
        let set = decode_x86(&regs);
        for isa in [Isa::AmxTile, Isa::AmxInt8, Isa::AmxBf16, Isa::Avx512F] {
            assert!(set.has(isa), "{isa}");
        }
        // Same CPU, OS without AMX tile state: AMX is not usable, AVX-512 still is.
        let set = decode_x86(&CpuidRegs { xcr0: XCR0_AVX512, ..regs });
        assert!(!set.has(Isa::AmxInt8));
        assert!(set.has(Isa::Avx512F));
    }

    #[test]
    fn cpu_support_without_os_state_is_not_support() {
        // AVX-512 reported by CPUID, but the OS enabled only AVX state.
        let set = decode_x86(&CpuidRegs {
            leaf7_ebx: LEAF7_AVX2 | LEAF7_AVX512_FBWVL,
            ..alder_lake()
        });
        assert!(set.has(Isa::Avx2));
        assert!(!set.has(Isa::Avx512F));
        // No OSXSAVE at all: nothing that needs extended state.
        let set = decode_x86(&CpuidRegs {
            leaf1_ecx: LEAF1_AVX_FAMILY & !OSXSAVE,
            ..alder_lake()
        });
        assert!(set.has(Isa::Sse42));
        assert!(!set.has(Isa::Avx));
        assert!(!set.has(Isa::Avx2));
    }

    /// The Arm detector on a Snapdragon X (Oryon, Armv8.6+): DotProd and I8MM must both be reported,
    /// or the reference profile's I8MM (and the matrix's Arm I8MM kernels row) does not describe
    /// what the live probe returns there. On Windows the standard library learns these from
    /// `IsProcessorFeaturePresent`; whether it queries the I8MM constant is exactly what this test
    /// checks on real hardware.
    #[cfg(all(windows, target_arch = "aarch64"))]
    #[test]
    #[ignore = "requires a Snapdragon X (Oryon) Windows-on-Arm machine and an aarch64 build"]
    fn live_windows_on_arm_reports_dotprod_and_i8mm() {
        let set = detect();
        assert!(set.has(Isa::Neon));
        assert!(set.has(Isa::DotProd), "{set:?}");
        assert!(set.has(Isa::I8mm), "{set:?}");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn live_detection_agrees_with_the_standard_library() {
        let set = detect();
        assert_eq!(set.has(Isa::Avx2), std::arch::is_x86_feature_detected!("avx2"));
        assert_eq!(set.has(Isa::Fma), std::arch::is_x86_feature_detected!("fma"));
        assert_eq!(set.has(Isa::Avx512F), std::arch::is_x86_feature_detected!("avx512f"));
        assert_eq!(set.has(Isa::Avx512Vnni), std::arch::is_x86_feature_detected!("avx512vnni"));
    }
}
