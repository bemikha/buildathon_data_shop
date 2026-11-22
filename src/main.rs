// Shopping Sessions pipeline (v3).
//
// Stages (all parallel with rayon, data moved between stages by hash partitioning
// so every lookup table stays in cache):
//   links   -> union-find over interned ids, person = byte-wise smallest id
//   events  -> SIMD CSV scan, validate per copy, partition by bucket(event_id)
//   dedup   -> per bucket keep smallest ingest_ts, drop invalid, partition by bucket(user_id)
//   clicks  -> scan, validate, partition by bucket(user_id)
//   resolve -> per user bucket map user_id -> person (partition join), partition by bucket(person)
//   people  -> per person bucket sort, sessionize, attribute, aggregate
//   write   -> parquet
use ahash::AHashMap;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use rayon::prelude::*;
use std::cmp::Ordering;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

type B = &'static [u8];
/// A string packed into 128 bits as (hi, lo). Strings of up to 15 bytes are stored
/// big-endian, left-aligned, with their length in the lowest byte, so tuple order
/// equals byte-wise string order. Longer strings get a 120-bit hash and the low byte
/// 0xFF; their bytes are kept in LONG when order or output needs them.
type K = (u64, u64);

const DAY_MS: i64 = 86_400_000;
const GAP_MS: i64 = 1_800_000;
const WEEK_MS: i64 = 7 * DAY_MS;
const NB_BITS: u32 = 10;
const NB: usize = 1 << NB_BITS;

/// Bytes of long keys, sharded by hash so threads rarely contend. Values point into the
/// input mappings, which live until exit.
const LSH: usize = 256;
static LONG: OnceLock<Vec<Mutex<AHashMap<K, B>>>> = OnceLock::new();

fn long_shard(k: K) -> &'static Mutex<AHashMap<K, B>> {
    let t = LONG.get_or_init(|| (0..LSH).map(|_| Mutex::new(AHashMap::new())).collect());
    &t[(k.0 >> 56) as usize % LSH]
}

thread_local! {
    /// Direct-mapped cache of long keys this thread already registered.
    static LONG_SEEN: RefCell<Vec<K>> = RefCell::new(vec![(0, 0); 4096]);
}

/// 128-bit hash of a byte string, one pass, two independent multiply-rotate lanes.
#[inline]
fn hash128(b: &[u8]) -> (u64, u64) {
    let mut a = 0x243F_6A88_85A3_08D3u64 ^ b.len() as u64;
    let mut c = 0x1319_8A2E_0370_7344u64 ^ (b.len() as u64).rotate_left(32);
    let mut i = 0;
    while i + 8 <= b.len() {
        let w = u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        a = (a ^ w).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29);
        c = (c ^ w).wrapping_mul(0xC2B2_AE3D_27D4_EB4F).rotate_left(31);
        i += 8;
    }
    if i < b.len() {
        let mut t = [0u8; 8];
        t[..b.len() - i].copy_from_slice(&b[i..]);
        let w = u64::from_le_bytes(t) ^ 0x8000_0000_0000_0000;
        a = (a ^ w).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29);
        c = (c ^ w).wrapping_mul(0xC2B2_AE3D_27D4_EB4F).rotate_left(31);
    }
    let f = |mut x: u64| {
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
        x ^= x >> 33;
        x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        x ^ (x >> 33)
    };
    let (fa, fc) = (f(a ^ c.rotate_left(17)), f(c ^ a.rotate_left(41)));
    (fa, fc)
}

#[inline(always)]
fn pack(b: &[u8], keep: bool) -> K {
    let len = b.len();
    if len <= 15 {
        let ptr = b.as_ptr();
        let v = if len == 0 {
            0u128
        } else if (ptr as usize & 4095) <= 4096 - 16 {
            // The 16-byte read stays inside the page holding the field, so it is mapped.
            let raw = unsafe { std::ptr::read_unaligned(ptr as *const [u8; 16]) };
            u128::from_be_bytes(raw) & (!0u128 << (8 * (16 - len)))
        } else {
            let mut buf = [0u8; 16];
            buf[..len].copy_from_slice(b);
            u128::from_be_bytes(buf)
        };
        let v = v | len as u128;
        ((v >> 64) as u64, v as u64)
    } else {
        pack_long(b, keep)
    }
}

#[inline(never)]
fn pack_long(b: &[u8], keep: bool) -> K {
    let (h1, h2) = hash128(b);
    let k = (h1, h2 | 0xFF);
    if keep {
        // Fields are slices of the input mappings, which are never unmapped.
        let st: B = unsafe { std::slice::from_raw_parts(b.as_ptr(), b.len()) };
        LONG_SEEN.with(|s| {
            let mut s = s.borrow_mut();
            let slot = (k.0 >> 52) as usize & 4095;
            if s[slot] != k {
                s[slot] = k;
                long_shard(k).lock().unwrap().entry(k).or_insert(st);
            }
        });
    }
    k
}

#[inline(always)]
fn is_long(k: K) -> bool {
    k.1 & 0xFF >= 0xFE
}

/// A click_id: ids up to 15 bytes are packed; a longer one keeps a pointer to its bytes
/// in the input mapping (low byte 0xFE). Click ids are only ever compared, never hashed.
#[inline(always)]
fn pack_cid(b: &[u8]) -> K {
    if b.len() <= 15 {
        pack(b, false)
    } else {
        (b.as_ptr() as u64, (b.len() as u64) << 8 | 0xFE)
    }
}

fn long_bytes(k: K) -> B {
    if k.1 & 0xFF == 0xFE {
        return unsafe { std::slice::from_raw_parts(k.0 as *const u8, (k.1 >> 8) as usize) };
    }
    long_shard(k).lock().unwrap().get(&k).copied().unwrap_or(&[])
}

fn key_bytes(k: K) -> Vec<u8> {
    if is_long(k) {
        long_bytes(k).to_vec()
    } else {
        let v = ((k.0 as u128) << 64) | k.1 as u128;
        let buf = v.to_be_bytes();
        buf[..buf[15] as usize].to_vec()
    }
}

#[inline(always)]
fn cmp_key(a: K, b: K) -> Ordering {
    if !is_long(a) && !is_long(b) {
        a.cmp(&b)
    } else if is_long(a) && is_long(b) {
        long_bytes(a).cmp(long_bytes(b))
    } else {
        key_bytes(a).cmp(&key_bytes(b))
    }
}

#[inline(always)]
fn mix(k: K) -> u64 {
    let mut x = k.1 ^ k.0.rotate_left(23).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 31;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 29;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 32)
}

#[inline(always)]
fn bucket(k: K) -> usize {
    (mix(k) >> (64 - NB_BITS)) as usize
}

#[inline(always)]
fn h32(k: K) -> u64 {
    mix(k) & 0xFFFF_FFFF
}

// ---------------------------------------------------------------- files

fn map_file(p: &Path) -> B {
    let f = std::fs::File::open(p).unwrap_or_else(|e| panic!("open {p:?}: {e}"));
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    if len == 0 {
        return &[];
    }
    let m = unsafe { memmap2::MmapOptions::new().map(&f) }.unwrap_or_else(|e| panic!("mmap {p:?}: {e}"));
    // No madvise(WillNeed): the parallel scan faults pages in as it goes.
    let m: &'static memmap2::Mmap = Box::leak(Box::new(m));
    &m[..]
}

fn csv_files(dir: &Path) -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().map(|x| x == "csv").unwrap_or(false) && p.is_file() {
                v.push(p);
            }
        }
    }
    v.sort();
    v
}

/// Bodies of CSV files (after the header line), in chunks that end on line boundaries.
fn chunks_of(files: &[PathBuf], target: usize) -> Vec<B> {
    let maps: Vec<B> = files.par_iter().map(|p| map_file(p)).collect();
    let mut out = Vec::new();
    for data in maps {
        let body = match memchr::memchr(b'\n', data) {
            Some(i) => &data[i + 1..],
            None => continue,
        };
        let mut s = body;
        while !s.is_empty() {
            if s.len() <= target {
                out.push(s);
                break;
            }
            let cut = match memchr::memchr(b'\n', &s[target..]) {
                Some(i) => target + i + 1,
                None => s.len(),
            };
            out.push(&s[..cut]);
            s = &s[cut..];
        }
    }
    out
}

// ---------------------------------------------------------------- SIMD line scanner

#[inline(always)]
fn masks64(p: &[u8; 64]) -> (u64, u64, u64) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    unsafe {
        use std::arch::x86_64::*;
        let v = _mm512_loadu_si512(p.as_ptr() as *const __m512i);
        return (
            _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'\n' as i8)),
            _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b',' as i8)),
            _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'"' as i8)),
        );
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512bw")))]
    unsafe {
        use std::arch::x86_64::*;
        let nl = _mm256_set1_epi8(b'\n' as i8);
        let cm = _mm256_set1_epi8(b',' as i8);
        let qt = _mm256_set1_epi8(b'"' as i8);
        let v0 = _mm256_loadu_si256(p.as_ptr() as *const __m256i);
        let v1 = _mm256_loadu_si256(p.as_ptr().add(32) as *const __m256i);
        let m = |a: __m256i, b: __m256i| {
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(a, b)) as u32 as u64
        };
        return (
            m(v0, nl) | m(v1, nl) << 32,
            m(v0, cm) | m(v1, cm) << 32,
            m(v0, qt) | m(v1, qt) << 32,
        );
    }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    unsafe {
        use std::arch::x86_64::*;
        let nl = _mm_set1_epi8(b'\n' as i8);
        let cm = _mm_set1_epi8(b',' as i8);
        let qt = _mm_set1_epi8(b'"' as i8);
        let (mut a, mut b, mut c) = (0u64, 0u64, 0u64);
        for i in 0..4 {
            let v = _mm_loadu_si128(p.as_ptr().add(i * 16) as *const __m128i);
            a |= (_mm_movemask_epi8(_mm_cmpeq_epi8(v, nl)) as u16 as u64) << (i * 16);
            b |= (_mm_movemask_epi8(_mm_cmpeq_epi8(v, cm)) as u16 as u64) << (i * 16);
            c |= (_mm_movemask_epi8(_mm_cmpeq_epi8(v, qt)) as u16 as u64) << (i * 16);
        }
        (a, b, c)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let (mut a, mut b, mut c) = (0u64, 0u64, 0u64);
        for (i, &x) in p.iter().enumerate() {
            a |= ((x == b'\n') as u64) << i;
            b |= ((x == b',') as u64) << i;
            c |= ((x == b'"') as u64) << i;
        }
        (a, b, c)
    }
}

/// Calls `f` with the N fields of every well-formed line of `chunk` (empty slice = null).
/// Only `\n` ends a line; a field is either free of quotes or wrapped in exactly one pair.
#[inline(always)]
fn scan<const N: usize>(chunk: &[u8], mut f: impl FnMut(&[&[u8]; N])) {
    let len = chunk.len();
    if len == 0 {
        return;
    }
    let base = chunk.as_ptr();
    // one spare slot absorbs extra fields of malformed lines
    let mut fs: [(u32, u32); 8] = [(0, 0); 8];
    assert!(N < 8);
    let mut nf = 0usize;
    let mut bad = false;
    let mut fstart = 0usize;
    let mut start_q = 0u32;
    let mut qbase = 0u32;
    let mut pos = 0usize;
    #[inline(always)]
    fn close(base: *const u8, len: usize, fstart: usize, p: usize, q: u32, fs: &mut [(u32, u32); 8], nf: &mut usize, bad: &mut bool, n: usize) {
        let flen = p - fstart;
        let first = unsafe { *base.add(fstart.min(len - 1)) };
        let last = unsafe { *base.add(p.saturating_sub(1)) };
        let wrapped = (q == 2) & (flen >= 2) & (first == b'"') & (last == b'"');
        *bad |= !((q == 0) | wrapped) | (*nf >= n);
        let t = wrapped as usize;
        let k = (*nf).min(7);
        fs[k] = ((fstart + t) as u32, (p - t) as u32);
        *nf += 1;
    }
    let emit = |fs: &[(u32, u32); 8], f: &mut dyn FnMut(&[&[u8]; N])| {
        let mut out: [&[u8]; N] = [&[]; N];
        for i in 0..N {
            out[i] = unsafe { std::slice::from_raw_parts(base.add(fs[i].0 as usize), (fs[i].1 - fs[i].0) as usize) };
        }
        f(&out);
    };
    let mut fdyn = |x: &[&[u8]; N]| f(x);
    while pos < len {
        let (nl, cm, qt) = if pos + 64 <= len {
            masks64(chunk[pos..pos + 64].try_into().unwrap())
        } else {
            let mut buf = [0u8; 64];
            buf[..len - pos].copy_from_slice(&chunk[pos..]);
            masks64(&buf)
        };
        let mut st = nl | cm;
        while st != 0 {
            let bit = st.trailing_zeros();
            let p = pos + bit as usize;
            let qcum = qbase + (qt & ((1u64 << bit) - 1)).count_ones();
            close(base, len, fstart, p, qcum - start_q, &mut fs, &mut nf, &mut bad, N);
            fstart = p + 1;
            start_q = qcum;
            if (nl >> bit) & 1 != 0 {
                if !bad && nf == N {
                    emit(&fs, &mut fdyn);
                }
                nf = 0;
                bad = false;
            }
            st &= st - 1;
        }
        qbase += qt.count_ones();
        pos += 64;
    }
    if fstart < len || nf > 0 {
        close(base, len, fstart, len, qbase - start_q, &mut fs, &mut nf, &mut bad, N);
        if !bad && nf == N {
            emit(&fs, &mut fdyn);
        }
    }
}


/// Fast line scanner. Per window of whole lines (up to 64 KiB): one pass computes
/// newline/comma/quote masks per 64-byte block, flattens all delimiter positions and
/// records, for every newline, its index among the delimiters and the number of
/// quotes before it. A line is then kept iff it has exactly N-1 commas and its quote
/// count equals 2 x the number of fields wrapped in quotes (so no other quote occurs).
/// Lines longer than a window and a final line without newline use `scan`.
fn scan_lines<const N: usize>(chunk: &[u8], mut f: impl FnMut(&[&[u8]; N])) {
    let len = chunk.len();
    if len == 0 {
        return;
    }
    SCAN_BUF.with(|cell| {
    let mut guard = cell.borrow_mut();
    let (pos, nls) = &mut *guard;
    if pos.len() < WIN + 72 {
        pos.resize(WIN + 72, 0);
        nls.resize(WIN + 1, (0, 0));
    }
    let mut start = 0usize;
    while start < len {
        let lim = (start + WIN).min(len);
        let wend = if lim == len {
            len
        } else {
            match memchr::memrchr(b'\n', &chunk[start..lim]) {
                Some(i) => start + i + 1,
                None => {
                    let e = match memchr::memchr(b'\n', &chunk[start..]) {
                        Some(i) => start + i + 1,
                        None => len,
                    };
                    scan::<N>(&chunk[start..e], &mut f);
                    start = e;
                    continue;
                }
            }
        };
        let w = &chunk[start..wend];
        let wl = w.len();
        let mut np = 0usize;
        let mut nn = 0usize;
        let mut qacc = 0u32;
        let mut off = 0usize;
        while off < wl {
            let (nl, cm, qt) = if off + 64 <= wl {
                masks64(w[off..off + 64].try_into().unwrap())
            } else {
                let mut buf = [0u8; 64];
                buf[..wl - off].copy_from_slice(&w[off..]);
                masks64(&buf)
            };
            let st = nl | cm;
            let mut nm = nl;
            while nm != 0 {
                let below = (nm & nm.wrapping_neg()) - 1;
                unsafe {
                    *nls.get_unchecked_mut(nn) = ((np as u32) + (st & below).count_ones(), qacc + (qt & below).count_ones());
                }
                nn += 1;
                nm &= nm - 1;
            }
            let mut m = st;
            let mut k = np;
            while m != 0 {
                for _ in 0..4 {
                    unsafe { *pos.get_unchecked_mut(k) = (off as u32) + m.trailing_zeros() };
                    m &= m.wrapping_sub(1);
                    k += 1;
                }
            }
            np += st.count_ones() as usize;
            qacc += qt.count_ones();
            off += 64;
        }
        let mut a = 0usize; // index of the line's first delimiter
        let mut ls = 0usize; // line start
        let mut pq = 0u32; // quotes before the line
        for &(ni, q) in &nls[..nn] {
            let ni = ni as usize;
            if ni - a == N - 1 {
                let lq = q - pq;
                let mut out: [&[u8]; N] = [&[]; N];
                let mut s0 = ls;
                let mut wrapped = 0u32;
                for (j, o) in out.iter_mut().enumerate() {
                    let e0 = unsafe { *pos.get_unchecked(a + j) } as usize;
                    let fld = unsafe { w.get_unchecked(s0..e0) };
                    if lq != 0 && fld.len() >= 2 && fld[0] == b'"' && fld[fld.len() - 1] == b'"' {
                        wrapped += 1;
                        *o = &fld[1..fld.len() - 1];
                    } else {
                        *o = fld;
                    }
                    s0 = e0 + 1;
                }
                if lq == 2 * wrapped {
                    f(&out);
                }
            }
            ls = unsafe { *pos.get_unchecked(ni) } as usize + 1;
            a = ni + 1;
            pq = q;
        }
        if ls < wl {
            scan::<N>(&w[ls..], &mut f);
        }
        start = wend;
    }
    });
}

const WIN: usize = 1 << 14;
thread_local! {
    static SCAN_BUF: RefCell<(Vec<u32>, Vec<(u32, u32)>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

// ---------------------------------------------------------------- timestamps

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

#[inline(always)]
fn d2(b: &[u8], i: usize) -> Option<i64> {
    let (a, c) = (b[i].wrapping_sub(b'0'), b[i + 1].wrapping_sub(b'0'));
    if a > 9 || c > 9 {
        return None;
    }
    Some((a * 10 + c) as i64)
}

/// Validated `YYYY-MM-DD` -> days since epoch, through a small direct-mapped cache.
struct Dates {
    tags: [(u64, u16); 64],
    days: [i64; 64],
}
impl Dates {
    fn new() -> Self {
        Dates { tags: [(0, 0); 64], days: [0; 64] }
    }
    #[inline(always)]
    fn get(&mut self, b: &[u8]) -> Option<i64> {
        let w = u64::from_le_bytes(b[0..8].try_into().unwrap());
        let t2 = u16::from_le_bytes([b[8], b[9]]);
        let slot = ((w ^ (t2 as u64) << 40).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 58) as usize;
        if self.tags[slot] == (w, t2) {
            return Some(self.days[slot]);
        }
        let days = date_slow(b)?;
        self.tags[slot] = (w, t2);
        self.days[slot] = days;
        Some(days)
    }
}

fn date_slow(b: &[u8]) -> Option<i64> {
    if b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let y = d2(b, 0)? * 100 + d2(b, 2)?;
    let mo = d2(b, 5)?;
    let d = d2(b, 8)?;
    if !(1..=12).contains(&mo) || d < 1 {
        return None;
    }
    let dim = match mo {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 {
                29
            } else {
                28
            }
        }
    };
    if d > dim {
        return None;
    }
    Some(days_from_civil(y, mo, d))
}

#[allow(dead_code)]
fn parse_ts_ref(b: &[u8], dc: &mut Dates) -> Option<i64> {
    if !((b.len() == 20 || b.len() == 25) && b[4] == b'-') {
        return parse_f3(b);
    }
    let off_ms = if b.len() == 20 {
        if b[19] != b'Z' {
            return None;
        }
        0
    } else {
        let sign = match b[19] {
            b'+' => 1,
            b'-' => -1,
            _ => return None,
        };
        if b[22] != b':' {
            return None;
        }
        let oh = d2(b, 20)?;
        let om = d2(b, 23)?;
        if oh > 23 || om > 59 {
            return None;
        }
        sign * (oh * 60 + om) * 60_000
    };
    if b[7] != b'-' || b[10] != b'T' || b[13] != b':' || b[16] != b':' {
        return None;
    }
    let h = d2(b, 11)?;
    let mi = d2(b, 14)?;
    let s = d2(b, 17)?;
    if h > 23 || mi > 59 || s > 59 {
        return None;
    }
    let days = dc.get(b)?;
    Some((days * 86_400 + h * 3600 + mi * 60 + s) * 1000 - off_ms)
}


const HI4: u64 = 0xF0F0_F0F0_F0F0_F0F0;
const ZERO8: u64 = 0x3030_3030_3030_3030;

/// Every byte selected by `mask` of the little-endian word `w` is an ASCII digit.
#[inline(always)]
fn swar_digits(w: u64, mask: u64) -> bool {
    let x = (w ^ ZERO8) & mask;
    (x & HI4) == 0 && (x.wrapping_add(0x0606_0606_0606_0606) & HI4) == 0
}

/// Value of 8 ASCII digits loaded little-endian (first digit most significant).
#[inline(always)]
fn swar_parse8(w: u64) -> u64 {
    let lo = (w & 0x0f00_0f00_0f00_0f00) >> 8;
    let hi = (w & 0x000f_000f_000f_000f) * 10;
    let c = lo + hi;
    let lo = (c & 0x00ff_0000_00ff_0000) >> 16;
    let hi = (c & 0x0000_00ff_0000_00ff) * 100;
    let c = lo + hi;
    let lo = (c & 0x0000_ffff_0000_0000) >> 32;
    let hi = (c & 0x0000_0000_0000_ffff) * 10000;
    lo + hi
}

#[inline(always)]
fn ld64(b: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(b[i..i + 8].try_into().unwrap())
}

/// Same contract as `parse_ts_ref`, with SWAR digit checks and conversions.
#[inline(always)]
fn parse_ts(b: &[u8], dc: &mut Dates) -> Option<i64> {
    let n = b.len();
    if !((n == 20 || n == 25) && b[4] == b'-') {
        if n == 13 {
            let (w0, w1) = (ld64(b, 0), ld64(b, 5));
            if !(swar_digits(w0, !0) && swar_digits(w1, !0)) {
                return None;
            }
            return Some((swar_parse8(w0) * 100_000 + swar_parse8(w1) % 100_000) as i64);
        }
        return parse_f3(b);
    }
    // b[11..19] = "HH:MM:SS"
    let t = ld64(b, 11);
    if (t & 0x0000_FF00_00FF_0000) != 0x0000_3A00_003A_0000 || b[10] != b'T' || !swar_digits(t, 0xFFFF_00FF_FF00_FFFF) {
        return None;
    }
    let x = t ^ ZERO8;
    let h = (x & 0xFF) * 10 + ((x >> 8) & 0xFF);
    let mi = ((x >> 24) & 0xFF) * 10 + ((x >> 32) & 0xFF);
    let sec = ((x >> 48) & 0xFF) * 10 + ((x >> 56) & 0xFF);
    if h > 23 || mi > 59 || sec > 59 {
        return None;
    }
    let off_ms = if n == 20 {
        if b[19] != b'Z' {
            return None;
        }
        0
    } else {
        let sign = match b[19] {
            b'+' => 1,
            b'-' => -1,
            _ => return None,
        };
        if b[22] != b':' {
            return None;
        }
        let oh = d2(b, 20)?;
        let om = d2(b, 23)?;
        if oh > 23 || om > 59 {
            return None;
        }
        sign * (oh * 60 + om) * 60_000
    };
    let days = dc.get(b)?;
    Some((days * 86_400 + (h * 3600 + mi * 60 + sec) as i64) * 1000 - off_ms)
}

/// ingest_ts is always valid F1 (`YYYY-MM-DDTHH:MM:SSZ`): parse it without format checks
/// (a value of another length, or an unknown date, rejects the line instead of panicking).
#[inline(always)]
fn parse_ingest(b: &[u8], dc: &mut Dates) -> Option<i64> {
    if b.len() != 20 {
        return parse_ts(b, dc);
    }
    let days = dc.get(b)?;
    let x = ld64(b, 11) ^ ZERO8;
    let h = (x & 0xFF) * 10 + ((x >> 8) & 0xFF);
    let mi = ((x >> 24) & 0xFF) * 10 + ((x >> 32) & 0xFF);
    let sec = ((x >> 48) & 0xFF) * 10 + ((x >> 56) & 0xFF);
    Some((days * 86_400 + (h * 3600 + mi * 60 + sec) as i64) * 1000)
}

#[inline(always)]
fn parse_f3(b: &[u8]) -> Option<i64> {
    if b.is_empty() {
        return None;
    }
    let mut v: i64 = 0;
    for &c in b {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        v = v.checked_mul(10)?.checked_add(d as i64)?;
    }
    Some(v)
}

fn manifest_value(text: &str, key: &str) -> i64 {
    let k = format!("\"{key}\"");
    let i = text.find(&k).unwrap_or_else(|| panic!("manifest: {key}")) + k.len();
    let rest = &text[i..];
    let a = rest.find('"').unwrap() + 1;
    let b = a + rest[a..].find('"').unwrap();
    parse_ts(rest[a..b].as_bytes(), &mut Dates::new()).unwrap_or_else(|| panic!("manifest ts {key}"))
}

#[inline(always)]
fn price_ok(b: &[u8]) -> Option<u32> {
    let mut v: u64 = 0;
    for &c in b {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        v = v * 10 + d as u64;
        if v > 10_000_000 {
            return None;
        }
    }
    if v >= 1 {
        Some(v as u32)
    } else {
        None
    }
}

// ---------------------------------------------------------------- partitioning


/// Copies `n` records with non-temporal stores: the destination is written without
/// first being read into the cache (no read-for-ownership on fresh memory).
#[inline(always)]
unsafe fn stream_copy<T>(src: *const T, dst: *mut T, n: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::mem::size_of::<T>() % 8 == 0 && std::mem::align_of::<T>() >= 8 {
            let words = n * std::mem::size_of::<T>() / 8;
            let s = src as *const i64;
            let d = dst as *mut i64;
            for k in 0..words {
                std::arch::x86_64::_mm_stream_si64(d.add(k), *s.add(k));
            }
            return;
        }
    }
    std::ptr::copy_nonoverlapping(src, dst, n);
}

/// A vector grouped into NB contiguous buckets.
struct Parts<T> {
    data: Vec<T>,
    off: Vec<u32>, // NB + 1
}
impl<T: Copy> Parts<T> {
    fn build(v: &[T], key: impl Fn(&T) -> usize) -> Parts<T> {
        with_scratch(&S_BS, |bs| Self::build_with(v, key, bs))
    }
    fn build_with(v: &[T], key: impl Fn(&T) -> usize, bs: &mut Vec<u16>) -> Parts<T> {
        bs.extend(v.iter().map(|x| key(x) as u16));
        Self::build_pre(v, bs)
    }
    /// Groups `v` by the precomputed bucket of each record.
    fn build_pre(v: &[T], bs: &[u16]) -> Parts<T> {
        const W: usize = 8; // records per write-combining buffer
        let mut cnt = vec![0u32; NB + 1];
        for &b in bs.iter() {
            cnt[b as usize + 1] += 1;
        }
        for i in 0..NB {
            cnt[i + 1] += cnt[i];
        }
        let mut pos = cnt.clone();
        let mut data: Vec<T> = Vec::with_capacity(v.len());
        let mut wc: Vec<[std::mem::MaybeUninit<T>; W]> = Vec::with_capacity(NB);
        unsafe { wc.set_len(NB) };
        let mut fill = vec![0u8; NB];
        unsafe {
            let p = data.as_mut_ptr();
            for (x, &b) in v.iter().zip(bs.iter()) {
                let b = b as usize;
                let f = fill.get_unchecked_mut(b);
                wc.get_unchecked_mut(b)[*f as usize].write(*x);
                *f += 1;
                if *f as usize == W {
                    let d = pos.get_unchecked_mut(b);
                    stream_copy(wc.get_unchecked(b).as_ptr() as *const T, p.add(*d as usize), W);
                    *d += W as u32;
                    *f = 0;
                }
            }
            for b in 0..NB {
                let f = fill[b] as usize;
                if f > 0 {
                    std::ptr::copy_nonoverlapping(wc[b].as_ptr() as *const T, p.add(pos[b] as usize), f);
                }
            }
            data.set_len(v.len());
            #[cfg(target_arch = "x86_64")]
            std::arch::x86_64::_mm_sfence();
        }
        Parts { data, off: cnt }
    }
    #[inline]
    fn get(&self, b: usize) -> &[T] {
        &self.data[self.off[b] as usize..self.off[b + 1] as usize]
    }
}

/// Records grouped into NB buckets, one vector per bucket, filled through per-bucket
/// write-combining buffers flushed with non-temporal stores (no scratch copy, no
/// counting pass).
struct BParts<T> {
    b: Vec<Vec<T>>,
}
impl<T: Copy> BParts<T> {
    #[inline]
    fn get(&self, b: usize) -> &[T] {
        &self.b[b]
    }
}

const WCN: usize = 8;

struct Writer<T> {
    parts: Vec<Vec<T>>,
    wc: Vec<[std::mem::MaybeUninit<T>; WCN]>,
    fill: Vec<u8>,
}
impl<T: Copy> Writer<T> {
    fn new(total_hint: usize) -> Self {
        let each = total_hint / NB + total_hint / NB / 3 + 16;
        let mut wc = Vec::with_capacity(NB);
        unsafe { wc.set_len(NB) };
        Writer { parts: (0..NB).map(|_| Vec::with_capacity(each)).collect(), wc, fill: vec![0; NB] }
    }
    #[inline(always)]
    fn push(&mut self, bk: usize, x: T) {
        unsafe {
            let f = self.fill.get_unchecked_mut(bk);
            self.wc.get_unchecked_mut(bk)[*f as usize].write(x);
            *f += 1;
            if *f as usize == WCN {
                *f = 0;
                let v = self.parts.get_unchecked_mut(bk);
                if v.capacity() - v.len() < WCN {
                    v.reserve(v.len().max(WCN));
                }
                stream_copy(self.wc.get_unchecked(bk).as_ptr() as *const T, v.as_mut_ptr().add(v.len()), WCN);
                v.set_len(v.len() + WCN);
            }
        }
    }
    fn finish(mut self) -> BParts<T> {
        for bk in 0..NB {
            let f = self.fill[bk] as usize;
            if f > 0 {
                let v = &mut self.parts[bk];
                v.reserve(f);
                unsafe {
                    std::ptr::copy_nonoverlapping(self.wc[bk].as_ptr() as *const T, v.as_mut_ptr().add(v.len()), f);
                    v.set_len(v.len() + f);
                }
            }
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_sfence();
        }
        BParts { b: self.parts }
    }
}

fn gather_into<T: Copy>(parts: &[Parts<T>], b: usize, v: &mut Vec<T>) {
    let n: usize = parts.iter().map(|p| p.get(b).len()).sum();
    v.clear();
    v.reserve(n);
    for p in parts {
        v.extend_from_slice(p.get(b));
    }
}

// Per-thread scratch buffers: reusing them avoids faulting in fresh zeroed pages.
use std::cell::RefCell;
use std::thread::LocalKey;
thread_local! {
    static S_EV: RefCell<Vec<Ev>> = const { RefCell::new(Vec::new()) };
    static S_EV2: RefCell<Vec<Ev>> = const { RefCell::new(Vec::new()) };
    static S_EV3: RefCell<Vec<Ev>> = const { RefCell::new(Vec::new()) };
    static S_CK: RefCell<Vec<Ck>> = const { RefCell::new(Vec::new()) };
    static S_CK2: RefCell<Vec<Ck>> = const { RefCell::new(Vec::new()) };
    static S_CK3: RefCell<Vec<Ck>> = const { RefCell::new(Vec::new()) };
    static S_BS: RefCell<Vec<u16>> = const { RefCell::new(Vec::new()) };
}
fn with_scratch<T: 'static, R>(key: &'static LocalKey<RefCell<Vec<T>>>, f: impl FnOnce(&mut Vec<T>) -> R) -> R {
    key.with(|c| {
        let mut v = c.take();
        v.clear();
        let r = f(&mut v);
        v.clear();
        c.replace(v);
        r
    })
}

// ---------------------------------------------------------------- records

/// The dedup half of one parsed copy of an event, grouped by bucket(event_id).
/// `ingest` is seconds from window_start; `loc` is the copy's payload index in its
/// chunk (u32::MAX for an invalid copy).
#[derive(Clone, Copy)]
struct KeyRec {
    eid: K,
    ingest: i32,
    loc: u32,
}

/// The payload half of a valid copy, grouped by bucket(user_id).
/// `pc` = price | category << 24; price 0 means not a purchase.
#[derive(Clone, Copy)]
struct Pay {
    user: K,
    tsoff: u32, // ts - window_start, ms
    pc: u32,
}

struct ParsedChunk {
    keys: BParts<KeyRec>,
    pays: BParts<Pay>,
    longs: BParts<(K, B)>, // (key, bytes) of every long user_id seen, by user bucket
    off: Vec<u32>, // payload index of each user bucket's first payload
    win: Vec<std::sync::atomic::AtomicU64>, // dedup winners, by payload index
}

/// A surviving event. `key` = h32(person) << 32 | (ts - window_start).
#[derive(Clone, Copy)]
struct Ev {
    name: K,
    key: u64,
    price: u32,
    cat: u8,
    purchase: bool,
}

/// A valid click. `key` = h32(person) << 32 | (ts - (window_start - 7 days)).
#[derive(Clone, Copy)]
struct Ck {
    name: K,
    cid: K,
    key: u64,
    camp: u32,
}

struct Links {
    entries: Parts<(K, K, u64)>, // (id, person, mix(person)) grouped by bucket(id)
    tables: Vec<Vec<u32>>,  // per bucket: open addressing over entries
}
impl Links {
    /// Resolves an id that falls in bucket b.
    #[inline(always)]
    fn name_in(&self, b: usize, id: K) -> K {
        self.name_hash_in(b, id, mix(id)).0
    }
    /// Person and mix(person) of an id in bucket b whose mix is `hid`.
    #[inline(always)]
    fn name_hash_in(&self, b: usize, id: K, hid: u64) -> (K, u64) {
        let t = unsafe { self.tables.get_unchecked(b) };
        let e = self.entries.get(b);
        let mask = t.len() - 1;
        let mut h = (hid as usize) & mask;
        loop {
            let s = unsafe { *t.get_unchecked(h) };
            if s == u32::MAX {
                return (id, hid);
            }
            let (k, n, hn) = unsafe { *e.get_unchecked(s as usize) };
            if k == id {
                return (n, hn);
            }
            h = (h + 1) & mask;
        }
    }
}

fn build_table<T>(items: &[T], key: impl Fn(&T) -> K) -> Vec<u32> {
    let cap = (items.len() * 2).next_power_of_two().max(8);
    let mask = cap - 1;
    let mut t = vec![u32::MAX; cap];
    for (i, x) in items.iter().enumerate() {
        let mut h = (mix(key(x)) as usize) & mask;
        while t[h] != u32::MAX {
            h = (h + 1) & mask;
        }
        t[h] = i as u32;
    }
    t
}

struct Ctx {
    ws: i64,
    we: i64,
    products: AHashMap<K, u8>,
    unknown_cat: u8,
    campaigns: AHashMap<K, u32>,
}

fn parse_events(chunk: B, cx: &Ctx) -> ParsedChunk {
    let est = chunk.len() / 70 + 16;
    let mut keys: Writer<KeyRec> = Writer::new(est);
    let mut pays: Writer<Pay> = Writer::new(est);
    let mut longs: Writer<(K, B)> = Writer::new(est / 4);
    let mut rank = vec![0u32; NB];
    let mut dc = Dates::new();
    let ws_s = cx.ws.div_euclid(1000);
    scan_lines::<7>(chunk, |f| {
        if f[0].is_empty() {
            return;
        }
        let ingest = match parse_ingest(f[3], &mut dc) {
            Some(t) => t,
            None => return, // ingest_ts is always valid F1
        };
        let mut kr = KeyRec {
            eid: pack(f[0], false),
            ingest: (ingest.div_euclid(1000) - ws_s).clamp(i32::MIN as i64, i32::MAX as i64) as i32,
            loc: u32::MAX,
        };
        'v: {
            if f[1].is_empty() || f[5].is_empty() {
                break 'v;
            }
            let ts = match parse_ts(f[2], &mut dc) {
                Some(t) if t >= cx.ws && t < cx.we => t,
                _ => break 'v,
            };
            let t = f[4];
            let purchase = match t.len() {
                8 if u64::from_le_bytes(t.try_into().unwrap()) == u64::from_le_bytes(*b"purchase") => true,
                4 if {
                    let w = u32::from_le_bytes(t.try_into().unwrap());
                    w == u32::from_le_bytes(*b"view") || w == u32::from_le_bytes(*b"cart")
                } => false,
                _ => break 'v,
            };
            let mut pc = 0u32;
            if purchase {
                let price = match price_ok(f[6]) {
                    Some(p) => p,
                    None => break 'v,
                };
                let cat = match cx.products.get(&pack(f[5], false)) {
                    Some(&c) => c,
                    None => cx.unknown_cat,
                };
                pc = price | (cat as u32) << 24;
            }
            if ingest - ts > DAY_MS {
                break 'v;
            }
            let user = pack(f[1], false);
            let ub = bucket(user);
            if is_long(user) {
                // bytes of long ids are registered later, once per id, per user bucket
                longs.push(ub, (user, unsafe { std::slice::from_raw_parts(f[1].as_ptr(), f[1].len()) }));
            }
            // position after grouping by user bucket = offset of ub + rank within ub
            kr.loc = (ub as u32) << 22 | rank[ub];
            rank[ub] += 1;
            pays.push(ub, Pay { user, tsoff: (ts - cx.ws) as u32, pc });
        }
        keys.push(bucket(kr.eid), kr);
    });
    assert!(rank.iter().all(|&r| r < 1 << 22));
    let pays = pays.finish();
    let keys = keys.finish();
    let longs = longs.finish();
    let mut off = vec![0u32; NB + 1];
    for i in 0..NB {
        off[i + 1] = off[i] + pays.get(i).len() as u32;
    }
    let win = (0..off[NB] as usize / 64 + 1).map(|_| std::sync::atomic::AtomicU64::new(0)).collect();
    ParsedChunk { keys, pays, longs, off, win }
}

fn parse_clicks(chunk: B, cx: &Ctx) -> Parts<Ck> {
    with_scratch(&S_CK, |v| parse_clicks_into(chunk, cx, v))
}

fn parse_clicks_into(chunk: B, cx: &Ctx, v: &mut Vec<Ck>) -> Parts<Ck> {
    v.reserve(chunk.len() / 40 + 16);
    let mut dc = Dates::new();
    let lo = cx.ws - WEEK_MS;
    scan_lines::<4>(chunk, |f| {
        if f[0].is_empty() || f[1].is_empty() || f[2].is_empty() {
            return;
        }
        let camp = match cx.campaigns.get(&pack(f[2], false)) {
            Some(&c) => c,
            None => return,
        };
        let ts = match parse_ts(f[3], &mut dc) {
            Some(t) if t >= lo && t < cx.we => t,
            _ => return,
        };
        v.push(Ck { name: pack(f[1], false), cid: pack_cid(f[0]), key: (ts - lo) as u64, camp });
    });
    Parts::build(v, |c| bucket(c.name))
}

/// Dedup one event_id bucket: among the copies of each event keep the one with the
/// smallest ingest_ts and, if that copy is valid, mark its payload as a winner.
fn dedup(chunks: &[ParsedChunk], b: usize) {
    let n: usize = chunks.iter().map(|c| c.keys.get(b).len()).sum();
    let cap = (n * 2).next_power_of_two().max(16);
    let mask = cap - 1;
    let mut slots: Vec<u64> = vec![u64::MAX; cap]; // chunk << 32 | index
    let rec = |s: u64| -> &KeyRec { unsafe { chunks.get_unchecked((s >> 32) as usize).keys.get(b).get_unchecked(s as u32 as usize) } };
    for (c, ch) in chunks.iter().enumerate() {
        for (i, r) in ch.keys.get(b).iter().enumerate() {
            let me = (c as u64) << 32 | i as u64;
            let mut h = (mix(r.eid) as usize) & mask;
            loop {
                let s = unsafe { *slots.get_unchecked(h) };
                if s == u64::MAX {
                    slots[h] = me;
                    break;
                }
                let o = rec(s);
                if o.eid == r.eid {
                    if r.ingest < o.ingest {
                        slots[h] = me;
                    }
                    break;
                }
                h = (h + 1) & mask;
            }
        }
    }
    for &s in &slots {
        if s == u64::MAX {
            continue;
        }
        let loc = rec(s).loc;
        if loc != u32::MAX {
            let ch = &chunks[(s >> 32) as usize];
            let pos = ch.off[(loc >> 22) as usize] + (loc & ((1 << 22) - 1));
            ch.win[(pos >> 6) as usize].fetch_or(1 << (pos & 63), std::sync::atomic::Ordering::Relaxed);
        }
    }
}

// ---------------------------------------------------------------- sessions + attribution

struct UserRow {
    name: K,
    day: i32,
    sessions: i64,
    events: i64,
    revenue: i64,
    top: i32, // -1 = null
}

#[derive(Default, Clone, Copy)]
struct CampAgg {
    sessions: i64,
    purchasing: i64,
    revenue: i64,
    assisted: i64,
}

struct Acc {
    camps: Vec<CampAgg>, // camp * ndays + (day - day0)
    cnt: Vec<u32>,       // per campaign: clicks in the current window
    active: Vec<u32>,
    users: Vec<Vec<UserRow>>,
}

struct Shared<'a> {
    cat_rank: &'a [u32],
    organic: u32,
    ndays: usize,
    day0: i32,
    ws: i64,
    cbase: i64,
}

// Person order used by both lists: (low 12 bits of h32, h32, name).
#[inline(always)]
fn pk_e(e: &Ev) -> (u64, u64, K) {
    let h = e.key >> 32;
    (h & 0xFFF, h, e.name)
}
#[inline(always)]
fn pk_c(c: &Ck) -> (u64, u64, K) {
    let h = c.key >> 32;
    (h & 0xFFF, h, c.name)
}


/// Stable sort of `src` by the low 31 bits of `key` (time offset) into `dst`:
/// three LSD counting passes of 11 bits over packed time << 32 | index.
fn radix_by_time<T: Copy>(src: &[T], dst: &mut [T], key: &impl Fn(&T) -> u64) {
    let m = src.len();
    let mut a: Vec<u64> = src.iter().enumerate().map(|(i, x)| (key(x) & 0x7FFF_FFFF) << 32 | i as u64).collect();
    let mut b: Vec<u64> = vec![0; m];
    for pass in 0..3 {
        let sh = 32 + 11 * pass;
        let mut cnt = vec![0u32; 2049];
        for &v in &a {
            cnt[((v >> sh) & 0x7FF) as usize + 1] += 1;
        }
        for i in 0..2048 {
            cnt[i + 1] += cnt[i];
        }
        for &v in &a {
            let d = ((v >> sh) & 0x7FF) as usize;
            b[cnt[d] as usize] = v;
            cnt[d] += 1;
        }
        std::mem::swap(&mut a, &mut b);
    }
    for (d, &v) in dst.iter_mut().zip(a.iter()) {
        *d = src[(v & 0xFFFF_FFFF) as usize];
    }
}

thread_local! {
    static S_KEYS: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

/// Sorts records by `key` = person hash (high 32 bits) << 32 | time offset (low 32 bits).
/// A counting sort on the low 12 bits of the hash forms groups (mostly one person each);
/// each group is then ordered through packed u64 keys
/// hash[12..32] << 44 | time << 13 | index, so only 8-byte values are compared and moved.
/// Records with equal keys keep their input order. Big groups or times of 2^31 ms or more
/// use a plain sort of the records.
fn group_sort<T: Copy>(v: &[T], tmp: &mut Vec<T>, out: &mut Vec<T>, key: impl Fn(&T) -> u64) {
    const SB: usize = 4096;
    let mut cnt = [0u32; SB + 1];
    for x in v {
        cnt[((key(x) >> 32) & 0xFFF) as usize + 1] += 1;
    }
    for i in 0..SB {
        cnt[i + 1] += cnt[i];
    }
    let mut pos = cnt;
    let n = v.len();
    tmp.clear();
    tmp.reserve(n);
    out.clear();
    out.reserve(n);
    unsafe {
        let p = tmp.as_mut_ptr();
        for x in v {
            let i = pos.get_unchecked_mut(((key(x) >> 32) & 0xFFF) as usize);
            p.add(*i as usize).write(*x);
            *i += 1;
        }
        tmp.set_len(n);
        out.set_len(n);
    }
    with_scratch(&S_KEYS, |keys| {
        for g in 0..SB {
            let (a, b) = (cnt[g] as usize, cnt[g + 1] as usize);
            let m = b - a;
            if m == 0 {
                continue;
            }
            let src = &tmp[a..b];
            let dst = &mut out[a..b];
            if m == 1 {
                dst[0] = src[0];
                continue;
            }
            let mut tor = 0u64;
            for x in src {
                tor |= key(x);
            }
            if m < (1 << 13) && (tor & 0x8000_0000) == 0 {
                keys.clear();
                for (i, x) in src.iter().enumerate() {
                    let k = key(x);
                    keys.push((k >> 44) << 44 | (k & 0x7FFF_FFFF) << 13 | i as u64);
                }
                keys.sort_unstable();
                for (d, &k) in dst.iter_mut().zip(keys.iter()) {
                    *d = unsafe { *src.get_unchecked((k & 0x1FFF) as usize) };
                }
            } else if (tor >> 32) == (src.iter().fold(!0u64, |a, x| a & key(x)) >> 32) && m < (1 << 32) && (tor & 0x8000_0000) == 0 {
                // one person (all hashes equal): LSD radix sort of time << 32 | index
                radix_by_time(src, dst, &key);
            } else {
                let mut pairs: Vec<(u64, u32)> = src.iter().enumerate().map(|(i, x)| (key(x), i as u32)).collect();
                pairs.sort_unstable();
                for (d, &(_, i)) in dst.iter_mut().zip(pairs.iter()) {
                    *d = src[i as usize];
                }
            }
        }
    });
}

fn process_bucket(pev: &[Parts<Ev>], pck: &[Parts<Ck>], b: usize, sh: &Shared, acc: &mut Acc) {
    with_scratch(&S_EV, |e1| {
        with_scratch(&S_EV2, |e2| {
        with_scratch(&S_EV3, |e3| {
            with_scratch(&S_CK, |c1| {
                with_scratch(&S_CK2, |c2| {
                with_scratch(&S_CK3, |c3| {
                    gather_into(pev, b, e1);
                    group_sort(e1, e2, e3, |e| e.key);
                    gather_into(pck, b, c1);
                    group_sort(c1, c2, c3, |c| c.key);
                    // clicks at the same person and millisecond: smallest click_id first
                    let mut i = 0;
                    while i < c3.len() {
                        let mut j = i + 1;
                        while j < c3.len() && c3[j].key == c3[i].key {
                            j += 1;
                        }
                        if j - i > 1 {
                            c3[i..j].sort_by(|a, b| cmp_key(a.cid, b.cid));
                        }
                        i = j;
                    }
                    process_sorted(e3, c3, sh, acc);
                })
                })
            })
        })
        })
    })
}

fn process_sorted(evs: &mut [Ev], cks: &mut [Ck], sh: &Shared, acc: &mut Acc) {
    // Two people with the same 32-bit hash would interleave: fall back to a full sort.
    let clash_e = evs.windows(2).any(|w| w[0].key >> 32 == w[1].key >> 32 && w[0].name != w[1].name);
    let clash_c = cks.windows(2).any(|w| w[0].key >> 32 == w[1].key >> 32 && w[0].name != w[1].name);
    if clash_e || clash_c {
        evs.sort_unstable_by(|a, b| (pk_e(a), a.key).cmp(&(pk_e(b), b.key)));
        cks.sort_unstable_by(|a, b| (pk_c(a), a.key).cmp(&(pk_c(b), b.key)).then_with(|| cmp_key(a.cid, b.cid)));
    }
    let mut users = Vec::new();
    let mut cats: Vec<(u8, i64)> = Vec::new();
    let ndays = sh.ndays;
    const LO32: u64 = 0xFFFF_FFFF;

    let mut ci = 0usize;
    let mut i = 0usize;
    while i < evs.len() {
        let pk = pk_e(&evs[i]);
        let name = evs[i].name;
        let mut j = i + 1;
        while j < evs.len() && pk_e(&evs[j]) == pk {
            j += 1;
        }
        while ci < cks.len() && pk_c(&cks[ci]) < pk {
            ci += 1;
        }
        let c0 = ci;
        while ci < cks.len() && pk_c(&cks[ci]) == pk {
            ci += 1;
        }
        let pc = &cks[c0..ci];
        let pe = &evs[i..j];
        let cts = |c: &Ck| sh.cbase + (c.key & LO32) as i64;

        let (mut lo, mut hi) = (0usize, 0usize);
        let mut cur: Option<UserRow> = None;
        cats.clear();
        let mut k = 0;
        while k < pe.len() {
            let start = sh.ws + (pe[k].key & LO32) as i64;
            let mut m = k + 1;
            while m < pe.len() && (pe[m].key & LO32) - (pe[m - 1].key & LO32) <= GAP_MS as u64 {
                m += 1;
            }
            let day = start.div_euclid(DAY_MS) as i32;
            if cur.as_ref().map(|r| r.day != day).unwrap_or(false) {
                let mut r = cur.take().unwrap();
                r.top = top_cat(&cats, sh.cat_rank);
                users.push(r);
                cats.clear();
            }
            let mut rev = 0i64;
            let mut purch = false;
            for e in &pe[k..m] {
                if e.purchase {
                    purch = true;
                    rev += e.price as i64;
                    match cats.iter_mut().find(|c| c.0 == e.cat) {
                        Some(c) => c.1 += e.price as i64,
                        None => cats.push((e.cat, e.price as i64)),
                    }
                }
            }
            let r = cur.get_or_insert(UserRow { name, day, sessions: 0, events: 0, revenue: 0, top: -1 });
            r.sessions += 1;
            r.events += (m - k) as i64;
            r.revenue += rev;

            // attribution window [start - 7d, start]
            while hi < pc.len() && cts(&pc[hi]) <= start {
                let c = pc[hi].camp as usize;
                if acc.cnt[c] == 0 {
                    acc.active.push(c as u32);
                }
                acc.cnt[c] += 1;
                hi += 1;
            }
            while lo < hi && cts(&pc[lo]) < start - WEEK_MS {
                acc.cnt[pc[lo].camp as usize] -= 1;
                lo += 1;
            }
            let credit = if lo < hi {
                let t = pc[hi - 1].key;
                let mut f = hi - 1;
                while f > lo && pc[f - 1].key == t {
                    f -= 1;
                }
                pc[f].camp
            } else {
                sh.organic
            };
            let di = (day - sh.day0) as usize;
            let a = &mut acc.camps[credit as usize * ndays + di];
            a.sessions += 1;
            a.purchasing += purch as i64;
            a.revenue += rev;
            let cnt = &acc.cnt;
            acc.active.retain(|&c| cnt[c as usize] > 0);
            for &c in &acc.active {
                acc.camps[c as usize * ndays + di].assisted += 1;
            }
            k = m;
        }
        if let Some(mut r) = cur.take() {
            r.top = top_cat(&cats, sh.cat_rank);
            users.push(r);
        }
        for &c in &acc.active {
            acc.cnt[c as usize] = 0;
        }
        acc.active.clear();
        i = j;
    }
    acc.users.push(users);
}

fn top_cat(cats: &[(u8, i64)], rank: &[u32]) -> i32 {
    let mut best: Option<(u8, i64)> = None;
    for &(c, v) in cats {
        best = match best {
            Some((bc, bv)) if !(v > bv || (v == bv && rank[c as usize] < rank[bc as usize])) => Some((bc, bv)),
            _ => Some((c, v)),
        };
    }
    best.map(|b| b.0 as i32).unwrap_or(-1)
}

// ---------------------------------------------------------------- links

fn build_links(chunks: &[B]) -> Links {
    let pairs: Vec<(K, K)> = chunks
        .par_iter()
        .flat_map_iter(|c| {
            let mut v = Vec::new();
            scan_lines::<2>(c, |&[a, b]| {
                if !a.is_empty() && !b.is_empty() {
                    v.push((pack(a, true), pack(b, true)));
                }
            });
            v
        })
        .collect();
    // Intern ids with an open-addressing table.
    let cap = (pairs.len() * 4).next_power_of_two().max(16);
    let mask = cap - 1;
    let mut table = vec![u32::MAX; cap];
    let mut ids: Vec<K> = Vec::with_capacity(pairs.len() * 2);
    let mut intern = |k: K, ids: &mut Vec<K>| -> u32 {
        let mut h = (mix(k) as usize) & mask;
        loop {
            let s = table[h];
            if s == u32::MAX {
                table[h] = ids.len() as u32;
                ids.push(k);
                return ids.len() as u32 - 1;
            }
            if ids[s as usize] == k {
                return s;
            }
            h = (h + 1) & mask;
        }
    };
    let ipairs: Vec<(u32, u32)> = pairs.iter().map(|&(a, b)| (intern(a, &mut ids), intern(b, &mut ids))).collect();
    let n = ids.len();
    let mut p: Vec<u32> = (0..n as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            let g = p[p[x as usize] as usize];
            p[x as usize] = g;
            x = g;
        }
        x
    }
    for &(a, b) in &ipairs {
        let (ra, rb) = (find(&mut p, a), find(&mut p, b));
        if ra != rb {
            if ra < rb {
                p[rb as usize] = ra;
            } else {
                p[ra as usize] = rb;
            }
        }
    }
    // Each person is named by its byte-wise smallest id.
    let mut best: Vec<K> = ids.clone();
    let mut root: Vec<u32> = vec![0; n];
    for i in 0..n {
        let r = find(&mut p, i as u32);
        root[i] = r;
        let r = r as usize;
        if cmp_key(ids[i], best[r]) == Ordering::Less {
            best[r] = ids[i];
        }
    }
    let entries: Vec<(K, K, u64)> = (0..n).map(|i| { let nm = best[root[i] as usize]; (ids[i], nm, mix(nm)) }).collect();
    let entries = Parts::build(&entries, |e| bucket(e.0));
    let tables: Vec<Vec<u32>> = (0..NB).into_par_iter().map(|b| build_table(entries.get(b), |e| e.0)).collect();
    Links { entries, tables }
}

// ---------------------------------------------------------------- parquet

fn write_parquet(path: &Path, batch: arrow_array::RecordBatch) {
    use parquet::arrow::ArrowWriter;
    use parquet::basic::{Compression, Encoding};
    use parquet::file::properties::{EnabledStatistics, WriterProperties};
    let f = std::fs::File::create(path).unwrap_or_else(|e| panic!("create {path:?}: {e}"));
    let props = WriterProperties::builder()
        .set_compression(Compression::UNCOMPRESSED)
        .set_dictionary_enabled(false)
        .set_encoding(Encoding::PLAIN)
        .set_statistics_enabled(EnabledStatistics::None)
        .build();
    let mut w = ArrowWriter::try_new(std::io::BufWriter::with_capacity(1 << 20, f), batch.schema(), Some(props)).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

// ---------------------------------------------------------------- main


fn fuzz_all() {
    let mut seed: u64 = 0x1234_5678_9abc_def1;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    // --- scanner
    let alpha = [b'a', b'b', b',', b',', b'"', b'\n', b' ', b'\r', b'x', b'1'];
    for round in 0..4000 {
        let mut buf: Vec<u8> = Vec::new();
        let lines = if round % 40 == 0 { 3000 + rnd() % 3000 } else { rnd() % 80 };
        for _ in 0..lines {
            match rnd() % 6 {
                0 => {
                    let n = if rnd() % 400 == 0 { 70000 } else { rnd() % 300 };
                    for _ in 0..n { buf.push(alpha[(rnd() % alpha.len() as u64) as usize]); }
                }
                _ => {
                    let nf = 1 + rnd() % 8;
                    for i in 0..nf {
                        if i > 0 { buf.push(b','); }
                        let fl = rnd() % 25;
                        let q = rnd() % 4 == 0;
                        if q { buf.push(b'"'); }
                        for _ in 0..fl { buf.push([b'a', b'z', b'9', b' ', b'\r', b'"', b'b', b'c'][(rnd() % 8) as usize]); }
                        if q { buf.push(b'"'); }
                    }
                }
            }
            if rnd() % 50 != 0 { buf.push(b'\n'); }
        }
        fn run<const N: usize>(b: &[u8], fast: bool) -> Vec<Vec<Vec<u8>>> {
            let mut out = Vec::new();
            if fast { scan_lines::<N>(b, |f| out.push(f.iter().map(|x| x.to_vec()).collect())); }
            else { scan::<N>(b, |f| out.push(f.iter().map(|x| x.to_vec()).collect())); }
            out
        }
        if !(run::<7>(&buf, true) == run::<7>(&buf, false) && run::<4>(&buf, true) == run::<4>(&buf, false) && run::<2>(&buf, true) == run::<2>(&buf, false)) {
            eprintln!("SCAN MISMATCH round {round} len {}", buf.len());
            if buf.len() < 3000 { eprintln!("{:?}", String::from_utf8_lossy(&buf)); }
            std::process::exit(1);
        }
    }
    eprintln!("scan fuzz ok");
    // --- timestamps
    let mut d1 = Dates::new();
    let mut d2c = Dates::new();
    let bases = ["2026-10-03T23:58:10Z", "2026-10-03T19:58:10-04:00", "2028-02-29T00:00:00Z", "2027-02-29T00:00:00Z",
                 "2026-13-01T00:00:00Z", "2026-10-03T24:00:00Z", "2026-10-03T23:59:60Z", "2026-10-03T23:59:59+23:59",
                 "2026-10-03T23:59:59+24:00", "2026-10-03T23:59:59+05:60", "1791071890000", "0001791071890000",
                 "86400000", "1791071890000.0", "2026-10-03 23:58:10Z", "2026-10-03t23:58:10Z", "2026-10-03T23:58:10z",
                 "2000-02-29T12:30:45Z", "1900-02-29T12:30:45Z", "2026-04-31T00:00:00Z", "2026-10-03T23:58:10+0200x"];
    let chars = b"0123456789-:TZ+ zt.\x00\xff";
    let mut n = 0u64;
    for i in 0..3_000_000u64 {
        let mut v: Vec<u8> = bases[(rnd() % bases.len() as u64) as usize].as_bytes().to_vec();
        match i % 4 {
            0 => {}
            1 => { let k = (rnd() % v.len() as u64) as usize; v[k] = chars[(rnd() % chars.len() as u64) as usize]; }
            2 => { for _ in 0..3 { let k = (rnd() % v.len() as u64) as usize; v[k] = b'0' + (rnd() % 10) as u8; } }
            _ => { let l = 1 + rnd() % 22; v = (0..l).map(|_| b'0' + (rnd() % 10) as u8).collect(); if rnd() % 3 == 0 { v.truncate(13.min(v.len())); } }
        }
        let a = parse_ts(&v, &mut d1);
        let b = parse_ts_ref(&v, &mut d2c);
        if a != b {
            eprintln!("TS MISMATCH {:?}: fast {:?} ref {:?}", String::from_utf8_lossy(&v), a, b);
            std::process::exit(1);
        }
        n += a.is_some() as u64;
    }
    eprintln!("ts fuzz ok ({n} valid)");
    std::process::exit(0);
}

fn main() {
    if std::env::var("FUZZ").is_ok() {
        fuzz_all();
    }
    let t0 = std::time::Instant::now();
    unsafe {
        // Back large allocations with transparent huge pages.
        libmimalloc_sys::mi_option_set(libmimalloc_sys::mi_option_large_os_pages, 1);
        // mi_option_purge_delay (v2 enum value 15): never return freed memory, so it is reused without re-faulting
        libmimalloc_sys::mi_option_set(15, -1);
    }
    if let Some(n) = std::env::var("NCPU").ok().and_then(|v| v.parse::<usize>().ok()) {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(n.max(1)).build_global();
    }
    let data = PathBuf::from(std::env::var("DATA_DIR").unwrap_or_else(|_| "/data".into()));
    let out = PathBuf::from(std::env::var("OUTPUT_DIR").unwrap_or_else(|_| "/output".into()));
    let verbose = std::env::var("VERBOSE").is_ok();
    let lap = |what: &str| {
        if verbose {
            eprintln!("{:>8.3}s {what}", t0.elapsed().as_secs_f64());
        }
    };

    let ((ev_chunks, ck_chunks), link_chunks) = rayon::join(
        || {
            rayon::join(
                || chunks_of(&csv_files(&data.join("events")), 16 << 20),
                || chunks_of(&csv_files(&data.join("clicks")), 8 << 20),
            )
        },
        || chunks_of(&csv_files(&data.join("links")), 1 << 20),
    );
    lap("mapped");

    let mtext = std::fs::read_to_string(data.join("manifest.json")).expect("manifest");
    let ws = manifest_value(&mtext, "window_start");
    let we = manifest_value(&mtext, "window_end");
    let cbase = ws - WEEK_MS;
    assert!(we - cbase < (1i64 << 32), "window too long for 32-bit offsets");

    // products / categories
    let mut cat_names: Vec<Vec<u8>> = Vec::new();
    let mut cat_idx: AHashMap<Vec<u8>, u8> = AHashMap::new();
    let mut products: AHashMap<K, u8> = AHashMap::new();
    for c in chunks_of(&[data.join("products.csv")], usize::MAX) {
        scan::<2>(c, |&[p, c]| {
            if p.is_empty() || c.is_empty() {
                return;
            }
            let n = cat_names.len();
            let id = *cat_idx.entry(c.to_vec()).or_insert_with(|| {
                assert!(n < 255, "too many categories");
                cat_names.push(c.to_vec());
                n as u8
            });
            products.insert(pack(p, false), id);
        });
    }
    let unknown_cat = cat_names.len() as u8;
    cat_names.push(b"unknown".to_vec());
    let mut order: Vec<usize> = (0..cat_names.len()).collect();
    order.sort_by(|&a, &b| cat_names[a].cmp(&cat_names[b]));
    let mut cat_rank = vec![0u32; cat_names.len()];
    for (r, &c) in order.iter().enumerate() {
        cat_rank[c] = r as u32;
    }

    // campaigns
    let mut campaigns: AHashMap<K, u32> = AHashMap::new();
    let mut camp_names: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    for c in chunks_of(&[data.join("campaigns.csv")], usize::MAX) {
        scan::<2>(c, |&[c, ch]| {
            if c.is_empty() {
                return;
            }
            campaigns.insert(pack(c, false), camp_names.len() as u32);
            camp_names.push((c.to_vec(), ch.to_vec()));
        });
    }
    let organic = camp_names.len() as u32;
    camp_names.push((b"organic".to_vec(), b"organic".to_vec()));
    let cx = Ctx { ws, we, products, unknown_cat, campaigns };

    // Links are built alongside event parsing and dedup, which do not need them.
    let (links, (chunks, clicks)) = rayon::join(
        || build_links(&link_chunks),
        || {
            let chunks: Vec<ParsedChunk> = ev_chunks.par_iter().map(|c| parse_events(c, &cx)).collect();
            lap("events parsed");
            let (_, clicks): ((), Vec<Parts<Ck>>) = rayon::join(
                || (0..NB).into_par_iter().for_each(|b| dedup(&chunks, b)),
                || ck_chunks.par_iter().map(|c| parse_clicks(c, &cx)).collect(),
            );
            lap("deduped + clicks");
            (chunks, clicks)
        },
    );
    lap("links ready");

    // Partition join: per user bucket, keep winning payloads, resolve ids to people,
    // regroup by person.
    let resolved: Vec<(Parts<Ev>, Parts<Ck>)> = (0..NB)
        .into_par_iter()
        .map(|b| {
            // register the bytes of this bucket's long user ids, once each
            {
                let mut seen: ahash::AHashSet<K> = ahash::AHashSet::new();
                for ch in &chunks {
                    for &(k, bytes) in ch.longs.get(b) {
                        if seen.insert(k) {
                            long_shard(k).lock().unwrap().entry(k).or_insert(bytes);
                        }
                    }
                }
            }
            let pe = with_scratch(&S_EV, |e| {
                with_scratch(&S_BS, |bs| {
                    let n: usize = chunks.iter().map(|c| c.pays.get(b).len()).sum();
                    e.reserve(n);
                    bs.reserve(n);
                    for ch in &chunks {
                        let base = ch.off[b] as usize;
                        for (i, p) in ch.pays.get(b).iter().enumerate() {
                            let pos = base + i;
                            if (ch.win[pos >> 6].load(std::sync::atomic::Ordering::Relaxed) >> (pos & 63)) & 1 == 0 {
                                continue;
                            }
                            let (name, hn) = links.name_hash_in(b, p.user, mix(p.user));
                            let price = p.pc & 0xFF_FFFF;
                            e.push(Ev { name, key: (hn & 0xFFFF_FFFF) << 32 | p.tsoff as u64, price, cat: (p.pc >> 24) as u8, purchase: price != 0 });
                            bs.push((hn >> (64 - NB_BITS)) as u16);
                        }
                    }
                    Parts::build_pre(e, bs)
                })
            });
            let pc = with_scratch(&S_CK, |c| {
                with_scratch(&S_BS, |bs| {
                    let n: usize = clicks.iter().map(|p| p.get(b).len()).sum();
                    c.reserve(n);
                    bs.reserve(n);
                    for cp in &clicks {
                        for x in cp.get(b) {
                            let (name, hn) = links.name_hash_in(b, x.name, mix(x.name));
                            c.push(Ck { name, cid: x.cid, key: x.key | (hn & 0xFFFF_FFFF) << 32, camp: x.camp });
                            bs.push((hn >> (64 - NB_BITS)) as u16);
                        }
                    }
                    Parts::build_pre(c, bs)
                })
            });
            (pe, pc)
        })
        .collect();
    drop(chunks);

    drop(clicks);
    let (pev, pck): (Vec<Parts<Ev>>, Vec<Parts<Ck>>) = resolved.into_iter().unzip();
    lap("resolved");

    let day0 = ws.div_euclid(DAY_MS) as i32;
    let ndays = ((we - 1).div_euclid(DAY_MS) as i32 - day0 + 1) as usize;
    let ncamp = camp_names.len();
    let sh = Shared { cat_rank: &cat_rank, organic, ndays, day0, ws, cbase };
    let accs: Vec<Acc> = (0..NB)
        .into_par_iter()
        .fold(
            || Acc { camps: vec![CampAgg::default(); ncamp * ndays], cnt: vec![0; ncamp], active: Vec::new(), users: Vec::new() },
            |mut acc, b| {
                process_bucket(&pev, &pck, b, &sh, &mut acc);
                acc
            },
        )
        .collect();
    lap("sessions + attribution");

    // ---- outputs
    use arrow_array::builder::StringBuilder;
    use arrow_array::{ArrayRef, Date32Array, Int64Array, RecordBatch, StringArray};
    let ud = out.join("user_daily");
    let cd = out.join("campaign_daily");
    std::fs::create_dir_all(&ud).unwrap();
    std::fs::create_dir_all(&cd).unwrap();

    let groups: Vec<&Vec<UserRow>> = accs.iter().flat_map(|a| a.users.iter()).collect();
    let nparts = 8usize;
    let per = groups.len().div_ceil(nparts).max(1);
    let cat_strs: Vec<String> = cat_names.iter().map(|c| String::from_utf8_lossy(c).into_owned()).collect();

    let write_users = || {
        (0..nparts).into_par_iter().for_each(|p| {
            let rows: Vec<&UserRow> = groups.iter().skip(p * per).take(per).flat_map(|g| g.iter()).collect();
            let mut nb = StringBuilder::with_capacity(rows.len(), rows.len() * 12);
            for r in &rows {
                if is_long(r.name) {
                    let b = key_bytes(r.name);
                    nb.append_value(std::str::from_utf8(&b).unwrap_or("\u{FFFD}"));
                } else {
                    let v = (((r.name.0 as u128) << 64) | r.name.1 as u128).to_be_bytes();
                    let bytes = &v[..v[15] as usize];
                    if u128::from_be_bytes(v) & 0x8080_8080_8080_8080_8080_8080_8080_8000 == 0 {
                        // ASCII (the length byte is masked out): valid UTF-8
                        nb.append_value(unsafe { std::str::from_utf8_unchecked(bytes) });
                    } else {
                        nb.append_value(std::str::from_utf8(bytes).unwrap_or("\u{FFFD}"));
                    }
                }
            }
            let batch = RecordBatch::try_from_iter(vec![
                ("user_id", Arc::new(nb.finish()) as ArrayRef),
                ("day", Arc::new(Date32Array::from_iter_values(rows.iter().map(|u| u.day))) as ArrayRef),
                ("sessions", Arc::new(Int64Array::from_iter_values(rows.iter().map(|u| u.sessions))) as ArrayRef),
                ("events", Arc::new(Int64Array::from_iter_values(rows.iter().map(|u| u.events))) as ArrayRef),
                ("revenue_cents", Arc::new(Int64Array::from_iter_values(rows.iter().map(|u| u.revenue))) as ArrayRef),
                (
                    "top_category",
                    Arc::new(StringArray::from_iter(
                        rows.iter().map(|u| if u.top < 0 { None } else { Some(cat_strs[u.top as usize].as_str()) }),
                    )) as ArrayRef,
                ),
            ])
            .unwrap();
            write_parquet(&ud.join(format!("part-{p}.parquet")), batch);
        })
    };
    let write_camps = || {
        let mut tot = vec![CampAgg::default(); ncamp * ndays];
        for a in &accs {
            for (t, x) in tot.iter_mut().zip(&a.camps) {
                t.sessions += x.sessions;
                t.purchasing += x.purchasing;
                t.revenue += x.revenue;
                t.assisted += x.assisted;
            }
        }
        let mut rows: Vec<(usize, usize, CampAgg)> = Vec::new();
        for c in 0..ncamp {
            for d in 0..ndays {
                let a = tot[c * ndays + d];
                if a.sessions > 0 || a.assisted > 0 {
                    rows.push((c, d, a));
                }
            }
        }
        let st = |b: &Vec<u8>| String::from_utf8_lossy(b).into_owned();
        let batch = RecordBatch::try_from_iter(vec![
            ("campaign_id", Arc::new(StringArray::from_iter_values(rows.iter().map(|r| st(&camp_names[r.0].0)))) as ArrayRef),
            ("channel", Arc::new(StringArray::from_iter_values(rows.iter().map(|r| st(&camp_names[r.0].1)))) as ArrayRef),
            ("day", Arc::new(Date32Array::from_iter_values(rows.iter().map(|r| day0 + r.1 as i32))) as ArrayRef),
            ("sessions", Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.2.sessions))) as ArrayRef),
            ("purchasing_sessions", Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.2.purchasing))) as ArrayRef),
            ("revenue_cents", Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.2.revenue))) as ArrayRef),
            ("assisted_sessions", Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.2.assisted))) as ArrayRef),
        ])
        .unwrap();
        write_parquet(&cd.join("part-0.parquet"), batch);
    };
    rayon::join(write_users, write_camps);
    lap("written");
    // Skip freeing gigabytes of memory.
    std::process::exit(0);
}
