/*
 * Local definitions of glibc 2.38's C23-conformant strtol family so the
 * cdylib loads on its claimed manylinux_2_34 baseline (Ubuntu 22.04 LTS,
 * RHEL 9 base, Debian 12, etc.).
 *
 * The pyke-built ONNX Runtime binaries that ort-sys downloads are compiled
 * on a glibc >= 2.38 host; on that host <stdlib.h> macro-redirects
 * strtol/strtoll/strtoul/strtoull (and their _l variants) to __isoc23_*
 * exports new in glibc 2.38. Those redirected calls show up as undefined
 * references inside libort_sys's .rlib and propagate into our final cdylib,
 * which then fails to dlopen on any glibc < 2.38 — even though our wheel is
 * tagged manylinux_2_34. auditwheel does not catch this because the
 * __isoc23_* exports are unversioned, so the dynamic VERNEED table still
 * caps at GLIBC_2.34.
 *
 * Only compile these when building against a glibc that actually lacks the
 * symbols (< 2.38, which is what the manylinux_2_34 release container has).
 * On a newer glibc the C library already provides strong __isoc23_* exports;
 * adding our own would shadow them process-wide via symbol interposition
 * (and has caused a SIGSEGV in the `cargo test` build on Ubuntu 24.04) — so
 * on >= 2.38 this is an empty translation unit and the static lib the build
 * links is a harmless no-op.
 *
 * The C23 versions only differ from the originals by accepting `0b...`
 * binary-literal prefixes when base is 0 or 2; ORT's internal callers parse
 * decimal/hex model metadata and do not exercise that path.
 */

#define _GNU_SOURCE
#include <features.h>

#if defined(__GLIBC__) && !__GLIBC_PREREQ(2, 38)

#include <locale.h>
#include <stdlib.h>

long __isoc23_strtol(const char *nptr, char **endptr, int base) {
    return strtol(nptr, endptr, base);
}

long long __isoc23_strtoll(const char *nptr, char **endptr, int base) {
    return strtoll(nptr, endptr, base);
}

unsigned long __isoc23_strtoul(const char *nptr, char **endptr, int base) {
    return strtoul(nptr, endptr, base);
}

unsigned long long __isoc23_strtoull(const char *nptr, char **endptr, int base) {
    return strtoull(nptr, endptr, base);
}

long long __isoc23_strtoll_l(const char *nptr, char **endptr, int base, locale_t loc) {
    return strtoll_l(nptr, endptr, base, loc);
}

unsigned long long __isoc23_strtoull_l(const char *nptr, char **endptr, int base, locale_t loc) {
    return strtoull_l(nptr, endptr, base, loc);
}

#endif /* glibc < 2.38 */
