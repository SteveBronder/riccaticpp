#ifndef INCLUDE_RICCATI_MACROS_HPP
#define INCLUDE_RICCATI_MACROS_HPP

namespace riccati {

#ifdef __GNUC__ // 1

/**
 * If statements predicate tagged with this attribute are expected to
 * be true most of the time. This effects inlining decisions.
 */
#ifndef likely // 2
#define likely(x) __builtin_expect(!!(x), 1)
#endif // 2

#ifndef unlikely // 2
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif // 2

#ifdef __has_attribute // 2
#if __has_attribute(noinline) && __has_attribute(cold) // 3
#ifndef RICCATI_COLD_PATH // 4
/**
 * Functions tagged with this attribute are not inlined and moved
 *  to a cold branch to tell the CPU to not attempt to pre-fetch
 *  the associated function.
 */
#define RICCATI_COLD_PATH __attribute__((noinline, cold))
#endif // 4
#endif // 3
#endif // 2

#ifndef RICCATI_COLD_PATH // 2
#define RICCATI_COLD_PATH
#endif // 2
#ifndef RICCATI_NO_INLINE // 2
#define RICCATI_NO_INLINE __attribute__((noinline))
#endif // 2

#ifndef RICCATI_ALWAYS_INLINE // 2
#define RICCATI_ALWAYS_INLINE __attribute__((always_inline)) inline
#endif

/**
 * Functions tagged with this attribute are pure functions, i.e. they
 * do not modify any global state and only depend on their input arguments.
 */
#ifndef RICCATI_PURE
#define RICCATI_PURE __attribute__((pure))
#endif

#else
#define likely(x) (x)
#define unlikely(x) (x)
#define RICCATI_COLD_PATH
#define RICCATI_NO_INLINE
#define RICCATI_ALWAYS_INLINE
#define RICCATI_PURE
#endif



}  // namespace riccati

#endif
