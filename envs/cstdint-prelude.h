/* Force-included via CXXFLAGS for the habitat-sim build (see build.sh).
 * GCC 13+ dropped transitive <cstdint> includes; the vendored corrade/magnum
 * predate that and fail on std::uint32_t. Guarded so CMake's -std=c++98
 * feature test still compiles: <cstdint> requires C++11, and an unconditional
 * -include cstdint makes cmake 3.14 mark ALL compile features unsupported,
 * which silently breaks the OpenEXR subproject configuration.
 */
#if defined(__cplusplus) && __cplusplus >= 201103L
#include <cstdint>
#endif
