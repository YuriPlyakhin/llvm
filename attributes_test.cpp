// compile: clang++ -fsycl -fsycl-device-code-split=off attributes_test.cpp
// ONEAPI_DEVICE_SELECTOR=opencl:cpu ./a.out
#include <sycl/detail/core.hpp>
#include <sycl/sub_group.hpp>
#include <sycl/usm.hpp>
#include <iostream>
#include <string>
#include <functional>
#include <map>

using namespace sycl;

class WG16 {
 public:
  int *out;
  int *wg_size_out;
  int *sg_size_out;
  [[sycl::reqd_work_group_size(16)]] void operator()(nd_item<1> it) const {
    out[it.get_global_id(0)] = static_cast<int>(it.get_global_id(0)) * 2 + 1;
    if (it.get_local_id(0) == 0 && it.get_group(0) == 0) {
      *wg_size_out = static_cast<int>(it.get_local_range(0));
      *sg_size_out = static_cast<int>(it.get_sub_group().get_local_range()[0]);
    }
  };
};

class WG32 {
 public:
  int *out;
  int *wg_size_out;
  int *sg_size_out;
  [[sycl::reqd_work_group_size(32)]] void operator()(nd_item<1> it) const {
    out[it.get_global_id(0)] = static_cast<int>(it.get_global_id(0)) * 2 + 1;
    if (it.get_local_id(0) == 0 && it.get_group(0) == 0) {
      *wg_size_out = static_cast<int>(it.get_local_range(0));
      *sg_size_out = static_cast<int>(it.get_sub_group().get_local_range()[0]);
    }
  };
};

class WGBIG {
 public:
  int *out;
  int *wg_size_out;
  int *sg_size_out;
  [[sycl::reqd_work_group_size(16384)]] void operator()(nd_item<1> it) const {
    out[it.get_global_id(0)] = static_cast<int>(it.get_global_id(0)) * 2 + 1;
    if (it.get_local_id(0) == 0 && it.get_group(0) == 0) {
      *wg_size_out = static_cast<int>(it.get_local_range(0));
      *sg_size_out = static_cast<int>(it.get_sub_group().get_local_range()[0]);
    }
  };
};

class SGSMALL {
 public:
  int *out;
  int *wg_size_out;
  int *sg_size_out;
  [[sycl::reqd_sub_group_size(2)]] void operator()(nd_item<1> it) const {
    out[it.get_global_id(0)] = static_cast<int>(it.get_global_id(0)) * 2 + 1;
    if (it.get_local_id(0) == 0 && it.get_group(0) == 0) {
      *wg_size_out = static_cast<int>(it.get_local_range(0));
      *sg_size_out = static_cast<int>(it.get_sub_group().get_local_range()[0]);
    }
  };
};

class SG16 {
 public:
  int *out;
  int *wg_size_out;
  int *sg_size_out;
  [[sycl::reqd_sub_group_size(16)]] void operator()(nd_item<1> it) const {
    out[it.get_global_id(0)] = static_cast<int>(it.get_global_id(0)) * 2 + 1;
    if (it.get_local_id(0) == 0 && it.get_group(0) == 0) {
      *wg_size_out = static_cast<int>(it.get_local_range(0));
      *sg_size_out = static_cast<int>(it.get_sub_group().get_local_range()[0]);
    }
  };
};

class SG32 {
 public:
  int *out;
  int *wg_size_out;
  int *sg_size_out;
  [[sycl::reqd_sub_group_size(32)]] void operator()(nd_item<1> it) const {
    out[it.get_global_id(0)] = static_cast<int>(it.get_global_id(0)) * 2 + 1;
    if (it.get_local_id(0) == 0 && it.get_group(0) == 0) {
      *wg_size_out = static_cast<int>(it.get_local_range(0));
      *sg_size_out = static_cast<int>(it.get_sub_group().get_local_range()[0]);
    }
  };
};

class SGBIG {
 public:
  int *out;
  int *wg_size_out;
  int *sg_size_out;
  [[sycl::reqd_sub_group_size(128)]] void operator()(nd_item<1> it) const {
    out[it.get_global_id(0)] = static_cast<int>(it.get_global_id(0)) * 2 + 1;
    if (it.get_local_id(0) == 0 && it.get_group(0) == 0) {
      *wg_size_out = static_cast<int>(it.get_local_range(0));
      *sg_size_out = static_cast<int>(it.get_sub_group().get_local_range()[0]);
    }
  };
};

template <typename KernelT>
void run_kernel(queue &q, const std::string &name) {
    constexpr size_t N = KernelT::launch_range;
    constexpr size_t L = KernelT::local_size;
    std::cout << "Running kernel: " << name << " [range=" << N << " local=" << L << "]\n";
    int *buf = malloc_shared<int>(N, q);
    int *wg_size_dev = malloc_shared<int>(1, q);
    int *sg_size_dev = malloc_shared<int>(1, q);
    for (size_t i = 0; i < N; ++i) buf[i] = 0;
    *wg_size_dev = -1;
    *sg_size_dev = -1;
    try {
        KernelT k;
        k.out = buf;
        k.wg_size_out = wg_size_dev;
        k.sg_size_out = sg_size_dev;
        q.submit([&](handler &cgh) {
            cgh.parallel_for(nd_range<1>(N, L), k);
        }).wait_and_throw();
        int first = buf[0], last = buf[N - 1];
        bool ok = (first == 1) && (last == static_cast<int>((N - 1) * 2 + 1));
        std::cout << "  wg_size=" << *wg_size_dev
                  << " sg_size=" << *sg_size_dev << "\n";
        std::cout << "  OK first=" << first << " last=" << last
                  << (ok ? " (verified)" : " (MISMATCH)") << "\n";
    } catch (sycl::exception &e) {
        std::cout << "  FAILED: " << e.what() << "\n";
    }
    free(buf, q);
    free(wg_size_dev, q);
    free(sg_size_dev, q);
}

struct K_WG16   : WG16   { static constexpr size_t launch_range = 16;   static constexpr size_t local_size = 16;  };
struct K_WG32   : WG32   { static constexpr size_t launch_range = 32;   static constexpr size_t local_size = 32;  };
struct K_WGBIG  : WGBIG  { static constexpr size_t launch_range = 16384; static constexpr size_t local_size = 16384;};
struct K_SGSMALL    : SGSMALL    { static constexpr size_t launch_range = 64;   static constexpr size_t local_size = 64;  };
struct K_SG16   : SG16   { static constexpr size_t launch_range = 64;   static constexpr size_t local_size = 64;  };
struct K_SG32   : SG32   { static constexpr size_t launch_range = 64;   static constexpr size_t local_size = 64;  };
struct K_SGBIG   : SGBIG   { static constexpr size_t launch_range = 128;  static constexpr size_t local_size = 128; };

int main(int argc, char **argv) {
    queue q;
    device d = q.get_device();

    std::cout << "Device name: "
              << d.get_info<info::device::name>() << "\n";

    std::cout << "Max work-group size: "
              << d.get_info<info::device::max_work_group_size>() << "\n";

    std::cout << "Supported sub-group sizes:";
    for (size_t s : d.get_info<info::device::sub_group_sizes>())
        std::cout << " " << s;
    std::cout << "\n";

    std::map<std::string, std::function<void()>> runners = {
        {"WG16",  [&]{ run_kernel<K_WG16>(q,  "WG16");  }},
        {"WG32",  [&]{ run_kernel<K_WG32>(q,  "WG32");  }},
        {"WGBIG", [&]{ run_kernel<K_WGBIG>(q, "WGBIG"); }},
        {"SGSMALL",   [&]{ run_kernel<K_SGSMALL>(q,   "SGSMALL");   }},
        {"SG16",  [&]{ run_kernel<K_SG16>(q,  "SG16");  }},
        {"SG32",  [&]{ run_kernel<K_SG32>(q,  "SG32");  }},
        {"SGBIG",  [&]{ run_kernel<K_SGBIG>(q,  "SGBIG");  }},
    };

    for (int i = 1; i < argc; ++i) {
        std::string name = argv[i];
        auto it = runners.find(name);
        if (it == runners.end()) {
            std::cout << "Unknown kernel: " << name << "\n";
            continue;
        }
        it->second();
    }
    return 0;
}
