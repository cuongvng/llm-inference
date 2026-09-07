// `llm-devinfo` -- prints the device/VRAM baseline that PLAN.md's budget table
// is built on. `--kv` emits the one-line form meant for pasting into
// docs/vram_budget.md; `--out FILE` appends it to a log.
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>

#include "llm/device_info.h"

int main(int argc, char** argv) {
  bool kv_only = false;
  std::string out_path;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--kv") {
      kv_only = true;
    } else if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "-h" || arg == "--help") {
      std::printf("usage: llm-devinfo [--kv] [--out FILE]\n");
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }

  const llm::DeviceBaseline b = llm::query_device_baseline(0);
  const std::string kv = llm::format_baseline_kv(b);

  std::printf("%s", kv_only ? (kv + "\n").c_str() : llm::format_baseline(b).c_str());

  if (!out_path.empty()) {
    std::ofstream out(out_path, std::ios::app);
    if (!out) {
      std::fprintf(stderr, "cannot open %s for writing\n", out_path.c_str());
      return 1;
    }
    const std::time_t now = std::time(nullptr);
    char stamp[32] = {};
    std::strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
    out << stamp << " " << kv << "\n";
  }
  return 0;
}
