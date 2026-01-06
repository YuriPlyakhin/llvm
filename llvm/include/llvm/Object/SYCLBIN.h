//===- SYCLBIN.h - SYCLBIN binary format support ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_SYCLBIN_H
#define LLVM_OBJECT_SYCLBIN_H

#include "llvm/ADT/SmallString.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/SYCLPostLink/ModuleSplitter.h"
#include "llvm/Support/MemoryBuffer.h"
#include <string>

namespace llvm {

namespace object {

// Representation of a SYCLBIN binary object.
class SYCLBIN {
public:
  SYCLBIN(const SYCLBIN &Other) = delete;
  SYCLBIN &operator=(const SYCLBIN &Other) = delete;

  uint32_t getVersion() const {
    assert(OffloadBinaries.size() > 0 &&
           "SYCLBIN should contain at least 1 offload binary.");
    return OffloadBinaries[0]->getVersion();
  }

  enum class BundleState : uint8_t { Input = 0, Object = 1, Executable = 2 };

  struct SYCLBINModuleDesc {
    std::string ArchString;
    llvm::Triple TargetTriple;
    std::vector<module_split::SplitModule> SplitModules;
  };

  class SYCLBINDesc {
  public:
    SYCLBINDesc(BundleState State, ArrayRef<SYCLBINModuleDesc> ModuleDescs);

    SYCLBINDesc(const SYCLBINDesc &Other) = delete;
    SYCLBINDesc(SYCLBINDesc &&Other) = default;

    SYCLBINDesc &operator=(const SYCLBINDesc &Other) = delete;
    SYCLBINDesc &operator=(SYCLBINDesc &&Other) = default;

  private:
    struct ImageDesc {
      SmallString<0> Metadata;
      SmallString<0> FilePath;
    };

    struct AbstractModuleDesc {
      SmallString<0> Metadata;
      SmallVector<ImageDesc, 4> IRModuleDescs;
      SmallVector<ImageDesc, 4> NativeDeviceCodeImageDescs;
    };

    std::unique_ptr<llvm::util::PropertySetRegistry> GlobalMetadata;
    SmallVector<AbstractModuleDesc, 4> AbstractModuleDescs;

    friend class SYCLBIN;
  };

  /// Serialize \p Desc.
  static SmallString<0> write(const SYCLBIN::SYCLBINDesc &Desc);

  /// Deserialize the contents of \p Source to produce a SYCLBIN object.
  static Expected<std::unique_ptr<SYCLBIN>> read(MemoryBufferRef Source);

  struct IRModule {
    std::unique_ptr<llvm::util::PropertySetRegistry> Metadata;
    StringRef RawIRBytes;
  };
  struct NativeDeviceCodeImage {
    std::unique_ptr<llvm::util::PropertySetRegistry> Metadata;
    StringRef RawDeviceCodeImageBytes;
  };

  struct AbstractModule {
    std::unique_ptr<llvm::util::PropertySetRegistry> Metadata;
    SmallVector<IRModule> IRModules;
    SmallVector<NativeDeviceCodeImage> NativeDeviceCodeImages;
  };

  std::unique_ptr<llvm::util::PropertySetRegistry> GlobalMetadata;
  SmallVector<AbstractModule, 4> AbstractModules;

  private:
  SmallVector<std::unique_ptr<OffloadBinary>> OffloadBinaries;
};

} // namespace object

} // namespace llvm

#endif
