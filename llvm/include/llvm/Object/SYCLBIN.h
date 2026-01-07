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
      ImageKind TheImageKind = ImageKind::IMG_None;
      llvm::Triple TargetTriple;
      std::string ArchString;
      SmallString<0> FilePath;
    };

    struct AbstractModuleDesc {
      std::unique_ptr<llvm::util::PropertySetRegistry> Metadata;
      SmallVector<ImageDesc, 4> IRModuleDescs;
      SmallVector<ImageDesc, 4> NativeDeviceCodeImageDescs;
    };

    std::unique_ptr<llvm::util::PropertySetRegistry> GlobalMetadata;
    SmallVector<AbstractModuleDesc, 4> AbstractModuleDescs;

    friend class SYCLBIN;
  };

  /// Serialize \p Desc.
  static Error write(const SYCLBIN::SYCLBINDesc &Desc, raw_ostream &OS);

  /// Deserialize the contents of \p Source to produce a SYCLBIN object.
  static Expected<std::unique_ptr<SYCLBIN>> read(MemoryBufferRef Source);

private:
  SYCLBIN() {}
  SYCLBIN(SmallVector<std::unique_ptr<OffloadBinary>> OB)
      : OffloadBinaries(std::move(OB)) {}
  SmallVector<std::unique_ptr<OffloadBinary>> OffloadBinaries;

  /// The current version of the binary used for backwards compatibility.
  static constexpr uint32_t
      [[deprecated("Use OffloadBinary format instead.")]] CurrentVersion = 1;

  /// Magic number used to identify SYCLBIN files.
  static constexpr uint32_t
      [[deprecated("Use OffloadBinary format instead.")]] MagicNumber =
          0x53594249;

  struct [[deprecated("Use OffloadBinary format instead.")]] alignas(8)
      FileHeaderType {
    uint32_t Magic;
    uint32_t Version;
    uint32_t AbstractModuleCount;
    uint32_t IRModuleCount;
    uint32_t NativeDeviceCodeImageCount;
    uint64_t MetadataByteTableSize;
    uint64_t BinaryByteTableSize;
    uint64_t GlobalMetadataOffset;
    uint64_t GlobalMetadataSize;
  };

  struct [[deprecated("Use OffloadBinary format instead.")]] alignas(8)
      AbstractModuleHeaderType {
    uint64_t MetadataOffset;
    uint64_t MetadataSize;
    uint32_t IRModuleCount;
    uint32_t IRModuleOffset;
    uint32_t NativeDeviceCodeImageCount;
    uint32_t NativeDeviceCodeImageOffset;
  };

  struct [[deprecated("Use OffloadBinary format instead.")]] alignas(8)
      IRModuleHeaderType {
    uint64_t MetadataOffset;
    uint64_t MetadataSize;
    uint64_t RawIRBytesOffset;
    uint64_t RawIRBytesSize;
  };

  struct [[deprecated("Use OffloadBinary format instead.")]] alignas(8)
      NativeDeviceCodeImageHeaderType {
    uint64_t MetadataOffset;
    uint64_t MetadataSize;
    uint64_t BinaryBytesOffset;
    uint64_t BinaryBytesSize;
  };
};

} // namespace object

} // namespace llvm

#endif
