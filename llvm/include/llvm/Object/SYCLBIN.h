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

// Representation of a SYCLBIN binary object, extends OffloadBinary.
class SYCLBIN : public OffloadBinary {
public:
  SYCLBIN(const SYCLBIN &Other) = delete;
  SYCLBIN &operator=(const SYCLBIN &Other) = delete;

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
  };

};

} // namespace object

} // namespace llvm

#endif
