//===- SYCLBIN.cpp - SYCLBIN binary format support --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/SYCLBIN.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Support/Error.h"

using namespace llvm;
using namespace llvm::object;
using OffloadingImage = OffloadBinary::OffloadingImage;

SYCLBIN::SYCLBINDesc::SYCLBINDesc(BundleState State,
                                  ArrayRef<SYCLBINModuleDesc> ModuleDescs) {
  // Write global metadata.
  GlobalMetadata = std::make_unique<llvm::util::PropertySetRegistry>();
  GlobalMetadata->add(llvm::util::PropertySetRegistry::SYCLBIN_GLOBAL_METADATA,
                      "state", static_cast<uint32_t>(State));

  // We currently create a single abstract module per split module.
  // Some of these should be merged in the future.
  size_t NumAMs = 0;
  for (const SYCLBINModuleDesc &MD : ModuleDescs)
    NumAMs += MD.SplitModules.size();
  AbstractModuleDescs.reserve(NumAMs);

  for (const SYCLBINModuleDesc &MD : ModuleDescs) {
    for (const module_split::SplitModule &SM : MD.SplitModules) {
      AbstractModuleDesc &AMD = AbstractModuleDescs.emplace_back();

      // Write module metadata to the abstract module metadata.
      raw_svector_ostream AMMetadataOS(AMD.Metadata);
      SM.Properties.write(AMMetadataOS);

      ImageDesc ID;
      // Copy the filepath.
      ID.FilePath = SM.ModuleFilePath;

      // Create metadata and save the descriptor to the right collection.
      raw_svector_ostream IDMetadataOS(ID.Metadata);
      if (MD.ArchString.empty()) {
        // If the arch string is empty, it must be an IR module.
        llvm::util::PropertySetRegistry IRMMetadata;
        // TODO: Determine type from the input.
        IRMMetadata.add(
            llvm::util::PropertySetRegistry::SYCLBIN_IR_MODULE_METADATA, "type",
            /*SPIR-V*/ 0);
        IRMMetadata.add(
            llvm::util::PropertySetRegistry::SYCLBIN_IR_MODULE_METADATA,
            "target", MD.TargetTriple.str());
        IRMMetadata.write(IDMetadataOS);
        AMD.IRModuleDescs.emplace_back(std::move(ID));
      } else {
        // If the arch string is empty, it must be an native device code image.
        llvm::util::PropertySetRegistry NDCIMetadata;
        NDCIMetadata.add(llvm::util::PropertySetRegistry::
                             SYCLBIN_NATIVE_DEVICE_CODE_IMAGE_METADATA,
                         "arch", MD.ArchString);
        NDCIMetadata.add(llvm::util::PropertySetRegistry::
                             SYCLBIN_NATIVE_DEVICE_CODE_IMAGE_METADATA,
                         "target", MD.TargetTriple.str());
        NDCIMetadata.write(IDMetadataOS);
        AMD.NativeDeviceCodeImageDescs.emplace_back(std::move(ID));
      }
    }
  }
}

// TODO: update interface to return Error.
SmallString<0> SYCLBIN::write(const SYCLBIN::SYCLBINDesc &Desc) {
  SmallVector<OffloadingImage> Images;
  SmallVector<SmallString<128>> Buffers;

  // Write global metadata image.
  OffloadingImage GlobalMDI{};
  GlobalMDI.TheOffloadKind = OffloadKind::OFK_SYCL;
  GlobalMDI.Flags = OIF_NoImage;
  Desc.GlobalMetadata->write(GlobalMDI.StringData, Buffers);
  Images.emplace_back(GlobalMDI);

  for (const SYCLBINDesc::AbstractModuleDesc &AMD : Desc.AbstractModuleDescs) {
    // Store IR modules.
    for (const SYCLBINDesc::ImageDesc &IRMD : AMD.IRModuleDescs) {
      OffloadingImage OI{};

      auto FileBufferOrError =
          llvm::MemoryBuffer::getFileOrSTDIN(IRMD.FilePath);
      if (!FileBufferOrError)
        return createFileError(IRMD.FilePath, FileBufferOrError.getError());
      OI.Image = std::move(*FileBufferOrError);

      Images.emplace_back(OI);
    }

    // Store native device code images.
    for (const SYCLBINDesc::ImageDesc &NDCID : AMD.NativeDeviceCodeImageDescs) {
      OffloadingImage OI{};

      auto FileBufferOrError =
          llvm::MemoryBuffer::getFileOrSTDIN(NDCID.FilePath);
      if (!FileBufferOrError)
        return createFileError(NDCID.FilePath, FileBufferOrError.getError());
      OI.Image = std::move(*FileBufferOrError);

      Images.emplace_back(OI);
    }
  }

  // Write abstract module metadata.
  for (const SYCLBINDesc::AbstractModuleDesc &AMD : Desc.AbstractModuleDescs)
    OS << AMD.Metadata;

  // Write IR module metadata.
  for (const SYCLBINDesc::AbstractModuleDesc &AMD : Desc.AbstractModuleDescs)
    for (const SYCLBINDesc::ImageDesc &IRMD : AMD.IRModuleDescs)
      OS << IRMD.Metadata;

  // Write native device code image metadata.
  for (const SYCLBINDesc::AbstractModuleDesc &AMD : Desc.AbstractModuleDescs)
    for (const SYCLBINDesc::ImageDesc &NDCID : AMD.NativeDeviceCodeImageDescs)
      OS << NDCID.Metadata;

  return OffloadBinary::write(Images);
}

Expected<std::unique_ptr<SYCLBIN>> SYCLBIN::read(MemoryBufferRef Source) {

}