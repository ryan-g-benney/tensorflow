/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_JIT_DISABLE_FUNCTIONAL_OPS_LOWERING_FOR_XLA_PASS_H_
#define TENSORFLOW_COMPILER_JIT_DISABLE_FUNCTIONAL_OPS_LOWERING_FOR_XLA_PASS_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {

// Auto-clustering lowers functional control flow ops (While/If/Case) into
// Switch/Merge before MarkForCompilationPass runs. That can cause only
// fragments of a loop/branch body to end up in a cluster instead of the whole
// op, which silently corrupts results for some patterns. jit_compile=True
// never lowers these ops in the first place, so it doesn't hit this.
//
// This pass clears the `_lower_using_switch_merge` attribute on functional
// control flow nodes whenever global JIT is enabled, so auto-clustering skips
// the lowering the same way jit_compile=True already does.
//
// The attribute is only cleared for nodes that XLA can compile on every
// candidate device (checked with RecursiveCompilabilityChecker, using the
// same operation filter as MarkForCompilationPass). Ops that auto-clustering
// would reject anyway — e.g. loop bodies containing ops without XLA kernels,
// tf.data ops, or resource accesses in multi-device training — keep the
// attribute and are lowered by LowerFunctionalOpsPass as before. Skipping
// the lowering for those would leave a functional op that can fail placement
// when resource or reference edges span devices (as seen with parameter
// server training).
class DisableFunctionalOpsLoweringForXlaPass : public GraphOptimizationPass {
 public:
  DisableFunctionalOpsLoweringForXlaPass() = default;

  absl::Status Run(const GraphOptimizationPassOptions& options) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_DISABLE_FUNCTIONAL_OPS_LOWERING_FOR_XLA_PASS_H_
