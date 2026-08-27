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

#include "tensorflow/compiler/jit/disable_functional_ops_lowering_for_xla_pass.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/jit/compilability_check_util.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

namespace {

// Returns the effective global JIT level for `options`. Unlike
// `GetGlobalJitLevelForGraph`, this also works when `options.session_options`
// is null, which can happen when auto-clustering is enabled purely via the
// TF_XLA_FLAGS=--tf_xla_auto_jit=N environment variable and the execution
// path never populates a SessionOptions. In that case we combine just the
// flag-provided auto-jit level with whether the graph is a single-GPU graph,
// mirroring what `GetGlobalJitLevelForGraph` does for a DEFAULT ConfigProto
// setting.
OptimizerOptions::GlobalJitLevel GetEffectiveGlobalJitLevel(
    const GraphOptimizationPassOptions& options, const Graph& graph) {
  if (options.session_options != nullptr) {
    return GetGlobalJitLevelForGraph(options);
  }

  const XlaAutoJitFlag& auto_jit_flag =
      GetMarkForCompilationPassFlags()->xla_auto_jit_flag;
  auto level_or_off = [](int32_t level) {
    return level == OptimizerOptions::DEFAULT
               ? OptimizerOptions::OFF
               : static_cast<OptimizerOptions::GlobalJitLevel>(level);
  };
  return IsSingleGpuGraph(graph)
             ? level_or_off(auto_jit_flag.optimization_level_single_gpu)
             : level_or_off(auto_jit_flag.optimization_level_general);
}

// Builds one compilability checker per candidate device type, mirroring the
// filter MarkForCompilationPass uses during auto-clustering. Returns an empty
// vector if some candidate device type has no XLA backend, in which case the
// caller must conservatively keep the lowering: the functional op might be
// placed on that device, where it can never be clustered.
std::vector<RecursiveCompilabilityChecker> BuildCompilabilityCheckers(
    const GraphOptimizationPassOptions& options) {
  // Make sure the XLA kernels are registered before querying them, the same
  // way MarkForCompilationPass does before its compilability checks.
  XlaOpRegistry::RegisterCompilationKernels();

  std::vector<DeviceType> device_types;
  bool skip_unregistered_device_types = false;
  if (options.device_set != nullptr) {
    device_types = options.device_set->PrioritizedDeviceTypeList();
  } else {
    // No device information (e.g. some unit-test paths). Fall back to checking
    // against the CPU and GPU JIT devices, skipping any that is not linked
    // into this binary.
    device_types.emplace_back(DEVICE_CPU);
    device_types.emplace_back(DEVICE_GPU);
    skip_unregistered_device_types = true;
  }

  std::vector<RecursiveCompilabilityChecker> checkers;
  checkers.reserve(device_types.size());
  for (const DeviceType& device_type : device_types) {
    const XlaOpRegistry::DeviceRegistration* registration;
    if (!XlaOpRegistry::GetCompilationDevice(device_type.type(),
                                             &registration)) {
      if (skip_unregistered_device_types) continue;
      return {};
    }
    RecursiveCompilabilityChecker::OperationFilter filter =
        CreateOperationFilter(*registration);
    // Match the restrictions MarkForCompilationPass applies during
    // auto-clustering, so we never skip the lowering for an op that
    // auto-clustering would then refuse to compile.
    filter.require_always_compilable = true;
    filter.allow_string_consts = false;
    filter.allow_collective_reduce_v2 = false;
    filter.allow_unique_op = false;
    checkers.emplace_back(std::move(filter),
                          DeviceType(registration->compilation_device_name));
  }
  return checkers;
}

}  // namespace

absl::Status DisableFunctionalOpsLoweringForXlaPass::Run(
    const GraphOptimizationPassOptions& options) {
  if (options.graph == nullptr || options.graph->get() == nullptr) {
    return absl::OkStatus();
  }
  Graph* graph = options.graph->get();
  if (GetEffectiveGlobalJitLevel(options, *graph) < OptimizerOptions::ON_1) {
    return absl::OkStatus();
  }

  std::vector<Node*> candidates;
  for (Node* n : graph->op_nodes()) {
    if (!n->IsIfNode() && !n->IsCaseNode() && !n->IsWhileNode()) continue;
    bool lower = false;
    if (TryGetNodeAttr(n->attrs(), "_lower_using_switch_merge", &lower) &&
        lower) {
      candidates.push_back(n);
    }
  }
  if (candidates.empty()) {
    return absl::OkStatus();
  }

  std::vector<RecursiveCompilabilityChecker> checkers =
      BuildCompilabilityCheckers(options);
  if (checkers.empty()) {
    return absl::OkStatus();
  }

  const FunctionLibraryDefinition* flib_def =
      options.flib_def != nullptr ? options.flib_def : &graph->flib_def();
  Env* env = options.session_options != nullptr ? options.session_options->env
                                                : Env::Default();
  OptimizerOptions opts;
  auto pflr = std::make_unique<ProcessFunctionLibraryRuntime>(
      nullptr, env, /*config=*/nullptr, TF_GRAPH_DEF_VERSION, flib_def, opts);
  FunctionLibraryRuntime* lib_runtime =
      pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);

  // Only skip the lowering for ops that XLA can compile no matter which
  // device they end up placed on. If auto-clustering would reject the op
  // (e.g. a loop body containing ops without XLA kernels, tf.data ops, or
  // resource accesses in multi-device training), keep the attribute so
  // LowerFunctionalOpsPass lowers it to Switch/Merge as before; otherwise the
  // un-lowered op can fail placement with resource or reference edges that
  // span devices.
  for (Node* n : candidates) {
    bool compilable_on_all_devices =
        absl::c_all_of(checkers, [&](const RecursiveCompilabilityChecker& c) {
          return c.IsCompilableNode(*n, lib_runtime);
        });
    if (compilable_on_all_devices) {
      n->ClearAttr("_lower_using_switch_merge");
    }
  }
  return absl::OkStatus();
}

}  // namespace tensorflow
