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
#include <vector>

#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

// An op with no XLA kernel, so any function body containing it is not
// compilable by XLA.
REGISTER_OP("DisableLoweringPassMissingXlaKernel")
    .Input("i: float")
    .Output("o: float");

using FDH = FunctionDefHelper;

// cond(x: float) -> bool: x < 10, built from ops with XLA kernels.
FunctionDef CompilableCond() {
  const Tensor kTen = test::AsScalar<float>(10.0f);
  return FDH::Define(
      "CompilableCond", {"x: float"}, {"r: bool"}, {},
      {
          {{"ten"}, "Const", {}, {{"value", kTen}, {"dtype", DT_FLOAT}}},
          {{"r"}, "Less", {"x", "ten"}, {{"T", DT_FLOAT}}},
      });
}

// body(x: float) -> float: x + 1, built from ops with XLA kernels.
FunctionDef CompilableBody() {
  const Tensor kOne = test::AsScalar<float>(1.0f);
  return FDH::Define(
      "CompilableBody", {"x: float"}, {"y: float"}, {},
      {
          {{"one"}, "Const", {}, {{"value", kOne}, {"dtype", DT_FLOAT}}},
          {{"y"}, "Add", {"x", "one"}, {{"T", DT_FLOAT}}},
      });
}

// body(x: float) -> float built around an op XLA cannot compile.
FunctionDef UncompilableBody() {
  return FDH::Define(
      "UncompilableBody", {"x: float"}, {"y: float"}, {},
      {
          {{"y"}, "DisableLoweringPassMissingXlaKernel", {"x"}, {}},
      });
}

std::unique_ptr<FunctionLibraryDefinition> MakeFlibDef() {
  FunctionDefLibrary flib;
  *flib.add_function() = CompilableCond();
  *flib.add_function() = CompilableBody();
  *flib.add_function() = UncompilableBody();
  return std::make_unique<FunctionLibraryDefinition>(OpRegistry::Global(),
                                                     flib);
}

NameAttrList FuncAttr(absl::string_view name) {
  NameAttrList fn;
  fn.set_name(std::string(name));
  return fn;
}

absl::Status AddPlaceholder(Graph* graph, absl::string_view name,
                            DataType dtype, Node** node) {
  return NodeBuilder(name, "Placeholder")
      .Attr("dtype", dtype)
      .Finalize(graph, node);
}

absl::Status AddWhileNode(Graph* graph, absl::string_view name, Node* input,
                          absl::string_view cond_fn, absl::string_view body_fn,
                          Node** node) {
  return NodeBuilder(name, "While")
      .Input(std::vector<NodeBuilder::NodeOut>{NodeBuilder::NodeOut(input, 0)})
      .Attr("_lower_using_switch_merge", true)
      .Attr("cond", FuncAttr(cond_fn))
      .Attr("body", FuncAttr(body_fn))
      .Attr("T", DataTypeVector{DT_FLOAT})
      .Finalize(graph, node);
}

absl::Status AddIfNode(Graph* graph, absl::string_view name, Node* cond,
                       Node* input, absl::string_view branch_fn, Node** node) {
  return NodeBuilder(name, "If")
      .Input(cond, 0)
      .Input(std::vector<NodeBuilder::NodeOut>{NodeBuilder::NodeOut(input, 0)})
      .Attr("_lower_using_switch_merge", true)
      .Attr("Tcond", DT_BOOL)
      .Attr("Tin", DataTypeVector{DT_FLOAT})
      .Attr("Tout", DataTypeVector{DT_FLOAT})
      .Attr("then_branch", FuncAttr(branch_fn))
      .Attr("else_branch", FuncAttr(branch_fn))
      .Finalize(graph, node);
}

absl::Status AddCaseNode(Graph* graph, absl::string_view name,
                         Node* branch_index, Node* input,
                         absl::string_view branch_fn, Node** node) {
  return NodeBuilder(name, "Case")
      .Input(branch_index, 0)
      .Input(std::vector<NodeBuilder::NodeOut>{NodeBuilder::NodeOut(input, 0)})
      .Attr("_lower_using_switch_merge", true)
      .Attr("Tin", DataTypeVector{DT_FLOAT})
      .Attr("Tout", DataTypeVector{DT_FLOAT})
      .Attr("branches", std::vector<NameAttrList>{FuncAttr(branch_fn)})
      .Finalize(graph, node);
}

bool HasLowerUsingSwitchMergeAttr(const Node* n) {
  bool lower = false;
  return TryGetNodeAttr(n->attrs(), "_lower_using_switch_merge", &lower) &&
         lower;
}

absl::Status RunPass(std::unique_ptr<Graph>* graph,
                     SessionOptions* session_options,
                     FunctionLibraryDefinition* flib_def) {
  GraphOptimizationPassOptions options;
  options.graph = graph;
  options.session_options = session_options;
  options.flib_def = flib_def;
  DisableFunctionalOpsLoweringForXlaPass pass;
  return pass.Run(options);
}

SessionOptions SessionOptionsWithJitLevel(
    OptimizerOptions::GlobalJitLevel level) {
  SessionOptions session_options;
  session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_global_jit_level(level);
  return session_options;
}

TEST(DisableFunctionalOpsLoweringForXlaPassTest,
     ClearsAttrOnCompilableNodesWhenGlobalJitEnabled) {
  auto flib_def = MakeFlibDef();
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  Node* x;
  TF_ASSERT_OK(AddPlaceholder(graph.get(), "x", DT_FLOAT, &x));
  Node* while_node;
  TF_ASSERT_OK(AddWhileNode(graph.get(), "my_while", x, "CompilableCond",
                            "CompilableBody", &while_node));
  Node* pred;
  TF_ASSERT_OK(AddPlaceholder(graph.get(), "pred", DT_BOOL, &pred));
  Node* if_node;
  TF_ASSERT_OK(
      AddIfNode(graph.get(), "my_if", pred, x, "CompilableBody", &if_node));
  Node* branch_index;
  TF_ASSERT_OK(
      AddPlaceholder(graph.get(), "branch_index", DT_INT32, &branch_index));
  Node* case_node;
  TF_ASSERT_OK(AddCaseNode(graph.get(), "my_case", branch_index, x,
                           "CompilableBody", &case_node));

  SessionOptions session_options =
      SessionOptionsWithJitLevel(OptimizerOptions::ON_1);
  TF_ASSERT_OK(RunPass(&graph, &session_options, flib_def.get()));

  EXPECT_FALSE(HasLowerUsingSwitchMergeAttr(while_node));
  EXPECT_FALSE(HasLowerUsingSwitchMergeAttr(if_node));
  EXPECT_FALSE(HasLowerUsingSwitchMergeAttr(case_node));
}

// Regression test: a loop whose body XLA cannot compile (e.g. tf.data ops or
// parameter server variable access) must keep the attribute, so that
// LowerFunctionalOpsPass still lowers it to Switch/Merge. Otherwise the
// un-lowered While op can fail placement when resource or reference edges
// span devices.
TEST(DisableFunctionalOpsLoweringForXlaPassTest,
     PreservesAttrOnUncompilableWhileWhenGlobalJitEnabled) {
  auto flib_def = MakeFlibDef();
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  Node* x;
  TF_ASSERT_OK(AddPlaceholder(graph.get(), "x", DT_FLOAT, &x));
  Node* while_node;
  TF_ASSERT_OK(AddWhileNode(graph.get(), "my_while", x, "CompilableCond",
                            "UncompilableBody", &while_node));

  SessionOptions session_options =
      SessionOptionsWithJitLevel(OptimizerOptions::ON_2);
  TF_ASSERT_OK(RunPass(&graph, &session_options, flib_def.get()));

  EXPECT_TRUE(HasLowerUsingSwitchMergeAttr(while_node));
}

TEST(DisableFunctionalOpsLoweringForXlaPassTest,
     PreservesAttrWhenGlobalJitDisabled) {
  auto flib_def = MakeFlibDef();
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  Node* x;
  TF_ASSERT_OK(AddPlaceholder(graph.get(), "x", DT_FLOAT, &x));
  Node* while_node;
  TF_ASSERT_OK(AddWhileNode(graph.get(), "my_while", x, "CompilableCond",
                            "CompilableBody", &while_node));

  SessionOptions session_options =
      SessionOptionsWithJitLevel(OptimizerOptions::OFF);
  TF_ASSERT_OK(RunPass(&graph, &session_options, flib_def.get()));

  EXPECT_TRUE(HasLowerUsingSwitchMergeAttr(while_node));
}

TEST(DisableFunctionalOpsLoweringForXlaPassTest, NoOpWhenGraphIsMissing) {
  auto flib_def = MakeFlibDef();
  SessionOptions session_options =
      SessionOptionsWithJitLevel(OptimizerOptions::ON_2);

  GraphOptimizationPassOptions options;
  options.session_options = &session_options;
  options.flib_def = flib_def.get();
  DisableFunctionalOpsLoweringForXlaPass pass;
  TF_ASSERT_OK(pass.Run(options));
}

class DisableFunctionalOpsLoweringForXlaPassFlagsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    flags_ = GetMarkForCompilationPassFlags();
    original_ = flags_->xla_auto_jit_flag;
  }
  void TearDown() override { flags_->xla_auto_jit_flag = original_; }

  MarkForCompilationPassFlags* flags_;
  XlaAutoJitFlag original_;
};

TEST_F(DisableFunctionalOpsLoweringForXlaPassFlagsTest,
       FallsBackToFlagsWhenSessionOptionsIsNull) {
  auto flib_def = MakeFlibDef();
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  Node* x;
  TF_ASSERT_OK(AddPlaceholder(graph.get(), "x", DT_FLOAT, &x));
  Node* while_node;
  TF_ASSERT_OK(AddWhileNode(graph.get(), "my_while", x, "CompilableCond",
                            "CompilableBody", &while_node));

  flags_->xla_auto_jit_flag.optimization_level_single_gpu =
      OptimizerOptions::ON_2;
  flags_->xla_auto_jit_flag.optimization_level_general = OptimizerOptions::ON_2;

  TF_ASSERT_OK(RunPass(&graph, /*session_options=*/nullptr, flib_def.get()));

  EXPECT_FALSE(HasLowerUsingSwitchMergeAttr(while_node));
}

TEST_F(DisableFunctionalOpsLoweringForXlaPassFlagsTest,
       PreservesAttrWhenSessionOptionsIsNullAndFlagsAreOff) {
  auto flib_def = MakeFlibDef();
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  Node* x;
  TF_ASSERT_OK(AddPlaceholder(graph.get(), "x", DT_FLOAT, &x));
  Node* while_node;
  TF_ASSERT_OK(AddWhileNode(graph.get(), "my_while", x, "CompilableCond",
                            "CompilableBody", &while_node));

  flags_->xla_auto_jit_flag.optimization_level_single_gpu =
      OptimizerOptions::OFF;
  flags_->xla_auto_jit_flag.optimization_level_general = OptimizerOptions::OFF;

  TF_ASSERT_OK(RunPass(&graph, /*session_options=*/nullptr, flib_def.get()));

  EXPECT_TRUE(HasLowerUsingSwitchMergeAttr(while_node));
}

}  // namespace
}  // namespace tensorflow
