#pragma once

#include <memory>
#include <string>

#include <pfvm/compiler/compiler_context.h>
#include <pfvm/compiler/custom_onnx_ops.h>
#include <pfvm/compiler/model.h>
#include <pfvm/compiler/onnx.h>
#include <pfvm/tools/util.h>

namespace pfvm {
namespace runtime {

inline std::unique_ptr<Model> CompileModel(const std::string& onnx_filename, const bool use_gradient) {
    RegisterCustomOnnxOperatorSetSchema();
    onnx::ModelProto xmodel(LoadLargeProto<onnx::ModelProto>(onnx_filename));
    auto model = std::make_unique<Model>(xmodel);
    Graph* graph = model->mutable_graph();
    g_fixed_batch_norm = true;
    // To avoid Clip-11 and Pad-11 which ONNX-chainer does not support yet.
    // TODO(hamaji): Remove this after updating ONNX-chainer.
    g_opset_version = 9;
    RunDefaultPassesBeforeGradient(graph);
    if (use_gradient) {
        GenerateGradientNodesTo(graph, graph, {"vc"});
    }
    g_skip_inference = true;
    {
        CompilerContext ctx(graph);
        RunDefaultPasses(ctx);
    }
    return model;
}

}  // namespace runtime
}  // namespace pfvm
