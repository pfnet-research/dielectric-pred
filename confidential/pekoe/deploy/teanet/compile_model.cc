#include <stdio.h>
#include <stdlib.h>

#include <string>

#include <pfvm/common/log.h>
#include <pfvm/common/protoutil.h>
#include <pfvm/common/strutil.h>
#include <pfvm/compiler/atenvm/emitter.h>
#include <pfvm/compiler/flags.h>
#include <pfvm/compiler/gradient.h>
#include <pfvm/compiler/model.h>
#include <pfvm/compiler/onnx.h>
#include <pfvm/compiler/passes.h>
#include <pfvm/runtime/pfvm.h>
#include <pfvm/runtime/pfvm_var.h>
#include <pfvm/runtime/serializer.h>
#include <pfvm/tools/util.h>

#include "obfuscator/io_torch.h"

#include "compile_model.h"

namespace pfvm {
namespace runtime {
namespace {

void ExtractParams(const std::string& model_onnx,
                   const std::string& output_filename,
                   bool use_gradient) {
    std::unique_ptr<Model> model(CompileModel(model_onnx, use_gradient));

    {
        onnx::ModelProto xmodel;
        model->ToONNX(&xmodel);
        const std::string& out_onnx = output_filename + ".onnx";
        std::ofstream ofs(out_onnx);
        CHECK(ofs) << "Failed to open output ONNX: " << out_onnx;
        CHECK(xmodel.SerializeToOstream(&ofs));
    }

    InOuts params(LoadParams(model->graph()));
    std::map<std::string, std::string> renamed_params;
    {
        size_t i = 0;
        for (const auto& p : params) {
            CHECK(renamed_params.emplace(p.first, StrCat(i)).second);
            ++i;
        }
    }

    PFVMProgramProto pfvm_prog;
    atenvm::Emit(CompilerContext(model->mutable_graph()), &pfvm_prog, false);
    StripPFVMProgram(&pfvm_prog);

    for (auto&& input_sig : pfvm_prog.input_sigs) {
        std::string* input_name = &input_sig.name;
        auto found = renamed_params.find(*input_name);
        if (found == renamed_params.end()) {
            continue;
        }
        *input_name = found->second;
    }
    for (auto&& inst : pfvm_prog.instructions) {
        if (inst.op != PFVMOpcode::In) {
            continue;
        }
        PFVMValueProto* input = &inst.inputs[0];
        CHECK(!input->s().empty());
        auto found = renamed_params.find(input->s());
        if (found == renamed_params.end()) {
            continue;
        }
        input->set_s(found->second);
    }

    {
        const std::string out_chxvm = output_filename + ".chxvm";
        std::ofstream ofs(out_chxvm);
        CHECK(ofs) << "Failed to open output PFVM: " << out_chxvm;
        SerializeProgram(pfvm_prog, ofs);
    }

    FILE* fp = fopen((output_filename + ".params").c_str(), "wb");
    DumpInt(fp, params.size());
    for (const auto& p : params) {
        auto found = renamed_params.find(p.first);
        CHECK(found != renamed_params.end()) << p.first;
        const at::Tensor& a = p.second.GetArray();
        DumpString(fp, found->second);
        DumpArray(fp, a);
    }
    fclose(fp);
}

}  // namespace
}  // namespace runtime
}  // namespace pfvm

int main(int argc, const char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s model.onnx target (0 or 1)", argv[0]);
        exit(1);
    }
    pfvm::runtime::ExtractParams(argv[1], argv[2], atoi(argv[3]));
}
