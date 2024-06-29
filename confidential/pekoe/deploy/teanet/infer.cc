#include <assert.h>
#include <sys/mman.h>

#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <sstream>
#include <tuple>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ATen/Functions.h>
#include <ATen/TensorIndexing.h>
#include <ATen/Tensor.h>
#include <ATen/core/DimVector.h>

#include <pfvm/common/strutil.h>
#include <pfvm/runtime/aten_util.h>
#include <pfvm/runtime/pfvm.h>
#include <pfvm/runtime/pfvm_var.h>
#include <pfvm/runtime/serializer.h>

#include "obfuscator/coder.h"
#include "obfuscator/io_torch.h"

#ifdef DEV
#include <pfvm/common/protoutil.h>
#include <pfvm/compiler/atenvm/emitter.h>
#include <pfvm/compiler/dtype.h>
#include <pfvm/compiler/flags.h>
#include <pfvm/compiler/gradient.h>
#include <pfvm/compiler/graph.h>
#include <pfvm/compiler/model.h>
#include <pfvm/compiler/onnx.h>
#include <pfvm/compiler/passes.h>
#include <pfvm/tools/util.h>
#include "compile_model.h"
#endif

extern "C" {
    extern uint8_t _binary_target_chxvm_enc_start[];
    extern uint8_t _binary_target_chxvm_enc_end[];
    extern uint8_t _binary_target_params_enc_start[];
    extern uint8_t _binary_target_params_enc_end[];
    extern uint8_t _binary_target_forward_chxvm_enc_start[];
    extern uint8_t _binary_target_forward_chxvm_enc_end[];
    extern uint8_t _binary_target_forward_params_enc_start[];
    extern uint8_t _binary_target_forward_params_enc_end[];
    extern uint8_t rodata_start[];
    extern uint8_t rodata_end[];
}

namespace py = pybind11;
using py::operator""_a;

using namespace pfvm;
using namespace pfvm::runtime;

namespace {

int g_trace = 0;
std::string g_dump_outputs_dir;

}

at::DimVector ToChxShape(py::array x) {
    at::DimVector shape;
    for (int i = 0; i < x.ndim(); ++i) {
        shape.push_back(x.shape()[i]);
    }
    return shape;
}

at::ScalarType ToChxDtype(const py::dtype& npdtype) {
    switch (npdtype.kind()) {
        case 'b':
            return at::kBool;
        case 'i':
            switch (npdtype.itemsize()) {
                case 1:
                    return at::kChar;
                case 2:
                    return at::kShort;
                case 4:
                    return at::kInt;
                case 8:
                    return at::kLong;
                default:
                    break;
            }
            break;
        case 'u':
            switch (npdtype.itemsize()) {
                case 1:
                    return at::kByte;
                default:
                    break;
            }
            break;
        case 'f':
            switch (npdtype.itemsize()) {
                case 2:
                    return at::kHalf;
                case 4:
                    return at::kFloat;
                case 8:
                    return at::kDouble;
                default:
                    break;
            }
            break;
        default:
            break;
    }
    // TODO(hamaji): Support other dtypes.
    fprintf(stderr, "Unsupported dtype in input: %c\n", npdtype.kind());
    abort();
}

std::vector<ssize_t> FromChxShape(at::IntArrayRef s) {
    return std::vector<ssize_t>(s.begin(), s.end());
}

py::dtype FromChxDtype(at::ScalarType t) {
    switch (t) {
        case at::kInt:
            return py::dtype::of<int32_t>();
        case at::kLong:
            return py::dtype::of<int64_t>();
        case at::kFloat:
            return py::dtype::of<float>();
        case at::kDouble:
            return py::dtype::of<double>();
        default:
            // TODO(hamaji): Support other dtypes.
            fprintf(stderr, "Unsupported dtype in output: %d\n",
                    static_cast<int>(t));
            abort();
    }
}

at::Tensor MakeArray(py::array x) {
    py::buffer_info info = x.request();
    at::DimVector shape = ToChxShape(x);
    at::ScalarType dtype = ToChxDtype(x.dtype());

    at::Tensor a = at::empty(shape, dtype);
    memcpy(a.data_ptr(), info.ptr, a.nbytes());
    a = a.to(GetDefaultDevice());
    return a;
}

PFVMVar MakeInputVar(py::array x) {
    return PFVMVar(MakeArray(x));
}

PFVMVar MakeInputSeqVar(const std::vector<py::array>& xs) {
    auto arrays = std::make_shared<PFVMSequence>();
    for (py::array x : xs) {
        arrays->emplace_back(MakeArray(x));
    }
    return PFVMVar(arrays);
}

py::array MakeOutputArray(at::Tensor a) {
    a = a.cpu();
    py::dtype dtype = FromChxDtype(a.scalar_type());
    std::vector<ssize_t> shape = FromChxShape(a.sizes());
    py::array array(dtype, shape, a.data_ptr());
    return array;
}

enum ErrorEnumCC {
    NoError = 0,
    AtomsFarAwayError = 1,
    CellTooSmallError = 2,
    GhostAtomsTooManyError = 3,
};

class ModelCC {
public:
    explicit ModelCC(int gpu_id, bool _use_gradient) : use_gradient(_use_gradient) {
        std::lock_guard<std::mutex> lock(mutex);

        at::Device device(StrCat("cuda:", gpu_id));
        SetDefaultDevice(device);

#ifdef DEV
        fprintf(stderr, "Loading model (dev)...\n");

        const std::string& model_onnx = "target_onnx/model.onnx";
        if (g_trace) {
            g_compiler_log = true;
        }
        std::unique_ptr<Model> model(CompileModel(model_onnx, this->use_gradient));
        if (g_trace) {
            std::cerr << model->graph().DebugString() << std::endl;
        }
        inputs = LoadParams(model->graph());

        PFVMProgramProto program;
        atenvm::Emit(CompilerContext(model->mutable_graph()), &program, false);
        chxvm.reset(new PFVM(program));
#else
        uint8_t *chxvm_start, *chxvm_end, *params_start, *params_end;
        uint32_t model_seed, params_seed;
        if (this->use_gradient) {
            chxvm_start = _binary_target_chxvm_enc_start;
            chxvm_end = _binary_target_chxvm_enc_end;
            params_start = _binary_target_params_enc_start;
            params_end = _binary_target_params_enc_end;
            model_seed = MODEL_SEED;
            params_seed = PARAMS_SEED;
        } else {
            chxvm_start = _binary_target_forward_chxvm_enc_start;
            chxvm_end = _binary_target_forward_chxvm_enc_end;
            params_start = _binary_target_forward_params_enc_start;
            params_end = _binary_target_forward_params_enc_end;
            model_seed = MODEL_FORWARD_SEED;
            params_seed = PARAMS_FORWARD_SEED;
        }

        PFVMProgramProto program;
        DecodeRange(model_seed,
                    chxvm_start,
                    chxvm_end);
        {
            std::string buf(chxvm_start, chxvm_end);
            std::istringstream iss(buf);
            DeserializeProgram(iss, &program);
        }
        EncodeRange(model_seed,
                    chxvm_start,
                    chxvm_end);
        chxvm.reset(new PFVM(program));

        DecodeRange(params_seed,
                    params_start,
                    params_end);
        uint8_t* p = params_start;
        int64_t num_params = LoadInt(&p);
        // fprintf(stderr, "num_params=%d\n", (int)num_params);
        for (int64_t i = 0; i < num_params; ++i) {
            const std::string& name = LoadString(&p);
            // fprintf(stderr, "name=%s\n", name.c_str());
            const at::Tensor& a = LoadArray(&p).to(GetDefaultDevice());
            inputs.emplace(name, PFVMVar(a));
        }
        EncodeRange(params_seed,
                    params_start,
                    params_end);
#endif
    }


    std::tuple<at::Tensor, at::Tensor, at::Tensor> Run(
        const at::Tensor& vc_at,
        const at::Tensor& sp_at,
        const at::Tensor& a1_at,
        const at::Tensor& a2_at,
        const at::Tensor& batch_at,
        const at::Tensor& batch_edge_at,
        const int n_batch,
        const at::Tensor& x_add,
        const at::Tensor& calc_mode_type
    ) {
        at::Tensor out_batch = at::zeros({n_batch}, at::TensorOptions(at::kDouble).device(GetDefaultDevice()));

        inputs["vc"] = PFVMVar(vc_at);
        inputs["sp"] = PFVMVar(sp_at);
        inputs["a1"] = PFVMVar(a1_at);
        inputs["a2"] = PFVMVar(a2_at);
        inputs["ob"] = PFVMVar(out_batch);
        inputs["ba"] = PFVMVar(batch_at);
        inputs["be"] = PFVMVar(batch_edge_at);
        inputs["xa"] = PFVMVar(x_add);
        inputs["cm"] = PFVMVar(calc_mode_type);

        at::ScalarType dtype = vc_at.scalar_type();
#ifdef DEV
        // std::cerr << "batch_size=" << batch_size << std::endl;
#endif
        if (this->use_gradient) {
            at::Tensor grad_e = at::ones({n_batch, 1}, at::TensorOptions(dtype).device(GetDefaultDevice()));
            at::Tensor grad_c = at::zeros({sp_at.size(0), 1}, at::TensorOptions(dtype).device(GetDefaultDevice()));
            inputs["grad_in@e"] = PFVMVar(grad_e);
            inputs["grad_in@c"] = PFVMVar(grad_c);
        }

        PFVMOptions options;
#ifdef DEV
        options.trace_level = g_trace;
        options.dump_outputs_dir = g_dump_outputs_dir;
#endif
        InOuts outputs = chxvm->Run(inputs, options);
        if (this->use_gradient) {
            return std::make_tuple(
                outputs["e"].GetArray(),
                outputs["c"].GetArray(),
                outputs["grad_out@vc"].GetArray().toType(at::kDouble)
            );
        } else {
            return std::make_tuple(
                outputs["e"].GetArray(),
                outputs["c"].GetArray(),
                at::Tensor()
            );
        }
    }


private:
    //chainerx::Context ctx;
    std::unique_ptr<PFVM> chxvm;
    InOuts inputs;
    std::mutex mutex;
    bool use_gradient;
};

class EstimatorCC {
public:
    explicit EstimatorCC(int gpu_id) : 
        model(new ModelCC(gpu_id, true)),
        model_forward(new ModelCC(gpu_id, false))
    {
        this->cutoff = 6.0;

        double cell_corners_raw[8*3];
        for (int i = 0; i < 8; i++) {
            cell_corners_raw[i*3]   = (i / 4);
            cell_corners_raw[i*3+1] = ((i / 2) % 2);
            cell_corners_raw[i*3+2] = (i % 2);
        }
        this->cell_corners = at::from_blob(cell_corners_raw, {8, 3}, at::kDouble).to(GetDefaultDevice());
        this->coordinates_saved = false;
        this->use_book_keeping = false;
        this->book_keeping_skin = 0.0;
    }

    std::vector<py::array> Run(py::array sv, py::array sp, py::array cell, unsigned int n_batch, py::array x_add, py::array calc_mode_type, bool calc_force) {
        auto sv_at = MakeArray(sv);
        auto sp_at = MakeArray(sp);
        auto cell_at = MakeArray(cell);
        auto x_add_at = MakeArray(x_add);
        auto calc_mode_type_at = MakeArray(calc_mode_type);
        sv_at = at::sub(sv_at, at::mm(this->fraction_prev, cell_at));

        at::Tensor batch_at = this->batch_prev;
        at::Tensor a1_at, a2_at, sh_at, vc_at, batch_edge_at;

        if (this->use_book_keeping) {
            auto vc_all = at::sub(at::sub(sv_at.index_select(0, this->a1_prev), sv_at.index_select(0, this->a2_prev)), at::mm(this->sh_prev, cell_at));
            auto rsq_all = at::sum(at::mul(vc_all, vc_all), 1);
            auto within_cutoff = at::le(rsq_all, this->cutoff * this->cutoff);
            a1_at = at::masked_select(this->a1_prev, within_cutoff);
            a2_at = at::masked_select(this->a2_prev, within_cutoff);
            sh_at = at::masked_select(this->sh_prev, within_cutoff.reshape({within_cutoff.size(0), 1})).reshape({a1_at.size(0), this->sh_prev.size(1)});
            batch_edge_at = at::masked_select(this->batch_edge_prev, within_cutoff);
            vc_at = at::masked_select(vc_all, within_cutoff.reshape({within_cutoff.size(0), 1})).reshape({a1_at.size(0), this->sh_prev.size(1)});
        } else {
            a1_at = this->a1_prev;
            a2_at = this->a2_prev;
            sh_at = this->sh_prev;
            batch_edge_at = this->batch_edge_prev;
            vc_at = at::sub(at::sub(sv_at.index_select(0, a1_at), sv_at.index_select(0, a2_at)), at::mm(sh_at, cell_at));
        }

        vc_at = vc_at.toType(at::kFloat);

        at::ScalarType itype = at::kLong;

        std::tuple<at::Tensor, at::Tensor, at::Tensor> outputs;
        if (calc_force) {
            outputs = model->Run(vc_at, sp_at, a1_at, a2_at, batch_at, batch_edge_at, n_batch, x_add_at, calc_mode_type_at);
        } else {
            outputs = model_forward->Run(vc_at, sp_at, a1_at, a2_at, batch_at, batch_edge_at, n_batch, x_add_at, calc_mode_type_at);
        }
        auto energy = std::get<0>(outputs);
        auto charges = std::get<1>(outputs);

        if (!calc_force) {
            std::vector<py::array> output_arrays;
            output_arrays.push_back(MakeOutputArray(energy));
            output_arrays.push_back(MakeOutputArray(charges));
            return output_arrays;
        }

        auto output_grad = std::get<2>(outputs);

        at::Tensor forces_zeros = at::zeros({sp.shape(0), 3}, at::TensorOptions(at::kDouble).device(GetDefaultDevice()));
        at::Tensor forces = at::sub(forces_zeros.scatter_add(0, a2_at.reshape({a2_at.size(0), 1}).expand({a2_at.size(0), 3}), output_grad), forces_zeros.scatter_add(0, a1_at.reshape({a1_at.size(0), 1}).expand({a1_at.size(0), 3}), output_grad));
        int64_t virial_force_index_raw[6] = {0,1,2,1,2,0};
        int64_t virial_vec_index_raw[6] = {0,1,2,2,0,1};
        at::Tensor virial_force_index = at::from_blob(virial_force_index_raw, {6}, at::TensorOptions(itype)).to(GetDefaultDevice());
        at::Tensor virial_vec_index = at::from_blob(virial_vec_index_raw, {6}, at::TensorOptions(itype)).to(GetDefaultDevice());
        at::Tensor virial_edge = at::mul(output_grad.index_select(1, virial_force_index), vc_at.toType(at::kDouble).index_select(1, virial_vec_index));
        at::Tensor virial = at::zeros({n_batch, 6}, at::TensorOptions(at::kDouble).device(GetDefaultDevice()));
        virial = virial.scatter_add(0, batch_edge_at.to(GetDefaultDevice()).reshape({batch_edge_at.size(0), 1}).expand({batch_edge_at.size(0), 6}), virial_edge.to(GetDefaultDevice()));

        std::vector<py::array> output_arrays;
        output_arrays.push_back(MakeOutputArray(energy));
        output_arrays.push_back(MakeOutputArray(charges));
        output_arrays.push_back(MakeOutputArray(forces));
        output_arrays.push_back(MakeOutputArray(virial));
        return output_arrays;
    }


    void set_cutoff(const double _cutoff){
        this->cutoff = _cutoff;
        this->reset_book_keeping();
    }

    void set_book_keeping(const bool book_keeping_flag, const double skin){
        this->use_book_keeping = book_keeping_flag;
        if (book_keeping_flag) {
            this->book_keeping_skin = skin;
        } else {
            this->book_keeping_skin = 0.0;
        }
    }

    std::tuple<bool, ErrorEnumCC, int> book_keeping_check_rebuild(const py::array coordinates, const py::array cell, bool is_pbc){
        auto cell_at = MakeArray(cell);
        if (this->coordinates_saved) {
            auto co_at = MakeArray(coordinates);
            if (co_at.size(0) != this->coordinates_prev.size(0)) return std::make_tuple(true, ErrorEnumCC::NoError, -1);
            co_at = at::sub(co_at, at::mm(this->fraction_prev, cell_at));
            auto atom_diff = at::sub(this->coordinates_prev, co_at);
            double atom_diffsq = at::max(at::sum(at::mul(atom_diff, atom_diff), 1)).item<double>();
            double cell_change_margin = 0.0;
            if (is_pbc) {
                auto corner_pos_diff = at::sub(at::mm(this->cell_corners, cell_at), this->cell_corners_prev);
                cell_change_margin = std::sqrt(at::max(at::sum(at::mul(corner_pos_diff, corner_pos_diff), 1)).item<double>());
            }
            double half_skin = 0.5 * std::max(0.0, this->book_keeping_skin - 2.0 * cell_change_margin);
            if (atom_diffsq < half_skin * half_skin) {
                return std::make_tuple(false, ErrorEnumCC::NoError, this->sh_prev.size(0));
            }
        }
        if (is_pbc) {
            at::Tensor xn = at::cross(cell_at[1], cell_at[2]);
            at::Tensor yn = at::cross(cell_at[2], cell_at[0]);
            at::Tensor zn = at::cross(cell_at[0], cell_at[1]);
            xn = at::div(xn, at::norm(xn));
            yn = at::div(yn, at::norm(yn));
            zn = at::div(zn, at::norm(zn));
            double cell_dx = std::abs(at::sum(at::mul(cell_at[0], xn), 0).item<double>());
            double cell_dy = std::abs(at::sum(at::mul(cell_at[1], yn), 0).item<double>());
            double cell_dz = std::abs(at::sum(at::mul(cell_at[2], zn), 0).item<double>());
            if (std::min(cell_dx, std::min(cell_dy, cell_dz)) < this->cutoff + 1.5 * this->book_keeping_skin) {
                return std::make_tuple(true, ErrorEnumCC::CellTooSmallError, -1);
            }
        }
        return std::make_tuple(true, ErrorEnumCC::NoError, -1);
    }

    void send_params(const py::array fractional, const py::array a1, const py::array a2, const py::array sh, const py::array ba, const py::array be){
        auto fractional_t = MakeArray(fractional);
        this->fraction_prev = fractional_t;
        this->a1_prev = MakeArray(a1);
        this->a2_prev = MakeArray(a2);
        this->sh_prev = MakeArray(sh);
        this->batch_prev = MakeArray(ba);
        this->batch_edge_prev = MakeArray(be);

        this->coordinates_saved = false;
    }

    void send_params_for_book_keeping(const py::array fractional, const py::array a1, const py::array a2, const py::array sh, const py::array ba, const py::array be, const py::array coordinates, const py::array cell, bool is_pbc){
        this->send_params(fractional, a1, a2, sh, ba, be);

        auto coordinates_t = MakeArray(coordinates);
        auto cell_t = MakeArray(cell);
        this->coordinates_prev = at::sub(coordinates_t, at::mm(this->fraction_prev, cell_t));
        this->cell_prev = cell_t;
        this->cell_corners_prev = at::mm(this->cell_corners, cell_t);

        this->coordinates_saved = true;
    }

    void reset_book_keeping(){
        this->coordinates_saved = false;
    }

private:
    std::unique_ptr<ModelCC> model, model_forward;
    double cutoff;
    bool use_book_keeping;
    bool coordinates_saved;
    double book_keeping_skin;
    at::Tensor coordinates_prev, cell_prev, cell_corners_prev, fraction_prev;
    at::Tensor a1_prev, a2_prev, sh_prev, batch_prev, batch_edge_prev;
    at::Tensor cell_corners;
};

class PreprocessorCC {
public:
    static std::tuple<py::array, py::array, py::array, ErrorEnumCC> preprocess_pbc(const py::array coordinates, const py::array cell, const py::array pbc, const double cutoff, const long max_atoms){
        const at::Tensor coordinates_at = MakeArray(coordinates);
        const at::Tensor cell_at = MakeArray(cell);
        const at::Tensor pbc_at = MakeArray(pbc);
        at::Tensor ghost_shift, ghost_refs, total_coordinates;
        at::Tensor atom_index1 = at::zeros({0}, at::TensorOptions(at::kLong).device(GetDefaultDevice()));
        at::Tensor atom_index2 = at::zeros({0}, at::TensorOptions(at::kLong).device(GetDefaultDevice()));
        at::Tensor shift = at::zeros({0, 3}, at::TensorOptions(at::kLong).device(GetDefaultDevice()));
        if (at::all(at::eq(pbc_at, 0)).item<bool>()) {
            at::Tensor left_ind, right_ind;
            std::tie(right_ind, left_ind) = calc_neighbors(coordinates_at, coordinates_at, cutoff);
            at::Tensor is_larger = at::flatten(at::nonzero(at::lt(left_ind, right_ind)));
            atom_index1 = at::index_select(left_ind, 0, is_larger);
            atom_index2 = at::index_select(right_ind, 0, is_larger);
            shift = at::zeros({atom_index1.size(0), 3}, at::TensorOptions(at::kLong).device(GetDefaultDevice()));
            return std::make_tuple(MakeOutputArray(atom_index1), MakeOutputArray(atom_index2), MakeOutputArray(shift), ErrorEnumCC::NoError);
        } else {
            ErrorEnumCC res_error;
            std::tie(ghost_shift, ghost_refs, res_error) = ghost_copy(coordinates_at, cell_at, pbc_at, cutoff, max_atoms);
            if(res_error != ErrorEnumCC::NoError){
                return std::make_tuple(MakeOutputArray(atom_index1), MakeOutputArray(atom_index2), MakeOutputArray(shift), res_error);
            }
            total_coordinates = at::add(at::index_select(coordinates_at, 0, ghost_refs), at::matmul(ghost_shift.toType(coordinates_at.scalar_type()), cell_at));
        }
        if (max_atoms != -1 && (coordinates_at.size(0) * total_coordinates.size(0)) > (max_atoms * max_atoms)) {
            return std::make_tuple(MakeOutputArray(atom_index1), MakeOutputArray(atom_index2), MakeOutputArray(shift), ErrorEnumCC::GhostAtomsTooManyError);
        }

        std::tie(atom_index1, atom_index2, shift) = get_neighbors_single(coordinates_at, total_coordinates, ghost_refs, ghost_shift, cutoff);

        return std::make_tuple(MakeOutputArray(atom_index1), MakeOutputArray(atom_index2), MakeOutputArray(shift), ErrorEnumCC::NoError);
    }

    static std::tuple<py::array, py::array> wrap_coordinates(const py::array coordinates, const py::array cell, const py::array cell_inv, const py::array pbc){
        const at::Tensor coordinates_at = MakeArray(coordinates);
        const at::Tensor pbc_at = MakeArray(pbc);
        if (at::all(at::eq(pbc_at, 0)).item<bool>()) {
            return std::make_tuple(coordinates, MakeOutputArray(at::zeros_like(coordinates_at)));
        }
        const at::Tensor cell_at = MakeArray(cell);
        const at::Tensor cell_inv_at = MakeArray(cell_inv);
        
        at::Tensor fractional = at::floor(at::matmul(coordinates_at, cell_inv_at));
        if ((pbc_at[0] == 0).item<bool>()) {
            fractional.index_put_({at::indexing::Ellipsis, 0}, 0.0);
        }
        if ((pbc_at[1] == 0).item<bool>()) {
            fractional.index_put_({at::indexing::Ellipsis, 1}, 0.0);
        }
        if ((pbc_at[2] == 0).item<bool>()) {
            fractional.index_put_({at::indexing::Ellipsis, 2}, 0.0);
        }
        const at::Tensor wrap_shift = at::neg(at::matmul(fractional, cell_at));
        const at::Tensor ret = at::add(coordinates_at, wrap_shift);
        return std::make_tuple(MakeOutputArray(ret), MakeOutputArray(fractional));
    }

    static std::tuple<at::Tensor, at::Tensor> calc_neighbors(const at::Tensor& coordinates, const at::Tensor& total_coordinates, const double cutoff){
        const at::Tensor distances = at::sum(at::pow(at::sub(at::unsqueeze(coordinates, 0), at::unsqueeze(total_coordinates, 1)), 2), 2);
        const at::Tensor within_cutoff = at::le(distances, cutoff * cutoff);
        const std::vector<at::Tensor> indices = at::where(within_cutoff);
        const at::Tensor right_ind = indices.at(0);
        const at::Tensor left_ind = indices.at(1);
        return std::make_tuple(left_ind, right_ind);        
    }

    static std::tuple<at::Tensor, at::Tensor, at::Tensor> get_neighbors_single(const at::Tensor& coordinates, const at::Tensor& total_coordinates, const at::Tensor& ghost_refs, const at::Tensor& shift_coordinates, const double cutoff){
        at::Tensor left_ind, right_ind;
        std::tie(left_ind, right_ind) = calc_neighbors(coordinates, total_coordinates, cutoff);
        const at::Tensor right_ind_refs = at::index_select(ghost_refs, 0, right_ind);

        const at::Tensor larger_indices = at::flatten(at::nonzero(at::lt(left_ind, right_ind_refs)));
        const at::Tensor left_ind_larger = at::index_select(left_ind, 0, larger_indices);
        const at::Tensor right_ind_larger = at::index_select(right_ind, 0, larger_indices);

        const at::Tensor same_indices = at::flatten(at::nonzero(at::eq(left_ind, right_ind_refs)));
        const at::Tensor left_ind_same = at::index_select(left_ind, 0, same_indices);
        const at::Tensor right_ind_same = at::index_select(right_ind, 0, same_indices);
        const at::Tensor shift_diff_same_atom = at::sub(at::index_select(shift_coordinates, 0, left_ind_same), at::index_select(shift_coordinates, 0, right_ind_same));
        const at::Tensor shift_diff_same_atom_0 = at::select(shift_diff_same_atom, 1, 0);
        const at::Tensor shift_diff_same_atom_1 = at::select(shift_diff_same_atom, 1, 1);
        const at::Tensor shift_diff_same_atom_2 = at::select(shift_diff_same_atom, 1, 2);
        const at::Tensor x_gt = at::gt(shift_diff_same_atom_0, 0);
        const at::Tensor x_eq = at::eq(shift_diff_same_atom_0, 0);
        const at::Tensor y_gt = at::gt(shift_diff_same_atom_1, 0);
        const at::Tensor y_eq = at::eq(shift_diff_same_atom_1, 0);
        const at::Tensor z_gt = at::gt(shift_diff_same_atom_2, 0);
        const at::Tensor half_same_atom = at::flatten(at::nonzero(at::logical_or(x_gt,
            at::logical_and(x_eq,
                at::logical_or(y_gt,
                    at::logical_and(y_eq, z_gt)
                )
            )
        )));

        const at::Tensor atom_index1 = at::cat({left_ind_larger, at::index_select(left_ind_same, 0, half_same_atom)});
        const at::Tensor atom_index2_raw = at::cat({right_ind_larger, at::index_select(right_ind_same, 0, half_same_atom)});
        const at::Tensor shift = at::index_select(shift_coordinates, 0, atom_index2_raw);

        const at::Tensor atom_index2 = at::index_select(ghost_refs, 0, atom_index2_raw);

        return std::make_tuple(atom_index1, atom_index2, shift);
    }

    static std::tuple<std::tuple<at::Tensor, at::Tensor, at::Tensor>, std::tuple<double, double, double>> cell_width(const at::Tensor& cell){
        at::Tensor xn = at::cross(cell[1], cell[2]);
        at::Tensor yn = at::cross(cell[2], cell[0]);
        at::Tensor zn = at::cross(cell[0], cell[1]);
        xn = at::div(xn, at::norm(xn));
        yn = at::div(yn, at::norm(yn));
        zn = at::div(zn, at::norm(zn));
        const double cell_dx = std::abs(at::sum(at::mul(cell[0], xn), 0).item<double>());
        const double cell_dy = std::abs(at::sum(at::mul(cell[1], yn), 0).item<double>());
        const double cell_dz = std::abs(at::sum(at::mul(cell[2], zn), 0).item<double>());
        return std::make_tuple(std::make_tuple(xn, yn, zn), std::make_tuple(cell_dx, cell_dy, cell_dz));
    }

    static std::tuple<at::Tensor, at::Tensor> _extend_ghost_along_axis(const at::Tensor& current_refs, const at::Tensor& current_shift, const int rep_x, const at::Tensor& xdl, const double cell_dx, const at::Tensor& shift_x, const double cutoff){
        const int n_atoms = current_refs.size(0);
        const at::Tensor repeated_current_refs = current_refs.repeat({2 * rep_x + 1});
        const at::Tensor repeated_current_shift = current_shift.repeat({2 * rep_x + 1, 1});
        const at::Tensor xdl_refs = at::index_select(xdl, 0, repeated_current_refs);
        const at::Tensor rep_x_range = at::cat({at::arange(rep_x + 1, at::TensorOptions(at::kLong).device(GetDefaultDevice())), at::neg(at::arange(1, rep_x + 1, at::TensorOptions(at::kLong).device(GetDefaultDevice())))});
        const at::Tensor cell_step = at::repeat_interleave(rep_x_range, n_atoms);
        const at::Tensor dists = at::max(at::add(at::mul(at::sub(cell_step, 1), cell_dx), xdl_refs), at::sub(at::mul(at::neg(cell_step), cell_dx), xdl_refs));
        const at::Tensor within_cutoff = at::flatten(at::nonzero(at::lt(dists, cutoff)));
        const at::Tensor ghost_refs = at::index_select(repeated_current_refs, 0, within_cutoff);
        const at::Tensor ghost_shift = at::index_select(at::add(repeated_current_shift, at::mul(at::unsqueeze(cell_step, 1), shift_x)), 0, within_cutoff);
        return std::make_tuple(ghost_refs, ghost_shift);
    }

    static std::tuple<at::Tensor, at::Tensor, ErrorEnumCC> ghost_copy(const at::Tensor& coordinates, const at::Tensor& cell, const at::Tensor& pbc, const double cutoff, const long max_atoms){
        at::Tensor ghost_shift, ghost_refs;
        at::Tensor xn, yn, zn;
        double cell_dx, cell_dy, cell_dz;
        auto cell_width_result = cell_width(cell);
        std::tie(xn, yn, zn) = std::get<0>(cell_width_result);
        std::tie(cell_dx, cell_dy, cell_dz) = std::get<1>(cell_width_result);
        // std::tie(std::tie(xn, yn, zn), std::tie(cell_dx, cell_dy, cell_dz)) = cell_width(cell);

        if(cell_dx == 0.0 || cell_dy == 0.0 || cell_dz == 0.0){
            return std::make_tuple(ghost_shift, ghost_refs, ErrorEnumCC::CellTooSmallError);
        }

        const int rep_x = static_cast<int>(std::floor(cutoff / cell_dx)) + 1;
        const int rep_y = static_cast<int>(std::floor(cutoff / cell_dy)) + 1;
        const int rep_z = static_cast<int>(std::floor(cutoff / cell_dz)) + 1;

        const unsigned long n_atoms_mergin = 10;
        const unsigned long n_max_atoms_matrix = (max_atoms + n_atoms_mergin) * (max_atoms + n_atoms_mergin);
        const unsigned long n_real_atoms_with_mergin = coordinates.size(0) + n_atoms_mergin;
        const unsigned long n_ghost_atoms_optimistic = (2L * rep_x - 1) * (2L * rep_y - 1) * (2L * rep_z - 1) * n_real_atoms_with_mergin;
        if(max_atoms != -1 && n_real_atoms_with_mergin * n_ghost_atoms_optimistic > n_max_atoms_matrix){
            return std::make_tuple(ghost_shift, ghost_refs, ErrorEnumCC::CellTooSmallError);
        }

        const at::Tensor xdl = at::abs(at::matmul(coordinates, xn));
        const at::Tensor ydl = at::abs(at::matmul(coordinates, yn));
        const at::Tensor zdl = at::abs(at::matmul(coordinates, zn));

        const int n_atoms = coordinates.size(0);
        ghost_shift = at::zeros({n_atoms, 3}, at::TensorOptions(at::kLong).device(GetDefaultDevice()));
        ghost_refs = at::arange(n_atoms, at::TensorOptions(at::kLong).device(GetDefaultDevice()));

        int64_t shift_x_raw[3] = {1, 0, 0};
        int64_t shift_y_raw[3] = {0, 1, 0};
        int64_t shift_z_raw[3] = {0, 0, 1};
        const at::Tensor shift_x = at::from_blob(shift_x_raw, {1, 3}, at::TensorOptions(at::kLong)).to(GetDefaultDevice());
        const at::Tensor shift_y = at::from_blob(shift_y_raw, {1, 3}, at::TensorOptions(at::kLong)).to(GetDefaultDevice());
        const at::Tensor shift_z = at::from_blob(shift_z_raw, {1, 3}, at::TensorOptions(at::kLong)).to(GetDefaultDevice());

        if(pbc[0].item<int64_t>()){
            std::tie(ghost_refs, ghost_shift) = _extend_ghost_along_axis(ghost_refs, ghost_shift, rep_x, xdl, cell_dx, shift_x, cutoff);
        }
        if(pbc[1].item<int64_t>()){
            std::tie(ghost_refs, ghost_shift) = _extend_ghost_along_axis(ghost_refs, ghost_shift, rep_y, ydl, cell_dy, shift_y, cutoff);
        }
        if(pbc[2].item<int64_t>()){
            std::tie(ghost_refs, ghost_shift) = _extend_ghost_along_axis(ghost_refs, ghost_shift, rep_z, zdl, cell_dz, shift_z, cutoff);
        }

        return std::make_tuple(ghost_shift, ghost_refs, ErrorEnumCC::NoError);
    }
};

std::vector<py::array> Run(
    const std::shared_ptr<EstimatorCC>& model,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& sv,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& sp,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& cell,
    const unsigned int n_batch,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& x_add,
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& calc_mode_type,
    bool calc_force
){
    return model->Run(sv, sp, cell, n_batch, x_add, calc_mode_type, calc_force);
}

void set_cutoff(const std::shared_ptr<EstimatorCC>& model, const double cutoff){
    model->set_cutoff(cutoff);
}

void set_book_keeping(const std::shared_ptr<EstimatorCC>& model, const bool book_keeping_flag, const double skin){
    model->set_book_keeping(book_keeping_flag, skin);
}

std::tuple<bool, ErrorEnumCC, int> book_keeping_check_rebuild(
    const std::shared_ptr<EstimatorCC>& model,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& coordinates,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& cell,
    bool is_pbc
){
    return model->book_keeping_check_rebuild(coordinates, cell, is_pbc);
}

void send_params(
    const std::shared_ptr<EstimatorCC>& model,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& fractional,
    const py::array_t<long, py::array::c_style | py::array::forcecast>& a1,
    const py::array_t<long, py::array::c_style | py::array::forcecast>& a2,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& sh,
    const py::array_t<long, py::array::c_style | py::array::forcecast>& ba,
    const py::array_t<long, py::array::c_style | py::array::forcecast>& be
){
    model->send_params(fractional, a1, a2, sh, ba, be);
}

void send_params_for_book_keeping(
    const std::shared_ptr<EstimatorCC>& model,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& fractional,
    const py::array_t<long, py::array::c_style | py::array::forcecast>& a1,
    const py::array_t<long, py::array::c_style | py::array::forcecast>& a2,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& sh,
    const py::array_t<long, py::array::c_style | py::array::forcecast>& ba,
    const py::array_t<long, py::array::c_style | py::array::forcecast>& be,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& coordinates,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& cell,
    bool is_pbc
){
    model->send_params_for_book_keeping(fractional, a1, a2, sh, ba, be, coordinates, cell, is_pbc);
}

void reset_book_keeping(const std::shared_ptr<EstimatorCC>& model){
    model->reset_book_keeping();
}

std::tuple<py::array, py::array, py::array, ErrorEnumCC> preprocess_pbc(
    const std::shared_ptr<EstimatorCC>& model,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& coordinates,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& cell,
    const py::array_t<long, py::array::c_style | py::array::forcecast>& pbc,
    const double cutoff,
    const int max_atoms
){
    return PreprocessorCC::preprocess_pbc(coordinates, cell, pbc, cutoff, max_atoms);
}

std::tuple<py::array, py::array> wrap_coordinates(
    const std::shared_ptr<EstimatorCC>& model,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& coordinates,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& cell,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& cell_inv,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& pbc
){
    return PreprocessorCC::wrap_coordinates(coordinates, cell, cell_inv, pbc);
}

std::shared_ptr<EstimatorCC> Load(int gpu_id) {
    return std::make_shared<EstimatorCC>(gpu_id);
}

void Crash(const std::shared_ptr<EstimatorCC>& model) {
    delete model.get();
}

void SetTrace(int t) {
    g_trace = t;
}

void SetDumpOutputsDir(const std::string& d) {
    g_dump_outputs_dir = d;
}

void InitModel(py::module& m) {
    py::class_<EstimatorCC, std::shared_ptr<EstimatorCC>> c{m, "Model"};
    c.def("l", &Load, "PLEASE DO NOT REVERSE THIS :(");
    c.def("z", &fread, "PLEASE DO NOT REVERSE THIS :(");
    // This one is the real entry point.
    c.def("r", &Run, "PLEASE DO NOT REVERSE THIS :(");
    c.def("sc", &set_cutoff, "PLEASE DO NOT REVERSE THIS :(");
    c.def("bks", &set_book_keeping, "PLEASE DO NOT REVERSE THIS :(");
    c.def("bkc", &book_keeping_check_rebuild, "PLEASE DO NOT REVERSE THIS :(");
    c.def("sp", &send_params, "PLEASE DO NOT REVERSE THIS :(");
    c.def("spb", &send_params_for_book_keeping, "PLEASE DO NOT REVERSE THIS :(");
    c.def("bkr", &reset_book_keeping, "PLEASE DO NOT REVERSE THIS :(");
    c.def("ppp", &preprocess_pbc, "PLEASE DO NOT REVERSE THIS :(");
    c.def("ppw", &wrap_coordinates, "PLEASE DO NOT REVERSE THIS :(");
    //c.def("w", &chainerx::SetDefaultContext, "PLEASE DO NOT REVERSE THIS :(");

    py::enum_<ErrorEnumCC>(c, "ErrorEnumCC")
    .value("NoError", ErrorEnumCC::NoError)
    .value("AtomsFarAwayError", ErrorEnumCC::AtomsFarAwayError)
    .value("CellTooSmallError", ErrorEnumCC::CellTooSmallError)
    .value("GhostAtomsTooManyError", ErrorEnumCC::GhostAtomsTooManyError)
    .export_values();
}

__attribute__((constructor(101)))
void init() {
#ifndef DEV
    mprotect(rodata_start, (rodata_end - rodata_start + 4095) & ~4095,
             PROT_READ | PROT_WRITE);
    DecodeRange(RODATA_SEED, rodata_start, rodata_end);
#if 0
    fprintf(stderr, "decoded: %p-%p (%zx)\n",
            rodata_start, rodata_end, rodata_end - rodata_start);
#endif
#endif
}

PYBIND11_MODULE(libinfer, m) {  // NOLINT
    m.doc() = "infer";

    InitModel(m);

    m.def("v", &InitModel, "MINAIDE ONEGAI ('-')");
    m.def("b", &Run, "MINAIDE ONEGAI ('-')");
    m.def("u", &PFVM::num_variables, "MINAIDE ONEGAI ('-')");
    m.def("k", &mmap, "MINAIDE ONEGAI ('-')");
    // This one is the real entry point.
    m.def("l", &Load, "MINAIDE ONEGAI ('-')");
    m.def("o", [](){}, "MINAIDE ONEGAI ('-')");
    //m.def("p", &chainerx::SetDefaultDevice, "MINAIDE ONEGAI ('-')");
    m.def("z", [](){ return 42; }, "MINAIDE ONEGAI ('-')");
    m.def("w", &init, "MINAIDE ONEGAI ('-')");
    m.def("a", &PFVMOpaque::DebugString, "MINAIDE ONEGAI ('-')");
    m.def("c", &Crash, "MINAIDE ONEGAI ('-')");
    m.def("t", &SetTrace, "MINAIDE ONEGAI ('-')");
    m.def("d", &SetDumpOutputsDir, "MINAIDE ONEGAI ('-')");

}

