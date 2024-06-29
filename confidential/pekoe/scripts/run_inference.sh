#!/bin/bash

set -eux

pfvm_root=~/dev/pfvm
iterations=10

if [ ! -f teanet_onnx/model.onnx ] ; then
    rm -rf teanet_onnx
    python3 examples/pekoe/ase_molecule.py --output-onnx teanet_onnx --large-md 5
fi

for test_data_index in 0 1 ; do
    for device in '-d cuda' '-d cpu' '--use_openvino' ; do
        for backprop in '' '--backprop'; do
            dev_name=${device//[- ]/}

            out_prefix="${dev_name}_${test_data_index}"
            if [ "${backprop}" != '' ] ; then
                out_prefix="backprop_${out_prefix}"
            fi
            out_prefix="teanet_onnx/$out_prefix"

            ${pfvm_root}/build/tools/run_onnx teanet_onnx ${device} ${backprop} \
                        -I $iterations \
                        --skip_inference \
                        --chrome_tracing "${out_prefix}_trace.json" \
                        --test_case test_data_set_${test_data_index} \
                        --report_json "${out_prefix}_report.json" \
                        --compiler_log \
                &> "${out_prefix}.log" || echo "Unexpected exit code: $?"
        done
    done
done

zip -r teanet_onnx.zip teanet_onnx
