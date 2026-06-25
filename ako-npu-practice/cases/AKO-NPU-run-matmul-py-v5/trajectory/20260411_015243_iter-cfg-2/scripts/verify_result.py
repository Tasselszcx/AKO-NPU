#!/usr/bin/python3
# coding=utf-8

import sys
import numpy as np

# For float32 output from fp16 matmul, use reasonable tolerances
# fp16 has ~3.3 decimal digits of precision, matmul accumulates K=256 terms
relative_tol = 1e-3
absolute_tol = 1e-3
error_tol = 1e-4


def verify_result(output, golden):
    output = np.fromfile(output, dtype=np.float32).reshape(-1)
    golden = np.fromfile(golden, dtype=np.float32).reshape(-1)
    
    # Check both atol and rtol as required by OPTIMIZE.md
    different_element_results = np.isclose(output,
                                           golden,
                                           rtol=relative_tol,
                                           atol=absolute_tol,
                                           equal_nan=True)
    different_element_indexes = np.where(different_element_results == False)[0]
    for index in range(min(len(different_element_indexes), 20)):
        real_index = different_element_indexes[index]
        golden_data = golden[real_index]
        output_data = output[real_index]
        denom = abs(golden_data) if abs(golden_data) > 1e-10 else 1.0
        print(
            "data index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f" %
            (real_index, golden_data, output_data,
             abs(output_data - golden_data) / denom))
    error_ratio = float(different_element_indexes.size) / golden.size
    print("error ratio: %.6f, tolerance: %.4f (rtol=%.1e, atol=%.1e)" % (error_ratio, error_tol, relative_tol, absolute_tol))
    return error_ratio <= error_tol


if __name__ == '__main__':
    try:
        res = verify_result(sys.argv[1], sys.argv[2])
        if not res:
            raise ValueError("[ERROR] result error")
        else:
            print("test pass!")
    except Exception as e:
        print(e)
        sys.exit(1)
