#!/usr/bin/env python
# COPYRIGHT 2020. Fred Fung. Boston University.
"""
Prepare JSON file for submission/evaluation.
"""
import json
import os
import sys

from absl import flags

flags.DEFINE_string("path_to_output",
                    "/research/fung/experiments/RETRIEVAL-EVALUATION_12_22.21:45:26.645397/logs",
                    "Path to output logs folder.")

if __name__ == '__main__':
    flags.FLAGS(sys.argv)
    results = dict()
    list_of_outputs = os.listdir(flags.FLAGS.path_to_output)
    for query in list_of_outputs:
        with open(os.path.join(flags.FLAGS.path_to_output, query)) as f:
            lines = f.readlines()
        results[query.split(".")[0]] = list()
        for l in lines:
            results[query.split(".")[0]].append(l.strip())

    with open("results.json", "w") as f:
        json.dump(results, f)
