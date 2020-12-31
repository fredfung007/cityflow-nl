#!/usr/bin/env python
# COPYRIGHT 2020. Fred Fung. Boston University.
"""
Compute evaluation metrics for the baseline model.
"""
import json


def evaluate_retrieval_results(gt_tracks, results):
    recall_5 = 0
    recall_10 = 0
    mrr = 0
    for query in gt_tracks:
        result = results[query]
        target = gt_tracks[query]
        try:
            rank = result.index(target)
        except ValueError:
            rank = 100
        if rank < 10:
            recall_10 += 1
        if rank < 5:
            recall_5 += 1
        mrr += 1.0 / (rank + 1)
    recall_5 /= len(gt_tracks)
    recall_10 /= len(gt_tracks)
    mrr /= len(gt_tracks)
    print("Recall@5 is %.4f" % recall_5)
    print("Recall@10 is %.4f" % recall_10)
    print("MRR is %.4f" % mrr)


if __name__ == '__main__':
    with open("data/test-gt.json") as f:
        gt_tracks = json.load(f)
    with open("baseline/results.json") as f:
        results = json.load(f)
    evaluate_retrieval_results(gt_tracks, results)
