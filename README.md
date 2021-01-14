# Natural Language-Based Vehicle Retrieval

This dataset is curated for the Natural Language (NL) Based Vehicle Retrieval
Challenge Track of the 2021 AI City Workshop.

## Contents in this repository
`data/train-tracks.json` is a dictionary of all 2,498 vehicle tracks in the
training split. Each vehicle track is annotated with three natural language
descriptions of the target and is assigned a universally unique identifier
(UUID).  The file is structured as
```json
{
  "track-uuid-1": {
    "frames": ["file-1.jpg", ..., "file-n.jpg"],
    "boxes": [[742, 313, 171, 129], ..., [709, 304, 168, 125]],
    "nl": [
      "A gray pick-up truck goes ...", 
      "A dark pick-up runs ...", 
      "Pick-up truck that goes ..."
    ]
  },
  "track-uuid-2": ...
}
```
The files under the `frames` attribute are paths in the CityFlow Benchmark [2] used
in Challenge Track 2 of the 2021 AI City Challenge.

`data/test-tracks.json` contains 530 tracks of target vehicles. The structure
of this file is identical to the training split, except that the natural
language descriptions are removed.

`data/test-queiries.json` contains 530 queries. Each consists of three natural
language descriptions of the vehicle target annotated by different annotators.
Each query is assigned a UUID that is later used in results submission.  The
structure of this file is as follows:
```json
{
  "query-uuid-1": [
    "A dark red SUV drives straight through an intersection.",
    "A red MPV crosses an empty intersection.",
    "A red SUV going straight down the street."
  ],
  "query-uuid-2": ...
}
```

The `baseline/` directory contains a baseline model that measures the similarity
between language descriptions and frame crops in a track. Details of this model
can be found in [1].

## Problem Definition

Teams should retrieve and rank the provided vehicle tracks for each of the
queries. A baseline retrieval model is provided as a demo for a start point for
participating teams.

## Submission Format
For each query, teams should submit a list of the testing tracks ranked by
their retrieval model.  One JSON file should be submitted containing a
dictionary in the following format:
```json
{
  "query-uuid-1": ["track-uuid-i", ..., "track-uuid-j"],
  "query-uuid-2": ["track-uuid-m", ..., "track-uuid-n"],
  ...
}
```

A sample JSON file of submission for the baseline model is available in
`baseline/baseline-results.json`.

## Evaluation Metrics
The Vehicle Retrieval by NL Descriptions task is evaluated using standard
metrics for retrieval tasks.  We use the Mean Reciprocal Rank (MRR) [3] as the
main evaluation metric. Recall @ 5, Recall @ 10, and Recall @ 25 are also
evaluated for all submissions.

The provided baseline model’s MRR is 0.0269, Recall @ 5 is 0.0264, Recall @ 10 is
0.0491, Recall @ 25 is 0.1113.


## Citations
Please cite this work:

[1] Feng, Qi, et al. "CityFlow-NL: Tracking and Retrieval of Vehicles at City
Scale by Natural Language Descriptions." arXiv preprint. arXiv:2101.04741. 

## References
[2] Tang, Zheng, et al. "Cityflow: A city-scale benchmark for multi-target
multi-camera vehicle tracking and re-identification." CVPR. 2019.

[3] Voorhees, Ellen M. "The TREC-8 question answering track report." Trec. 
Vol. 99. 1999.
