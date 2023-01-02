# Natural Language-Based Vehicle Retrieval

This dataset is curated for the Natural Language (NL) Based Vehicle Retrieval
Challenge Track of the 2023 AI City Workshop.

Workshop summary papers with this challenge track are available at:
<https://arxiv.org/abs/2204.10380> and <https://arxiv.org/abs/2104.12233>.

```bibtex
@InProceedings{Naphade_2021_CVPR,
    author    = {Naphade, Milind and Wang, Shuo and Anastasiu, David C. and Tang, Zheng and Chang, Ming-Ching and Yang, Xiaodong and Yao, Yue and Zheng, Liang and Chakraborty, Pranamesh and Lopez, Christian E. and Sharma, Anuj and Feng, Qi and Ablavsky, Vitaly and Sclaroff, Stan},
    title     = {The 5th AI City Challenge},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2021},
    pages     = {4263-4273}
}
@InProceedings{Naphade_2022_CVPR,
    author    = {Naphade, Milind and Wang, Shuo and Anastasiu, David C. and Tang, Zheng and Chang, Ming-Ching and Yao, Yue and Zheng, Liang and Rahman, Mohammed Shaiqur and Venkatachalapathy, Archana and Sharma, Anuj and Feng, Qi and Ablavsky, Vitaly and Sclaroff, Stan and Chakraborty, Pranamesh and Li, Alice and Li, Shangru and Chellappa, Rama},
    title     = {The 6th AI City Challenge},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {3347-3356}
}
```

## Contents in this repository

`data/extract_vdo_frms.py` is a Python script that is used to extract frames
from the provided videos. Please use this script to extract frames, so that the
path configurations in JSON files are consistent.

`data/train-tracks.json` is a dictionary of all 2,155 vehicle tracks in the
training split. Each vehicle track is annotated with three natural language (NL)
descriptions of the target and is assigned a universally unique identifier
(UUID). The file is structured as

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

The files under the *frames* attribute are paths in the CityFlow Benchmark [2].
The *nl* attribute contains the three natural language descriptions annotated for
this vehicle track. The *nl_other_views* attribute is a list of all other natural
language descriptions we collected for the same vehicle target, but for another
view point or another time.

`data/test-tracks.json` contains 184 tracks of candidate target vehicles. The
structure of this file is identical to the training split, except that the
natural language descriptions are removed.

`data/test-queries.json` contains 184 queries. Each consists of three natural
language descriptions of the vehicle target annotated by different annotators
under the nl attributes. Same as the training split, the nl_other_views is a
list of all other natural language descriptions we collected for the same
vehicle target, but for another view point or another time. Teams may choose to
use these additional descriptions during inference if needed. Each query is
assigned a UUID that is later used in results submission. The structure of this
file is as follows:

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

## Usage of Pre-trained Models and Additional Data

> **Warning**
> Teams **cannot** use models that are pre-trained on the CityFlow Benchmark,
> *e.g.* ResNet used for the Re-ID or MTMC tracks in previous AI City
> Challenges. Teams may use additional publicly available training datasets that
> were not collected specifically for language-based, traffic-related vision
> tasks. If in doubt, please contact the organizers.

## Submission Format

For each query, teams should submit a list of the testing tracks ranked by their
retrieval model.  One JSON file should be submitted containing a dictionary in
the following format:

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
main evaluation metric. Recall @ 5 and Recall @ 10 are also evaluated for all
submissions.

## Citations

Please cite this work:

[1] Feng, Qi, et al. "CityFlow-NL: Tracking and Retrieval of Vehicles at City
Scale by Natural Language Descriptions." arXiv preprint. arXiv:2101.04741.

## References

[2] Tang, Zheng, et al. "CityFlow: A city-scale benchmark for multi-target
multi-camera vehicle tracking and re-identification." CVPR. 2019.

[3] Voorhees, Ellen M. "The TREC-8 question answering track report." Trec. Vol.
99. 1999.
