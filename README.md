# Vehicle Retrieval by Natural Language Descriptions

The Vehicle Retrieval by Natural Language Specification in the 2021 AI City
Challenge Track 5.

## Problem Definition

For the purpose of this task, we utilize the proposed CityFlow-NL Benchmark in
a \textit{single-view} setup. For each single-view track, we bundle it with
three different NL descriptions.

We divide the CityFlow-NL into training and testing splits. We build the
testing split with groups of NL descriptions as query sets and the goal is to
rank all single-view tracks based on each query set.

## Evaluation Metrics
The Vehicle Retrieval by NL Descriptions task is evaluated using standard
metrics for retrieval tasks~\cite{manning2008introduction}.  We use the Mean
Reciprocal Rank (MRR) as the main evaluation metric. Recall @ 1, Recall @ 5,
and Recall @ 10 are also evaluated for all approaches.

We evaluate each approach with top-10 tracks for each query set, \ie tracks
that are ranked after the 10-th are ignored during evaluation.
