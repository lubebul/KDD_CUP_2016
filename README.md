# KDD CUP 2016
This repository holds our solution(Team @@ - 26th place) for [KDD CUP 2016](https://kddcup2016.azurewebsites.net/).

## Task
Given the accepted paper lists of the top CS conferences (ex: KDD, ICML, SIGMOD, etc), the goal is to predict the paper submitted by which institution are most likely to be accepted in 2016.

## Our Solution: OMNI-Prop (AAAI'15)
This is a classical node ranking task. A naive approach is to propagate the academic influence score that associated with each node(researcher) on the co-author graph.
However, we found traditional label propagation algorithms are not suitable for this task. As they require the homophily correlation of nodes (i.e. the influence score of a node and its neighbors' must be similar). While from our experience, papers are usually co-authored by Prof. and the PhD student, which obviously are not having similar academic influence yet.
Therefore, we choose to implement [OMNI-Prop(AAAI'15)](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9382), a propagation algorithm that are specially designed to cope with both homophily and heterophily correlations.
Following the competition evaluation rule, we designed to propagate two scores on the co-author graph:
 * the # of accepted papers of the researcher
 * the # of co-authors

Because the academic influence of a paper is inverse proportional to the # of co-authors.