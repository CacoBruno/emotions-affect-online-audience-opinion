import numpy as np
from typing import List

import math

 
def opinionEuclideanDistance(influValenceScore, influStrengthPredict, commentValenceScoreMean, commentStrengthPredictMean):
    """
    Calculates the Euclidean distance between two points in a two-dimensional space.

    Parameters:
    influValenceScore (float): X-axis value of the first point.
    influStrengthPredict (float): Y-axis value of the first point.
    commentValenceScoreMean (float): X-axis value of the second point.
    commentStrengthPredictMean (float): Y-axis value of the second point.

    Returns:
    float: Euclidean distance between the two points.
    """
    distance = math.sqrt((influValenceScore - commentValenceScoreMean) ** 2 + 
                         (influStrengthPredict - commentStrengthPredictMean) ** 2)
    return distance


