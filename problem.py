import json
import os
from dataclasses import dataclass
from itertools import tee
from os.path import join as pjoin

import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.score_types import BaseScoreType
from rampwf.prediction_types.base import BasePrediction
from rampwf.utils.importing import import_module_from_source
from rampwf.workflows import Estimator
from sklearn.model_selection import StratifiedShuffleSplit

DATA_HOME = "data"
RANDOM_STATE = 777

# --------------------------------------
# 0) Utils to manipulate data
# --------------------------------------


@dataclass
class WalkSignal:
    """Wrapper class around a numpy array containing a walk signal (with metadata)"""
    code: str
    age: int
    gender: str
    height: float
    weight: int
    bmi: float
    laterality: str
    sensor: str
    pathology_group: str
    is_control: str
    foot: str  # left or right
    signal: "typing.Any"  # numpy array or pandas dataframe

    @classmethod
    def load_from_file(cls, code, data_home=DATA_HOME):
        fname = pjoin(data_home, code)
        with open(fname + ".json", "r") as file_handle:
            metadata = json.load(file_handle)
        signal = pd.read_csv(fname + ".csv", sep=",")  # left and right feet

        left_foot_cols = ["LAV", "LAX", "LAY",
                          "LAZ", "LRV", "LRX", "LRY", "LRZ"]
        right_foot_cols = ["RAV", "RAX", "RAY",
                           "RAZ", "RRV", "RRX", "RRY", "RRZ"]

        left_foot = cls(code=code,
                        age=metadata["Age"],
                        gender=metadata["Gender"],
                        height=metadata["Height"],
                        weight=metadata["Weight"],
                        bmi=metadata["BMI"],
                        laterality=metadata["Laterality"],
                        sensor=metadata["Sensor"],
                        pathology_group=metadata["PathologyGroup"],
                        is_control=metadata["IsControl"],
                        foot="left",
                        signal=signal[left_foot_cols].rename(columns=lambda name: name[1:]))
        right_foot = cls(code=code,
                         age=metadata["Age"],
                         gender=metadata["Gender"],
                         height=metadata["Height"],
                         weight=metadata["Weight"],
                         bmi=metadata["BMI"],
                         laterality=metadata["Laterality"],
                         sensor=metadata["Sensor"],
                         pathology_group=metadata["PathologyGroup"],
                         is_control=metadata["IsControl"],
                         foot="right",
                         signal=signal[right_foot_cols].rename(columns=lambda name: name[1:]))

        return left_foot, right_foot


def load_steps(code, data_home=DATA_HOME):
    """Return two lists of steps (left and right feet).

    Arguments:
        code {str} -- code of the trial, e.g. "2-10"

    Keyword Arguments:
        data_home {str} -- folder where the files lie (default: {DATA_HOME})

    Returns:
        [tuple(list)] -- two lists of steps (left foot, right foot)
    """
    fname = pjoin(data_home, code)
    with open(fname + ".json", "r") as file_handle:
        metadata = json.load(file_handle)
    return metadata["LeftFootActivity"], metadata["RightFootActivity"]


def _read_data(path, train_or_test="train"):
    """Return the list of signals and steps for the train or test data set

    Arguments:
        path {str} -- folder where the train and test data are

    Keyword Arguments:
        train_or_test {str} -- train or test (default: {"train"})

    Returns:
        [tupe(List[WalkSignal], List)] -- (list of signals, list of lists of steps)
    """
    folder = pjoin(path, DATA_HOME, train_or_test)
    code_list = [fname.split(".")[0] for fname in os.listdir(
        folder) if fname.endswith(".csv")]

    X = list()
    y = list()
    for code in code_list:
        left_signal, right_signal = WalkSignal.load_from_file(code, folder)
        left_steps, right_steps = load_steps(code, folder)
        X.extend((left_signal, right_signal))
        y.extend((left_steps, right_steps))

    return X, np.array(y, dtype=list)


def is_iterable(obj):
    """To check if an object is iterable."""
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True

# --------------------------------------
# 2) Object implementing the score type
# --------------------------------------


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def _check_step_list(step_list):
    """Some sanity checks."""

    for step in step_list:
        assert len(
            step) == 2, f"A step consists of a start and an end: {step}."
        start, end = step
        assert start < end, f"start should be before end: {step}."


def count_detected(step_list_1, step_list_2):
    """A detected step is counted as correct if the mean of its start and end times lies
    inside an annotated step. An annotated step can only be detected one time.If several
    detected steps correspond to the same annotated step, all but one are considered as false.
    The precision is the number of correctly detected steps divided by the total number of
    detected steps.

    The lists y_true_ and y_pred are lists of steps, for instance:
        - step_list_1: [[357, 431], [502, 569], [633, 715], [778, 849], [907, 989]]
        - step_list_2: [[293, 365], [422, 508], [565, 642], [701, 789]]


    Arguments:
        step_list_2 {List} -- steps list 1
        step_list_2 {List} -- step list 2

    Returns:
        int -- number of detected steps in the first list
    """
    _check_step_list(step_list_1)
    _check_step_list(step_list_2)

    detected_index_set = set()
    n_detected = 0
    for (start, end) in step_list_2:
        mid = (start+end)//2
        for (index, (start_true, end_true)) in enumerate(step_list_1):
            if (index not in detected_index_set) and (start_true <= mid < end_true):
                n_detected += 1
                detected_index_set.add(index)
                break

    return n_detected


def _step_detection_precision(y_true, y_pred):
    """A detected step is counted as correct if the mean of its start and end times lies
        inside an annotated step.

    Arguments:
        y_true {List} -- list of true list of steps
        y_pred {List} -- predicted steps

    Returns:
        float -- precision, between 0.0 and 1.0

    """
    n_detected = 0
    n_steps_pred = 0
    for (step_true, step_pred) in zip(y_true, y_pred):
        n_detected += count_detected(step_true, step_pred)
        n_steps_pred += len(step_pred)

    if n_steps_pred == 0:  # no step was predicted
        return 0.0
    return n_detected/n_steps_pred


def _step_detection_recall(y_true, y_pred):
    """A detected step is counted as correct if the mean of its start and end times lies
        inside an annotated step.

    Arguments:
        y_true {List} -- list of true list of steps
        y_pred {List} -- predicted steps

    Returns:
        float -- recall, between 0.0 and 1.0

    """
    n_detected = 0
    n_steps_true = 0
    for (step_true, step_pred) in zip(y_true, y_pred):
        n_detected += count_detected(step_true, step_pred)
        n_steps_true += len(step_true)
    return n_detected/n_steps_true


class FScoreStepDetection(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name="F-score (step detection)", precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred) -> float:
        """Geometric between precision and recall.

        The lists y_true_ and y_pred are lists of lists of steps, for instance:
            - y_true: [[[907, 989]] [[357, 431], [502, 569]], [[633, 715], [778, 849]]]
            - y_pred: [[[293, 365]], [[422, 508], [565, 642]], [[701, 789]]]

        Arguments:
            y_true {List} -- true steps
            y_pred {List} -- predicted steps

        Returns:
            float -- f-score, between 0.0 and 1.0
        """
        prec = _step_detection_precision(y_true, y_pred)
        rec = _step_detection_recall(y_true, y_pred)

        if prec+rec < 1e-6:
            return 0.0
        return 2*prec*rec/(prec+rec)

# --------------------------------------
# 3) Prediction types
# --------------------------------------


class Predictions(BasePrediction):

    def __init__(self, y_pred=None, y_true=None, n_samples=None):
        """Essentially the same as in a regression task, but the prediction is a list not a float."""
        if y_pred is not None:
            self.y_pred = np.array(y_pred, dtype=list)
        elif y_true is not None:
            self.y_pred = np.array(y_true, dtype=list)
        elif n_samples is not None:
            # self.n_columns == 0:
            shape = (n_samples)
            self.y_pred = np.empty(shape, dtype=list)
            self.y_pred.fill(np.nan)
        else:
            raise ValueError(
                'Missing init argument: y_pred, y_true, or n_samples')
        self.check_y_pred_dimensions()

    @property
    def valid_indexes(self):
        """Return valid indices (e.g., a cross-validation slice)."""
        if len(self.y_pred.shape) == 1:
            return ~pd.isnull(self.y_pred)
        elif len(self.y_pred.shape) == 2:
            return ~pd.isnull(self.y_pred[:, 0])
        else:
            raise ValueError('y_pred.shape > 2 is not implemented')

    def check_y_pred_dimensions(self):
        pass

    @classmethod
    def combine(cls, predictions_list, index_list=None):
        """Dummy function. Here, combining consists in taking the first prediction."""
        combined_predictions = cls(y_pred=predictions_list[0].y_pred)
        return combined_predictions


def make_step_detection():
    return Predictions

# --------------------------------------
# 4) Ramp problem definition
# --------------------------------------


problem_title = "Step Detection with Inertial Measurement Units"
Predictions = make_step_detection()
workflow = Estimator()
score_types = [FScoreStepDetection(name="F-score (step detection)")]


def get_train_data(path="."):
    return _read_data(path, 'train')


def get_test_data(path="."):
    return _read_data(path, 'test')


def get_cv(X, y):
    """
    In this cross-validation scheme, the proportion should have instances of each pathology in
    sufficient proportion, therefore the cross-validation is stratified according to the
    `pathology_group` attribute.

    TODO: Check if we should do the same with `sensor`.
    TODO: Ensure that for a single trial, the left and right signals are not in different folds
    and test/train sets.
    """
    cv = StratifiedShuffleSplit(
        n_splits=5, test_size=0.2, random_state=RANDOM_STATE)
    pathology_list = [signal.pathology_group for signal in X]
    return cv.split(X, pathology_list)
