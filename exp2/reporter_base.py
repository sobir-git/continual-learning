from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import List, Union, TypeVar, Type, Iterator, Sequence, Callable, Dict

import numpy as np
import sklearn
import torch
from typing_extensions import Protocol

from exp2.model_state import ModelState
from logger import Logger, get_accuracy
from utils.confusion_matrix import rectangular_confusion_matrix

T = TypeVar('T')
Tc = TypeVar("Tc", covariant=True)


class HasFinalContent(Protocol[Tc]):
    _final_content: Tc

    def compute_final_content(self) -> Tc: ...

    def is_ended(self) -> bool: ...


class ReporterBase:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__post_init__()

    def __post_init__(self):
        """Used to initialize instance variables."""
        pass


class ParentReporter(ReporterBase):
    children: List['ChildReporter']

    def __post_init__(self):
        super(ParentReporter, self).__post_init__()
        self.children = []

    def add_reporter(self, reporter):
        self.children.append(reporter)

    def find_descendant_nodes(self, class_: Type[T]) -> Iterator[T]:
        """Find all descendant nodes that are instances of the given class."""
        for child in self.children:
            if isinstance(child, class_):
                yield child
            if isinstance(child, ParentReporter):
                yield from child.find_descendant_nodes(class_)

    def find_descendant_node(self, class_: Type[T]) -> T:
        """Find one descendant node that is instance of the given class."""
        d = self.find_descendant_nodes(class_)
        return next(d)


class ChildReporter(ReporterBase, ABC):
    _ended_parents: List[ParentReporter]
    _parents: Iterator[ParentReporter]

    def __init__(self, parents: Iterator[ParentReporter], autoend: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parents = tuple(parents)
        self.autoend = autoend
        if parents:
            for p in parents:
                p.add_reporter(self)

    def __post_init__(self):
        super(ChildReporter, self).__post_init__()
        self._ended_parents = []

    def update(self, parent: ParentReporter, content):
        assert parent in self._parents

    def parent_ended(self, parent: ParentReporter, content):
        """Inform that a parent reporter has ended."""
        ended_parents = self._ended_parents
        parents = self._parents
        assert parent in parents
        assert parent not in ended_parents, "A parent can inform only once when it ends"
        ended_parents.append(parent)

        if self.autoend and isinstance(self, EndableReporter):
            if set(ended_parents) == set(parents):
                # all parents have been ended, end the report now
                self.end()

    def get_ended_parents(self):
        return self._ended_parents[:]

    def get_parents(self):
        return self._parents


class BasicReporter(ChildReporter, ParentReporter):
    """A report being child and parent"""

    def update(self, parent: ParentReporter, content):
        super(BasicReporter, self).update(parent, content)
        for child in self.children:
            child.update(parent, content)


class EndableReporter(BasicReporter):
    """
    A reporter that can end by calling .end() method and informing its children about it with its final content.
    Updateable, child and parent.
    """
    _ended: bool = False
    _final_content = None

    @abstractmethod
    def compute_final_content(self):
        """This method is called only once during .end() before informing the children the value of it."""
        ...

    def get_final_content(self: HasFinalContent[Tc]) -> Tc:
        if not self.is_ended():
            raise AttributeError('Report has not been ended yet.')
        return self._final_content

    def _end(self, final_content):
        self._final_content = final_content
        self._ended = True

    def is_ended(self):
        return self._ended

    def end(self):
        """End the report with the content = self.compute_final_content().
        This method should be called only once."""
        assert not self._ended
        final_content = self.compute_final_content()
        self._end(final_content)
        for child in self.children:
            child.parent_ended(self, final_content)

    def __del__(self):
        if not self.is_ended():
            raise ZeroDivisionError(self, "Report has not been ended while it is getting out of scope."
                                          " Did you forget to call .end() method?")


class SourceReporter(ParentReporter):
    children: List[ChildReporter]

    def __init__(self):
        super().__init__()

    def update(self, state: ModelState):
        for child in self.children:
            child.update(self, state)

    def end(self):
        for child in self.children:
            child.parent_ended(self, None)


class LoggerBase(EndableReporter):
    """A reporter that logs."""

    def __init__(self, logger: Logger, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger

    @abstractmethod
    def log_at_end(self):
        """This method is called immediately after .end(). So you can use .get_final_content()"""

    def end(self):
        super().end()
        self.log_at_end()


TReporter = TypeVar("TReporter", bound=ReporterBase)


class MetricLogger(LoggerBase):
    """Logs a simple key-value metric at the end, like accuracy, average loss."""

    def __init__(self, name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name

    def log(self, value):
        self.logger.log({self.name: value})

    @abstractmethod
    def compute_final_value(self):
        """Method that is called at the .end()"""

    def compute_final_content(self):
        return self.compute_final_value()

    def get_value(self):
        return self.get_final_content()

    def log_at_end(self):
        value = self.get_value()
        self.log(value)


class Concatenator(EndableReporter):
    """Concatenates tensors/array/values returned by method .extract_tensor().
    The result is automatically computed at .end() and accessible via .get_final_content()"""

    def __init__(self, *args, dim=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def __post_init__(self):
        super(Concatenator, self).__post_init__()
        self._tensors = []

    @abstractmethod
    def extract_tensor(self, content) -> Tc:
        """Extract the required tensor or numpy array or even any value that you want to concatenate."""

    def update(self, _, content):
        tensor = self.extract_tensor(content)
        self._tensors.append(tensor)

    def compute_final_content(self) -> Tc:
        if len(self._tensors) == 0:
            return None
        if isinstance(self._tensors[0], torch.Tensor):
            return torch.cat(self._tensors, self.dim)
        elif isinstance(self._tensors[0], np.ndarray):
            return np.concatenate(self._tensors, self.dim)
        else:
            return np.array(self._tensors)


class IncrementalMetricLogger(MetricLogger, Concatenator):
    """Incrementally computes a metric at each update and logs at the end."""

    def __init__(self, reduce_func: Union[str, Callable] = 'mean', *args, **kwargs):
        super().__init__(*args, **kwargs)
        if reduce_func == 'mean':
            reduce_func = np.mean
        self.reduce_func = reduce_func

    @abstractmethod
    def extract_values(self, content) -> Union[float, int, List[Union[float, int]], np.ndarray, torch.Tensor]:
        ...

    def compute_final_value(self):
        concatenated = Concatenator.compute_final_content(self)
        return self.reduce_func(concatenated)

    def extract_tensor(self, content) -> Tc:
        return self.extract_values(content)


class LabelGathererBase(Concatenator):
    @abstractmethod
    def extract_labels(self, content):
        ...

    def extract_tensor(self, content) -> Tc:
        return self.extract_labels(content)

    def get_labels(self):
        return self.get_final_content()


class IdsGatherer(Concatenator):
    def __init__(self, source: SourceReporter):
        super().__init__(parents=[source])

    def extract_tensor(self, content: ModelState) -> Tc:
        return content.ids


class LabelGatherer(LabelGathererBase):
    """Gathers labels from ModelState object."""

    def __init__(self, source: SourceReporter):
        parents = [source]
        super().__init__(parents)

    def extract_labels(self, content: ModelState):
        labels = content.labels_np
        return labels

    def get_labels(self):
        return self.get_final_content()


class PredictionReporterBase:
    """Has .get_predictions() method."""

    @abstractmethod
    def get_predictions(self):
        ...


class PredictionReporter(PredictionReporterBase, Concatenator):
    """Obtains predictions at each update and reports them at the end."""

    @abstractmethod
    def obtain_predictions(self, content):  # TODO: ctrl.accuracy is too low: https://wandb.ai/sobir/exp2/runs/12hj00w5
        """Obtain prediction from given update content. Called at each update."""
        ...

    def extract_tensor(self, content) -> Tc:
        return self.obtain_predictions(content)

    def get_predictions(self):
        return self.get_final_content()


class AccuracyLogger(MetricLogger):
    def __init__(self, logger: Logger, label_gatherer: LabelGatherer,
                 prediction_reporter: PredictionReporterBase, name: str):
        parents = [label_gatherer, prediction_reporter]
        super().__init__(name=name, logger=logger, parents=parents)
        self.label_gatherer = label_gatherer
        self.prediction_reporter = prediction_reporter

    def compute_final_value(self):
        predictions = self.prediction_reporter.get_predictions()
        labels = self.label_gatherer.get_labels()
        accuracy = get_accuracy(predictions=predictions, labels=labels)
        return accuracy


TClassnames = Sequence[Union[int, str]]


class ConfusionMatrixReporterBase(EndableReporter):
    def __init__(self, prediction_reporter: PredictionReporterBase, label_gatherer: LabelGatherer, *args, **kwargs):
        parents = [prediction_reporter, label_gatherer]
        super().__init__(parents)
        self._prediction_reporter = prediction_reporter
        self._label_gatherer = label_gatherer

    def get_predictions(self):
        return self._prediction_reporter.get_predictions()

    def get_labels(self):
        return self._label_gatherer.get_labels()


@dataclass
class ConfusionMatrixReport:
    confusion_matrix: np.ndarray
    classes: List[int]


class ConfusionMatrixReporter(ConfusionMatrixReporterBase):

    def __init__(self, prediction_reporter: PredictionReporterBase, label_gatherer: LabelGatherer,
                 classes: List[int] = None):
        super().__init__(prediction_reporter, label_gatherer)
        self.classes = classes

    def compute_final_content(self):
        predictions = self.get_predictions()
        labels = self.get_labels()
        matrix = sklearn.metrics.confusion_matrix(labels, predictions, labels=self.classes)
        if self.classes is None:
            self.classes = list(range(matrix.shape[0]))
        return ConfusionMatrixReport(matrix, self.classes)


@dataclass
class RectangularConfusionMatrixReport:
    confusion_matrix: np.ndarray
    true_classes: List[int]  # actual classes, making up rows of confusion matrix
    pred_classes: List[int]  # predictable classes, making up columns of confusion matrix


class RectangularConfusionMatrixReporter(ConfusionMatrixReporterBase):

    def __init__(self, prediction_reporter: PredictionReporterBase, label_gatherer: LabelGatherer,
                 true_classes: List[int] = None, pred_classes: List[int] = None):
        super().__init__(prediction_reporter, label_gatherer)
        self.pred_classes = pred_classes
        self.true_classes = true_classes

    def compute_final_content(self):
        predictions = self.get_predictions()
        labels = self.get_labels()
        matrix = rectangular_confusion_matrix(labels, predictions, true_labels=self.true_classes,
                                              pred_labels=self.pred_classes)
        if self.true_classes is None:
            self.true_classes = list(range(matrix.shape[0]))
        if self.pred_classes is None:
            self.pred_classes = list(range(matrix.shape[1]))
        return RectangularConfusionMatrixReport(matrix, self.true_classes, self.pred_classes)


class ConfusionMatrixLogger(LoggerBase):
    def __init__(self, logger: Logger, confusion_reporter: ConfusionMatrixReporterBase, name: str, title: str,
                 classnames: Dict[int, str] = None, *args, **kwargs):
        parents = [confusion_reporter]
        super().__init__(parents=parents, logger=logger, *args, **kwargs)
        self.confusion_reporter = confusion_reporter
        self.name = name
        self.title = title
        self.classnames = classnames

    def log_at_end(self):
        report = self.confusion_reporter.get_final_content()
        confusion_matrix = report.confusion_matrix
        classes, true_classes, pred_classes = None, None, None
        if isinstance(report, ConfusionMatrixReport):
            classes = report.classes
        elif isinstance(report, RectangularConfusionMatrixReport):
            true_classes = report.true_classes
            pred_classes = report.pred_classes

        def map_classes(_classes):
            if _classes is not None:
                return [self.classnames[cls] for cls in _classes]

        if self.classnames is not None:
            classes = map_classes(classes)
            true_classes = map_classes(true_classes)
            pred_classes = map_classes(pred_classes)

        self.logger.log_confusion_matrix(confusion_matrix, labels=classes, true_labels=true_classes,
                                         pred_labels=pred_classes, title=self.title, name=self.name)

    def compute_final_content(self):
        pass
