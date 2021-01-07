import itertools
from collections import defaultdict

from utils import np_a_in_b
from exp2.classifier import Classifier
from exp2.controller import Controller
from exp2.predictor import Givens, Predictor
from exp2.reporter_strings import *
from exp2.reporter_base import *
from logger import Logger, get_accuracy


class IdsLogger(LoggerBase):

    def __init__(self, ids_gatherer: IdsGatherer, logger: Logger, output_file: str, mode='w'):
        super().__init__(logger, parents=[ids_gatherer])
        self.ids_gatherer = ids_gatherer
        self.output_file = output_file
        self.mode = mode

    def compute_final_content(self):
        pass

    def log_at_end(self):
        ids = self.ids_gatherer.get_final_content()
        ids = ids.tolist()
        with open(self.output_file, self.mode) as f:
            f.write(str(ids))
            f.write('\n')


class ControllerPredictionReporter(PredictionReporter):
    def __init__(self, source: SourceReporter):
        parents = [source]
        super().__init__(parents)

    def obtain_predictions(self, content: 'ModelState'):
        ctrl_outputs = content.controller_state.outputs
        controller = content.get_controller()
        predictions = controller.get_predictions(ctrl_outputs)
        return predictions


class OutputsReporter(Concatenator):
    """Reports outputs at the end"""

    @abstractmethod
    def extract_outputs(self, content):
        pass

    def extract_tensor(self, content) -> Tc:
        return self.extract_outputs(content)

    def get_outputs(self):
        return self.get_final_content()


class ControllerOutputsReporter(OutputsReporter):
    def __init__(self, source: SourceReporter):
        parents = [source]
        super().__init__(parents)

    def extract_outputs(self, content: 'ModelState'):
        ctrl_outputs = content.controller_state.outputs
        return ctrl_outputs


class ControllerConfusionMatrixLogger(ConfusionMatrixLogger):
    def __init__(self, logger: Logger, confusion_reporter: ConfusionMatrixReporter, classnames: TClassnames = None):
        super().__init__(logger, confusion_reporter, CTRL_CONF_MTX, CTRL_CONF_MTX_TITLE, classnames)


class ControllerLabelGatherer(LabelGatherer):
    def extract_labels(self, content: 'ModelState'):
        controller = content.get_controller()
        labels = content.labels_np
        ctrl_labels = controller.group_labels(labels)
        return ctrl_labels


class ControllerAccuracyLogger(AccuracyLogger):
    def __init__(self, logger: Logger, label_gatherer: ControllerLabelGatherer,
                 prediction_reporter: ControllerPredictionReporter, name: str):
        super().__init__(name=name, logger=logger, label_gatherer=label_gatherer,
                         prediction_reporter=prediction_reporter)
        self.label_gatherer = label_gatherer
        self.prediction_reporter = prediction_reporter

    def compute_final_value(self):
        predictions = self.prediction_reporter.get_predictions()
        labels = self.label_gatherer.get_labels()
        accuracy = get_accuracy(predictions=predictions, labels=labels)
        return accuracy


class ControllerLossLogger(IncrementalMetricLogger):
    def __init__(self, source: SourceReporter, logger, name):
        parents = [source]
        super().__init__(parents=parents, logger=logger, name=name)

    def extract_values(self, state: 'ModelState'):
        ctrl_state = state.controller_state
        loss = ctrl_state.loss
        assert loss is not None
        batch_size = state.batchsize
        losses = np.full(batch_size, loss.item())  # we want average sample loss
        return losses


class ClassifierBaseReporter(BasicReporter):
    def __init__(self, classifier: Classifier, **kwargs):
        super().__init__(**kwargs)
        self.classifier = classifier

    def get_classifier_state(self, state: 'ModelState'):
        return state.get_classifier_state(self.classifier)


class ClassifierLossLogger(ClassifierBaseReporter, IncrementalMetricLogger):
    def __init__(self, classifier, source: SourceReporter, name: str, logger: Logger):
        super().__init__(classifier=classifier, parents=[source], name=name, logger=logger)

    def extract_values(self, state: 'ModelState'):
        clf_state = self.get_classifier_state(state)
        loss, batch_size = clf_state.loss, clf_state.batchsize
        losses = np.full(batch_size, loss.item())  # we want average sample loss
        return losses


class ClassifierPredictionReporter(ClassifierBaseReporter, PredictionReporter):
    def __init__(self, classifier: Classifier, source: SourceReporter, is_open: bool, is_exclusive: bool):
        super().__init__(classifier, parents=[source])
        self.is_open = is_open
        self.is_exclusive = is_exclusive

    def obtain_predictions(self, state: 'ModelState'):
        clf_state = self.get_classifier_state(state)
        outputs = clf_state.outputs
        labels = clf_state.labels_np
        classes = self.classifier.classes
        if self.is_exclusive:
            indices = np_a_in_b(labels, classes)
            outputs = outputs[indices]
        predictions = self.classifier.get_predictions(outputs, is_open=self.is_open)
        return predictions

    def get_predictions(self):
        return self.get_final_content()


class ClassifierLabelGatherer(ClassifierBaseReporter, LabelGatherer):
    def __init__(self, is_exclusive: bool, classifier: Classifier, source: SourceReporter, ):
        super().__init__(classifier, source=source)
        self.is_exclusive = is_exclusive

    def extract_labels(self, state: 'ModelState'):
        labels_np = state.labels_np
        if self.is_exclusive:
            classes = self.classifier.classes
            indices = np_a_in_b(labels_np, classes)
            labels_np = labels_np[indices]
        else:
            labels_np = self.classifier.map_other(labels_np)
        return labels_np


class ClassifierAccuracyLogger(ClassifierBaseReporter, AccuracyLogger):
    def __init__(self, classifier: Classifier, name, logger: Logger, label_gatherer: ClassifierLabelGatherer,
                 prediction_reporter: ClassifierPredictionReporter):
        super().__init__(classifier, label_gatherer=label_gatherer, prediction_reporter=prediction_reporter, name=name,
                         logger=logger)

        assert label_gatherer.is_exclusive == prediction_reporter.is_exclusive
        self.label_gatherer = label_gatherer
        self.prediction_reporter = prediction_reporter


class PredictionGivensReporter(EndableReporter):
    """Reports Givens object that is used as input to prediction algorithms."""

    def __init__(self, classifiers: List[Classifier],
                 ctrl_outputs_reporter: ControllerOutputsReporter,
                 ctrl_prediction_reporter: ControllerPredictionReporter,
                 clf_open_prediction_reporters: List[ClassifierPredictionReporter],
                 clf_closed_prediction_reporters: List[ClassifierPredictionReporter]):
        parents = [ctrl_outputs_reporter, ctrl_prediction_reporter, *clf_open_prediction_reporters,
                   *clf_closed_prediction_reporters]
        super().__init__(parents)
        self.clf_closed_prediction_reporters = clf_closed_prediction_reporters
        self.clf_open_prediction_reporters = clf_open_prediction_reporters
        self.ctrl_prediction_reporter = ctrl_prediction_reporter
        self.ctrl_outputs_reporter = ctrl_outputs_reporter
        self.classifiers = classifiers

    def compute_final_content(self):
        clf_preds_open = [reporter.get_predictions() for reporter in self.clf_open_prediction_reporters]
        clf_preds_closed = [reporter.get_predictions() for reporter in self.clf_closed_prediction_reporters]
        givens = Givens(self.classifiers,
                        self.ctrl_outputs_reporter.get_outputs(),
                        self.ctrl_prediction_reporter.get_predictions(),
                        clf_preds_open=clf_preds_open,
                        clf_preds_closed=clf_preds_closed)
        return givens


class FinalPredictionReporter(PredictionReporterBase, EndableReporter):
    def __init__(self, givens_reporter: PredictionGivensReporter, algorithm: Predictor):
        super().__init__([givens_reporter])
        self.algorithm = algorithm
        self.givens_reporter = givens_reporter

    def compute_final_content(self):
        givens = self.givens_reporter.get_final_content()
        predictions = self.algorithm(givens)
        return predictions

    def get_predictions(self):
        return self.get_final_content()


class ClassifierReporter:
    def __init__(self, config, logger, source: SourceReporter, classifier: Classifier, mode: str):
        self.source = source
        self.config = config
        assert mode in ['train', 'validation']
        if mode == 'train':
            self._create_classifier_train_reporter(logger, source, classifier)
        else:
            self._create_classifier_validation_reporter(logger, source, classifier)

    def get_average_loss(self):
        loss_logger = self.source.find_descendant_node(ClassifierLossLogger)
        return loss_logger.get_value()

    def get_accuracy(self, is_open, is_exclusive):
        predicate = lambda x: x.is_open == is_open and x.is_exclusive == is_exclusive
        nodes = self.source.find_descendant_nodes(ClassifierAccuracyLogger)
        accuracy_logger = next(filter(predicate, nodes))
        return accuracy_logger.get_value()

    def _create_classifier_train_reporter(self, logger: Logger, source: SourceReporter, classifier: Classifier):
        """Report loss."""
        loss_name = CLF_LOSS.format(idx=classifier.idx, is_train=True)
        ClassifierLossLogger(classifier, source, loss_name, logger)

        # log ids
        ids_gatherer = IdsGatherer(source)
        output_file = self.config.logdir + '/' + 'clf_train_ids.txt'
        IdsLogger(ids_gatherer, logger, output_file, mode='a')
        return source

    def _create_classifier_validation_reporter(self, logger: Logger, source: SourceReporter, classifier: Classifier):
        """Report loss, accuracy"""
        name = CLF_LOSS.format(idx=classifier.idx, is_validation=True)
        ClassifierLossLogger(classifier, source, name, logger)

        # log ids
        ids_gatherer = IdsGatherer(source)
        output_file = self.config.logdir + '/' + 'clf_val_ids.txt'
        IdsLogger(ids_gatherer, logger, output_file, mode='a')
        return source


class ControllerReporter:
    def __init__(self, config, logger: Logger, source: SourceReporter, controller: Controller, mode: str):
        self.source = source
        self.config = config
        assert mode in ['train', 'validation']
        if mode == 'train':
            self._create_controller_train_reporter(logger, source)
        else:
            self._create_controller_validation_reporter(logger, source, controller)

    def get_average_loss(self):
        loss_logger = self.source.find_descendant_node(ControllerLossLogger)
        return loss_logger.get_value()

    def get_accuracy(self):
        accuracy_logger = self.source.find_descendant_node(ControllerAccuracyLogger)
        return accuracy_logger.get_value()

    def _create_controller_train_reporter(self, logger: Logger, source: SourceReporter):
        """Report loss."""
        ControllerLossLogger(source, logger, name=CTRL_TRAIN_LOSS)
        ids_gatherer = IdsGatherer(source)
        output_file = self.config.logdir + '/' + 'ctrl_train_ids.txt'
        IdsLogger(ids_gatherer, logger, output_file, mode='a')
        return source

    def _create_controller_validation_reporter(self, logger: Logger, source: SourceReporter, controller: Controller):
        """Report loss, accuracy"""
        prediction_reporter = ControllerPredictionReporter(source)
        label_gatherer = ControllerLabelGatherer(source)
        ControllerLossLogger(source, logger, name=CTRL_VAL_LOSS)
        ControllerAccuracyLogger(logger, label_gatherer, prediction_reporter, CTRL_VAL_ACC)
        ids_gatherer = IdsGatherer(source)
        output_file = self.config.logdir + '/' + 'ctrl_val_ids.txt'
        IdsLogger(ids_gatherer, logger, output_file, mode='a')
        return source


def create_test_reporter(config, logger: Logger, source, classifiers: List[Classifier], controller: Controller,
                         classes: List[int]):
    """
    Reported metrics:
        - overall accuracy
        - controller
            - accuracy
            - confusion matrix
        - per classifier
            - open accuracy (with class weights)
            - closed accuracy
            - confusion matrices (open and closed)

    Args:
        config:
        logger:
        source:
        classifiers:
        controller:

    Returns:

    """
    ctrl_prediction_reporter = ControllerPredictionReporter(source)
    clf_names = [clf.idx for clf in controller.classifiers]
    ctrl_label_gatherer = ControllerLabelGatherer(source)
    ctrl_confusion_reporter = ConfusionMatrixReporter(ctrl_prediction_reporter, ctrl_label_gatherer)
    ControllerConfusionMatrixLogger(logger, ctrl_confusion_reporter, clf_names)
    ControllerAccuracyLogger(logger, ctrl_label_gatherer, ctrl_prediction_reporter, CTRL_ACC)

    ids_gatherer = IdsGatherer(source)
    output_file = config.logdir + '/' + 'test_ids.txt'
    IdsLogger(ids_gatherer, logger, output_file, mode='a')

    clf_predictors = defaultdict(list)

    for clf in classifiers:
        for is_open, is_exclusive in itertools.product([True, False], repeat=2):
            name = CLF_ACC.format(clf.idx, is_open, is_exclusive, is_test=True)
            label_gatherer = ClassifierLabelGatherer(is_exclusive, clf, source)
            prediction_reporter = ClassifierPredictionReporter(clf, source, is_open, is_exclusive)
            ClassifierAccuracyLogger(clf, name, logger, label_gatherer, prediction_reporter)
            clf_classes = clf.classes + [clf.other_label]
            conf_reporter = ConfusionMatrixReporter(prediction_reporter, label_gatherer, clf_classes)
            name = CLF_CONF_MTX.format(idx=clf.idx, is_open=is_open, is_exclusive=is_exclusive)
            title = CLF_CONF_MTX_TITLE.format(idx=clf.idx)
            ConfusionMatrixLogger(logger, conf_reporter, name=name, title=title)

            if not is_exclusive:
                clf_predictors[is_open].append(prediction_reporter)

    clf_open_prediction_reporters = clf_predictors[True]
    clf_closed_prediction_reporters = clf_predictors[False]
    plain_label_gatherer = LabelGatherer(source)
    ctrl_outputs_reporter = ControllerOutputsReporter(source)
    givens_reporter = PredictionGivensReporter(classifiers, ctrl_outputs_reporter, ctrl_prediction_reporter,
                                               clf_open_prediction_reporters, clf_closed_prediction_reporters)
    algorithms = [cls() for cls in Predictor.__subclasses__()]
    for algorithm in algorithms:
        final_prediction_reporter = FinalPredictionReporter(givens_reporter, algorithm)
        acc_name = FINAL_ACC.format(name=algorithm.name)
        AccuracyLogger(logger, plain_label_gatherer, final_prediction_reporter, acc_name)
        final_confusion = ConfusionMatrixReporter(final_prediction_reporter, plain_label_gatherer, classes)
        name = FINAL_CONF_MTX.format(name=algorithm.name)
        title = FINAL_CONF_MTX_TITLE.format(name=algorithm.name)
        ConfusionMatrixLogger(logger, final_confusion, name=name, title=title)
    return source
