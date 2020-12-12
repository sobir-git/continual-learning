import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from logger import Logger
from utils import AverageMeter, Timer, get_prediction


class TrainerBase:
    def __init__(self, opt, model, logger: Logger, device, optimizer):
        self.optimizer = optimizer
        self.opt = opt
        self.model = model
        self.logger = logger
        self.device = device

    def _before_train(self):
        self.logger.push_pref('train')
        self.model.train()

    def _after_train(self):
        self.logger.commit()
        self.logger.pop_pref()

    def _before_test(self):
        self.logger.push_pref('test')
        self.model.eval()

    def _after_test(self):
        self.logger.commit()
        self.logger.pop_pref()

    def _get_log_every(self, n_batches):
        return max(n_batches // 5, 1)

    def set_logger(self, logger: Logger):
        self.logger = logger

    def train(self, loader, n_loops):
        raise NotImplementedError

    def test(self, loader: DataLoader, classnames, mask):
        raise NotImplementedError


class StandardTrainer(TrainerBase):
    _default_criterion = torch.nn.CrossEntropyLoss()

    def __init__(self, opt, logger: Logger, model, device, optimizer, criterion=None):
        super().__init__(opt, model, logger, device, optimizer)
        assert not opt.regularization == 'cutmix', "we cannot apply cutmix"
        self.criterion = (criterion if criterion else self._default_criterion).to(device)

    def _train(self, x, y):
        """
        Trains the model with given the given batch (x, y). Assumes x, y are in the same device as the model
        Args:
            x: input
            y: labels

        Returns:
            Loss (single number)
        """
        output = self.model(x)
        loss = self.criterion(output, y)

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.clip)  # Always be safe than sorry
        self.optimizer.step()

        return loss

    def train(self, dataloader: DataLoader, num_loops=1):
        """
        Trains the model on the given dataloader for num_loops passes.
        Args:
            dataloader:
            num_loops: number of passes over the dataloader

        Returns:
            The total time it took for data-loading/processing stuff.
        """
        self._before_train()

        device = self.device

        data_time = Timer()  # the time it takes for forward+backward+step
        loss_meter = AverageMeter()
        log_every = self._get_log_every(len(dataloader))

        for i in range(num_loops):
            for batch_idx, (inputs, labels) in enumerate(data_time.get_timed_generator(dataloader)):
                inputs, label = inputs.to(device), labels.to(device)
                loss = self._train(inputs, labels)
                loss_meter.update(loss, inputs.size(0))

                if batch_idx % log_every == 0:
                    self.logger.log({'loss': loss_meter.avg, 'percent': (batch_idx + 1) / len(dataloader)}, commit=True)
                    # reset meters
                    loss_meter.reset()

        self._after_train()
        return data_time.total

    @torch.no_grad()
    def test(self, loader: DataLoader, classnames, mask):
        """
        Test the model.
        Returns loss, accuracy.
        """

        self._before_test()
        device = self.device

        # holds predictions of main(combined) and indiviual branches
        trues = []
        preds = []
        loss_meter = AverageMeter()
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = self.model(inputs)
            loss_meter.update(self.criterion(output, labels), inputs.size(0))

            probs = torch.softmax(output, 1)
            preds.extend(get_prediction(probs, mask))
            trues.extend(labels)

        # report confusion matrix, accuracies, recalls
        confmatrix = confusion_matrix(trues, preds, labels=range(len(classnames)))
        self.logger.log_confusion_matrix(confmatrix, classnames)
        accuracy = self.logger.log_accuracies(confmatrix, classnames)
        loss = self.logger.log({'loss': loss_meter.avg})

        self._after_test()
        return loss, accuracy
