CTRL_EPOCH = 'ctrl/epoch'
CTRL_TRAIN_LOSS = 'ctrl/tr_loss'
CTRL_VAL_LOSS = 'ctrl/val_loss'
CTRL_CONF_MTX_TITLE = 'Confusion(Controller)'
CTRL_CONF_MTX = 'ctrl.conf_mtx'
CTRL_ACC = 'ctrl.accuracy'
CLF_EPOCH = 'clf/epoch'
CLF_CONF_MTX_TITLE = "Confusion (classifier {idx})"
FINAL_ACC = "final/{name}.accuracy"
FINAL_CONF_MTX = "final/{name}.conf_mtx"
FINAL_CONF_MTX_TITLE = "Confusion final ({name})"


class CLF_CONF_MTX:
    @staticmethod
    def format(idx, is_open, is_exclusive):
        openness = 'open' if is_open else 'closed'
        inclusiveness = '_excl' if is_exclusive else ''
        s = f'clf/{idx}/{openness}{inclusiveness}.conf_mtx'
        return s


class CLF_LOSS:
    @staticmethod
    def format(idx, is_train=False, is_validation=False):
        is_train_str = 'tr_' if is_train else ''
        is_validation_str = 'val_' if is_validation else ''
        s = f'clf/{idx}/{is_train_str}{is_validation_str}loss'
        return s


class CLF_ACC:
    @staticmethod
    def format(idx, is_open=True, is_exclusive=False, is_test=False, is_validation=False):
        openness = 'open' if is_open else 'closed'
        inclusiveness = '_excl' if is_exclusive else ''
        s = f'clf/{idx}/{openness}{inclusiveness}.accuracy'
        return s
