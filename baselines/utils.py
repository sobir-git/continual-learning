import torch
from torch.optim import lr_scheduler as sch


def create_optimizer(config, model, lr):
    if config.faster_output_learning_rate:
        final = get_final_layer(model)
        all_except_final = list(filter(lambda p: all(p is not i for i in final.parameters()), model.parameters()))
        optimizer = torch.optim.SGD([{'params': all_except_final},
                                     {'params': final.parameters(), 'lr': lr * 10}],
                                    momentum=0.9, lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return optimizer


def create_lr_scheduler(config, optimizer):
    if config.lr_scheduler == 'exp':
        return sch.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.gamma)
    elif config.lr_scheduler == 'reduce_on_plateu':
        return sch.ReduceLROnPlateau(optimizer, 'min', factor=config.gamma,
                                     patience=config.lr_patience,
                                     verbose=True, min_lr=config.min_lr)
    else:
        raise ValueError(f"config.lr_scheduler be one of 'exp', 'reduce_on_plateu' but got {config.lr_scheduler}")


def scheduler_step(lr_scheduler, loss):
    if isinstance(lr_scheduler, sch.ExponentialLR):
        lr_scheduler.step()
    elif isinstance(lr_scheduler, sch.StepLR):
        lr_scheduler.step()
    elif isinstance(lr_scheduler, sch.ReduceLROnPlateau):
        lr_scheduler.step(loss)


def get_last_learning_rate(lr_scheduler):
    try:
        return lr_scheduler.get_last_lr()[0]
    except AttributeError:
        pass

    try:
        return lr_scheduler._last_lr[0]
    except AttributeError:
        pass

    return lr_scheduler.optimizer.param_groups[0]['lr']


def get_final_layer(model) -> torch.nn.Linear:
    while not isinstance(model, torch.nn.Linear):
        last_child = list(model.children())[-1]
        model = last_child
    return model


def get_dataset_weight(dataset):
    if len(dataset) == 0:
        return 0
    return len(dataset.classes) / len(dataset)
