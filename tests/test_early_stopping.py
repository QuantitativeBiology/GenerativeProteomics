import pytest
from models.early_stopping import EarlyStopping

def test_early_stopping_triggers():
    patience = 3
    early_stopper = EarlyStopping(patience=patience)

    losses = [1.0, 0.9, 0.9, 0.91, 0.92, 0.93]
    epochs = list(range(1, 7))

    for loss, epoch in zip(losses, epochs):
        print(f"Loss: {loss}, Epoch: {epoch}")
        early_stopper.step(loss, epoch)

    assert early_stopper.stop is True
    assert early_stopper.counter == patience
    assert early_stopper.best_value == 0.9


def test_early_stopping_resets_on_improvement():
    patience = 2
    early_stopper = EarlyStopping(patience=patience)

    early_stopper.step(1.0, 1)  
    early_stopper.step(0.9, 2)  
    early_stopper.step(0.95, 3)  # no improvement: counter=1
    early_stopper.step(0.85, 4)  # improvement: counter resets

    assert early_stopper.counter == 0
    assert early_stopper.best_value == 0.85
    assert early_stopper.stop is False