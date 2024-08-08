from matplotlib import pyplot as plt


def plot_lr_schedule(scheduler, num_epochs):
    """
    Utility function to plot the learning rate schedule.

    Args:
        scheduler (DecayingCosineAnnealingWarmRestarts): The scheduler to validate.
        num_epochs (int): Number of epochs to simulate.

    Returns:
        None: Displays a plot of the learning rate schedule.
    """
    lr_history = []
    for epoch in range(num_epochs):
        current_lrs = [group['lr'] for group in scheduler.optimizer.param_groups]
        lr_history.append(current_lrs)
        scheduler.step()

    epochs = list(range(num_epochs))
    min_lrs = [min(lrs) for lrs in lr_history]
    max_lrs = [max(lrs) for lrs in lr_history]

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, min_lrs, label='Min LR', color='blue')
    plt.plot(epochs, max_lrs, label='Max LR', color='red')
    plt.fill_between(epochs, min_lrs, max_lrs, alpha=0.2, color='gray')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.yscale('log')  # Use log scale for better visualization
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.show()
