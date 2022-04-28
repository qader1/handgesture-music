from tqdm.autonotebook import tqdm
import torch.optim as optim
import torch.utils.data as data

from models import *
from logger import *
from dataset import HandGestures
import albumentations as al
from albumentations.pytorch import ToTensorV2
import cv2

from sklearn.metrics import classification_report
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_data(batch_size: int, path: str):
    """
    function that creates an instance of the custom dataset, applies augmentations and returns
    3 instances of dataloader, one for training, one for training validation and one for testing
    :param batch_size:
    :param path: path of the data folder
    :return: 3 dataloader instances for train, validation and test
    """
    # choose data augmentation.
    # CLAHE equalizes the intensities of colors, and creates artifacts that serve as noise.
    # color jitter changes brightness saturation and hue. (color augmentation)
    # fancyPCA changes colors (color augmentation)
    # randomGamma increases gamma (color augmentation)
    # downscale (lower resolution)
    # ISONoise adds ISO noise (makes the image grainy)
    # Rotates the image with a certain range
    # Normalize uses the mean and std of ImageNet
    data_transform = al.Compose([
        al.CLAHE(clip_limit=50, tile_grid_size=(8, 20), p=.3),
        al.OneOf([al.ColorJitter(brightness=.3, saturation=.2, hue=0.1),
                 al.FancyPCA(),
                 al.RandomGamma(gamma_limit=(30, 120))], p=.3),
        al.OneOf([al.Blur(blur_limit=5, p=1),
                  al.ISONoise(color_shift=(.01, .10), intensity=(.1, .6), p=.25)], p=.3),
        al.GridDistortion(num_steps=15, distort_limit=.5,
                          interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REPLICATE, p=.3),
        al.Rotate(limit=(-21, 21), p=.4),
        al.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(transpose_mask=True)
    ])

    data_transform_t = al.Compose([
        al.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(transpose_mask=True)
    ])

    train_dataset, test_dataset = HandGestures(path, transform=data_transform)\
        .train_test_split(test_transform=data_transform_t)

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*batch_size, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*batch_size, shuffle=False)
    return train_loader, train_loader_at_eval, test_loader


def train(epochs, train_dataloader, val_dataloader, model, optimizer, loss_fn):
    """
    function to train model. each epoch both train and validation loss and accuracy are printed to the console to
    observe the training behaviour.

    :param epochs: no. of epochs to train
    :param train_dataloader: training data as a dataloader object
    :param val_dataloader: test data; dataloader object
    :param model: architecture to train
    :param optimizer: optimizing algorithm
    :param loss_fn: loss function
    :return: model and a tuple of metrics
    """
    loss_train_epochs = []
    accuracy_train_epochs = []
    loss_test_epochs = []
    accuracy_test_epochs = []
    last_30 = int(epochs * 0.25)
    current_optimizer = optimizer
    if type(optimizer) == tuple:
        current_optimizer = optimizer[0]
    for epoch in range(epochs):
        running_loss_train = 0
        correct_train = 0
        total_train = 0

        # set the model into train mode
        model.train()
        for inputs, targets in tqdm(train_dataloader):
            # forward + backward + optimize
            current_optimizer.zero_grad()
            outputs = model(inputs.to(device))
            targets = targets.squeeze().long().to(device)
            loss = loss_fn(outputs, targets)
            loss.backward()
            if type(optimizer) == tuple:
                if epoch >= last_30:
                    current_optimizer = optimizer[1]

            current_optimizer.step()
            running_loss_train += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()

        loss_train_epochs.append(running_loss_train/len(train_dataloader))
        accuracy_train_epochs.append(100.*correct_train/total_train)

        running_loss_test = 0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            model.eval()
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.squeeze().long().to(device)

                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                running_loss_test += loss.item()
                _, predicted = outputs.max(1)
                total_test += targets.size(0)
                correct_test += predicted.eq(targets).sum().item()

        loss_test_epochs.append(running_loss_test/len(val_dataloader))
        accuracy_test_epochs.append(100.*correct_test/total_test)
        print(f'epoch: {epoch}',
              f'train loss {loss_train_epochs[epoch]:4f}, test loss {loss_test_epochs[epoch]:4f}',
              f'train accu {accuracy_train_epochs[epoch]:4f}, test accu {accuracy_test_epochs[epoch]:4f}', sep='\n')
    return model, (loss_train_epochs, accuracy_train_epochs, loss_test_epochs, accuracy_test_epochs)


def test(model, dataloader):
    """
    function that calculates evaluation metrics for the trained model. it's used for both the test set, and training set
    without augmentation
    :param model: model to evaluate
    :param dataloader: test set or training set without the augmentation
    :return: classification report string from scikit-learn
    """
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_predicted = torch.tensor([]).to(device)
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs.to(device))

            targets = targets.squeeze().long()
            outputs = outputs.softmax(dim=-1)
            targets = targets.float().resize_(len(targets), 1)
            targets = targets.to(device)

            y_true = torch.cat((y_true, targets), 0)
            y_predicted = torch.cat((y_predicted, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_predicted.detach().cpu().numpy()

        return classification_report(y_true.squeeze(), y_score.argmax(axis=1))


def start(args):
    """
    function that takes arguments from the start subparser and instantiate the model, optimizer and loss function
    according to them.
    it calls the training, test and log functions.

    :param args: arguments in subparser. model number as integer, type of optimizer, learning rate, batch size and
    no of epochs
    :return: None
    """
    data_path = r'C:\Users\Aboud\Datasets\NUS_hand_posture_data_2\Data'
    epochs = args.epochs
    train_data, train_val_data, test_data = get_data(args.batch_size, data_path)
    lr = args.lr
    model = eval('Var'+args.model)(num_classes=8)
    model = model.to(device)
    sgd = optim.SGD(model.parameters(), lr=lr, momentum=.9)
    adam = optim.Adam(model.parameters(), lr=lr)
    if args.optimizer == 'SGD':
        optimizer = sgd
    elif args.optimizer == 'Adam':
        optimizer = adam
    else:
        optimizer = (adam, sgd)
    loss_fn = nn.CrossEntropyLoss()

    trained_model, scores = train(epochs,
                                  train_data,
                                  test_data,
                                  model,
                                  optimizer,
                                  loss_fn)

    loss_train, acc_train, loss_test, acc_test = scores

    figs = plot_train_test(loss_train,
                           acc_train,
                           loss_test,
                           acc_test)

    reports = test(trained_model, train_val_data), test(trained_model, test_data)

    log(model,
        args.batch_size,
        args.lr,
        args.optimizer,
        epochs,
        figs,
        reports,
        (acc_train[-1], acc_test[-1]),
        len(train_data.dataset),
        len(test_data.dataset))


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()
new = subparsers.add_parser('start')
positional = new.add_argument_group('required arguments')
positional.add_argument('-ep',
                        '--epochs',
                        type=int,
                        help='number of epochs',
                        required=True)

positional.add_argument('-bs',
                        '--batch-size',
                        type=int,
                        dest='batch_size',
                        help='mini batch size',
                        required=True)

positional.add_argument('-lr',
                        '--learning-rate',
                        type=float,
                        default=0.001,
                        dest='lr',
                        help='learning rate',
                        required=True)

positional.add_argument('-m',
                        '--model',
                        type=str,
                        dest='model',
                        choices=[str(x) for x in range(1, 9)],
                        help='which model to train 1, 2,.. as arguments',
                        required=True)

positional.add_argument('-op',
                        '--optimizer',
                        type=str,
                        dest='optimizer',
                        choices=['SGD', 'Adam', 'Both'],
                        help='SGD/Adam optimizer as arguments',
                        required=True)

new.set_defaults(func=start)
args = parser.parse_args()
args.func(args)

