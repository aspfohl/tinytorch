from mnist import MNIST

import tinytorch

mndata = MNIST(".data/mnist")
images, labels = mndata.load_training()


BACKEND = tinytorch.make_tensor_backend(tinytorch.FastOps)
BATCH = 16

# Number of classes (10 digits)
C = 10

# Size of images (height and width)
H, W = 28, 28


def RParam(*shape):
    r = 0.1 * (tinytorch.rand(shape, backend=BACKEND) - 0.5)
    return tinytorch.Parameter(r)


class Linear(tinytorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        batch, in_size = x.shape
        return (
            x.view(batch, in_size) @ self.weights.value.view(in_size, self.out_size)
        ).view(batch, self.out_size) + self.bias.value


class Conv2d(tinytorch.Module):
    def __init__(self, in_channels, out_channels, kh, kw):
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kh, kw)
        self.bias = RParam(out_channels, 1, 1)

    def forward(self, input):
        return tinytorch.conv2d(input, self.weights.value) + self.bias.value


class Network(tinytorch.Module):
    """
    Implement a CNN for MNist classification based on LeNet.

    This model should implement the following procedure:

    1. Apply a convolution with 4 output channels and a 3x3 kernel followed by a ReLU (save to self.mid)
    2. Apply a convolution with 8 output channels and a 3x3 kernel followed by a ReLU (save to self.out)
    3. Apply 2D pooling (either Avg or Max) with 4x4 kernel.
    4. Flatten channels, height, and width. (Should be size BATCHx392)
    5. Apply a Linear to size 64 followed by a ReLU and Dropout with rate 25%
    6. Apply a Linear to size C (number of classes).
    7. Apply a logsoftmax over the class dimension.
    """

    def __init__(self):
        super().__init__()

        # For vis
        self.mid = None
        self.out = None

        # layers
        self.conv1 = Conv2d(1, 4, 3, 3)
        self.conv2 = Conv2d(4, 8, 3, 3)
        self.linear1 = Linear(392, 64)
        self.linear2 = Linear(64, C)

    def forward(self, x):
        # Apply a convolution with 4 output channels and a 3x3 kernel followed by a ReLU
        self.mid = self.conv1.forward(x).relu()

        # Apply a convolution with 8 output channels and a 3x3 kernel followed by a ReLU
        self.out = self.conv2.forward(self.mid).relu()

        # Apply 2D pooling (either Avg or Max) with 4x4 kernel
        pooled = tinytorch.nn.avgpool2d(self.out, (4, 4))

        # Flatten channels, height, and width. (Should be size BATCHx392)
        flattened = pooled.view(BATCH, 392)

        # Apply a Linear to size 64 followed by a ReLU and Dropout with rate 25
        lineared = self.linear1.forward(flattened).relu()
        if self.training:
            lineared = tinytorch.nn.dropout(lineared, 0.25)

        # Apply a Linear to size C (number of classes)
        lineared2 = self.linear2.forward(lineared)

        # Apply a logsoftmax over the class dimension
        return tinytorch.nn.logsoftmax(lineared2, 1)


def make_mnist(start, stop):
    ys = []
    X = []
    for i in range(start, stop):
        y = labels[i]
        vals = [0.0] * 10
        vals[y] = 1.0
        ys.append(vals)
        X.append([[images[i][h * W + w] for w in range(W)] for h in range(H)])
    return X, ys


def default_log_fn(epoch, total_loss, correct, losses, model):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class ImageTrain:
    def __init__(self):
        self.model = Network()

    def run_one(self, x):
        return self.model.forward(tinytorch.tensor([x], backend=BACKEND))

    def train(
        self, data_train, data_val, learning_rate, max_epochs=500, log_fn=default_log_fn
    ):
        (X_train, y_train) = data_train
        (X_val, y_val) = data_val
        self.model = Network()
        model = self.model
        n_training_samples = len(X_train)
        optim = tinytorch.optim.SGD(self.model.parameters(), learning_rate)
        losses = []
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0

            model.train()
            for batch_num, example_num in enumerate(
                range(0, n_training_samples, BATCH)
            ):

                if n_training_samples - example_num <= BATCH:
                    continue
                y = tinytorch.tensor(
                    y_train[example_num : example_num + BATCH], backend=BACKEND
                )
                x = tinytorch.tensor(
                    X_train[example_num : example_num + BATCH], backend=BACKEND
                )
                x.requires_grad_(True)
                y.requires_grad_(True)
                # Forward
                out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)
                prob = (out * y).sum(1)
                loss = -(prob / y.shape[0]).sum()

                assert loss.backend == BACKEND
                loss.view(1).backward()

                total_loss += loss[0]

                # Update
                optim.step()

            losses.append(total_loss)

            model.eval()

            correct = 0
            for val_example_num in range(0, 1 * BATCH, BATCH):
                y = tinytorch.tensor(
                    y_val[val_example_num : val_example_num + BATCH], backend=BACKEND,
                )
                x = tinytorch.tensor(
                    X_val[val_example_num : val_example_num + BATCH], backend=BACKEND,
                )
                out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)
                for i in range(BATCH):
                    m = -1000
                    ind = -1
                    for j in range(C):
                        if out[i, j] > m:
                            ind = j
                            m = out[i, j]
                    if y[i, ind] == 1.0:
                        correct += 1
            log_fn(epoch, total_loss, correct, losses, model)

            total_loss = 0.0
            model.train()


if __name__ == "__main__":
    data_train, data_val = (make_mnist(0, 5000), make_mnist(10000, 10500))
    ImageTrain().train(data_train, data_val, learning_rate=0.01)
