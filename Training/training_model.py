def train(model, x_train, y_train):
    model.fit(x_train, y_train)


def pretrain(model, x, y):
    train(model, x, y)
