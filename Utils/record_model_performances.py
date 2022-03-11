import matplotlib.pyplot as plt


def record_pretrained_model(model, x, y):
    return record_trained_model(model, x, y)


def record_trained_model(model, x, y):
    print(model.predict(x))
    print(y)
    return


def evaluate_model(model, history, x_test, y_test):
    # Evaluate model
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    print(f"Evaluation = {model.evaluate(x_test, y_test)}")

