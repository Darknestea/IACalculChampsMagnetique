def record_pretrained_model(model, x, y):
    return record_trained_model(model, x, y)


def record_trained_model(model, x, y):
    print(model.predict(x))
    print(y)
    return
