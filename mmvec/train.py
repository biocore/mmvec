import torch


def mmvec_training_loop(microbes, metabolites, model, optimizer,
    batch_size, epochs):
    for epoch in range(epochs):

        draws = torch.multinomial(microbes,
                                  batch_size,
                                  replacement=True).T

        mmvec_model = model(draws, metabolites)

        optimizer.zero_grad()
        mmvec_model.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"loss: {mmvec_model.item()}\nBatch #: {epoch}")
