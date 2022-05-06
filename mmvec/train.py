import torch

def mmvec_training_loop(model, learning_rate, batch_size, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 betas=(0.8, 0.9), maximize=True)
    for epoch in range(epochs):
        batch_iterations = int(model.nnz / batch_size)

        for batch in range(batch_iterations):

            draws = torch.multinomial(model.microbe_relative_freq,
                                      batch_size,
                                      replacement=True).T

            mmvec_model = model(draws)

            optimizer.zero_grad()
            mmvec_model.backward()
            optimizer.step()

        if epoch % 500 == 0:
            print(f"loss: {mmvec_model.item()}\nBatch #: {epoch}")
