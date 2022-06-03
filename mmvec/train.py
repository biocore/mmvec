import torch

def mmvec_training_loop(model, learning_rate, batch_size, epochs,
        summary_interval):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 betas=(0.8, 0.9), maximize=True)
    for epoch in range(epochs):
        batch_iterations = int(model.nnz / batch_size)

        for batch in range(batch_iterations):
            #iteration = epoch*batch_iterations + batch + 1
            iteration = epoch * model.nnz // batch_size

            draws = torch.multinomial(
                        model.microbe_relative_freq(model.microbes_train),
                        batch_size,
                        replacement=True).T

            loss = model(draws, model.metabolites_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()






        if epoch % summary_interval == 0:

            with torch.no_grad():
                cv_draw = torch.multinomial(
                        model.microbe_relative_freq(model.microbes_test),
                        batch_size,
                        replacement=True).T
                cv_loss = model(cv_draw, model.metabolites_test)
                yield (str(iteration), loss.item(), cv_loss.item())
