import os

from InfoGAN.train import train, Params




name = "experiment_10_latents"
os.makedirs(name, exist_ok=True)
parameters = Params(name, batch_size=5, code_dim=2, latent_dim=10)
with open(name+"/params.txt", "w") as text_file:
    text_file.write(str(parameters._asdict()))
train(parameters)

name = "experiment_50_latents"
os.makedirs(name, exist_ok=True)
parameters = Params(name, batch_size=5, code_dim=2, latent_dim=50)
with open(name+"/params.txt", "w") as text_file:
    text_file.write(str(parameters._asdict()))
train(parameters)


name = "experiment_5_latents"
os.makedirs(name, exist_ok=True)
parameters = Params(name, batch_size=5, code_dim=2, latent_dim=5)
with open(name+"/params.txt", "w") as text_file:
    text_file.write(str(parameters._asdict()))
train(parameters)


