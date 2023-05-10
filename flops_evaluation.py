import style_transfer_models
from thop import profile, clever_format

model_vae = style_transfer_models.VAE()
model_picsart = style_transfer_models.PicsartAPI()
model_transformer = style_transfer_models.Transformer()

thops1 = model_vae.get_flops()
print(thops1)

thops2 = model_transformer.get_flops()
print(thops2)
