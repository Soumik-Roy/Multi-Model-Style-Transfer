import torch
import torchvision.transforms as transforms
from  PIL import Image
import numpy as np
import requests
from Models.ST_VAE.libs.models import encoder4
from Models.ST_VAE.libs.models import decoder4
from Models.ST_VAE.libs.Matrix import MulLayer
import torch.nn as nn
import Models.StyTR2.models.transformer as transformer
import Models.StyTR2.models.StyTR as StyTR

class VAE():
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
            transforms.Resize(512),
            transforms.Lambda(lambda x: x[:3])
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(self.device)


        vgg = encoder4()
        dec = decoder4()
        matrix = MulLayer(z_dim=256)
        vgg.load_state_dict(torch.load(   "Models/ST_VAE/models/vgg_r41.pth", map_location=torch.device(self.device)))
        dec.load_state_dict(torch.load(   "Models/ST_VAE/models/dec_r41.pth", map_location=torch.device(self.device)))
        matrix.load_state_dict(torch.load("Models/ST_VAE/models/matrix_r41_new.pth", map_location=torch.device(self.device)))

        vgg.to(self.device)
        dec.to(self.device)
        matrix.to(self.device)
        matrix.eval()
        vgg.eval()
        dec.eval()
        self.vgg = vgg
        self.dec = dec
        self.matrix = matrix

    def transform_image(self, content, ref, *args):

        content = self.transform(content).unsqueeze(0).to(self.device)
        ref = self.transform(ref).unsqueeze(0).to(self.device)
        # print(content.shape, ref.shape)

        with torch.no_grad():
            sF = self.vgg(ref)
            cF = self.vgg(content)
            feature, _, _ = self.matrix(cF['r41'], sF['r41'])
            prediction = self.dec(feature)

            prediction = prediction.data[0].cpu().permute(1, 2, 0)

        # t1 = time.time()
        #print("===> Processing: %s || Timer: %.4f sec." % (str(i), (t1 - t0)))

        prediction = prediction * 255.0
        prediction = prediction.clamp(0, 255)


        transformed_img = Image.fromarray(np.uint8(prediction))
        return transformed_img

class PicsartAPI():
    def __init__(self):
        self.url = "https://api.picsart.io/tools/1.0/styletransfer"
        self.files=[]
        self.headers = {"accept": "application/json", "X-Picsart-API-Key": "ImRHEZp0gqjD7mi6ZRykD1CVToKjnZPc"}
    
    def transform_image(self, content, style, content_url, style_url):
        payload={
            "reference_image_url": style_url, 
            "image_url": content_url,
            # "format": "JPG", 
            # "output_type": "cutout",
        }

        response = requests.request("POST", self.url, headers=self.headers, data=payload, files=self.files)
        # print(response.json())

        transformed_img = None
        if(response.status_code==200):
            transformed_img = Image.open(requests.get(response.json()['data']['url'], stream=True).raw)
        return transformed_img

class Transformer():
    def __init__(self):
        content_size=512
        style_size=512
        crop='store_true'

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        vgg = StyTR.vgg
        vgg.load_state_dict(torch.load("Models/StyTR2/experiments/vgg_normalised.pth"))
        vgg = nn.Sequential(*list(vgg.children())[:44])

        decoder = StyTR.decoder
        Trans = transformer.Transformer()
        embedding = StyTR.PatchEmbed()

        decoder.eval()
        Trans.eval()
        vgg.eval()
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        state_dict = torch.load("Models/StyTR2/experiments/decoder_iter_160000.pth")
        for k, v in state_dict.items():
            #namekey = k[7:] # remove `module.`
            namekey = k
            new_state_dict[namekey] = v
        decoder.load_state_dict(new_state_dict)

        new_state_dict = OrderedDict()
        state_dict = torch.load("Models/StyTR2/experiments/transformer_iter_160000.pth")
        for k, v in state_dict.items():
            #namekey = k[7:] # remove `module.`
            namekey = k
            new_state_dict[namekey] = v
        Trans.load_state_dict(new_state_dict)

        new_state_dict = OrderedDict()
        state_dict = torch.load("Models/StyTR2/experiments/embedding_iter_160000.pth")
        for k, v in state_dict.items():
            #namekey = k[7:] # remove `module.`
            namekey = k
            new_state_dict[namekey] = v
        embedding.load_state_dict(new_state_dict)

        network = StyTR.StyTrans(vgg,decoder,embedding,Trans)
        network.eval()
        network.to(device)

        self.device = device
        self.network = network

        self.content_tf = self.test_transform(content_size, crop)
        self.style_tf = self.test_transform(style_size, crop)

    def test_transform(self, size, crop):
        transform_list = []
    
        if size != 0: 
            transform_list.append(transforms.Resize(size))
        if crop:
            transform_list.append(transforms.CenterCrop(size))
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)
        return transform
    def style_transform(self, h,w):
        k = (h,w)
        size = int(np.max(k))
        print(type(size))
        transform_list = []    
        transform_list.append(transforms.CenterCrop((h,w)))
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)
        return transform

    def content_transform(self):
        
        transform_list = []   
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)
        return transform
    
    def transform_image(self, content_img, style_img, *args):
        content_tf1 = self.content_transform()       
        content = self.content_tf(content_img)

        h,w,c=np.shape(content)    
        style_tf1 = self.style_transform(h,w)
        style = self.style_tf(style_img)


        style = style.to(self.device).unsqueeze(0)
        content = content.to(self.device).unsqueeze(0)

        with torch.no_grad():
            output= self.network(content,style)       
        output = output[0].cpu()
        transformed_img = transforms.ToPILImage()(output[0])

        return transformed_img
