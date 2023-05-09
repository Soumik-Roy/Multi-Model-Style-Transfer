import requests
from PIL import Image
import base64
from io import BytesIO

def upload_img(img):
    url = "https://api.imgur.com/3/image"
    
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())


    payload={'image': img_str}
    files=[]
    headers = {
        'Authorization': 'Client-ID 52d1141cb13f609'
    }

    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    return response.json()

def delete_img(upload_response):
    url = f"https://api.imgur.com/3/image/{upload_response['data']['deletehash']}"

    payload={}
    files=[]
    headers = {
        'Authorization': 'Client-ID 52d1141cb13f609'
    }

    response = requests.request("DELETE", url, headers=headers, data=payload, files=files)
    return response.json()

def get_url(upload_response):
    # print(upload_response)
    url = upload_response['data']['link']
    url.replace("\\", '')
    return url