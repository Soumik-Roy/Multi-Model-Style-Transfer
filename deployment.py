import streamlit as st
import pandas as pd
import numpy as np
from  PIL import Image, ImageEnhance
from streamlit import config
import style_transfer_models
import time
import utils as utils

st.set_page_config(page_title="Style Transfer - DLOps", page_icon="🎭")

st.markdown(
            f'''
            <style>
                .css-1544g2n {{
                    padding-top: {3}rem;
                }}

                .css-1y4p8pa {{
                    padding-top: {2}rem;
                }}
            </style>
            ''',
            unsafe_allow_html=True,
)

model_vae = style_transfer_models.VAE()
model_picsart = style_transfer_models.PicsartAPI()
# model_transformer = style_transfer_models.Transformer()

st.sidebar.title("Configurations")
st.title("Style Transfer using Different Architectures")



###### SIDEBAR  #######

st.sidebar.header("Select Model Architecture")

model_options = {
    'Variational AE': model_vae,
    'Pics-Art API' : model_picsart,
    'Transformer' : None,
    'Compare All'.upper() : "all",
}

selected_model = st.sidebar.radio("Models",tuple(model_options.keys()))

with st.sidebar.expander("About the App"):
     st.write("""
        Use this simple app to convert your normal Images into different styles.\nUpload an Image that you want to transform, and another Image which is the Style reference. What you get as a result is the original image stylized according to the reference style Image.\n\nThis app was created by Soumik, Yash and Stuti as a part of the DLOps project for the course CSL4020: Deep Learning offered at IIT Jodhpur during Jan-May 2023.
     """)




####### MAIN PAGE  ########

st.subheader('Upload your Image and Style Reference')
content_file = st.file_uploader("Image to Style", type=['jpg','png','jpeg'])
style_file = st.file_uploader("Style Reference Image", type=['jpg','png','jpeg'])

cont_img = None
style_img = None

if content_file is not None or style_file is not None:
    
    col1, col2 = st.columns( [0.5, 0.5])
    with col1:
        st.markdown('<p style="text-align: center;">Original</p>',unsafe_allow_html=True)
        if content_file is not None:
            cont_img = Image.open(content_file)
            st.image(cont_img,width=300)  

    with col2:
        st.markdown('<p style="text-align: center;">Style Reference</p>',unsafe_allow_html=True)
        if style_file is not None:
            style_img = Image.open(style_file)
            st.image(style_img,width=300)

###### STYLE TRANSFER BUTTON #######

transformed_img = None
transformation_time = None

if st.button("Transform", type='primary'):
    if(cont_img is None or style_img is None):
        st.error("Please upload both the images")

    else:
        model = model_options[selected_model]
        if model=='all':
            with st.spinner("Styling✨ image with all models (can take a few minutes)..."):
                cont_json = utils.upload_img(cont_img)
                style_json = utils.upload_img(style_img)
                cont_url = utils.get_url(cont_json)
                style_url = utils.get_url(style_json)

                imgs = []
                times = []

                model_names = ['Variational AE', 'Pics-Art API', 'Transformer']

                for i in range(len(model_names)):
                    # print(model_names[i])
                    _model = model_options[model_names[i]]
                    if _model==None or _model=='all':
                        continue
                    t0 = time.time()
                    trf_img = _model.transform_image(cont_img, style_img, cont_url, style_url)
                    t1 = time.time()
                    imgs.append({'name':model_names[i],'img':trf_img})
                    times.append(t1-t0)

                    if(len(imgs)==2):
                        transformed_img = imgs
                        transformation_time = times
                transformed_img = imgs
                transformation_time = times

                utils.delete_img(cont_json)
                utils.delete_img(style_json)

        elif model!=None:
            with st.spinner("Adding style✨ to your image..."):
                cont_url = ""
                style_url = ""
                if(selected_model[-3:]=='API'):
                    cont_json = utils.upload_img(cont_img)
                    style_json = utils.upload_img(style_img)
                    cont_url = utils.get_url(cont_json)
                    style_url = utils.get_url(style_json)

                t0 = time.time()
                transformed_img = model.transform_image(cont_img, style_img, cont_url, style_url)
                t1 = time.time()

                transformation_time = t1-t0
                if(selected_model[-3:]=='API'):
                    utils.delete_img(cont_json)
                    utils.delete_img(style_json)

        else:
            st.error("Model not available yet, we're trying our best to get it running..")
        transforming = False
        # picklefile.close()

###### TRANSFORMED IMAGES #######

if type(transformed_img)==list:
        col1, col2 = st.columns( [0.5, 0.5])
        with col1:
            if len(transformed_img)>0:
                st.markdown(f'<p style="text-align: center;">{transformed_img[0]["name"]}</p>',unsafe_allow_html=True)
                if content_file is not None:
                    st.image(transformed_img[0]['img'],width=300, caption=f"generated in {transformation_time[0]:.1f} seconds")

        with col2:
            if len(transformed_img)>1:
                st.markdown(f'<p style="text-align: center;">{transformed_img[1]["name"]}</p>',unsafe_allow_html=True)
                if content_file is not None:
                    st.image(transformed_img[1]['img'],width=300, caption=f"generated in {transformation_time[1]:.1f} seconds")
        
        col3, col4 = st.columns( [0.5, 0.5])
        with col3:
            if len(transformed_img)>2:
                st.markdown(f'<p style="text-align: center;">{transformed_img[2]["name"]}</p>',unsafe_allow_html=True)
                if content_file is not None:
                    st.image(transformed_img[2]['img'],width=300, caption=f"generated in {transformation_time[2]:.1f} seconds")
elif transformed_img is not None:
    st.subheader('Transformed Image')
    st.image(transformed_img,width=300, caption=f"generated in {transformation_time:.1f} seconds")
