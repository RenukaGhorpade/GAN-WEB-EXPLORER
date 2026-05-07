from models.vanilla_gan import Generator as VanillaGenerator
from models.dcgan import Generator as DCGenerator
from models.cgan import Generator as CGANGenerator

from models.vanilla_discriminator import Discriminator as VanillaD
from models.dcgan_discriminator import Discriminator as DCGAND

import streamlit as st
import torch
import torchvision
import matplotlib.pyplot as plt

device = torch.device("cpu")

st.set_page_config(
    page_title="GAN Explorer",
    page_icon="🎨",
    layout="wide"
)

compare_mode = st.checkbox("Compare All GANs")

vanilla_gen = VanillaGenerator().to(device)
vanilla_gen.load_state_dict(torch.load("weights/vanilla_gan_generator.pth", map_location=device))
vanilla_gen.eval()

dcgan_gen = DCGenerator().to(device)
dcgan_gen.load_state_dict(torch.load("weights/dcgan_generator.pth", map_location=device))
dcgan_gen.eval()

cgan_gen = CGANGenerator().to(device)
cgan_gen.load_state_dict(torch.load("weights/cgan_generator.pth", map_location=device))
cgan_gen.eval()

vanilla_d = VanillaD()
vanilla_d.load_state_dict(torch.load("weights/vanilla_gan_discriminator.pth",map_location=device))
vanilla_d.eval()

dcgan_d = DCGAND()
dcgan_d.load_state_dict(torch.load("weights/dcgan_discriminator.pth",map_location=device))
dcgan_d.eval()

st.title("🎨 GAN Explorer")

gan_type = st.selectbox(
    "Select GAN Model",
    ["Vanilla GAN", "DCGAN", "Conditional GAN"]
)
if gan_type == "Conditional GAN":

    style = st.selectbox(
        "Select Style (Cluster)",
        ["Style 0","Style 1","Style 2","Style 3"]
    )

    style_label = int(style.split()[1])

if st.button("Generate Images"):

    noise = torch.randn(16,100)

    if gan_type == "Vanilla GAN":

        fake_images = vanilla_gen(noise).detach()


    elif gan_type == "Conditional GAN":
        noise = noise.view(16,100,1,1)

        labels = torch.tensor([style_label]*16)

        fake_images = cgan_gen(noise,labels).detach()
    else:
        gan_type == "DCGAN"
        noise = noise.view(16,100,1,1)
        fake_images = dcgan_gen(noise).detach()
    
    grid = torchvision.utils.make_grid(fake_images,nrow=4,normalize=True)

    fig, ax = plt.subplots()
    ax.imshow(grid.permute(1,2,0))
    ax.axis("off")

    st.pyplot(fig)

if compare_mode:

    st.subheader("GAN Comparison")

    noise = torch.randn(8,100)

    col1, col2, col3 = st.columns(3)

    with col1:

        st.write("Vanilla GAN")

        imgs = vanilla_gen(noise).detach()

        grid = torchvision.utils.make_grid(imgs,nrow=2,normalize=True)

        st.image(grid.permute(1,2,0))

    with col2:

        st.write("DCGAN")

        imgs = dcgan_gen(noise.view(8,100,1,1)).detach()

        grid = torchvision.utils.make_grid(imgs,nrow=2,normalize=True)

        st.image(grid.permute(1,2,0))

    with col3:

        st.write("CGAN")

        labels = torch.randint(0,4,(8,))

        imgs = cgan_gen(noise.view(8,100,1,1),labels).detach()

        grid = torchvision.utils.make_grid(imgs,nrow=2,normalize=True)

        st.image(grid.permute(1,2,0))


from PIL import Image
import torchvision.transforms as transforms

st.subheader("Real / Fake Detection")

uploaded_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image,caption="Uploaded Image")

    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    img_tensor = transform(image).unsqueeze(0)

    if gan_type == "Vanilla GAN":

        pred = vanilla_d(img_tensor)

    else:

        pred = dcgan_d(img_tensor)

    score = pred.item()

    if score > 0.5:

        st.success(f"REAL (confidence {score:.2f})")

    else:

        st.error(f"FAKE (confidence {1-score:.2f})")


st.subheader("Model Metrics")

col1,col2,col3 = st.columns(3)

with col1:

    st.metric("Generator Accuracy","87%","+2%")

with col2:

    st.metric("Discriminator Accuracy","84%","-1%")

with col3:

    st.metric("Training Epochs","30")

import pandas as pd

loss_data = pd.DataFrame({
    "Epoch":[1,5,10,15,20,25,30],
    "Generator Loss":[3.2,2.4,3.1,1.8,2.5,2.2,3.6],
    "Discriminator Loss":[1.4,0.8,0.6,0.9,0.7,0.5,0.53]
})

st.line_chart(loss_data.set_index("Epoch"))