# 🎨 ColorFlow

*Retrieval-Augmented Image Sequence Colorization*

**Authors:** Junhao Zhuang, Xuan Ju, Zhaoyang Zhang, Yong Liu, Shiyi Zhang, Chun Yuan, Ying Shan

<a href='https://zhuang2002.github.io/ColorFlow/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
<a href='https://huggingface.co/spaces/TencentARC/ColorFlow'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a> &nbsp;
<a href="https://arxiv.org/abs/2412.11815"><img src="https://img.shields.io/static/v1?label=Arxiv Preprint&message=ColorFlow&color=red&logo=arxiv"></a> &nbsp;
<a href="https://huggingface.co/TencentARC/ColorFlow"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue"></a>

**Your star means a lot for us to develop this project!** :star:

<img src='https://zhuang2002.github.io/ColorFlow/fig/teaser.png'/>

# Manga Dataset Preparation and ColorFlow Usage

This README provides a guide on how to prepare a manga dataset and use the ColorFlow tool for processing.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.x
- Git
- FFmpeg

## Step 1: Prepare the Manga Dataset

1. **Obtain Manga Images:**
   - Visit the Genshin Impact manga website to download or view manga images.
   - Example links:
     - [Genshin Impact Manga](https://genshin.hoyoverse.com/m/zh-tw/manga)
     - [Specific Manga Detail](https://genshin.hoyoverse.com/zh-tw/manga/detail/104885?mute=1)

2. **Prepare Reference Image:**
   - Ensure you have at least one image with two characters for reference. This image will be used for colorization or other processing tasks.

## Step 2: Install Required Software

1. **Update and Install Dependencies:**
   ```bash
   sudo apt-get update && sudo apt-get install cbm ffmpeg
   ```

2. **Clone the ColorFlow Repository:**
   ```bash
   git clone https://huggingface.co/spaces/svjack/ColorFlow && cd ColorFlow
   ```

3. **Install Python Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

## Step 3: Run the ColorFlow Application

1. **Start the Application:**
   ```bash
   python app.py
   ```
- Take Genshin Impact Manga for ColorFlow for example
- Input Image:
![input_1](https://github.com/user-attachments/assets/93f8c687-119f-4625-b965-2a124c45f956)

![image_2](https://github.com/user-attachments/assets/aec412b4-6f1f-4cdf-aa8e-83b1cdee961c)


- Colored Image:
![image_3](https://github.com/user-attachments/assets/da185d84-9b2a-4f78-a5d5-8d1d5190349a)

![image_4](https://github.com/user-attachments/assets/0439d0ec-254a-4879-a4df-77f9767de32e)

- API Method
```python
### 使用api 方法

from gradio_client import Client, handle_file
from PIL import Image
import pandas as pd

#client = Client("https://a9b5f5cb5cf97de36e.gradio.live/")
client = Client("http://127.0.0.1:7860")
result = client.predict(
    query_image_=handle_file('nlu.png'),
    input_style="Sketch",
    resolution="640x640",
    api_name="/extract_line_image"
)
print(result)

final_result = client.predict(
		reference_images=[handle_file("nlutree.jpg")],
		resolution="640x640",
		seed=0,
		input_style="Sketch",
		num_inference_steps=30,
		api_name="/colorize_image"
)
print(final_result)
Image.open(pd.DataFrame(final_result)["image"][0])

Image.open(pd.DataFrame(final_result)["image"][2])

result = client.predict(
    query_image_=handle_file('nlu.png'),
    input_style="GrayImage(ScreenStyle)",
    resolution="640x640",
    api_name="/extract_line_image"
)
print(result)

final_result = client.predict(
		reference_images=[handle_file("nlutree.jpg")],
		resolution="640x640",
		seed=0,
		input_style="GrayImage(ScreenStyle)",
		num_inference_steps=30,
		api_name="/colorize_image"
)
print(final_result)
Image.open(pd.DataFrame(final_result)["image"][0])

Image.open(pd.DataFrame(final_result)["image"][2])

#### sketch 会亮和鲜艳一些

#### https://github.com/TencentARC/ColorFlow/issues/3
```

### 🌟 Abstract 

Automatic black-and-white image sequence colorization while preserving character and object identity (ID) is a complex task with significant market demand, such as in cartoon or comic series colorization. Despite advancements in visual colorization using large-scale generative models like diffusion models, challenges with controllability and identity consistency persist, making current solutions unsuitable for industrial application.

To address this, we propose **ColorFlow**, a three-stage diffusion-based framework tailored for image sequence colorization in industrial applications. Unlike existing methods that require per-ID finetuning or explicit ID embedding extraction, we propose a novel robust and generalizable **Retrieval Augmented Colorization** pipeline for colorizing images with relevant color references.

Our pipeline also features a dual-branch design: one branch for color identity extraction and the other for colorization, leveraging the strengths of diffusion models. We utilize the self-attention mechanism in diffusion models for strong in-context learning and color identity matching.

To evaluate our model, we introduce **ColorFlow-Bench**, a comprehensive benchmark for reference-based colorization. Results show that ColorFlow outperforms existing models across multiple metrics, setting a new standard in sequential image colorization and potentially benefiting the art industry.

### 📰 News

- **Release Date:** 2024.12.17 - Inference code and model weights have been released! 🎉

### 📋 TODO

- ✅ Release inference code and model weights
- ⬜️ Release training code

### 🚀 Getting Started

Follow these steps to set up and run ColorFlow on your local machine:

- **Clone the Repository**
  
  Download the code from our GitHub repository:
  ```bash
  git clone https://github.com/TencentARC/ColorFlow
  cd ColorFlow
  ```

- **Set Up the Python Environment**

  Ensure you have Anaconda or Miniconda installed, then create and activate a Python environment and install required dependencies:
  ```bash
  conda create -n colorflow python=3.8.5
  conda activate colorflow
  pip install -r requirements.txt
  ```

- **Run the Application**

  You can launch the Gradio interface for PowerPaint by running the following command:
  ```bash
  python app.py
  ```

- **Access ColorFlow in Your Browser**

  Open your browser and go to `http://localhost:7860`. If you're running the app on a remote server, replace `localhost` with your server's IP address or domain name. To use a custom port, update the `server_port` parameter in the `demo.launch()` function of app.py.

### 🎉 Demo

You can [try the demo](https://huggingface.co/spaces/TencentARC/ColorFlow) of ColorFlow on Hugging Face Space.

### 🛠️ Method

The overview of ColorFlow. This figure presents the three primary components of our framework: the **Retrieval-Augmented Pipeline (RAP)**, the **In-context Colorization Pipeline (ICP)**, and the **Guided Super-Resolution Pipeline (GSRP)**. Each component is essential for maintaining the color identity of instances across black-and-white image sequences while ensuring high-quality colorization.

<img src="https://zhuang2002.github.io/ColorFlow/fig/flowchart.png" width="1000">

🤗 We welcome your feedback, questions, or collaboration opportunities. Thank you for trying ColorFlow!

### 📄 Acknowledgments

We would like to acknowledge the following open-source projects that have inspired and contributed to the development of ColorFlow:

- **ScreenStyle**: https://github.com/msxie92/ScreenStyle
- **MangaLineExtraction_PyTorch**: https://github.com/ljsabc/MangaLineExtraction_PyTorch

We are grateful for the valuable resources and insights provided by these projects.

### 📞 Contact

- **Junhao Zhuang**  
  Email: [zhuangjh23@mails.tsinghua.edu.cn](mailto:zhuangjh23@mails.tsinghua.edu.cn)

### 📜 Citation

```
@misc{zhuang2024colorflow,
title={ColorFlow: Retrieval-Augmented Image Sequence Colorization},
author={Junhao Zhuang and Xuan Ju and Zhaoyang Zhang and Yong Liu and Shiyi Zhang and Chun Yuan and Ying Shan},
year={2024},
eprint={2412.11815},
archivePrefix={arXiv},
primaryClass={cs.CV},
url={https://arxiv.org/abs/2412.11815},
}
```

### 📄 License

Please refer to our [license file](LICENSE) for more details.
