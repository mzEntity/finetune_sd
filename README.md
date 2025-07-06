# 微调stable-diffusion

使用图片数据集微调stable-diffusion，查看效果

## 使用的数据集

UTKFace
[UTKFace](https://susanqq.github.io/UTKFace/)

## 步骤

**运行dataset_prepare.py，将下载好的本地h5数据集展开，并上传至huggingface**

需要获得huggingface的登陆token，如果出现网络问题，请设置代理
```python
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
```

存在现成已经上传的数据集：[mzvltr/UTKFace_192](https://huggingface.co/datasets/mzvltr/UTKFace_192)

**运行train_text_to_image.py，微调stable-diffusion-v1-5**

请参考更具体的教程[huggingface微调sd教程](https://huggingface.co/docs/diffusers/training/text2image)，并按照它的步骤进行
这里仅是将核心代码拷贝过来，仅供参考

**运行sample.py，使用微调的sd为年龄为1-60生成图片，每个标签1000张**

在有4个GPU的机器上运行sample.sh，可批量生成

**运行sample_other.py，生成60岁以上的图片，验证泛化性**

