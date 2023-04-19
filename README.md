# Edit Everything: Text-Guided Generative System for Images Editing


[Defeng Xie](xiedefeng@oppo.com), [Ruichen Wang](wangruichen@oppo.com),[Jian Ma](majian2@oppo.com),[Chen Chen](chenchen4@oppo.com),[Haonan Lu](luhaonan@oppo.com),[Dong Yang](dongyang3-c@my.cityu.edu.hk),[Fobo Shi](foboshi99@gmail.com),[Xiaodong Lin](lin@business.rutgers.edu)   
**OPPO Research Institute**

> <font size=4.8> "While drawing I discover what I realy want to say" --Dario Fo </font>

We propose a text-guided generative system without any finetuning (zero-shot). We achieve a great performance for image editing by the implement of [Segment Anything Model](https://github.com/facebookresearch/segment-anything)+[CLIP](https://github.com/openai/CLIP)+[Stable Diffusion](https://github.com/Stability-AI/stablediffusion). This work firstly provides an efficient solution for Chinese scenarios. 

*The related code and **technical report** are coming soon*

![](./assets/幻灯片1.PNG)

**Approach**

The text-guided generative system consists of [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything), [CLIP](https://github.com/openai/CLIP), and [Stable Diffusion (SD)](https://github.com/Stability-AI/stablediffusion). First, we obtain all segments of an image by SAM. Second, based on the **source prompt**,  we rerank these segments and find the target segment with the highest score calculated by our CLIP. The **source prompt** is a text which describes an object of an image. Finally, we write a **target prompt** to guide our Stable Diffusion to generate a new object to replace the detected area. 

For Chinese scenarios, our CLIP is pre-trained on Chinese corpus and image pairs. And Stable Diffusion is also pre-trained on Chinese corpus. So our work first achieves the text-guided generative system in Chinese scenarios. For English scenarios, we just implement open-source [CLIP](https://github.com/openai/CLIP) and [Stable Diffusion](https://github.com/Stability-AI/stablediffusion).

*Note:  In this demo, we use [Taiyi-CLIP-Roberta-large-326M-Chinese](Taiyi-CLIP-Roberta-large-326M-Chinese) and  [Taiyi-Stable-Diffusion-1B-Chinese-v0.1](https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1). Thus real examples of this demo are slightly different from those shown in the README*.


**More Example**

One object can be edited at a time

![](./assets/幻灯片5.PNG)

![](./assets/幻灯片6.PNG)

![](./assets/幻灯片9.PNG)

![](./assets/幻灯片2.PNG)

![](./assets/幻灯片3.PNG)

![](./assets/幻灯片4.PNG)

![](./assets/幻灯片7.PNG)

![](./assets/幻灯片8.PNG)


Multiple objects of an image are edited step by step

![](./assets/多张3.png)

![](./assets/多张.png)

![](./assets/多张2.png)










**Next to do**

we will further improve zero-shot performances in our system. In the future, we will add area-guided and point-guided tools.

**Citing**
```

```
