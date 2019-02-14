# attention-based-neural-machine-translation
利用RNN实现基于注意力的神经机器翻译


采用编码器和解码器结构。

解码器端加上注意力机制，注意力机制让网络在解码的时候能够“集中注意力”在编码输出的某些部分上，而不仅仅依赖于简单的内容向量。

输出翻译结果：

![](https://raw.githubusercontent.com/zqs01/pic/master/11.png)