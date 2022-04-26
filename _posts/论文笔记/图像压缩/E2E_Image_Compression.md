
# End-to-end optimized image compression

Balle  google，深度学习图像压缩开山之作，入门必读

[代码地址](https://github.com/tensorflow/compression)

## 简介

整体算法分为三个部分：非线性分析变换（编码器），均匀量化器和非线性合成边变换（解码器），在整个训练框架上联合优化整个模型的率失真性能。在某些条件下，松弛损失函数可以解释为由变分自动编码器实现的生成模型的对数似然，即和变分自编码器的损失函数某种情况下有点像。另外，每一个折衷参数对应一个压缩模型，所以整个 R-D 曲线是由模型构成的 R-D 点形成的。

非线性分析变换，均匀量化器和非线性合成变换。变换是在卷积线性滤波器和非线性激活函数的三个连续阶段中构建的。与大多数卷积神经网络不同，选择联合非线性来实现局部增益控制的形式，其灵感来自用于模拟生物神经元的那些。使用随机梯度下降的变体，我们在训练图像数据库上联合优化整个模型的率失真性能，引入由量化器产生的不连续损失函数的连续代理。在某些条件下，松弛损失函数可以被解释为生成模型的对数似然性，如变分自动编码器所实现的。然而，与这些模型不同，压缩模型必须在速率 - 失真曲线的任何给定点处操作，如权衡参数所指定的那样。在一组独立的测试图像中，我们发现优化的方法通常表现出比标准JPEG和JPEG 2000压缩方法更好的速率 - 失真性能。更重要的是，我们观察到所有图像在所有比特率下的视觉质量都有显着提高，这得到了使用MS-SSIM的客观质量估算的支持。


内容
==

整体算法结构以及流程如下：
-------------

![](https://img-blog.csdnimg.cn/20201205140728571.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyMjgxNDI1,size_16,color_FFFFFF,t_70)

![](https://img-blog.csdnimg.cn/202012042010161.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyMjgxNDI1,size_16,color_FFFFFF,t_70)  
x x x 与 x ^ \hat{x} x^ 分别代表输入的原图和经过编解码器后的重建图片。 g a g_a ga​表示[编码器](https://so.csdn.net/so/search?q=%E7%BC%96%E7%A0%81%E5%99%A8&spm=1001.2101.3001.7020)提供的非线性分析变换， y y y 即由输入图片经过编码器网络后得到的潜在特征，通过量化器 q q q 后，得到 量化后结果： y ^ \hat{y} y^​，再通过 g s g_s gs​解码器重建图片结果.

其中，通过对 y y y 的码率估计得到 R R R, 计算原图 x x x 和 x ^ \hat{x} x^ 的失真得到 D D D， g p g_p gp​是指一种失真变换，可以将原图和重建图进行通过 g p g_p gp​ 转化到感知空间上进行计算，，直观的作用可以理解为失真计算： D = d ( x , x ^ ) D=d(x,\hat{x}) D=d(x,x^), 例如 PSNR,MS-SSIM，或者其他感知域如 VMAF 等。通过得到的 R R R 和 D D D 进行率失真联合优化，定义[损失函数](https://so.csdn.net/so/search?q=%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0&spm=1001.2101.3001.7020)为： L = λ ⋅ D + R L=\lambda\cdot D+R L=λ⋅D+R 通过 λ \lambda λ 参数进行码率的选择控制， λ \lambda λ 大则训练得到的模型的重建图失真，而压缩后的码流大，反之亦然。

**激活层改进**：受到生物神经元的启发，在编码器与解码器中，论文使用了 **GDN**（广义除数归一化），该激活层在论文中《Density modeling of image using a generalized normalization transformation》有着详细的描述，该激活层实践中比 BN 层更适用于图像重建工作。其中与 BN 层的差别：BN 层重新调整网络中线性滤波器的响应，以便将其保持在合理的操作范围内。这种类型的归一化不同于局部增益控制，因为在所有空间位置上重新缩放因子是相同的。此外，一旦训练完成，缩放参数通常是固定的，这将归一化转换为数据的仿射变换，不同于 GDN 那样，GDN 是空间自适应的，并且可能是高度非线性的。

**量化问题**：传统的图像编码中，量化有着不可微分的问题，为分段函数，且在非边界点的导数为 0，截断反向传播，在边界点则不存在导数。作者采用基于_概率模型连续松弛_的代理损失函数（没理解什么意思），用加性均匀噪声代替量化步骤（就是加了 - 0.5 到 0.5 区间的均匀噪声）。

前向传播
----

在非线性分析变换（编码器）中 g a g_a ga​ 包含有三个阶段：卷积，降采样（就是用的卷积层进行的降采样工作），GDN 归一化。以下公式预警：

**卷积**： v i k （ m , n ） = ∑ j ( h k , i j ∗ u j k ) ( m , n ) + c k , i v_{i}^{k}（m,n）=\sum\limits_{j}(h_{k,i j}*u_j^{k})(m,n)+c_{k,i} vik​（m,n）=j∑​(hk,ij​∗ujk​)(m,n)+ck,i​ v i k （ m , n ) v_{i}^{k}（m,n) vik​（m,n) 是 i i i 个输出的通道在第 k k k 层的网络的输入，总共有 3 层，所以 k 最大为 3，其中（ m m m, n n n）代表长宽的二维的位置。 ∗ * ∗是卷积操作，上述公式描述输入在  
经过卷积层后的输出。

**采样** w i k ( m , n ) = v i k （ s k m , s k n ） w_i^{k}(m,n)=v_{i}^{k}（s_km,s_kn） wik​(m,n)=vik​（sk​m,sk​n） 上述公式， s k s_k sk​是采样因子，即几倍采样。一般采样与卷积运算在代码中统一实现。

**GDN** u i k + 1 ( m , n ) = w i k ( m , n ) ( β k , i + ∑ j r k , i j ( w j k ( m , n ) ) 2 ) u_i^{k+1}(m,n)= \dfrac{w_i^{k}(m,n)}{\sqrt{(\beta_{k,i}+\sum\limits_jr_{k,ij}(w_j^{k}(m,n))^2)}} uik+1​(m,n)=(βk,i​+j∑​rk,ij​(wjk​(m,n))2) ​wik​(m,n)​所有的待优化参数 h h h（卷积核权重）, c c c（卷积核偏置）, r r r（GDN 的归一化系数）都在整个端到端中被优化。

后续在解码端中的公式不例举辽，这部分内容说白了就是把神经网络作用给抽象公式化。

变换优化问题
------

     在传统的图像编码中，对于 DCT 生成的系数需要进行量化，通常采用了矢量量化的形式并且结合熵编码进行率失真控制。在该论文并没有直接在编码空间进行最优量化（矢量）, 而是通过一个固定的标量量化器进行量化（四舍五入），试图让分线性变换（编码器）进行自动学习，以学习到的方式扭曲量化空间，从而有效实现矢量量化。整体[框架](https://so.csdn.net/so/search?q=%E6%A1%86%E6%9E%B6&spm=1001.2101.3001.7020)通过以下公式进行优化： L = − E [ l o g 2 P q ] + λ ⋅ E [ d ( z , z ^ ) ] L=-E[log_2P_q]+\lambda\cdot E[d(z,\hat{z})] L=−E[log2​Pq​]+λ⋅E[d(z,z^)] 通过计算熵与失真进行率失真优化，其中两个期望值通过训练集的平均值进行拟合。并且通过非线性分析学习量化特征即矢量量化结构，则本文采用标准量化： y i ^ = q i = r o u n d ( y i ) \hat{y_i}=q_i=round(y_i) yi​^​=qi​=round(yi​) i i i 覆盖了所有的，待编码的数值，并且 y i ^ \hat{y_i} yi​^​ 的边缘密度一系列离散概率密度质量给出: P q i (n) = ∫ n − 1 / 2 n + 1 / 2 p y i ( t ) d t P_{q_i}(n)=\int_{n-1/2}^{n+1/2}p_{y_i}(t)dt Pqi​​(n)=∫n−1/2n+1/2​pyi​​(t)dt 即 p y i (t) p_{y_i}(t) pyi​​(t) 是 y y y 的概率密度函数，由于四舍五入，在某一整数的（-0.5，0.5）区间内都会量化为该整数，则通过积分的形式计算这一区间内的数值的出现概率，得到量化后的整数的出现概率。

     上述公式均涉及到量化问题，但是量化会导致不可微分，阻断反向传播优化的问题。论文中采用了添加（-0.5，0.5）范围的均匀噪声，在训练过程中，即采用这种形式近似可微以用于反向传播的优化，在推理过程中，则依旧使用 r o u n d round round 函数进行四舍五入（因为不用进行优化了），使用均匀噪声有两个优点：

首先， y ~ = y + n o i s e \tilde{y} = y+noise y~​=y+noise 的密度函数是 q q q 的概率质量函数的连续松弛

![](https://img-blog.csdnimg.cn/20201205124820457.png#pic_center)

y i y_i yi​是编码空间的元素，就是需要被编码的值， y i ^ \hat{y_i} yi​^​ 是四舍五入后的值， y i ~ \tilde{y_i} yi​~​ 是通过添加噪声后的值，上述是三者的概率密度函数（PDF），离散的 y i ^ \hat{y_i} yi​^​是概率质量函数（PMF），其中黑点的数值由实线在各自区间内取积分得到的，但是其实不一定等于虚线在区间内的积分，只能是近似关系（论文中说相等，我不信 =-=）。即可以通过这种近似关系，合理等于两者的微分熵，换个说法就是加均匀噪声后，不影响码率的估计。

其次，独立的均匀噪声就其边际矩（我也不知道什么是边际矩）而言近似于量化误差，并且经常被用作量化误差的模型（Gray and Neuhoff，1998）。

针对 y 的宽松概率模型和熵代码在代码空间中假设独立的边际，则通过进行参数建模以减少码率估计模型与实际模型之间的差距，具体使用精细采样的分段线性函数，这些函数与一维直方图的更新类似（请参见附录）即通过神经网络生成编码点的 PMF，通过 PMF 的积分求得每个特征点对应的可导形式的概率值，通过对概率值求自信息得到估计的码率点，在前向传播中，通过训练阶段生成的神经网络生成分位点与 CDF，将 CDF 与待编码点输出 range coder 进行熵编码。  
以用于熵编码。 由于编码数值的概率密度函数被均匀噪声平滑，这部分会导致一定的误差，但是通过减小采样间隔可以使得模型误差任意小。

整个过程可以公式得到： L ( θ , ϕ ) = E x , △ y [ − ∑ i l o g 2 p y i ˉ ( g a ( x ; ϕ ) + △ y ) ; ψ (i) + λ d ( g p ( g s ( g a ( x ) ; ϕ ) ) , g p ( x ) ) ] L(\theta,\phi)=E_{x,\triangle y}[-\sum_ilog_2p_{\bar{y_i}}(g_a(x;\phi)+\triangle y);\psi_{(i)}+\lambda d(g_p(g_s(g_a(x); \phi)),g_p(x))] L(θ,ϕ)=Ex,△y​[−i∑​log2​pyi​ˉ​​(ga​(x;ϕ)+△y);ψ(i)​+λd(gp​(gs​(ga​(x);ϕ)),gp​(x))] 量化以及 ψ \psi ψ的分段线性逼近很适合随机优化（我看不出 =-=！）

变分推导问题
------

再论，等下一篇超先验网络结构模型进行推导吧。

实验结果
====

![](https://img-blog.csdnimg.cn/2020120513551030.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyMjgxNDI1,size_16,color_FFFFFF,t_70)  
左图为量化所引起的误差和均匀噪声引起的误差的关系，右图为离散情况下的熵率以及可微分情况下的熵率的关系，可以得出，本文提出的量化模型以及熵率估计模型与实际中量化引起的误差和待编码值得熵率近似一致。

![](https://img-blog.csdnimg.cn/20201205135848579.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyMjgxNDI1,size_16,color_FFFFFF,t_70)  
以及在实际中，得到得 R-D 性能，PSNR,MS-SSIM 性能指标熵超过了 JPEG2000 压缩标准。其中在 PSNR 度量标准上是通过 MSE 为失真指标进行训练的，而 MS-SSIM 是在 MS-SSIM 指标上进行训练到。并不是一种失真模型对应两个失真指标。即 MSE 训练了六个 lambda 的模型，MS-SSIM 训练了六个 lambda 的模型。  
![](https://img-blog.csdnimg.cn/20201205144744618.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyMjgxNDI1,size_16,color_FFFFFF,t_70)  
![](https://img-blog.csdnimg.cn/20201205144758699.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyMjgxNDI1,size_16,color_FFFFFF,t_70)  
![](https://img-blog.csdnimg.cn/20201205144809445.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyMjgxNDI1,size_16,color_FFFFFF,t_70)  
通过主观感受可以看出，本文提出的算法优于 JPEG 以及 JPEG2000。

结论
==

该论文还是着重与量化和熵率估计的问题，论述了添加均匀噪声的量化的合理性以及对于熵率估计的影响，包括提出了如何使得熵率估计模型可微分以及 CDF 的学习以用于熵编码的情况。并且确定了在[图像压缩](https://so.csdn.net/so/search?q=%E5%9B%BE%E5%83%8F%E5%8E%8B%E7%BC%A9&spm=1001.2101.3001.7020)框架中，标量量化可以通过编码器的非线性分析达到矢量量化的效果（期望这种非线性分析可以学习得到）。