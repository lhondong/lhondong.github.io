# Deep Learning in Latent Space for Video Prediction and Compression

先做图像压缩，然后使用前一帧的latent representation预测下一帧的latent representation。然后做 element-wise的残差，如果预测的很好的话，残差是稀疏的，熵很低。

$$
p_{\theta}\left(x,z\right)=p_{\theta}\left(z\right) \cdot p_{\theta}\left(x \mid z\right)
$$

$$
p_{\theta}\left(x_{1: T}, z_{1: T}\right)=\prod_{t=1}^{T} p_{\theta}\left(z_{t} \mid z_{<t}\right) \cdot p_{\theta}\left(x_{t} \mid z_{t}\right)
$$

$$
p_{\theta}\left(z_{T} \mid z_{<T}\right)=\prod_{t=2}^{T} \frac{p_{\theta}\left(z_{1: t}\right)}{p_{\theta}\left(z_{1: t-1}\right)}
$$

$$
\begin{aligned}
\mathcal{L}=\lambda \cdot &\left\{\mathbb{E}_{z \sim p_{z}}\left[\log \left(1-D\left(G\left(P\left(z_{<t} \mid x_{<t}\right)\right)\right)\right)\right]\right.\\
&\left.+\mathbb{E}_{z \sim p_{o p t}}\left[\log \left(D\left(G\left(z_{o p t} \mid x_{<t}\right)\right)\right)\right]\right\} \\
&+(1-\lambda) \cdot \mathbb{E}_{p(z<t)}\left[\log p\left(z_{t} \mid z_{<t}\right)\right]
\end{aligned}
$$

$$
\min_r f(r) \quad \text { subject to } r \in S
$$

$$
f(r)=d_{l}\left(z_{o p t},(\hat{z}+r)\right)+d_{s}(x, G(\hat{z}+r)) \quad \text { subject to } r \in S
$$

$$
\min _{r} f(r)+g(u) \quad \text { subject to } r=u
$$

$$
g(u)=\left\{\begin{array}{cc}
0 & \text { if } u \in S \\
+\infty & \text { otherwise }
\end{array}\right.
$$

$$
r_{k+1}=\underset{r}{\arg \min } f(r)+\frac{\mu}{2} \cdot\left\|r-u_{k}+\eta_{k}\right\|_{2}^{2}
$$

$$
u_{k+1}=\underset{u}{\arg \min } g(u)+\frac{\mu}{2} \cdot\left\|r_{k+1}-u+\eta_{k}\right\|_{2}^{2}
$$

$$
\eta_{k+1}=\eta_{k}+r_{k+1}-u_{k+1}
$$

$$
e(t)=\left\|z_{o p t}-\tilde{z}_{t}\right\|_{2}
$$

$$
S(t)=1-\frac{e(t)-\min _{\tau}(e(\tau))}{\max _{\tau}(e(\tau))}, \quad \tau=t-T, t-T+1, \dots, t
$$