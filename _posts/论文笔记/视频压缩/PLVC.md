
$$
\min _{G} \max _{D} \mathbb{E}[f(D(\boldsymbol{x}))]+\mathbb{E}[g(D(G(\boldsymbol{y})))]
$$

$$
\min _{G} \max _{D} \mathbb{E}[f(D(\boldsymbol{x} \mid \boldsymbol{c}))]+\mathbb{E}[g(D(\hat{\boldsymbol{x}} \mid \boldsymbol{c}))]
$$

$$
\mathcal{L}_{w}=\sum_{i=1}^{N} R\left(\boldsymbol{y}_{i}\right)+\lambda \cdot d\left(\hat{\boldsymbol{x}}_{i}, \boldsymbol{x}_{i}\right)
$$

$$
\mathcal{L}_{D}= \sum_{i=1}^{N}\left(-\log \left(1-D\left(\hat{\boldsymbol{x}}_{i}, \hat{\boldsymbol{x}}_{i-1} \mid \boldsymbol{y}_{i}, \boldsymbol{m}_{i}, \boldsymbol{h}_{i-1}^{D}\right)\right)\right. \left.-\log D\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{i-1} \mid \boldsymbol{y}_{i}, \boldsymbol{m}_{i}, \boldsymbol{h}_{i-1}^{D}\right)\right)
$$

$$
\mathcal{L}_{G}=\sum_{i=1}^{N}\left(\alpha \cdot R\left(\boldsymbol{y}_{i}\right)+\lambda^{\prime} \cdot d(\hat{\boldsymbol{x}}, \boldsymbol{x})\right. \left.-\beta \cdot \log D\left(\hat{\boldsymbol{x}}_{i}, \hat{\boldsymbol{x}}_{i-1} \mid \boldsymbol{y}_{i}, \boldsymbol{m}_{i}, \boldsymbol{h}_{i-1}^{D}\right)\right)
$$