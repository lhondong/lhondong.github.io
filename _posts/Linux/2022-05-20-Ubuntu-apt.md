---
title: "Ubuntu 的 APT 工具简介"
subtitle: "apt-get"
layout: post
author: "L Hondong"
header-img: "img/post-bg-34.jpg"
tags:
  - linux
  - apt
---

Ubuntu 系统中的 APT 工具用两个：**apt-get** & **apt-cache**。`apt-get` 主要用于软件包的安装和卸载操作，`apt-cache` 主要用于软件包的搜索查询操作。对于本地软件包的管理，Ubuntu 系统中还有一个更强大的工具 **dpkg**，暂不讨论。

以下列举这两个工具的具体操作：

## 1. 更新或升级操作

```shell
apt-get update              # 更新源  
apt-get upgrade             # 更新所有已安装的包  
apt-get dist-upgrade        # 发行版升级（如，从10.10到11.04）  
```

## 2. 安装或重装类操作

```shell
apt-get install <pkg>              # 安装软件包 <pkg>，多个软件包用空格隔开  
apt-get install --reinstall <pkg>  # 重新安装软件包 <pkg>  
apt-get install -f <pkg>           # 修复安装（破损的依赖关系）软件包 <pkg>  
```


## 3. 卸载类操作

```shell
apt-get remove <pkg>        # 删除软件包 <pkg>（不包括配置文件）  
apt-get purge <pkg>         # 删除软件包 <pkg>（包括配置文件）  
```

## 4. 下载清除类操作

```shell
apt-get source <pkg>         # 下载 pkg 包的源代码到当前目录  
apt-get download <pkg>       # 下载 pkg 包的二进制包到当前目录  
apt-get source -d <pkg>      # 下载完源码包后，编译  
apt-get build-dep <pkg>      # 构建 pkg 源码包的依赖环境（编译环境？）  
apt-get clean         # 清除缓存(/var/cache/apt/archives/{,partial}下)中所有已下载的包  
apt-get autoclean     # 类似于 clean，但清除的是缓存中过期的包（即已不能下载或者是无用的包）  
apt-get autoremove    # 删除因安装软件自动安装的依赖，而现在不需要的依赖包  
```

## 5. 查询类操作

```shell
apt-cache stats               # 显示系统软件包的统计信息  
apt-cache search <pkg>        # 使用关键字 pkg 搜索软件包  
apt-cache show   <pkg_name>   # 显示软件包 pkg_name 的详细信息  
apt-cache depends <pkg>       # 查看 pkg 所依赖的软件包  
apt-cache rdepends <pkg>      # 查看 pkg 被那些软件包所依赖  
```
