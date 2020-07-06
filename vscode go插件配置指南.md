---
typora-root-url: ./
typora-copy-images-to: images
---

# VsCode Go插件配置最佳实践指南

[toc]

## VsCode Go插件工作原理

* 原理

VsCode Go插件的工作原理与其它的ide是不一样的，比如idea(goland)，它是通过一系列go的小工具来完成ide的相关功能。比如智能提示，代码导航（查看引用，查看源码，查看接口），符号搜索，括号匹配，代码段之类的语言功能等。

但是如果是启用了go language server，那就是用的vscode的lsp来工作的，不是用go的小工具，推荐大家用go language server．

* 为什么你的VsCode Go插件不能正常工作

那些插件所依赖小工具如果不能正确工作（比如没有正确安装，比如版本不对，go path不对，没有下载成功），你的VsCode Go插件不能工作。

## VsCode Go的正确安装方式

* 先从扩展管理中安装Go插件

![image-20200614232656622](/images/image-20200614232656622.png)

## VsCode Go插件依赖的工具

* 安装Go插件所依赖的go tools

按ctrl+shift+p 调出命令面板，输入go install tools 选Go: Install/Update Tools

![image-20200614233129826](/images/image-20200614233129826.png)

### go tools

![image-20200614233209144](/images/image-20200614233209144.png)

这个阶段可能会失败。

很多人就卡在这一步就进行不下去了。这个得自己想一下办法。网上现在的办法都是让你去下载下来放到某个目录，心智负担太大还容易搞错。

我介绍2个方法

方法1：开国外vpn或者代理，让它安装成功。

方法2：设置go proxy.

> go env -w  GOPROXY=<https://goproxy.cn>
> 清空缓存 go clean --modcache

### go path配置

gopath直接在环境变量中设置就可以了，不用单独在vscode中设置。

关于go path的设置，还有一个问题，就是要不要设置2个的问题。

设置2个的目的是为了将工程放在第二个gopath下面，第一个是放go get的

在设置2个的时候，执行go get的会默认下载到一个gopath，但这个对于vscode来说可能会有点问题。

vscode中可以为在vscode中安装的go tools设置一个单独的目录
具体设置项为 Tools Gopath，使用ctrl+, 然后输入tools gopath ，在下方填你想独立存放刚才第二步安装的工具的存放的地方了。

![image-20200618130923478](/images/image-20200618130923478.png)

> Go: Tools Gopath
> Location to install the Go tools that the extension depends on if you don't want them in your GOPATH.

### go mod相关

如果你现在使用了go mod模式，就不用纠结配置几个gopath的问题，只配置一个就好了。
vscode的go mod支持需要启用language server
按ctrl+, （注意是ctrl + 英文状态的逗号）调出配置界面，输入go.lang

![image-20200618130729380](/images/image-20200618130729380.png)

把 Use Language Server设置选中状态即开启了gopls了，这时vscode就会很好的支持go mod类型的项目了。

## VsCode Go插件在Go项目中的正确配置

### 如何运行

如果是一个单独的main.go这种，现在你的vscode应该可以工作了，但是在工程里面可能不行。

工程一般有2种 结构

一种是有src目录，就是go 代码放在工程目录下面的src目录下面，这就可能会导致一些项目不能正确加载，比如非go mod项目。

这时候在工程目录下面建一个.vscode目录，然后在.vscode目录下面创建.settings.json文件

在里面设置gopath

![image-20200618131505964](/images/image-20200618131505964.png)

如果你的main包不在src目录下面，则需要设置cwd，也就是工作时切换到的目录

这时候可以选中你的main.go，按ctrl+F5，start Without Debuging，开始运行了．

说一点，如果是go mod的工程，这个gopath就无需要配置了．

### 如何Debug和运行

vscode里面正常的工程项目main包一般在cmd下面，不同的项目可能不同，但一定有一个main包。

选中你的main.go，按F5，这时就会弹出一个需要创建launch.json，点create a launch.json，直接默认的选中go，就会在.vscode目录下生成一个launch.json文件

![image-20200618132108552](/images/image-20200618132108552.png)

生成的launch.json如下

![image-20200705185650862](/images/image-20200705185650862.png)

这时候一般的程序都可以开始调试了，但是如果你的main.go启动是需要一些配置文件话，需要配置一个cwd的属性．

`cwd` - current working directory for finding dependencies and other files

加上cwd的属性如下

##

![ ](/images/image-20200705190109398.png)

到此，设置好断点，选中入口文件，按F5就可以进入debug了，然后f10单步之类的快捷键，可以看界面上的提示，自己点几下就知道乍玩了．

## VsCode Go插件的那些常用快捷

ctrl+p　文件搜索快捷键

ctrl+shift+p　命令快捷键

ctrl+shift+k　删除一行

alt+左方向键　回到上一次编辑的地方

ctrl+鼠标左键，跳到方法定义的地方

## 写在最后

要不要切换到vscode，这个不重要，重要的是你选择一个ide并灵活熟练使用他．还有，最后，尽量不要使用破解版本的idea(goland)．

## 求关注

我是coding老王，做服务端开发已经有十多年了，熟悉java，golang，Python，对大数据和机器学习，也有一些钻研，欢迎关注和交流。
