---
sidebar_position: 1
---
https://setuptools.pypa.io/en/latest/userguide/entry_point.html
# 入口点

入口点是一种元数据，可以在安装时由包公开。它们是Python生态系统中非常有用的功能，在两种情况下特别方便：

1.该包希望提供在终端上运行的命令。此功能称为控制台脚本。该命令还可以打开GUI，在这种情况下，它被称为GUI脚本。控制台脚本的一个例子是pip包提供的脚本，它允许您在终端中运行pip install等命令。

2.软件包希望通过插件定制其功能。例如，测试框架pytest允许通过pytest11入口点进行自定义，语法突出显示工具pygments允许使用入口点pygments.styles指定其他样式。

## 1 控制台脚本

让我们从控制台脚本开始。首先考虑一个没有入口点的例子。想象一下这样定义的包：

```
project_root_directory
├── pyproject.toml        # and/or setup.cfg, setup.py
└── src
    └── timmins
        ├── __init__.py
        └── ...
```

with `__init__.py` as:
```
def hello_world():
    print("Hello world")
```

现在，假设我们想提供一些从命令行执行函数hello_world（）的方法。一种方法是创建一个文件src/timmins/__main__.py，提供如下钩子：

```
from . import hello_world

if __name__ == '__main__':
    hello_world()
```

然后，在安装了timmins包之后，我们可以通过runpy模块调用hello_world（）函数，如下所示：

```
$ python -m timmins
Hello world
```

除了使用__main__.py的这种方法，您还可以创建一个用户友好的CLI可执行文件，该文件可以在没有python-m的情况下直接调用。在上面的示例中，要创建调用timmins.hello_world的命令hello world，请在配置中添加一个控制台脚本入口点：

pyproject.toml
```
[project.scripts]
hello-world = "timmins:hello_world"
```

setup.cfg
```
[options.entry_points]
console_scripts =
    hello-world = timmins:hello_world
```

setup.py
```
from setuptools import setup

setup(
    # ...,
    entry_points={
        'console_scripts': [
            'hello-world = timmins:hello_world',
        ]
    }
)
```

安装包后，用户可以通过在命令行上简单地调用hello world来调用该函数：

```
$ hello-world
Hello world
```

请注意，用作控制台脚本的任何函数，即本例中的hello_world（），都不应接受任何参数。如果你的函数需要用户的任何输入，你可以在函数体内使用常规的命令行参数解析实用程序，如argparse，来解析通过sys.argv给出的用户输入。

您可能已经注意到，我们使用了一种特殊的语法来指定控制台脚本必须调用的函数，即我们用冒号编写了timmins:hello_world：分隔包名和函数名。本文档的最后一节讨论了此语法的完整规范，这可用于指定位于包中任何位置的函数，而不仅仅是__init__.py中的函数。
## 2 GUI脚本
除了console_scripts之外，Setuptools还支持gui_scripts，它将在不在终端窗口中运行的情况下启动gui应用程序。

例如，如果我们有一个与之前目录结构相同的项目，其中__init__.py文件包含以下内容：

```
import PySimpleGUI as sg

def hello_world():
    sg.Window(title="Hello world", layout=[[]], margins=(100, 50)).read()
```

然后，我们可以添加一个GUI脚本入口点：

project.toml
```
[project.gui-scripts]
hello-world = "timmins:hello_world"
```

setup.cfg
```
[options.entry_points]
gui_scripts =
    hello-world = timmins:hello_world
```

setup.py
```
from setuptools import setup

setup(
    # ...,
    entry_points={
        'gui_scripts': [
            'hello-world = timmins:hello_world',
        ]
    }
)
```

现在，运行：

```
$ hello-world

```

将打开一个标题为“Hello world”的小应用程序窗口。

请注意，与控制台脚本一样，用作GUI脚本的任何函数都不应接受任何参数，并且任何用户输入都可以在函数体内解析。GUI脚本也使用相同的语法（在上一节中讨论）来指定要调用的函数。

注意：
> console_scripts和gui_scripts之间的区别仅影响Windows系统。[1]console_scripts被包装在控制台可执行文件中，因此它们被附加到控制台，可以使用sys.stdin、sys.stdout和sys.stderr进行输入和输出。gui_scripts被包装在gui可执行文件中，因此可以在没有控制台的情况下启动它们，但除非应用程序代码重定向它们，否则不能使用标准流。其他平台没有同样的区别。

> 控制台和GUI脚本之所以有效，是因为在幕后，像pip这样的安装程序会围绕被调用的函数创建包装脚本。例如，上述两个示例中的hello world入口点将创建一个命令hello world，并启动如下脚本：

```
 import sys
from timmins import hello_world
sys.exit(hello_world()) 
```

控制台/GUI脚本是入口点更一般概念的一种用途。另一种就是实现插件。

## 3 插件的入口点

让我们考虑一个简单的例子来了解如何实现与插件对应的入口点。假设我们有一个具有以下目录结构的包timmins：

```
timmins
├── pyproject.toml        # and/or setup.cfg, setup.py
└── src
    └── timmins
        └── __init__.py
```

在src/timmins/__init__.py中，我们有以下代码：

```
def hello_world():
    print('Hello world')
```
基本上，我们定义了一个hello_world（）函数，它将打印文本“hello world”。现在，假设我们想以不同的方式打印文本“Hello world”。当前函数只是按原样打印文本——假设我们想要另一种将文本括在感叹号内的样式：

```
!!! Hello world !!!
```

让我们看看如何使用插件来实现这一点。首先，让我们将文本的印刷风格与文本本身分开。换句话说，我们可以将src/timmins/__init__.py中的代码更改为如下内容：

```
def display(text):
    print(text)

def hello_world():
    display('Hello world')
```

在这里，display（）函数控制打印文本的样式，hello_world（）函数调用display（（）函数打印文本“hello world”。

现在display（）函数只是按原样打印文本。为了能够对其进行自定义，我们可以执行以下操作。让我们引入一组名为timmins.display的新入口点，并期望实现此入口点的插件包提供类似display（）的函数。接下来，为了能够自动发现实现此入口点的插件包，我们可以使用importlib.metadata模块，如下所示：

```
from importlib.metadata import entry_points
display_eps = entry_points(group='timmins.display')
```

注：
每个importlib.metadata。EntryPoint对象是一个包含名称、组和值的对象。例如，在如下所述设置插件包后，上述代码中的display_eps将如下所示：
```

(
    EntryPoint(name='excl', value='timmins_plugin_fancy:excl_display', group='timmins.display'),
    ...,
)
```

display_eps现在将是EntryPoint对象的列表，每个对象都引用由一个或多个已安装的插件包定义的类似display（）的函数。然后，要导入一个特定的类似display（）的函数，让我们选择与第一个发现的入口点对应的函数，我们可以使用load（）方法，如下所示：
```

display = display_eps[0].load()
```

最后，一个明智的行为是，如果我们找不到任何自定义display（）函数的插件包，我们应该回到默认实现，按原样打印文本。包括此行为后，src/timmins/__init__.py中的代码最终变为：

```
from importlib.metadata import entry_points
display_eps = entry_points(group='timmins.display')
try:
    display = display_eps[0].load()
except IndexError:
    def display(text):
        print(text)

def hello_world():
    display('Hello world')
```

timmins相关的设置就完成了。接下来，我们需要实现一个插件来实现入口点timmins.display。让我们将这个插件命名为timmins-plugin-fance，并使用以下目录结构进行设置：

```
timmins-plugin-fancy
├── pyproject.toml        # and/or setup.cfg, setup.py
└── src
    └── timmins_plugin_fancy
        └── __init__.py
```

然后，在src/timmins_plugin_fancy/__init__.py中，我们可以放置一个名为excl_display（）的函数，用于打印被感叹号包围的给定文本：

```
def excl_display(text):
    print('!!!', text, '!!!')
```

这是我们希望提供给timmins包的类似display（）的函数。我们可以通过在timmins-plugin-fance的配置中添加以下内容来实现：

project.toml
```
# Note the quotes around timmins.display in order to escape the dot .
[project.entry-points."timmins.display"]
excl = "timmins_plugin_fancy:excl_display"
```

setup.cfg
```
[options.entry_points]
timmins.display =
    excl = timmins_plugin_fancy:excl_display
```

setup.py
```
from setuptools import setup

setup(
    # ...,
    entry_points = {
        'timmins.display': [
            'excl = timmins_plugin_fancy:excl_display'
        ]
    }
)
```

基本上，此配置表示我们是组timmins.display下的一个入口点。入口点名为excl，它指的是由timmins-plugin-fanic包定义的函数excl_display。

现在，如果我们同时安装timmins和timmins-plugin-fance，我们应该得到以下结果：

```
>>> from timmins import hello_world
>>> hello_world()
!!! Hello world !!!
```

而如果我们只安装timmins而不安装timmins-plugin-fance，我们应该得到以下结果：

```
>>> from timmins import hello_world
>>> hello_world()
Hello world
```

我们的插件还可以在组timmins.display下定义多个入口点。例如，在src/timmins_plugin_fancy/__init__.py中，我们可以有两个类似display（）的函数，如下所示：

```
def excl_display(text):
    print('!!!', text, '!!!')

def lined_display(text):
    print(''.join(['-' for _ in text]))
    print(text)
    print(''.join(['-' for _ in text]))
```

然后，timmins插件fancy的配置将更改为：
project.toml
```
[project.entry-points."timmins.display"]
excl = "timmins_plugin_fancy:excl_display"
lined = "timmins_plugin_fancy:lined_display"
```

setup.cfg
```
[options.entry_points]
timmins.display =
    excl = timmins_plugin_fancy:excl_display
    lined = timmins_plugin_fancy:lined_display
```

setup.py
```
from setuptools import setup

setup(
    # ...,
    entry_points = {
        'timmins.display': [
            'excl = timmins_plugin_fancy:excl_display',
            'lined = timmins_plugin_fancy:lined_display',
        ]
    }
)
```

在timmins库中，我们还可以使用不同的加载入口点策略。例如，我们可以搜索特定的显示样式：

```
display_eps = entry_points(group='timmins.display')
try:
    display = display_eps['lined'].load()
except KeyError:
    # if the 'lined' display is not available, use something else
    ...
```

另一点是，在这个特定的例子中，我们使用插件来定制函数的行为（display（））。一般来说，我们可以使用入口点使插件不仅可以自定义函数的行为，还可以自定义整个类和模块的行为。这与控制台/GUI脚本的情况不同，后者的入口点只能引用函数。用于指定入口点的语法与控制台/GUI脚本的语法相同，并在上一节中进行了讨论。

加载和导入入口点的推荐方法是importlib.metadata模块，它是Python 3.8以来标准库的一部分，自Python 3.10以来是非临时的。对于较旧版本的Python，应使用其后端口importlib_metadata。使用后端口时，唯一需要做的更改是将importlib.metadata替换为importlib_metadata，即。

```
from importlib_metadata import entry_points
...
```

总之，入口点允许包通过插件打开其功能进行自定义。请求入口点的包不需要对实现入口点的插件有任何依赖或先验知识，下游用户能够通过将实现入口点地插件拉到一起来组合功能。