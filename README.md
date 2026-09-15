# 基于朴素贝叶斯的垃圾邮件过滤与纠错系统

## 1. 项目简介

本项目是一个基于 Python、Tkinter、SQLite 和伯努利朴素贝叶斯实现的教学型垃圾邮件过滤系统。

系统模拟多个用户之间的邮件发送和接收，邮件到达收件人后先经过固定关键词规则过滤，再由伯努利朴素贝叶斯分类器计算垃圾邮件概率。用户发现误判后可以人工纠错，系统会将反馈写入训练数据，使后续概率计算发生变化。

项目重点是把“贝叶斯公式、关键词特征、邮件过滤、用户反馈学习”串成一个可以操作和演示的完整过程。

## 2. 功能特性

- 固定垃圾关键词过滤
- 伯努利朴素贝叶斯分类
- 垃圾邮件概率计算
- 邮件发送、收件箱、已发送、垃圾回收站
- 用户人工纠错
- 根据反馈更新训练数据
- Tkinter 图形界面
- SQLite 本地持久化

## 3. 技术栈

- Python
- Tkinter
- SQLite
- Bernoulli Naive Bayes
- unittest

本项目主要使用 Python 标准库，不需要额外第三方依赖。

## 4. 项目结构

```text
naive-bayes-spam-filter/
├── data/                 # 运行时生成 mail.db
├── docs/                 # 演示说明和截图目录
├── tests/                # 单元测试
├── bayes.py              # 伯努利朴素贝叶斯算法
├── database.py           # SQLite 建表、初始化和数据访问
├── mail_service.py       # 邮件分类、收发、纠错业务
├── main.py               # 程序入口
├── ui.py                 # Tkinter 图形界面
├── README.md
├── requirements.txt
└── .gitignore
```

## 5. 算法说明

### 固定关键词规则

系统将 `优惠`、`中奖` 设置为固定垃圾关键词。邮件主题或正文命中任一关键词时，直接判定为垃圾邮件并进入垃圾回收站。此时贝叶斯概率不参与最终判定，界面中概率显示为 `-`。

### 伯努利朴素贝叶斯

未命中固定规则的邮件进入贝叶斯分类器。系统使用以下关键词作为 0/1 二值特征：

```text
限时、领取、点击、链接、免费、现金、礼品、奖励、恭喜、促销、优惠、中奖
```

每个关键词只判断是否出现，不统计出现次数。模型根据 `training_data` 表中的训练样本计算先验概率和条件概率。

### 拉普拉斯平滑

为避免某个关键词在某类样本中从未出现导致概率为 0，条件概率使用拉普拉斯平滑：

```text
P(w|C) = (类别 C 中包含关键词 w 的邮件数 + 1) / (类别 C 邮件总数 + 2)
```

内部使用 log 概率计算，避免多个小概率相乘造成浮点下溢。

### 分类阈值

贝叶斯分类结果会归一化为垃圾概率和正常概率。当垃圾概率 `>= 55%` 时，系统判定为垃圾邮件；否则判定为正常邮件。

### 用户反馈学习

用户可以将垃圾邮件标记为正常，也可以将正常邮件标记为垃圾。系统会把“主题 + 正文”和用户标签写入 `training_data`，来源为 `user_feedback`。后续分类会重新读取最新训练数据，因此概率会随反馈变化。

固定关键词规则优先级高于贝叶斯模型。用户反馈会影响贝叶斯概率，但不会自动取消 `优惠`、`中奖` 这类强规则。

## 6. 环境要求

- Python 3.10 或更高版本
- Windows、macOS 或 Linux 桌面环境
- Python 安装需包含 Tkinter

## 7. 运行方式

从 GitHub 克隆后运行：

```powershell
git clone https://github.com/dyabn/naive-bayes-spam-filter.git
cd naive-bayes-spam-filter
python main.py
```

程序首次启动时会自动创建并初始化：

```text
data/mail.db
```

数据库文件属于运行时数据，已被 `.gitignore` 排除，不提交到 GitHub。

## 8. 使用说明

1. 在顶部选择当前用户，例如 `张三`、`李四` 或 `王五`。
2. 点击左侧 `+ 写邮件`，选择收件人、填写主题和正文，然后发送。
3. 点击 `收件箱`、`已发送`、`垃圾回收站` 查看不同文件夹。
4. 选择邮件后，右侧会显示主题、正文、概率、判定方式、判定原因和检测关键词。
5. 在收件箱中可点击 `标记为垃圾邮件`。
6. 在垃圾回收站中可点击 `这不是垃圾邮件`。
7. 已发送邮件不允许纠错，避免影响收件人的邮件视图。

## 9. 演示流程

### 普通邮件

- 发件人：张三
- 收件人：李四
- 主题：项目会议
- 正文：明天下午三点召开项目会议，请准时参加。
- 预期：李四收件箱出现邮件，判定方式为朴素贝叶斯，正常概率高。

### 固定规则垃圾邮件

- 发件人：张三
- 收件人：李四
- 主题：活动通知
- 正文：恭喜中奖，优惠活动开始，请及时领取。
- 预期：李四垃圾回收站出现邮件，判定方式为固定关键词规则，概率显示为 `-`。

### 贝叶斯垃圾邮件

- 发件人：张三
- 收件人：李四
- 主题：礼品通知
- 正文：限时领取礼品，请点击链接查看奖励。
- 预期：未命中 `优惠/中奖`，由贝叶斯判定为垃圾邮件，界面显示较高垃圾概率和检测关键词。

### 人工纠错

在垃圾回收站中选择贝叶斯垃圾邮件，点击 `这不是垃圾邮件`。邮件会移回收件箱，同时反馈样本写入训练数据。再次发送相同或相似内容时，垃圾概率会发生变化。

## 10. 测试

运行全部自动测试：

```powershell
python -m unittest discover -s tests -v
```

运行语法编译检查：

```powershell
python -m py_compile database.py bayes.py mail_service.py ui.py main.py
```

## 11. 项目截图

建议在完成人工演示后，将截图保存到：

```text
docs/images/
```

推荐文件名：

```text
01_main_interface.png
02_normal_mail.png
03_rule_spam.png
04_bayes_spam.png
05_feedback_correction.png
```

截图可用于课程报告或答辩 PPT。

## 12. Git 提交说明

本项目按阶段提交，历史清晰对应开发过程：

```text
init: initialize project structure
feat: implement SQLite database
feat: implement Bernoulli Naive Bayes classifier
feat: add spam keyword filtering rules
feat: implement mail sending and folder workflows
feat: implement classification correction and feedback learning
feat: add Tkinter graphical user interface
docs: improve README and add demo checklist
```
