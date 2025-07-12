# 极客时间彭靖田-大模型微调训练营

这份“AI大模型微调整训练营第0期”课程大纲，围绕AI大模型微调，涵盖理论、技术、工具及实战多方面内容，以下是全部课程内容：

### 一、AI大模型四阶段总览
- **课程内容**：深度解读AI四轮浪潮（技术浪潮：人工智能、机器学习、深度学习、大语言模型；商业浪潮：智能硬件、机器人、元宇宙、AI原生应用）；剖析浪潮下AI大模型机遇与个体机会；介绍AI大模型四阶段技术总览，包括提示工程（Prompt Engineering）、微调（Fine-Tuning）、大模型训练（Pre-Training）、强化学习（RLHF）。
- **时间安排**：11月29日（周三）20:00 - 22:00

### 二、大模型微调技术原理揭秘（上）
- **课程内容**：
    - 预训练模型Fine-Tuning与演进，讲述预训练模型的演进历程，基于Transformer的大语言模型，以及基于Transformer的预训练模型微调技术。
    - 大模型高效微调技术PEFT初探，介绍Adapter Tuning、Prefix Tuning、Prompt Tuning、P-Tuning v2。
- **时间安排**：12月3日（周日）19:00 - 22:00

### 三、大模型微调技术原理揭秘（下）
- **课程内容**：
    - 大模型轻量级高效微调方法LoRA，涵盖LoRA（Low-Rank Adaptation of LLMs）、LoRA Adapter优势、LoRA Adapter for PEFT、AdaLoRA（Adaptive Budget Allocation for PEFT）。
    - 少样本PEFT新方法IA3。
    - 统一微调框架UniPELT。
- **时间安排**：12月6日（周三）20:00 - 22:00

### 四、ChatGPT大模型训练技术解读
- **课程内容**：
    - 基于人类反馈的强化学习微调RLHF，介绍有监督微调Supervised-Fine Tuning（SFT）预训练、奖励模型Reward Model（RM）、强化学习微调Reinforcement Learning from Human Feedback（RLHF）。
    - 混合专家模型Mixture of Experts（MoE）技术架构揭秘，包括动态路由模型Switch Transformer、专家选择模型Expert Choice、通用语言生成模型General Language Model（GLM）。
- **时间安排**：12月10日（周日）19:00 - 22:00

### 五、大模型开发工具库Hugging Face Transformers（上）
- **课程内容**：
    - Transformers库概要，讲解Transformers 3.0+新特性、使用Transformers原因、Transformers库核心概念与功能、安装Transformers库。
    - Transformers Pipeline流水线，包括了解Pipeline、开箱即用（pipelines API）、使用pipeline快速实现情感判断任务、使用pipeline快速实现摘要任务。
- **时间安排**：12月13日（周三）20:00 - 22:00

### 六、大模型开发工具库Hugging Face Transformers（中）
- **课程内容**：
    - Transformers Model模型，介绍Transformers模型类型列表、模型架构介绍、模型保存。
    - Transformers Tokenizer分词器，包括探索分词器、Transformers分词器加载。
- **时间安排**：无

### 七、大模型高效微调工具库Hugging Face PEFT
- **课程内容**：
    - Hugging Face PEFT库，介绍PEFT是什么、PEFT典型使用场景、PEFT基础概念与核心功能、安装PEFT库。
    - 实战，包括使用PeftConfig定义和存储模型参数、使用PeftModel.from_pretrained加载大模型、使用Auto Class实现LoRA。
- **时间安排**：12月17日（周日）19:00 - 22:00

### 八、实战：使用QLoRA实现ChatGLM-6B全参数微调（上）
- **课程内容**：
    - 模型微调数据集准备流程，包括使用Hugging Face Datasets获取开源数据集、数据清洗、数据加载与编码解码。
    - 使用PEFT QLoRA微调ChatGLM-6B，介绍定义Transformers管道、加载模型、使用模型。
- **时间安排**：12月20日（周三）20:00 - 22:00

### 九、实战：使用QLoRA实现ChatGLM-6B全参数微调（下）
- **课程内容**：使用私有化数据集训练微调，包括使用QLoRA+LoRA高质量微调、数据隐私保护、训练环境搭建、ChatGLM-Bloomberg-6B服务。
- **时间安排**：12月24日（周日）19:00 - 22:00

### 十、个性化ChatBot开发环境搭建
- **课程内容**：
    - Jupyter Lab交互式开发环境。
    - Visual Studio Code开发插件。
    - 多模态开发环境搭建。
    - 向量数据库Chroma。
- **时间安排**：无

### 十一、实战：基于微调ChatGLM-6B打造个性化ChatBot
- **课程内容**：
    - 产品设计与功能规划，包括技术方案与架构设计、使用LangChain PromptTemplate实现提示工程、使用LangChain私有化ChatGLM-6B。
    - 生产级ChatBot部署，包括Docker容器部署、使用Docker Compose搭建运行环境、编译ChatBot docker镜像、使用Docker部署ChatBot聊天服务。
- **时间安排**：12月27日（周三）20:00 - 22:00

### 十二、实战：结合检索增强生成（RAG）的ChatBot
- **课程内容**：
    - 嵌入技术Embedding 101，包括Embedding是什么、常见Embedding模型、OpenAI Embedding的模型关系。
    - 向量数据库Chroma 101，包括Chroma主要功能接口、Chroma主流功能。
    - 搭建ChatBot知识库，包括使用GPT生成领域知识、向向量数据库添加知识、使用LangChain Prompt模板向向量数据库提问。
    - 使用LangChain实现结合检索增强生成（RAG）的ChatBot。
- **时间安排**：1月3日（周三）20:00 - 22:00

### 十三、分布式大模型微调整训练框架Microsoft DeepSpeed
- **课程内容**：
    - DeepSpeed框架简介，包括DeepSpeed是什么、DeepSpeed价值定位、DeepSpeed开源生态。
    - DeepSpeed核心模块解读，包括DeepSpeed - Training模块、DeepSpeed ZeRO优化器、DeepSpeed - Inference模块、DeepSpeed - Science模块。
    - DeepSpeed分布式技术架构，包括DeepSpeed分布式训练、DeepSpeed分布式推理、DeepSpeed ZeRO优化器。
    - DeepSpeed Zero Redundancy Optimizer（ZeRO）技术，包括模型微调内存资源优化技术、ZeRO - Offload、ZeRO - Infinity。
    - 端到端RLHF训练系统：DeepSpeed - Chat，包括整合Hugging Face简化模型训练和推理体验、DeepSpeed - Chat关键技术。
    - 使用DeepSpeed进行LoRA Finetune。
- **时间安排**：1月7日（周日）19:00 - 22:00

### 十四、实战：使用DeepSpeed Chat实现RLHF模型微调
- **课程内容**：
    - Meta OPT系列模型介绍。
    - 使用DeepSpeed Chat RLHF训练Meta OPT模型，包括数据准备（Hugging Face Datasets）、训练准备、训练模型、Reward RLHF训练方法。
    - 使用AI全栈云平台运行模型服务。
- **时间安排**：1月10日（周三）20:00 - 22:00

### 十五、国产化适配实战：基于华为昇腾910微调训练ChatGLM-6B
- **课程内容**：
    - 华为昇腾（HUAWEI Ascend）AI处理器介绍，包括昇腾AI处理器架构、AI训练处理器昇腾Ascend 910。
    - 在昇腾Ascend 910上模型微调ChatGLM-6B，包括使用MindSpore调用Ascend 910、使用ChatGLM-6B进行模型训练、使用ChatGLM-6B进行模型推理。
- **时间安排**：1月14日（周日）20:00 - 22:00

### 十六、Meta LLaMA-2大模型家族
- **课程内容**：
    - LLaMA-2大模型系列介绍，包括LLaMA-2大模型介绍、官方支持LLaMA-2模型微调。
    - 在Hugging Face使用Meta官方LLaMA-2模型，包括申请和使用Meta LLaMA-2访问权限、部署运行Meta官方LLaMA-2模型。
- **时间安排**：1月17日（周三）20:00 - 22:00

### 十七、预训练Meta LLaMA-2大模型（上）
- **课程内容**：
    - 训练数据准备，包括在Hugging Face选择合适的开源数据集、使用Datasets库进行中文数据集获取、数据质量检查。
    - 模型结构转换，包括将Meta官方LLaMA-2模型转换为Hugging Face模型格式。
    - 使用QLoRA技术预训练LLaMA-2-7B大模型，包括选择合适数量Nvidia GPU训练版本、预训练数据准备。
- **时间安排**：1月21日（周日）19:00 - 22:00

### 十八、预训练Meta LLaMA-2大模型（下）
- **课程内容**：
    - 使用QLoRA技术预训练LLaMA-2-7B大模型（续），包括模型训练参数设置、模型训练过程监控。
    - 实战Meta LLaMA-2预训练。
- **时间安排**：1月24日（周三）20:00 - 22:00 

# 课程对比表
|课程|企业级 Agents 开发实战营|大模型应用开发实战营|
| ---- | ---- | ---- |
|课程特点|专注生产实战，覆盖从立项到部署的完整开发流程|侧重基础知识传授，手把手教学提示工程、基于 GPT 的开发和 LangChain 框架，涵盖硬件、理论和开发实践|
| |通过 3 个不同工作场景实战演练，提供扩展空间，方便学员按需发展|实战项目重在引导入门，适配 OpenAI 和 LangChain 最新技术与框架|
|面向群体|生产级应用开发者和项目管理人员|大模型开发爱好者和初学者|
|教学方式|通过端到端企业级项目研发，全面提升实战能力|理论剖析与实战操作并重，案例驱动技能学习|
|实战项目对比|GitHub Sentinel：项目管理和更新推送工具，提升团队协作效率与项目管理便捷性，可扩展为信息流订阅和总结服务 Agent<br><br>LanguageMentor：在线英语私教，提供高效语言学习体验，可扩展为 100 + 语种语言教练 Agent<br><br>ChatPPT：支持多模态输入的 PPT 生成 Agent，可扩展为企业自动化流程提效 Agent|深度剖析和实现热门大模型项目（OpenAI - Translator、RAG、AutoGPT 等），通过多个落地案例助力熟练上手大模型应用开发| 
