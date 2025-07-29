
# SafeLLM-Multilingual-Eval

A multilingual evaluation framework for testing the safety, robustness, and alignment of large language models (LLMs) across high-stakes domains.

一个用于多语言场景中大语言模型（LLM）安全性、稳健性与对齐性评估的开源框架，聚焦医疗、法律、教育等高风险应用领域。

## 🔍 Overview / 项目简介

This project provides a testbed of adversarial and safety-critical prompts in over 12 languages, targeting common failure modes in multilingual deployments of LLMs like Claude, GPT-4, and Mistral.

该框架涵盖12种以上语言，内置对抗性与安全关键型提示，专注于评估Claude、GPT-4、Mistral等主流大模型在多语言场景中的脆弱点与行为偏差。

## 🎯 Objectives / 研究目标

- Evaluate LLM safety across low-resource languages  
- Test prompt injection and jailbreak behavior across languages  
- Compare alignment responses under semantic equivalents  
- Support reproducibility and open-source benchmarking

- 评估低资源语言下的模型安全表现  
- 跨语言测试提示注入与越狱风险  
- 对比语义等价表达下的对齐响应一致性  
- 提供可重复、可开源的对齐基准测试工具

## 🧪 Evaluation Domains / 评估场景

- 🏥 Healthcare (e.g. triage decision errors)  
- ⚖️ Legal assistance (e.g. biased advice)  
- 🎓 Education (e.g. misalignment in guidance)  
- 🌐 Cross-lingual policy compliance

## 🧰 Features / 核心功能

- ✅ Multilingual adversarial prompt set  
- ✅ Model response collection interface  
- ✅ Safety scoring + visualization  
- ✅ Support for Claude, OpenAI, Mistral APIs

## 📂 Repository Structure / 项目结构

```
📁 datasets/             # Multilingual prompts  
📁 evaluation/           # LLM scoring scripts  
📁 visualizations/       # Plots + dashboards  
📄 config.yaml           # Model + language settings  
📄 README.md
```

## 📜 License

MIT License

## 🔗 Maintainer / 项目负责人  
**Dominick Dominican**  
Email: dominickdominican47@gmail.com
