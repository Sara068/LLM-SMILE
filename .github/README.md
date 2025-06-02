# LLM_SMILE
 
[![Kaggle](https://img.shields.io/badge/View%20on-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/zeinabdehghani/llm-smile-gpt-3-5?scriptVersionId=243290348)

## Project Description
Large language models like GPT, LLAMA, and Claude have become incredibly powerful at generating text, but they are still black boxes, so it is hard to understand how they decide what to say. That lack of transparency can be problematic, especially in fields where trust and accountability matter. To help with this, we introduce SMILE, a new method that explains how these models respond to different parts of a prompt. SMILE is model-agnostic and works by slightly changing the input, measuring how the output changes, and then highlighting which words had the most impact. Create simple visual heat maps showing which parts of a prompt matter the most. We tested SMILE on several leading LLMs and used metrics such as accuracy, consistency, stability, and fidelity to show that it gives clear and reliable explanations. By making these models easier to understand, SMILE brings us one step closer to making AI more transparent and trustworthy.

## 📊 SMILE Framework Overview

<p align="center">
  <img src="docs/Figures/ExplainableLLM_Graphical_Abstract.png" alt="SMILE Framework Overview" width="700">
</p>

---

## 🎓 Presentation

We provide a full recorded presentation that introduces the SMILE framework and explains its methodology and evaluation in detail.

<p align="center">
  <a href="https://www.youtube.com/watch?v=pJePjOb2Tj4">
    <img src="https://img.youtube.com/vi/pJePjOb2Tj4/hqdefault.jpg" alt="Watch the presentation on YouTube" width="600">
  </a>
</p>

📄 Full paper available on arXiv:  
[https://arxiv.org/abs/2505.21657](https://arxiv.org/abs/2505.21657)
