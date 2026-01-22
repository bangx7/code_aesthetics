# Code Aesthetics with Agentic Reward Feedback
<div align="center">
<a href='https://bangx7.github.io/code-aesthetics/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
  <a href="https://huggingface.co/SamuelBang/AesCoder-4B"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-ffc107?color=ffc107&logoColor=white"/></a>
  <br>
  <a href="https://arxiv.org/abs/2510.23272"><b>Paper Link</b>👁️</a>
</div>
<div align="center">
<p>
<sup>1,2</sup><a href="https://bangx7.github.io" target="_blank">Bang Xiao</a><sup>#</sup>,</span>
                <span class="author-block">
                  <sup>1,3</sup><a href="https://github.com/JackLingjie" target="_blank">Lingjie Jiang</a><sup>#</sup>,</span>
                  <span class="author-block">
                    <sup>1</sup><a href="https://www.microsoft.com/en-us/research/people/shaohanh/" target="_blank">Shaohan Huang</a><sup>✉</sup>,</span>
                    <span class="author-block">
                      <sup>1</sup><a href="https://www.microsoft.com/en-us/research/people/tengchaolv/" target="_blank">Tengchao Lv</a>,
                    </span>
                    <span class="author-block">
                      <sup>1</sup><a href="https://www.microsoft.com/en-us/research/people/yupanhuang/" target="_blank">Yupan Huang</a>,
                    </span>
                    <span class="author-block">
                      <sup>1</sup><a href="https://yushuiwx.github.io/" target="_blank">Xun Wu</a>,
                    </span>
                    <span class="author-block">
                      <sup>1</sup><a href="https://www.microsoft.com/en-us/research/people/lecu/" target="_blank">Lei Cui</a>,
                    </span>
                    <span class="author-block">
                      <sup>1</sup><a href="https://www.microsoft.com/en-us/research/people/fuwei/" target="_blank">Furu Wei</a>
                    </span>
</p>
  <p>
    <sup>1</sup>Microsoft Research Asia &nbsp;&nbsp;
    <sup>2</sup>Zhiyuan College, Shanghai Jiao Tong University &nbsp;&nbsp;
    <sup>3</sup>Peking University<br>
    <sup>#</sup>Equal Contribution
    <sup>✉</sup>Corresponding author
  </p>
</div>

 ## 🎉 News

- __[2026.01.23]__: Release the OpenDesign benchmark.

- __[2026.01.12]__: Release the [AesCode](https://huggingface.co/datasets/SamuelBang/AesCode-358K) dataset.
- __[2025.10.29]__: Release the [AesCoder-4B](https://huggingface.co/SamuelBang/AesCoder-4B/) model.
- __[2025.10.27]__: Release the [Project Page](https://bangx7.github.io/code-aesthetics/) and the [Arxiv](https://arxiv.org/abs/2510.23272) version.

## 📷 Abstract
Large Language Models (LLMs) have become valuable assistants for developers in code-related tasks. While LLMs excel at traditional programming tasks such as code generation and bug fixing, they struggle with visually-oriented coding tasks, often producing suboptimal aesthetics. In this paper, we introduce a new pipeline to enhance the aesthetic quality of LLM-generated code. We first construct AesCode-358K, a large-scale instruction-tuning dataset focused on code aesthetics. Next, we propose agentic reward feedback, a multi-agent system that evaluates executability, static aesthetics, and interactive aesthetics. Building on this, we develop GRPO-AR, which integrates these signals into the GRPO algorithm for joint optimization of functionality and code aesthetics. Finally, we develop OpenDesign, a benchmark for assessing code aesthetics. Experimental results show that combining supervised fine-tuning on AesCode-358K with reinforcement learning using agentic reward feedback significantly improves performance on OpenDesign and also enhances results on existing benchmarks such as PandasPlotBench. Notably, our AesCoder-4B surpasses GPT-4o and GPT-4.1, and achieves performance comparable to large open-source models with 480B-685B parameters, underscoring the effectiveness of our approach.


## To-do List
- [x] Release paper and project page
- [x] Release our AesCoder model
- [x] Release AesCode dataset
- [x] Release OpenDesign benchmark code

## &#x1F4DA; Citation
If you find this codebase useful for your research, please use the following entry.
```BibTeX
@misc{xiao2025codeaestheticsagenticreward,
      title={Code Aesthetics with Agentic Reward Feedback}, 
      author={Bang Xiao and Lingjie Jiang and Shaohan Huang and Tengchao Lv and Yupan Huang and Xun Wu and Lei Cui and Furu Wei},
      year={2025},
      eprint={2510.23272},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.23272}, 
}
```
