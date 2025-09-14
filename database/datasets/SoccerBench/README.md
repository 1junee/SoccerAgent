---
license: cc-by-nc-sa-4.0
---
Hi all the soccer fans and all those who are interested in our project of **"Multi-Agent System for Comprehensive Soccer Understanding"**, here is the official source of our proposed benchmark **SoccerBench**. This benchmark is the largest and most comprehensive soccer-specific benchmark, featuring around 10K standardized multimodal (text, image, video) multi-choice QA pairs across 14 distinct understanding tasks, curated through automated pipelines and manual verification.

More details of this benchmark and entire project could be refered from the following links:
[Paper](https://arxiv.org/pdf/2505.03735) | [Code](https://github.com/jyrao/SoccerAgent) | [Arxiv](https://arxiv.org/abs/2505.03735)

In **qa** folder, you could find 14 json files corresponding to different tasks as following figure shows (In paper, q7 was omitted, the rest of the qa task would be moved foward. For example, the q10 here is the q9 in the passage.) The form of the qa was like:

```
  {
    "Q": "How many appearances did the midfielder who is replacing Antoine Griezmann in this video make for Atletico Madrid from 2002 to 2018?",
    "materials": [
      "materials/q12/SoccerReplay-1988/europe_champions-league_2023-2024/2023-11-07_atletico-de-madrid-celtic-fc-champions-league/2_19_01.mp4"
    ],
    "openA": "13 appearances.",
    "closeA": "O4",
    "O1": "25 appearances",
    "O2": "7 appearances",
    "O3": "18 appearances",
    "O4": "13 appearances"
  },
```

The value of the key "materials" represents the image/video files used in this question. According files are in **materials** folder with similar folder structure.

<figure>
    <img src="https://jyrao.github.io/SoccerAgent/static/images/benchmark.png" alt="Proposed Benchmark Overview" width="80%">
    <figcaption><strong>Proposed Benchmark Overview.</strong> SoccerBench QA Generation Pipeline. We construct multi-choice QA samples based on SoccerWiki and other existing datasets. Some representative examples for each task are presented for reference.</figcaption>
</figure>

<figure>
    <img src="https://jyrao.github.io/SoccerAgent/static/images/benchmark_table.png" alt="Data Statistics of SoccerBench" width="80%">
    <figcaption><strong>Data Statistics of SoccerBench.</strong> For each, we present its name, QA type, source materials, and curation strategies. Here, SN and SR-1988 represent the SoccerNet and Soccer-Replay-1988, respectively, while LLM denotes DeepSeek-v3.</figcaption>
</figure>

For any problem, feel free to contact us via email: jy_rao@sjtu.edu.cn or zifengli@sjtu.edu.cn

For citation, you could follow this format:

      @inproceedings{rao2025soccceragent,
          title = {Multi-Agent System for Comprehensive Soccer Understanding},
          author = {Rao, Jiayuan and Li, Zifeng and Wu, Haoning and Zhang, Ya and Wang, Yanfeng and Xie, Weidi},
          booktitle = {ACM Multimedia 2025},
          year = {2025}
      }