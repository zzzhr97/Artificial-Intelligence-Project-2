# AI-project2

人工智能-project2：Use adversarial search and reinforcement  learning to produce an AI to play Gomoku.

## 记录

- *v0.0*:
  - *v0.0.0*: 
    - Test. 
- *v0.1*: 
  - *v0.1.0*: 
    - Add basic files.
- *v0.2*:  
  - *v0.2.0*: 
    - Implement basic minimax alg. without eval.  
- *v0.3*:
  - *v0.3.0*: 
    - Implement basic minimax alg. to successfully test without eval.  
  - *v0.3.1*: 
    - Add parameter about evaluate function and *TODO* to mark the code needed to be implemented.
- *v0.4*:
  - *v0.4.0*: 
    - All *TODO* completed.
- *v0.5*:
  - *v0.5.0*: 
    - `eval.py` improved.
  - *v0.5.1*: 
    - Fix some bugs and upload a relatively complete version.
  - *v0.5.2*: 
    - Add some indicators to adjust parameters.
  - *v0.5.3*:
    - Improve `README.md` and add `.gitignore`
- *v0.6*:
  - *v0.6.0*:
    - Restore to initial eval function due to some bugs in developed eval function.
- *v1.0*:
  - *v1.0.0*:
    - Finish RL part.
  - *v1.0.1*:
    - Modify `README.md` and modify some codes in Minimax.
    - Add `.gitattributes`.
  - *v1.0.2*:
    - Improve RL part.
    - Add 8-connectivity to RL.
    - Add `config.ini` to control the training process of RL.

## RL运行方法
- 将模型权重文件放到文件夹中，如`./RL/models/policy_Simple__best_epoch8350.pth`
- 在`scripts/run.bat`中，更改参数：
  - `load_path` 模型文件路径，如`./RL/models/policy_Simple__best_epoch8350.pth`
  - `internal_model` 特征提取器模型，有`Simple`和`Res`两种选择，请根据模型权重文件名来判断
  - `n_play` 模拟次数，值越高，理论上讲模型越强，但所需时间也会增加
  - `k` 权重衰减系数，在0~1之间。`k`越小，模型越倾向于在已下位置附近的位置进行落子；反之，则越能公平地考虑棋盘上的每一个落子位置。
- 运行`scripts/run`

## 参考
[Issue 25](https://github.com/junxiaosong/AlphaZero_Gomoku/issues/25)