# Soft_Actor_Critic_DeepLearning
### 深度学习小组作业，用PyTorch复现ICML2018的Soft Actor-Critic (SAC) 模型，欢迎指正！
#### Reference
##### 1 Haarnoja, Tuomas et al. “Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.” ArXiv abs/1801.01290 (2018): n. pag.
##### 2 Haarnoja, Tuomas et al. “Soft Actor-Critic Algorithms and Applications.” ArXiv abs/1812.05905 (2018): n. pag. 

***

You can create the environment as follows:
```bash
conda create -n sac python=3.10.12
conda activate sac
pip install -r requirements.txt
```

After installing the requirement.txt, you also need to download mujoco_py in the main folder. Here we provide the instruction on Linux:
```bash
git clone https://github.com/openai/mujoco-py.git
cd mujoco-py
pip install -r requirements.txt
pip3 install -U 'mujoco-py<2.2,>=2.1'
sudo gedit ~/.bashrc
```

On the last line of file .bashrc, add
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia 
```
then
```bash
source ~/.bashrc
```

***

In the experiment part, you can run as follows:

###### Part 1
for Humanoid:
```bash
python main.py --env-name Humanoid-v4 --alpha 0.05
python main.py --env-name Humanoid-v4 --alpha 0.05 --tau 1 --target_update_interval 1000
```
for HalfCheetah:
```bash
python main.py --env-name HalfCheetah-v4 --alpha 0.2
python main.py --env-name HalfCheetah-v4 --alpha 0.2 --tau 1 --target_update_interval 1000
```
for Ant:
```bash
python main.py --env-name Ant-v4 --alpha 0.2
python main.py --env-name Ant-v4 --alpha 0.2 --tau 1 --target_update_interval 1000
```
for Swimmer:
```bash
python main.py --env-name Swimmer-v4 --alpha 0.2
python main.py --env-name Swimmer-v4 --alpha 0.2 --tau 1 --target_update_interval 1000
```
###### Part 2
```bash
python main.py --env-name Humanoid-v4 --alpha 0.05 --tau 0.1 
python main.py --env-name Humanoid-v4 --alpha 0.05 --tau 0.01
python main.py --env-name Humanoid-v4 --alpha 0.05 --tau 0.001
```
###### Part 3
When use SACv2, run
```bash
python main.py --model v2 --env-name Humanoid-v4 --alpha 0.0036
```
All the result are in tensorboard loggers, which you can check by running
```bash
tensorboard --logdir=run --host=127.0.0.1
```
and then log in ```http://127.0.0.1:6006```.


