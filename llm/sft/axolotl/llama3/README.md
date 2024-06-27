### Requirements

```shell
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl

pip3 install packaging ninja
pip3 install -e '.[flash-attn,deepspeed]'
```


_Login to huggingface_
```shell
huggingface-cli login
```

_Using wandb to log_
```shell
wandb login
```

### Training
```shell
accelerate launch -m axolotl.cli.train llama3-7b.yaml
```