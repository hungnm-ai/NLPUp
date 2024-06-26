### Requirements

```shell
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl

pip3 install packaging ninja
pip3 install -e '.[flash-attn,deepspeed]'
```


```shell
huggingface-cli login
```

### Training
```shell
accelerate launch -m axolotl.cli.train llama3-7b.yaml

```