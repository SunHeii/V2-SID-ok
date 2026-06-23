from trl import SFTTrainer, SFTConfig
import trl
import transformers
import inspect

print("trl =", trl.__version__)
print("transformers =", transformers.__version__)

print(inspect.signature(SFTConfig.__init__))
print(inspect.signature(SFTTrainer.__init__))