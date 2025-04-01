import json
import requests
from utils import cal_local_llm_t5

test_input = "summarize: The quick brown fox jumps over the lazy dog."
output = cal_local_llm_t5(test_input, port=6000)
print("📝 模型输出：", output)
