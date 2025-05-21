"""
统一LLM接口实现，支持本地Hugging Face和远程API模式
"""

import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Generator
from llama_index.core.llms import (
    CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
)
from llama_index.core.llms.callbacks import llm_completion_callback
from openai import OpenAI

class BaseLLM(ABC):
    """LLM基础接口类"""

    @abstractmethod
    def __call__(self, text: str, max_tokens: int = 300, return_full_text: bool = False, **kwargs) -> List[
        Dict[str, str]]:
        """调用LLM生成文本"""
        pass

    @abstractmethod
    def apply_chat_template(self, messages: List[Dict[str, str]], tokenize: bool = False,
                            add_generation_prompt: bool = True) -> str:
        """应用对话模板"""
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """编码文本为token IDs"""
        pass

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """从token IDs解码为文本"""
        pass


class HuggingFaceLLM(BaseLLM):
    """HuggingFace本地LLM实现"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, text: str, max_tokens: int = 300, return_full_text: bool = False, **kwargs) -> List[
        Dict[str, str]]:
        temperature = kwargs.get('temperature', 0.4)
        top_p = kwargs.get('top_p', 0.9)
        do_sample = kwargs.get('do_sample', True)

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # 处理生成的文本
        if return_full_text:
            results = [self.tokenizer.decode(outputs[0], skip_special_tokens=True)]
        else:
            # 只返回新生成的文本部分
            new_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            results = [new_text]

        return [{'generated_text': text} for text in results]

    def apply_chat_template(self, messages: List[Dict[str, str]], tokenize: bool = False,
                            add_generation_prompt: bool = True) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=tokenize,
                                                  add_generation_prompt=add_generation_prompt)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)


class SaturnLLM(BaseLLM):
    """Saturn API LLM实现"""

    def __init__(self, client, tokenizer, model_name="meta/llama-3_1-8b-instruct"):
        self.client = client
        self.tokenizer = tokenizer
        self.model_name = model_name

    def __call__(self, text: str, max_tokens: int = 300, return_full_text: bool = False, **kwargs) -> List[
        Dict[str, str]]:
        response = self.client.completions.create(
            model=self.model_name,
            prompt=text,
            max_tokens=max_tokens,
            temperature=kwargs.get('temperature', 0.4),
            top_p=kwargs.get('top_p', 0.9),
            stream=False
        )
        return [{'generated_text': response.choices[0].text}]

    def apply_chat_template(self, messages: List[Dict[str, str]], tokenize: bool = False,
                            add_generation_prompt: bool = True) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=tokenize,
                                                  add_generation_prompt=add_generation_prompt)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)


class WrappedCustomLLM(CustomLLM):
    """适配LlamaIndex的LLM包装器"""

    def __init__(self, llm_interface):
        super().__init__()
        self.llm_interface = llm_interface

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = self.llm_interface(prompt, **kwargs)[0]['generated_text']
        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> Generator[CompletionResponse, None, None]:
        response = self.llm_interface(prompt, **kwargs)[0]['generated_text']

        # 简单模拟流式输出，实际应用中需要根据接口特性调整
        text = ""
        text += response
        yield CompletionResponse(text=text, delta=response)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4096,
            num_output=1024,
            model_name="unified-llm"
        )


def create_llm(mode: str, model_name: str, hf_token: str, saturn_token: str,
               use_4bit: bool = False, use_8bit: bool = False, device: str = "auto"):
    """
    根据指定模式创建LLM接口

    Args:
        mode: LLM模式 ('hf' 或 'nim')
        model_name: 模型名称
        hf_token: Hugging Face token
        saturn_token: Saturn API token
        use_4bit: 是否使用4bit量化
        use_8bit: 是否使用8bit量化
        device: 设备设置

    Returns:
        BaseLLM: 创建的LLM接口实例
    """
    from utils import init_tokenizer_and_model

    if mode == "hf":
        tokenizer, model = init_tokenizer_and_model(
            model_name, hf_token, use_4bit, use_8bit, device, load_model=True
        )
        return HuggingFaceLLM(model, tokenizer)
    else:
        from retrieval import NimClient
        tokenizer, _ = init_tokenizer_and_model(
            model_name, hf_token, load_model=False
        )
        nim_client = NimClient(saturn_token)
        return SaturnLLM(nim_client.client, tokenizer)