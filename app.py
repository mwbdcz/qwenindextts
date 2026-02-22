# -*- coding: utf-8 -*-
"""
Qwen3-TTS 有声小说配音 WebUI - 改进版
支持三种模式：语音克隆、音色设计、自定义音色
改进：音色持久化存储、自动加载、音色管理
"""

import os
import time
import torch
import soundfile as sf
import numpy as np
import gradio as gr
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import base64

# ==================== 目录管理 ====================

# 创建输出目录
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# 音色存储目录
VOICES_DIR = Path("saved_voices")
VOICES_DIR.mkdir(exist_ok=True)

# 音色元数据文件
VOICES_META_FILE = VOICES_DIR / "voices_metadata.json"


# ==================== 音色持久化管理 ====================

class VoiceStorage:
    """音色持久化存储管理"""
    
    def __init__(self, voices_dir: Path = VOICES_DIR, meta_file: Path = VOICES_META_FILE):
        self.voices_dir = voices_dir
        self.meta_file = meta_file
        self.voices_metadata: Dict[str, dict] = {}
        self.load_metadata()
    
    def load_metadata(self):
        """加载音色元数据"""
        if self.meta_file.exists():
            try:
                with open(self.meta_file, 'r', encoding='utf-8') as f:
                    self.voices_metadata = json.load(f)
                print(f"✅ 已加载 {len(self.voices_metadata)} 个已保存的音色")
            except Exception as e:
                print(f"⚠️ 加载音色元数据失败：{e}")
                self.voices_metadata = {}
        else:
            print("ℹ️ 未找到已保存的音色")
            self.voices_metadata = {}
    
    def save_metadata(self):
        """保存音色元数据"""
        try:
            with open(self.meta_file, 'w', encoding='utf-8') as f:
                json.dump(self.voices_metadata, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"❌ 保存音色元数据失败：{e}")
            return False
    
    def save_voice(self, voice_name: str, text: str, instruct: str, 
                   wav: np.ndarray, sr: int, language: str = "Chinese") -> Tuple[bool, str]:
        """保存音色到文件"""
        try:
            # 创建音色目录
            voice_dir = self.voices_dir / voice_name
            voice_dir.mkdir(exist_ok=True)
            
            # 保存音频文件
            audio_path = voice_dir / "reference.wav"
            sf.write(str(audio_path), wav, sr)
            
            # 保存元数据
            meta = {
                "name": voice_name,
                "text": text,
                "instruct": instruct,
                "language": language,
                "sr": sr,
                "duration": len(wav) / sr,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "audio_path": str(audio_path),
            }
            
            # 保存元数据到 JSON
            meta_path = voice_dir / "metadata.json"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            
            # 更新内存中的元数据
            self.voices_metadata[voice_name] = meta
            self.save_metadata()
            
            return True, f"✅ 音色 '{voice_name}' 已保存"
        except Exception as e:
            return False, f"❌ 保存失败：{str(e)}"
    
    def load_voice(self, voice_name: str) -> Optional[dict]:
        """加载音色"""
        if voice_name in self.voices_metadata:
            meta = self.voices_metadata[voice_name]
            audio_path = Path(meta["audio_path"])
            if audio_path.exists():
                wav, sr = sf.read(str(audio_path))
                meta["wav"] = wav
                meta["sr"] = sr
                return meta
        return None
    
    def delete_voice(self, voice_name: str) -> Tuple[bool, str]:
        """删除音色"""
        try:
            if voice_name not in self.voices_metadata:
                return False, f"❌ 音色 '{voice_name}' 不存在"
            
            # 删除音色目录
            voice_dir = self.voices_dir / voice_name
            if voice_dir.exists():
                shutil.rmtree(voice_dir)
            
            # 更新元数据
            del self.voices_metadata[voice_name]
            self.save_metadata()
            
            return True, f"✅ 音色 '{voice_name}' 已删除"
        except Exception as e:
            return False, f"❌ 删除失败：{str(e)}"
    
    def get_voice_list(self) -> List[str]:
        """获取音色列表"""
        return list(self.voices_metadata.keys())
    
    def get_voice_info(self, voice_name: str) -> Optional[dict]:
        """获取音色详细信息"""
        return self.voices_metadata.get(voice_name)
    
    def download_voice(self, voice_name: str) -> Optional[str]:
        """获取音色文件路径用于下载"""
        if voice_name in self.voices_metadata:
            meta = self.voices_metadata[voice_name]
            audio_path = Path(meta["audio_path"])
            if audio_path.exists():
                return str(audio_path)
        return None
    
    def scan_existing_voices(self):
        """扫描已存在的音色目录"""
        if not self.voices_dir.exists():
            return
        
        for voice_dir in self.voices_dir.iterdir():
            if voice_dir.is_dir() and voice_dir.name != "__pycache__":
                meta_path = voice_dir / "metadata.json"
                audio_path = voice_dir / "reference.wav"
                
                if meta_path.exists() and audio_path.exists():
                    try:
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                        voice_name = voice_dir.name
                        if voice_name not in self.voices_metadata:
                            self.voices_metadata[voice_name] = meta
                            print(f"📁 发现已保存的音色：{voice_name}")
                    except Exception as e:
                        print(f"⚠️ 加载音色 {voice_dir.name} 失败：{e}")
        
        self.save_metadata()


# 全局音色存储管理器
voice_storage = VoiceStorage()

# ==================== 模型管理 ====================

class ModelManager:
    """模型加载和管理"""
    
    def __init__(self):
        self.base_model = None
        self.voice_design_model = None
        self.custom_voice_model = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
    def load_base_model(self, model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"):
        """加载 Base 模型（语音克隆）"""
        if self.base_model is None:
            from qwen_tts import Qwen3TTSModel
            print(f"正在加载 Base 模型：{model_path}")
            self.base_model = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map=self.device,
                dtype=self.dtype,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
            )
            print("✅ Base 模型加载完成")
        return self.base_model
    
    def load_voice_design_model(self, model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"):
        """加载 VoiceDesign 模型（音色设计）"""
        if self.voice_design_model is None:
            from qwen_tts import Qwen3TTSModel
            print(f"正在加载 VoiceDesign 模型：{model_path}")
            self.voice_design_model = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map=self.device,
                dtype=self.dtype,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
            )
            print("✅ VoiceDesign 模型加载完成")
        return self.voice_design_model
    
    def load_custom_voice_model(self, model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"):
        """加载 CustomVoice 模型（自定义音色）"""
        if self.custom_voice_model is None:
            from qwen_tts import Qwen3TTSModel
            print(f"正在加载 CustomVoice 模型：{model_path}")
            self.custom_voice_model = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map=self.device,
                dtype=self.dtype,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
            )
            print("✅ CustomVoice 模型加载完成")
        return self.custom_voice_model
    
    def get_supported_speakers(self) -> List[str]:
        """获取支持的音色列表"""
        if self.custom_voice_model:
            return self.custom_voice_model.get_supported_speakers()
        return ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", 
                "Ryan", "Aiden", "Ono_Anna", "Sohee"]
    
    def get_supported_languages(self) -> List[str]:
        """获取支持的语言列表"""
        if self.custom_voice_model:
            return self.custom_voice_model.get_supported_languages()
        return ["Auto", "Chinese", "English", "Japanese", "Korean", 
                "German", "French", "Russian", "Portuguese", "Spanish", "Italian"]


# 全局模型管理器
model_manager = ModelManager()

# ==================== 音频处理工具 ====================

def save_audio(wavs: np.ndarray, sr: int, prefix: str = "output") -> str:
    """保存音频文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"{prefix}_{timestamp}.wav"
    sf.write(str(output_path), wavs, sr)
    return str(output_path)


def save_batch_audio(wavs: List[np.ndarray], sr: int, prefix: str = "output") -> List[str]:
    """批量保存音频文件"""
    paths = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, wav in enumerate(wavs):
        output_path = OUTPUT_DIR / f"{prefix}_{timestamp}_{i}.wav"
        sf.write(str(output_path), wav, sr)
        paths.append(str(output_path))
    return paths


# ==================== 模式一：语音克隆 ====================

def voice_clone_fn(
    ref_audio: Optional[str],
    ref_text: str,
    syn_text: str,
    language: str,
    x_vector_only: bool = False,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """语音克隆功能"""
    try:
        progress(0.1, desc="加载模型...")
        model = model_manager.load_base_model()
        
        if not ref_audio:
            return None, "❌ 请上传参考音频文件"
        
        if not ref_text.strip():
            return None, "❌ 请填写参考音频的文本内容"
        
        if not syn_text.strip():
            return None, "❌ 请填写要合成的文本内容"
        
        progress(0.3, desc="处理音频...")
        
        # 语言处理
        lang = "Auto" if language == "Auto" else language
        
        # 生成参数
        gen_kwargs = {
            "text": syn_text,
            "language": lang,
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "x_vector_only_mode": x_vector_only,
            "max_new_tokens": 2048,
            "do_sample": True,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 0.9,
            "repetition_penalty": 1.05,
        }
        
        progress(0.6, desc="生成音频...")
        wavs, sr = model.generate_voice_clone(**gen_kwargs)
        
        progress(0.9, desc="保存音频...")
        output_path = save_audio(wavs[0] if isinstance(wavs, list) else wavs, sr, "voice_clone")
        
        progress(1.0, desc="完成!")
        return output_path, f"✅ 生成成功！时长：{len(wavs[0])/sr:.2f}秒"
        
    except Exception as e:
        return None, f"❌ 错误：{str(e)}"


def voice_clone_batch_fn(
    ref_audio: Optional[str],
    ref_text: str,
    syn_texts: str,
    language: str,
    x_vector_only: bool = False,
    progress=gr.Progress()
) -> Tuple[List[str], str]:
    """语音克隆批量功能"""
    try:
        progress(0.1, desc="加载模型...")
        model = model_manager.load_base_model()
        
        if not ref_audio:
            return [], "❌ 请上传参考音频文件"
        
        # 解析批量文本（每行一条）
        text_list = [t.strip() for t in syn_texts.strip().split('\n') if t.strip()]
        if not text_list:
            return [], "❌ 请填写要合成的文本内容"
        
        progress(0.3, desc="创建语音克隆 prompt...")
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
        )
        
        progress(0.5, desc="批量生成音频...")
        lang = "Auto" if language == "Auto" else language
        languages = [lang] * len(text_list)
        
        wavs, sr = model.generate_voice_clone(
            text=text_list,
            language=languages,
            voice_clone_prompt=prompt_items,
            max_new_tokens=2048,
        )
        
        progress(0.8, desc="保存音频...")
        output_paths = save_batch_audio(wavs, sr, "voice_clone_batch")
        
        progress(1.0, desc="完成!")
        return output_paths, f"✅ 批量生成成功！共 {len(output_paths)} 个文件"
        
    except Exception as e:
        return [], f"❌ 错误：{str(e)}"


# ==================== 模式二：音色设计 ====================

def voice_design_fn(
    text: str,
    language: str,
    instruct: str,
    voice_name: str,
    save_voice: bool = True,
    progress=gr.Progress()
) -> Tuple[str, str, gr.Dropdown]:
    """音色设计功能"""
    try:
        progress(0.1, desc="加载模型...")
        model = model_manager.load_voice_design_model()
        
        if not text.strip():
            return None, "❌ 请填写文本内容", gr.update()
        
        if not instruct.strip():
            return None, "❌ 请填写音色描述", gr.update()
        
        if not voice_name.strip():
            return None, "❌ 请填写音色名称", gr.update()
        
        progress(0.3, desc="生成音色...")
        lang = "Auto" if language == "Auto" else language
        
        wavs, sr = model.generate_voice_design(
            text=text,
            language=lang,
            instruct=instruct,
            max_new_tokens=2048,
        )
        
        progress(0.7, desc="保存音频...")
        output_path = save_audio(wavs[0] if isinstance(wavs, list) else wavs, sr, "voice_design")
        
        # 保存音色到文件
        save_msg = ""
        if save_voice:
            success, msg = voice_storage.save_voice(
                voice_name=voice_name,
                text=text,
                instruct=instruct,
                wav=wavs[0] if isinstance(wavs, list) else wavs,
                sr=sr,
                language=language,
            )
            save_msg = f" | {msg}"
        
        progress(1.0, desc="完成!")
        
        # 更新音色列表
        voice_list = voice_storage.get_voice_list()
        dropdown_update = gr.update(choices=voice_list, value=voice_name if voice_name in voice_list else None)
        
        return output_path, f"✅ 音色设计成功！时长：{len(wavs[0])/sr:.2f}秒{save_msg}", dropdown_update
        
    except Exception as e:
        return None, f"❌ 错误：{str(e)}", gr.update()


def voice_design_to_clone_fn(
    voice_name: str,
    syn_text: str,
    language: str,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """使用保存的音色进行克隆"""
    try:
        if not voice_name:
            return None, "❌ 请选择音色"
        
        voice_data = voice_storage.load_voice(voice_name)
        if not voice_data:
            return None, f"❌ 音色 '{voice_name}' 不存在或已损坏"
        
        if not syn_text.strip():
            return None, "❌ 请填写要合成的文本内容"
        
        progress(0.1, desc="加载模型...")
        model = model_manager.load_base_model()
        
        progress(0.3, desc="创建语音克隆 prompt...")
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=(voice_data["wav"], voice_data["sr"]),
            ref_text=voice_data["text"],
            x_vector_only_mode=False,
        )
        
        progress(0.6, desc="生成音频...")
        lang = "Auto" if language == "Auto" else language
        
        wavs, sr = model.generate_voice_clone(
            text=syn_text,
            language=lang,
            voice_clone_prompt=prompt_items,
            max_new_tokens=2048,
        )
        
        progress(0.9, desc="保存音频...")
        output_path = save_audio(wavs[0] if isinstance(wavs, list) else wavs, sr, f"design_clone_{voice_name}")
        
        progress(1.0, desc="完成!")
        return output_path, f"✅ 使用音色 '{voice_name}' 生成成功！时长：{len(wavs[0])/sr:.2f}秒"
        
    except Exception as e:
        return None, f"❌ 错误：{str(e)}"


def voice_design_batch_fn(
    voice_name: str,
    syn_texts: str,
    language: str,
    progress=gr.Progress()
) -> Tuple[List[str], str]:
    """使用保存的音色批量克隆"""
    try:
        if not voice_name:
            return [], "❌ 请选择音色"
        
        voice_data = voice_storage.load_voice(voice_name)
        if not voice_data:
            return [], f"❌ 音色 '{voice_name}' 不存在或已损坏"
        
        # 解析批量文本（每行一条）
        text_list = [t.strip() for t in syn_texts.strip().split('\n') if t.strip()]
        if not text_list:
            return [], "❌ 请填写要合成的文本内容"
        
        progress(0.1, desc="加载模型...")
        model = model_manager.load_base_model()
        
        progress(0.3, desc="创建语音克隆 prompt...")
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=(voice_data["wav"], voice_data["sr"]),
            ref_text=voice_data["text"],
            x_vector_only_mode=False,
        )
        
        progress(0.5, desc="批量生成音频...")
        lang = "Auto" if language == "Auto" else language
        languages = [lang] * len(text_list)
        
        wavs, sr = model.generate_voice_clone(
            text=text_list,
            language=languages,
            voice_clone_prompt=prompt_items,
            max_new_tokens=2048,
        )
        
        progress(0.8, desc="保存音频...")
        output_paths = save_batch_audio(wavs, sr, f"design_clone_batch_{voice_name}")
        
        progress(1.0, desc="完成!")
        return output_paths, f"✅ 批量生成成功！共 {len(output_paths)} 个文件"
        
    except Exception as e:
        return [], f"❌ 错误：{str(e)}"


# ==================== 音色管理功能 ====================

def refresh_voice_list() -> Tuple[gr.Dropdown, str]:
    """刷新音色列表"""
    voice_storage.scan_existing_voices()
    voice_list = voice_storage.get_voice_list()
    msg = f"✅ 已刷新，共 {len(voice_list)} 个音色" if voice_list else "ℹ️ 暂无已保存的音色"
    return gr.update(choices=voice_list), msg


def get_voice_details(voice_name: str) -> str:
    """获取音色详细信息"""
    if not voice_name:
        return "请选择一个音色"
    
    info = voice_storage.get_voice_info(voice_name)
    if not info:
        return f"❌ 音色 '{voice_name}' 不存在"
    
    details = f"""
    ### 📊 音色信息
    
    - **名称**: {info.get('name', 'N/A')}
    - **语言**: {info.get('language', 'N/A')}
    - **时长**: {info.get('duration', 0):.2f} 秒
    - **创建时间**: {info.get('created_at', 'N/A')}
    - **示例文本**: {info.get('text', 'N/A')[:100]}...
    - **音色描述**: {info.get('instruct', 'N/A')[:100]}...
    """
    return details


def delete_voice_fn(voice_name: str) -> Tuple[gr.Dropdown, str]:
    """删除音色"""
    if not voice_name:
        return gr.update(), "❌ 请选择要删除的音色"
    
    success, msg = voice_storage.delete_voice(voice_name)
    voice_list = voice_storage.get_voice_list()
    
    if success:
        return gr.update(choices=voice_list, value=None), msg
    else:
        return gr.update(), msg


def download_voice_fn(voice_name: str) -> Tuple[Optional[str], str]:
    """下载音色"""
    if not voice_name:
        return None, "❌ 请选择要下载的音色"
    
    audio_path = voice_storage.download_voice(voice_name)
    if audio_path:
        return audio_path, f"✅ 音色 '{voice_name}' 已准备好下载"
    else:
        return None, f"❌ 音色 '{voice_name}' 不存在"


# ==================== 模式三：自定义音色 ====================

def custom_voice_fn(
    text: str,
    language: str,
    speaker: str,
    instruct: str,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """自定义音色生成功能"""
    try:
        progress(0.1, desc="加载模型...")
        model = model_manager.load_custom_voice_model()
        
        if not text.strip():
            return None, "❌ 请填写文本内容"
        
        progress(0.3, desc="生成音频...")
        lang = "Auto" if language == "Auto" else language
        
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=lang,
            speaker=speaker,
            instruct=instruct if instruct.strip() else None,
            max_new_tokens=2048,
        )
        
        progress(0.8, desc="保存音频...")
        output_path = save_audio(wavs[0] if isinstance(wavs, list) else wavs, sr, f"custom_{speaker}")
        
        progress(1.0, desc="完成!")
        return output_path, f"✅ 使用音色 '{speaker}' 生成成功！时长：{len(wavs[0])/sr:.2f}秒"
        
    except Exception as e:
        return None, f"❌ 错误：{str(e)}"


def custom_voice_batch_fn(
    texts: str,
    language: str,
    speaker: str,
    instruct: str,
    progress=gr.Progress()
) -> Tuple[List[str], str]:
    """自定义音色批量生成功能"""
    try:
        progress(0.1, desc="加载模型...")
        model = model_manager.load_custom_voice_model()
        
        # 解析批量文本（每行一条）
        text_list = [t.strip() for t in texts.strip().split('\n') if t.strip()]
        if not text_list:
            return [], "❌ 请填写文本内容"
        
        progress(0.3, desc="批量生成音频...")
        lang = "Auto" if language == "Auto" else language
        languages = [lang] * len(text_list)
        speakers = [speaker] * len(text_list)
        instructs = [instruct if instruct.strip() else "" for _ in range(len(text_list))]
        
        wavs, sr = model.generate_custom_voice(
            text=text_list,
            language=languages,
            speaker=speakers,
            instruct=instructs,
            max_new_tokens=2048,
        )
        
        progress(0.8, desc="保存音频...")
        output_paths = save_batch_audio(wavs, sr, f"custom_batch_{speaker}")
        
        progress(1.0, desc="完成!")
        return output_paths, f"✅ 批量生成成功！共 {len(output_paths)} 个文件"
        
    except Exception as e:
        return [], f"❌ 错误：{str(e)}"


# ==================== WebUI 界面 ====================

def create_webui():
    """创建 WebUI 界面"""
    
    # 自定义 CSS
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
    }
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 30px;
    }
    .mode-card {
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .success-msg {
        color: #28a745;
        font-weight: bold;
    }
    .error-msg {
        color: #dc3545;
        font-weight: bold;
    }
    .audio-player {
        border-radius: 10px;
        margin: 10px 0;
    }
    .btn-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        font-weight: bold;
    }
    .btn-primary:hover {
        opacity: 0.9;
    }
    .voice-info-box {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(title="Qwen3-TTS 有声小说配音", css=custom_css, theme=gr.themes.Soft()) as demo:
        
        # 标题
        gr.Markdown("""
        # 🎙️ Qwen3-TTS 有声小说配音工作室
        ### 专业的 AI 语音生成工具 | 支持语音克隆、音色设计、自定义音色
        """)
        
        with gr.Tabs() as tabs:
            
            # ==================== 标签页 1：语音克隆 ====================
            with gr.TabItem("🎤 语音克隆", id="tab_clone"):
                gr.Markdown("""
                ### 📋 使用说明
                1. 上传一段参考音频（建议 3-10 秒，清晰的人声）
                2. 填写参考音频的文本内容
                3. 输入要合成的文本内容
                4. 点击生成按钮
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📁 参考音频设置")
                        ref_audio_input = gr.Audio(
                            label="参考音频",
                            type="filepath",
                            sources=["upload", "microphone"],
                        )
                        ref_text_input = gr.Textbox(
                            label="参考音频文本",
                            placeholder="请输入参考音频中的文字内容...",
                            lines=3,
                        )
                        x_vector_only = gr.Checkbox(
                            label="仅使用音色向量（快速但质量略低）",
                            value=False,
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📝 合成设置")
                        syn_text_input = gr.Textbox(
                            label="要合成的文本",
                            placeholder="请输入要合成的文字内容...",
                            lines=3,
                        )
                        clone_language = gr.Dropdown(
                            choices=model_manager.get_supported_languages(),
                            value="Auto",
                            label="语言",
                        )
                        
                        clone_btn = gr.Button(
                            "🚀 生成语音",
                            variant="primary",
                            size="lg",
                        )
                
                with gr.Row():
                    clone_output_audio = gr.Audio(
                        label="生成结果",
                        type="filepath",
                    )
                    clone_status = gr.Textbox(
                        label="状态",
                        interactive=False,
                    )
                
                # 批量克隆
                with gr.Accordion("📦 批量克隆", open=False):
                    batch_syn_texts = gr.Textbox(
                        label="批量文本（每行一条）",
                        placeholder="第一句文本...\n第二句文本...\n第三句文本...",
                        lines=5,
                    )
                    clone_batch_btn = gr.Button("🚀 批量生成", variant="primary")
                    clone_batch_output = gr.File(
                        label="批量生成结果",
                        file_count="multiple",
                    )
                    clone_batch_status = gr.Textbox(label="批量状态", interactive=False)
                
                # 绑定事件
                clone_btn.click(
                    fn=voice_clone_fn,
                    inputs=[ref_audio_input, ref_text_input, syn_text_input, 
                           clone_language, x_vector_only],
                    outputs=[clone_output_audio, clone_status],
                )
                
                clone_batch_btn.click(
                    fn=voice_clone_batch_fn,
                    inputs=[ref_audio_input, ref_text_input, batch_syn_texts, 
                           clone_language, x_vector_only],
                    outputs=[clone_batch_output, clone_batch_status],
                )
            
            # ==================== 标签页 2：音色设计 ====================
            with gr.TabItem("🎨 音色设计", id="tab_design"):
                gr.Markdown("""
                ### 📋 使用说明
                1. 填写一段示例文本
                2. 用自然语言描述你想要的音色（如：温柔的少女音、沉稳的男声等）
                3. 给音色起个名字保存
                4. 可以使用保存的音色进行克隆
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🎨 音色设计")
                        design_text_input = gr.Textbox(
                            label="示例文本",
                            placeholder="请输入示例文本...",
                            lines=3,
                            value="哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
                        )
                        design_instruct_input = gr.Textbox(
                            label="音色描述",
                            placeholder="请用自然语言描述你想要的音色...",
                            lines=4,
                            value="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
                        )
                        design_language = gr.Dropdown(
                            choices=model_manager.get_supported_languages(),
                            value="Chinese",
                            label="语言",
                        )
                        design_voice_name = gr.Textbox(
                            label="音色名称",
                            placeholder="给这个音色起个名字...",
                            value="my_voice_001",
                        )
                        save_voice_checkbox = gr.Checkbox(
                            label="保存到音色库",
                            value=True,
                        )
                        
                        design_btn = gr.Button(
                            "🎨 设计音色",
                            variant="primary",
                            size="lg",
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🎤 使用保存的音色克隆")
                        saved_voices_dropdown = gr.Dropdown(
                            choices=voice_storage.get_voice_list(),
                            label="已保存的音色",
                            interactive=True,
                        )
                        refresh_voice_btn = gr.Button("🔄 刷新音色列表", size="sm")
                        clone_from_design_text = gr.Textbox(
                            label="要合成的文本",
                            placeholder="请输入要合成的文字内容...",
                            lines=3,
                        )
                        design_clone_language = gr.Dropdown(
                            choices=model_manager.get_supported_languages(),
                            value="Auto",
                            label="语言",
                        )
                        
                        design_clone_btn = gr.Button(
                            "🚀 使用音色克隆",
                            variant="primary",
                            size="lg",
                        )
                
                with gr.Row():
                    design_output_audio = gr.Audio(
                        label="设计的音色",
                        type="filepath",
                    )
                    design_status = gr.Textbox(
                        label="状态",
                        interactive=False,
                    )
                
                with gr.Row():
                    design_clone_output_audio = gr.Audio(
                        label="克隆结果",
                        type="filepath",
                    )
                    design_clone_status = gr.Textbox(
                        label="状态",
                        interactive=False,
                    )
                
                # 绑定事件
                design_btn.click(
                    fn=voice_design_fn,
                    inputs=[design_text_input, design_language, design_instruct_input, 
                           design_voice_name, save_voice_checkbox],
                    outputs=[design_output_audio, design_status, saved_voices_dropdown],
                )
                
                design_clone_btn.click(
                    fn=voice_design_to_clone_fn,
                    inputs=[saved_voices_dropdown, clone_from_design_text, design_clone_language],
                    outputs=[design_clone_output_audio, design_clone_status],
                )
                
                refresh_voice_btn.click(
                    fn=refresh_voice_list,
                    inputs=[],
                    outputs=[saved_voices_dropdown, design_status],
                )
                
                # 音色管理区域
                with gr.Accordion("📂 音色管理", open=False):
                    gr.Markdown("### 管理已保存的音色")
                    
                    with gr.Row():
                        manage_voice_dropdown = gr.Dropdown(
                            choices=voice_storage.get_voice_list(),
                            label="选择音色",
                            interactive=True,
                        )
                        manage_refresh_btn = gr.Button("🔄 刷新", size="sm")
                    
                    voice_info_display = gr.Textbox(
                        label="音色详情",
                        interactive=False,
                        lines=8,
                    )
                    
                    with gr.Row():
                        download_voice_btn = gr.Button("📥 下载音色", variant="secondary")
                        delete_voice_btn = gr.Button("🗑️ 删除音色", variant="stop")
                    
                    download_voice_output = gr.File(
                        label="下载",
                        visible=False,
                    )
                    manage_status = gr.Textbox(
                        label="状态",
                        interactive=False,
                    )
                    
                    # 绑定事件
                    manage_voice_dropdown.change(
                        fn=get_voice_details,
                        inputs=[manage_voice_dropdown],
                        outputs=[voice_info_display],
                    )
                    
                    manage_refresh_btn.click(
                        fn=refresh_voice_list,
                        inputs=[],
                        outputs=[manage_voice_dropdown, manage_status],
                    )
                    
                    download_voice_btn.click(
                        fn=download_voice_fn,
                        inputs=[manage_voice_dropdown],
                        outputs=[download_voice_output, manage_status],
                    )
                    
                    delete_voice_btn.click(
                        fn=delete_voice_fn,
                        inputs=[manage_voice_dropdown],
                        outputs=[manage_voice_dropdown, manage_status],
                    )
                
                # 批量克隆
                with gr.Accordion("📦 批量克隆", open=False):
                    design_batch_syn_texts = gr.Textbox(
                        label="批量文本（每行一条）",
                        placeholder="第一句文本...\n第二句文本...\n第三句文本...",
                        lines=5,
                    )
                    design_batch_btn = gr.Button("🚀 批量生成", variant="primary")
                    design_batch_output = gr.File(
                        label="批量生成结果",
                        file_count="multiple",
                    )
                    design_batch_status = gr.Textbox(label="批量状态", interactive=False)
                    
                    design_batch_btn.click(
                        fn=voice_design_batch_fn,
                        inputs=[saved_voices_dropdown, design_batch_syn_texts, design_clone_language],
                        outputs=[design_batch_output, design_batch_status],
                    )
            
            # ==================== 标签页 3：自定义音色 ====================
            with gr.TabItem("👤 自定义音色", id="tab_custom"):
                gr.Markdown("""
                ### 📋 使用说明
                1. 选择内置的预设音色
                2. 输入要合成的文本
                3. 可选：添加语气指令（如：用开心的语气说）
                4. 点击生成按钮
                """)
                speaker = model_manager.get_supported_speakers()
                # 音色说明
                speaker_info = {
                    "Vivian": "明亮、略带个性的年轻女声（中文）",
                    "Serena": "温暖、温柔的年轻女声（中文）",
                    "Uncle_Fu": "成熟男性低音，醇厚音色（中文）",
                    "Dylan": "年轻的北京男声，清晰自然（中文 - 北京方言）",
                    "Eric": "活泼的成都男声，略带沙哑的明亮感（中文 - 四川方言）",
                    "Ryan": "充满活力的男声，节奏感强（英文）",
                    "Aiden": "阳光的美式男声，中频清晰（英文）",
                    "Ono_Anna": "俏皮的日本女声，轻盈灵活（日文）",
                    "Sohee": "温暖的韩国女声，情感丰富（韩文）",
                }
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🎤 音色选择")
                        speaker_dropdown = gr.Dropdown(
                            choices=speaker,
                            value=speaker[0],
                            label="选择音色",
                        )
                        speaker_info_display = gr.Textbox(
                            label="音色说明",
                            value=speaker_info[speaker[0]],
                            interactive=False,
                        )
                        
                        def update_speaker_info(speaker):
                            return speaker_info.get(speaker, "")
                        
                        speaker_dropdown.change(
                            fn=update_speaker_info,
                            inputs=[speaker_dropdown],
                            outputs=[speaker_info_display],
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📝 合成设置")
                        custom_text_input = gr.Textbox(
                            label="要合成的文本",
                            placeholder="请输入要合成的文字内容...",
                            lines=3,
                        )
                        custom_language = gr.Dropdown(
                            choices=model_manager.get_supported_languages(),
                            value="Auto",
                            label="语言",
                        )
                        custom_instruct_input = gr.Textbox(
                            label="语气指令（可选）",
                            placeholder="如：用开心的语气说、用愤怒的语气说...",
                            lines=2,
                        )
                        
                        custom_btn = gr.Button(
                            "🚀 生成语音",
                            variant="primary",
                            size="lg",
                        )
                
                with gr.Row():
                    custom_output_audio = gr.Audio(
                        label="生成结果",
                        type="filepath",
                    )
                    custom_status = gr.Textbox(
                        label="状态",
                        interactive=False,
                    )
                
                # 批量生成
                with gr.Accordion("📦 批量生成", open=False):
                    batch_custom_texts = gr.Textbox(
                        label="批量文本（每行一条）",
                        placeholder="第一句文本...\n第二句文本...\n第三句文本...",
                        lines=5,
                    )
                    custom_batch_btn = gr.Button("🚀 批量生成", variant="primary")
                    custom_batch_output = gr.File(
                        label="批量生成结果",
                        file_count="multiple",
                    )
                    custom_batch_status = gr.Textbox(label="批量状态", interactive=False)
                
                # 绑定事件
                custom_btn.click(
                    fn=custom_voice_fn,
                    inputs=[custom_text_input, custom_language, speaker_dropdown, custom_instruct_input],
                    outputs=[custom_output_audio, custom_status],
                )
                
                custom_batch_btn.click(
                    fn=custom_voice_batch_fn,
                    inputs=[batch_custom_texts, custom_language, speaker_dropdown, custom_instruct_input],
                    outputs=[custom_batch_output, custom_batch_status],
                )
            
            # ==================== 标签页 4：使用说明 ====================
            with gr.TabItem("📖 使用说明", id="tab_help"):
                gr.Markdown("""
                ## 🎙️ Qwen3-TTS 有声小说配音工作室
                
                ### 功能介绍
                
                #### 1️⃣ 语音克隆模式
                - 上传参考音频，克隆说话人的音色
                - 支持批量生成多条文本
                - 适合需要固定角色声音的有声书
                
                #### 2️⃣ 音色设计模式
                - 通过自然语言描述设计独特音色
                - **设计的音色会保存到本地文件，重启后仍然可用**
                - 可以传输给 Base 模型进行批量克隆
                - 适合创建独特的角色声音
                
                #### 3️⃣ 自定义音色模式
                - 使用 9 种内置高质量音色
                - 支持多语言和方言
                - 可添加语气指令控制情感表达
                - 适合快速生成标准配音
                
                ### 💡 最佳实践
                
                1. **参考音频质量**：建议使用 3-10 秒清晰、无背景噪音的人声
                2. **文本分段**：长文本建议分段处理，每段不超过 200 字
                3. **音色设计**：描述越详细，生成的音色越符合预期
                4. **批量处理**：使用批量功能可以提高效率，保持音色一致性
                5. **音色保存**：设计的音色会自动保存到 `saved_voices` 文件夹
                
                ### 🔧 技术信息
                
                - 模型：Qwen3-TTS-12Hz-1.7B 系列
                - 支持语言：中文、英文、日文、韩文、德文、法文、俄文、葡萄牙文、西班牙文、意大利文
                - 流式生成：支持超低延迟流式输出
                - 音色控制：支持自然语言指令控制语调、情感、语速
                - 音色存储：`saved_voices/` 文件夹，每个音色一个子目录
                
                ### 📁 文件结构
                
                ```
                saved_voices/
                ├── voices_metadata.json    # 音色元数据
                ├── my_voice_001/
                │   ├── reference.wav       # 参考音频
                │   └── metadata.json       # 音色信息
                └── my_voice_002/
                    ├── reference.wav
                    └── metadata.json
                ```
                
                ### 📞 支持
                
                - [GitHub](https://github.com/QwenLM/Qwen3-TTS)
                - [Hugging Face](https://huggingface.co/collections/Qwen/qwen3-tts)
                - [ModelScope](https://modelscope.cn/collections/Qwen/Qwen3-TTS)
                """)
        
        # 页脚
        gr.Markdown("""
        ---
        <div style="text-align: center; color: #666;">
        <p>🎙️ Qwen3-TTS 有声小说配音工作室 | Powered by Qwen Team</p>
        <p>基于 Qwen3-TTS-12Hz-1.7B 系列模型构建</p>
        </div>
        """)
    
    return demo


# ==================== 主程序 ====================

if __name__ == "__main__":
    # 扫描已存在的音色
    print("🔍 扫描已保存的音色...")
    voice_storage.scan_existing_voices()
    
    # 创建并启动 WebUI
    demo = create_webui()
    
    # 启动服务
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )