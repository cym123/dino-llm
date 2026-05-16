# 导入工具包
from dataclasses import dataclass  # 用来定义数据结构（简单的小盒子）
from typing import Dict, List       # 类型提示，告诉变量是什么格式（字典、列表）

from dinollm.message import DetokenizeMsg  # 模型传过来的“解码消息”
from transformers import PreTrainedTokenizerBase  # 分词器基类（模型专用）

# ==================== 工具函数1：判断是不是中文字符 ====================
# 作用：检查一个字符是不是中文（用来处理中文流式输出不卡顿）
def _is_chinese_char(cp: int):
    """Checks whether CP is the codepoint of a CJK character."""
    # 下面一大串数字 = 中文在电脑里的编码范围
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)
        or (cp >= 0x20000 and cp <= 0x2A6DF)
        or (cp >= 0x2A700 and cp <= 0x2B73F)
        or (cp >= 0x2B740 and cp <= 0x2B81F)
        or (cp >= 0x2B820 and cp <= 0x2CEAF)
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)
    ):
        return True  # 是中文
    return False     # 不是中文

# ==================== 工具函数2：找到能正常显示的文字 ====================
# 作用：流式输出时，保证不输出半截乱码、不输出半截单词
def find_printable_text(text: str):
    """Returns the longest printable substring of text that contains only entire words."""
    
    # 如果结尾是换行，直接返回
    if text.endswith("\n"):
        return text
    # 如果最后一个字符是中文，直接返回（中文不用等空格）
    elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text
    # 如果倒数第二个是中文，返回除最后一个外的所有
    elif len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]
    # 英文：只返回到最后一个空格，保证不输出半截单词
    else:
        return text[: text.rfind(" ") + 1]

# ==================== 数据结构：每个用户的解码状态 ====================
# 作用：给每个用户存“已经解码了什么、输出到哪了”
@dataclass
class DecodeStatus:
    decoded_ids: List[int]    # 已经解码的 token ID（模型输出的数字）
    decoded_str: str         # 已经解码好的完整字符串
    read_offset: int         # 已经读到哪个位置了
    surr_offset: int         # 辅助解码的偏移
    sent_offset: int         # 已经发送给前端的字符串长度

# ==================== 核心类：DetokenizeManager（解码管理器） ====================
# 作用：把模型输出的【数字ID】 → 变成【人类能看懂的文字】
# 并且支持**流式一段一段输出**
class DetokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # 每个用户一个独立的解码状态（uid → 解码状态）
        self.decode_map: Dict[int, DecodeStatus] = {}
        self.tokenizer = tokenizer              # 模型的分词器
        self.eos_token_id = self.tokenizer.eos_token_id  # 结束标记ID

    # 核心方法：把模型输出的 token 解码成文字
    def detokenize(self, msgs: List[DetokenizeMsg]) -> List[str]:
        read_ids: List[List[int]] = []
        surr_ids: List[List[int]] = []

        # 遍历每个用户的消息
        for msg in msgs:
            # 如果用户第一次来，新建一个解码状态
            if msg.uid not in self.decode_map:
                self.decode_map[msg.uid] = DecodeStatus(
                    decoded_ids=[],
                    decoded_str="",
                    read_offset=0,
                    surr_offset=0,
                    sent_offset=0,
                )
            s = self.decode_map[msg.uid]  # 拿到这个用户的状态

            # 不是结束的话，把新 token 存起来
            if not (msg.finished and msg.next_token == self.eos_token_id):
                s.decoded_ids.append(msg.next_token)

            # 记录需要解码的区间
            read_ids.append(s.decoded_ids[s.surr_offset :])
            surr_ids.append(s.decoded_ids[s.surr_offset : s.read_offset])

        # 批量解码：数字 → 文字
        read_texts = self.tokenizer.batch_decode(read_ids)
        surr_texts = self.tokenizer.batch_decode(surr_ids)

        incremental_strs: List[str] = []  # 最终要返回的增量文字

        # 遍历每个用户的结果
        for msg, read_str, surr_str in zip(msgs, read_texts, surr_texts, strict=True):
            s = self.decode_map[msg.uid]
            new_text = read_str[len(surr_str) :]  # 拿到新增的文字

            # print("原始解码文本 new_text :", repr(new_text))


            # 如果新增文字正常，没有乱码
            if len(new_text) > 0 and not new_text.endswith("�"):
                output_str = s.decoded_str + new_text
                s.decoded_str = output_str
                s.surr_offset = s.read_offset
                s.read_offset = len(s.decoded_ids)
            else:
                # 有乱码，只输出能正常显示的部分
                new_text = find_printable_text(new_text)
                output_str = s.decoded_str + new_text

            # 计算这次要输出的【增量文字】（流式输出关键！）
            incremental_output = output_str[s.sent_offset :]
            s.sent_offset = len(output_str)  # 更新已发送长度
            incremental_strs.append(incremental_output)

            # 如果结束了，删除用户状态
            if msg.finished:
                del self.decode_map[msg.uid]

        return incremental_strs  # 返回给前端的增量文字