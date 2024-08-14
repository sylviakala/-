from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from template import STAGE_ANALYZER_INCEPTION_PROMPT,STAGE_1_TEMPLATE,STAGE_2_TEMPLATE,STAGE_3_TEMPLATE,SUMMARY_TEMPLATE

class StageAnalyzerChain(LLMChain):
    """通过查看聊天记录判断是否要转换阶段."""

    @classmethod
    def from_llm(cls, llm, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        stage_analyzer_inception_prompt_template = STAGE_ANALYZER_INCEPTION_PROMPT
        prompt = PromptTemplate(
            template=stage_analyzer_inception_prompt_template,
            input_variables=[
                "conversation_history",
                "question"
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

class ConversationChain_stage_1(LLMChain):
    """破冰阶段，接收提示作为输入，生成 AI 消息"""

    @classmethod
    def from_llm(cls, llm, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        conversation_stage_1_template = STAGE_1_TEMPLATE
        prompt = PromptTemplate(
            template=conversation_stage_1_template,
            input_variables=[
                "conversation_history",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
    
class ConversationChain_stage_2(LLMChain):
    """探索阶段，接收提示作为输入，生成 AI 消息"""

    @classmethod
    def from_llm(cls, llm, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        conversation_stage_2_template = STAGE_2_TEMPLATE
        prompt = PromptTemplate(
            template=conversation_stage_2_template,
            input_variables=[
                "conversation_history",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
    
class ConversationChain_stage_3(LLMChain):
    """总结阶段，接收提示作为输入，生成 AI 消息"""

    @classmethod
    def from_llm(cls, llm, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        conversation_stage_3_template = STAGE_3_TEMPLATE
        prompt = PromptTemplate(
            template=conversation_stage_3_template,
            input_variables=[
                "conversation_history",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose) 
    
    
    class ConversationSummaryChain(LLMChain):
        '''总结用户历史消息'''

    @classmethod
    def from_llm(cls, llm, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        conversation_summary_template = SUMMARY_TEMPLATE
        prompt = PromptTemplate(
            template=conversation_summary_template,
            input_variables=[
                "conversation_history",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
