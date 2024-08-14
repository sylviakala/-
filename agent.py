from typing import Any, Callable, Dict, List, Union
import streamlit as st
import os

from pydantic import Field
from langchain_openai import OpenAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.agents import LLMSingleActionAgent,AgentExecutor

from chain import StageAnalyzerChain,ConversationChain_stage_1,ConversationChain_stage_2,ConversationChain_stage_3
from stage import CONVERSATION_STAGES
from template import *

class ConversationAgent():
    stage_analyzer_chain: StageAnalyzerChain = Field(...)
    conversation_agent = Field()
   

    conversation_history = []
    conversation_stage_id: str = "1"
    current_conversation_stage: str = CONVERSATION_STAGES.get("1")

    llm = OpenAI(
            temperature= 0.7,
            #openai_api_key=st.secrets['api']['key'],
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            #base_url=st.secrets['api']['base_url']，
         )

    '''清空对话历史，初始化代理'''
    def seed_agent(self):
        self.conversation_history.clear()
        print("——Seed Successful——")

    '''返回对话历史'''
    def show_chat_history(self):
        return self.conversation_history

    '''根据传入的 key 获取对应的对话阶段描述'''
    def retrieve_conversation_stage(self, key):
        return CONVERSATION_STAGES.get(key)

    '''测试输出最后一条历史记录'''
    def fake_step(self):
        input_text = self.conversation_history[-1]
        ai_message = self._respond_with_tools(str(input_text), verbose=True)
        print(ai_message,type(ai_message['output']))

    '''如果在阶段x,调用_respond_stage_x()'''
    def step(self):
        input_text = self.conversation_history[-1]
        print(str(input_text)+'input_text****')

        # if int(self.conversation_stage_id) == 1:
        #     ai_message = self._respond_without_tools(str(input_text),verbose=True)
        # else:
        #     chat_message = self._respond_without_tools(str(input_text), verbose=True)
        #     recommend_message = self.recommend_product()
        #     print(recommend_message,len(recommend_message))
        #     if len(recommend_message)<=5:
        #         ai_message = chat_message
        #     else:
        #         ai_message = chat_message + '\n\n' + recommend_message

            # output_dic = self._respond_with_tools(str(input_text),verbose=True)
            # ai_message = str(output_dic['output'])

        if int(self.conversation_stage_id) == 1:
            ai_message = self._respond_stage_1(str(input_text),verbose=True)
        elif int(self.conversation_stage_id) == 2:
            ai_message = self._respond_stage_2(str(input_text),verbose=True)
        else :
            ai_message = self._respond_stage_3(str(input_text),verbose=True)


        print(ai_message,type(ai_message))

        ai_message = "AI:"+str(ai_message)
        self.conversation_history.append(ai_message)
        # print(f"——系统返回消息'{ai_message}'，并添加到history里——")
        return ai_message.lstrip('AI:')

    def human_step(self,input_text):
        human_message = input_text
        human_message = "用户: " + human_message
        self.conversation_history.append(human_message)
        # print(f"——用户输入消息'{human_message}'，并添加到history里——")
        return human_message

    def generate_stage_analyzer(self,verbose: bool = False):
        self.stage_analyzer_chain = StageAnalyzerChain.from_llm(
            llm=self.llm,
            verbose=verbose
        )

        print("成功构造一个StageAnalyzerChain")


    def determine_conversation_stage(self,question):
        self.question = question
        print('-----进入阶段判断方法-----')
        self.conversation_stage_id = self.stage_analyzer_chain.run(
            conversation_history=self.conversation_history,
            question=self.question
        )

        print(f"Conversation Stage ID: {self.conversation_stage_id}")
        print(type(self.conversation_stage_id))
        self.current_conversation_stage = self.retrieve_conversation_stage(
            self.conversation_stage_id
        )
        print(f"Conversation Stage: {self.current_conversation_stage}")

    # def _respond_without_tools(self,input_text,verbose: bool = False):
    #     self.conversation_agent_without_tool = ConversationChain.from_llm(
    #         llm=self.llm,
    #         verbose=verbose
    #     )

    #     response = self.conversation_agent_without_tool.run(
    #         question = input_text,
    #         conversation_history=self.conversation_history,
    #     )

    def _respond_stage_1(self,input_text,verbose: bool = False):
        self.conversation_agent = ConversationChain_stage_1.from_llm(
            llm=self.llm,
            verbose=verbose
        )

        response = self.conversation_agent.run(
            question = input_text,
            conversation_history=self.conversation_history,
        )
        return response


    def _respond_stage_2(self,input_text,verbose: bool = False):
        self.conversation_agent = ConversationChain_stage_2.from_llm(
            llm=self.llm,
            verbose=verbose
        )

        response = self.conversation_agent.run(
            question = input_text,
            conversation_history=self.conversation_history,
        )
        return response
    

    def _respond_stage_3(self,input_text,verbose: bool = False):
        self.conversation_agent = ConversationChain_stage_3.from_llm(
            llm=self.llm,
            verbose=verbose
        )

        response = self.conversation_agent.run(
            question = input_text,
            conversation_history=self.conversation_history,
        )
        return response
