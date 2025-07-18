import os
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

# 環境変数（APIキー）読み込み
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# LLMのセットアップ
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="gpt-3.5-turbo",
    temperature=0.7
)

# Streamlit UI
st.title("💡専門家に相談できるLLMアプリ")
st.write("このアプリでは、専門家の視点からアドバイスをもらえます。")
st.write("下記の専門家を選び、相談内容を入力してください。")

# ラジオボタンで専門家選択
expert = st.radio(
    "相談する専門家を選択してください：",
    ("フィットネストレーナー", "キャリアアドバイザー", "メンタルヘルスカウンセラー")
)

# 入力欄
user_input = st.text_input("相談内容を入力してください：")

# 実行ボタン
if st.button("相談する"):
    if not user_input:
        st.warning("相談内容を入力してください。")
    else:
        # 専門家ごとのSystem Prompt
        system_prompt = {
            "フィットネストレーナー": "あなたはプロのフィットネストレーナーです。健康や運動に関する相談に専門的に答えてください。",
            "キャリアアドバイザー": "あなたは経験豊富なキャリアアドバイザーです。就職や転職の悩みに的確なアドバイスをしてください。",
            "メンタルヘルスカウンセラー": "あなたは信頼できるメンタルヘルスカウンセラーです。心の悩みに寄り添って助言をしてください。"
        }

        # LangChain用プロンプトテンプレート
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt[expert]),
            ("human", "{question}")
        ])

        # Chain化
        chain = LLMChain(llm=llm, prompt=prompt)

        # 実行
        with st.spinner("考え中..."):
            response = chain.run({"question": user_input})
        st.success("回答が届きました：")
        st.write(response)
