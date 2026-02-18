import streamlit as st
import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
import io

# アプリの基本設定
st.set_page_config(page_title="グアテマラ農村データ増幅器", layout="wide")

st.title("🇬🇹 グアテマラ農村データ・ジェネレーター")
st.markdown("""
### 概要
100件程度のスモールデータから、統計的特徴を維持した合成データを生成します。
本アプリは**Gaussian Copula（ガウス・コピュラ）**技法を用いて、変数間の相関関係を数学的に再現します。
""")

# サイドバー：設定
st.sidebar.header("生成設定")
num_samples = st.sidebar.number_input("生成するデータ件数", min_value=100, max_value=5000, value=1000)

# ファイルアップローダー
uploaded_file = st.file_uploader("元となるデータ（ExcelまたはCSV）をアップロードしてください", type=["csv", "xlsx"])

if uploaded_file:
    # データの読み込み
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📊 元データの確認 (Top 5)")
    st.write(df.head())

    # 合成データの生成ボタン
    if st.button("合成データの生成"):
        with st.spinner('Gaussian Copula モデルで学習・生成中...'):
            try:
                # 1. SDVのメタデータ自動検出
                metadata = Metadata.detect_from_dataframe(data=df)
                
                # 2. Gaussian Copula シンセサイザーの訓練
                synthesizer = GaussianCopulaSynthesizer(metadata)
                synthesizer.fit(df)
                
                # 3. データ生成
                synthetic_df = synthesizer.sample(num_rows=num_samples)
                
                # 4. データクレンジング（PC負荷軽減・安定化のため）
                # 小数点以下の精度を5桁に丸め、異常値を防ぐ
                synthetic_df = synthetic_df.round(5)
                # 欠損値の適切な処理
                synthetic_df = synthetic_df.fillna('')

                st.success(f"✅ {num_samples}件の合成データ生成が完了しました！")
                
                st.subheader("✨ 生成されたデータ (Top 5)")
                st.write(synthetic_df.head())

                # 5. CSVダウンロードの準備
                # utf-8-sig を使うことで、Excelで開いても文字化けせず、構造も安定します
                csv_data = synthetic_df.to_csv(index=False).encode('utf-8-sig')
                
                st.download_button(
                    label="生成されたCSVを保存",
                    data=csv_data,
                    file_name="synthetic_data.csv",
                    mime="text/csv"
                )

                # --- 統計比較 ---
                st.divider()
                st.subheader("📈 統計的妥当性のチェック (平均値の比較)")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("【元データ】")
                    st.write(df.describe().loc['mean'])
                with col2:
                    st.write("【合成データ】")
                    st.write(synthetic_df.describe().loc['mean'])

            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
                st.info("ヒント: 列に特殊な記号が含まれていないか、空の行がないか確認してください。")