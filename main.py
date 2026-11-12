
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import zipfile
import io

st.set_page_config(
    page_title="AI 로봇 장애물 회피 예측기",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI 로봇 장애물 회피 예측기")
st.write("초음파 센서 데이터를 활용하여 로봇의 이동 방향을 분석하는 프로젝트입니다.")

DATA_URL = "https://archive.ics.uci.edu/static/public/194/wall+following+robot+navigation+data.zip"

@st.cache_data
def load_data():
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        data_file = next(
            name for name in z.namelist()
            if name.endswith("sensor_readings_24.data")
        )
        with z.open(data_file) as f:
            df = pd.read_csv(f, header=None, skipinitialspace=True)

    columns = [f"US{i}" for i in range(1, 25)] + ["Class"]
    df.columns = columns

    return df


try:
    df = load_data()

    st.success("데이터를 성공적으로 불러왔습니다!")

    st.header("1. 데이터 기본 정보")

    col1, col2 = st.columns(2)
    col1.metric("전체 데이터 개수", f"{len(df):,}개")
    col2.metric("전체 속성 개수", f"{len(df.columns)}개")

    st.subheader("데이터 미리보기")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("결측값 확인")
    missing = df.isnull().sum()
    st.dataframe(
        missing.rename("결측값 개수"),
        use_container_width=True
    )

    st.subheader("이동 방향별 데이터 개수")
    st.dataframe(
        df["Class"].value_counts().rename("데이터 개수"),
        use_container_width=True
    )

    st.divider()

    st.header("2. 로봇의 이동 방향별 데이터 분포")

    direction_names = {
        "Move-Forward": "직진",
        "Slight-Right-Turn": "약간 우회전",
        "Sharp-Right-Turn": "급격한 우회전",
        "Slight-Left-Turn": "약간 좌회전"
    }

    direction_counts = (
        df["Class"]
        .value_counts()
        .rename_axis("이동 방향")
        .reset_index(name="데이터 개수")
    )

    direction_counts["이동 방향"] = (
        direction_counts["이동 방향"]
        .map(direction_names)
        .fillna(direction_counts["이동 방향"])
    )

    order = ["직진", "약간 우회전", "급격한 우회전", "약간 좌회전"]
    direction_counts["이동 방향"] = pd.Categorical(
        direction_counts["이동 방향"],
        categories=order,
        ordered=True
    )
    direction_counts = direction_counts.sort_values("이동 방향")

    fig = px.bar(
        direction_counts,
        x="이동 방향",
        y="데이터 개수",
        text="데이터 개수",
        title="로봇의 이동 방향별 데이터 분포",
        labels={
            "이동 방향": "로봇의 이동 방향",
            "데이터 개수": "데이터 개수"
        }
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="로봇의 이동 방향",
        yaxis_title="데이터 개수",
        uniformtext_minsize=10,
        uniformtext_mode="hide"
    )

    st.plotly_chart(fig, use_container_width=True)

    most_common = direction_counts.loc[
        direction_counts["데이터 개수"].idxmax()
    ]
    least_common = direction_counts.loc[
        direction_counts["데이터 개수"].idxmin()
    ]

    st.subheader("그래프 해석")

    st.write(
        f"- 데이터가 가장 많은 이동 방향은 **{most_common['이동 방향']}**이며, "
        f"총 {most_common['데이터 개수']:,}개입니다."
    )
    st.write(
        f"- 데이터가 가장 적은 이동 방향은 **{least_common['이동 방향']}**이며, "
        f"총 {least_common['데이터 개수']:,}개입니다."
    )
    st.write(
        "- 이동 방향별 데이터 개수의 차이를 통해 데이터가 특정 방향에 "
        "치우쳐 있는지 확인할 수 있습니다."
    )

except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.error(f"오류 내용: {e}")
