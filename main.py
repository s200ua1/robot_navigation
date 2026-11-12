
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import zipfile
import io

# -----------------------------
# 기본 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="AI 로봇 장애물 회피 예측기",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI 로봇 장애물 회피 예측기")
st.write(
    "초음파 센서 데이터를 활용하여 "
    "로봇의 이동 방향을 분석하는 프로젝트입니다."
)

# -----------------------------
# 데이터 불러오기
# -----------------------------
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/194/"
    "wall+following+robot+navigation+data.zip"
)

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
            df = pd.read_csv(
                f,
                header=None,
                skipinitialspace=True
            )

    columns = [f"US{i}" for i in range(1, 25)] + ["Class"]
    df.columns = columns

    df["Class"] = df["Class"].astype(str).str.strip()

    for col in columns[:24]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=columns)

    return df


# -----------------------------
# 이동 방향 이름 변환
# -----------------------------
direction_names = {
    "Move-Forward": "직진",
    "Slight-Right-Turn": "약간 우회전",
    "Sharp-Right-Turn": "급격한 우회전",
    "Slight-Left-Turn": "약간 좌회전"
}

direction_order = [
    "Move-Forward",
    "Slight-Right-Turn",
    "Sharp-Right-Turn",
    "Slight-Left-Turn"
]

sensor_columns = [f"US{i}" for i in range(1, 25)]


# -----------------------------
# 앱 실행
# -----------------------------
try:
    df = load_data()

    st.success("데이터를 성공적으로 불러왔습니다!")

    # -----------------------------
    # 1. 데이터 기본 정보
    # -----------------------------
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
        df["Class"]
        .value_counts()
        .rename("데이터 개수"),
        use_container_width=True
    )

    st.divider()

    # -----------------------------
    # 2. 첫 번째 그래프
    # -----------------------------
    st.header("2. 로봇의 이동 방향별 데이터 분포")

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

    fig1 = px.bar(
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

    fig1.update_traces(textposition="outside")

    fig1.update_layout(
        xaxis_title="로봇의 이동 방향",
        yaxis_title="데이터 개수"
    )

    st.plotly_chart(fig1, use_container_width=True)

    most_common = direction_counts.loc[
        direction_counts["데이터 개수"].idxmax()
    ]

    least_common = direction_counts.loc[
        direction_counts["데이터 개수"].idxmin()
    ]

    st.subheader("그래프 해석")

    st.write(
        f"- 데이터가 가장 많은 이동 방향은 "
        f"**{most_common['이동 방향']}**이며, "
        f"{most_common['데이터 개수']:,}개입니다."
    )

    st.write(
        f"- 데이터가 가장 적은 이동 방향은 "
        f"**{least_common['이동 방향']}**이며, "
        f"{least_common['데이터 개수']:,}개입니다."
    )

    st.write(
        "- 이동 방향별 데이터 개수의 차이를 통해 "
        "데이터가 특정 방향에 치우쳐 있는지 확인할 수 있습니다."
    )

    st.divider()

    # -----------------------------
    # 3. 두 번째 그래프
    # -----------------------------
    st.header("3. 이동 방향별 초음파 센서 측정값 분포")

    st.write(
        "이동 방향을 선택하면 해당 방향에서 "
        "US1부터 US24까지 센서 측정값이 어떻게 분포하는지 확인할 수 있습니다."
    )

    selected_direction = st.selectbox(
        "분석할 로봇의 이동 방향을 선택하세요.",
        options=direction_order,
        format_func=lambda x: direction_names.get(x, x)
    )

    # 선택한 이동 방향의 데이터만 추출
    selected_df = df[
        df["Class"] == selected_direction
    ].copy()

    # 센서 데이터를 세로 형태로 변환
    melted_df = selected_df.melt(
        value_vars=sensor_columns,
        var_name="초음파 센서",
        value_name="센서 측정값"
    )

    # 센서 순서 유지
    melted_df["초음파 센서"] = pd.Categorical(
        melted_df["초음파 센서"],
        categories=sensor_columns,
        ordered=True
    )

    # 상자그림 생성
    fig2 = px.box(
        melted_df,
        x="초음파 센서",
        y="센서 측정값",
        category_orders={
            "초음파 센서": sensor_columns
        },
        title=(
            "이동 방향별 초음파 센서 측정값 분포 - "
            + direction_names[selected_direction]
        ),
        labels={
            "초음파 센서": "초음파 센서",
            "센서 측정값": "센서 측정값"
        },
        points=False
    )

    fig2.update_layout(
        xaxis_title="초음파 센서",
        yaxis_title="센서 측정값"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------
    # 그래프 해석
    # -----------------------------
    st.subheader("그래프 해석")

    st.write(
        f"- 현재 선택한 이동 방향은 "
        f"**{direction_names[selected_direction]}**입니다."
    )

    st.write(
        f"- 이 방향의 데이터는 총 "
        f"**{len(selected_df):,}개**입니다."
    )

    st.write(
        "- 상자그림의 가운데 선은 중앙값을 나타내며, "
        "상자의 크기는 측정값이 분포한 범위를 보여 줍니다."
    )

    st.write(
        "- 센서별 상자그림을 비교하면 어떤 센서에서 "
        "측정값의 차이가 크거나 작은지 확인할 수 있습니다."
    )

    st.write(
        "- 이러한 차이를 분석하면 로봇의 이동 방향과 "
        "초음파 센서 측정값 사이의 관계를 이해하는 데 도움이 됩니다."
    )

except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.error(f"오류 내용: {e}")
