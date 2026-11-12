
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import zipfile
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
    "로봇의 이동 방향을 분석하고 예측하는 프로젝트입니다."
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
# 이동 방향 이름
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
    # 2. 이동 방향별 데이터 분포
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
    # 3. 이동 방향별 센서 측정값 분포
    # -----------------------------
    st.header("3. 이동 방향별 초음파 센서 측정값 분포")

    st.write(
        "이동 방향을 선택하면 해당 방향에서 "
        "US1부터 US24까지 센서 측정값의 분포를 확인할 수 있습니다."
    )

    selected_direction = st.selectbox(
        "분석할 로봇의 이동 방향을 선택하세요.",
        options=direction_order,
        format_func=lambda x: direction_names.get(x, x)
    )

    selected_df = df[
        df["Class"] == selected_direction
    ].copy()

    melted_df = selected_df.melt(
        value_vars=sensor_columns,
        var_name="초음파 센서",
        value_name="센서 측정값"
    )

    melted_df["초음파 센서"] = pd.Categorical(
        melted_df["초음파 센서"],
        categories=sensor_columns,
        ordered=True
    )

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
        "상자의 크기는 측정값의 분포 범위를 보여 줍니다."
    )

    st.write(
        "- 센서별 상자그림을 비교하면 어떤 센서에서 "
        "측정값의 차이가 크거나 작은지 확인할 수 있습니다."
    )

    st.write(
        "- 이러한 차이를 분석하면 로봇의 이동 방향과 "
        "초음파 센서 측정값 사이의 관계를 이해하는 데 도움이 됩니다."
    )

    st.divider()

    # -----------------------------
    # 4. 인공지능 모델 학습
    # -----------------------------
    st.header("4. 인공지능 모델 학습 및 이동 방향 예측")

    st.write(
        "24개의 초음파 센서 측정값을 이용하여 "
        "로봇의 이동 방향을 예측하는 두 가지 인공지능 모델을 학습합니다."
    )

    st.subheader("학습 데이터 설정")

    # 독립변수와 종속변수 설정
    X = df[sensor_columns]
    y = df["Class"]

    # 학습용 80%, 테스트용 20%로 분리
    # stratify를 사용하여 이동 방향별 비율을 유지
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("학습용 데이터", f"{len(X_train):,}개")
    col2.metric("테스트용 데이터", f"{len(X_test):,}개")
    col3.metric("사용 센서 개수", "24개")

    st.write(
        "- 독립변수(X): US1~US24 초음파 센서 측정값"
    )
    st.write(
        "- 종속변수(y): 로봇의 이동 방향(Class)"
    )
    st.write(
        "- 데이터 분할: 학습용 80%, 테스트용 20%"
    )

    st.divider()

    # -----------------------------
    # 모델 학습 버튼
    # -----------------------------
    st.subheader("인공지능 모델 학습")

    if st.button("🤖 두 가지 인공지능 모델 학습하기", type="primary"):

        with st.spinner("인공지능 모델을 학습하고 있습니다..."):

            # 로지스틱 회귀
            logistic_model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=2000,
                    random_state=42
                ))
            ])

            logistic_model.fit(X_train, y_train)

            logistic_predictions = logistic_model.predict(X_test)

            logistic_accuracy = accuracy_score(
                y_test,
                logistic_predictions
            )

            # 랜덤 포레스트
            random_forest_model = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )

            random_forest_model.fit(X_train, y_train)

            forest_predictions = random_forest_model.predict(X_test)

            forest_accuracy = accuracy_score(
                y_test,
                forest_predictions
            )

        # 학습 결과 저장
        st.session_state["logistic_model"] = logistic_model
        st.session_state["random_forest_model"] = random_forest_model
        st.session_state["logistic_predictions"] = logistic_predictions
        st.session_state["forest_predictions"] = forest_predictions
        st.session_state["logistic_accuracy"] = logistic_accuracy
        st.session_state["forest_accuracy"] = forest_accuracy
        st.session_state["y_test"] = y_test

        st.success("두 가지 인공지능 모델의 학습이 완료되었습니다!")

    # -----------------------------
    # 학습 결과 표시
    # -----------------------------
    if "logistic_model" in st.session_state:

        st.subheader("모델별 예측 정확도")

        logistic_accuracy = st.session_state["logistic_accuracy"]
        forest_accuracy = st.session_state["forest_accuracy"]

        col1, col2 = st.columns(2)

        col1.metric(
            "로지스틱 회귀 정확도",
            f"{logistic_accuracy * 100:.2f}%"
        )

        col2.metric(
            "랜덤 포레스트 정확도",
            f"{forest_accuracy * 100:.2f}%"
        )

        # 정확도 비교 그래프
        accuracy_df = pd.DataFrame({
            "모델": ["로지스틱 회귀", "랜덤 포레스트"],
            "정확도": [
                logistic_accuracy * 100,
                forest_accuracy * 100
            ]
        })

        fig3 = px.bar(
            accuracy_df,
            x="모델",
            y="정확도",
            text=accuracy_df["정확도"].map(
                lambda x: f"{x:.2f}%"
            ),
            title="인공지능 모델별 예측 정확도 비교",
            labels={
                "모델": "인공지능 모델",
                "정확도": "정확도(%)"
            },
            range_y=[0, 100]
        )

        fig3.update_traces(textposition="outside")

        st.plotly_chart(fig3, use_container_width=True)

        # 더 높은 정확도의 모델 찾기
        if logistic_accuracy > forest_accuracy:
            best_model_name = "로지스틱 회귀"
            best_accuracy = logistic_accuracy
        elif forest_accuracy > logistic_accuracy:
            best_model_name = "랜덤 포레스트"
            best_accuracy = forest_accuracy
        else:
            best_model_name = "두 모델"
            best_accuracy = logistic_accuracy

        st.subheader("모델 학습 결과 해석")

        st.write(
            f"- 더 높은 정확도를 보인 모델은 "
            f"**{best_model_name}**입니다."
        )

        st.write(
            f"- 해당 모델의 정확도는 "
            f"**{best_accuracy * 100:.2f}%**입니다."
        )

        st.write(
            "- 정확도는 테스트 데이터 중 실제 이동 방향을 "
            "올바르게 예측한 데이터의 비율입니다."
        )

        st.write(
            "- 정확도만으로 모든 성능을 판단하기는 어려우므로 "
            "다음 단계에서 혼동 행렬과 정밀도, 재현율 등의 "
            "지표도 함께 분석할 예정입니다."
        )

        st.divider()

        # -----------------------------
        # 실제값과 예측값 비교
        # -----------------------------
        st.subheader("실제 이동 방향과 예측 결과 비교")

        y_test = st.session_state["y_test"]

        comparison_df = pd.DataFrame({
            "실제 이동 방향": y_test.map(direction_names).values,
            "로지스틱 회귀 예측": pd.Series(
                st.session_state["logistic_predictions"]
            ).map(direction_names).values,
            "랜덤 포레스트 예측": pd.Series(
                st.session_state["forest_predictions"]
            ).map(direction_names).values
        })

        st.write("테스트 데이터 중 앞의 10개 결과입니다.")

        st.dataframe(
            comparison_df.head(10),
            use_container_width=True
        )

        st.caption(
            "예측 결과가 실제 이동 방향과 같으면 올바르게 예측한 것입니다."
        )

    else:
        st.info(
            "위의 '두 가지 인공지능 모델 학습하기' 버튼을 누르면 "
            "모델 학습과 정확도 비교 결과를 확인할 수 있습니다."
        )

except Exception as e:
    st.error("데이터를 불러오거나 분석하는 중 오류가 발생했습니다.")
    st.error(f"오류 내용: {e}")
