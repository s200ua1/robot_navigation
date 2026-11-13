
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import requests
import zipfile
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

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
# 데이터 설정
# -----------------------------
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/194/"
    "wall+following+robot+navigation+data.zip"
)

sensor_columns = [f"US{i}" for i in range(1, 25)]

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

korean_order = [
    "직진",
    "약간 우회전",
    "급격한 우회전",
    "약간 좌회전"
]


# -----------------------------
# 데이터 불러오기
# -----------------------------
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

    columns = sensor_columns + ["Class"]
    df.columns = columns

    df["Class"] = df["Class"].astype(str).str.strip()

    for col in sensor_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=columns)

    return df


# -----------------------------
# 혼동 행렬 표시 함수
# -----------------------------
def show_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=direction_order
    )

    cm_df = pd.DataFrame(
        cm,
        index=korean_order,
        columns=korean_order
    )

    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
        title=title,
        labels={
            "x": "예측한 이동 방향",
            "y": "실제 이동 방향",
            "color": "데이터 개수"
        }
    )

    fig.update_layout(
        xaxis_title="예측한 이동 방향",
        yaxis_title="실제 이동 방향"
    )

    st.plotly_chart(fig, use_container_width=True)

    return cm


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
        df["Class"].value_counts().rename("데이터 개수"),
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

    direction_counts["이동 방향"] = pd.Categorical(
        direction_counts["이동 방향"],
        categories=korean_order,
        ordered=True
    )

    direction_counts = direction_counts.sort_values("이동 방향")

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
    st.plotly_chart(fig1, use_container_width=True)

    most_common = direction_counts.loc[
        direction_counts["데이터 개수"].idxmax()
    ]

    least_common = direction_counts.loc[
        direction_counts["데이터 개수"].idxmin()
    ]

    st.subheader("그래프 해석")

    st.write(
        f"- 데이터가 가장 많은 방향은 "
        f"**{most_common['이동 방향']}**이며 "
        f"{most_common['데이터 개수']:,}개입니다."
    )

    st.write(
        f"- 데이터가 가장 적은 방향은 "
        f"**{least_common['이동 방향']}**이며 "
        f"{least_common['데이터 개수']:,}개입니다."
    )

    st.write(
        "- 이동 방향별 데이터 개수의 차이를 통해 "
        "데이터가 특정 방향에 치우쳐 있는지 확인할 수 있습니다."
    )

    st.divider()

    # -----------------------------
    # 3. 센서 측정값 분포
    # -----------------------------
    st.header("3. 이동 방향별 초음파 센서 측정값 분포")

    st.write(
        "이동 방향을 선택하면 해당 방향에서 "
        "US1부터 US24까지 센서 측정값의 분포를 확인할 수 있습니다."
    )

    selected_direction = st.selectbox(
        "분석할 로봇의 이동 방향을 선택하세요.",
        options=direction_order,
        format_func=lambda x: direction_names[x]
    )

    selected_df = df[
        df["Class"] == selected_direction
    ].copy()

    melted_df = selected_df.melt(
        value_vars=sensor_columns,
        var_name="초음파 센서",
        value_name="센서 측정값"
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

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("그래프 해석")

    st.write(
        f"- 현재 선택한 이동 방향은 "
        f"**{direction_names[selected_direction]}**입니다."
    )

    st.write(
        f"- 해당 방향의 데이터는 총 {len(selected_df):,}개입니다."
    )

    st.write(
        "- 상자그림의 가운데 선은 중앙값을 나타내며, "
        "상자의 크기는 측정값의 분포 범위를 보여 줍니다."
    )

    st.write(
        "- 센서별 측정값의 차이를 비교하여 "
        "이동 방향과 센서 데이터 사이의 관계를 탐색할 수 있습니다."
    )

    st.divider()

    # -----------------------------
    # 4. 인공지능 모델 학습
    # -----------------------------
    st.header("4. 인공지능 모델 학습 및 이동 방향 예측")

    st.write(
        "24개의 초음파 센서 측정값을 이용하여 "
        "로봇의 이동 방향을 예측하는 모델을 학습합니다."
    )

    X = df[sensor_columns]
    y = df["Class"]

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

    st.write("- 독립변수(X): US1~US24")
    st.write("- 종속변수(y): Class")
    st.write("- 데이터 분할: 학습용 80%, 테스트용 20%")

    st.divider()

    st.subheader("인공지능 모델 학습")

    if st.button(
        "🤖 두 가지 인공지능 모델 학습하기",
        type="primary"
    ):

        with st.spinner("모델을 학습하고 있습니다..."):

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

    # 학습 결과 확인
    if "logistic_model" in st.session_state:

        logistic_accuracy = st.session_state["logistic_accuracy"]
        forest_accuracy = st.session_state["forest_accuracy"]

        st.subheader("모델별 예측 정확도")

        col1, col2 = st.columns(2)

        col1.metric(
            "로지스틱 회귀 정확도",
            f"{logistic_accuracy * 100:.2f}%"
        )

        col2.metric(
            "랜덤 포레스트 정확도",
            f"{forest_accuracy * 100:.2f}%"
        )

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

        comparison_df = pd.DataFrame({
            "실제 이동 방향": st.session_state["y_test"].map(
                direction_names
            ).values,
            "로지스틱 회귀 예측": pd.Series(
                st.session_state["logistic_predictions"]
            ).map(direction_names).values,
            "랜덤 포레스트 예측": pd.Series(
                st.session_state["forest_predictions"]
            ).map(direction_names).values
        })

        st.subheader("실제 이동 방향과 예측 결과 비교")
        st.dataframe(
            comparison_df.head(10),
            use_container_width=True
        )

    else:
        st.info(
            "모델 학습 버튼을 눌러 두 가지 인공지능 모델을 학습해 주세요."
        )

    st.divider()

    # -----------------------------
    # 5. 인공지능 모델 성능 평가
    # -----------------------------
    st.header("5. 인공지능 모델 성능 평가 및 비교")

    st.write(
        "두 모델의 예측 결과를 바탕으로 "
        "정확도, 정밀도, 재현율, F1 점수를 비교합니다."
    )

    if "logistic_model" not in st.session_state:

        st.warning(
            "아직 학습된 모델이 없습니다. "
            "먼저 4번 항목에서 '두 가지 인공지능 모델 학습하기' "
            "버튼을 눌러 주세요."
        )

    else:

        y_test_saved = st.session_state["y_test"]

        logistic_pred = st.session_state["logistic_predictions"]
        forest_pred = st.session_state["forest_predictions"]

        # 평가 지표 계산
        def calculate_metrics(y_true, y_pred):

            return {
                "정확도": accuracy_score(y_true, y_pred),
                "정밀도": precision_score(
                    y_true,
                    y_pred,
                    labels=direction_order,
                    average="macro",
                    zero_division=0
                ),
                "재현율": recall_score(
                    y_true,
                    y_pred,
                    labels=direction_order,
                    average="macro",
                    zero_division=0
                ),
                "F1 점수": f1_score(
                    y_true,
                    y_pred,
                    labels=direction_order,
                    average="macro",
                    zero_division=0
                )
            }

        logistic_metrics = calculate_metrics(
            y_test_saved,
            logistic_pred
        )

        forest_metrics = calculate_metrics(
            y_test_saved,
            forest_pred
        )

        # -----------------------------
        # 평가 지표 비교 표
        # -----------------------------
        st.subheader("모델별 평가 지표 비교")

        metrics_df = pd.DataFrame(
            {
                "로지스틱 회귀": logistic_metrics,
                "랜덤 포레스트": forest_metrics
            }
        )

        st.dataframe(
            metrics_df.style.format("{:.4f}"),
            use_container_width=True
        )

        st.caption(
            "정밀도, 재현율, F1 점수는 네 가지 이동 방향을 "
            "동일한 비중으로 평가하는 Macro Average 방식입니다."
        )

        # -----------------------------
        # 평가 지표 그래프
        # -----------------------------
        metrics_chart_df = metrics_df.reset_index()
        metrics_chart_df = metrics_chart_df.rename(
            columns={"index": "평가 지표"}
        )

        metrics_chart_df = metrics_chart_df.melt(
            id_vars="평가 지표",
            var_name="모델",
            value_name="점수"
        )

        fig4 = px.bar(
            metrics_chart_df,
            x="평가 지표",
            y="점수",
            color="모델",
            barmode="group",
            text=metrics_chart_df["점수"].map(
                lambda x: f"{x:.3f}"
            ),
            title="인공지능 모델별 평가 지표 비교",
            range_y=[0, 1]
        )

        st.plotly_chart(fig4, use_container_width=True)

        st.divider()

        # -----------------------------
        # 혼동 행렬
        # -----------------------------
        st.subheader("로지스틱 회귀 혼동 행렬")

        st.write(
            "실제 이동 방향과 모델이 예측한 이동 방향을 비교합니다."
        )

        cm_logistic = show_confusion_matrix(
            y_test_saved,
            logistic_pred,
            "로지스틱 회귀 혼동 행렬"
        )

        st.write(
            "- 혼동 행렬의 대각선에 있는 숫자는 "
            "올바르게 예측한 데이터의 개수입니다."
        )

        st.write(
            "- 대각선 이외의 숫자는 다른 이동 방향으로 "
            "잘못 예측한 데이터의 개수입니다."
        )

        st.divider()

        st.subheader("랜덤 포레스트 혼동 행렬")

        cm_forest = show_confusion_matrix(
            y_test_saved,
            forest_pred,
            "랜덤 포레스트 혼동 행렬"
        )

        st.write(
            "- 대각선의 숫자가 클수록 해당 이동 방향을 "
            "정확하게 예측한 데이터가 많다는 의미입니다."
        )

        st.write(
            "- 대각선 밖의 숫자를 비교하면 "
            "어떤 이동 방향을 서로 혼동하는지 알 수 있습니다."
        )

        st.divider()

        # -----------------------------
        # 모델 성능 종합 비교
        # -----------------------------
        st.subheader("모델 성능 종합 분석")

        # F1 점수를 기준으로 비교
        logistic_f1 = logistic_metrics["F1 점수"]
        forest_f1 = forest_metrics["F1 점수"]

        if logistic_f1 > forest_f1:
            best_name = "로지스틱 회귀"
            best_f1 = logistic_f1
        elif forest_f1 > logistic_f1:
            best_name = "랜덤 포레스트"
            best_f1 = forest_f1
        else:
            best_name = "두 모델"
            best_f1 = logistic_f1

        st.success(
            f"F1 점수 기준으로 더 높은 성능을 보인 모델은 "
            f"**{best_name}**입니다. "
            f"(F1 점수: {best_f1:.4f})"
        )

        st.write(
            f"- 로지스틱 회귀의 F1 점수: "
            f"{logistic_f1:.4f}"
        )

        st.write(
            f"- 랜덤 포레스트의 F1 점수: "
            f"{forest_f1:.4f}"
        )

        st.write(
            "- 정확도는 전체 예측 중 정답을 맞힌 비율입니다."
        )

        st.write(
            "- 정밀도는 특정 이동 방향이라고 예측한 데이터 중 "
            "실제로 그 방향인 데이터의 비율입니다."
        )

        st.write(
            "- 재현율은 실제로 특정 이동 방향인 데이터 중 "
            "모델이 올바르게 찾아낸 데이터의 비율입니다."
        )

        st.write(
            "- F1 점수는 정밀도와 재현율을 함께 고려한 지표입니다."
        )

        st.write(
            "- 혼동 행렬을 통해 모델이 어떤 이동 방향을 "
            "잘 예측하고 어떤 방향을 혼동하는지 확인할 수 있습니다."
        )

except Exception as e:
    st.error("데이터를 불러오거나 분석하는 중 오류가 발생했습니다.")
    st.error(f"오류 내용: {e}")
