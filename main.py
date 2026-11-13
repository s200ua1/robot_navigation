
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import zipfile
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# --------------------------------------------------
# 1. 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="AI 로봇 장애물 회피 예측기",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI 로봇 장애물 회피 예측기")
st.write(
    "초음파 센서 데이터를 분석하여 로봇의 이동 방향을 예측하는 인공지능 프로그램입니다."
)

SENSOR_COLUMNS = [f"US{i}" for i in range(1, 25)]

CLASS_NAMES = [
    "Move-Forward",
    "Slight-Right-Turn",
    "Sharp-Right-Turn",
    "Slight-Left-Turn"
]

CLASS_KOREAN = {
    "Move-Forward": "직진",
    "Slight-Right-Turn": "약간 우회전",
    "Sharp-Right-Turn": "급격한 우회전",
    "Slight-Left-Turn": "약간 좌회전"
}

# --------------------------------------------------
# 2. 데이터 불러오기 및 전처리
# --------------------------------------------------

@st.cache_data
def load_data():
    url = (
        "https://archive.ics.uci.edu/static/public/194/"
        "wall+following+robot+navigation+data.zip"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        data_file = next(
            name for name in z.namelist()
            if name.endswith("sensor_readings_24.data")
        )

        with z.open(data_file) as file:
            df = pd.read_csv(file, header=None)

    df.columns = SENSOR_COLUMNS + ["Class"]

    for col in SENSOR_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=SENSOR_COLUMNS + ["Class"])
    df["Class"] = df["Class"].astype(str).str.strip()

    return df


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.error(f"오류 내용: {e}")
    st.info("인터넷 연결 상태를 확인한 후 다시 실행해 주세요.")
    st.stop()


# --------------------------------------------------
# 3. 데이터 기본 정보
# --------------------------------------------------

st.header("1. 데이터 확인하기")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("전체 데이터 수", f"{len(df):,}개")

with col2:
    st.metric("초음파 센서 수", "24개")

with col3:
    st.metric("이동 방향 종류", df["Class"].nunique())

with st.expander("데이터 미리 보기"):
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("데이터 결측값 확인"):
    missing = df.isnull().sum()
    st.dataframe(
        missing.rename("결측값 개수"),
        use_container_width=True
    )


# --------------------------------------------------
# 4. 이동 방향별 데이터 분포
# --------------------------------------------------

st.divider()
st.header("2. 로봇의 이동 방향별 데이터 분포")

class_counts = (
    df["Class"]
    .value_counts()
    .reindex(CLASS_NAMES, fill_value=0)
    .reset_index()
)

class_counts.columns = ["이동 방향", "데이터 개수"]
class_counts["이동 방향"] = class_counts["이동 방향"].map(
    CLASS_KOREAN
)

fig = px.bar(
    class_counts,
    x="이동 방향",
    y="데이터 개수",
    text="데이터 개수",
    title="로봇의 이동 방향별 데이터 분포"
)

fig.update_traces(textposition="outside")
fig.update_layout(
    xaxis_title="이동 방향",
    yaxis_title="데이터 개수"
)

st.plotly_chart(fig, use_container_width=True)

st.info(
    "이 그래프는 각 이동 방향에 해당하는 데이터가 얼마나 있는지 보여 줍니다. "
    "방향별 데이터 개수의 차이를 확인할 수 있습니다."
)


# --------------------------------------------------
# 5. 이동 방향에 따른 센서값 비교
# --------------------------------------------------

st.divider()
st.header("3. 이동 방향에 따른 초음파 센서값 분석")

selected_class = st.selectbox(
    "확인할 이동 방향을 선택하세요.",
    CLASS_NAMES,
    format_func=lambda x: CLASS_KOREAN.get(x, x)
)

selected_data = df[df["Class"] == selected_class]

melted_data = selected_data.melt(
    id_vars=["Class"],
    value_vars=SENSOR_COLUMNS,
    var_name="초음파 센서",
    value_name="측정값"
)

fig_box = px.box(
    melted_data,
    x="초음파 센서",
    y="측정값",
    title=f"{CLASS_KOREAN[selected_class]}일 때 센서별 측정값 분포",
    points=False
)

fig_box.update_layout(
    xaxis_title="초음파 센서",
    yaxis_title="센서 측정값"
)

st.plotly_chart(fig_box, use_container_width=True)

st.info(
    "상자그림을 통해 선택한 이동 방향에서 각 초음파 센서가 측정한 값의 "
    "분포와 차이를 비교할 수 있습니다."
)


# --------------------------------------------------
# 6. 인공지능 모델 학습
# --------------------------------------------------

st.divider()
st.header("4. 인공지능 모델 학습하기")

st.write(
    "US1~US24의 초음파 센서 측정값을 입력 데이터로 사용하고, "
    "Class를 예측 대상으로 설정합니다."
)

if st.button("두 인공지능 모델 학습하기", type="primary"):

    X = df[SENSOR_COLUMNS]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 로지스틱 회귀
    logistic_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000)
    )

    logistic_model.fit(X_train, y_train)
    logistic_pred = logistic_model.predict(X_test)

    # 랜덤 포레스트
    forest_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    forest_model.fit(X_train, y_train)
    forest_pred = forest_model.predict(X_test)

    # 모델과 평가 결과 저장
    st.session_state["logistic_model"] = logistic_model
    st.session_state["forest_model"] = forest_model
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test
    st.session_state["logistic_pred"] = logistic_pred
    st.session_state["forest_pred"] = forest_pred

    st.success("두 인공지능 모델의 학습이 완료되었습니다!")

if "logistic_model" in st.session_state:
    st.success("학습된 모델이 준비되어 있습니다.")
else:
    st.warning("아직 모델이 학습되지 않았습니다. 위 버튼을 눌러 먼저 학습해 주세요.")


# --------------------------------------------------
# 7. 모델 성능 평가
# --------------------------------------------------

st.divider()
st.header("5. 인공지능 모델 성능 평가")

if "logistic_model" not in st.session_state:

    st.info("모델 성능을 확인하려면 먼저 4단계에서 모델을 학습해 주세요.")

else:

    y_test = st.session_state["y_test"]
    logistic_pred = st.session_state["logistic_pred"]
    forest_pred = st.session_state["forest_pred"]

    def calculate_metrics(y_true, y_pred):
        return {
            "정확도": accuracy_score(y_true, y_pred),
            "정밀도": precision_score(
                y_true, y_pred,
                labels=CLASS_NAMES,
                average="macro",
                zero_division=0
            ),
            "재현율": recall_score(
                y_true, y_pred,
                labels=CLASS_NAMES,
                average="macro",
                zero_division=0
            ),
            "F1 점수": f1_score(
                y_true, y_pred,
                labels=CLASS_NAMES,
                average="macro",
                zero_division=0
            )
        }

    logistic_metrics = calculate_metrics(y_test, logistic_pred)
    forest_metrics = calculate_metrics(y_test, forest_pred)

    comparison = pd.DataFrame(
        [logistic_metrics, forest_metrics],
        index=["로지스틱 회귀", "랜덤 포레스트"]
    )

    st.subheader("모델별 성능 비교")
    st.dataframe(
        comparison.style.format("{:.3f}"),
        use_container_width=True
    )

    chart_data = comparison.reset_index().rename(
        columns={"index": "모델"}
    )

    chart_data = chart_data.melt(
        id_vars="모델",
        var_name="평가 지표",
        value_name="점수"
    )

    fig_metrics = px.bar(
        chart_data,
        x="평가 지표",
        y="점수",
        color="모델",
        barmode="group",
        text_auto=".3f",
        title="인공지능 모델 성능 비교"
    )

    fig_metrics.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig_metrics, use_container_width=True)

    st.subheader("혼동 행렬 비교")

    col1, col2 = st.columns(2)

    for col, title, predictions in [
        (col1, "로지스틱 회귀", logistic_pred),
        (col2, "랜덤 포레스트", forest_pred)
    ]:
        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=CLASS_NAMES
        )

        matrix_korean = [
            CLASS_KOREAN[x] for x in CLASS_NAMES
        ]

        fig_matrix = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=matrix_korean,
                y=matrix_korean,
                text=matrix,
                texttemplate="%{text}",
                colorscale="Blues"
            )
        )

        fig_matrix.update_layout(
            title=title,
            xaxis_title="예측한 방향",
            yaxis_title="실제 방향"
        )

        with col:
            st.plotly_chart(
                fig_matrix,
                use_container_width=True
            )

    logistic_f1 = logistic_metrics["F1 점수"]
    forest_f1 = forest_metrics["F1 점수"]

    if logistic_f1 > forest_f1:
        st.info(
            "로지스틱 회귀 모델의 F1 점수가 더 높습니다. "
            "이번 평가 데이터에서는 로지스틱 회귀 모델이 더 좋은 "
            "균형 성능을 보였습니다."
        )
    elif forest_f1 > logistic_f1:
        st.info(
            "랜덤 포레스트 모델의 F1 점수가 더 높습니다. "
            "이번 평가 데이터에서는 랜덤 포레스트 모델이 더 좋은 "
            "균형 성능을 보였습니다."
        )
    else:
        st.info("두 모델의 F1 점수가 같습니다.")


# --------------------------------------------------
# 8. 새로운 센서값으로 이동 방향 예측
# --------------------------------------------------

st.divider()
st.header("6. AI 로봇 이동 방향 예측하기")

st.write(
    "초음파 센서 측정값 24개를 입력하면 두 인공지능 모델이 "
    "로봇의 이동 방향을 각각 예측합니다."
)

if "logistic_model" not in st.session_state:

    st.warning(
        "예측 기능을 사용하려면 먼저 4단계에서 인공지능 모델을 학습해 주세요."
    )

else:

    st.subheader("센서값 입력하기")

    input_method = st.radio(
        "센서값 입력 방법을 선택하세요.",
        ["직접 입력하기", "기존 데이터에서 샘플 선택하기"],
        horizontal=True
    )

    # 기존 데이터 샘플 선택
    if input_method == "기존 데이터에서 샘플 선택하기":

        sample_index = st.selectbox(
            "예측에 사용할 데이터 번호를 선택하세요.",
            range(min(len(df), 1000)),
            format_func=lambda x: f"{x + 1}번 데이터"
        )

        sample_values = df.iloc[sample_index][SENSOR_COLUMNS]

    else:
        sample_values = pd.Series(
            df[SENSOR_COLUMNS].median().values,
            index=SENSOR_COLUMNS
        )

    st.caption(
        "각 센서의 측정값을 입력해 주세요. "
        "기존 데이터에서 샘플을 선택하면 해당 데이터의 측정값이 기본값으로 표시됩니다."
    )

    with st.form("prediction_form"):

        sensor_values = {}

        # 24개 센서 입력칸을 4열로 배치
        for start in range(0, 24, 4):

            columns = st.columns(4)

            for i in range(start, min(start + 4, 24)):

                sensor_name = SENSOR_COLUMNS[i]

                with columns[i - start]:
                    sensor_values[sensor_name] = st.number_input(
                        sensor_name,
                        value=float(sample_values[sensor_name]),
                        format="%.4f",
                        key=f"prediction_{input_method}_{sample_index if input_method == '기존 데이터에서 샘플 선택하기' else 'manual'}_{sensor_name}"
                    )

        predict_button = st.form_submit_button(
            "로봇 이동 방향 예측하기",
            type="primary",
            use_container_width=True
        )

    if predict_button:

        input_df = pd.DataFrame(
            [[sensor_values[col] for col in SENSOR_COLUMNS]],
            columns=SENSOR_COLUMNS
        )

        logistic_model = st.session_state["logistic_model"]
        forest_model = st.session_state["forest_model"]

        # 두 모델의 예측
        logistic_prediction = logistic_model.predict(input_df)[0]
        forest_prediction = forest_model.predict(input_df)[0]

        logistic_probabilities = logistic_model.predict_proba(input_df)[0]
        forest_probabilities = forest_model.predict_proba(input_df)[0]

        logistic_classes = logistic_model.classes_
        forest_classes = forest_model.classes_

        st.divider()
        st.subheader("예측 결과")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 로지스틱 회귀")
            st.success(
                f"예측 방향: **{CLASS_KOREAN.get(logistic_prediction, logistic_prediction)}**"
            )

        with col2:
            st.markdown("#### 랜덤 포레스트")
            st.success(
                f"예측 방향: **{CLASS_KOREAN.get(forest_prediction, forest_prediction)}**"
            )

        # 예측 결과 일치 여부
        if logistic_prediction == forest_prediction:
            st.success(
                "두 모델이 같은 이동 방향을 예측했습니다."
            )
        else:
            st.warning(
                "두 모델의 예측 결과가 다릅니다. "
                "모델마다 학습한 판단 기준이 다르기 때문일 수 있습니다."
            )

        st.divider()
        st.subheader("이동 방향별 예측 확률 비교")

        probability_rows = []

        for class_name in CLASS_NAMES:

            logistic_probability = 0
            forest_probability = 0

            for i, name in enumerate(logistic_classes):
                if name == class_name:
                    logistic_probability = logistic_probabilities[i]

            for i, name in enumerate(forest_classes):
                if name == class_name:
                    forest_probability = forest_probabilities[i]

            probability_rows.append({
                "이동 방향": CLASS_KOREAN[class_name],
                "로지스틱 회귀": logistic_probability,
                "랜덤 포레스트": forest_probability
            })

        probability_df = pd.DataFrame(probability_rows)

        probability_chart = probability_df.melt(
            id_vars="이동 방향",
            var_name="모델",
            value_name="예측 확률"
        )

        probability_chart["예측 확률(%)"] = (
            probability_chart["예측 확률"] * 100
        )

        fig_probability = px.bar(
            probability_chart,
            x="이동 방향",
            y="예측 확률(%)",
            color="모델",
            barmode="group",
            text=probability_chart["예측 확률(%)"].map(
                lambda x: f"{x:.1f}%"
            ),
            title="모델별 이동 방향 예측 확률"
        )

        fig_probability.update_traces(
            textposition="outside"
        )

        fig_probability.update_layout(
            yaxis_title="예측 확률(%)",
            xaxis_title="이동 방향",
            yaxis_range=[0, 110]
        )

        st.plotly_chart(
            fig_probability,
            use_container_width=True
        )

        st.subheader("예측 확률 표")
        display_probability = probability_df.copy()

        display_probability["로지스틱 회귀"] = (
            display_probability["로지스틱 회귀"] * 100
        ).map(lambda x: f"{x:.2f}%")

        display_probability["랜덤 포레스트"] = (
            display_probability["랜덤 포레스트"] * 100
        ).map(lambda x: f"{x:.2f}%")

        st.dataframe(
            display_probability,
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "예측 확률은 각 모델이 해당 이동 방향이라고 판단한 정도를 나타냅니다. "
            "실제 로봇의 움직임을 보장하는 값은 아닙니다."
        )

# --------------------------------------------------
# 마무리
# --------------------------------------------------

st.divider()

st.caption(
    "본 프로그램은 UCI Wall-Following Robot Navigation Data를 활용하여 "
    "초음파 센서값에 따른 로봇의 이동 방향을 예측하는 학습용 프로그램입니다."
)
