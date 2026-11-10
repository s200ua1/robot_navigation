
import streamlit as st
import pandas as pd
from ucimlrepo import fetch_ucirepo


# ---------------------------------
# 1. 페이지 기본 설정
# ---------------------------------
st.set_page_config(
    page_title="AI 로봇 장애물 회피 예측기",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI 로봇 장애물 회피 예측기")

st.write("""
초음파 센서 데이터를 이용하여 로봇의 이동 방향을 예측하는
인공지능 프로젝트입니다.

UCI Machine Learning Repository의
Wall-Following Robot Navigation Data를 사용합니다.
""")

st.divider()


# ---------------------------------
# 2. 데이터 읽기
# ---------------------------------
@st.cache_data
def load_data():

    # UCI 데이터셋 불러오기
    robot_data = fetch_ucirepo(id=194)

    # 24개 초음파 센서 데이터
    X = robot_data.data.features.copy()

    # 로봇의 이동 방향 데이터
    y = robot_data.data.targets.copy()

    # 센서 데이터가 24개인지 확인
    if X.shape[1] != 24:
        raise ValueError(
            f"센서 데이터의 열 개수가 24개가 아닙니다. "
            f"현재 열 개수: {X.shape[1]}"
        )

    # 열 이름 지정
    X.columns = [f"US{i}" for i in range(1, 25)]

    # 정답 데이터 확인
    if y.shape[1] != 1:
        raise ValueError(
            "이동 방향 데이터의 열 개수가 예상과 다릅니다."
        )

    # 이동 방향 열 이름 지정
    y.columns = ["이동 방향"]

    # 센서 데이터와 이동 방향 데이터 합치기
    df = pd.concat(
        [X, y],
        axis=1
    )

    return df


# ---------------------------------
# 3. 데이터 불러오기 실행 및 오류 처리
# ---------------------------------
try:

    with st.spinner("로봇 센서 데이터를 불러오는 중입니다..."):

        df = load_data()

    st.success("데이터를 성공적으로 불러왔습니다!")

except Exception as e:

    st.error("데이터를 불러오는 중 오류가 발생했습니다.")

    st.write("**오류 종류:**", type(e).__name__)
    st.write("**오류 내용:**", str(e))

    st.info("""
    다음 사항을 확인해 주세요.

    1. 인터넷 연결이 정상적인지 확인합니다.
    2. requirements.txt에 필요한 라이브러리가 있는지 확인합니다.
    3. UCI Machine Learning Repository 서버가 정상적으로 작동하는지 확인합니다.
    4. Streamlit Cloud의 앱 로그에서 자세한 오류를 확인합니다.
    """)

    st.stop()


# ---------------------------------
# 4. 데이터 전체 크기
# ---------------------------------
st.header("1. 데이터 기본 정보")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="전체 데이터 행 개수",
        value=f"{df.shape[0]:,}개"
    )

with col2:
    st.metric(
        label="전체 데이터 열 개수",
        value=f"{df.shape[1]}개"
    )

with col3:
    st.metric(
        label="초음파 센서 개수",
        value="24개"
    )


st.write("전체 데이터는 24개의 초음파 센서 측정값과 1개의 이동 방향으로 구성되어 있습니다.")

st.divider()


# ---------------------------------
# 5. 데이터 앞부분 확인
# ---------------------------------
st.header("2. 데이터 미리보기")

st.write("실제 수집된 로봇 초음파 센서 데이터의 앞부분입니다.")

st.dataframe(
    df.head(10),
    use_container_width=True
)


# ---------------------------------
# 6. 결측값 확인
# ---------------------------------
st.divider()

st.header("3. 결측값 확인")

missing_by_column = df.isnull().sum()
total_missing = int(missing_by_column.sum())

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="전체 결측값 개수",
        value=f"{total_missing:,}개"
    )

with col2:
    st.metric(
        label="결측값이 있는 열 개수",
        value=f"{(missing_by_column > 0).sum()}개"
    )

if total_missing == 0:

    st.success("결측값이 없습니다. 모든 데이터가 정상적으로 입력되어 있습니다.")

else:

    st.warning("일부 데이터에 결측값이 존재합니다.")

    st.write("열별 결측값 개수")

    st.dataframe(
        missing_by_column[missing_by_column > 0]
        .rename("결측값 개수")
        .to_frame(),
        use_container_width=True
    )


# ---------------------------------
# 7. 이동 방향별 데이터 개수
# ---------------------------------
st.divider()

st.header("4. 로봇 이동 방향별 데이터 개수")

st.write("""
각 초음파 센서 측정 상황에서 로봇이 어떤 방향으로
이동하도록 결정되었는지 확인합니다.
""")

direction_counts = df["이동 방향"].value_counts()

st.dataframe(
    direction_counts.rename("데이터 개수").to_frame(),
    use_container_width=True
)

st.write("**이동 방향별 데이터 개수 확인 결과**")

for direction, count in direction_counts.items():

    percentage = count / len(df) * 100

    st.write(
        f"- {direction}: {count:,}개 "
        f"({percentage:.2f}%)"
    )


# ---------------------------------
# 8. 데이터 열 이름 확인
# ---------------------------------
st.divider()

st.header("5. 데이터 열 정보")

st.write("앞의 24개 열은 초음파 센서 측정값이며, 마지막 열은 로봇의 이동 방향입니다.")

st.code(
    ", ".join(df.columns),
    language="text"
)

st.success("데이터 읽기 및 기본 확인이 완료되었습니다.")
