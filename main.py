
import streamlit as st
import pandas as pd
import requests
from io import StringIO


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

    # UCI 공식 데이터 파일 주소
    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00194/sensor_readings_24.data",
        "https://archive.ics.uci.edu/static/public/194/sensor_readings_24.data"
    ]

    last_error = None

    for url in urls:

        try:
            response = requests.get(
                url,
                timeout=30,
                headers={"User-Agent": "Mozilla/5.0"}
            )

            response.raise_for_status()

            # 다운로드한 데이터 읽기
            df = pd.read_csv(
                StringIO(response.text),
                header=None,
                sep=r"\s*,\s*",
                engine="python"
            )

            # 데이터 열 개수 확인
            if df.shape[1] != 25:
                raise ValueError(
                    f"예상한 열 개수는 25개이지만 "
                    f"실제 열 개수는 {df.shape[1]}개입니다."
                )

            # 열 이름 지정
            columns = [
                f"US{i}" for i in range(1, 25)
            ] + ["이동 방향"]

            df.columns = columns

            # 센서 데이터 숫자형으로 변환
            for col in columns[:24]:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="raise"
                )

            # 이동 방향 앞뒤 공백 제거
            df["이동 방향"] = (
                df["이동 방향"].astype(str).str.strip()
            )

            # 빈 데이터 확인
            if df.empty:
                raise ValueError("데이터가 비어 있습니다.")

            return df

        except Exception as e:
            last_error = e

    raise RuntimeError(
        f"데이터 다운로드에 실패했습니다. 마지막 오류: {last_error}"
    )


# ---------------------------------
# 3. 데이터 불러오기
# ---------------------------------
try:

    with st.spinner("로봇 센서 데이터를 불러오는 중입니다..."):

        df = load_data()

    st.success("데이터를 성공적으로 불러왔습니다!")

except Exception as e:

    st.error("데이터를 불러오는 중 오류가 발생했습니다.")

    st.write("오류 종류:", type(e).__name__)
    st.write("오류 내용:", str(e))

    st.info("""
    다음 사항을 확인해 주세요.

    1. 인터넷 연결 상태를 확인합니다.
    2. UCI 데이터 서버가 정상적으로 작동하는지 확인합니다.
    3. requirements.txt에 필요한 라이브러리가 있는지 확인합니다.
    4. Streamlit Cloud의 Manage app에서 로그를 확인합니다.
    """)

    st.stop()


# ---------------------------------
# 4. 데이터 기본 정보
# ---------------------------------
st.header("1. 데이터 기본 정보")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "전체 데이터 행 개수",
        f"{df.shape[0]:,}개"
    )

with col2:
    st.metric(
        "전체 데이터 열 개수",
        f"{df.shape[1]}개"
    )

with col3:
    st.metric(
        "초음파 센서 개수",
        "24개"
    )

st.write(
    "24개의 초음파 센서 측정값과 "
    "1개의 로봇 이동 방향으로 구성되어 있습니다."
)

st.divider()


# ---------------------------------
# 5. 데이터 미리보기
# ---------------------------------
st.header("2. 데이터 미리보기")

st.write("실제 수집된 로봇 초음파 센서 데이터의 앞부분입니다.")

st.dataframe(
    df.head(10),
    use_container_width=True
)

st.divider()


# ---------------------------------
# 6. 결측값 확인
# ---------------------------------
st.header("3. 결측값 확인")

missing_by_column = df.isnull().sum()

total_missing = int(missing_by_column.sum())

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "전체 결측값 개수",
        f"{total_missing:,}개"
    )

with col2:
    st.metric(
        "결측값이 있는 열 개수",
        f"{(missing_by_column > 0).sum()}개"
    )

if total_missing == 0:

    st.success("결측값이 없습니다.")

else:

    st.warning("일부 데이터에 결측값이 있습니다.")

    st.dataframe(
        missing_by_column[
            missing_by_column > 0
        ].rename("결측값 개수").to_frame(),
        use_container_width=True
    )

st.divider()


# ---------------------------------
# 7. 이동 방향별 데이터 개수
# ---------------------------------
st.header("4. 로봇 이동 방향별 데이터 개수")

st.write("""
각 초음파 센서 측정 상황에서 로봇이
어떤 방향으로 이동했는지 확인합니다.
""")

direction_counts = df["이동 방향"].value_counts()

st.dataframe(
    direction_counts.rename(
        "데이터 개수"
    ).to_frame(),
    use_container_width=True
)

st.write("**이동 방향별 데이터 개수**")

for direction, count in direction_counts.items():

    percentage = count / len(df) * 100

    st.write(
        f"- {direction}: {count:,}개 "
        f"({percentage:.2f}%)"
    )

st.divider()


# ---------------------------------
# 8. 데이터 열 정보
# ---------------------------------
st.header("5. 데이터 열 정보")

st.write("""
앞의 24개 열은 초음파 센서 측정값이며,
마지막 열은 로봇의 이동 방향입니다.
""")

st.code(
    ", ".join(df.columns),
    language="text"
)

st.success("데이터 읽기 및 기본 확인이 완료되었습니다.")
