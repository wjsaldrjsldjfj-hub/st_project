
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="비와 따릉이 이용량 회귀분석",
    page_icon="🚲",
    layout="wide"
)

st.title("🚲 비가 따릉이 이용량에 영향을 미칠까?")
st.markdown(
    "2026년 서울시 공공자전거 대여 건수와 "
    "기상청 시간별 강수량 자료를 이용해 관계를 분석합니다."
)

BASE_DIR = Path(__file__).parent

BIKE_FILE = BASE_DIR / "서울특별시 공공자전거 일별 대여건수_26.1-6(1).csv"
WEATHER_FILE = BASE_DIR / "OBS_ASOS_TIM_20261127133619.csv"


# --------------------------------------------------
# CSV 읽기
# --------------------------------------------------

@st.cache_data
def read_csv_auto(file_path):
    for encoding in ["cp949", "euc-kr", "utf-8-sig", "utf-8"]:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"파일 인코딩을 읽을 수 없습니다: {file_path.name}")


@st.cache_data
def load_data():
    bike = read_csv_auto(BIKE_FILE)
    weather = read_csv_auto(WEATHER_FILE)

    # 필요한 열 확인
    required_bike = ["대여일자", "대여건수"]
    required_weather = ["일시", "강수량(mm)"]

    for col in required_bike:
        if col not in bike.columns:
            raise ValueError(f"따릉이 파일에 '{col}' 열이 없습니다.")

    for col in required_weather:
        if col not in weather.columns:
            raise ValueError(f"기상청 파일에 '{col}' 열이 없습니다.")

    # 날짜 및 숫자 변환
    bike["대여일자"] = pd.to_datetime(
        bike["대여일자"], errors="coerce"
    )
    bike["대여건수"] = pd.to_numeric(
        bike["대여건수"], errors="coerce"
    )

    weather["일시"] = pd.to_datetime(
        weather["일시"], errors="coerce"
    )
    weather["강수량(mm)"] = pd.to_numeric(
        weather["강수량(mm)"], errors="coerce"
    )

    bike = bike.dropna(subset=["대여일자", "대여건수"])
    weather = weather.dropna(subset=["일시"])

    # 날짜 단위로 변환
    bike["날짜"] = bike["대여일자"].dt.normalize()
    weather["날짜"] = weather["일시"].dt.normalize()

    # 따릉이 날짜별 대여 건수
    bike_daily = (
        bike.groupby("날짜", as_index=False)
        .agg(대여건수=("대여건수", "sum"))
    )

    # 기상청 날짜별 강수량 집계
    # 결측값은 0으로 바꾸지 않음
    weather_daily = (
        weather.groupby("날짜")
        .agg(
            관측시간수=("강수량(mm)", "count"),
            기록시간수=("일시", "size"),
            관측강수량합계=(
                "강수량(mm)",
                lambda x: x.sum(min_count=1)
            )
        )
        .reset_index()
    )

    # 두 자료 결합
    merged = pd.merge(
        bike_daily,
        weather_daily,
        on="날짜",
        how="inner"
    )

    merged = merged.sort_values("날짜")

    return merged


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------

try:
    if not BIKE_FILE.exists():
        st.error(f"따릉이 데이터 파일을 찾을 수 없습니다: {BIKE_FILE.name}")
        st.stop()

    if not WEATHER_FILE.exists():
        st.error(f"기상청 데이터 파일을 찾을 수 없습니다: {WEATHER_FILE.name}")
        st.stop()

    data = load_data()

except Exception as e:
    st.error(f"데이터를 읽는 중 오류가 발생했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 데이터 설명
# --------------------------------------------------

st.subheader("1. 분석 데이터")

col1, col2, col3 = st.columns(3)

col1.metric(
    "따릉이 전체 자료",
    f"{data['대여건수'].notna().sum():,}일"
)

col2.metric(
    "기상청 자료와 날짜가 겹치는 날",
    f"{len(data):,}일"
)

valid_weather = data.dropna(subset=["관측강수량합계"]).copy()

col3.metric(
    "강수량이 확인되는 날",
    f"{len(valid_weather):,}일"
)

if not valid_weather.empty:
    st.write(
        f"**분석 기간:** "
        f"{valid_weather['날짜'].min():%Y-%m-%d} ~ "
        f"{valid_weather['날짜'].max():%Y-%m-%d}"
    )

    st.write(
        f"**강수량 유효 관측시간 평균:** "
        f"{valid_weather['관측시간수'].mean():.1f}시간 / 하루"
    )

st.warning(
    "기상청 파일은 시간별 자료이며, 관측되지 않은 시간이 많습니다. "
    "따라서 관측된 시간 강수량의 합계가 실제 하루 전체 강수량보다 "
    "작을 수 있습니다. 관측 강수량이 0이라고 해서 하루 종일 "
    "비가 오지 않았다고 단정할 수 없습니다."
)

st.divider()


# --------------------------------------------------
# 관측시간 필터
# --------------------------------------------------

st.subheader("2. 분석 조건")

min_hours = st.slider(
    "하루에 필요한 최소 유효 강수량 관측시간",
    min_value=1,
    max_value=24,
    value=1,
    help="관측된 시간이 적은 날짜를 분석에서 제외할 수 있습니다."
)

analysis = valid_weather[
    valid_weather["관측시간수"] >= min_hours
].copy()

st.caption(
    f"현재 조건으로 분석할 수 있는 날짜: {len(analysis)}일"
)

if len(analysis) < 3:
    st.error(
        "회귀분석을 하기에는 데이터가 너무 적습니다. "
        "최소 관측시간을 낮춰 주세요."
    )
    st.stop()

if analysis["관측강수량합계"].nunique() < 2:
    st.error("강수량 값이 모두 같아 회귀분석을 할 수 없습니다.")
    st.stop()


# --------------------------------------------------
# 3. 회귀분석
# --------------------------------------------------

st.subheader("3. 강수량과 따릉이 대여 건수의 회귀분석")

x = analysis["관측강수량합계"].to_numpy(dtype=float)
y = analysis["대여건수"].to_numpy(dtype=float)

# 선형회귀
slope, intercept = np.polyfit(x, y, 1)

predicted = slope * x + intercept

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]

# 결정계수
ss_res = np.sum((y - predicted) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)

r_squared = (
    1 - ss_res / ss_tot
    if ss_tot != 0
    else 0
)

# 회귀 그래프
line_x = np.linspace(x.min(), x.max(), 100)
line_y = slope * line_x + intercept

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="실제 관측값",
        marker=dict(
            size=9,
            opacity=0.7
        ),
        text=analysis["날짜"].dt.strftime("%Y-%m-%d"),
        hovertemplate=(
            "날짜: %{text}<br>"
            "관측 강수량 합계: %{x:.1f}mm<br>"
            "따릉이 대여 건수: %{y:,.0f}건"
            "<extra></extra>"
        )
    )
)

fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="선형회귀선",
        line=dict(width=3)
    )
)

fig.update_layout(
    title="관측된 시간 강수량 합계와 따릉이 대여 건수",
    xaxis_title="관측된 시간 강수량 합계 (mm)",
    yaxis_title="따릉이 대여 건수 (건)",
    hovermode="closest",
    height=550,
    legend=dict(orientation="h", y=1.1)
)

st.plotly_chart(fig, use_container_width=True)

# 회귀 결과
c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "회귀계수(기울기)",
    f"{slope:,.2f}"
)

c2.metric(
    "절편",
    f"{intercept:,.2f}"
)

c3.metric(
    "상관계수",
    f"{correlation:.3f}"
)

c4.metric(
    "결정계수 R²",
    f"{r_squared:.3f}"
)

st.markdown("**회귀식**")

st.latex(
    rf"\hat{{y}} = {slope:.2f}x + {intercept:.2f}"
)

st.info(
    f"관측된 시간 강수량 합계가 1mm 증가할 때 "
    f"회귀모형은 따릉이 대여 건수가 평균적으로 "
    f"{slope:,.2f}건 변하는 관계를 나타냅니다. "
    "이는 회귀선의 수학적 해석이며, 실제 강수량 1mm 증가의 "
    "인과적 효과를 의미하지는 않습니다."
)

st.caption(
    "R²는 이 자료에서 회귀선이 대여 건수의 변동을 얼마나 설명하는지 "
    "나타내는 값입니다. 미래 예측의 정확도를 의미하지는 않습니다."
)

st.divider()


# --------------------------------------------------
# 4. 강수량 구간별 비교
# --------------------------------------------------

st.subheader("4. 관측 강수량 구간별 따릉이 대여 건수 비교")

# 관측 강수량 합계 기준 구간
bins = [-0.001, 0, 1, 5, 10, 20, np.inf]
labels = [
    "0mm",
    "0 초과~1mm",
    "1 초과~5mm",
    "5 초과~10mm",
    "10 초과~20mm",
    "20mm 초과"
]

analysis["강수량 구간"] = pd.cut(
    analysis["관측강수량합계"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

grouped = (
    analysis.groupby("강수량 구간", observed=False)
    .agg(
        평균대여건수=("대여건수", "mean"),
        날짜수=("대여건수", "count")
    )
    .reset_index()
)

fig2 = px.bar(
    grouped,
    x="강수량 구간",
    y="평균대여건수",
    text="평균대여건수",
    hover_data={
        "날짜수": True,
        "평균대여건수": ":,.0f"
    },
    title="관측 강수량 구간별 평균 따릉이 대여 건수"
)

fig2.update_traces(
    texttemplate="%{text:,.0f}건",
    textposition="outside"
)

fig2.update_layout(
    xaxis_title="관측된 시간 강수량 합계 구간",
    yaxis_title="평균 대여 건수 (건)",
    height=450
)

st.plotly_chart(fig2, use_container_width=True)

st.dataframe(
    grouped.rename(
        columns={
            "강수량 구간": "관측 강수량 구간",
            "평균대여건수": "평균 대여 건수",
            "날짜수": "날짜 수"
        }
    ),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "각 구간의 평균 대여 건수를 비교한 그래프입니다. "
    "관측시간이 불완전한 날짜가 포함될 수 있으므로 "
    "실제 강수량 구간별 비교로 단정하지 않도록 주의해야 합니다."
)

st.divider()


# --------------------------------------------------
# 5. 날짜별 변화
# --------------------------------------------------

st.subheader("5. 날짜별 따릉이 대여 건수 변화")

time_data = analysis.sort_values("날짜")

fig3 = go.Figure()

fig3.add_trace(
    go.Scatter(
        x=time_data["날짜"],
        y=time_data["대여건수"],
        mode="lines+markers",
        name="따릉이 대여 건수",
        line=dict(width=2)
    )
)

fig3.update_layout(
    title="날짜별 따릉이 대여 건수",
    xaxis_title="날짜",
    yaxis_title="대여 건수 (건)",
    height=430
)

st.plotly_chart(fig3, use_container_width=True)


# --------------------------------------------------
# 6. 분석 데이터 표
# --------------------------------------------------

st.subheader("6. 실제 분석에 사용된 데이터")

show_data = analysis[
    [
        "날짜",
        "대여건수",
        "관측강수량합계",
        "관측시간수",
        "기록시간수"
    ]
].copy()

show_data = show_data.rename(
    columns={
        "날짜": "날짜",
        "대여건수": "따릉이 대여 건수",
        "관측강수량합계": "관측 강수량 합계(mm)",
        "관측시간수": "유효 관측시간",
        "기록시간수": "기록된 시간 수"
    }
)

st.dataframe(
    show_data,
    use_container_width=True,
    hide_index=True
)

csv_data = show_data.to_csv(
    index=False,
    encoding="utf-8-sig"
)

st.download_button(
    label="분석 데이터 CSV 다운로드",
    data=csv_data,
    file_name="따릉이_강수량_회귀분석.csv",
    mime="text/csv"
)

st.divider()

st.subheader("7. 분석 결과 해석 시 주의할 점")

st.markdown(
    """
    1. 이 분석은 강수량과 따릉이 대여 건수 사이의 **상관관계**를 확인합니다.
    2. 회귀분석 결과만으로 비가 대여 건수 변화의 원인이라고 증명할 수는 없습니다.
    3. 기온, 요일, 공휴일, 계절 등의 다른 요인도 대여 건수에 영향을 줄 수 있습니다.
    4. 시간별 강수량 자료에 결측이 많으므로 하루 전체 강수량을 정확하게 반영하지 못할 수 있습니다.
    5. 보다 정확한 분석을 위해서는 날짜별 24시간 관측이 충분히 확보된 일강수량 자료를 사용하는 것이 좋습니다.
    """
)
