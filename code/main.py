
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from io import BytesIO
from zipfile import ZipFile
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="비와 따릉이 이용량 회귀분석",
    page_icon="🚲",
    layout="wide"
)

st.title("🚲 비가 공용 자전거 이용량에 정말 영향을 미칠까?")
st.markdown("""
서울시 공공자전거 따릉이 대여이력과 기상청의 실제 관측자료를 이용해
**강수량과 따릉이 대여량 사이의 관계**를 회귀분석합니다.
""")

st.info(
    "분석 기간: 2026년 6월 1일 ~ 6월 30일\n\n"
    "분석 대상: 서울시 일별 따릉이 대여 건수와 실제 일강수량"
)


# --------------------------------------------------
# 파일 읽기 함수
# --------------------------------------------------

def read_csv_with_encoding(file, **kwargs):
    """여러 인코딩을 시도해 CSV 파일을 읽습니다."""
    encodings = ["cp949", "utf-8-sig", "utf-8"]

    last_error = None

    for encoding in encodings:
        try:
            file.seek(0)
            return pd.read_csv(
                file,
                encoding=encoding,
                **kwargs
            )
        except Exception as e:
            last_error = e

    raise ValueError(f"CSV 파일을 읽을 수 없습니다: {last_error}")


def load_weather(weather_file):
    """기상청 일별 관측자료를 읽습니다."""

    weather = read_csv_with_encoding(weather_file)

    required_columns = [
        "일시",
        "평균기온(°C)",
        "일강수량(mm)"
    ]

    for col in required_columns:
        if col not in weather.columns:
            raise ValueError(
                f"기상자료에 '{col}' 열이 없습니다."
            )

    weather["날짜"] = pd.to_datetime(
        weather["일시"],
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    numeric_columns = [
        "평균기온(°C)",
        "최저기온(°C)",
        "최고기온(°C)",
        "일강수량(mm)",
        "평균 풍속(m/s)",
        "평균 상대습도(%)",
        "합계 일조시간(hr)"
    ]

    for col in numeric_columns:
        if col in weather.columns:
            weather[col] = pd.to_numeric(
                weather[col],
                errors="coerce"
            )

    # 일강수량이 비어 있는 날은 강수량이 측정되지 않은 날로 처리
    # 원본 결측 여부는 별도 열에 보존
    weather["강수량_원자료결측"] = (
        weather["일강수량(mm)"].isna()
    )

    weather["일강수량(mm)"] = (
        weather["일강수량(mm)"].fillna(0)
    )

    return weather


def load_bike_from_zip(uploaded_file):
    """ZIP 내부의 따릉이 CSV를 날짜별로 집계합니다."""

    daily_counts = []

    zip_bytes = BytesIO(uploaded_file.getvalue())

    with ZipFile(zip_bytes) as archive:

        csv_files = [
            name for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_files:
            raise ValueError(
                "ZIP 파일 안에서 CSV 파일을 찾을 수 없습니다."
            )

        csv_name = csv_files[0]

        with archive.open(csv_name) as bike_file:

            # 인코딩 확인
            encoding = None

            for enc in ["cp949", "utf-8-sig", "utf-8"]:
                try:
                    bike_file.seek(0)
                    header = pd.read_csv(
                        bike_file,
                        encoding=enc,
                        nrows=0
                    )
                    encoding = enc
                    break
                except Exception:
                    continue

            if encoding is None:
                raise ValueError(
                    "따릉이 CSV 파일의 인코딩을 확인할 수 없습니다."
                )

            if "대여일시" not in header.columns:
                raise ValueError(
                    "따릉이 자료에 '대여일시' 열이 없습니다."
                )

            bike_file.seek(0)

            # 큰 파일을 여러 묶음으로 나누어 처리
            progress = st.progress(0)
            status = st.empty()

            chunk_size = 300000

            for i, chunk in enumerate(
                pd.read_csv(
                    bike_file,
                    encoding=encoding,
                    usecols=["대여일시"],
                    chunksize=chunk_size
                )
            ):

                dates = pd.to_datetime(
                    chunk["대여일시"],
                    errors="coerce"
                )

                daily = (
                    dates.dt.strftime("%Y-%m-%d")
                    .value_counts()
                )

                daily_counts.append(daily)

                status.text(
                    f"따릉이 대여자료 처리 중... {i + 1}번째 묶음"
                )

            progress.progress(100)
            status.text("따릉이 자료 처리 완료!")

    if not daily_counts:
        raise ValueError("따릉이 대여자료가 비어 있습니다.")

    result = (
        pd.concat(daily_counts)
        .groupby(level=0)
        .sum()
        .rename("따릉이_대여건수")
        .reset_index()
    )

    result = result.rename(
        columns={"index": "날짜"}
    )

    return result


def load_bike_from_csv(uploaded_file):
    """일반 CSV 형태의 따릉이 대여이력을 집계합니다."""

    daily_counts = []

    uploaded_file.seek(0)

    encoding = None

    for enc in ["cp949", "utf-8-sig", "utf-8"]:
        try:
            uploaded_file.seek(0)
            header = pd.read_csv(
                uploaded_file,
                encoding=enc,
                nrows=0
            )
            encoding = enc
            break
        except Exception:
            continue

    if encoding is None:
        raise ValueError("따릉이 CSV 인코딩을 확인할 수 없습니다.")

    if "대여일시" not in header.columns:
        raise ValueError("따릉이 자료에 '대여일시' 열이 없습니다.")

    uploaded_file.seek(0)

    progress = st.progress(0)
    status = st.empty()

    for i, chunk in enumerate(
        pd.read_csv(
            uploaded_file,
            encoding=encoding,
            usecols=["대여일시"],
            chunksize=300000
        )
    ):

        dates = pd.to_datetime(
            chunk["대여일시"],
            errors="coerce"
        )

        daily = dates.dt.strftime(
            "%Y-%m-%d"
        ).value_counts()

        daily_counts.append(daily)

        status.text(
            f"따릉이 대여자료 처리 중... {i + 1}번째 묶음"
        )

    progress.progress(100)
    status.text("따릉이 자료 처리 완료!")

    result = (
        pd.concat(daily_counts)
        .groupby(level=0)
        .sum()
        .rename("따릉이_대여건수")
        .reset_index()
    )

    result = result.rename(
        columns={"index": "날짜"}
    )

    return result


# --------------------------------------------------
# 파일 업로드
# --------------------------------------------------

st.header("1. 분석 자료 업로드")

col1, col2 = st.columns(2)

with col1:
    weather_file = st.file_uploader(
        "기상청 일별 관측자료 CSV",
        type=["csv"],
        key="weather"
    )

with col2:
    bike_file = st.file_uploader(
        "따릉이 대여이력 ZIP 또는 CSV",
        type=["zip", "csv"],
        key="bike"
    )


# --------------------------------------------------
# 데이터 분석 시작
# --------------------------------------------------

if weather_file is not None and bike_file is not None:

    try:

        with st.spinner("기상자료를 읽는 중입니다..."):
            weather = load_weather(weather_file)

        with st.spinner("따릉이 대여자료를 분석하는 중입니다..."):

            if bike_file.name.lower().endswith(".zip"):
                bike = load_bike_from_zip(bike_file)
            else:
                bike = load_bike_from_csv(bike_file)

        # 날짜 기준으로 두 자료 결합
        data = pd.merge(
            weather,
            bike,
            on="날짜",
            how="inner"
        )

        data["날짜"] = pd.to_datetime(data["날짜"])

        # 2026년 6월만 분석
        data = data[
            (data["날짜"] >= "2026-06-01")
            & (data["날짜"] <= "2026-06-30")
        ].copy()

        data = data.sort_values("날짜")

        if len(data) < 3:
            st.error(
                "분석할 날짜가 부족합니다. "
                "두 자료의 날짜가 서로 일치하는지 확인해 주세요."
            )
            st.stop()

        # 요일 계산
        weekday_names = {
            0: "월요일",
            1: "화요일",
            2: "수요일",
            3: "목요일",
            4: "금요일",
            5: "토요일",
            6: "일요일"
        }

        data["요일"] = (
            data["날짜"].dt.dayofweek.map(weekday_names)
        )

        # 비 여부
        data["비여부"] = np.where(
            data["일강수량(mm)"] > 0,
            "비가 온 날",
            "비가 오지 않은 날"
        )

        st.success(
            f"데이터 결합 완료! 총 {len(data)}일의 자료를 분석합니다."
        )

        # --------------------------------------------------
        # 기본 데이터 확인
        # --------------------------------------------------

        st.header("2. 분석 자료 확인")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "분석 날짜",
            f"{len(data)}일"
        )

        c2.metric(
            "총 따릉이 대여",
            f"{data['따릉이_대여건수'].sum():,}건"
        )

        c3.metric(
            "비가 온 날",
            f"{(data['일강수량(mm)'] > 0).sum()}일"
        )

        c4.metric(
            "비가 오지 않은 날",
            f"{(data['일강수량(mm)'] == 0).sum()}일"
        )

        st.subheader("날짜별 분석 데이터")

        display_columns = [
            "날짜",
            "요일",
            "따릉이_대여건수",
            "일강수량(mm)",
            "비여부",
            "평균기온(°C)"
        ]

        display_columns = [
            c for c in display_columns
            if c in data.columns
        ]

        st.dataframe(
            data[display_columns],
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------
        # 비가 온 날과 오지 않은 날 비교
        # --------------------------------------------------

        st.header("3. 비가 온 날과 오지 않은 날의 이용량 비교")

        rain_comparison = (
            data.groupby("비여부")["따릉이_대여건수"]
            .agg(["mean", "count"])
            .reset_index()
        )

        rain_comparison.columns = [
            "날씨",
            "평균 대여 건수",
            "날짜 수"
        ]

        no_rain = data[
            data["일강수량(mm)"] == 0
        ]["따릉이_대여건수"]

        rain = data[
            data["일강수량(mm)"] > 0
        ]["따릉이_대여건수"]

        a, b = st.columns(2)

        with a:
            st.metric(
                "비가 오지 않은 날 평균 대여",
                f"{no_rain.mean():,.0f}건"
                if len(no_rain) else "자료 없음"
            )

        with b:
            st.metric(
                "비가 온 날 평균 대여",
                f"{rain.mean():,.0f}건"
                if len(rain) else "자료 없음"
            )

        if len(no_rain) > 0 and len(rain) > 0:
            difference = (
                (rain.mean() - no_rain.mean())
                / no_rain.mean()
                * 100
            )

            st.write(
                f"비가 온 날의 평균 대여량은 "
                f"비가 오지 않은 날보다 **{abs(difference):.1f}% "
                f"{'적습니다.' if difference < 0 else '많습니다.'}**"
            )

        fig_compare = px.bar(
            rain_comparison,
            x="날씨",
            y="평균 대여 건수",
            color="날씨",
            text_auto=".0f",
            title="비 여부에 따른 평균 따릉이 대여 건수",
            labels={
                "날씨": "날씨",
                "평균 대여 건수": "평균 대여 건수"
            }
        )

        st.plotly_chart(
            fig_compare,
            use_container_width=True
        )

        # --------------------------------------------------
        # 단순 선형회귀
        # --------------------------------------------------

        st.header("4. 단순 선형회귀 분석")

        st.markdown("""
        강수량이 따릉이 대여 건수에 어떤 관계가 있는지 분석합니다.

        - 독립변수 X: 하루 강수량(mm)
        - 종속변수 Y: 하루 따릉이 대여 건수
        """)

        X = data[["일강수량(mm)"]]
        y = data["따릉이_대여건수"]

        if X["일강수량(mm)"].nunique() < 2:
            st.warning(
                "강수량 값의 차이가 없어 회귀선을 계산할 수 없습니다."
            )
        else:

            simple_model = LinearRegression()
            simple_model.fit(X, y)

            predictions = simple_model.predict(X)

            slope = simple_model.coef_[0]
            intercept = simple_model.intercept_

            r2 = r2_score(y, predictions)
            mae = mean_absolute_error(y, predictions)
            mse = mean_squared_error(y, predictions)

            r = np.corrcoef(
                data["일강수량(mm)"],
                data["따릉이_대여건수"]
            )[0, 1]

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "회귀계수",
                f"{slope:,.1f}"
            )

            m2.metric(
                "결정계수 R²",
                f"{r2:.3f}"
            )

            m3.metric(
                "상관계수",
                f"{r:.3f}"
            )

            m4.metric(
                "평균 절대 오차 MAE",
                f"{mae:,.0f}"
            )

            st.write("**회귀식**")

            st.latex(
                rf"\hat{{y}} = {intercept:,.1f} "
                rf"+ ({slope:,.1f})x"
            )

            st.write(
                f"강수량이 1mm 증가할 때 회귀식은 "
                f"따릉이 대여 건수가 약 **{abs(slope):,.1f}건 "
                f"{'감소' if slope < 0 else '증가'}**하는 관계를 나타냅니다."
            )

            # 회귀선 그래프
            x_line = np.linspace(
                data["일강수량(mm)"].min(),
                data["일강수량(mm)"].max(),
                100
            )

            y_line = simple_model.predict(
                pd.DataFrame({
                    "일강수량(mm)": x_line
                })
            )

            fig_reg = go.Figure()

            fig_reg.add_trace(
                go.Scatter(
                    x=data["일강수량(mm)"],
                    y=data["따릉이_대여건수"],
                    mode="markers",
                    name="실제 데이터",
                    text=data["날짜"].dt.strftime("%Y-%m-%d"),
                    hovertemplate=(
                        "날짜: %{text}<br>"
                        "강수량: %{x}mm<br>"
                        "대여 건수: %{y:,}건"
                        "<extra></extra>"
                    )
                )
            )

            fig_reg.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name="선형회귀선"
                )
            )

            fig_reg.update_layout(
                title="강수량과 따릉이 대여 건수의 선형회귀",
                xaxis_title="일강수량(mm)",
                yaxis_title="따릉이 대여 건수",
                hovermode="closest"
            )

            st.plotly_chart(
                fig_reg,
                use_container_width=True
            )

            st.caption(
                "R²는 회귀모델이 대여 건수의 변동을 설명하는 정도입니다. "
                "높을수록 이 자료에서 회귀선의 설명력이 높다는 뜻입니다."
            )

        # --------------------------------------------------
        # 다중 선형회귀
        # --------------------------------------------------

        st.header("5. 다중 선형회귀 분석")

        st.markdown("""
        따릉이 이용량은 비뿐만 아니라 기온과 요일에도 영향을 받을 수 있습니다.

        따라서 평균기온과 요일을 함께 고려하여
        강수량의 관계를 다시 분석합니다.
        """)

        if "평균기온(°C)" not in data.columns:
            st.warning(
                "평균기온 자료가 없어 다중회귀 분석을 진행할 수 없습니다."
            )

        else:

            multi_data = data[
                [
                    "따릉이_대여건수",
                    "일강수량(mm)",
                    "평균기온(°C)",
                    "요일"
                ]
            ].dropna()

            if len(multi_data) < 5:
                st.warning(
                    "다중회귀 분석에 필요한 데이터가 부족합니다."
                )

            else:

                # 요일을 숫자형 변수로 변환
                weekday_dummy = pd.get_dummies(
                    multi_data["요일"],
                    prefix="요일",
                    drop_first=True,
                    dtype=int
                )

                X_multi = pd.concat(
                    [
                        multi_data[
                            [
                                "일강수량(mm)",
                                "평균기온(°C)"
                            ]
                        ].reset_index(drop=True),
                        weekday_dummy.reset_index(drop=True)
                    ],
                    axis=1
                )

                y_multi = multi_data[
                    "따릉이_대여건수"
                ]

                multi_model = LinearRegression()
                multi_model.fit(X_multi, y_multi)

                multi_predictions = multi_model.predict(
                    X_multi
                )

                multi_r2 = r2_score(
                    y_multi,
                    multi_predictions
                )

                multi_mae = mean_absolute_error(
                    y_multi,
                    multi_predictions
                )

                multi_mse = mean_squared_error(
                    y_multi,
                    multi_predictions
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "다중회귀 R²",
                    f"{multi_r2:.3f}"
                )

                c2.metric(
                    "다중회귀 MAE",
                    f"{multi_mae:,.0f}건"
                )

                c3.metric(
                    "다중회귀 MSE",
                    f"{multi_mse:,.0f}"
                )

                st.subheader("다중회귀 변수별 회귀계수")

                coefficient_table = pd.DataFrame({
                    "변수": X_multi.columns,
                    "회귀계수": multi_model.coef_
                })

                coefficient_table = pd.concat(
                    [
                        pd.DataFrame({
                            "변수": ["절편"],
                            "회귀계수": [multi_model.intercept_]
                        }),
                        coefficient_table
                    ],
                    ignore_index=True
                )

                st.dataframe(
                    coefficient_table,
                    use_container_width=True,
                    hide_index=True
                )

                rain_coefficient = (
                    multi_model.coef_[
                        list(X_multi.columns).index("일강수량(mm)")
                    ]
                )

                st.write(
                    f"평균기온과 요일을 함께 고려했을 때, "
                    f"강수량의 회귀계수는 **{rain_coefficient:,.1f}**입니다."
                )

                if rain_coefficient < 0:
                    st.info(
                        "다른 변수들을 함께 고려한 모델에서도 "
                        "강수량이 많을수록 대여 건수가 감소하는 관계가 나타났습니다."
                    )
                elif rain_coefficient > 0:
                    st.info(
                        "다른 변수들을 함께 고려한 모델에서는 "
                        "강수량이 많을수록 대여 건수가 증가하는 관계가 나타났습니다."
                    )
                else:
                    st.info(
                        "다른 변수들을 함께 고려했을 때 "
                        "강수량의 선형적인 관계가 거의 나타나지 않았습니다."
                    )

                # 실제값과 예측값 비교
                prediction_data = pd.DataFrame({
                    "실제 대여 건수": y_multi,
                    "예측 대여 건수": multi_predictions
                })

                fig_prediction = px.scatter(
                    prediction_data,
                    x="실제 대여 건수",
                    y="예측 대여 건수",
                    title="다중회귀 실제값과 예측값 비교"
                )

                min_value = min(
                    prediction_data.min().min(),
                    prediction_data.max().min()
                )

                max_value = max(
                    prediction_data.max().max(),
                    prediction_data.min().max()
                )

                fig_prediction.add_trace(
                    go.Scatter(
                        x=[min_value, max_value],
                        y=[min_value, max_value],
                        mode="lines",
                        name="완벽한 예측 기준선"
                    )
                )

                st.plotly_chart(
                    fig_prediction,
                    use_container_width=True
                )

        # --------------------------------------------------
        # 결론 및 데이터 다운로드
        # --------------------------------------------------

        st.header("6. 분석 결과 정리")

        st.markdown("""
        다음 내용을 중심으로 수행평가 결론을 작성할 수 있습니다.

        1. 비가 온 날과 오지 않은 날의 평균 대여 건수 차이
        2. 강수량과 따릉이 대여 건수의 상관관계
        3. 단순 선형회귀에서 강수량의 회귀계수와 결정계수
        4. 평균기온과 요일을 고려한 다중회귀 결과
        5. 분석 결과가 실제로 비의 영향이라고 단정할 수 있는지
        """)

        st.warning(
            "주의: 이 자료는 2026년 6월 한 달, 총 30일의 관측자료입니다. "
            "회귀분석은 변수 사이의 관계를 보여줄 뿐, 비가 대여량 감소의 "
            "유일한 원인이라는 것을 증명하지는 않습니다. "
            "또한 대여 건수는 이용률 자체가 아니라 이용량을 나타냅니다."
        )

        csv_data = data.to_csv(
            index=False,
            encoding="utf-8-sig"
        )

        st.download_button(
            label="분석에 사용한 데이터 CSV 다운로드",
            data=csv_data,
            file_name="seoul_bike_weather_analysis_202606.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(
            f"분석 중 오류가 발생했습니다: {e}"
        )

        st.info(
            "기상청 일별 CSV와 따릉이 대여이력 ZIP 파일이 "
            "올바른 자료인지 확인해 주세요."
        )

else:

    st.info(
        "위쪽에서 기상청 일별 관측자료 CSV와 "
        "따릉이 대여이력 ZIP 파일을 모두 업로드하면 분석이 시작됩니다."
    )
