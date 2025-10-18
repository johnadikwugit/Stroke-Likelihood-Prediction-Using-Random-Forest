import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from sklearn.preprocessing import LabelEncoder

# ---- helpers ----
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Replace missing data in bmi with the mean
    df['bmi'].fillna(df['bmi'].mean(), inplace=True)
    return df

class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode
        self.encoders = {}

    def fit(self, X, y=None):
        self.encoders = {}
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            self.encoders[col] = LabelEncoder().fit(X[col])
        return self

    def transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].transform(X[col])
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].inverse_transform(X[col])
        return output


def app():
    st.title(" Predicting Stroke likelihood :orange[Dashboard] :bar_chart:")
    st.write("You are in 'dashboard' page.")

    # ---------- data ----------
    df = load_data("healthcare-dataset-stroke-data.csv")

    # Categorical / continuous
    cat_cols = [
        'gender', 'hypertension', 'heart_disease', 'ever_married',
        'work_type', 'Residence_type', 'smoking_status', 'stroke'
    ]
    cont_cols = ['age', 'avg_glucose_level', 'bmi']

    # Correlation candidates (after encoding, these will all be numeric)
    cat_cols_correlation = [
        'gender', 'hypertension', 'heart_disease', 'ever_married',
        'work_type', 'Residence_type', 'smoking_status', 'stroke',
        'age', 'avg_glucose_level', 'bmi'
    ]

    # Frame subsets
    df_cat = df[cat_cols]
    df_cont = df[cont_cols]

    # ---------- sidebar ----------
    # Stroke pie
    colors = ['#FFA500', '#6AC0F8']
    table2 = px.pie(
        df,
        names='stroke',
        color='stroke',
        color_discrete_sequence=colors,
        template='ggplot2',
        title="Stroke Percentage"
    )

    st.sidebar.markdown("## Target Variable")
    st.sidebar.plotly_chart(table2, use_container_width=True)
    st.sidebar.markdown("## Settings")

    # IMPORTANT: do not allow selecting 'stroke' as the category (prevents duplicate in groupby)
    cat_cols_no_target = [c for c in cat_cols if c != 'stroke']
    cat_selected = st.sidebar.selectbox('Categorical Variables', cat_cols_no_target)
    cont_selected = st.sidebar.selectbox('Continuous Variables', cont_cols)
    cont_multi_selected = st.sidebar.multiselect(
        'Correlation Matrix',
        cat_cols_correlation,
        default=cat_cols_correlation
    )

    # ---------- categorical distribution (stacked bars) ----------
    # Safe groupby (no duplicate keys); rename before reset_index to avoid collisions
    df_cat_counts = (
        df_cat.groupby([cat_selected, 'stroke'])
              .size()
              .rename('count')
              .reset_index()
    )

    df_cat_counts_0 = df_cat_counts[df_cat_counts['stroke'] == 0]
    df_cat_counts_1 = df_cat_counts[df_cat_counts['stroke'] == 1]

    fig1 = go.Figure(data=[
        go.Bar(name='stroke=0', x=df_cat_counts_0[cat_selected], y=df_cat_counts_0['count']),
        go.Bar(name='stroke=1', x=df_cat_counts_1[cat_selected], y=df_cat_counts_1['count'])
    ])
    fig1.update_layout(
        height=400,
        width=500,
        margin={'l': 50, 'r': 50, 't': 50, 'b': 50},
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        barmode='stack',
        xaxis_title='Category',
        yaxis_title='Count'
    )

    # ---------- continuous distribution (by stroke) ----------
    li_cont0 = df[df['stroke'] == 0][cont_selected].values.tolist()
    li_cont1 = df[df['stroke'] == 1][cont_selected].values.tolist()
    cont_data = [li_cont0, li_cont1]
    group_labels = ['stroke=0', 'stroke=1']

    fig2 = ff.create_distplot(
        cont_data,
        group_labels,
        show_hist=False,
        show_rug=False
    )
    fig2.update_layout(
        height=400,
        width=500,
        margin={'l': 20, 'r': 20, 't': 0, 'b': 0},
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        xaxis_title='Distribution',
        yaxis_title='Measure'
    )

    # ---------- correlation matrix ----------
    # Encode selected categoricals to numeric for correlation
    multi = MultiColumnLabelEncoder(
        columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    )
    df_encoded = multi.fit_transform(df.copy())

    # Keep only requested columns and compute corr
    corr = df_encoded[cont_multi_selected].corr().round(3)
    fig_corr = go.Figure([
        go.Heatmap(
            z=corr.values,
            x=corr.index.values,
            y=corr.columns.values
        )
    ])
    fig_corr.update_layout(
        height=300,
        width=1200,
        margin={'l': 20, 'r': 20, 't': 0, 'b': 0}
    )

    # ---------- layout ----------
    left_column, right_column = st.columns(2)
    left_column.subheader('Categorical Variable Distribution: ' + cat_selected)
    right_column.subheader('Continuous Variable Distribution: ' + cont_selected)
    left_column.plotly_chart(fig1, use_container_width=True)
    right_column.plotly_chart(fig2, use_container_width=True)

    st.subheader('Correlation Matrix')
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("### Dataset Statistics")
    st.write(df.describe())
