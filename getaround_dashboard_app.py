import streamlit as st
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from typing import Tuple, List
from pydantic import BaseModel,Field
from pydantic_core import PydanticUndefined
from typing import Literal, get_origin, get_args

import os
from pathlib import Path

import random

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from getaround_schemas.price_predict import RentalPricePredictInput
import requests

import statsmodels.api as sm

st.set_page_config(
    page_title="Getaround Insights Dashboard",
    page_icon="🚗 ",
    layout="wide"
)

DATA_FOLDER_PATH = "data"
base_dir = os.path.dirname(__file__)
#DELAY_ANALYSIS_XL_FILE_PATH = os.path.join("..",DATA_FOLDER_PATH, "get_around_delay_analysis.xlsx")
#DELAY_ANALYSIS_XL_FILE_PATH = os.path.join(base_dir,'..', DATA_FOLDER_PATH, 'get_around_delay_analysis.xlsx')
DELAY_ANALYSIS_XL_FILE_PATH = os.path.join(DATA_FOLDER_PATH, 'get_around_delay_analysis.xlsx')

# Constants
DECIMAL_FORMAT_2 = "%.2f"
def format_2_decimals(number) -> str:
    return (DECIMAL_FORMAT_2 % number)

def decimal_format_str(nb_decimals:int) -> str:
    return "{:."+str(nb_decimals)+"f}"

def format_decimals(number, nb_decimals:int) ->str:
    return decimal_format_str(nb_decimals).format(number)

CHART_VALUES_COUNT = 12
CHART_STEP = 60
CHART_CUSTOM_BINS = np.arange(CHART_STEP, (CHART_VALUES_COUNT + 1) * CHART_STEP, CHART_STEP)

def generate_cumulative_rentals_by_threshold_chart(delay_analysis_delta_with_previous_dtf:pd.DataFrame, point_index:int) -> Tuple[go.Figure, Tuple[float,float,float]]:

    col = delay_analysis_delta_with_previous_dtf['time_delta_with_previous_rental_in_minutes']
    max_val = col.max()
    custom_bins = np.arange(0, max_val + CHART_STEP, CHART_STEP)

    hist, bin_edges = np.histogram(col, bins=custom_bins, density=False)
    cumulative_counts = np.cumsum(hist)

    proportions = cumulative_counts / cumulative_counts[-1]  # Normalize to 1
    # Use midpoints for x values
    bin_midpoints = 0.5 * (np.array(bin_edges[:-1]) + np.array(bin_edges[1:]))

    #base = delay_analysis_delta_with_previous_dtf.shape[0]
    #counts = proportions * base
    #print("counts")
    #print(counts)

    fig = go.Figure()

    # Primary Y-Axis (Proportion)
    fig.add_trace(go.Scatter(
        x=bin_midpoints,
        y=proportions,
        name="Cumulative Proportion",
        yaxis="y1",
        mode="lines+markers"
    ))

    # Secondary Y-Axis (Proportion × constant)
    fig.add_trace(go.Scatter(
        x=bin_midpoints,
        y=cumulative_counts,
        name="Cumulative Count",
        yaxis="y2",
        mode="lines+markers",
        line=dict(dash='dot')
    ))

    fig.add_vline(x=bin_midpoints[point_index], line_width=2, line_dash="dash", line_color="red")

    # Layout with dual y-axes
    fig.update_layout(
        title="Cumulative Distribution of rentals affected by Threshold",
        xaxis=dict(title="Threshold - Time delta with previous rental (minutes)"),
        yaxis=dict(title="Proportion", tickformat=".0%"),
        yaxis2=dict(
            title="Count",
            overlaying="y",
            side="right"
        ),
        #width=800,
        #height=500,
        legend=dict(x=0.5, y=1.1, xanchor="center", orientation="h")
    )
    return (fig, (bin_midpoints[point_index],cumulative_counts[point_index],proportions[point_index] ))


def generate_cumulative_rentals_by_threshold_and_scope_chart(delay_analysis_dataframe:pd.DataFrame,distrib_column:str
                                                               ,chart_title:str, xaxis_title:str, yaxis_title:str, point_index:int, quantile_zoom:float= None ) -> Tuple[go.Figure, List[Tuple[str,float,float,float]]]:
    
    col = delay_analysis_dataframe[distrib_column]
    max_val = col.max()
    custom_bins = np.arange(0, max_val + CHART_STEP, CHART_STEP)
    bin_midpoints = 0.5 * (custom_bins[:-1] + custom_bins[1:])

    scopes = sorted(delay_analysis_dataframe["checkin_type"].unique())
    n_rows = len(scopes)

    cumulative_counts_list = []
    cumulative_props_list = []
    title_list = []
    y_max_list = []
    output_data_points: List[Tuple[str, float, float,float]] = []

    # préparer les données pour chaque scope et pour les sample points
    for i, scope in enumerate(scopes):
        data = delay_analysis_dataframe[delay_analysis_dataframe["checkin_type"] == scope][distrib_column]
        hist, _ = np.histogram(data, bins=custom_bins, density=False)
        cumulative_counts = np.cumsum(hist)
        proportions = cumulative_counts / cumulative_counts[-1]
        total = cumulative_counts[-1]

        cumulative_counts_list.append(cumulative_counts)
        cumulative_props_list.append(proportions)
        title_list.append(f"{scope} (total : {total})")
        y_max_list.append(total)
        output_data_points.append((scope, bin_midpoints[point_index], cumulative_counts[point_index],proportions[point_index]))

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=title_list,
        specs=[[{"secondary_y": True}] for _ in range(n_rows)]  # <--- clé pour avoir double axe
    )

    for i, scope in enumerate(scopes):
        row = i + 1

        # Courbe prop cumul
        fig.add_trace(
            go.Scatter(
                x=bin_midpoints,
                y=cumulative_props_list[i],
                mode="lines+markers",
                name=f"{scope} (proportion)",
                customdata=cumulative_counts_list[i],
                hovertemplate=(
                    "Seuil (min): %{x}<br>"
                    "Cumulated proportion: %{y:.1%}<br>"
                    "Cumulated count: %{customdata}<extra></extra>"
                )
            ),
            row=row,
            col=1,
            secondary_y=False
        )

        fig.add_shape(
            type="line",
            x0=bin_midpoints[point_index], x1=bin_midpoints[point_index],
            y0=0, y1=y_max_list[i],
            line=dict(color="red", dash="dash"),
            xref=f"x{row}" if row > 1 else "x",
            yref=f"y{row+ n_rows}" if row > 1 else "y2",  # <--- référence à l’axe secondaire
            row=row, col=1
        )

        # Courbe cumul
        fig.add_trace(
            go.Scatter(
                x=bin_midpoints,
                y=cumulative_counts_list[i],
                mode="lines",
                line=dict(dash="dot"),
                name=f"{scope} (count)"
            ),
            row=row,
            col=1,
            secondary_y=True
        )

    fig.update_layout(
        height=300 * n_rows,
        title_text=chart_title,
        xaxis_title=xaxis_title,
        template="plotly_white"
    )

    # Mise en forme des axes Y
    for i in range(n_rows):
        fig.update_yaxes(title_text="Proportion", tickformat=".0%", row=i+1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Count", row=i+1, col=1, secondary_y=True)

    # Couper au quantile quantile_zoom (utile lorsqu'il y a un palier à cause d'outliers)
    if quantile_zoom is not None:
        xmax = col.quantile(quantile_zoom)
        fig.update_xaxes(range=[0, xmax])

    return (fig, output_data_points)

def compute_rentals_with_previous_rental_dtf(delay_analysis_dtf:pd.DataFrame) -> pd.DataFrame:
    return delay_analysis_dtf.loc[delay_analysis_dtf['previous_ended_rental_id'].notnull()]


def compute_rentals_with_previous_rental_join_dtf(delay_analysis_dtf:pd.DataFrame) -> pd.DataFrame:
    rentals_with_previous_rental = delay_analysis_dtf.loc[delay_analysis_dtf['previous_ended_rental_id'].notnull()]
    return pd.merge(delay_analysis_dtf,rentals_with_previous_rental, left_on=['rental_id'], right_on=['previous_ended_rental_id'], how='inner', suffixes=['_previous', '_next'])

def compute_drivers_late_next_checkin_join_dtf(delay_analysis_dtf:pd.DataFrame) -> pd.DataFrame:
    rentals_with_previous_rental = compute_rentals_with_previous_rental_dtf(delay_analysis_dtf)
    rentals_with_delay_checkout_dtf = delay_analysis_dtf.loc[delay_analysis_dtf['delay_at_checkout_in_minutes'] > 0 ]

    drivers_late_next_checking_dtf = pd.merge(rentals_with_delay_checkout_dtf,rentals_with_previous_rental, left_on=['rental_id'], right_on=['previous_ended_rental_id'], how='inner', suffixes=['_previous', '_next'])
    return drivers_late_next_checking_dtf.loc[drivers_late_next_checking_dtf['delay_at_checkout_in_minutes_previous'] > drivers_late_next_checking_dtf['time_delta_with_previous_rental_in_minutes_next']]


def test_diff_proportions(df, subset_cond, success_cond):
    """
    Test if success proportion in a subset is
    significantly different from that of global set.
    """
    # Global
    n_D = len(df)
    x_D = success_cond.sum()

    # Subset
    subset = df[subset_cond]
    n_S = len(subset)
    x_S = success_cond[subset_cond].sum()

    # Z-test pour proportions
    count = [x_S, x_D]
    nobs = [n_S, n_D]
    stat, pval = sm.stats.proportions_ztest(count, nobs)

    return {
        "prop_global": x_D / n_D if n_D > 0 else None,
        "prop_subset": x_S / n_S if n_S > 0 else None,
        "p_value": pval
    }

st.title("Getaround insight")
st.subheader("Getaround delay analysis", divider=True)

@st.cache_data()
def load_data() -> pd.DataFrame:
    try:
        session = boto3.Session(aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"), 
                                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
        s3 = session.resource("s3")
        bucket = s3.Bucket("pfe-getaround-bucket")
        DATA_FOLDER_PATH = "data"
        FILE_NAME = 'get_around_delay_analysis.xlsx'
        
        local_path = f"/tmp/{FILE_NAME}"
        if not os.path.exists(local_path):
            bucket.download_file("".join([DATA_FOLDER_PATH, "/",FILE_NAME]), local_path) 
        dtf =  pd.read_excel(local_path)
        print(f" ✅ {FILE_NAME} downloaded successfully")
        return dtf
    except (BotoCoreError, ClientError) as s3_err:
        print(f"❌ S3 error while downloading {FILE_NAME}: {s3_err}")
        return None
    except ValueError as ve:
        print(f"❌ Pandas failed to parse Excel file {FILE_NAME}: {ve}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error while loading {FILE_NAME}: {e}")
        return None
        

data_load_state = st.text('Loading data...')
delay_analysis_dtf = load_data()
data_load_state.text("")

sample_size = 10

if st.checkbox('Show sample'):
    n = 10  # Number of values
    step = 10
    sample_size = st.selectbox(label='Select sample size', options=np.arange(step, (n + 1) * step, step)) 
    st.subheader('sample')
    st.write(delay_analysis_dtf.head(sample_size))   

st.markdown(f'Dataset size : {delay_analysis_dtf.shape[0]}')

col1, col2 = st.columns(2)
with col1:
    st.markdown('**Checkin Type distribution**')
    st.write((
        delay_analysis_dtf['checkin_type']
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .reset_index()
    ))
    
with col2:
    st.markdown('**State distribution**')
    st.write((
        delay_analysis_dtf['state']
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .reset_index()
    ))

#st.markdown("---")
st.subheader("Impacts of a minimum delay between two rentals", divider='grey')
delay_analysis_delta_with_previous_dtf = delay_analysis_dtf.query("time_delta_with_previous_rental_in_minutes.notnull()")
#st.markdown("---")
#st.markdown("<br>", unsafe_allow_html=True)
#st.subheader('')
st.markdown('<h5> Which share of our owner’s revenue would potentially be affected by the feature ? </h5>', unsafe_allow_html=True)
#st.subheader('Which share of our owner’s revenue would potentially be affected by the feature ?' )
st.markdown('<span style="color: var(--primary-color)">The Proportion of rentals with a previous rental and with a delay < 12h with previous rentals helps answer this question</span>',unsafe_allow_html=True)
st.markdown('<span style="color: var(--primary-color)">*To estimate owner share revenue proportion we discard canceled rentals</span>',unsafe_allow_html=True)
#st.markdown('**time_delta_with_previous_rental_in_minutes**')
#st.markdown('Difference in minutes between this rental planned start time and the previous rental planned end time (when lower than 12 hours, NULL if higher)')
proportion_delta_with_previous = 100*delay_analysis_delta_with_previous_dtf.shape[0]/delay_analysis_dtf.shape[0]
proportion_delta_with_previous_formatted = format_decimals(proportion_delta_with_previous,2)

valid_rentals = delay_analysis_dtf.loc[delay_analysis_dtf['state'] != 'canceled']
chained_rentals = valid_rentals.loc[valid_rentals['previous_ended_rental_id'].notnull()]
owners_share_impacted_formatted = format_decimals(100*chained_rentals.shape[0]/valid_rentals.shape[0],2)

proportion_delay_analysis_delta_with_previous = chained_rentals.shape[0]/valid_rentals.shape[0]
st.markdown(f'Owners revenue share impacted : <span style="color: var(--primary-color); font-weight:bold;">{owners_share_impacted_formatted} % </span>',unsafe_allow_html=True)

st.markdown('<h5> How many rentals would be affected by the feature depending on the threshold and scope we choose ? </h5>', unsafe_allow_html=True)
st.markdown('Rentals affected by the threshold : if delay1 < threshold.')
st.markdown('Hence if delay2 < delay1 < threshold => rentals with delay2 also concerned, so cumulative histogram makes sense')
rental_distribution_delay_threshold_col, rental_distribution_delay_threshold_scope_col = st.columns(2)

with rental_distribution_delay_threshold_col:
    sample_index= 3
    fig, sample = generate_cumulative_rentals_by_threshold_chart(delay_analysis_delta_with_previous_dtf,sample_index)
    st.plotly_chart(fig, use_container_width=True)
    sample_percentage = 100*sample[2]
    sample_percentage_formatted = format_decimals(sample_percentage,2)
    whole_population_percentage_formatted = format_decimals(sample[2]*proportion_delta_with_previous,2)
    st.markdown(f"<div style='margin-left:30px;'>✅With a threshold of {int(sample[0])} minutes, approximatively {int(sample[1])} rentals are concerned. It represents {sample_percentage_formatted} % of the rentals with a previous rental, and {sample_percentage_formatted}% * {proportion_delta_with_previous_formatted}% (={whole_population_percentage_formatted}%) of the whole population.</div>", unsafe_allow_html=True)

with rental_distribution_delay_threshold_scope_col:
    sample_index= 3
    fig, data_points = generate_cumulative_rentals_by_threshold_and_scope_chart(delay_analysis_dataframe=delay_analysis_delta_with_previous_dtf
                                                                     ,distrib_column='time_delta_with_previous_rental_in_minutes'
                                                                     ,chart_title='Cumulative Distribution of rentals affected by Threshold / scope'
                                                                     ,xaxis_title='Threshold - Time delta with previous rental (minutes)'
                                                                     ,yaxis_title="Cumulative count",point_index=sample_index)
    st.plotly_chart(fig, use_container_width=True)   
    for i in range(0,len(data_points)):
        scope_sample = data_points[i]
        sample_percentage = 100*scope_sample[3]
        sample_percentage_formatted = format_decimals(sample_percentage,2)
        whole_population_percentage_formatted = format_decimals(int(scope_sample[2])*100/delay_analysis_dtf.shape[0],2)
        st.markdown(f"<div style='margin-left:30px;'>✅ for {scope_sample[0]}, with a threshold of {int(scope_sample[1])} minutes, approximatively {int(scope_sample[2])} rentals are concerned.It represents {sample_percentage_formatted} % of the rentals with a previous rental (with scope {scope_sample[0]}), and {whole_population_percentage_formatted}% of the whole population.</div>", unsafe_allow_html=True)


st.markdown("<br><br>", unsafe_allow_html=True)    
st.markdown('<h5> How often are drivers late for the next check-in? How does it impact the next driver ?  </h5>', unsafe_allow_html=True)
st.markdown('<h6> 1. How often are drivers late for the next check-in ?  </h6>', unsafe_allow_html=True)

drivers_late_next_checking_dtf = compute_drivers_late_next_checkin_join_dtf(delay_analysis_dtf)
rentals_with_previous_rental = compute_rentals_with_previous_rental_dtf(delay_analysis_dtf)
st.markdown(f'number of drivers late for next check-in : {drivers_late_next_checking_dtf.shape[0]}')
st.markdown(f'proportion of drivers late for next check-in : {format_decimals(100*drivers_late_next_checking_dtf.shape[0]/rentals_with_previous_rental.shape[0],2)} %')

col1, col2, col3 = st.columns([3, 6, 1])
with col1:
    st.markdown('**Checkin Type distribution**')
    st.write((
        drivers_late_next_checking_dtf['checkin_type_previous']
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .reset_index()
    ))

rentals_with_previous_rental_join = compute_rentals_with_previous_rental_join_dtf(delay_analysis_dtf)
subset_cond = ((rentals_with_previous_rental_join['delay_at_checkout_in_minutes_previous'] > 0 ) & (rentals_with_previous_rental_join['delay_at_checkout_in_minutes_previous'] > rentals_with_previous_rental_join['time_delta_with_previous_rental_in_minutes_next']))
test_diff_proportions(rentals_with_previous_rental_join
    , subset_cond = subset_cond
    , success_cond  = (rentals_with_previous_rental_join['checkin_type_previous'] == "mobile"))

st.markdown('✅ There is more chance for driver to be late for next-checkin when scope is <b> mobile </b> (z-test is significant)', unsafe_allow_html=True)

st.markdown('<h6> 2.How does it impact the next driver ?  </h6>', unsafe_allow_html=True)
st.markdown('Is it fair to assume that when driver is late for next check-in, then next driver has more chance to cancel his rental or to be late too for for next check-in.')
st.markdown("Let's compute canceling rate / dalay at checkout rate comparing with reference (drivers with previous checking) ")

#--------------------------------------------------------------------------------------------

next_driver_returning_late_z_test = test_diff_proportions(rentals_with_previous_rental_join
    , subset_cond=subset_cond
    , success_cond  = (rentals_with_previous_rental_join['delay_at_checkout_in_minutes_next'] > 0))

next_driver_cancel_z_test = test_diff_proportions(rentals_with_previous_rental_join
    , subset_cond=subset_cond
    , success_cond  = (rentals_with_previous_rental_join['state_next'] == "canceled"))

header = pd.MultiIndex.from_tuples([
    ("Metric","Name"),
    ("Metric","Within late drivers for next check-in"),
    ("Metric","Within all rentals with previous rentals (reference)"),
    ("Two-proportion significance test", "p-value"),
    ("Two-proportion significance test", "diff significative")
])

data = [
    ['proportion (%) of next drivers returning car late',
        format_decimals(float(next_driver_returning_late_z_test['prop_subset']),2),
        format_decimals(float(next_driver_returning_late_z_test['prop_global']),2),
        format_decimals(float(next_driver_returning_late_z_test['p_value']),5),
        str(float(next_driver_returning_late_z_test['p_value']) < 0.05)
    ]
    , ['proportion (%) of next drivers canceling their rental',
        format_decimals(float(next_driver_cancel_z_test['prop_subset']),2),
        format_decimals(float(next_driver_cancel_z_test['prop_global']),2),
        format_decimals(float(next_driver_cancel_z_test['p_value']),5),
        str(float(next_driver_cancel_z_test['p_value']) < 0.05)
    ]
]


df = pd.DataFrame(data, columns=header)

st.dataframe(df)  # or st.table(df)
st.markdown('✅ So when driver is late for next checkin, it has an impact on next driver (next driver has more chance to return car late)', unsafe_allow_html=True)
#-----------------------------------------------------------------------------------------------


st.markdown('<h5> How many problematic cases will it solve depending on the chosen threshold and scope ?  </h5>', unsafe_allow_html=True)
st.markdown("""We compute min_delay_between_rentals = delay_at_checkout_in_minutes_previous for rentals with driver late for next check-in.
Thus with this delay / threshold it solves problematic case (drivers late for next check-in).
Then by computing cumulative distribution function based on this new field / threshold, it gives the number of problematic cases solved by threshold""")

min_delay_threshold_col, empty_col = st.columns(2)
with min_delay_threshold_col:
    drivers_late_next_checking_dtf['min_delay_between_rentals'] = drivers_late_next_checking_dtf['time_delta_with_previous_rental_in_minutes_next'] + (drivers_late_next_checking_dtf['delay_at_checkout_in_minutes_previous']-drivers_late_next_checking_dtf['time_delta_with_previous_rental_in_minutes_next'])
    drivers_late_next_checking_threshold_dtf = drivers_late_next_checking_dtf.loc[:,['min_delay_between_rentals', 'checkin_type_previous' ]].copy()
    drivers_late_next_checking_threshold_dtf.rename(columns={'checkin_type_previous':'checkin_type'}, inplace=True)
    sample_index = 3
    fig, data_points = generate_cumulative_rentals_by_threshold_and_scope_chart(delay_analysis_dataframe=drivers_late_next_checking_threshold_dtf
                                                                        ,distrib_column='min_delay_between_rentals'
                                                                        ,chart_title='Cumulative Distribution of rentals potentialy solved by Threshold'
                                                                        ,xaxis_title='Threshold - Time delta with previous rental (minutes)'
                                                                        ,yaxis_title="Cumulative count",point_index=sample_index, quantile_zoom=0.95)

    st.plotly_chart(fig, use_container_width=True) 
    for i in range(0,len(data_points)):
        scope_sample = data_points[i]
        st.markdown(f"<div style='margin-left:30px;'>✅ for {scope_sample[0]}, with a threshold of {int(scope_sample[1])} minutes, approximatively {int(scope_sample[2])} rentals are potentialy solved.</div>", unsafe_allow_html=True)

#------------------------------------ Predict section ----------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.subheader("Getaround car rental price prediction", divider=True)

RENTAL_PRICE_PRED_FASTAPI_URL = "https://pieric-getaround-api.hf.space/predict"

inputs = {}


with st.form("prediction_form"):
    for field_name, field_info in RentalPricePredictInput.model_fields.items():
        field_type = field_info.annotation
        default = field_info.default
        if default is PydanticUndefined:
            default = None

        # --- Case 1: Literal (restricted choices) ---
        if get_origin(field_type) is Literal:
            choices = list(get_args(field_type))
            index = 0
            if default is not None and default in choices:
                index = choices.index(default)
            inputs[field_name] = st.selectbox(field_name, choices, index=index)

        # --- Case 2: Boolean ---
        elif field_type is bool:
            inputs[field_name] = st.checkbox(field_name, value=default or False)

        # --- Case 3: Int or float ---
        elif field_type in [int, float]:
            # In v2, constraints are in metadata
            ge = None
            for meta in field_info.metadata:
                if hasattr(meta, "ge"):
                    ge = meta.ge
            min_val = ge if ge is not None else 0

            inputs[field_name] = st.number_input(
                field_name, min_value=min_val, value=default or 0
            )

        # --- Case 4: String ---
        elif field_type is str:
            inputs[field_name] = st.text_input(field_name, value=default or "")

        else:
            st.warning(f"⚠️ Field {field_name} of type {field_type} not handled automatically")

    submitted = st.form_submit_button("Predict Rental Price")

# --- Build request if submitted ---
if submitted:
    try:
        payload = RentalPricePredictInput(**inputs).model_dump() 
        st.write("📦 Payload sent to API:", payload)  # Debugging helper
        response = requests.post(RENTAL_PRICE_PRED_FASTAPI_URL, json=payload)
        if response.status_code == 200:
            prediction = response.json()
            st.success(f"💰 Predicted Rental Price: {prediction['prediction']} €")
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"❌ Validation failed: {e}")