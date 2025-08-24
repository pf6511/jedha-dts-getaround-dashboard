import streamlit as st
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from typing import Tuple

import os
from pathlib import Path

import random

import boto3
from botocore.exceptions import BotoCoreError, ClientError

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

    hist, bin_edges = np.histogram(delay_analysis_delta_with_previous_dtf['time_delta_with_previous_rental_in_minutes'], bins=CHART_CUSTOM_BINS, density=False)
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
                                                               ,chart_title:str, xaxis_title:str, yaxis_title:str, point_index:int) -> go.Figure:
    bin_midpoints = 0.5 * (CHART_CUSTOM_BINS[:-1] + CHART_CUSTOM_BINS[1:])

    # Prepare subplot layout
    scopes = sorted(delay_analysis_dataframe["checkin_type"].unique())
    n_rows = len(scopes)

    cumulative_counts_list = []
    title_list = []
    y_max_list = []
    for i, scope in enumerate(scopes):
        data = delay_analysis_dataframe[delay_analysis_dataframe["checkin_type"] == scope][distrib_column]
        hist, _ = np.histogram(data, bins=CHART_CUSTOM_BINS, density=False)
        cumulative_counts = np.cumsum(hist)
        cumulative_counts_list.append(cumulative_counts)
        max = cumulative_counts[len(cumulative_counts)-1]
        title_list.append((scope + ' (total : ' + str(max) + ')'))
        y_max_list.append(max)

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles= title_list
    )

    # Add traces
    for i, scope in enumerate(scopes):
        #data = delay_analysis_delta_with_previous_dtf[delay_analysis_delta_with_previous_dtf["checkin_type"] == scope]["time_delta_with_previous_rental_in_minutes"]
        #hist, _ = np.histogram(data, bins=custom_bins, density=False)
        #cumulative_counts = np.cumsum(hist)
        #proportions = cumulative_counts / cumulative_counts[-1]  # Normalize to 1
        row = i + 1

        fig.add_trace(
            go.Scatter(
                x=bin_midpoints,
                y=cumulative_counts_list[i],
                mode='lines+markers',
                name=scope,
                showlegend=True
            ),
            row=row,
            col=1
        )

        fig.add_shape(
            type='line',
            x0=bin_midpoints[point_index], x1=bin_midpoints[point_index],
            y0=0.0, y1=y_max_list[i],
            line=dict(color='red', dash='dash'),
            xref=f'x{row}' if row > 1 else 'x',  # 'x' for first subplot
            yref=f'y{row}' if row > 1 else 'y',
            row=row, col=1
        )

    # Layout
    fig.update_layout(
        height=300 * n_rows,
        #width=800,
        title_text=chart_title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_white"
    )
    return fig

def compute_rentals_with_previous_rental_dtf(delay_analysis_dtf:pd.DataFrame) -> pd.DataFrame:
    rentals_with_previous_rental = delay_analysis_dtf.loc[delay_analysis_dtf['previous_ended_rental_id'].notnull()]
    rentals_with_previous_rental = pd.merge(delay_analysis_dtf,rentals_with_previous_rental, left_on=['rental_id'], right_on=['previous_ended_rental_id'], how='inner', suffixes=['_previous', '_next'])

def compute_drivers_late_next_checkin_join_dtf(delay_analysis_dtf:pd.DataFrame) -> pd.DataFrame:
    rentals_with_previous_rental = compute_rentals_with_previous_rental_dtf(delay_analysis_dtf)
    rentals_with_delay_checkout_dtf = delay_analysis_dtf.loc[delay_analysis_dtf['delay_at_checkout_in_minutes'] > 0 ]

    drivers_late_next_checking_dtf = pd.merge(rentals_with_delay_checkout_dtf,rentals_with_previous_rental, left_on=['rental_id'], right_on=['previous_ended_rental_id'], how='inner', suffixes=['_previous', '_next'])
    return drivers_late_next_checking_dtf.loc[drivers_late_next_checking_dtf['delay_at_checkout_in_minutes_previous'] > drivers_late_next_checking_dtf['time_delta_with_previous_rental_in_minutes_next']]



st.title("Getaround insight")
st.subheader("Getaround delay analysis")

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

st.markdown("---")
st.subheader("Impacts of a minimum delay between two rentals", divider=True)
delay_analysis_delta_with_previous_dtf = delay_analysis_dtf.query("time_delta_with_previous_rental_in_minutes.notnull()")
#st.markdown("---")
#st.markdown("<br>", unsafe_allow_html=True)
#st.subheader('')
st.markdown('<h5> Which share of our owner’s revenue would potentially be affected by the feature ? </h5>', unsafe_allow_html=True)
#st.subheader('Which share of our owner’s revenue would potentially be affected by the feature ?' )
st.markdown('<span style="color: var(--primary-color)">The Proportion of rentals with a previous rental and with a delay < 12h with previous rentals helps answer this question</span>',unsafe_allow_html=True)
#st.markdown('**time_delta_with_previous_rental_in_minutes**')
#st.markdown('Difference in minutes between this rental planned start time and the previous rental planned end time (when lower than 12 hours, NULL if higher)')
proportion_delta_with_previous = 100*delay_analysis_delta_with_previous_dtf.shape[0]/delay_analysis_dtf.shape[0]
proportion_delta_with_previous_formatted = format_decimals(proportion_delta_with_previous,2)
st.markdown(f'Number of rentals with previous rental  < 12h  : <span style="color: var(--primary-color); font-weight:bold;">{delay_analysis_delta_with_previous_dtf.shape[0]} ({proportion_delta_with_previous_formatted} % ) </span>',unsafe_allow_html=True)


st.markdown('<h5> How many rentals would be affected by the feature depending on the threshold and scope we choose ? </h5>', unsafe_allow_html=True)
st.markdown('Rentals affected by the threshold : if delay1 < threshold.')
st.markdown('Hence if delay2 < delay1 < threshold => rentals with delay2 also concerned, so cumulative histogram makes sense')
rental_distribution_delay_threshold_col, rental_distribution_delay_threshold_scope_col = st.columns(2)

with rental_distribution_delay_threshold_col:
    fig, sample = generate_cumulative_rentals_by_threshold_chart(delay_analysis_delta_with_previous_dtf,2)
    st.plotly_chart(fig, use_container_width=True)
    sample_percentage = 100*sample[2]
    sample_percentage_formatted = format_decimals(sample_percentage,2)
    st.markdown(f'With a threshold of {int(sample[0])} minutes, approximatively {int(sample[1])} rentals are concerned. It represents {sample_percentage_formatted} % of the rentals with a previous rental, and {sample_percentage_formatted}% * {proportion_delta_with_previous_formatted}% of the whole population.')

with rental_distribution_delay_threshold_scope_col:
    fig = generate_cumulative_rentals_by_threshold_and_scope_chart(delay_analysis_dataframe=delay_analysis_delta_with_previous_dtf
                                                                     ,distrib_column='time_delta_with_previous_rental_in_minutes'
                                                                     ,chart_title='Cumulative Distribution of rentals affected by Threshold / scope'
                                                                     ,xaxis_title='Threshold - Time delta with previous rental (minutes)'
                                                                     ,yaxis_title="Cumulative count",point_index=2)
    st.plotly_chart(fig, use_container_width=True)    
    
st.markdown('<h5> How often are drivers late for the next check-in? How does it impact the next driver ?  </h5>', unsafe_allow_html=True)

st.markdown('<h6> 1. How often are drivers late for the next check-in ?  </h6>', unsafe_allow_html=True)

drivers_late_next_checking_dtf = compute_drivers_late_next_checkin_join_dtf(delay_analysis_dtf)
rentals_with_previous_rental = compute_rentals_with_previous_rental_dtf(delay_analysis_dtf)
st.markdown(f'proportion of drivers late for next check-in : {format_decimals(100*drivers_late_next_checking_dtf.shape[0]/rentals_with_previous_rental.shape[0],2)} %')

st.markdown('<h6> 2.How does it impact the next driver ?  </h6>', unsafe_allow_html=True)
st.markdown('Is it fair to assume that when driver is late for next check-in, then next driver has more chance to cancel his rental or to be late too for for next check-in.')
st.markdown("Let's compute canceling rate / dalay at checkout rate comparing with reference (drivers with previous checking) ")

late_driver_impact_next_col, sep_col, driver_previous_checkin_ref_col  = st.columns([4,0.1,4])
with late_driver_impact_next_col:

    st.markdown('**Within late drivers for next check-in**', unsafe_allow_html=True)

    metric_names = ['proportion (%) of next drivers returning car in advance'
                    , 'proportion (%) of next drivers returning car late'
                    ,'proportion (%) of next drivers canceling their rental' ]
    data= [{} for _ in range(len(metric_names))]
    for i, name in enumerate(metric_names):
        data[i]['name'] = name

    data[0]['metric'] = format_decimals(100*drivers_late_next_checking_dtf.query("delay_at_checkout_in_minutes_next < 0").shape[0]/drivers_late_next_checking_dtf.shape[0],2)
    data[1]['metric'] = format_decimals(100*drivers_late_next_checking_dtf.query("delay_at_checkout_in_minutes_next > 0").shape[0]/drivers_late_next_checking_dtf.shape[0],2)
    data[2]['metric'] = format_decimals(100*drivers_late_next_checking_dtf.query('state_next == "canceled"').shape[0]/drivers_late_next_checking_dtf.shape[0],2)


    data[0]['breakdown'] = (drivers_late_next_checking_dtf.loc[drivers_late_next_checking_dtf['delay_at_checkout_in_minutes_next'] < 0]['checkin_type_previous']
        .value_counts(normalize=True)
        .mul(100).round(2).reset_index()
    )

    data[1]['breakdown'] = (drivers_late_next_checking_dtf.loc[drivers_late_next_checking_dtf['delay_at_checkout_in_minutes_next'] > 0]['checkin_type_previous']
        .value_counts(normalize=True)
        .mul(100).round(2).reset_index()
    )

    data[2]['breakdown'] = (drivers_late_next_checking_dtf.loc[drivers_late_next_checking_dtf['state_next'] ==  'canceled']['checkin_type_previous']
        .value_counts(normalize=True)
        .mul(100).round(2).reset_index()
    )

    col_name, col_metric, col_bkdown = st.columns([2,1,3])
    col_name.markdown('**Metric**')
    col_metric.markdown('**Value**')
    col_bkdown.markdown('**Breakdown**')

    for row in data:
        col_name, col_metric, col_bkdown = st.columns([2,1,3])
        col_name.write(row['name'])
        col_metric.write(row['metric'])
        with col_bkdown:
            with st.expander("Details"):
                st.write(row['breakdown'])

with sep_col:
    st.markdown(
        """<div style='border-left: 2px solid lightgray; height: 100%;'></div>""",
        unsafe_allow_html=True
    )

with driver_previous_checkin_ref_col:
    st.markdown('**Within all rentals with previous rentals (reference)**', unsafe_allow_html=True)

    metric_names = ['proportion (%) of next drivers returning car in advance'
                    , 'proportion (%) of next drivers returning car late'
                    ,'proportion (%) of next drivers canceling their rental' ]
    data= [{} for _ in range(len(metric_names))]
    for i, name in enumerate(metric_names):
        data[i]['name'] = name

    data[0]['metric'] = format_decimals(100*rentals_with_previous_rental.query("delay_at_checkout_in_minutes_next < 0").shape[0]/rentals_with_previous_rental.shape[0],2)
    data[1]['metric'] = format_decimals(100*rentals_with_previous_rental.query("delay_at_checkout_in_minutes_next > 0").shape[0]/rentals_with_previous_rental.shape[0],2)
    data[2]['metric'] = format_decimals(100*rentals_with_previous_rental.query('state_next == "canceled"').shape[0]/rentals_with_previous_rental.shape[0],2)


    data[0]['breakdown'] = (rentals_with_previous_rental.loc[rentals_with_previous_rental['delay_at_checkout_in_minutes_next'] < 0]['checkin_type']
        .value_counts(normalize=True)
        .mul(100).round(2).reset_index()
    )

    data[1]['breakdown'] = (rentals_with_previous_rental.loc[rentals_with_previous_rental['delay_at_checkout_in_minutes_next'] > 0]['checkin_type']
        .value_counts(normalize=True)
        .mul(100).round(2).reset_index()
    )

    data[2]['breakdown'] = (rentals_with_previous_rental.loc[rentals_with_previous_rental['state_next'] ==  'canceled']['checkin_type']
        .value_counts(normalize=True)
        .mul(100).round(2).reset_index()
    )

    col_name, col_metric, col_bkdown = st.columns([2,1,3])
    col_name.markdown('**Metric**')
    col_metric.markdown('**Value**')
    col_bkdown.markdown('**Breakdown**')

    for row in data:
        col_name, col_metric, col_bkdown = st.columns([2,1,3])
        col_name.write(row['name'])
        col_metric.write(row['metric'])
        with col_bkdown:
            with st.expander("Details"):
                st.write(row['breakdown'])


st.markdown('<h5> How many problematic cases will it solve depending on the chosen threshold and scope ?  </h5>', unsafe_allow_html=True)
st.markdown("""We compute min_delay_between_rentals = delay_at_checkout_in_minutes_previous for rentals with driver late for next check-in.
Thus with this delay / threshold it solves problematic case (drivers late for next check-in).
Then by computing cumulative distribution function based on this new field / threshold, it gives the number of problematic cases solved by threshold""")

min_delay_threshold_col, empty_col = st.columns(2)
with min_delay_threshold_col:
    drivers_late_next_checking_dtf['min_delay_between_rentals'] = drivers_late_next_checking_dtf['time_delta_with_previous_rental_in_minutes_next'] + (drivers_late_next_checking_dtf['delay_at_checkout_in_minutes_previous']-drivers_late_next_checking_dtf['time_delta_with_previous_rental_in_minutes_next'])
    drivers_late_next_checking_threshold_dtf = drivers_late_next_checking_dtf.loc[:,['min_delay_between_rentals', 'checkin_type_previous' ]].copy()
    drivers_late_next_checking_threshold_dtf.rename(columns={'checkin_type_previous':'checkin_type'}, inplace=True)

    fig = generate_cumulative_rentals_by_threshold_and_scope_chart(delay_analysis_dataframe=drivers_late_next_checking_threshold_dtf
                                                                        ,distrib_column='min_delay_between_rentals'
                                                                        ,chart_title='Cumulative Distribution of rentals potentialy solved by Threshold'
                                                                        ,xaxis_title='Threshold - Time delta with previous rental (minutes)'
                                                                        ,yaxis_title="Cumulative count",point_index=2)

    st.plotly_chart(fig, use_container_width=True) 

st.subheader("Getaround car rental price prediction")

RENTAL_PRICE_PRED_FASTAPI_URL = "https://pieric-mlflow-server-demo.hf.space/predict"