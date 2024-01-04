import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.express as px
from config import *
from plotly.subplots import make_subplots
import pandas as pd
SAVE = True

# todo make title more informative
# todo save paths in config
# todo create folders for each plot
def plot_heatmap(mice_info_lst):

        if SAVE:
            fig.write_image(f"../graphs/heatmap_brain_activity/png/brain_activity_{pair}_{date}_{trial_num}.png")
            fig.write_html(f"../graphs/heatmap_brain_activity/html/brain_activity__{pair}_{date}_{trial_num}.html")
        fig.show()



def plot_area_activity_histogram(data):
    for col in data.columns:
        fig = px.histogram(y=data[col], color=data.win, title=col)
        if SAVE:
            fig.write_html(f"../graphs/histogram_area/histogram_{col}_winner_loser.html")

def plot_bar_area_activity(all_data):
    win_loss_mean = all_data.groupby('win').mean()
    win_loss_std = all_data.groupby('win').std()
    win_df = pd.DataFrame(win_loss_mean.loc[0], ).reset_index()
    win_df.columns = ["area name", "value"]
    loss_df = pd.DataFrame(win_loss_mean.iloc[1]).reset_index()
    loss_df.columns = ["area name", "value"]
    fig = go.Figure(data=[
        go.Bar(name='winner', x=win_df["area name"], y=win_df["value"], error_y=dict(type='data', array=win_loss_std)),
        go.Bar(name='losser', x=loss_df["area name"], y=loss_df["value"], error_y=dict(type='data', array=win_loss_std))
    ])
    # Change the bar mode
    fig.update_layout(barmode='group', title_text='Calcium Values Average by Area Winner and losser',
                      yaxis_title='Calcium Value Average',
                      xaxis_title='Brain Area')
    fig.show()
    fig.write_html(f"../graphs/brain_activiy_average_losser_and_winner.html")

def plot_PCA(pca_data, variance, n_dim):
    if n_dim == 2:
        fig = px.scatter(pca_data, x=0, y=1, color=target,
                         title=f'Total Explained Variance: {variance:.2f}%')
    elif n_dim ==3:
        fig = px.scatter_3d(
            pca_data, x=0, y=1, z=2, color=target,
            title=f'Total Explained Variance: {variance:.2f}%',
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )
    if SAVE:
        fig.write_html(f"../graphs/PCA/PCA_{n_dim}_component.html")

