import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.express as px
from config import *

# todo make title more informative
# todo save paths in config
# todo create folders for each plot
def plot_heatmap(trial_data, names):
    if trial_data["win"][0] == 0:
        title = "loser"
    else:
        title = "winner"
    trial_data = trial_data.drop(['win'],axis=1)
    fig = go.Figure(data=go.Heatmap(
        z=trial_data,
        x=names,
        colorscale='Viridis'))
    fig.update_layout(
    title=title,
    yaxis_nticks=24
        ,xaxis=dict(
        showticklabels=True  # Set showticklabels to False to hide x-axis labels
    ))
    # fig.show()

def plot_area_activity_histogram(data):
    for col in data.columns:
        fig = px.histogram(y=data[col], color=data.win, title=col)
        fig.write_html(f"../graphs/histogram_area/histogram_{col}_winner_loser.html")


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
    fig.write_html(f"../graphs/PCA/PCA_{n_dim}_component.html")

