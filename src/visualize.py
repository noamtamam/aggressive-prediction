import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.express as px

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