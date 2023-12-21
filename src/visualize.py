import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

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