import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.express as px
from config import *
from plotly.subplots import make_subplots
import pandas as pd
SAVE = True


"""
gets the list of the data and plot for each couploe aplot of the winner mouse in the right and 
loser on the left
"""
def plot_heatmap(mice_info_lst: list):
    for i, pair_data in enumerate(mice_info_lst):
        mouse_1 = pair_data["mouse_1_data"]
        mouse_2 = pair_data["mouse_2_data"]
        date = pair_data["date"]
        pair = pair_data["pair_name"]
        winning_mouse = pair.split("_")[1 + 2]
        trial_num = pair_data["trial_num"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Loser mouse's Brain activity ",
                                                            "Winner mouse's brain activity"),
                                                            shared_yaxes=True,)
        #### mouse 1 plot
        win_loss_1 = mouse_1["win"][0]
        trial_data_1 = mouse_1.drop(['win'], axis=1)
        fig.add_trace(go.Heatmap(
            z=trial_data_1,
            x=area_names,
            colorscale='Viridis',
            coloraxis="coloraxis",
        ),
            row=1, col=win_loss_1 + 1)

        #### mouse 2 plot
        win_loss_2 = mouse_2["win"][0]
        trial_data_2 = mouse_2.drop(['win'], axis=1)
        fig.add_trace(go.Heatmap(
            z=trial_data_2,
            x=area_names,
            colorscale='Viridis',
            coloraxis="coloraxis",
            # colorbar={"title": 'Calcium data value'}
        ),
            row=1, col=win_loss_2 + 1)

        fig.update_layout(
            title="Brain activity difference between winner and loser by frame "
                  + '<br>' + '<span style="font-size: 12px;">' +
                  f"{date} {pair} trial {trial_num} <b> winning mouse:</b> {winning_mouse}",
            # yaxis_nticks=24,
             xaxis=dict(
                showticklabels=True  # Set showticklabels to False to hide x-axis labels,
            ),
            coloraxis={'colorscale': 'viridis', },
            coloraxis_colorbar={"title": "Calcium Value"}
        )
        fig.update_yaxes(title_text="Frame Number")

        # fig.write_image(f"../graphs/heatmap_brain_activity/png/brain_activity_{pair}_{date}_{trial_num}_{i}.png")
        fig.write_html(f"../graphs/heatmap_brain_activity/html/brain_activity__{pair}_{date}_{trial_num}_{i}.html")


"""
box plot for the all data frame
"""
def box_plot_area_activity(df):
    df['win'] = df['win'].replace({1:'winner', 0:'losser'})
    melted_df = pd.melt(df, id_vars='win', var_name='Feature')
    fig = px.box(melted_df, x='Feature', y='value', color='win', points=False,
                 title='Calcium values distribution in brian areas by Winner/Loser',
                 labels={'value': 'Values', 'win': 'winner / loser', 'Feature': 'Areas Names'})
    # fig.show()
    fig.write_html("../graphs/brain_activity_average_loss.html")
    fig.write_image("../graphs/brain_activity_average_loss.png")

"""
calculate the mean of each area of the loser and the winner and plot the different between them in a bar plot
"""
def plot_diffreneces(all_data):
    win_loss_mean = all_data.groupby('win').mean() # mean for each area
    different_mean = win_loss_mean.iloc[0] - win_loss_mean.iloc[1] # get difference between winner and loser
    different_mean_sort = different_mean.sort_values(ascending=False) # sort by value
    fig = go.Figure(go.Bar(y=different_mean_sort, x=different_mean_sort.index))
    fig.update_layout(title_text="Average Difference Between Losser and Winner by area",
    plot_bgcolor = 'rgba(0, 0, 0, 0)',)
    fig.update_yaxes(title= "Calcium Average Difference ")
    fig.update_xaxes(title= "Area names")
    fig.write_html("../graphs/Difference_between_areas_no_abs_losser.html")
    fig.write_image("../graphs/Difference_between_areas_no_abs_losser.png")
    fig.show()

"""
plot the coefficients of the SVM model by decreasing order
"""
def plot_features_importance(svm_m):
    # Access the coefficients (weights) for each feature
    coefficients = svm_m.coef_[0]

    # Create a DataFrame for plotting
    coef_df = pd.DataFrame({'Features': area_names, 'Coefficients': coefficients})

    # Sort the DataFrame by absolute values of coefficients
    coef_df = coef_df.reindex(coef_df['Coefficients'].abs().sort_values(ascending=False).index)

    # Plotting coefficients using Plotly
    fig = px.bar(coef_df, x='Features', y='Coefficients',
                 title='Sorted Coefficients of SVM with Linear Kernel',
                 labels={'Coefficients': 'Coefficient Values'},
                 template='plotly_white')

    fig.update_xaxes(tickangle=45, tickmode='array', tickvals=list(range(len(coef_df))), ticktext=coef_df['Features'])
    fig.write_html("../graphs/SVM_Coefficients.html")
    fig.write_image("../graphs/SVM_Coefficients.png")
    fig.show()
"""
plot the data on the new principal components
"""
def plot_PCA(pca_data, variance, components, n_dim):
    loadings_first_component = components.iloc[0, :]  # Extract loadings for the first component

    # Create a DataFrame with features and their loadings
    loadings_data = {'Features': loadings_first_component.index, 'Loadings': loadings_first_component.values}
    loadings_df_sorted = pd.DataFrame(loadings_data).sort_values(by='Loadings', key=abs, ascending=False)

    # Plotting loadings for the first principal component using Plotly
    fig = px.bar(loadings_df_sorted, x='Features', y='Loadings',
                 title='Coefficients of the First Principal Component',
                 labels={'Loadings': 'Coefficients Values'},
                 template='plotly_white')
    fig.update_xaxes(title="Area names")
    fig.update_xaxes(tickangle=45, tickmode='array', tickvals=list(range(len(loadings_df_sorted))),
                     ticktext=loadings_df_sorted['Features'])

    fig.show()
    fig.write_html("../graphs/PCA/PC1_Components.html")
    fig.write_image("../graphs/PCA/PC1_Components.png")
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

