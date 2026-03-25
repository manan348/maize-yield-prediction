"""
Module for creating visualizations with Plotly.
Extracted from: maize_yield_prediction.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_yield_distribution(df, column='Grain Yield [bu/A]', nbins=40):
    """
    Plot histogram of yield distribution.
    
    Args:
        df (pd.DataFrame): Data containing yield column
        column (str): Column name to plot
        nbins (int): Number of histogram bins
        
    Returns:
        go.Figure: Plotly figure
    """
    fig = px.histogram(
        df.dropna(subset=[column]),
        x=column,
        nbins=nbins,
        title='Yield Distribution',
        color_discrete_sequence=['steelblue'],
        template='plotly_white'
    )
    fig.update_layout(height=400)
    return fig


def plot_yield_by_location_box(df):
    """
    Plot yield distribution by location as box plot.
    
    Args:
        df (pd.DataFrame): Data with Field-Location and Grain Yield columns
        
    Returns:
        go.Figure: Plotly figure
    """
    fig = px.box(
        df,
        x='Field-Location',
        y='Grain Yield [bu/A]',
        color='Field-Location',
        title='Yield Distribution by Location',
        template='plotly_white'
    )
    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-45,
        height=500
    )
    return fig


def plot_scatter_yield_vs_feature(df, x_col, y_col='Grain Yield [bu/A]', trendline=None):
    """
    Plot scatter of yield vs another feature.
    
    Args:
        df (pd.DataFrame): Data to plot
        x_col (str): X-axis column
        y_col (str): Y-axis column (default: yield)
        trendline (str): Trendline type (None, 'ols', etc.)
        
    Returns:
        go.Figure: Plotly figure
    """
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color='Field-Location',
        trendline=trendline,
        title=f'Yield vs {x_col}',
        template='plotly_white',
        opacity=0.5
    )
    fig.update_layout(height=450)
    return fig


def plot_weather_by_location_bar(df, column, color_scale='RdYlGn_r', ascending=False):
    """
    Plot weather feature by location as bar chart.
    
    Args:
        df (pd.DataFrame): Weather data
        column (str): Column to plot
        color_scale (str): Plotly color scale
        ascending (bool): Sort ascending or descending
        
    Returns:
        go.Figure: Plotly figure
    """
    fig = px.bar(
        df.sort_values(column, ascending=ascending),
        x='Field-Location',
        y=column,
        color=column,
        color_continuous_scale=color_scale,
        title=f'{column} by Location',
        template='plotly_white'
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=450,
        showlegend=False
    )
    return fig


def plot_pca_variance_explained(pca, n_comp):
    """
    Plot PCA variance explained by component.
    
    Args:
        pca: Fitted PCA transformer
        n_comp (int): Number of components
        
    Returns:
        go.Figure: Plotly figure
    """
    variance = pca.explained_variance_ratio_ * 100
    cumvar = np.cumsum(variance)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=[f'PCA_{i+1}' for i in range(n_comp)],
            y=variance,
            name='Individual Variance',
            marker_color='steelblue'
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=[f'PCA_{i+1}' for i in range(n_comp)],
            y=cumvar,
            name='Cumulative Variance',
            line=dict(color='red', width=2),
            mode='lines+markers'
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title='PCA Variance Explained by Component',
        template='plotly_white',
        height=450
    )
    fig.update_yaxes(title_text='Individual Variance (%)', secondary_y=False)
    fig.update_yaxes(title_text='Cumulative Variance (%)', secondary_y=True)
    
    return fig


def plot_actual_vs_predicted(y_test, y_pred, r2_score):
    """
    Plot actual vs predicted values with perfect fit line.
    
    Args:
        y_test (np.array): Actual test values
        y_pred (np.array): Predicted values
        r2_score (float): R² score for title
        
    Returns:
        go.Figure: Plotly figure
    """
    df_pred = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Error': np.abs(y_test - y_pred).round(2)
    })

    fig = px.scatter(
        df_pred,
        x='Actual', y='Predicted',
        color='Error',
        color_continuous_scale='RdYlGn_r',
        hover_data=['Error'],
        title=f'Actual vs Predicted Yield (R²={r2_score:.3f})',
        labels={
            'Actual': 'Actual Yield (bu/A)',
            'Predicted': 'Predicted Yield (bu/A)'
        },
        template='plotly_white',
        opacity=0.7
    )
    
    # Add perfect fit line
    fig.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines',
        name='Perfect Fit',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    fig.update_layout(height=500)
    return fig


def plot_residuals(y_pred, y_test):
    """
    Plot residuals (actual - predicted) vs predicted values.
    
    Args:
        y_pred (np.array): Predicted values
        y_test (np.array): Actual test values
        
    Returns:
        go.Figure: Plotly figure
    """
    residuals = y_test - y_pred
    
    fig = px.scatter(
        x=y_pred, y=residuals,
        color=np.abs(residuals),
        color_continuous_scale='RdYlGn_r',
        title='Residual Plot',
        labels={
            'x': 'Predicted Yield (bu/A)',
            'y': 'Residual (Actual - Predicted)'
        },
        template='plotly_white',
        opacity=0.6
    )
    
    fig.add_hline(y=0, line_dash='dash', line_color='red', line_width=2)
    fig.update_layout(height=450)
    return fig


def plot_cv_scores(cv_scores):
    """
    Plot cross-validation R² scores by fold.
    
    Args:
        cv_scores (np.array): CV scores from each fold
        
    Returns:
        go.Figure: Plotly figure
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f'Fold {i+1}' for i in range(len(cv_scores))],
        y=cv_scores,
        marker_color=['green' if s > 0 else 'red' for s in cv_scores],
        text=[f'{s:.3f}' for s in cv_scores],
        textposition='outside'
    ))
    
    fig.add_hline(
        y=cv_scores.mean(),
        line_dash='dash', line_color='blue',
        annotation_text=f'Mean: {cv_scores.mean():.3f}'
    )
    
    fig.update_layout(
        title='Cross Validation R² Scores (5-fold)',
        yaxis_title='R² Score',
        template='plotly_white',
        height=400
    )
    return fig


def plot_feature_importance(df_imp):
    """
    Plot feature importance as horizontal bar chart.
    
    Args:
        df_imp (pd.DataFrame): Features with 'Feature' and 'Importance' columns
        
    Returns:
        go.Figure: Plotly figure
    """
    color_map = {
        'Genetics (PCA)': 'steelblue',
        'Plant Trait': 'green',
        'Season Weather': 'orange',
        'Critical Weather': 'red',
        'Environmental': 'purple'
    }
    
    fig = px.bar(
        df_imp.sort_values('Importance', ascending=True),
        x='Importance', y='Feature',
        color='Type' if 'Type' in df_imp.columns else None,
        orientation='h',
        title='Feature Importance',
        template='plotly_white',
        color_discrete_map=color_map if 'Type' in df_imp.columns else None
    )
    fig.update_layout(height=max(400, len(df_imp) * 20))
    return fig


def plot_best_locations(df_locations):
    """
    Plot predicted yields for different locations.
    
    Args:
        df_locations (pd.DataFrame): Locations with predicted yields
        
    Returns:
        go.Figure: Plotly figure
    """
    fig = px.bar(
        df_locations.sort_values('Yield', ascending=False),
        x='Location', y='Yield',
        color='Yield',
        color_continuous_scale='RdYlGn',
        title='Predicted Yield by Location',
        text='Yield',
        template='plotly_white'
    )
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, height=450, showlegend=False)
    return fig


def plot_ge_interaction(df_ge):
    """
    Plot G×E (Genotype × Environment) interaction.
    
    Args:
        df_ge (pd.DataFrame): Data with Location, Yield, and Hybrid columns
        
    Returns:
        go.Figure: Plotly figure
    """
    fig = px.line(
        df_ge,
        x='Location', y='Yield',
        color='Hybrid',
        markers=True,
        title='G×E Interaction',
        template='plotly_white'
    )
    fig.add_hline(y=df_ge['Yield'].mean(), line_dash='dash', annotation_text='Average')
    fig.update_layout(height=500, xaxis_tickangle=-45)
    return fig


def plot_yield_heatmap(pivot_data):
    """
    Plot yield predictions as heatmap.
    
    Args:
        pivot_data (pd.DataFrame): Pivot table with yields (rows=genotypes, cols=locations)
        
    Returns:
        go.Figure: Plotly figure
    """
    fig = px.imshow(
        pivot_data,
        color_continuous_scale='RdYlGn',
        title='Yield Heatmap',
        text_auto='.1f',
        template='plotly_white'
    )
    fig.update_layout(height=max(400, pivot_data.shape[0] * 30))
    return fig
