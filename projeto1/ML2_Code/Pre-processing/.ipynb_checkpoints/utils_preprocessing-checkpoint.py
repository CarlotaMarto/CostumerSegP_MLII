import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math


def outlier_check_IQR(data: pd.DataFrame, variables: list,
                      outlier_type: str = 'normal',
                      return_dataframe: bool = False) -> None:
    """
    Evaluate the outliers of a dataset

    Prints the percentage of the dataset that is retained
    after excluding outliers and can optionally return a
    filtered DataFrame without outliers.

    Parameters:
        ----------
         - data (pd.DataFrame): The DataFrame containing the data.
         - variables (list): The column names of the variables to be evaluated.
         - type (string): The type of outliers to evaluate.
         Defaults to 'normal'.
         - return_dataframe(boolean): Whether to return the filtered dataset
         without outliers.

    Returns:
        ----------
        None, optionally a filtered Dataframe
    """
    dic_type = {'normal': 1.5, 'extreme': 3}
    data_no_out = data.copy()
    for variable in variables:
        P_25 = np.nanpercentile(data_no_out[variable], 25)
        P_75 = np.nanpercentile(data_no_out[variable], 75)
        IQR = P_75 - P_25
        data_no_out = data_no_out[(data_no_out[variable] < P_75 +
                                   dic_type[outlier_type] * IQR)
                                  & (data_no_out[variable] > P_25 -
                                     dic_type[outlier_type] * IQR)]
    print(f"""Excluding all {outlier_type} outliers, we are left with
          {round((len(data_no_out)/len(data))*100, 2)}% of our dataset""")
    if return_dataframe:
        return data_no_out


def cor_heatmap(cor: pd.DataFrame) -> None:
    '''
    Plot a correlation heatmap

    Function to plot a correlation heatmap from a dataframe of correlations.

    Arguments:
        ----------
         - cor(pd.DataFrame): DataFrame of correlations between variables

    Returns:
        ----------
         - None, although a heatmap is produced.
    '''
    mask = np.triu(np.ones_like(cor, dtype=bool))
    plt.figure(figsize=(20, 16))
    sns.heatmap(data=cor, annot=True,
                cmap=sns.color_palette("coolwarm", as_cmap=True),
                fmt='.2', mask=mask, vmin=-1, vmax=1)
    plt.show()


def pie_chart(data: pd.DataFrame, variable: str, colors: list,
              labels: list = None, legend: list = [], title_: str = None,
              autopct: str = '%1.1f%%') -> None:
    """
    Plot a pie chart based on the data.

    Parameters:
        ----------
         - data (pd.DataFrame): The DataFrame containing the data.
         - variable (str): The column name of the variable to be plotted.
         - colors (list): List of colors for the pie chart.
         - labels (list, optional): List of labels for each category.
         Defaults to None.
         - legend (list, optional): List of legend labels. Defaults to [].
         - title_ (str): Title to be ploted. Defaults to None.
         - autopct (str, optional): Format string for autopct parameter
         of the pie chart. Defaults to '%1.1f%%'.

    Returns:
        ----------
         None, although a plot is produced
    """
    # Count the occurrences of each value in the variable
    counts = data[variable].value_counts()

    # Plot the pie chart with specified parameters
    plt.figure(figsize=(8, 6))
    plt.pie(counts, colors=colors, labels=labels, startangle=90,
            autopct=autopct, textprops={'fontsize': 25})

    if len(legend) != 0:
        # Add a legend if provided
        plt.legend(legend, fontsize=16, bbox_to_anchor=(0.6, 0.9))

    if title_:
        # Add title if provided
        plt.title(title_)

    plt.show()


def set_plot_properties(ax: plt.axes, x_label: str, y_label: str,
                        y_lim: list = [], title_: str = None) -> None:
    """
    Set properties for the plot.

    Parameters:
        ----------
         - ax (matplotlib.axes.Axes): The axes object to draw the plot onto.
         - x_label (str): Label for the x-axis.
         - y_label (str): Label for the y-axis.
         - y_lim (list): List containing [min, max] values
         for the y-axis limit.
         - title_ (str): Title to be ploted. Defaults to None.

    Returns:
        ----------
         None
    """
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title_:
        plt.title(title_)
    if len(y_lim) != 0:
        ax.set_ylim(y_lim)


def bar_chart(ax: plt.axes, data: pd.DataFrame, variable: str,
              x_label: str, y_label: str = 'Count', y_lim: list = [],
              legend: list = [], color: str = 'lightblue',
              annotate: bool = False, title_: str = None) -> None:
    """
    Plot a bar chart based on the data.

    Parameters:
        ----------
         - ax (matplotlib.axes.Axes): The axes object to draw the plot onto.
         - data (pd.DataFrame): The DataFrame containing the data.
         - variable (str): The column name of the variable to be plotted.
         - x_label (str): Label for the x-axis.
         - y_label (str, optional): Label for the y-axis. Defaults to 'Count'.
         - y_lim (list, optional): List containing [min, max] values for the
         y-axis limit. Defaults to [].
         - legend (list, optional): List of legend labels. Defaults to [].
         - color (str, optional): Color for the bars. Defaults to 'lightblue'.
         - annotate (bool, optional): Whether to annotate the bars with their
         values. Defaults to False.
         - title_ (str): Title to be ploted. Defaults to None.

    Returns:
        ----------
         None, but a plot is produced
    """
    # Count the occurrences of each value in the variable
    counts = data[variable].value_counts()
    x = counts.index
    y = counts.values

    ax.bar(x, y, color=color)  # Plot the bar chart with specified color
    ax.set_xticks(x)  # Set the x-axis tick positions
    if len(legend) != 0:
        ax.set_xticklabels(legend)  # Set the x-axis tick labels if provided

    if annotate:
        for i, v in enumerate(y):
            ax.text(i, v, str(v), ha='center', va='bottom', fontsize=12)
            # Annotate the bars with their values
    set_plot_properties(ax, x_label, y_label, y_lim, title_)


def line_chart(ax: plt.axes, data: pd.DataFrame, variable: str, x_label: str,
               y_label: str = 'Count', color: str = 'black',
               fill: bool = False, title_: str = None) -> None:
    """
    Plot a line chart based on the data.

    Parameters:
        ----------
         - ax (matplotlib.axes.Axes): The axes object to draw the plot onto.
         - data (pd.DataFrame): The DataFrame containing the data.
         - variable (str): The column name of the variable to be plotted.
         - x_label (str): Label for the x-axis.
         - y_label (str, optional): Label for the y-axis. Defaults to 'Count'.
         - color (str, optional): Color for the bars. Defaults to 'black'.
         - fill(bool): Whether to fill the plot. Defaaults to False
         - title_ (str): Title to be ploted. Defaults to None.

    Returns:
        ----------
         None, but a plot is produced
    """
    # Count the occurrences of each value in the variable
    counts = data[variable].value_counts()
    counts_sorted = counts.sort_index()  # Sort the counts by index
    x = counts_sorted.index
    y = counts_sorted.values
    # Plot the line chart with specified color and marker
    ax.plot(x, y, marker='o', color=color)
    if fill:
        # Fill the area under the line if fill is True
        ax.fill_between(x, y, color='cadetblue', alpha=0.25)
    set_plot_properties(ax, x_label, y_label, y_lim=[], title_=title_)


def histogram_grid(data: pd.DataFrame, variables: list,
                   color: str = None, edgecolor: str = 'black') -> None:
    """
    Plot a histogram grid based on the data.

    Parameters:
        ----------
         - data (pd.DataFrame): The DataFrame containing the data.
         - variables (list): The column names of the variables to be plotted.
         - color (str, optional): Color for the bars. Defaults to None.
         - edgecolor (str, optional): Color for the bars edges.
         Defaults to 'black'.

    Returns:
        ----------
         None, but a plot is produced
    """
    a = math.ceil(math.sqrt(len(variables)))
    b = math.ceil(len(variables) / a)
    fig, axes = plt.subplots(a, b, figsize=(25, 25))
    axes = axes.flatten()
    for i, column in enumerate(variables):
        sns.histplot(x=column, data=data, ax=axes[i], color=color,
                     edgecolor=edgecolor)
        axes[i].set_title(column)
    axes_to_turn_off = (a * b) - len(variables)
    for i in range(1, axes_to_turn_off + 1):
        axes[-i].axis('off')
    plt.tight_layout()
    plt.show()
