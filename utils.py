import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import time
# import re
# import requests
from typing import List, Optional, Tuple, Union
from datetime import datetime
# from bs4 import BeautifulSoup
# from googletrans import Translator
# from scipy import stats
# from IPython.display import display
# from itertools import combinations
# from imblearn.over_sampling import SMOTE
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.feature_selection import SelectFromModel
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.feature_selection import RFE, SequentialFeatureSelector
# from sklearn.linear_model import LogisticRegression, LassoCV
# from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
# from sklearn.metrics import f1_score, classification_report, confusion_matrix
# from skopt import BayesSearchCV
# import umap
# import csv



##### VISUALIZATION

# CATEGORICAL BAR CHARTS
def bar_charts_categorical(df, feature, target):
    '''
    Generate categorical bar charts for a feature against a target variable.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.

    feature : str
        The categorical feature column name for which bar charts are generated.

    target : str
        The target variable column name used for comparison.

    Returns:
    --------
    None
    '''
    # Create a contingency table using crosstab
    cont_tab = pd.crosstab(df[feature], df[target], margins=True)
    # Extract categories from the index of the contingency table excluding the 'All' row
    categories = cont_tab.index[:-1]
        
    # Create a figure to hold subplots
    fig = plt.figure()
    
    # Subplot for frequency bar chart
    plt.subplot(121)
    # Plot bars for each category for all target values ('$y_i=0$', '$y_i=1$', '$y_i=2$')
    p1 = plt.bar(categories, cont_tab.iloc[:-1, 0].values, 0.55, color="lightcyan")
    p2 = plt.bar(categories, cont_tab.iloc[:-1, 1].values, 0.55, bottom=cont_tab.iloc[:-1, 0], color="powderblue")
    p3 = plt.bar(categories, cont_tab.iloc[:-1, 2].values, 0.55, bottom=cont_tab.iloc[:-1, 0]+cont_tab.iloc[:-1, 1], color="cadetblue")
    # Add legend for the bars
    plt.legend((p3[0], p2[0], p1[0]), (df[target].unique()))
    plt.title("Frequency bar chart")
    plt.xlabel(feature)
    plt.xticks(rotation=90 if df[feature].dtype != 'bool' else 0)    
    plt.ylabel("$Frequency$")

    # Calculate observed proportions for each category and target value
    obs_pct = np.array([np.divide(cont_tab.iloc[:-1, 0].values, cont_tab.iloc[:-1, 3].values), 
                        np.divide(cont_tab.iloc[:-1, 1].values, cont_tab.iloc[:-1, 3].values),
                        np.divide(cont_tab.iloc[:-1, 2].values, cont_tab.iloc[:-1, 3].values)])
      
    # Subplot for proportion bar chart
    plt.subplot(122)
    # Plot bars representing observed proportions for each category and target value
    p1 = plt.bar(categories, obs_pct[0], 0.55, color="antiquewhite")
    p2 = plt.bar(categories, obs_pct[1], 0.55, bottom=obs_pct[0], color="rosybrown")
    p3 = plt.bar(categories, obs_pct[2], 0.55, bottom=obs_pct[0]+obs_pct[1], color="indianred")
    # Add legend for the bars
    plt.legend((p3[0], p2[0], p1[0]), (df[target].unique()))
    plt.title("Proportion bar chart")
    plt.xlabel(feature)
    plt.xticks(rotation=90 if df[feature].dtype != 'bool' else 0)    
    plt.ylabel("$p$")

    # Show the plot
    plt.show()


# PROPERTIES OF PLOTS
def set_plot_properties(x_label, y_label, y_lim=[]):
    """
    Set properties of a plot axis.

    Args:
        ax (matplotlib.axes.Axes): The axis object of the plot.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        y_lim (list, optional): The limits for the y-axis. Defaults to [].

    Returns:
        None
    """
    plt.xlabel(x_label)  # Set the label for the x-axis
    plt.ylabel(y_label)  # Set the label for the y-axis
    if len(y_lim) != 0:
        plt.ylim(y_lim)  # Set the limits for the y-axis if provided


# BAR CHART
def plot_bar_chart(data, variable, x_label=None, y_label='Count', y_lim=[], legend=[], color='cadetblue', annotate=False, top=None, vertical=False):
    """
    Plot a bar chart based on the values of a variable in the given data.

    Args:
        data (pandas.DataFrame): The input data containing the variable.
        variable (str): The name of the variable to plot.
        x_label (str): The label for the x-axis.
        y_label (str, optional): The label for the y-axis. Defaults to 'Count'.
        y_lim (list, optional): The limits for the y-axis. Defaults to [].
        legend (list, optional): The legend labels. Defaults to [].
        color (str, optional): The color of the bars. Defaults to 'cadetblue'.
        annotate (bool, optional): Flag to annotate the bars with their values. Defaults to False.
        top (int or None, optional): The top value for plotting. Defaults to None.
        vertical (bool, optional): Flag to rotate x-axis labels vertically. Defaults to False.

    Returns:
        None
    """
    # Count the occurrences of each value in the variable
    counts = data[variable].value_counts()[:top] if top else data[variable].value_counts()
    x = counts.index  # Get x-axis values
    y = counts.values  # Get y-axis values
    
    # Sort x and y values together
    x, y = zip(*sorted(zip(x, y)))

    # Plot the bar chart with specified color
    plt.bar(x, y, color=color)
    
    # Set the x-axis tick positions and labels, rotate if vertical flag is True
    plt.xticks(rotation=90 if vertical else 0)

    # Annotate the bars with their values if annotate flag is True
    if annotate:
        for i, v in enumerate(y):
            plt.text(i, v, str(v), ha='center', va='bottom', fontsize=12)

    if x_label == None:
        x_label = variable

    set_plot_properties(x_label, y_label, y_lim) # Set plot properties using helper function

    plt.show()


# PIE CHART
def plot_pie_chart(data, variable, colors, labels=None, legend=[], autopct='%1.1f%%'):
    '''
    Plot a pie chart based on the values of a variable in the given data.

    Args:
        data (pandas.DataFrame): The input data containing the variable.
        variable (str): The name of the variable to plot.
        colors (list): The colors for each pie slice.
        labels (list, optional): The labels for each pie slice. Defaults to None.
        legend (list, optional): The legend labels. Defaults to [].
        autopct (str, optional): The format for autopct labels. Defaults to '%1.1f%%'.

    Returns:
        None
    '''
    counts = data[variable].value_counts()  # Count the occurrences of each value in the variable

    # Plot the pie chart with specified parameters
    plt.pie(counts, colors=colors, labels=labels, startangle=90, autopct=autopct, textprops={'fontsize': 21})
    plt.legend(legend if len(legend) > 0 else counts.index, 
               fontsize=16, bbox_to_anchor=(0.7, 0.9))  # Add a legend if provided
    
    plt.show()  # Display the pie chart


# HISTOGRAM
def plot_histogram(data, variable, x_label=None, y_label='Count', color='rosybrown'):
    '''
    Plot a histogram based on the values of a variable in the given data.

    Args:
        ax (matplotlib.axes.Axes): The axis object of the plot.
        data (pandas.DataFrame): The input data containing the variable.
        variable (str): The name of the variable to plot.
        x_label (str): The label for the x-axis.
        y_label (str, optional): The label for the y-axis. Defaults to 'Count'.
        color (str, optional): The color of the histogram bars. Defaults to 'cadetblue'.

    Returns:
        None
    '''
    plt.hist(data[variable], bins=50, color=color)  # Plot the histogram using 50 bins

    if x_label == None:
        x_label = variable

    set_plot_properties(x_label, y_label)  # Set plot properties using helper function

    plt.show()


# BOXPLOT
def plot_box(data, grouped_variable, by_variable):
    # Generate the boxplot
    data[[grouped_variable, by_variable]].boxplot(by=by_variable, color='#5F9EA0')

    # Remove the grid lines
    plt.grid(visible=None)

    # Remove the title
    plt.title(None)

    # Set xlabel and ylabel
    set_plot_properties(by_variable, grouped_variable)

    # Display the plot
    plt.show()


# SCATTER
def plot_scatter(data, variable1, variable2, color='cadetblue'):
    """
    Plot a scatter plot between two variables in the given data.

    Args:
        ax (matplotlib.axes.Axes): The axis object of the plot.
        data (pandas.DataFrame): The input data containing the variables.
        variable1 (str): The name of the first variable.
        variable2 (str): The name of the second variable.
        color (str, optional): The color of the scatter plot. Defaults to 'cadetblue'.

    Returns:
        None
    """
    plt.scatter(data[variable1], data[variable2], color=color, alpha=0.5, )  # Plot the scatter plot

    set_plot_properties(variable1, variable2)  # Set plot properties using helper function


# KDE
def plot_kde(data, variables, colors):
    # Create KDE plots for the scaled numerical columns
    for i, var in enumerate(variables):
        sns.kdeplot(data[var], color=colors[i])

    # Set the legend and label the x-axis
    plt.legend(variables, fontsize=12)
    plt.gca().set_xticks([])
    plt.xlabel('')

    # Display the plot
    plt.show()


# CORRELATION MATRIX
def plot_correlation_matrix(data, method):
    '''
    Plot a correlation matrix heatmap based on the given data.

    Args:
        data (pandas.DataFrame): The input data for calculating correlations.
        method (str): The correlation method to use.

    Returns:
        None
    '''
    corr = data.corr(method=method)  # Calculate the correlation matrix using the specified method

    mask = np.tri(*corr.shape, k=0, dtype=bool)  # Create a mask to hide the upper triangle of the matrix
    corr.where(mask, np.NaN, inplace=True)  # Set the upper triangle values to NaN

    plt.figure(figsize=(30, 15))  # Adjust the width and height of the heatmap as desired

    sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns,
                annot=True,
                vmin=-1, vmax=1,
                cmap=sns.diverging_palette(220, 10, n=20))  # Plot the correlation matrix heatmap

# CONFUSION MATRIX
def plot_confusion_matrix(ax, matrix, title, color_map='Blues'):
    sns.heatmap(matrix, annot=True, fmt='d', cmap=color_map, ax=ax)
    ax.set_title('{} Confusion Matrix'.format(title))
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')


# UMAP
def visualize_dimensionality_reduction(transformation, targets, predictions=None):
    '''
    Visualize the dimensionality reduction results using a scatter plot.

    Args:
        transformation (numpy.ndarray): The transformed data points after dimensionality reduction.
        targets (numpy.ndarray or list): The target labels or cluster assignments.
        predictions (list): List of True or False values indicating if each observation was well predicted.

    Returns:
        None
    '''
    if predictions is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2)

        # Create a scatter plot of the t-SNE output for predictions
        colors = ['lightgrey' if pred else 'indianred' for pred in predictions]
        ax2.scatter(transformation[:, 0], transformation[:, 1], c=colors)
        yes = plt.scatter([], [], c='lightgrey', label='Yes')
        no = plt.scatter([], [], c='indianred', label='No')
        ax2.legend(handles=[yes, no], title='Predicted')

    else:
        ax1 = plt.plot()

    # Convert object labels to categorical variables
    labels, targets_categorical = np.unique(targets, return_inverse=True)

    # Create a scatter plot of the t-SNE output
    cmap = plt.cm.tab20
    norm = plt.Normalize(vmin=0, vmax=len(labels) - 1)
    ax1.scatter(transformation[:, 0], transformation[:, 1], c=targets_categorical, cmap=cmap, norm=norm)

    # Create a legend with the class labels and corresponding colors
    handles = [ax1.scatter([], [], c=cmap(norm(i)), label=label) for i, label in enumerate(labels)]
    ax1.legend(handles=handles, title='Success')

    plt.show()


##### WEB SCRAPING
def fetch_soup(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup


def web_scraping_dges_areas(url):
    # Read DGES area/courses website html
    soup = fetch_soup(url)

    # Select the areas tab
    area_tab = soup.select_one('div.noprint')

    # Create lists with areas' names and links
    open_area = area_tab.find('li').text
    remaining_areas = [area.text for area in area_tab.find_all('a')]
    areas = [open_area] + remaining_areas
    links = [url + link.get('href')[-8:] for link in area_tab.find_all('a')]

    # Remove the last item from both lists
    areas.pop()
    links.pop()

    # Create an empty dictionary
    areas_dict = {}

    # Get the courses from each area
    for i, area in enumerate(areas):
        if i != 0:
            soup = fetch_soup(links[i-1])

        courses = soup.find_all('div', class_='lin-area-c2')
        courses = [course.text for course in courses]
        areas_dict[area] = courses

        time.sleep(1)

    return areas_dict


##### DICTIONARY TREATMENT

# TRANSLATOR
def dictionary_translator(dictionary, from_lang='pt', to_lang='en'):
    # Initialize a Translator object
    translator = Translator()

    # Initialize an empty dictionary to store translated key-value pairs
    translated_dictionary = {}

    # Iterate through key-value pairs in the input dictionary
    for key, values in dictionary.items():
        # Translate the key
        translated_key = translator.translate(key, src=from_lang, dest=to_lang).text
        # Initialize an empty list to store translated values for the key
        translated_dictionary[translated_key] = []

        # Iterate through values for the key
        for value in values:
            # Translate each value
            translated_value = translator.translate(value, src=from_lang, dest=to_lang).text
            # Append the translated value to the list of translated values
            translated_dictionary[translated_key].append(translated_value)
            
            # Pause for a while to avoid hitting rate limits
            time.sleep(2)

        # Pause for a while to avoid hitting rate limits
        time.sleep(5)

    return translated_dictionary


# CAPITALIZER
def dictionary_capitalizer(dictionary):
    # Initialize a new dictionary to store capitalized keys and values
    capitalized_dictionary = {}

    # Iterate through key-value pairs in the input dictionary
    for key, values in dictionary.items():
        # Capitalize the key
        capitalized_key = key.capitalize()
        # Capitalize each value in the list of values
        capitalized_values = [value.capitalize() for value in values]
        # Add the capitalized key and values to the new dictionary
        capitalized_dictionary[capitalized_key] = capitalized_values

    return capitalized_dictionary



##### DATA PREPROCESSING

# COLUMN NAMES
def format_column_name(data: pd.DataFrame, exclude_words: Optional[List[str]] = None) -> List[str]:
    ''' 
    Formats a column name by lowecasing all words except those in the exclusion list.
    '''
    cols = []
    for col in data.columns:
        # DEfine words to exclude
        if exclude_words is None:
            exclude_words = []
        exclude_words = set(word.lower() for word in exclude_words)  # Normalize for case-insensitivity
        
        # Format words
        words = col.split(' ')
        formatted_words = [word.lower() for word in words if word.lower() not in exclude_words]

        # Rejoin the words with '_'
        cols.append('_'.join(formatted_words))
    return cols


# DATA TYPES
def transform_variables_to_boolean(data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    '''
    Converts columns in the given DataFrame(s) to boolean type if they contain exactly two unique values.
    '''
    for col in data.columns:
        # Get and count unique non-null values in the column
        unique_values = data[col].dropna().unique()
        n_unique_values = len(unique_values)

        # Convert the column to boolean, if applicable
        if n_unique_values == 2:
            data[col] = data[col].astype(bool)

    return data


def datatype_distinction(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Distinguishes between the numerical and categorical columns in a DataFrame.
    '''
    # Select numerical columns using select_dtypes with np.number
    numerical = data.select_dtypes(include=np.number).copy()
    
    # Select categorical columns by excluding numerical types
    categorical = data.select_dtypes(exclude=np.number).copy()
    
    return numerical, categorical


# UNDERREPRESENTED CATEGORIES
def find_underrepresented_categories(dataframe: pd.DataFrame, categorical_columns: List, threshold: float = 0.01) -> None:
    """
    Identifies underrepresented categories in categorical columns of a dataframe.
    """
    underrepresented_categories = {}
    for col in categorical_columns.columns:
        category_counts = dataframe[col].value_counts(normalize=True)
        underrepresented = category_counts[category_counts < threshold]
        if not underrepresented.empty:
            underrepresented_categories[col] = underrepresented
    # Display the results
    for col, categories in underrepresented_categories.items():
        print(categories)
        print()


# DATA TRANSFORMATION
def transformation(technique, data: pd.DataFrame, column_transformer: Optional[bool] = False) -> pd.DataFrame:
    '''
    Applies the specified transformation technique to the DataFrame.

    Notes:
    ------
    - If column_transformer is False, the columns in the transformed DataFrame
      will retain the original column names.
    - If column_transformer is True, the method assumes that technique has a
      get_feature_names_out() method and uses it to get feature names for the
      transformed data, otherwise retains the original column names.
    '''
    # Apply the specified transformation technique to the data
    data_transformed = technique.transform(data)
    
    # Create a DataFrame from the transformed data
    data_transformed = pd.DataFrame(
        data_transformed,
        index=data.index,
        columns=technique.get_feature_names_out() if column_transformer else data.columns
    )
    
    return data_transformed


def data_transform(technique, X_train: Union[pd.DataFrame, pd.Series], 
    X_val: Optional[Union[pd.DataFrame, pd.Series]] = None, 
    column_transformer: bool = False) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    '''
    Fits a data transformation technique on the training data and applies the transformation 
    to both the training and validation data.

    Notes:
    ------
    - Fits the transformation technique on the training data (X_train).
    - Applies the fitted transformation to X_train and optionally to X_val if provided.
    '''
    # Fit the transformation technique on the training data
    technique.fit(X_train)
    
    # Apply transformation to the training data
    X_train_transformed = transformation(technique, X_train, column_transformer)
    
    # Apply transformation to the validation data if provided
    X_val_transformed = None
    if X_val is not None:
        X_val_transformed = transformation(technique, X_val, column_transformer)
        
    return X_train_transformed, X_val_transformed


# MISSING VALUES
def drop_missing_values(train: pd.DataFrame, test: pd.DataFrame, ax: int, drop_perc: float = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Calculate the number of missing values along the specified axis
    axis_nulls = train.isnull().sum(axis=ax)

    # Calculate the size of the train data along the specified axis
    if ax == 0:
        size = len(train.index)
    else:
        size = len(train.columns)

    # Calculate the percentage of missing values
    nulls_percentage = round(100 * axis_nulls / size, 1)
    
    # Initialize a list to store columns with high missing percentage
    to_drop = []
    count = 0
    
    # Print columns to remove
    print(f'REMOVE {'COLUMNS' if ax == 0 else 'ROWS'}')
    for obj, perc in nulls_percentage.items():
        if perc > drop_perc:
            print(f'{obj}: {perc}%')
            to_drop.append(obj)
            count += 1
    
    # Remove columns with high missing percentage
    train.drop(to_drop, axis=abs(ax-1), inplace=True)
    
    # Remove the same columns from the test data if ax is 0
    if ax == 0:
        test.drop(to_drop, axis=abs(ax-1), inplace=True)

    print('Total:', count)

    return train, test



# ENCODING
def one_hot_encoding(train, X_test, target):
    # Define X and y
    X_train = train.drop(columns=[target])

    # Filter the dataset with only the object data type columns
    X_train_obj = X_train.select_dtypes(include=['object'])

    # Get the number of unique values from the filtered dataset
    X_train_obj_nu = X_train_obj.nunique()

    # Get the columns with more than 2 unique values
    columns_to_encode = X_train_obj_nu.index[X_train_obj_nu > 2]

    # One-Hot
    ct = ColumnTransformer([
        ('oneHot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), columns_to_encode)
        ], remainder='passthrough')
    X_train, X_test = data_transform(ct, X_train, X_test, column_transformer=True)
    X_train.columns = X_train.columns.str.replace(r'(oneHot|remainder)__', '', regex=True)
    X_test.columns = X_test.columns.str.replace(r'(oneHot|remainder)__', '', regex=True)

    # Tranform variables with only two unique value to boolean
    X_train, X_test = transform_variables_to_boolean(X_train, X_test)

    return X_train, X_test


##### FEATURE ENGINEERING

# FEATURE CREATION
def calculate_age_from_dob(data: pd.DataFrame, dob_column: str) -> pd.DataFrame:
    """
    Get age from the date of birth column in a DataFrame.
    """
    today = datetime.today()
    
    # Convert DOB column to datetime format
    data[dob_column] = pd.to_datetime(data[dob_column], format='%d-%m-%Y', errors='coerce')
    
    # Calculate age
    data['age'] = data[dob_column].apply(lambda dob: today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day)) if pd.notnull(dob) else None)
    
    data.drop(columns=[dob_column], inplace=True)

    return data


def calculate_mean_difference_inner(data, grouped_variable, by_variables, new_column, means):
    # Merge data with mean values based on 'by_variables'
    merged_data = data.merge(means, on=by_variables, how='left')
    
    # Calculate mean difference
    merged_data[new_column] = merged_data[grouped_variable] - merged_data['mean_' + grouped_variable]
    
    # Drop redundant columns
    return merged_data.drop(columns=['mean_' + grouped_variable])


def calculate_mean_difference(train, test, grouped_variable, by_variables, new_column):
    # Calculate mean values for 'grouped_variable' grouped by 'by_variables'
    means = train.groupby(by_variables)[grouped_variable].mean().reset_index()
    means.columns = by_variables + ['mean_' + grouped_variable]

    # Apply mean difference calculation function to train and test sets
    train = calculate_mean_difference_inner(train, grouped_variable, by_variables, new_column, means)
    test = calculate_mean_difference_inner(test, grouped_variable, by_variables, new_column, means)

    return train, test



# RECLASSIFYING
# # Option 1: finished cycle [2nd cycle, 3rd cycle, high school, higher education] (ordinal)
# def get_ordinal_qualification(qualification):
#     if qualification == 'No school':
#         return 0    # No education
    
#     elif qualification in [f'{i}th grade' for i in range(3, 12)]:
#         return 1    # Basic education
    
#     elif '12' in qualification or qualification == 'Incomplete Bachelor\'s':
#         return 2    # Intermidiate education
    
#     else:
#         return 3    # Advanced education


# Option 2: year of education (numerical)
def get_years_of_education(qualification):
    years = re.findall(r'\d+', qualification)

    if qualification == 'No school':
        return 0    # No education
    
    elif years:
        return int(years[0])    # Total years
    
    elif qualification == 'Incomplete Bachelor\'s':
        return 13   # Considering 1 year in university
    
    elif qualification == 'Bachelor degree':
        return 15   # Bachelor's duration general rule is 3 years
    
    elif qualification == 'Post-Graduation':
        return 16   # Plus 1 year after Bachelor degree
    
    elif qualification == 'Master degree':
        return 17   # Plus 2 years after Bachelor degree
    
    elif qualification == 'PhD':
        return 21   # Plus 4 years after Master degree


##### FEATURE SELECTION

# CHI-SQUARE
def TestIndependence(X, y, var, alpha=0.05):
    '''
    Test the independence of a categorical variable with respect to the target variable.

    Parameters:
    -----------
    X : pandas.Series or array-like
        The independent categorical variable.

    y : pandas.Series or array-like
        The target variable.

    var : str
        The name of the variable being tested for importance.

    alpha : float, optional (default=0.05)
        The significance level for the test.

    Returns:
    --------
    None

    Notes:
    ------
    - Performs a chi-squared test of independence between X and y.
    - Compares the p-value with the significance level (alpha).
    - Prints whether the variable is important for prediction or not based on p-value.
    '''
    # Create a contingency table of observed frequencies
    dfObserved = pd.crosstab(y, X)
    
    # Perform chi-squared test and retrieve test statistics
    chi2, p, dof, expected = stats.chi2_contingency(dfObserved.values)
    
    # Create a DataFrame of expected frequencies
    dfExpected = pd.DataFrame(expected, columns=dfObserved.columns, index=dfObserved.index)
    
    # Determine the importance of the variable based on the p-value
    if p < alpha:
        important = True

    else:
        important = False
    

    return important


# RFE
def rfe(X, y):
    # Define a range of number of features to select
    nof_list = np.arange(1, len(X.columns))

    # Iterate through the range
    for n in range(len(nof_list)):
        # Initialize logistic regression model
        model = LogisticRegression(random_state=16)
        
        # Initialize Recursive Feature Elimination (RFE)
        rfe = RFE(model, n_features_to_select=nof_list[n])
        
        # Fit RFE on the data
        rfe.fit(X, y)

    # Return the selected features
    return rfe.support_



# SEQUENTIAL
def sequential_feature_selection(X, y):
    # Define the pipeline with SequentialFeatureSelector
    pipeline = Pipeline([
        ('sfs', SequentialFeatureSelector(LogisticRegression(random_state=16), 
                                          direction='forward', 
                                          scoring='f1_weighted', 
                                          n_jobs=-1))
    ])

    # Fit the pipeline on the data
    pipeline.fit(X, y)

    # Return the selected features
    return pipeline['sfs'].support_



# TREE-BASED
def tree_based_method(X, y, threshold='median'):
    '''
    Perform feature selection using the Extra Trees Classifier method.

    Parameters:
    -----------
    X : pandas.DataFrame or array-like
        The feature matrix.

    y : pandas.Series or array-like
        The target variable.

    threshold : str or float, optional (default='median')
        The feature importance threshold to select features.

    Returns:
    --------
    None

    Notes:
    ------
    - Uses Extra Trees Classifier for feature selection based on feature importances.
    - Prints the names of selected features based on the specified threshold.
    - Does not modify the original data.
    - Reference: https://scikit-learn.org/stable/modules/feature_selection.html
    '''
    # Initialize Extra Trees Classifier
    rf_model = ExtraTreesClassifier(n_estimators=100, random_state=16)

    # Fit the model to your data
    rf_model.fit(X, y)

    # Create a feature selector based on feature importances
    feature_selector = SelectFromModel(rf_model, prefit=True, threshold=threshold)

    # Return the selected feature indices
    return feature_selector.get_support()


# LASSO
def lasso_method(X, y):
    '''
    Perform feature selection using Lasso regression and visualize feature importance.

    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix.

    y : pandas.Series
        Target variable.

    Returns:
    --------
    None

    Notes:
    ------
    - Fits a LassoCV model to perform feature selection.
    - Plots and visualizes feature importance based on Lasso coefficients.
    '''
    # Fit LassoCV model
    reg = LassoCV()
    reg.fit(X, y)
    
    # Extract feature coefficients and index them with column names
    coef = pd.Series(reg.coef_, index=X.columns)
    coef_selected = coef[abs(coef) > 0.015]
        
    selected_features = []
    # Get the selected features by Lasso
    for col in X.columns:
        if col in coef_selected.keys():
            selected_features.append(True)
        else:
            selected_features.append(False)
            
    return selected_features



# GENERAL
def feature_selection_cv(X, y, scaler, imputer, threshold=2, min_freq=8, show=False, export_name=None):
    # Select boolean columns
    columns_bool = X.select_dtypes(include=['bool']).columns

    # Initialize stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=16)

    # Dictionary to store results
    dict_results = {}

    # Iterate over cross-validation folds
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        # Split data into train and test sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Data transformation
        X_train, X_test = data_transform(scaler, X_train, X_test)
        X_train, X_test = data_transform(imputer, X_train, X_test)

        ##### CATEGORICAL
        # Test Independence
        bool_variables_to_use = []
        for col in columns_bool:
            important = TestIndependence(X_train[col], y_train, col)
            if important:
                bool_variables_to_use.append(col)

        ##### NUMERICAL
        # RFE
        rfe_features = rfe(X_train, y_train)

        # Sequential
        sequential_features = sequential_feature_selection(X_train, y_train)

        # Tree-based
        tree_features = tree_based_method(X_train, y_train)

        # LASSO
        lasso_features = lasso_method(X_train, pd.Categorical(y_train).codes)

        ##### DECISIONS
        features = pd.DataFrame({
            'RFE': rfe_features,
            'Sequential': sequential_features,
            'Tree-based': tree_features,
            'Lasso': lasso_features},
            index=X_train.columns)

        features_sum = features.sum(axis=1)
        features_bool = (features_sum > threshold)

        for var, count in features_sum.items():
            if var in bool_variables_to_use and count == threshold:
                features_bool[var] = True

        # Display if specified
        if show:
            print(i)
            features['DECISION'] = features_bool
            features.loc['TOTAL'] = features.sum(axis=0)
            display(features)

        # Store results
        dict_results[i] = features_bool

    # Convert results to DataFrame
    results = pd.DataFrame(dict_results, index=X.columns).T

    # Calculate feature frequency
    features_freq = results.sum()

    # Display if specified
    if show:
        print('\nFEATURES FREQUENCY')
        print(features_freq)

    # Select features with frequency greater than minimum frequency
    features_to_use = features_freq[features_freq > min_freq - 1].index

    # Update X with selected features
    X = X[features_to_use]

    # Export if specified
    if export_name:
        X.to_csv(f'temp\\feature_selection\\{export_name}.csv')

    return X[features_to_use]



##### MODEL ASSUMPTIONS

# LINEAR DISCRIMINANT ANALYSIS
def pairwise_covariance_similarity(covariance_matrices):
    # Get the number of classes
    n_classes = len(covariance_matrices)
    
    # Initialize the matrix to store pairwise similarities
    similarities = np.zeros((n_classes, n_classes))

    # Calculate pairwise similarities
    for i, j in combinations(range(n_classes), 2):
        # Compute correlation coefficient between flattened covariance matrices
        similarity = np.corrcoef(covariance_matrices[i].flatten(), covariance_matrices[j].flatten())[0, 1]
        
        # Store similarity in both positions of the symmetric matrix
        similarities[i, j] = similarity
        similarities[j, i] = similarity
    
    return similarities



##### MODEL SELECTION
def model_evaluator(model, parameters, X, y, scaler, imputer, exhaustive=True, log=False):
    '''
    Evaluates a model using GridSearchCV to find the best parameters and their performance.

    Args:
    - model: Machine learning estimator (e.g., classifier or regressor)
    - parameters: Dictionary of parameters for the model
    - X: Input features (DataFrame or array-like)
    - y: Target variable (Series or array-like)
    - scaler: Scaler object (e.g., StandardScaler, MinMaxScaler), default=None
    - log: Boolean indicating whether to log results to a CSV file, default=False

    Returns:
    - None

    Prints the best parameters and best F1 score achieved by the model.

    If log=True, appends the model's best parameters, best accuracy score, and feature columns to 'record.csv'.
    '''
    # Create a Pipeline with the defined steps
    pipeline = Pipeline([
        ('scaler', scaler),
        ('imputer', imputer),
        ('estimator', model)
        ])

    if exhaustive:
    # Set up GridSearchCV
        search = GridSearchCV(estimator=pipeline,
                              param_grid=parameters,
                              scoring='f1_weighted',
                              return_train_score=True,
                              cv=10,
                              n_jobs=-1
                              )
    
    else:
    # Set up BayesSearchCV
        search = BayesSearchCV(estimator=pipeline,
                              search_spaces=parameters,
                              scoring='f1_weighted',
                              return_train_score=True,
                              cv=10,
                              n_jobs=-1
                              )

    # Fit the GridSearchCV object to find the best parameters
    search.fit(X, y)
    
    best_score = 0
    # Filter out parameter combinations where the difference between training and validation scores is bigger than the threshold
    for mean_train_score, mean_test_score, params in zip(search.cv_results_['mean_train_score'], 
                                                        search.cv_results_['mean_test_score'], 
                                                        search.cv_results_['params']):
        if mean_train_score - mean_test_score <= 0.05:
            if mean_test_score > best_score:
                best_score = mean_test_score
                best_parameters = params

    # Print the best parameters and best accuracy score
    print('Best parameters: {}'.format(best_parameters))
    print('Best score: {}'.format(best_score))

    # Log results to a CSV file if log=True
    if log:
        with open('record.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([X.columns, model, parameters, best_score])


def avg_score(model, X, y, scaler, imputer, oversampling=False): 
    '''
    Calculate the average F1 score for a given model using cross-validation.

    Parameters:
    -----------
    model : sklearn model object
        The model to evaluate.

    X : pandas.DataFrame
        Feature matrix.

    y : pandas.Series
        Target variable.

    scaler : Scaler object, optional (default=None)
        Scaler for feature scaling.

    Returns:
    --------
    str
        A string containing the average F1 score +/- its standard deviation for train and test sets.

    Notes:
    ------
    - Utilizes Stratified K-Fold cross-validation with 10 splits.
    - Computes F1 score for train and test sets and calculates their average and standard deviation.
    '''
    # Apply k-fold cross-validation
    skf = StratifiedKFold(n_splits=10)

    # Create lists to store the results from different folds
    score_train = []
    score_test = []
    timer = []

    for train_index, val_index in skf.split(X, y):
        # Get the indexes of the observations assigned for each partition
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Fit and transform scaler on training data
        X_train, X_val = data_transform(scaler, X_train, X_val)
        X_train, X_val = data_transform(imputer, X_train, X_val)

        if oversampling:
            X_train, y_train = SMOTE().fit_resample(X_train, y_train)
        
        # Start counting time
        begin = time.perf_counter()

        # Fit the model to the training data
        model.fit(X_train, y_train)

        # Finish counting time
        end = time.perf_counter()
        
        # Calculate F1 score for train and test sets
        value_train = f1_score(y_train, model.predict(X_train), average='weighted')
        value_test = f1_score(y_val, model.predict(X_val), average='weighted')
        
        # Append the F1 scores
        score_train.append(value_train)
        score_test.append(value_test)
        timer.append(end-begin)
 
    # Calculate the average and the standard deviation for time and f1 scores
    avg_time = round(np.mean(timer), 3)
    avg_train = round(np.mean(score_train), 3)
    avg_test = round(np.mean(score_test), 3)
    std_time = round(np.std(timer), 2)
    std_train = round(np.std(score_train), 2)
    std_test = round(np.std(score_test), 2)
    
    # Format and return the results as a string
    return (
        str(avg_time) + '+/-' + str(std_time),
        str(avg_train) + '+/-' + str(std_train),
        str(avg_test) + '+/-' + str(std_test)
    )


##### MODEL EVALUATION
def model_evaluation(model, X, y, scaler, imputer, 
                     get_confusion_matrix=False, 
                     get_dimensionality_reduction=False):
    # Splitting the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3,
                                                      stratify=y, shuffle=True,
                                                      random_state=16)
    
    # Scaling and imputing the data
    X_train, X_val = data_transform(scaler, X_train, X_val)
    X_train, X_val = data_transform(imputer, X_train, X_val)

    # Fitting the model on the training data
    model.fit(X_train, y_train)

    # Predictions on training and validation sets
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Printing classification reports
    print('CLASSIFICATION REPORT')
    print('\nTraining')
    print('\n', classification_report(y_train, y_train_pred))
    print('\nValidation')
    print('\n', classification_report(y_val, y_val_pred))

    # If specified, print confusion matrices
    if get_confusion_matrix:
        print('\n\nCONFUSION MATRIX')
        train_matrix = confusion_matrix(y_train, y_train_pred)
        val_matrix = confusion_matrix(y_val, y_val_pred)

        # Plotting the confusion matrix heatmaps
        fig, (ax1, ax2) = plt.subplots(1, 2)  # Creating subplots for train and validation matrices
        plot_confusion_matrix(ax1, train_matrix, 'Training Set', 
                              color_map=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True))
        plot_confusion_matrix(ax2, val_matrix, 'Validation Set', 
                              color_map=sns.color_palette("ch:s=-.2,r=.6", as_cmap=True))
        
        plt.show()

    # If specified, visualize dimensionality reduction
    if get_dimensionality_reduction:
        print('\n\nDIMENSIONALITY REDUCTION VISUALIZATION')
        umap_object = umap.UMAP(n_neighbors=10, min_dist=1, random_state=16)

        # Visualization on training data
        print('\nTraining')
        umap_embedding = umap_object.fit_transform(X_train)
        visualize_dimensionality_reduction(umap_embedding, y_train, (y_train_pred == y_train))

        # Visualization on validation data
        print('\nValidation')
        umap_embedding = umap_object.fit_transform(X_val)
        visualize_dimensionality_reduction(umap_embedding, y_val, (y_val_pred == y_val))