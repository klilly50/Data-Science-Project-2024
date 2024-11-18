# Seattle Library Checkouts

## Introduction
This project aims to develop predictive insights into library book checkouts using machine learning and time series analysis. Leveraging the ``Checkouts by Title" dataset provided by the Seattle Public Library, our analysis addresses two main questions: (1) can we predict the number of checkouts in the first year for a new book based on a number of features, and (2) can we forecast future checkouts over several months by analyzing past checkout patterns?

Both questions have implications for both inventory management and future acquisitions. Our first question aims to help libraries, bookstores, and publishers determine the number of copies of a new book they should buy/release. Our time series forecasting will help these same stakeholders estimate demand, supporting strategic planning for resource allocation. 

## Data Cleaning and Exploratory Data Analysis

### First Year Checkouts
Working with such a large dataset that included over 45 million entries introduced challenges, particularly in processing speed and memory limitations. We mitigated this by filtering out non-English titles and records with significant missing data. Though we initially hoped to use ISBNs as unique identifiers, inconsistencies rendered this unfeasible. Without a single unique identifier across columns, we developed alternative approaches to track items accurately.

From the ``Checkouts by Title" data set, we opted to keep the following key features to examine, focusing on fields essential to our questions about checkout patterns: Usage class (Physical vs. Digital), Material Type (Book, Sound disc, Audiobook, Ebook, Video disc, Other), Checkout year, Checkout month, Checkouts, Title, Creator, Subjects, and Publisher. Author names and book titles were standardized for consistency, and we categorized books by genre using text analysis on the subjects provided. To establish the number of checkouts in a book's first year, we segmented the data by checkout dates, noting that later records exhibited inconsistencies. Initial visualizations revealed unreliability in these periods, leading us to exclude data from certain time ranges.

To enhance our predictive capabilities, we created an additional feature to quantify author popularity. For each book, we calculated the total number of checkouts for the author’s other works in the years preceding the book’s release. This feature provides a measure of prior popularity, helping us assess whether an author’s established readership correlates with the initial demand for new titles. By incorporating this popularity metric, we aim to improve the accuracy of our first-year checkout predictions.

Our exploratory data analysis highlighted minimal overarching trends across the chosen features, underscoring the complexity and diversity in library checkouts. We anticipated that author popularity, as measured by the total checkouts of an author’s previous works, would be a strong predictor of a new book's first-year checkouts. However, our analysis revealed no significant correlation between these variables, challenging our initial expectations. Figure \textcolor{blue}{reference figure here} illustrates this lack of relationship, showing scattered data points with no clear trend, suggesting that factors beyond an author’s past popularity may play a larger role in predicting initial demand for new titles.

### Time Series Data
For our time series analysis, we aggregated the total number of checkouts per month across the entire dataset. To gain deeper insights, we also categorized this data by material type, allowing us to observe monthly trends within specific categories.

An initial visualization of this data is shown in Figure \textcolor{blue}{reference figure here} reveals distinct patterns in checkout trends over time for each material type. Notably, a sharp decline is visible during the early months of the COVID-19 pandemic, reflecting reduced access to physical library resources. In contrast, digital materials saw less of an impact during this period, illustrating a shift in user behavior when physical access was limited. In fact, digital media such as Ebooks appear to exponentially grow over time, suggesting the rise in digital media post pandemic. This trend provides valuable context for our forecasting models, as it highlights both long-term patterns and temporary disruptions that influence overall checkout behaviors.

Our ACF and PACF analysis of the time series data indicated a seasonal pattern, where checkouts are highly correlated with data from approximately a year prior. This repeating cycle suggests that library checkouts follow predictable annual trends. This strong autocorrelation at yearly intervals informs our forecasting approach, as it highlights the importance of past seasonal patterns in predicting future checkout behavior for various material types. \textcolor{red}{Should probably discuss negative correlation with data at 13 months??}

## Modeling Approach

### First Year Checkouts
Our baseline model, used to establish a reference point for prediction accuracy, was chosen to be the average number of checkouts in the first year for all books in the training set. This value turned out to be approximately 116 checkouts and was used as the predicted number of first year checkouts, regardless of book. Such a choice of baseline provides a straightforward metric for comparison.

To improve upon this baseline, we applied a variety of linear regression models to capture relationships between the features and target variable. We experimented with multiple regression techniques, including ordinary least squares, ridge, and lasso regression, each selected to test different ways of handling feature relationships and regularization. We also explored several other models, including random forest, \textcolor{red}{include other models Danielle tried}, which provided further insights into model performance and refinement.

### Time Series Data
We used a linear regression model as a baseline to establish an initial predictive framework for the time series. This baseline provides a simple trend line to gauge the general direction of monthly checkouts over time, helping us identify if more complex models can yield improvements.

To capture the seasonality and autocorrelation observed in the data, we also experimented with an AutoARIMA model. This model dynamically selects the best-fitting parameters for autoregressive and moving average components, allowing it to adapt to seasonal patterns and more complex dependencies in the data. To achieve more stationary results, we subtracted off the trend in our data before applying AutoARIMA. We determined the trend by fitting our data using \textcolor{red}{a third order polynomial.} Comparing AutoARIMA’s performance against the linear regression baseline enables us to evaluate how well each approach captures the nuanced temporal patterns in library checkouts. \textcolor{red}{This entire section might need modifying based on what we decide to use/do, but it's a start.}

We then compared our results obtained by removing the trend ourselves to those obtained using the time series analysis (tsa) functionality built into statsmodels in Python. This functionality automatically smooths out our data to calculate the trend using a state space approach, rather than needing to rely on an arbitrary fit as we did when applying AutoARIMA ourselves. Moreover, it can also automatically remove seasonal components from a time series. Finally, after removing seasonality and trend for us systematically, AutoARIMA can be applied to forecast forward.

## Results

TBD

